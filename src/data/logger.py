"""Prediction event logger backed by the SQLAlchemy data layer.

Provides convenience methods for recording predictions, attaching
ground-truth labels, querying recent predictions, and computing basic
throughput statistics.

Usage::

    from src.data.database import get_database
    from src.data.logger import PredictionLogger

    db = get_database()
    db.init_db()

    pl = PredictionLogger(db)
    pid = pl.log_prediction(
        features={"amount": 120.5, "hour": 14},
        predicted_label=0,
        confidence=0.92,
        model_version="3",
        latency_ms=4.3,
    )
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, func

from src.data.database import DatabaseManager, Prediction

logger = logging.getLogger(__name__)


class PredictionLogger:
    """High-level API for storing and retrieving prediction records.

    Args:
        db: A :class:`~src.data.database.DatabaseManager` instance used
            to obtain database sessions.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    # ------------------------------------------------------------------ #
    # Write operations
    # ------------------------------------------------------------------ #

    def log_prediction(
        self,
        features: dict[str, Any],
        predicted_label: int,
        confidence: float,
        model_version: str,
        latency_ms: float,
    ) -> int:
        """Persist a single prediction and return its database ID.

        Args:
            features: Raw feature dict (will be JSON-serialised).
            predicted_label: Model output class (e.g. 0 or 1).
            confidence: Predicted probability for the positive class.
            model_version: Model version string or integer label.
            latency_ms: Inference wall-clock time in milliseconds.

        Returns:
            The auto-generated primary-key ``id`` of the new row.
        """
        record = Prediction(
            features_json=json.dumps(features, default=str),
            predicted_label=predicted_label,
            confidence=confidence,
            model_version=str(model_version),
            latency_ms=latency_ms,
        )

        with self._db.get_session() as session:
            session.add(record)
            session.flush()  # populate record.id before commit
            pred_id: int = record.id

        logger.debug("Logged prediction id=%d (v=%s).", pred_id, model_version)
        return pred_id

    def log_ground_truth(self, prediction_id: int, true_label: int) -> bool:
        """Attach a ground-truth label to an existing prediction.

        Args:
            prediction_id: Primary key of the prediction to update.
            true_label: The actual class label.

        Returns:
            ``True`` if the prediction was found and updated, ``False``
            otherwise.
        """
        with self._db.get_session() as session:
            record: Prediction | None = session.get(Prediction, prediction_id)
            if record is None:
                logger.warning(
                    "Prediction id=%d not found – ground truth not stored.",
                    prediction_id,
                )
                return False

            record.true_label = true_label

        logger.debug(
            "Ground truth (label=%d) attached to prediction id=%d.",
            true_label,
            prediction_id,
        )
        return True

    def log_batch_predictions(
        self,
        predictions: list[dict[str, Any]],
    ) -> list[int]:
        """Bulk-insert multiple predictions in a single transaction.

        Each dict in *predictions* must contain the keys accepted by
        :meth:`log_prediction`:

        - ``features`` (dict)
        - ``predicted_label`` (int)
        - ``confidence`` (float)
        - ``model_version`` (str)
        - ``latency_ms`` (float)

        Args:
            predictions: List of prediction dictionaries.

        Returns:
            List of auto-generated primary-key IDs in insertion order.
        """
        records: list[Prediction] = []
        for entry in predictions:
            records.append(
                Prediction(
                    features_json=json.dumps(
                        entry["features"], default=str,
                    ),
                    predicted_label=entry["predicted_label"],
                    confidence=entry["confidence"],
                    model_version=str(entry["model_version"]),
                    latency_ms=entry["latency_ms"],
                )
            )

        with self._db.get_session() as session:
            session.add_all(records)
            session.flush()
            ids = [r.id for r in records]

        logger.info("Batch-logged %d predictions.", len(ids))
        return ids

    # ------------------------------------------------------------------ #
    # Read operations
    # ------------------------------------------------------------------ #

    def get_predictions(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        model_version: str | None = None,
        limit: int = 1000,
    ) -> list[Prediction]:
        """Query predictions with optional time-range and version filters.

        Args:
            start_time: Inclusive lower bound on ``timestamp``.
            end_time: Inclusive upper bound on ``timestamp``.
            model_version: Filter to a specific model version.
            limit: Maximum number of rows to return (newest first).

        Returns:
            List of :class:`Prediction` ORM instances ordered by
            descending timestamp.
        """
        with self._db.get_session() as session:
            query = session.query(Prediction)

            if start_time is not None:
                query = query.filter(Prediction.timestamp >= start_time)
            if end_time is not None:
                query = query.filter(Prediction.timestamp <= end_time)
            if model_version is not None:
                query = query.filter(
                    Prediction.model_version == model_version,
                )

            results: list[Prediction] = (
                query.order_by(desc(Prediction.timestamp))
                .limit(limit)
                .all()
            )

        return results

    def get_recent_predictions(self, n: int = 500) -> list[Prediction]:
        """Return the *n* most recent predictions.

        Args:
            n: Number of rows to fetch.

        Returns:
            List of :class:`Prediction` instances, newest first.
        """
        with self._db.get_session() as session:
            results: list[Prediction] = (
                session.query(Prediction)
                .order_by(desc(Prediction.timestamp))
                .limit(n)
                .all()
            )
        return results

    def get_predictions_with_ground_truth(
        self,
        n: int = 500,
    ) -> list[Prediction]:
        """Return predictions that have a ground-truth label attached.

        Only rows where ``true_label IS NOT NULL`` are included.

        Args:
            n: Maximum number of rows to return.

        Returns:
            List of :class:`Prediction` instances, newest first.
        """
        with self._db.get_session() as session:
            results: list[Prediction] = (
                session.query(Prediction)
                .filter(Prediction.true_label.isnot(None))
                .order_by(desc(Prediction.timestamp))
                .limit(n)
                .all()
            )
        return results

    # ------------------------------------------------------------------ #
    # Statistics
    # ------------------------------------------------------------------ #

    def get_prediction_stats(self) -> dict[str, Any]:
        """Compute aggregate statistics across all stored predictions.

        Returns:
            Dictionary with keys:

            - ``total_count`` – total number of prediction rows.
            - ``avg_latency_ms`` – mean inference latency.
            - ``throughput_per_min`` – approximate predictions per minute
              based on the time span between the earliest and latest
              prediction.
            - ``model_version`` – the most recent model version string
              (or ``None`` if the table is empty).
        """
        with self._db.get_session() as session:
            total_count: int = session.query(
                func.count(Prediction.id),
            ).scalar() or 0

            avg_latency: float | None = session.query(
                func.avg(Prediction.latency_ms),
            ).scalar()

            min_ts: datetime | None = session.query(
                func.min(Prediction.timestamp),
            ).scalar()
            max_ts: datetime | None = session.query(
                func.max(Prediction.timestamp),
            ).scalar()

            latest: Prediction | None = (
                session.query(Prediction)
                .order_by(desc(Prediction.timestamp))
                .first()
            )

        # Throughput calculation
        throughput: float = 0.0
        if min_ts is not None and max_ts is not None and min_ts != max_ts:
            span_minutes = (max_ts - min_ts).total_seconds() / 60.0
            if span_minutes > 0:
                throughput = total_count / span_minutes

        return {
            "total_count": total_count,
            "avg_latency_ms": round(avg_latency, 3) if avg_latency else 0.0,
            "throughput_per_min": round(throughput, 2),
            "model_version": latest.model_version if latest else None,
        }
