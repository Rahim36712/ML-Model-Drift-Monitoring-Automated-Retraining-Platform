"""Central orchestrator for all drift-detection checks.

``DriftManager`` ties together the three specialised detectors
(:class:`~src.monitoring.data_drift.DataDriftDetector`,
:class:`~src.monitoring.prediction_drift.PredictionDriftDetector`,
:class:`~src.monitoring.concept_drift.ConceptDriftDetector`) and the
persistence layer (:class:`~src.data.database.DatabaseManager`).

A single call to :meth:`DriftManager.run_drift_check` will:

1. Parse feature vectors from the raw prediction records.
2. Invoke whichever detectors are configured.
3. Persist individual :class:`~src.data.database.DriftResult` rows.
4. Return a unified :class:`DriftSummary`.

Usage::

    manager = DriftManager(
        db=get_database(),
        data_detector=data_det,
        prediction_detector=pred_det,
        concept_detector=concept_det,
    )
    summary = manager.run_drift_check(recent_predictions, feature_names)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlalchemy import desc

from src.data.database import DatabaseManager, DriftResult
from src.monitoring.concept_drift import ConceptDriftDetector, ConceptDriftResult
from src.monitoring.data_drift import DataDriftDetector, DataDriftResult
from src.monitoring.prediction_drift import (
    PredictionDriftDetector,
    PredictionDriftResult,
)

logger = logging.getLogger(__name__)

# Priority ordering used to collapse multiple severities
_SEVERITY_ORDER: dict[str, int] = {
    "none": 0,
    "healthy": 0,
    "warning": 1,
    "critical": 2,
}


# -------------------------------------------------------------------
# Result container
# -------------------------------------------------------------------


@dataclass(slots=True)
class DriftSummary:
    """Aggregated result from a full drift-check cycle.

    Attributes:
        data_drift: Result from the data-drift detector (``None`` if
            the detector was not configured).
        prediction_drift: Result from the prediction-drift detector.
        concept_drift: Result from the concept-drift detector.
        overall_status: Worst status across all checks — one of
            ``'healthy'``, ``'warning'``, ``'critical'``.
        timestamp: UTC time the check was initiated.
        checks_performed: Names of the detectors that were executed.
    """

    data_drift: DataDriftResult | None = None
    prediction_drift: PredictionDriftResult | None = None
    concept_drift: ConceptDriftResult | None = None
    overall_status: str = "healthy"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    checks_performed: list[str] = field(default_factory=list)


# -------------------------------------------------------------------
# Manager
# -------------------------------------------------------------------


class DriftManager:
    """Orchestrates drift checks, persistence, and history retrieval.

    Each detector is *optional* — pass ``None`` to skip that check type.

    Args:
        db: :class:`DatabaseManager` used for storing / querying drift
            results.
        data_detector: Optional data-drift detector.
        prediction_detector: Optional prediction-drift detector.
        concept_detector: Optional concept-drift detector.
    """

    def __init__(
        self,
        db: DatabaseManager,
        data_detector: DataDriftDetector | None = None,
        prediction_detector: PredictionDriftDetector | None = None,
        concept_detector: ConceptDriftDetector | None = None,
    ) -> None:
        self._db = db
        self._data_detector = data_detector
        self._prediction_detector = prediction_detector
        self._concept_detector = concept_detector
        self._latest_summary: DriftSummary | None = None

        detectors = [
            name
            for name, det in [
                ("data", data_detector),
                ("prediction", prediction_detector),
                ("concept", concept_detector),
            ]
            if det is not None
        ]
        logger.info(
            "DriftManager initialised with detectors: %s.",
            detectors or "(none)",
        )

    # ---- public API --------------------------------------------------

    def run_drift_check(
        self,
        recent_predictions: list[Any],
        feature_names: list[str],
    ) -> DriftSummary:
        """Execute all configured drift checks on *recent_predictions*.

        Each item in *recent_predictions* is expected to be a
        :class:`~src.data.database.Prediction` ORM row (with
        ``features_json``, ``predicted_label``, ``confidence``, and
        optionally ``true_label``).

        Args:
            recent_predictions: List of ``Prediction`` ORM objects.
            feature_names: Ordered list of feature names matching the
                columns stored in ``features_json``.

        Returns:
            A :class:`DriftSummary` with per-detector results and an
            overall status.
        """
        summary = DriftSummary()

        if not recent_predictions:
            logger.warning("run_drift_check called with an empty prediction list.")
            self._latest_summary = summary
            return summary

        # --- Extract raw arrays from the prediction records -----------
        features_matrix = self._extract_features(recent_predictions, feature_names)
        predicted_labels = np.array(
            [p.predicted_label for p in recent_predictions], dtype=np.int64,
        )
        confidences = np.array(
            [p.confidence for p in recent_predictions], dtype=np.float64,
        )

        # --- 1. Data drift --------------------------------------------
        if self._data_detector is not None:
            try:
                data_result = self._data_detector.check_drift(
                    features_matrix, feature_names,
                )
                summary.data_drift = data_result
                summary.checks_performed.append("data_drift")
            except Exception:
                logger.exception("Data drift check failed.")

        # --- 2. Prediction drift --------------------------------------
        if self._prediction_detector is not None:
            try:
                pred_result = self._prediction_detector.check_drift(
                    predicted_labels, confidences,
                )
                summary.prediction_drift = pred_result
                summary.checks_performed.append("prediction_drift")
            except Exception:
                logger.exception("Prediction drift check failed.")

        # --- 3. Concept drift (requires ground truth) -----------------
        if self._concept_detector is not None:
            true_labels = [
                p.true_label
                for p in recent_predictions
                if p.true_label is not None
            ]
            if true_labels:
                try:
                    # Match predicted labels to only those with ground truth
                    labelled_preds = [
                        p for p in recent_predictions if p.true_label is not None
                    ]
                    y_true = np.array(
                        [p.true_label for p in labelled_preds], dtype=np.int64,
                    )
                    y_pred = np.array(
                        [p.predicted_label for p in labelled_preds], dtype=np.int64,
                    )
                    concept_result = self._concept_detector.check_drift(y_true, y_pred)
                    summary.concept_drift = concept_result
                    summary.checks_performed.append("concept_drift")
                except Exception:
                    logger.exception("Concept drift check failed.")
            else:
                logger.info(
                    "Concept drift check skipped — no ground-truth labels "
                    "available in the prediction window."
                )

        # --- Determine overall status ---------------------------------
        summary.overall_status = self._resolve_overall_status(summary)

        # --- Persist to DB --------------------------------------------
        try:
            self._store_drift_results(summary)
        except Exception:
            logger.exception("Failed to persist drift results.")

        self._latest_summary = summary
        logger.info(
            "Drift check cycle complete: status=%s, checks=%s.",
            summary.overall_status,
            summary.checks_performed,
        )
        return summary

    def get_drift_history(
        self,
        drift_type: str | None = None,
        limit: int = 50,
    ) -> list[DriftResult]:
        """Query historical drift results from the database.

        Args:
            drift_type: Optional filter — one of ``'data'``,
                ``'prediction'``, ``'concept'``.  ``None`` returns all
                types.
            limit: Maximum number of rows to return.

        Returns:
            List of :class:`DriftResult` ORM rows ordered newest-first.
        """
        with self._db.get_session() as session:
            query = session.query(DriftResult)
            if drift_type is not None:
                query = query.filter(DriftResult.drift_type == drift_type)
            results: list[DriftResult] = (
                query.order_by(desc(DriftResult.timestamp))
                .limit(limit)
                .all()
            )
        logger.debug(
            "Retrieved %d drift history rows (type=%s).",
            len(results),
            drift_type or "all",
        )
        return results

    def get_latest_summary(self) -> DriftSummary | None:
        """Return the most recent :class:`DriftSummary`, if available.

        This is an in-memory cache of the last :meth:`run_drift_check`
        result — it does *not* query the database.

        Returns:
            The latest summary or ``None`` if no check has been run yet.
        """
        return self._latest_summary

    # ---- internal helpers --------------------------------------------

    @staticmethod
    def _extract_features(
        predictions: list[Any],
        feature_names: list[str],
    ) -> np.ndarray:
        """Parse ``features_json`` from each prediction into a 2-D array.

        Args:
            predictions: List of ``Prediction`` ORM objects.
            feature_names: Ordered column names to extract.

        Returns:
            ``np.ndarray`` of shape ``(n_predictions, n_features)``.
        """
        rows: list[list[float]] = []
        for pred in predictions:
            feature_dict: dict[str, float] = json.loads(pred.features_json)
            row = [float(feature_dict.get(name, 0.0)) for name in feature_names]
            rows.append(row)
        return np.array(rows, dtype=np.float64)

    @staticmethod
    def _resolve_overall_status(summary: DriftSummary) -> str:
        """Collapse individual check severities into one overall status.

        Returns the *worst* status from all completed checks.
        """
        severities: list[str] = []
        if summary.data_drift is not None:
            severities.append(summary.data_drift.severity)
        if summary.prediction_drift is not None:
            severities.append(summary.prediction_drift.severity)
        if summary.concept_drift is not None:
            severities.append(summary.concept_drift.severity)

        if not severities:
            return "healthy"

        worst = max(severities, key=lambda s: _SEVERITY_ORDER.get(s, 0))
        return "healthy" if worst == "none" else worst

    def _store_drift_results(self, summary: DriftSummary) -> None:
        """Persist individual drift metrics as ``DriftResult`` rows.

        Each sub-result is decomposed into one or more database rows so
        that historical queries can filter by drift type and metric name.
        """
        now = summary.timestamp
        rows: list[DriftResult] = []

        if summary.data_drift is not None:
            d = summary.data_drift
            rows.append(
                DriftResult(
                    timestamp=now,
                    drift_type="data",
                    metric_name="psi",
                    metric_value=d.overall_psi,
                    threshold=0.0,  # overall; per-feature thresholds vary
                    is_breached=d.is_drifted,
                    window_start=now,
                    window_end=now,
                    details_json=json.dumps(
                        {
                            "feature_psi": d.feature_psi,
                            "drifted_features": d.drifted_features,
                            "severity": d.severity,
                        }
                    ),
                )
            )

        if summary.prediction_drift is not None:
            p = summary.prediction_drift
            rows.append(
                DriftResult(
                    timestamp=now,
                    drift_type="prediction",
                    metric_name="hellinger",
                    metric_value=p.hellinger_distance,
                    threshold=0.0,
                    is_breached=p.is_drifted,
                    window_start=now,
                    window_end=now,
                    details_json=json.dumps(
                        {
                            "baseline_positive_rate": p.baseline_positive_rate,
                            "current_positive_rate": p.current_positive_rate,
                            "baseline_mean_confidence": p.baseline_mean_confidence,
                            "current_mean_confidence": p.current_mean_confidence,
                            "severity": p.severity,
                        }
                    ),
                )
            )

        if summary.concept_drift is not None:
            c = summary.concept_drift
            # Store one row per degraded metric for easy querying
            if c.degraded_metrics:
                for metric_name in c.degraded_metrics:
                    drop_val = -c.metric_deltas[metric_name]
                    rows.append(
                        DriftResult(
                            timestamp=now,
                            drift_type="concept",
                            metric_name=f"{metric_name}_drop",
                            metric_value=drop_val,
                            threshold=0.0,
                            is_breached=True,
                            window_start=now,
                            window_end=now,
                            details_json=json.dumps(
                                {
                                    "current_metrics": c.current_metrics,
                                    "baseline_metrics": c.baseline_metrics,
                                    "severity": c.severity,
                                }
                            ),
                        )
                    )
            else:
                # No degradation — store a single healthy row
                rows.append(
                    DriftResult(
                        timestamp=now,
                        drift_type="concept",
                        metric_name="overall",
                        metric_value=0.0,
                        threshold=0.0,
                        is_breached=False,
                        window_start=now,
                        window_end=now,
                        details_json=json.dumps(
                            {
                                "current_metrics": c.current_metrics,
                                "baseline_metrics": c.baseline_metrics,
                                "severity": c.severity,
                            }
                        ),
                    )
                )

        if rows:
            with self._db.get_session() as session:
                session.add_all(rows)
            logger.info("Stored %d drift result rows in the database.", len(rows))
