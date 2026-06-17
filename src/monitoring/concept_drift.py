"""Concept-drift detection via performance-metric degradation.

Concept drift occurs when the relationship between the input features
and the target variable changes — i.e. the model's learned mapping
becomes stale even though the input distribution may be stable.

Detection is performed by comparing current classification metrics
(accuracy, F1, precision, recall) computed from a labelled window
against stored baseline metrics.  If any metric drops by more than the
configured threshold the window is flagged as drifted.

Usage::

    detector = ConceptDriftDetector(
        baseline_metrics={"accuracy": 0.95, "f1": 0.90, ...},
        thresholds=settings.drift_thresholds.concept_drift,
    )
    result = detector.check_drift(y_true, y_pred)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.config.settings import ConceptDriftThresholds

logger = logging.getLogger(__name__)

# Required baseline metric keys
_REQUIRED_METRIC_KEYS = frozenset({"accuracy", "f1", "precision", "recall"})


# -------------------------------------------------------------------
# Result container
# -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConceptDriftResult:
    """Immutable snapshot of a concept-drift evaluation.

    Attributes:
        current_metrics: Newly computed classification metrics.
        baseline_metrics: Reference metrics the detector was initialised
            with.
        metric_deltas: ``current - baseline`` for each metric (negative
            values mean degradation).
        is_drifted: ``True`` when any metric drop exceeds the warning
            threshold.
        degraded_metrics: Names of metrics that dropped beyond their
            respective thresholds.
        severity: One of ``'none'``, ``'warning'``, ``'critical'``.
        timestamp: UTC time the check was performed.
    """

    current_metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    metric_deltas: dict[str, float]
    is_drifted: bool
    degraded_metrics: list[str]
    severity: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# -------------------------------------------------------------------
# Detector
# -------------------------------------------------------------------


class ConceptDriftDetector:
    """Detects concept drift by monitoring performance metric drops.

    Args:
        baseline_metrics: Dictionary with keys ``accuracy``, ``f1``,
            ``precision``, ``recall`` representing the reference
            (training/validation) performance.
        thresholds: ``ConceptDriftThresholds`` containing per-metric
            ``ThresholdPair`` objects (warning / critical).

    Raises:
        ValueError: If *baseline_metrics* is missing a required key.
    """

    def __init__(
        self,
        baseline_metrics: dict[str, float],
        thresholds: ConceptDriftThresholds,
    ) -> None:
        missing = _REQUIRED_METRIC_KEYS - set(baseline_metrics)
        if missing:
            raise ValueError(
                f"baseline_metrics is missing required keys: {sorted(missing)}"
            )

        self._baseline = {k: float(v) for k, v in baseline_metrics.items()}
        self._thresholds = thresholds

        # Map metric name → corresponding ThresholdPair on the config
        self._threshold_map: dict[str, tuple[float, float]] = {
            "accuracy": (
                thresholds.accuracy_drop.warning,
                thresholds.accuracy_drop.critical,
            ),
            "f1": (
                thresholds.f1_drop.warning,
                thresholds.f1_drop.critical,
            ),
            "precision": (
                thresholds.precision_drop.warning,
                thresholds.precision_drop.critical,
            ),
            "recall": (
                thresholds.recall_drop.warning,
                thresholds.recall_drop.critical,
            ),
        }

        logger.info(
            "ConceptDriftDetector initialised (baseline: %s).",
            ", ".join(f"{k}={v:.4f}" for k, v in self._baseline.items()),
        )

    # ---- metric calculation ------------------------------------------

    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """Compute standard binary-classification metrics.

        Args:
            y_true: 1-D array of ground-truth labels (0/1).
            y_pred: 1-D array of predicted labels (0/1).

        Returns:
            Dictionary with keys ``accuracy``, ``f1``, ``precision``,
            ``recall``.
        """
        y_true = np.asarray(y_true, dtype=np.int64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.int64).ravel()

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0.0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0.0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0.0)),
        }

    # ---- main entry point --------------------------------------------

    def check_drift(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> ConceptDriftResult:
        """Evaluate concept drift from labelled predictions.

        Computes current metrics, calculates deltas against the baseline,
        and classifies severity based on the configured thresholds.

        Args:
            y_true: 1-D array of ground-truth labels.
            y_pred: 1-D array of predicted labels.

        Returns:
            A :class:`ConceptDriftResult` with per-metric deltas and an
            overall severity assessment.
        """
        current = self.calculate_metrics(y_true, y_pred)

        deltas: dict[str, float] = {
            k: current[k] - self._baseline[k] for k in _REQUIRED_METRIC_KEYS
        }

        degraded_metrics: list[str] = []
        worst_severity = "none"

        for metric_name in _REQUIRED_METRIC_KEYS:
            drop = -deltas[metric_name]  # positive = degradation
            warn_thresh, crit_thresh = self._threshold_map[metric_name]

            if drop >= crit_thresh:
                degraded_metrics.append(metric_name)
                worst_severity = "critical"
            elif drop >= warn_thresh:
                degraded_metrics.append(metric_name)
                if worst_severity != "critical":
                    worst_severity = "warning"

        is_drifted = worst_severity != "none"

        result = ConceptDriftResult(
            current_metrics=current,
            baseline_metrics=dict(self._baseline),
            metric_deltas=deltas,
            is_drifted=is_drifted,
            degraded_metrics=sorted(degraded_metrics),
            severity=worst_severity,
        )

        logger.info(
            "Concept drift check complete: severity=%s, degraded=%s, "
            "deltas=[%s].",
            worst_severity,
            sorted(degraded_metrics) or "none",
            ", ".join(f"{k}={v:+.4f}" for k, v in deltas.items()),
        )
        return result
