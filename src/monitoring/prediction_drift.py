"""Prediction-drift detection via Hellinger distance.

Monitors for shifts in the model's *output* distribution — i.e. changes
in the predicted-label mix or confidence-score distribution — independent
of any ground-truth labels.

The primary metric is the **Hellinger distance** between the baseline
and current confidence-score histograms.  Auxiliary metrics track the
positive-prediction rate and mean confidence.

Usage::

    detector = PredictionDriftDetector(
        baseline_predictions=train_labels,
        baseline_confidences=train_probs,
        thresholds=settings.drift_thresholds.prediction_drift,
    )
    result = detector.check_drift(new_labels, new_probs)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from src.config.settings import PredictionDriftThresholds

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Result container
# -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PredictionDriftResult:
    """Immutable snapshot of a prediction-drift evaluation.

    Attributes:
        hellinger_distance: Hellinger distance between baseline and
            current confidence distributions.
        baseline_positive_rate: Fraction of positive predictions in the
            baseline set.
        current_positive_rate: Fraction of positive predictions in the
            current batch.
        baseline_mean_confidence: Mean predicted probability in baseline.
        current_mean_confidence: Mean predicted probability in current
            batch.
        is_drifted: ``True`` when the Hellinger distance exceeds the
            warning threshold.
        severity: One of ``'none'``, ``'warning'``, ``'critical'``.
        timestamp: UTC time the check was performed.
    """

    hellinger_distance: float
    baseline_positive_rate: float
    current_positive_rate: float
    baseline_mean_confidence: float
    current_mean_confidence: float
    is_drifted: bool
    severity: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# -------------------------------------------------------------------
# Detector
# -------------------------------------------------------------------


class PredictionDriftDetector:
    """Detects shifts in the model output distribution.

    Stores the baseline prediction labels and confidence scores and
    compares incoming batches using the Hellinger distance.

    Args:
        baseline_predictions: 1-D array of predicted labels (0/1) from
            the baseline period.
        baseline_confidences: 1-D array of predicted probabilities from
            the baseline period.
        thresholds: ``PredictionDriftThresholds`` containing ``hellinger``
            and ``distribution_shift`` warning / critical pairs.
    """

    def __init__(
        self,
        baseline_predictions: np.ndarray,
        baseline_confidences: np.ndarray,
        thresholds: PredictionDriftThresholds,
    ) -> None:
        self._baseline_predictions = np.asarray(
            baseline_predictions, dtype=np.int64,
        ).ravel()
        self._baseline_confidences = np.asarray(
            baseline_confidences, dtype=np.float64,
        ).ravel()
        self._thresholds = thresholds

        self._baseline_positive_rate: float = float(
            self._baseline_predictions.mean()
        )
        self._baseline_mean_confidence: float = float(
            self._baseline_confidences.mean()
        )

        logger.info(
            "PredictionDriftDetector initialised (baseline samples=%d, "
            "positive_rate=%.4f, mean_conf=%.4f).",
            len(self._baseline_predictions),
            self._baseline_positive_rate,
            self._baseline_mean_confidence,
        )

    # ---- statistical metric ------------------------------------------

    @staticmethod
    def calculate_hellinger(
        p: np.ndarray,
        q: np.ndarray,
        bins: int = 20,
    ) -> float:
        """Compute the Hellinger distance between two sample sets.

        Both arrays are first binned into normalised histograms over a
        shared support.  The Hellinger distance is then:

        .. math::

            H(P, Q) = \\frac{1}{\\sqrt{2}}
            \\sqrt{\\sum_{i} \\bigl(\\sqrt{p_i} - \\sqrt{q_i}\\bigr)^2}

        Args:
            p: 1-D array of observations from distribution *P*.
            q: 1-D array of observations from distribution *Q*.
            bins: Number of histogram bins.

        Returns:
            Hellinger distance in [0, 1].
        """
        p = np.asarray(p, dtype=np.float64).ravel()
        q = np.asarray(q, dtype=np.float64).ravel()

        # Shared bin edges over the union of both supports
        combined_min = min(float(p.min()), float(q.min()))
        combined_max = max(float(p.max()), float(q.max()))
        bin_edges = np.linspace(combined_min, combined_max, bins + 1)

        p_hist = np.histogram(p, bins=bin_edges)[0].astype(np.float64)
        q_hist = np.histogram(q, bins=bin_edges)[0].astype(np.float64)

        # Normalise to probability distributions
        p_hist = p_hist / p_hist.sum() if p_hist.sum() > 0 else p_hist
        q_hist = q_hist / q_hist.sum() if q_hist.sum() > 0 else q_hist

        hellinger: float = float(
            (1.0 / np.sqrt(2.0))
            * np.sqrt(np.sum((np.sqrt(p_hist) - np.sqrt(q_hist)) ** 2))
        )
        return hellinger

    # ---- main entry point --------------------------------------------

    def check_drift(
        self,
        current_predictions: np.ndarray,
        current_confidences: np.ndarray,
    ) -> PredictionDriftResult:
        """Compare current predictions against the baseline.

        Args:
            current_predictions: 1-D array of predicted labels (0/1).
            current_confidences: 1-D array of predicted probabilities.

        Returns:
            A :class:`PredictionDriftResult` summarising the assessment.
        """
        current_predictions = np.asarray(current_predictions, dtype=np.int64).ravel()
        current_confidences = np.asarray(current_confidences, dtype=np.float64).ravel()

        hellinger = self.calculate_hellinger(
            self._baseline_confidences,
            current_confidences,
        )

        current_positive_rate = float(current_predictions.mean())
        current_mean_confidence = float(current_confidences.mean())

        # Severity assessment
        if hellinger >= self._thresholds.hellinger.critical:
            severity = "critical"
        elif hellinger >= self._thresholds.hellinger.warning:
            severity = "warning"
        else:
            severity = "none"

        is_drifted = severity != "none"

        result = PredictionDriftResult(
            hellinger_distance=hellinger,
            baseline_positive_rate=self._baseline_positive_rate,
            current_positive_rate=current_positive_rate,
            baseline_mean_confidence=self._baseline_mean_confidence,
            current_mean_confidence=current_mean_confidence,
            is_drifted=is_drifted,
            severity=severity,
        )

        logger.info(
            "Prediction drift check complete: severity=%s, "
            "hellinger=%.4f, positive_rate_change=%.4f.",
            severity,
            hellinger,
            current_positive_rate - self._baseline_positive_rate,
        )
        return result

    # ---- utilities ---------------------------------------------------

    @staticmethod
    def get_prediction_rate_change(result: PredictionDriftResult) -> float:
        """Return the percentage change in positive-prediction rate.

        A positive value means the model is predicting the positive class
        *more* often than during the baseline period.

        Args:
            result: Output of :meth:`check_drift`.

        Returns:
            Percentage change (e.g. ``12.5`` means +12.5 %).
        """
        if result.baseline_positive_rate == 0.0:
            return 0.0
        change = (
            (result.current_positive_rate - result.baseline_positive_rate)
            / result.baseline_positive_rate
            * 100.0
        )
        return change
