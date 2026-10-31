"""Data-drift detection via PSI and KL Divergence.

Compares incoming feature distributions against a stored reference
(baseline) distribution.  Two complementary metrics are computed per
feature:

* **PSI** – Population Stability Index quantifies how much the
  distribution of a variable has shifted.
* **KL Divergence** – Kullback–Leibler divergence from the reference
  distribution to the current distribution.

Both metrics use quantile-based binning derived from the reference
distribution to guarantee stable bin edges across evaluations.

Usage::

    detector = DataDriftDetector(
        reference_data={"amount": ref_amounts, "age": ref_ages},
        thresholds=settings.drift_thresholds.data_drift,
    )
    result = detector.check_drift(current_matrix, ["amount", "age"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from functools import lru_cache

from src.config.settings import DataDriftThresholds

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Result container
# -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DataDriftResult:
    """Immutable snapshot of a data-drift evaluation.

    Attributes:
        feature_psi: PSI value per feature name.
        overall_psi: Maximum PSI across all features.
        feature_kl: KL divergence per feature name.
        is_drifted: ``True`` when any metric exceeds the warning threshold.
        severity: One of ``'none'``, ``'warning'``, ``'critical'``.
        drifted_features: Feature names whose PSI exceeds the warning
            threshold.
        timestamp: UTC time the check was performed.
    """

    feature_psi: dict[str, float]
    overall_psi: float
    feature_kl: dict[str, float]
    is_drifted: bool
    severity: str
    drifted_features: list[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# -------------------------------------------------------------------
# Detector
# -------------------------------------------------------------------


class DataDriftDetector:
    """Detects feature-level data drift using PSI and KL divergence.

    The detector stores a copy of the reference (training) distribution
    for every feature.  At check-time it bins the incoming data using
    the *same* quantile-derived edges and computes PSI + KL per feature.

    Args:
        reference_data: Mapping of feature name → 1-D numpy array of
            reference observations.
        thresholds: ``DataDriftThresholds`` with ``psi`` and
            ``kl_divergence`` warning / critical pairs.
    """

    _EPSILON: float = 1e-6

    def __init__(
        self,
        reference_data: dict[str, np.ndarray],
        thresholds: DataDriftThresholds,
    ) -> None:
        if not reference_data:
            raise ValueError("reference_data must contain at least one feature.")

        self._reference: dict[str, np.ndarray] = {
            name: np.asarray(arr, dtype=np.float64).ravel()
            for name, arr in reference_data.items()
        }
        self._thresholds = thresholds
        logger.info(
            "DataDriftDetector initialised with %d reference features "
            "(samples per feature ≈ %d).",
            len(self._reference),
            next(iter(self._reference.values())).shape[0],
        )

    # ---- statistical metrics -----------------------------------------

    @staticmethod
    def calculate_psi(
        expected: np.ndarray,
        actual: np.ndarray,
        bins: int = 10,
    ) -> float:
        """Compute the Population Stability Index (PSI).

        Uses quantile-based binning derived from *expected* so that the
        reference distribution has (roughly) uniform bucket densities.

        Args:
            expected: 1-D array of reference observations.
            actual: 1-D array of current observations.
            bins: Number of quantile bins.

        Returns:
            Non-negative PSI value.  Values ≤ 0.1 generally indicate
            no significant shift; 0.1–0.25 is moderate; > 0.25 is large.
        """
        p, q = DataDriftDetector._probabilities(expected, actual, bins)
        return float(np.sum((q - p) * np.log(q / p)))

    @staticmethod
    @lru_cache(maxsize=128)
    def _reference_histogram(raw, bins):
        expected = np.frombuffer(raw, dtype=np.float64)
        if len(expected) == 0 or not np.isfinite(expected).all():
            raise ValueError("Reference samples must be nonempty and finite")
        inner = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)[1:-1]))
        if len(np.unique(expected)) == 1:
            value = expected[0]
            inner = np.array([np.nextafter(value, -np.inf), np.nextafter(value, np.inf)])
        edges = np.r_[-np.inf, inner, np.inf]
        counts = np.histogram(expected, edges)[0].astype(float) + 1e-6
        return edges, counts / counts.sum()

    @staticmethod
    def _probabilities(expected, actual, bins):
        expected = np.asarray(expected, dtype=np.float64).ravel()
        actual = np.asarray(actual, dtype=np.float64).ravel()
        if bins < 2 or len(actual) == 0 or not np.isfinite(actual).all():
            raise ValueError("Current samples must be nonempty and finite; bins >= 2")
        edges, p = DataDriftDetector._reference_histogram(expected.tobytes(), bins)
        counts = np.histogram(actual, edges)[0].astype(float) + 1e-6
        return p, counts / counts.sum()

    @staticmethod
    def calculate_kl_divergence(
        expected: np.ndarray,
        actual: np.ndarray,
        bins: int = 10,
    ) -> float:
        """Compute KL divergence D_KL(P || Q).

        ``P`` is derived from *expected* (reference) and ``Q`` from
        *actual* (current).  Quantile binning is used for consistency
        with :meth:`calculate_psi`.

        Args:
            expected: 1-D array of reference observations.
            actual: 1-D array of current observations.
            bins: Number of quantile bins.

        Returns:
            Non-negative KL divergence value (in nats).
        """
        p, q = DataDriftDetector._probabilities(expected, actual, bins)
        return float(max(0.0, np.sum(p * np.log(p / q))))

    # ---- main entry point --------------------------------------------

    def check_drift(
        self,
        current_data: np.ndarray,
        feature_names: list[str],
    ) -> DataDriftResult:
        """Run PSI and KL divergence for every feature and assess drift.

        Args:
            current_data: 2-D array of shape ``(n_samples, n_features)``
                with the same column ordering as *feature_names*.
            feature_names: Column names matching the reference data keys.

        Returns:
            A :class:`DataDriftResult` summarising per-feature metrics,
            overall severity, and which features drifted.

        Raises:
            ValueError: If *feature_names* contains an unknown feature.
        """
        current_data = np.asarray(current_data, dtype=np.float64)
        if current_data.size == 0 or not np.isfinite(current_data).all():
            raise ValueError("Current feature samples must be nonempty and finite")
        if current_data.ndim == 1:
            current_data = current_data.reshape(-1, 1)

        if current_data.ndim != 2 or current_data.shape[1] != len(feature_names):
            raise ValueError("Feature matrix shape does not match schema")
        if set(feature_names) - set(self._reference):
            raise ValueError("Unknown reference feature")
        feature_psi: dict[str, float] = {}
        feature_kl: dict[str, float] = {}
        drifted_features: list[str] = []

        for idx, name in enumerate(feature_names):
            if name not in self._reference:
                logger.warning(
                    "Feature '%s' not found in reference data – skipping.", name
                )
                continue

            ref = self._reference[name]
            cur = current_data[:, idx]

            psi_val = self.calculate_psi(ref, cur)
            kl_val = self.calculate_kl_divergence(ref, cur)

            feature_psi[name] = psi_val
            feature_kl[name] = kl_val

            if psi_val >= self._thresholds.psi.warning or kl_val >= self._thresholds.kl_divergence.warning:
                drifted_features.append(name)

        overall_psi = max(feature_psi.values()) if feature_psi else 0.0

        # Determine severity from the worst-case PSI
        if overall_psi >= self._thresholds.psi.critical or max(feature_kl.values(), default=0) >= self._thresholds.kl_divergence.critical:
            severity = "critical"
        elif overall_psi >= self._thresholds.psi.warning or max(feature_kl.values(), default=0) >= self._thresholds.kl_divergence.warning:
            severity = "warning"
        else:
            severity = "none"

        is_drifted = severity != "none"

        result = DataDriftResult(
            feature_psi=feature_psi,
            overall_psi=overall_psi,
            feature_kl=feature_kl,
            is_drifted=is_drifted,
            severity=severity,
            drifted_features=drifted_features,
        )

        logger.info(
            "Data drift check complete: severity=%s, overall_psi=%.4f, "
            "drifted=%d/%d features.",
            severity,
            overall_psi,
            len(drifted_features),
            len(feature_psi),
        )
        return result

    # ---- utilities ---------------------------------------------------

    @staticmethod
    def get_top_drifted_features(
        result: DataDriftResult,
        n: int = 5,
    ) -> list[tuple[str, float]]:
        """Return the *n* features with the highest PSI.

        Args:
            result: Output of :meth:`check_drift`.
            n: Number of features to return.

        Returns:
            List of ``(feature_name, psi_value)`` tuples sorted descending.
        """
        sorted_features = sorted(
            result.feature_psi.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        return sorted_features[:n]
