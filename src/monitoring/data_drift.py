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
        epsilon = DataDriftDetector._EPSILON

        # Quantile edges from the reference distribution
        quantiles = np.linspace(0.0, 1.0, bins + 1)
        bin_edges = np.quantile(expected, quantiles)
        # Ensure unique edges (collapse duplicates) so np.histogram works
        bin_edges = np.unique(bin_edges)

        expected_counts = np.histogram(expected, bins=bin_edges)[0].astype(np.float64)
        actual_counts = np.histogram(actual, bins=bin_edges)[0].astype(np.float64)

        expected_pct = expected_counts / expected_counts.sum() + epsilon
        actual_pct = actual_counts / actual_counts.sum() + epsilon

        psi: float = float(
            np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        )
        return psi

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
        epsilon = DataDriftDetector._EPSILON

        quantiles = np.linspace(0.0, 1.0, bins + 1)
        bin_edges = np.quantile(expected, quantiles)
        bin_edges = np.unique(bin_edges)

        p_counts = np.histogram(expected, bins=bin_edges)[0].astype(np.float64)
        q_counts = np.histogram(actual, bins=bin_edges)[0].astype(np.float64)

        p = p_counts / p_counts.sum() + epsilon
        q = q_counts / q_counts.sum() + epsilon

        kl: float = float(np.sum(p * np.log(p / q)))
        return kl

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
        if current_data.ndim == 1:
            current_data = current_data.reshape(-1, 1)

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

            if psi_val >= self._thresholds.psi.warning:
                drifted_features.append(name)

        overall_psi = max(feature_psi.values()) if feature_psi else 0.0

        # Determine severity from the worst-case PSI
        if overall_psi >= self._thresholds.psi.critical:
            severity = "critical"
        elif overall_psi >= self._thresholds.psi.warning:
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
