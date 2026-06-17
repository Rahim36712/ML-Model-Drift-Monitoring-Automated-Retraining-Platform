"""Synthetic drift injection engine for testing drift detection.

Provides controlled methods for injecting various types of data and concept
drift into existing datasets.  Every method returns a **copy** of the input
data – the originals are never mutated.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# Supported scenario names used by :meth:`DriftInjector.create_drifted_dataset`
DriftScenario = Literal[
    "feature_shift",
    "scale_change",
    "noise",
    "label_flip",
    "class_imbalance",
    "severe",
]


class DriftInjector:
    """Inject controlled drift into feature matrices and label vectors.

    All public methods accept NumPy arrays and return **copies** so that the
    caller's original data is never modified.  This is critical for A/B
    comparisons between clean and drifted datasets.

    Example::

        injector = DriftInjector()
        X_drifted, y_drifted = injector.create_drifted_dataset(X, y, "severe")
    """

    # ------------------------------------------------------------------
    # Feature-level drift
    # ------------------------------------------------------------------

    @staticmethod
    def inject_feature_shift(
        X: np.ndarray,
        feature_indices: list[int] | None = None,
        shift_magnitude: float = 1.5,
    ) -> np.ndarray:
        """Add a constant shift to selected features.

        Simulates covariate drift where upstream data sources change their
        mean (e.g. a sensor re-calibration or currency conversion change).

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            feature_indices: Column indices to shift.  Defaults to the
                first five features.
            shift_magnitude: Value added to each selected column.

        Returns:
            A shifted copy of *X*.
        """
        X_shifted = X.copy()
        if feature_indices is None:
            feature_indices = list(range(min(5, X.shape[1])))

        X_shifted[:, feature_indices] += shift_magnitude
        logger.info(
            "Injected feature shift (magnitude=%.2f) on %d features",
            shift_magnitude,
            len(feature_indices),
        )
        return X_shifted

    @staticmethod
    def inject_scale_change(
        X: np.ndarray,
        feature_indices: list[int] | None = None,
        scale_factor: float = 3.0,
    ) -> np.ndarray:
        """Multiply selected features by a scale factor.

        Models variance drift – e.g. a change in measurement units or
        increased noise in an upstream pipeline.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            feature_indices: Column indices to scale.  Defaults to the
                first five features.
            scale_factor: Multiplicative factor applied to each selected
                column.

        Returns:
            A rescaled copy of *X*.
        """
        X_scaled = X.copy()
        if feature_indices is None:
            feature_indices = list(range(min(5, X.shape[1])))

        X_scaled[:, feature_indices] *= scale_factor
        logger.info(
            "Injected scale change (factor=%.2f) on %d features",
            scale_factor,
            len(feature_indices),
        )
        return X_scaled

    @staticmethod
    def inject_noise(
        X: np.ndarray,
        noise_std: float = 0.5,
    ) -> np.ndarray:
        """Add isotropic Gaussian noise to every feature.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            noise_std: Standard deviation of the Gaussian noise.

        Returns:
            A noisy copy of *X*.
        """
        rng = np.random.default_rng(seed=42)
        noise = rng.normal(loc=0.0, scale=noise_std, size=X.shape)
        X_noisy = X.copy() + noise
        logger.info("Injected Gaussian noise (std=%.2f) on all features", noise_std)
        return X_noisy

    # ------------------------------------------------------------------
    # Label-level drift (concept drift proxy)
    # ------------------------------------------------------------------

    @staticmethod
    def inject_label_flip(
        y: np.ndarray,
        flip_ratio: float = 0.15,
    ) -> np.ndarray:
        """Randomly flip a fraction of labels.

        Simulates concept drift where the decision boundary has shifted
        and ground-truth labels no longer match the original relationship.

        Args:
            y: Label vector of shape ``(n_samples,)``.
            flip_ratio: Proportion of labels to flip, in ``[0, 1]``.

        Returns:
            A copy of *y* with flipped labels.
        """
        rng = np.random.default_rng(seed=42)
        y_flipped = y.copy()

        n_flip = int(len(y) * flip_ratio)
        flip_indices = rng.choice(len(y), size=n_flip, replace=False)

        # Binary flip: 0 ↔ 1
        y_flipped[flip_indices] = 1 - y_flipped[flip_indices]

        logger.info(
            "Flipped %d / %d labels (%.1f%%)",
            n_flip,
            len(y),
            flip_ratio * 100,
        )
        return y_flipped

    # ------------------------------------------------------------------
    # Temporal / gradual drift
    # ------------------------------------------------------------------

    @staticmethod
    def inject_gradual_drift(
        X: np.ndarray,
        n_batches: int = 10,
        max_shift: float = 2.0,
    ) -> list[np.ndarray]:
        """Split data into batches with linearly increasing feature shift.

        Useful for simulating slow, real-world distribution drift.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            n_batches: Number of time-ordered batches to produce.
            max_shift: Maximum additive shift applied to the final batch.

        Returns:
            List of ``n_batches`` arrays, each containing a slice of *X*
            with progressively larger drift.
        """
        batch_size = len(X) // n_batches
        batches: list[np.ndarray] = []

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size if i < n_batches - 1 else len(X)
            batch = X[start:end].copy()

            # Linearly interpolate the shift magnitude
            shift = max_shift * (i / max(n_batches - 1, 1))
            batch += shift

            batches.append(batch)

        logger.info(
            "Created %d batches with gradual drift (max_shift=%.2f)",
            n_batches,
            max_shift,
        )
        return batches

    # ------------------------------------------------------------------
    # Class-balance drift
    # ------------------------------------------------------------------

    @staticmethod
    def inject_class_imbalance_shift(
        X: np.ndarray,
        y: np.ndarray,
        target_ratio: float = 0.15,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Oversample the minority class to reach a target ratio.

        This simulates a production scenario where the fraud rate suddenly
        increases.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Label vector of shape ``(n_samples,)``.
            target_ratio: Desired fraction of the minority class.

        Returns:
            Tuple ``(X_resampled, y_resampled)`` with oversampled minority.
        """
        rng = np.random.default_rng(seed=42)

        minority_mask = y == 1
        majority_mask = ~minority_mask

        X_minority = X[minority_mask]
        y_minority = y[minority_mask]
        X_majority = X[majority_mask]
        y_majority = y[majority_mask]

        n_majority = len(y_majority)
        n_target_minority = int(n_majority * target_ratio / (1 - target_ratio))

        if n_target_minority <= len(y_minority):
            logger.warning(
                "Target ratio (%.2f) already met – returning copy of original data.",
                target_ratio,
            )
            return X.copy(), y.copy()

        # Oversample minority with replacement
        extra_needed = n_target_minority - len(y_minority)
        extra_indices = rng.choice(len(y_minority), size=extra_needed, replace=True)

        X_resampled = np.vstack([X_majority, X_minority, X_minority[extra_indices]])
        y_resampled = np.concatenate(
            [y_majority, y_minority, y_minority[extra_indices]]
        )

        logger.info(
            "Class imbalance shifted: minority %.1f%% → %.1f%%  (n=%d → %d)",
            minority_mask.mean() * 100,
            target_ratio * 100,
            len(y),
            len(y_resampled),
        )
        return X_resampled, y_resampled

    # ------------------------------------------------------------------
    # Composite scenarios
    # ------------------------------------------------------------------

    def create_drifted_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scenario: DriftScenario,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply a named drift scenario to a dataset.

        Supported scenarios:

        * ``feature_shift`` – constant additive shift on the first 5 features.
        * ``scale_change`` – multiplicative scaling on the first 5 features.
        * ``noise`` – Gaussian noise added to all features.
        * ``label_flip`` – random label corruption.
        * ``class_imbalance`` – minority-class oversampling.
        * ``severe`` – combines *feature_shift*, *label_flip*, and *noise*
          for a worst-case stress test.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Label vector of shape ``(n_samples,)``.
            scenario: One of the supported scenario names.

        Returns:
            Tuple ``(X_drifted, y_drifted)``.

        Raises:
            ValueError: If *scenario* is not recognised.
        """
        logger.info("Applying drift scenario: %s", scenario)

        if scenario == "feature_shift":
            return self.inject_feature_shift(X), y.copy()

        if scenario == "scale_change":
            return self.inject_scale_change(X), y.copy()

        if scenario == "noise":
            return self.inject_noise(X), y.copy()

        if scenario == "label_flip":
            return X.copy(), self.inject_label_flip(y)

        if scenario == "class_imbalance":
            return self.inject_class_imbalance_shift(X, y)

        if scenario == "severe":
            X_drifted = self.inject_feature_shift(X)
            X_drifted = self.inject_noise(X_drifted)
            y_drifted = self.inject_label_flip(y)
            return X_drifted, y_drifted

        raise ValueError(
            f"Unknown drift scenario '{scenario}'.  "
            f"Choose from: feature_shift, scale_change, noise, "
            f"label_flip, class_imbalance, severe."
        )
