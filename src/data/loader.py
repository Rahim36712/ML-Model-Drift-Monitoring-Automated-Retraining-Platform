"""Dataset loading and preprocessing for fraud detection.

Handles loading the credit card fraud dataset (real or synthetic),
preprocessing with standardisation, and persisting reference distributions
and fitted preprocessors for downstream drift detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Column constants
# ---------------------------------------------------------------------------
_V_COLUMNS: list[str] = [f"V{i}" for i in range(1, 29)]
_FEATURE_COLUMNS: list[str] = _V_COLUMNS + ["Amount"]
_TARGET_COLUMN: str = "Class"


class DataLoader:
    """Load, preprocess, and persist fraud-detection datasets.

    Supports both the real Kaggle *creditcard.csv* file and an
    auto-generated synthetic substitute produced via
    ``sklearn.datasets.make_classification``.

    Args:
        data_dir: Root data directory.  Defaults to ``<PROJECT_ROOT>/data``.
    """

    def __init__(self, data_dir: Path = PROJECT_ROOT / "data") -> None:
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.reference_dir = self.data_dir / "reference"

        # Eagerly create directory tree so downstream code never fails on
        # missing parents.
        for d in (self.raw_dir, self.processed_dir, self.reference_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def load_dataset(self) -> pd.DataFrame:
        """Load the credit-card fraud dataset.

        Attempts to read ``data/raw/creditcard.csv``.  When the file is not
        present a synthetic dataset with similar statistical properties is
        generated instead.

        Returns:
            DataFrame with columns V1–V28, Amount, and Class.
        """
        csv_path = self.raw_dir / "creditcard.csv"

        if csv_path.exists():
            logger.info("Loading real dataset from %s", csv_path)
            df = pd.read_csv(csv_path)
            # Keep only the columns we need (drop Time if present)
            keep = [c for c in _FEATURE_COLUMNS + [_TARGET_COLUMN] if c in df.columns]
            return df[keep]

        logger.info("Real dataset not found – generating synthetic data.")
        print(
            "Using synthetic dataset. For real data, download creditcard.csv "
            "from Kaggle to data/raw/"
        )
        return self._generate_synthetic()

    @staticmethod
    def _generate_synthetic() -> pd.DataFrame:
        """Create a synthetic fraud dataset via ``make_classification``.

        The generator mirrors the real dataset's properties: 29 numeric
        features, binary target, and a heavy class imbalance (98 % / 2 %).

        Returns:
            DataFrame shaped identically to the real creditcard.csv.
        """
        X, y = make_classification(
            n_samples=10_000,
            n_features=29,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            weights=[0.98, 0.02],
            random_state=42,
        )

        columns = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        df = pd.DataFrame(X, columns=columns)
        df[_TARGET_COLUMN] = y

        logger.info(
            "Synthetic dataset created: %d samples, %.1f%% fraud",
            len(df),
            y.mean() * 100,
        )
        return df

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Standardise features and split into train / test sets.

        * The *Amount* column is scaled via ``StandardScaler``.
        * A stratified 80 / 20 split preserves class balance.
        * The fitted scaler is persisted for later reuse.

        Args:
            df: Raw DataFrame returned by :meth:`load_dataset`.

        Returns:
            Tuple of ``(X_train, X_test, y_train, y_test, feature_names)``.
        """
        feature_names: list[str] = [c for c in _FEATURE_COLUMNS if c in df.columns]
        X = df[feature_names].values.copy()
        y = df[_TARGET_COLUMN].values.copy()

        # Scale the Amount column (last feature column)
        amount_idx = feature_names.index("Amount")
        scaler = StandardScaler()
        X[:, amount_idx] = scaler.fit_transform(X[:, amount_idx].reshape(-1, 1)).ravel()

        # Persist the fitted scaler
        self.save_preprocessor(scaler)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )

        self.save_processed_data(X_train, X_test, y_train, y_test, feature_names)

        logger.info(
            "Preprocessing complete - train=%d, test=%d, features=%d",
            X_train.shape[0],
            X_test.shape[0],
            X_train.shape[1],
        )
        return X_train, X_test, y_train, y_test, feature_names

    def save_processed_data(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: list[str],
    ) -> None:
        """Persist train/test arrays for scripts and retraining jobs."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        path = self.processed_dir / "dataset.npz"
        np.savez_compressed(
            path,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=np.array(feature_names, dtype=object),
        )
        logger.info("Saved processed dataset to %s", path)

    def load_processed_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Load arrays saved by :meth:`save_processed_data`."""
        path = self.processed_dir / "dataset.npz"
        if not path.exists():
            raise FileNotFoundError(f"No processed dataset found at {path}")

        data = np.load(path, allow_pickle=True)
        feature_names = [str(name) for name in data["feature_names"].tolist()]
        return (
            data["X_train"],
            data["X_test"],
            data["y_train"],
            data["y_test"],
            feature_names,
        )

    # ------------------------------------------------------------------
    # Reference distributions
    # ------------------------------------------------------------------

    def save_reference_distribution(
        self,
        X_train: np.ndarray,
        feature_names: list[str],
    ) -> None:
        """Persist per-feature and combined reference distributions.

        One ``.npy`` file is written per feature plus a combined
        ``combined.npy`` containing the full training matrix.

        Args:
            X_train: Training feature matrix.
            feature_names: Ordered list of feature names matching columns.
        """
        self.reference_dir.mkdir(parents=True, exist_ok=True)

        for idx, name in enumerate(feature_names):
            np.save(self.reference_dir / f"{name}.npy", X_train[:, idx])

        np.save(self.reference_dir / "combined.npy", X_train)

        logger.info(
            "Saved reference distributions for %d features to %s",
            len(feature_names),
            self.reference_dir,
        )

    def create_reference_distribution(
        self,
        X_train: np.ndarray,
        feature_names: list[str],
    ) -> None:
        """Compatibility alias used by the implementation plan."""
        self.save_reference_distribution(X_train, feature_names)

    def load_reference_distribution(self) -> dict[str, np.ndarray]:
        """Load previously saved reference distributions.

        Returns:
            Dictionary mapping feature names (and ``'combined'``) to their
            reference arrays.

        Raises:
            FileNotFoundError: If no ``.npy`` files exist in the reference
                directory.
        """
        npy_files = list(self.reference_dir.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(
                f"No reference distributions found in {self.reference_dir}"
            )

        distributions: dict[str, np.ndarray] = {}
        for fpath in sorted(npy_files):
            distributions[fpath.stem] = np.load(fpath)

        logger.info("Loaded %d reference distributions", len(distributions))
        return distributions

    # ------------------------------------------------------------------
    # Preprocessor persistence
    # ------------------------------------------------------------------

    def save_preprocessor(self, scaler: Any) -> None:
        """Persist a fitted ``StandardScaler`` to disk.

        Args:
            scaler: Fitted scikit-learn scaler instance.
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        path = self.processed_dir / "scaler.joblib"
        joblib.dump(scaler, path)
        logger.info("Saved preprocessor to %s", path)

    def load_preprocessor(self) -> Any:
        """Load a previously saved preprocessor.

        Returns:
            The fitted scaler object.

        Raises:
            FileNotFoundError: If the scaler file does not exist.
        """
        path = self.processed_dir / "scaler.joblib"
        if not path.exists():
            raise FileNotFoundError(f"No preprocessor found at {path}")

        scaler = joblib.load(path)
        logger.info("Loaded preprocessor from %s", path)
        return scaler

    @staticmethod
    def get_feature_names() -> list[str]:
        """Return the canonical fraud-detection feature order."""
        return list(_FEATURE_COLUMNS)
