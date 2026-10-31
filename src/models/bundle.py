"""Immutable model packages and a persistent, leakage-free historical split."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import uuid
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.data.loader import DataLoader
from src.runtime import home

FEATURES = DataLoader.get_feature_names()


def fingerprint(X):
    values = np.ascontiguousarray(np.asarray(X, dtype="<f8"))
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in values]


def validate_dataset(frame):
    required = FEATURES + ["Class"]
    if not set(required).issubset(frame.columns):
        raise ValueError("Dataset requires V1–V28, Amount, and Class columns.")
    frame = frame[required].apply(pd.to_numeric, errors="raise").astype(float)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Dataset contains missing or nonfinite values.")
    if not frame.Class.isin([0, 1]).all() or frame.Class.nunique() != 2:
        raise ValueError("Class must contain both binary labels 0 and 1.")
    if frame.Class.value_counts().min() < 25:
        raise ValueError("Dataset needs at least 25 examples of each class.")
    # Duplicate feature rows never cross the historical split.
    if (frame.groupby(FEATURES, dropna=False).Class.nunique() > 1).any():
        raise ValueError("Identical feature rows have conflicting labels.")
    frame = frame.drop_duplicates(subset=FEATURES).reset_index(drop=True)
    if frame.Class.value_counts().min() < 25:
        raise ValueError("Dataset needs at least 25 unique examples of each class")
    return frame


def historical_data():
    path = home() / "historical.joblib"
    external = os.environ.get("MLOPS_DATASET_PATH")
    if path.exists():
        result = joblib.load(path)
        requested = str(Path(external).resolve()) if external else None
        if result.get("dataset_path") != requested:
            raise ValueError("This runtime belongs to another dataset. Select a new MLOPS_HOME.")
        if external and Path(external).exists():
            signature = [Path(external).stat().st_size, Path(external).stat().st_mtime_ns]
            if result.get("dataset_signature") != signature:
                raise ValueError(
                    "Source CSV changed. Use a new MLOPS_HOME to preserve holdout isolation."
                )
        return result
    frame = pd.read_csv(Path(external).resolve()) if external else DataLoader._generate_synthetic()
    frame = validate_dataset(frame)
    X = frame[FEATURES].to_numpy()
    y = frame.Class.to_numpy(dtype=int)
    train_x, test_x, train_y, test_y = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    result = {
        "train_x": train_x,
        "test_x": test_x,
        "train_y": train_y,
        "test_y": test_y,
        "source": "creditcard.csv" if external else "synthetic",
        "dataset_path": str(Path(external).resolve()) if external else None,
        "dataset_signature": [Path(external).stat().st_size, Path(external).stat().st_mtime_ns]
        if external
        else None,
        "holdout_hashes": set(fingerprint(test_x)),
        "training_hashes": set(fingerprint(train_x)),
    }
    temporary = path.with_suffix(".tmp")
    joblib.dump(result, temporary)
    temporary.replace(path)
    return result


def preprocessing():
    return ColumnTransformer([("amount", StandardScaler(), [28])], remainder="passthrough")


@dataclass(frozen=True)
class ModelBundle:
    version: int
    run_id: str
    pipeline: Pipeline
    reference: dict
    baseline_predictions: np.ndarray
    baseline_probabilities: np.ndarray
    baseline_metrics: dict
    source: str
    feature_names: tuple = tuple(FEATURES)


def save_bundle(bundle):
    folder = home() / "bundles"
    folder.mkdir(exist_ok=True)
    path = folder / f"v{bundle.version}.joblib"
    if path.exists():
        raise FileExistsError(f"Bundle {bundle.version} is immutable.")
    temp = folder / f".{uuid.uuid4().hex}.tmp"
    joblib.dump(bundle, temp)
    temp.replace(path)
    return path


def load_bundle(version):
    # Load only locally generated trusted artifacts; never accept arbitrary paths.
    bundle = joblib.load(home() / "bundles" / f"v{int(version)}.joblib")
    if bundle.version != int(version) or bundle.feature_names != tuple(FEATURES):
        raise ValueError("Artifact version or feature schema does not match.")
    probe = bundle.pipeline.predict_proba(np.zeros((1, len(FEATURES))))
    if probe.shape != (1, 2) or not np.isfinite(probe).all():
        raise ValueError("Artifact failed its serving validation.")
    return bundle
