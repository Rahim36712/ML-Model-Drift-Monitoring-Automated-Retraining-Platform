"""Data layer for the MLOps Drift Monitor platform.

Provides data loading/preprocessing utilities, SQLAlchemy ORM models,
a ``DatabaseManager`` for connection lifecycle management, and a
``PredictionLogger`` for recording inference events and ground-truth labels.
"""

from src.data.loader import DataLoader
from src.data.drift_injector import DriftInjector
from src.data.database import (
    Alert,
    Base,
    DatabaseManager,
    DriftResult,
    ModelVersion,
    Prediction,
    RetrainingEvent,
    get_database,
)
from src.data.logger import PredictionLogger

__all__: list[str] = [
    "Alert",
    "Base",
    "DatabaseManager",
    "DataLoader",
    "DriftInjector",
    "DriftResult",
    "ModelVersion",
    "Prediction",
    "PredictionLogger",
    "RetrainingEvent",
    "get_database",
]
