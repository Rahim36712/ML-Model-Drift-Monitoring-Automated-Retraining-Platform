"""Monitoring package for the MLOps Drift Monitor platform.

Provides three specialised drift detectors and a central
:class:`DriftManager` that orchestrates them:

* :class:`DataDriftDetector` – feature-distribution drift (PSI / KL).
* :class:`PredictionDriftDetector` – output-distribution drift
  (Hellinger distance).
* :class:`ConceptDriftDetector` – performance-degradation drift
  (metric drops).
* :class:`DriftManager` – runs all checks, persists results, and
  exposes history queries.
"""

from src.monitoring.concept_drift import ConceptDriftDetector, ConceptDriftResult
from src.monitoring.data_drift import DataDriftDetector, DataDriftResult
from src.monitoring.drift_manager import DriftManager, DriftSummary
from src.monitoring.prediction_drift import (
    PredictionDriftDetector,
    PredictionDriftResult,
)

__all__: list[str] = [
    "ConceptDriftDetector",
    "ConceptDriftResult",
    "DataDriftDetector",
    "DataDriftResult",
    "DriftManager",
    "DriftSummary",
    "PredictionDriftDetector",
    "PredictionDriftResult",
]
