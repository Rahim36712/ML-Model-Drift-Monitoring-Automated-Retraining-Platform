"""Pydantic schemas for the FastAPI prediction and monitoring API.

Provides validation models for requests and responses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Features input for a single prediction request."""

    features: dict[str, float] = Field(
        ...,
        description="Features mapping (V1-V28 and Amount) and their float values.",
        json_schema_extra={"example": {
            "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
            "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
            "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
            "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
            "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
            "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
            "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
            "Amount": 149.62
        }}
    )


class PredictionResponse(BaseModel):
    """Inference output for a single prediction request."""

    prediction_id: int = Field(..., description="Unique database ID of logged prediction.")
    predicted_label: int = Field(..., description="Predicted label (0=Legitimate, 1=Fraud).")
    confidence: float = Field(..., description="Model probability confidence for positive class (Fraud).")
    model_version: str = Field(..., description="Model version that produced this prediction.")
    latency_ms: float = Field(..., description="Inference latency in milliseconds.")
    timestamp: datetime = Field(..., description="Timestamp of inference.")


class BatchPredictionRequest(BaseModel):
    """Batch features inputs."""

    predictions: list[PredictionRequest] = Field(..., description="List of single prediction requests.")


class BatchPredictionResponse(BaseModel):
    """Batch prediction outputs."""

    predictions: list[PredictionResponse] = Field(..., description="List of single prediction responses.")


class GroundTruthRequest(BaseModel):
    """Ground truth feedback for a prediction event."""

    true_label: int = Field(..., ge=0, le=1, description="Actual class label (0=Legitimate, 1=Fraud).")


class GroundTruthResponse(BaseModel):
    """Status result of posting ground truth."""

    success: bool = Field(..., description="True if ground truth was recorded.")
    prediction_id: int = Field(..., description="ID of the updated prediction.")
    true_label: int = Field(..., description="The label recorded.")


class PredictionStatsResponse(BaseModel):
    """Basic usage and throughput stats."""

    total_count: int = Field(..., description="Total count of logged predictions.")
    avg_latency_ms: float = Field(..., description="Average inference latency in milliseconds.")
    throughput_per_min: float = Field(..., description="Predictions per minute throughput.")
    model_version: str | None = Field(None, description="Active model version string.")


class ModelVersionResponse(BaseModel):
    """Mirror of a model version metadata from DB."""

    version: int
    mlflow_run_id: str
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    auc_roc: float | None
    training_date: datetime
    is_production: bool
    deployed_at: datetime | None


class DriftMetricDetail(BaseModel):
    """Metric detail breakdown."""

    metric_name: str
    value: float
    threshold: float
    is_breached: bool


class DriftSummaryResponse(BaseModel):
    """Drift check summary status."""

    overall_status: str = Field(..., description="Overall health: healthy, warning, critical.")
    timestamp: datetime = Field(..., description="Time of drift check execution.")
    checks_performed: list[str] = Field(..., description="Detectors executed.")
    data_drift_breached: bool = Field(..., description="True if data drift thresholds breached.")
    prediction_drift_breached: bool = Field(..., description="True if prediction drift thresholds breached.")
    concept_drift_breached: bool = Field(..., description="True if concept drift thresholds breached.")
    details: dict[str, Any] = Field(..., description="Detailed raw stats dictionary.")


class AlertResponse(BaseModel):
    """Details of a system alert."""

    id: int
    timestamp: datetime
    severity: str
    drift_type: str
    message: str
    channel: str
    acknowledged: bool


class HealthResponse(BaseModel):
    """FastAPI service health status."""

    status: str = Field(..., description="Service health state (e.g. 'ok').")
    timestamp: datetime = Field(..., description="Current system time.")
    model_version: str | None = Field(None, description="Active production model version.")
    db_connected: bool = Field(..., description="True if database connection is functional.")
