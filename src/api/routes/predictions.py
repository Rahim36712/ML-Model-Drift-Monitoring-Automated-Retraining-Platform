"""Prediction endpoints for single, batch inference, and ground-truth submission."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.data.database import DatabaseManager
from src.data.logger import PredictionLogger
from src.data.loader import DataLoader
from src.pipeline.deployer import ModelProvider
from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    GroundTruthRequest,
    GroundTruthResponse,
    PredictionStatsResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Predictions"])


def get_db_manager() -> DatabaseManager:
    from src.data.database import get_database
    return get_database()


def get_logger(db: DatabaseManager = Depends(get_db_manager)) -> PredictionLogger:
    return PredictionLogger(db)


@router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict(
    request: PredictionRequest,
    db: DatabaseManager = Depends(get_db_manager),
    pred_logger: PredictionLogger = Depends(get_logger),
) -> dict[str, Any]:
    """Generate a prediction for a single transaction feature vector."""
    model, preprocessor, version, _ = ModelProvider.get_active_model()
    
    if model is None or preprocessor is None:
        logger.error("Inference request failed: model or preprocessor not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Active model is not loaded in memory. Ensure baseline is trained and deployed.",
        )

    start_time = time.perf_counter()

    try:
        # Extract features and sort to canonical order
        feature_names = DataLoader.get_feature_names()
        raw_features = request.features
        
        # Validate that all features exist
        missing_features = [f for f in feature_names if f not in raw_features]
        if missing_features:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Missing feature inputs: {missing_features}",
            )

        # Build feature vector
        vector = [raw_features[f] for f in feature_names]
        X = np.array(vector).reshape(1, -1)

        # Scale features using the preprocessor
        amount_idx = feature_names.index("Amount")
        X[:, amount_idx] = preprocessor.transform(X[:, amount_idx].reshape(-1, 1)).ravel()

        # Generate prediction
        predicted_label = int(model.predict(X)[0])
        # Obtain prediction confidence for class 1 (Fraud)
        confidence = float(model.predict_proba(X)[0, 1])
        
    except Exception as e:
        logger.exception("Error processing inference inputs.")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference processing failed: {str(e)}",
        )

    latency_ms = (time.perf_counter() - start_time) * 1000

    # Log to prediction DB
    try:
        version_str = str(version) if version is not None else "unknown"
        pred_id = pred_logger.log_prediction(
            features=raw_features,
            predicted_label=predicted_label,
            confidence=confidence,
            model_version=version_str,
            latency_ms=latency_ms,
        )
    except Exception:
        logger.exception("Failed to write prediction event log to database.")
        # We still return the prediction even if logging fails, to keep serving requests
        pred_id = -1

    return {
        "prediction_id": pred_id,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "model_version": str(version),
        "latency_ms": round(latency_ms, 3),
        "timestamp": datetime.now(timezone.utc),
    }


@router.post("/predict/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK)
def predict_batch(
    request: BatchPredictionRequest,
    db: DatabaseManager = Depends(get_db_manager),
    pred_logger: PredictionLogger = Depends(get_logger),
) -> dict[str, Any]:
    """Generate predictions for a batch of transactions."""
    model, preprocessor, version, _ = ModelProvider.get_active_model()
    
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Active model is not loaded in memory.",
        )

    feature_names = DataLoader.get_feature_names()
    amount_idx = feature_names.index("Amount")
    version_str = str(version) if version is not None else "unknown"

    records = []
    responses = []

    for idx, single_req in enumerate(request.predictions):
        start_time = time.perf_counter()
        raw_features = single_req.features
        
        # Verify columns
        vector = []
        for f in feature_names:
            if f not in raw_features:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Transaction index {idx} missing feature {f}",
                )
            vector.append(raw_features[f])
            
        X = np.array(vector).reshape(1, -1)
        X[:, amount_idx] = preprocessor.transform(X[:, amount_idx].reshape(-1, 1)).ravel()

        predicted_label = int(model.predict(X)[0])
        confidence = float(model.predict_proba(X)[0, 1])
        latency_ms = (time.perf_counter() - start_time) * 1000

        records.append({
            "features": raw_features,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "model_version": version_str,
            "latency_ms": latency_ms
        })

    # Log in bulk
    try:
        pred_ids = pred_logger.log_batch_predictions(records)
    except Exception:
        logger.exception("Failed batch log predictions to DB.")
        pred_ids = [-1] * len(request.predictions)

    for i, rec in enumerate(records):
        responses.append({
            "prediction_id": pred_ids[i],
            "predicted_label": rec["predicted_label"],
            "confidence": rec["confidence"],
            "model_version": version_str,
            "latency_ms": round(rec["latency_ms"], 3),
            "timestamp": datetime.now(timezone.utc),
        })

    return {"predictions": responses}


@router.post("/ground-truth/{prediction_id}", response_model=GroundTruthResponse)
def submit_ground_truth(
    prediction_id: int,
    request: GroundTruthRequest,
    pred_logger: PredictionLogger = Depends(get_logger),
) -> dict[str, Any]:
    """Submit ground-truth labels for past predictions to identify concept drift."""
    success = pred_logger.log_ground_truth(prediction_id, request.true_label)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction ID {prediction_id} not found in database.",
        )
    return {
        "success": True,
        "prediction_id": prediction_id,
        "true_label": request.true_label,
    }


@router.get("/predictions/stats", response_model=PredictionStatsResponse)
def get_stats(pred_logger: PredictionLogger = Depends(get_logger)) -> dict[str, Any]:
    """Get active throughput, average latency, and count statistics."""
    return pred_logger.get_prediction_stats()


@router.get("/predictions")
def list_predictions(
    limit: int = Query(100, ge=1, le=1000),
    model_version: str | None = Query(None),
    pred_logger: PredictionLogger = Depends(get_logger),
) -> list[dict[str, Any]]:
    """Retrieve historical prediction events, newest first."""
    predictions = pred_logger.get_predictions(model_version=model_version, limit=limit)
    
    results = []
    for p in predictions:
        import json
        results.append({
            "prediction_id": p.id,
            "timestamp": p.timestamp.isoformat(),
            "model_version": p.model_version,
            "features": json.loads(p.features_json) if p.features_json else {},
            "predicted_label": p.predicted_label,
            "confidence": p.confidence,
            "true_label": p.true_label,
            "latency_ms": p.latency_ms,
        })
    return results
