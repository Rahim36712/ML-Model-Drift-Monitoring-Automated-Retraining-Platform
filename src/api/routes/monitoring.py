"""Monitoring routes for drift analysis, health checks, and system alerts."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query, Request, BackgroundTasks

from src.data.database import DatabaseManager, DriftResult
from src.data.logger import PredictionLogger
from src.monitoring.drift_manager import DriftManager, DriftSummary
from src.alerting.alert_manager import AlertManager
from src.api.schemas import DriftSummaryResponse, AlertResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Monitoring"])


def get_db_manager() -> DatabaseManager:
    from src.data.database import get_database
    return get_database()


def get_alert_manager(db: DatabaseManager = Depends(get_db_manager)) -> AlertManager:
    return AlertManager(db)


def get_drift_manager(request: Request) -> DriftManager:
    """Retrieve DriftManager instance stored in FastAPI app state."""
    drift_mgr = getattr(request.app.state, "drift_manager", None)
    if drift_mgr is None:
        raise HTTPException(
            status_code=503,
            detail="Drift detection service is not initialized on the server.",
        )
    return drift_mgr


@router.post("/drift/run", response_model=DriftSummaryResponse)
def trigger_drift_check(
    request: Request,
    background_tasks: BackgroundTasks,
    db: DatabaseManager = Depends(get_db_manager),
    drift_mgr: DriftManager = Depends(get_drift_manager),
) -> dict[str, Any]:
    """Manually run a drift-detection check over recent predictions."""
    settings = request.app.state.settings
    window_size = settings.monitoring.window_size
    min_samples = settings.monitoring.window_size # or settings.monitoring.min_samples

    pred_logger = PredictionLogger(db)
    recent_predictions = pred_logger.get_recent_predictions(window_size)

    if len(recent_predictions) < settings.monitoring.min_samples:
        logger.warning(
            "Drift check skipped: not enough prediction samples. required=%d, current=%d",
            settings.monitoring.min_samples,
            len(recent_predictions),
        )
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient prediction logs to compute drift. Need at least {settings.monitoring.min_samples} samples (current: {len(recent_predictions)}).",
        )

    from src.data.loader import DataLoader
    feature_names = DataLoader.get_feature_names()

    # Run check
    summary: DriftSummary = drift_mgr.run_drift_check(recent_predictions, feature_names)

    # If any drift is breached, trigger the retraining decision engine
    decision_engine = getattr(request.app.state, "retraining_engine", None)
    retrain_pipeline = getattr(request.app.state, "retrain_pipeline", None)

    if decision_engine is not None and retrain_pipeline is not None:
        decision = decision_engine.evaluate(summary)
        decision_engine.record_decision(decision)
        if decision.should_retrain:
            logger.info("Retraining triggered manually via drift check. Reason: %s", decision.reason)
            # Run retraining asynchronously in background to prevent blocking HTTP response
            background_tasks.add_task(retrain_pipeline.execute, decision.reason)

    return _format_drift_summary_response(summary)


@router.get("/drift/latest", response_model=DriftSummaryResponse)
def get_latest_drift(
    request: Request,
    drift_mgr: DriftManager = Depends(get_drift_manager),
) -> dict[str, Any]:
    """Retrieve the latest computed drift check summary."""
    summary = drift_mgr.get_latest_summary()
    if summary is None:
        # If no summary in memory, try to construct one from DB or run a quick check if possible.
        raise HTTPException(
            status_code=404,
            detail="No drift checks have been executed yet.",
        )
    return _format_drift_summary_response(summary)


@router.get("/drift/history")
def get_drift_history(
    drift_type: str | None = Query(None, description="Filter by drift type: data, prediction, concept"),
    limit: int = Query(50, ge=1, le=200),
    drift_mgr: DriftManager = Depends(get_drift_manager),
) -> list[dict[str, Any]]:
    """Retrieve historical drift results from the database."""
    history: list[DriftResult] = drift_mgr.get_drift_history(drift_type=drift_type, limit=limit)
    
    results = []
    for row in history:
        import json
        results.append({
            "id": row.id,
            "timestamp": row.timestamp.isoformat(),
            "drift_type": row.drift_type,
            "metric_name": row.metric_name,
            "metric_value": row.metric_value,
            "threshold": row.threshold,
            "is_breached": row.is_breached,
            "details": json.loads(row.details_json) if row.details_json else {},
        })
    return results


@router.get("/model-health")
def get_model_health(
    request: Request,
    drift_mgr: DriftManager = Depends(get_drift_manager),
) -> dict[str, Any]:
    """Get high-level status of the model health and drift indices."""
    summary = drift_mgr.get_latest_summary()
    
    status_val = "healthy"
    details = {}
    
    if summary:
        status_val = summary.overall_status
        details = {
            "data_drift_status": summary.data_drift.severity if summary.data_drift else "none",
            "prediction_drift_status": summary.prediction_drift.severity if summary.prediction_drift else "none",
            "concept_drift_status": summary.concept_drift.severity if summary.concept_drift else "none",
            "last_checked": summary.timestamp.isoformat(),
        }
    else:
        status_val = "unknown"
        details = {"message": "No drift checks have run yet."}

    return {
        "status": status_val,
        "timestamp": datetime.now(timezone.utc),
        "details": details,
    }


@router.get("/alerts", response_model=list[AlertResponse])
def get_alerts(
    active_only: bool = Query(False, description="Filter to unacknowledged alerts only"),
    limit: int = Query(50, ge=1, le=100),
    alert_mgr: AlertManager = Depends(get_alert_manager),
) -> list[Any]:
    """Fetch active or historical alerts."""
    if active_only:
        return alert_mgr.get_active_alerts()[:limit]
    return alert_mgr.get_alert_history(limit=limit)


@router.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(
    alert_id: int,
    alert_mgr: AlertManager = Depends(get_alert_manager),
) -> dict[str, bool]:
    """Acknowledge an active alert."""
    success = alert_mgr.acknowledge_alert(alert_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Alert ID {alert_id} not found or already acknowledged.",
        )
    return {"success": True}


def _format_drift_summary_response(summary: DriftSummary) -> dict[str, Any]:
    """Helper to structure DriftSummary into API schema dict."""
    import json
    
    data_drift_breached = summary.data_drift.is_drifted if summary.data_drift else False
    pred_drift_breached = summary.prediction_drift.is_drifted if summary.prediction_drift else False
    concept_drift_breached = summary.concept_drift.is_drifted if summary.concept_drift else False

    # Extract all details
    details = {}
    if summary.data_drift:
        details["data_drift"] = {
            "overall_psi": summary.data_drift.overall_psi,
            "feature_psi": summary.data_drift.feature_psi,
            "feature_kl": summary.data_drift.feature_kl,
            "severity": summary.data_drift.severity,
            "drifted_features": summary.data_drift.drifted_features,
        }
    if summary.prediction_drift:
        details["prediction_drift"] = {
            "hellinger_distance": summary.prediction_drift.hellinger_distance,
            "baseline_positive_rate": summary.prediction_drift.baseline_positive_rate,
            "current_positive_rate": summary.prediction_drift.current_positive_rate,
            "baseline_mean_confidence": summary.prediction_drift.baseline_mean_confidence,
            "current_mean_confidence": summary.prediction_drift.current_mean_confidence,
            "severity": summary.prediction_drift.severity,
        }
    if summary.concept_drift:
        details["concept_drift"] = {
            "current_metrics": summary.concept_drift.current_metrics,
            "baseline_metrics": summary.concept_drift.baseline_metrics,
            "metric_deltas": summary.concept_drift.metric_deltas,
            "degraded_metrics": summary.concept_drift.degraded_metrics,
            "severity": summary.concept_drift.severity,
        }

    return {
        "overall_status": summary.overall_status,
        "timestamp": summary.timestamp,
        "checks_performed": summary.checks_performed,
        "data_drift_breached": data_drift_breached,
        "prediction_drift_breached": pred_drift_breached,
        "concept_drift_breached": concept_drift_breached,
        "details": details,
    }
