"""Model management endpoints including promotion, version listing, retraining, and rollbacks."""

from __future__ import annotations

import logging
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request, status

from src.data.database import DatabaseManager, ModelVersion
from src.pipeline.deployer import ModelDeployer, ModelProvider
from src.pipeline.retrain_pipeline import RetrainingPipeline
from src.api.schemas import ModelVersionResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Models"])


def get_db_manager() -> DatabaseManager:
    from src.data.database import get_database
    return get_database()


@router.get("/models/current", response_model=ModelVersionResponse)
def get_current_model(db: DatabaseManager = Depends(get_db_manager)) -> dict[str, Any]:
    """Retrieve details of the currently active production model."""
    with db.get_session() as session:
        row = (
            session.query(ModelVersion)
            .filter(ModelVersion.is_production == True)
            .one_or_none()
        )
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No active production model found in the database. Ensure a baseline model is trained.",
            )
        
        return {
            "version": row.version,
            "mlflow_run_id": row.mlflow_run_id,
            "accuracy": row.accuracy,
            "f1_score": row.f1_score,
            "precision": row.precision,
            "recall": row.recall,
            "auc_roc": row.auc_roc,
            "training_date": row.training_date,
            "is_production": row.is_production,
            "deployed_at": row.deployed_at,
        }


@router.get("/models/versions", response_model=list[ModelVersionResponse])
def get_all_versions(db: DatabaseManager = Depends(get_db_manager)) -> list[Any]:
    """List all registered model versions mirrored in the local database."""
    with db.get_session() as session:
        rows = (
            session.query(ModelVersion)
            .order_by(ModelVersion.version.desc())
            .all()
        )
        return rows


@router.post("/models/retrain", status_code=status.HTTP_202_ACCEPTED)
def trigger_manual_retrain(
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """Force an asynchronous model retraining job."""
    retrain_pipeline = getattr(request.app.state, "retrain_pipeline", None)
    if retrain_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retraining service is not initialized on the server.",
        )
        
    logger.info("Manual retraining triggered via API endpoint.")
    background_tasks.add_task(retrain_pipeline.execute, "manual_api_trigger")
    return {"message": "Model retraining job submitted successfully."}


@router.post("/models/rollback/{version}", response_model=dict[str, Any])
def rollback_model(
    version: int,
    request: Request,
    db: DatabaseManager = Depends(get_db_manager),
) -> dict[str, Any]:
    """Roll back the serving model to a previously registered version."""
    deployer = getattr(request.app.state, "deployer", None)
    if deployer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model deployer service is not initialized on the server.",
        )
        
    # Verify version exists in DB first
    with db.get_session() as session:
        exists = session.query(ModelVersion).filter(ModelVersion.version == version).count() > 0
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model version {version} not found in registered history.",
            )

    logger.info("Manual rollback to version %d requested.", version)
    success = deployer.rollback(version)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rollback to version {version} failed. Check server logs.",
        )
        
    return {
        "success": True,
        "message": f"Successfully rolled back to version {version}.",
        "version": version,
    }
