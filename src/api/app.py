"""FastAPI application factory.

Initialises the database, MLflow model registry, in-memory serving cache,
drift detection detectors, alerting channels, and periodic scheduler.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler

from src.config.settings import get_settings
from src.data.database import get_database, ModelVersion
from src.data.logger import PredictionLogger
from src.data.loader import DataLoader
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.models.registry import ModelRegistry
from src.monitoring.data_drift import DataDriftDetector
from src.monitoring.prediction_drift import PredictionDriftDetector
from src.monitoring.concept_drift import ConceptDriftDetector
from src.monitoring.drift_manager import DriftManager
from src.alerting.alert_manager import AlertManager
from src.decision.retraining_engine import RetrainingDecisionEngine
from src.pipeline.deployer import ModelDeployer, ModelProvider
from src.pipeline.retrain_pipeline import RetrainingPipeline

from src.api.middleware import TimingMiddleware, RequestLoggingMiddleware, global_exception_handler
from src.api.routes import health, predictions, monitoring, models

logger = logging.getLogger(__name__)


def schedule_periodic_drift_check(app: FastAPI) -> None:
    """Run periodic data, prediction, and concept drift check via scheduler."""
    settings = app.state.settings
    db = app.state.db
    drift_mgr = app.state.drift_manager
    retraining_engine = app.state.retraining_engine
    retrain_pipeline = app.state.retrain_pipeline

    logger.info("Executing scheduled periodic drift check...")
    
    try:
        pred_logger = PredictionLogger(db)
        recent_predictions = pred_logger.get_recent_predictions(settings.monitoring.window_size)
        
        if len(recent_predictions) < settings.monitoring.min_samples:
            logger.info(
                "Scheduled drift check skipped: insufficient prediction logs (%d/%d).",
                len(recent_predictions),
                settings.monitoring.min_samples,
            )
            return

        feature_names = DataLoader.get_feature_names()
        summary = drift_mgr.run_drift_check(recent_predictions, feature_names)

        # Evaluate retraining decision rules
        decision = retraining_engine.evaluate(summary)
        retraining_engine.record_decision(decision)
        
        if decision.should_retrain:
            logger.warning(
                "Scheduled drift check triggered retraining! Reason: %s",
                decision.reason,
            )
            # Run retraining
            retrain_pipeline.execute(decision.reason)
            
    except Exception:
        logger.exception("Scheduled periodic drift check failed.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager handling FastAPI startup and shutdown routines."""
    logger.info("Starting up MLOps Serving API...")
    
    # 1. Load configuration
    settings = get_settings()
    app.state.settings = settings

    # 2. Setup Database & Logger
    db = get_database(settings.database.url)
    db.init_db()
    app.state.db = db

    # 3. Setup MLflow registry
    registry = ModelRegistry(settings)
    app.state.registry = registry

    # 4. Try loading the production model into serving cache (ModelProvider)
    prod_version_row = None
    try:
        with db.get_session() as session:
            prod_version_row = (
                session.query(ModelVersion)
                .filter(ModelVersion.is_production == True)
                .one_or_none()
            )
        
        if prod_version_row:
            logger.info("Loading production model version %d from registry...", prod_version_row.version)
            prod_model = registry.load_model(version=prod_version_row.version)
            loader = DataLoader()
            preprocessor = loader.load_preprocessor()
            ModelProvider.set_active_model(
                model=prod_model,
                preprocessor=preprocessor,
                version=prod_version_row.version,
                mlflow_run_id=prod_version_row.mlflow_run_id
            )
        else:
            logger.warning("No production model marked in DB. Server starting without pre-loaded model.")
    except Exception:
        logger.exception("Error loading active production model at startup. API will be degraded.")

    # 5. Initialize alerting & decision engine
    alert_mgr = AlertManager(db)
    app.state.alert_manager = alert_mgr
    
    retraining_engine = RetrainingDecisionEngine(settings.drift_thresholds, db)
    app.state.retraining_engine = retraining_engine

    # 6. Initialize drift detectors & manager
    try:
        loader = DataLoader()
        ref_data = loader.load_reference_distribution()
        
        # Load baseline model details for Hellinger and metric comparisons
        # If reference distribution and preprocessor exist, we initialize detectors
        data_detector = DataDriftDetector(ref_data, settings.drift_thresholds.data_drift)
        
        # Set up prediction and concept detectors from real production state
        # when available. The deterministic fallback keeps cold starts usable.
        pred_logger = PredictionLogger(db)
        baseline_window = pred_logger.get_recent_predictions(settings.monitoring.window_size)
        if len(baseline_window) >= settings.monitoring.min_samples:
            baseline_preds = np.array([p.predicted_label for p in baseline_window], dtype=np.int64)
            baseline_confs = np.array([p.confidence for p in baseline_window], dtype=np.float64)
        else:
            baseline_preds = np.zeros(settings.monitoring.window_size, dtype=np.int64)
            baseline_confs = np.ones(settings.monitoring.window_size, dtype=np.float64) * 0.95

        prediction_detector = PredictionDriftDetector(
            baseline_predictions=baseline_preds,
            baseline_confidences=baseline_confs,
            thresholds=settings.drift_thresholds.prediction_drift
        )

        if prod_version_row:
            baseline_metrics = {
                "accuracy": float(prod_version_row.accuracy),
                "f1": float(prod_version_row.f1_score),
                "precision": float(prod_version_row.precision),
                "recall": float(prod_version_row.recall),
            }
        else:
            baseline_metrics = {"accuracy": 0.98, "f1": 0.95, "precision": 0.96, "recall": 0.94}
        concept_detector = ConceptDriftDetector(
            baseline_metrics=baseline_metrics,
            thresholds=settings.drift_thresholds.concept_drift
        )

        drift_mgr = DriftManager(
            db=db,
            data_detector=data_detector,
            prediction_detector=prediction_detector,
            concept_detector=concept_detector
        )
        app.state.drift_manager = drift_mgr
        logger.info("DriftManager and detectors successfully initialized.")
    except Exception:
        logger.exception("Failed to initialize drift detectors at startup (probably missing reference data).")
        # Initialize a degraded DriftManager without detectors to avoid crashing server startup
        app.state.drift_manager = DriftManager(db=db)

    # 7. Initialize training & retraining pipeline components
    trainer = ModelTrainer(settings)
    evaluator = ModelEvaluator()
    deployer = ModelDeployer(settings, db, registry)
    app.state.deployer = deployer
    
    retrain_pipeline = RetrainingPipeline(
        settings=settings,
        db=db,
        trainer=trainer,
        evaluator=evaluator,
        registry=registry,
        deployer=deployer,
        alert_manager=alert_mgr,
    )
    app.state.retrain_pipeline = retrain_pipeline

    # 8. Start APScheduler background monitor checks
    scheduler = BackgroundScheduler()
    app.state.scheduler = scheduler
    
    # Run a drift check every check_interval_minutes
    interval_min = settings.monitoring.check_interval_minutes
    scheduler.add_job(
        schedule_periodic_drift_check,
        "interval",
        minutes=interval_min,
        args=[app],
        id="scheduled_drift_check",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("APScheduler started: periodic drift checks run every %d minutes.", interval_min)

    yield

    # Shutdown routines
    logger.info("Shutting down MLOps Serving API...")
    if hasattr(app.state, "scheduler"):
        app.state.scheduler.shutdown(wait=False)
        logger.info("APScheduler stopped.")


def create_app() -> FastAPI:
    """App factory creating and configuring FastAPI instance."""
    import numpy as np # Ensure numpy is available in namespace
    
    app = FastAPI(
        title="MLOps Prediction & Serving API",
        description="Production serving layer for credit card fraud detection with real-time drift monitoring.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Logging and Timing Middlewares
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(TimingMiddleware)

    # Exception Handler
    app.add_exception_handler(Exception, global_exception_handler)

    # Include Routers
    app.include_router(health.router)
    app.include_router(predictions.router)
    app.include_router(monitoring.router)
    app.include_router(models.router)

    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:create_app", host="0.0.0.0", port=8000, reload=True)
