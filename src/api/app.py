"""Local serving API. One serving process owns activation and rollback."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import secrets
import time
from typing import Literal
import numpy as np
from fastapi import FastAPI, Depends, Header, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from src.api.schemas import PredictionRequest, BatchPredictionRequest, GroundTruthRequest
from src.config.settings import get_settings
from src.data.database import get_database, Prediction, ModelVersion, Alert
from src.data.logger import PredictionLogger
from src.jobs import JobQueue
from src.models.bundle import FEATURES
from src.models.registry import ModelRegistry
from src.pipeline.deployer import ModelProvider, ModelDeployer
from src.runtime import management_token
from src.service import monitor, dashboard_snapshot, serial

logger = logging.getLogger(__name__)


def require_management(x_mlops_token: str = Header(default="")):
    if not secrets.compare_digest(x_mlops_token, management_token()):
        raise HTTPException(401, "Management authentication required")


class DemoAction(BaseModel):
    scenario: Literal["stable", "drift"] = "stable"
    samples: int = Field(default=500, ge=100, le=2000)


class Feedback(BaseModel):
    prediction_id: int
    true_label: Literal[0, 1]


class FeedbackBatch(BaseModel):
    labels: list[Feedback] = Field(min_length=1, max_length=1000)


class ResetRequest(BaseModel):
    confirmation: Literal["RESET DEMO"]


def create_app():
    settings = get_settings()
    db = get_database(settings.database.url)
    db.init_db()
    queue = JobQueue(db)

    @asynccontextmanager
    async def lifespan(app):
        from src.process_lock import process_lock
        from src.runtime import home

        with process_lock(home() / "api.lock"):
            ModelProvider.set_active_model(None, None, None)
            registry = ModelRegistry(settings)
            app.state.deployer = ModelDeployer(settings, db, registry)
            with db.get_session() as session:
                row = session.query(ModelVersion).filter_by(is_production=True).one_or_none()
                version = row.version if row else None
            if version is not None and not app.state.deployer.deploy(version):
                logger.error("Persisted champion unavailable; readiness remains false")
            yield

    app = FastAPI(title="DriftWatch · Fraud Monitoring", version="2.0.0", lifespan=lifespan)
    app.state.db, app.state.settings, app.state.queue = db, settings, queue
    app.state.snapshot_cache = {}

    @app.middleware("http")
    async def latency(request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time"] = f"{elapsed:.3f}ms"
        ids = getattr(request.state, "prediction_ids", [])
        if ids:
            with db.get_session() as session:
                session.query(Prediction).filter(Prediction.id.in_(ids)).update(
                    {"request_latency_ms": elapsed}, synchronize_session=False
                )
        return response

    @app.exception_handler(Exception)
    async def unexpected(request, exc):
        logger.exception("Unhandled API error", exc_info=exc)
        return JSONResponse(
            status_code=500, content={"detail": "An internal error occurred; inspect service logs."}
        )

    @app.get("/health")
    def health():
        with db.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {
            "status": "ok",
            "db_connected": True,
            "model_version": ModelProvider.get_snapshot().version,
            "timestamp": datetime.now(timezone.utc),
        }

    @app.get("/ready")
    def ready():
        connected = True
        try:
            with db.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except Exception:
            connected = False
        value = {
            "ready": ModelProvider.is_loaded() and connected,
            "db_connected": connected,
            "model_version": ModelProvider.get_snapshot().version,
        }
        return JSONResponse(status_code=200 if value["ready"] else 503, content=value)

    def infer(items, request, source="api", scenario="live"):
        snapshot = ModelProvider.get_snapshot()
        if snapshot.model is None:
            raise HTTPException(503, "No model is loaded. Train the baseline first.")
        start = time.perf_counter()
        X = np.array([[item.features[name] for name in FEATURES] for item in items], dtype=float)
        if snapshot.preprocessor is not None:
            X[:, 28] = snapshot.preprocessor.transform(X[:, 28].reshape(-1, 1)).ravel()
        probabilities = snapshot.model.predict_proba(X)[:, 1]
        labels = snapshot.model.predict(X)
        elapsed = (time.perf_counter() - start) * 1000 / len(items)
        stamp = datetime.now(timezone.utc)
        with db.get_session() as session:
            rows = [
                Prediction(
                    features_json=json.dumps(item.features),
                    predicted_label=int(label),
                    confidence=float(probability),
                    model_version=str(snapshot.version),
                    latency_ms=elapsed,
                    timestamp=stamp,
                    source=source,
                    scenario=scenario,
                )
                for item, label, probability in zip(items, labels, probabilities)
            ]
            session.add_all(rows)
            session.flush()
            ids = [row.id for row in rows]
        request.state.prediction_ids = ids
        return [
            {
                "prediction_id": row_id,
                "predicted_label": int(label),
                "confidence": float(probability),
                "model_version": str(snapshot.version),
                "latency_ms": elapsed,
                "timestamp": stamp,
            }
            for row_id, label, probability in zip(ids, labels, probabilities)
        ]

    @app.post("/predict")
    def predict(payload: PredictionRequest, request: Request):
        return infer([payload], request)[0]

    @app.post("/predict/batch")
    def batch(payload: BatchPredictionRequest, request: Request):
        return {"predictions": infer(payload.predictions, request)}

    @app.post("/demo/predict/batch", dependencies=[Depends(require_management)])
    def demo_batch(
        payload: BatchPredictionRequest,
        request: Request,
        scenario: Literal["stable", "drift"] = "stable",
    ):
        return {"predictions": infer(payload.predictions, request, "demo", scenario)}

    @app.post("/ground-truth/batch")
    def feedback_batch(payload: FeedbackBatch):
        with db.get_session() as session:
            ids = [item.prediction_id for item in payload.labels]
            rows = {
                row.id: row
                for row in session.query(Prediction).filter(Prediction.id.in_(ids)).all()
            }
            if len(rows) != len(set(ids)):
                raise HTTPException(404, "One or more predictions were not found")
            for item in payload.labels:
                rows[item.prediction_id].true_label = item.true_label
        return {"success": True, "count": len(ids)}

    @app.post("/ground-truth/{prediction_id}")
    def ground_truth(prediction_id: int, payload: GroundTruthRequest):
        if not PredictionLogger(db).log_ground_truth(prediction_id, payload.true_label):
            raise HTTPException(404, "Prediction not found")
        return {"success": True, "prediction_id": prediction_id, "true_label": payload.true_label}

    @app.get("/predictions/stats")
    def stats():
        return PredictionLogger(db).get_prediction_stats()

    @app.get("/predictions")
    def predictions(limit: int = Query(100, ge=1, le=1000), model_version: str | None = None):
        return [
            serial(row)
            for row in PredictionLogger(db).get_predictions(
                model_version=model_version, limit=limit
            )
        ]

    @app.get("/models/current")
    def current():
        with db.get_session() as session:
            row = session.query(ModelVersion).filter_by(is_production=True).one_or_none()
            if row is None:
                raise HTTPException(404, "No champion registered")
            return serial(row)

    @app.get("/models/versions")
    def versions():
        with db.get_session() as session:
            return [
                serial(row)
                for row in session.query(ModelVersion).order_by(ModelVersion.version.desc()).all()
            ]

    @app.post("/models/retrain", status_code=202, dependencies=[Depends(require_management)])
    def retrain():
        job, created = queue.enqueue("Manual showcase retraining")
        return {"job_id": job["id"], "created": created, "status": job["status"]}

    @app.get("/jobs", dependencies=[Depends(require_management)])
    def jobs():
        return queue.list()

    @app.get("/jobs/{job_id}", dependencies=[Depends(require_management)])
    def job(job_id: str):
        row = queue.get(job_id)
        if not row:
            raise HTTPException(404, "Job not found")
        return row

    def activate(version, request):
        with db.get_session() as session:
            if session.query(ModelVersion).filter_by(version=version).one_or_none() is None:
                raise HTTPException(404, "Model version not found")
        if not hasattr(request.app.state, "deployer"):
            raise HTTPException(503, "Deployer not initialized")
        if not request.app.state.deployer.deploy(version):
            raise HTTPException(409, "Activation failed; previous serving snapshot retained")
        queue.state(
            "drift",
            {
                "overall_status": "unavailable",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_version": version,
                "window": None,
                "message": "New version: generate traffic and run a drift check",
            },
        )
        return {"success": True, "version": version}

    @app.post("/internal/models/activate/{version}", dependencies=[Depends(require_management)])
    def internal_activate(version: int, request: Request):
        return activate(version, request)

    @app.post("/models/rollback/{version}", dependencies=[Depends(require_management)])
    def rollback(version: int, request: Request):
        if any(
            row["status"] in ["RUNNING", "EVALUATING", "DEPLOYING", "QUEUED"]
            for row in queue.list()
        ):
            raise HTTPException(409, "Wait for the active job before rollback")
        return activate(version, request)

    @app.post("/drift/run", dependencies=[Depends(require_management)])
    def run_drift():
        try:
            return monitor(db, settings)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/drift/latest")
    @app.get("/model-health")
    def latest():
        return queue.state("drift") or {
            "overall_status": "unavailable",
            "message": "No drift checks yet",
        }

    @app.get("/drift/history")
    def history(limit: int = Query(50, ge=1, le=300)):
        from src.data.database import DriftResult

        with db.get_session() as session:
            return [
                serial(row)
                for row in session.query(DriftResult)
                .order_by(DriftResult.timestamp.desc())
                .limit(limit)
                .all()
            ]

    @app.get("/alerts")
    def alerts():
        with db.get_session() as session:
            return [
                serial(row)
                for row in session.query(Alert).order_by(Alert.timestamp.desc()).limit(200).all()
            ]

    @app.post("/alerts/{alert_id}/acknowledge", dependencies=[Depends(require_management)])
    def acknowledge(alert_id: int):
        with db.get_session() as session:
            row = session.get(Alert, alert_id)
            if row is None:
                raise HTTPException(404, "Alert not found")
            row.acknowledged = True
        return {"success": True}

    @app.get("/dashboard/snapshot", dependencies=[Depends(require_management)])
    def overview(model_version: str | None = None, hours: int = Query(24, ge=1, le=720)):
        key = (model_version, hours)
        cached = app.state.snapshot_cache.get(key)
        if cached and time.monotonic() - cached[0] < 1:
            return cached[1]
        value = jsonable_encoder(dashboard_snapshot(db, settings, model_version, hours))
        if len(app.state.snapshot_cache) > 30:
            app.state.snapshot_cache.clear()
        app.state.snapshot_cache[key] = (time.monotonic(), value)
        return value

    @app.post("/demo/traffic", status_code=202, dependencies=[Depends(require_management)])
    def demo(payload: DemoAction):
        if not ModelProvider.is_loaded():
            raise HTTPException(409, "Train a baseline before generating traffic")
        job, created = queue.enqueue(json.dumps(payload.model_dump()), "traffic")
        return {"job_id": job["id"], "created": created}

    @app.post("/demo/reset", dependencies=[Depends(require_management)])
    def reset(payload: ResetRequest):
        with db.engine.connect() as connection:
            connection.execute(text("BEGIN IMMEDIATE"))
            if connection.execute(
                text(
                    "SELECT COUNT(*) FROM jobs WHERE status IN ('QUEUED','RUNNING','EVALUATING','DEPLOYING')"
                )
            ).scalar():
                raise HTTPException(409, "Wait for active jobs before resetting demo telemetry")
            connection.execute(text("DELETE FROM predictions WHERE source='demo'"))
            connection.execute(text("DELETE FROM runtime_state WHERE key='drift'"))
            connection.commit()
        return {
            "success": True,
            "message": "Generated demo predictions cleared. Models and audit history preserved.",
        }

    return app
