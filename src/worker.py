"""Persistent single-flight training/demo worker, separate from serving."""

from datetime import datetime, timedelta, timezone
import json
import logging
import signal
import threading
import time
import uuid
import numpy as np
import requests
from sqlalchemy import text
from src.alerting.alert_manager import AlertManager
from src.config.settings import get_settings
from src.data.database import get_database, RetrainingEvent
from src.jobs import JobQueue
from src.models.bundle import FEATURES, historical_data
from src.models.evaluator import ModelEvaluator
from src.models.registry import ModelRegistry
from src.models.trainer import ModelTrainer
from src.pipeline.retrain_pipeline import RetrainingPipeline
from src.process_lock import process_lock
from src.runtime import api_url, home, management_token

logger = logging.getLogger(__name__)


class RemoteDeployer:
    def deploy(self, version):
        response = requests.post(
            f"{api_url()}/internal/models/activate/{version}",
            headers={"X-MLOps-Token": management_token()},
            timeout=120,
        )
        return response.ok and response.json().get("version") == version


def generate_traffic(queue, job, headers):
    action = json.loads(job["reason"])
    data = historical_data()
    rng = np.random.default_rng(2026)
    count = action["samples"]
    # Sample training-side examples only; never expose historical holdout rows.
    indices = rng.choice(len(data["train_x"]), count, replace=True)
    X = data["train_x"][indices].copy()
    y = data["train_y"][indices]
    X += rng.normal(0, 0.001, X.shape)
    if action["scenario"] == "drift":
        X[:, :5] += 3.0 * np.std(data["train_x"][:, :5], axis=0)
    for start in range(0, count, 100):
        vectors = X[start : start + 100]
        response = requests.post(
            f"{api_url()}/demo/predict/batch",
            params={"scenario": action["scenario"]},
            headers=headers,
            json={
                "predictions": [{"features": dict(zip(FEATURES, row.tolist()))} for row in vectors]
            },
            timeout=60,
        )
        response.raise_for_status()
        predictions = response.json()["predictions"]
        feedback = [
            {"prediction_id": row["prediction_id"], "true_label": int(label)}
            for row, label in zip(predictions, y[start : start + 100])
        ]
        response = requests.post(
            f"{api_url()}/ground-truth/batch", json={"labels": feedback}, timeout=30
        )
        response.raise_for_status()
        queue.update(
            job["id"],
            "RUNNING",
            {
                "message": "Real inference and label submission",
                "processed": min(start + 100, count),
                "total": count,
            },
        )
    return {
        "message": f"Generated {count} {action['scenario']} predictions through the API with source labels.",
        "processed": count,
        "scenario": action["scenario"],
        "source": data["source"],
    }


def run_worker():
    settings = get_settings()
    db = get_database(settings.database.url)
    db.init_db()
    queue = JobQueue(db)
    stop = threading.Event()
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, lambda *_: stop.set())
    with process_lock(home() / "worker.lock"):
        headers = {"X-MLOps-Token": management_token()}
        # Reconcile activation that succeeded just before a worker interruption.
        reconciled = []
        for job in queue.list():
            if job["status"] == "DEPLOYING" and job["result"].get("version"):
                try:
                    current = requests.get(f"{api_url()}/ready", timeout=5).json()
                    if current.get("model_version") == job["result"]["version"]:
                        reconciled.append(job)
                except requests.RequestException:
                    pass
        queue.recover()
        for job in reconciled:
            queue.update(
                job["id"],
                "COMPLETED",
                {**job["result"], "message": "Reconciled confirmed activation after restart"},
            )
            with db.get_session() as session:
                event_id = job["result"].get("event_id")
                event = session.get(RetrainingEvent, event_id) if event_id else None
                if event:
                    event.status = "COMPLETED"
                    event.new_version = job["result"]["version"]

        owner = uuid.uuid4().hex
        current_job = {"id": None}

        def heartbeat():
            while not stop.is_set():
                queue.state(
                    "worker",
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "job_id": current_job["id"],
                    },
                )
                stop.wait(3)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        registry = ModelRegistry(settings)
        pipeline = RetrainingPipeline(
            settings,
            db,
            ModelTrainer(settings),
            ModelEvaluator(),
            registry,
            RemoteDeployer(),
            AlertManager(db, settings.alerting),
        )
        last_check = time.monotonic()
        try:
            while not stop.is_set():
                job = queue.claim(owner)
                if job:
                    current_job["id"] = job["id"]
                    try:
                        if job["kind"] == "traffic":
                            result = generate_traffic(queue, job, headers)
                            queue.update(job["id"], "COMPLETED", result)
                        else:
                            pipeline.last_result = {}
                            deployed = pipeline.execute(
                                job["reason"],
                                lambda status, value: queue.update(job["id"], status, value),
                            )
                            status = (
                                "COMPLETED"
                                if deployed
                                else (
                                    "FAILED" if pipeline.last_result.get("failed") else "REJECTED"
                                )
                            )
                            queue.update(job["id"], status, pipeline.last_result)
                    except Exception as exc:
                        logger.exception("Job failed")
                        queue.update(job["id"], "FAILED", {"message": str(exc)})
                    finally:
                        current_job["id"] = None
                if time.monotonic() - last_check >= settings.monitoring.check_interval_minutes * 60:
                    try:
                        response = requests.post(
                            f"{api_url()}/drift/run", headers=headers, timeout=45
                        )
                        if response.status_code not in [200, 400]:
                            logger.warning("Scheduled monitoring returned %s", response.status_code)
                    except requests.RequestException:
                        logger.exception("Scheduled monitoring API unavailable")
                    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
                    with db.engine.begin() as connection:
                        connection.execute(
                            text("DELETE FROM predictions WHERE timestamp < :cutoff"),
                            {"cutoff": cutoff},
                        )
                        connection.execute(
                            text(
                                "DELETE FROM predictions WHERE id IN (SELECT id FROM predictions ORDER BY id DESC LIMIT -1 OFFSET 50000)"
                            )
                        )
                    last_check = time.monotonic()
                stop.wait(0.5)
        finally:
            stop.set()
            thread.join(timeout=5)
            queue.state("worker", {"timestamp": "2000-01-01T00:00:00+00:00", "job_id": None})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_worker()
