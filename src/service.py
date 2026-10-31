"""Monitoring and shared dashboard reads, independent of presentation code."""

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
import hashlib
import numpy as np
from src.alerting.alert_manager import AlertManager
from src.data.database import Prediction, DriftResult, Alert, ModelVersion, RetrainingEvent
from src.data.logger import PredictionLogger
from src.decision.retraining_engine import RetrainingDecisionEngine
from src.jobs import JobQueue
from src.models.bundle import FEATURES
from src.monitoring.data_drift import DataDriftDetector
from src.monitoring.prediction_drift import PredictionDriftDetector
from src.monitoring.concept_drift import ConceptDriftDetector
from src.monitoring.drift_manager import DriftManager
from src.pipeline.deployer import ModelProvider


def serial(row):
    return {column.name: getattr(row, column.name) for column in row.__table__.columns}


def monitor(db, settings):
    snapshot = ModelProvider.get_snapshot()
    if snapshot.bundle is None:
        raise ValueError("Train a baseline before checking drift.")
    bundle = snapshot.bundle
    records = PredictionLogger(db).get_predictions(
        model_version=str(bundle.version), limit=settings.monitoring.window_size
    )
    if len(records) < settings.monitoring.min_samples:
        raise ValueError(
            f"Generate at least {settings.monitoring.min_samples} predictions for this model first."
        )
    queue = JobQueue(db)
    labels_hash = hashlib.sha256(
        json.dumps([(row.id, row.true_label) for row in records]).encode()
    ).hexdigest()[:16]
    window = f"{bundle.version}:{max(row.id for row in records)}:{labels_hash}"
    previous = queue.state("drift")
    if previous and previous.get("window") == window:
        return previous
    labeled = [row for row in records if row.true_label is not None]
    labels = np.array([row.true_label for row in labeled])
    label_ready = len(labels) >= settings.monitoring.min_labeled_samples and all(
        np.sum(labels == value) >= settings.monitoring.min_class_samples for value in [0, 1]
    )
    manager = DriftManager(
        db,
        DataDriftDetector(bundle.reference, settings.drift_thresholds.data_drift),
        PredictionDriftDetector(
            bundle.baseline_predictions,
            bundle.baseline_probabilities,
            settings.drift_thresholds.prediction_drift,
        ),
        ConceptDriftDetector(bundle.baseline_metrics, settings.drift_thresholds.concept_drift)
        if label_ready
        else None,
    )
    manager._model_version = str(bundle.version)
    summary = manager.run_drift_check(records, FEATURES)
    expected = {"data_drift", "prediction_drift"} | ({"concept_drift"} if label_ready else set())
    failed = sorted(expected - set(summary.checks_performed))
    if failed:
        summary.overall_status = "unavailable"
    result = json.loads(json.dumps(asdict(summary), default=str))
    result.update(
        window=window,
        model_version=bundle.version,
        sample_count=len(records),
        labeled_count=len(labeled),
        failed_checks=failed,
        concept_note="Measured" if label_ready else "Insufficient labeled samples / class coverage",
        window_start=str(min(row.timestamp for row in records)),
        window_end=str(max(row.timestamp for row in records)),
    )
    alerts = AlertManager(db, settings.alerting)
    if summary.overall_status in ["warning", "critical"]:
        alerts.send_alert(
            summary.overall_status.upper(),
            "monitoring",
            "drift_window",
            1,
            0,
            f"{summary.overall_status.title()} drift for model v{bundle.version}; inspect Drift Analysis.",
        )
    decision = RetrainingDecisionEngine(settings.drift_thresholds, db).evaluate(summary)
    if (
        not failed
        and decision.should_retrain
        and not queue.cooldown(settings.monitoring.cooldown_minutes)
    ):
        job, created = queue.enqueue(decision.reason)
        result["job_id"] = job["id"]
        result["job_created"] = created
    result["decision"] = asdict(decision)
    queue.state("drift", result)
    return result


def dashboard_snapshot(db, settings, version=None, hours=24):
    queue = JobQueue(db)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    with db.get_session() as session:
        query = session.query(Prediction).filter(Prediction.timestamp >= cutoff)
        if version:
            query = query.filter(Prediction.model_version == str(version))
        total = query.count()
        labeled = query.filter(Prediction.true_label.is_not(None)).count()
        recent = query.order_by(Prediction.timestamp.desc(), Prediction.id.desc()).limit(5000).all()
        latencies = [row.latency_ms for row in recent]
        requests = [row.request_latency_ms for row in recent if row.request_latency_ms is not None]
        minute = datetime.now(timezone.utc) - timedelta(minutes=1)
        throughput = query.filter(Prediction.timestamp >= minute).count()
        models = [
            serial(row)
            for row in session.query(ModelVersion).order_by(ModelVersion.version.desc()).all()
        ]
        alerts = [
            serial(row)
            for row in session.query(Alert).order_by(Alert.timestamp.desc()).limit(200).all()
        ]
        drift_query = session.query(DriftResult).filter(DriftResult.timestamp >= cutoff)
        if version:
            drift_query = drift_query.filter(DriftResult.model_version == str(version))
        drifts = [
            serial(row)
            for row in drift_query.order_by(DriftResult.timestamp.desc(), DriftResult.id.desc())
            .limit(1000)
            .all()
        ]
        open_alerts = (
            session.query(Alert)
            .filter(Alert.acknowledged.is_(False), Alert.severity != "RESOLVED")
            .count()
        )
        events = [
            serial(row)
            for row in session.query(RetrainingEvent)
            .order_by(RetrainingEvent.timestamp.desc())
            .limit(30)
            .all()
        ]
    beat = queue.state("worker")
    worker_ok = bool(
        beat
        and (datetime.now(timezone.utc) - datetime.fromisoformat(beat["timestamp"])).total_seconds()
        < 30
    )
    drift = queue.state("drift")
    if version and drift and str(drift.get("model_version")) != str(version):
        drift = None
    snapshot = ModelProvider.get_snapshot()
    if drift:
        drift["stale"] = (
            datetime.now(timezone.utc) - datetime.fromisoformat(drift["timestamp"])
        ).total_seconds() > settings.monitoring.check_interval_minutes * 120 or drift[
            "model_version"
        ] != snapshot.version
    return {
        "timestamp": datetime.now(timezone.utc),
        "active_version": snapshot.version,
        "ready": ModelProvider.is_loaded(),
        "worker_online": worker_ok,
        "worker": beat,
        "source": snapshot.bundle.source if snapshot.bundle else "unavailable",
        "stats": {
            "open_alerts": open_alerts,
            "total": total,
            "labeled": labeled,
            "label_coverage": labeled / total if total else 0,
            "throughput_per_min": throughput,
            "p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
            "p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
            "p99_ms": float(np.percentile(latencies, 99)) if latencies else None,
            "request_p95_ms": float(np.percentile(requests, 95)) if requests else None,
            "latency_sample_count": len(latencies),
        },
        "predictions": [serial(row) for row in recent[:1000]],
        "models": models,
        "alerts": alerts,
        "drifts": drifts,
        "events": events,
        "jobs": queue.list(),
        "drift": drift,
        "thresholds": settings.drift_thresholds.model_dump(),
        "promotion": settings.promotion.model_dump(),
        "notifications": {
            "console": settings.alerting.console.enabled,
            "email": settings.alerting.email.enabled,
            "slack": settings.alerting.slack.enabled,
        },
    }
