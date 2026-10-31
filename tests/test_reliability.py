"""Boundary cases that previously caused misleading drift and job results."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from unittest.mock import Mock
import numpy as np
import pytest
from sqlalchemy import text
from src.alerting.alert_manager import AlertManager
from src.data.database import Alert, Prediction
from src.jobs import JobQueue
from src.models.bundle import validate_dataset
from src.models.evaluator import EvaluationResult, promotion_gates
from src.monitoring.data_drift import DataDriftDetector
from src.monitoring.prediction_drift import PredictionDriftDetector
from src.monitoring.drift_manager import DriftSummary
from src.monitoring.concept_drift import ConceptDriftResult
from src.decision.retraining_engine import RetrainingDecisionEngine


@pytest.mark.parametrize(
    "metric", [DataDriftDetector.calculate_psi, DataDriftDetector.calculate_kl_divergence]
)
def test_histogram_tails_and_constant_reference(metric):
    assert metric(np.ones(100), np.ones(100)) == pytest.approx(0)
    assert np.isfinite(metric(np.ones(100), np.full(100, 1000)))
    assert metric(np.ones(100), np.full(100, 1000)) > 0.25
    assert metric(np.arange(100), np.arange(100) + 1000) > 0.25
    with pytest.raises(ValueError):
        metric(np.ones(2), np.array([]))
    with pytest.raises(ValueError):
        metric(np.ones(2), np.array([np.nan]))


def test_constant_prediction_and_invalid_samples():
    assert PredictionDriftDetector.calculate_hellinger(np.ones(10), np.ones(10)) == 0
    with pytest.raises(ValueError):
        PredictionDriftDetector.calculate_hellinger(np.array([]), np.ones(10))


def test_f1_improvement_does_not_trigger_degradation(runtime):
    settings, _ = runtime
    concept = ConceptDriftResult(
        current_metrics={"f1": 0.98},
        baseline_metrics={"f1": 0.8},
        metric_deltas={"f1": 0.18},
        is_drifted=False,
        degraded_metrics=[],
        severity="none",
        timestamp=datetime.now(timezone.utc),
    )
    decision = RetrainingDecisionEngine(settings.drift_thresholds).evaluate(
        DriftSummary(concept_drift=concept)
    )
    assert not decision.should_retrain


def test_quality_gate_recall_and_historical_safety():
    def metrics(f1, recall):
        return EvaluationResult(0.9, f1, 0.9, recall, 0.9, [[1, 0], [0, 1]], "")

    champion = metrics(0.8, 0.9)
    assert promotion_gates((champion, metrics(0.81, 0.9)))["passed"]
    assert not promotion_gates((champion, metrics(0.9, 0.8)))["passed"]
    assert not promotion_gates((champion, metrics(0.78, 0.9)), (champion, metrics(0.9, 0.9)))[
        "passed"
    ]


def test_queue_single_claim_and_restart_recovery(runtime):
    queue = JobQueue(runtime[1])
    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(lambda _: queue.enqueue("test"), range(8)))
    assert len({row[0]["id"] for row in rows}) == 1
    with ThreadPoolExecutor(8) as pool:
        claims = list(pool.map(lambda i: queue.claim(str(i)), range(8)))
    assert sum(row is not None for row in claims) == 1
    queue.recover()
    assert queue.list()[0]["status"] == "INTERRUPTED"
    assert queue.enqueue("next")[1]


def test_notification_failure_is_durable_and_deduplicated(runtime):
    settings, db = runtime
    manager = AlertManager(db, settings.alerting)
    manager._email_notifier = Mock()
    manager._email_notifier.send.return_value = False
    manager.send_alert("CRITICAL", "data", "psi", 0.5, 0.25)
    manager.send_alert("CRITICAL", "data", "psi", 0.6, 0.25)
    with db.get_session() as session:
        alert = session.query(Alert).one()
        assert alert.occurrences == 2
        assert '"email": "failed"' in alert.delivery_json
    manager._email_notifier.send.assert_called_once()


def test_migration_idempotent(runtime):
    _, db = runtime
    db.init_db()
    db.init_db()
    with db.engine.connect() as connection:
        assert connection.execute(text("SELECT MAX(version) FROM schema_versions")).scalar() >= 2
        assert connection.execute(text("PRAGMA journal_mode")).scalar() == "wal"


def test_failed_activation_preserves_snapshot(runtime):
    from src.pipeline.deployer import ModelDeployer, ModelProvider

    model = object()
    ModelProvider.set_active_model(model, None, 7, "previous")
    before = ModelProvider.get_snapshot()
    registry = Mock()
    deployer = ModelDeployer(runtime[0], runtime[1], registry)
    assert not deployer.deploy(99999)
    assert ModelProvider.get_snapshot() is before
    registry.promote_to_production.assert_not_called()


def test_atomic_snapshot_under_concurrent_updates():
    from src.pipeline.deployer import ModelProvider
    from types import SimpleNamespace

    failures = []

    def update():
        for version in range(500):
            bundle = SimpleNamespace(version=version)
            ModelProvider.set_active_model(bundle, None, version, str(version), bundle)

    def read():
        for _ in range(3000):
            snapshot = ModelProvider.get_snapshot()
            if snapshot.bundle and snapshot.bundle.version != snapshot.version:
                failures.append(snapshot)

    with ThreadPoolExecutor(4) as pool:
        futures = [pool.submit(update), pool.submit(read), pool.submit(read)]
        for future in futures:
            future.result()
    assert not failures
    ModelProvider.set_active_model(None, None, None)


def test_dataset_validation():
    import pandas as pd
    from src.models.bundle import FEATURES

    frame = pd.DataFrame(np.zeros((50, 29)), columns=FEATURES)
    frame["Class"] = [0] * 25 + [1] * 25
    with pytest.raises(ValueError, match="conflicting"):
        validate_dataset(frame)
    frame["Amount"] = np.arange(50, dtype=float)
    assert len(validate_dataset(frame)) == 50
    frame.loc[0, "Amount"] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        validate_dataset(frame)


def test_monitoring_state_roundtrip_with_decision_timestamp(runtime):
    queue = JobQueue(runtime[1])
    value = {"decision": {"timestamp": datetime.now(timezone.utc)}, "overall_status": "healthy"}
    queue.state("drift", value)
    restored = queue.state("drift")
    assert isinstance(restored["decision"]["timestamp"], str)
    assert restored["overall_status"] == "healthy"


def test_upgrade_legacy_telemetry_database(tmp_path):
    import sqlite3
    from src.data.database import DatabaseManager

    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE predictions (
            id INTEGER PRIMARY KEY, timestamp DATETIME NOT NULL, model_version VARCHAR(64) NOT NULL,
            features_json TEXT NOT NULL, predicted_label INTEGER NOT NULL, confidence FLOAT NOT NULL,
            true_label INTEGER, latency_ms FLOAT NOT NULL)""")
        connection.execute("""INSERT INTO predictions VALUES
            (1, '2026-01-01 00:00:00', '1', '{}', 0, 0.1, NULL, 2.0)""")
    db = DatabaseManager("sqlite:///" + path.as_posix())
    try:
        db.init_db()
        db.init_db()
        with db.get_session() as session:
            row = session.get(Prediction, 1)
            assert row.source == "api"
            assert row.scenario == "live"
            assert row.request_latency_ms is None
    finally:
        db.engine.dispose()
