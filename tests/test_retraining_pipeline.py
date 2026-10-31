"""Integration: real MLflow, training, immutable artifacts, and rejection."""

from dataclasses import replace
from unittest.mock import Mock
import numpy as np
from src.alerting.alert_manager import AlertManager
from src.models.bundle import historical_data, load_bundle, save_bundle, fingerprint
from src.models.evaluator import ModelEvaluator
from src.models.trainer import ModelTrainer
from src.models.registry import ModelRegistry
from src.pipeline.deployer import ModelDeployer, ModelProvider
from src.pipeline.retrain_pipeline import RetrainingPipeline
from src.data.database import ModelVersion, RetrainingEvent
from src.data.logger import PredictionLogger


def make_pipeline(runtime, deployer=None):
    settings, db = runtime
    registry = ModelRegistry(settings)
    return RetrainingPipeline(
        settings,
        db,
        ModelTrainer(settings),
        ModelEvaluator(),
        registry,
        deployer or ModelDeployer(settings, db, registry),
        AlertManager(db, settings.alerting),
    )


def test_real_training_reload_rejection_and_rollback(runtime):
    pipeline = make_pipeline(runtime)
    assert pipeline.execute("baseline")
    settings, db = runtime
    original = ModelProvider.get_snapshot()
    # Exercise the real monitoring-to-state path, including decision timestamps.
    from src.service import monitor
    from src.jobs import JobQueue

    training = historical_data()
    logger = PredictionLogger(db)
    for row, label in zip(training["train_x"][:500], training["train_y"][:500]):
        prediction_id = logger.log_prediction(
            dict(zip(original.bundle.feature_names, row.tolist())),
            int(label),
            float(label),
            str(original.version),
            1.0,
        )
        logger.log_ground_truth(prediction_id, int(label))
    observed = monitor(db, settings)
    assert observed["sample_count"] == 500
    assert JobQueue(db).state("drift")["window"] == observed["window"]

    bundle = load_bundle(original.version)
    historical = historical_data()
    assert set(fingerprint(historical["test_x"])).isdisjoint(fingerprint(historical["train_x"]))
    np.testing.assert_allclose(
        bundle.pipeline.predict_proba(historical["test_x"][:5]),
        original.model.predict_proba(historical["test_x"][:5]),
    )
    # Inject a historical holdout row as feedback; it must never enter training.
    logger = PredictionLogger(db)
    sample = dict(zip(bundle.feature_names, historical["test_x"][0].tolist()))
    row_id = logger.log_prediction(sample, 0, 0.1, str(bundle.version), 1)
    logger.log_ground_truth(row_id, int(historical["test_y"][0]))
    assert not pipeline.execute("identical candidate must be rejected")
    assert not pipeline.last_result.get("failed")
    assert ModelProvider.get_snapshot() is original
    with db.get_session() as session:
        assert (
            session.query(RetrainingEvent).order_by(RetrainingEvent.id.desc()).first().status
            == "REJECTED"
        )
        assert session.query(ModelVersion).filter_by(is_production=True).count() == 1
    version2 = pipeline.registry.register_model(bundle.run_id)
    save_bundle(replace(bundle, version=version2))
    pipeline.registry.record_version_in_db(db, version2, bundle.run_id, bundle.baseline_metrics)
    assert pipeline.deployer.deploy(version2)
    assert pipeline.deployer.rollback(bundle.version)
    assert ModelProvider.get_snapshot().version == bundle.version


def test_failed_activation_not_reported_completed(runtime):
    deployer = Mock()
    deployer.deploy.return_value = False
    pipeline = make_pipeline(runtime, deployer)
    assert not pipeline.execute("failed activation")
    assert pipeline.last_result["failed"]
    with runtime[1].get_session() as session:
        assert session.query(RetrainingEvent).one().status == "FAILED"
        assert session.query(ModelVersion).filter_by(is_production=True).count() == 0
    assert not ModelProvider.is_loaded()
