"""Serialized serving activation with immutable version snapshots."""

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
import logging
from src.data.database import ModelVersion
from src.models.bundle import load_bundle

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServingSnapshot:
    model: object
    preprocessor: object
    version: int | None
    run_id: str | None
    bundle: object = None


class ModelProvider:
    _snapshot = ServingSnapshot(None, None, None, None)
    lock = RLock()

    @classmethod
    def set_active_model(cls, model, preprocessor, version, mlflow_run_id=None, bundle=None):
        cls._snapshot = ServingSnapshot(model, preprocessor, version, mlflow_run_id, bundle)

    @classmethod
    def get_snapshot(cls):
        return cls._snapshot

    @classmethod
    def get_active_model(cls):
        value = cls._snapshot
        return value.model, value.preprocessor, value.version, value.run_id

    @classmethod
    def is_loaded(cls):
        return cls._snapshot.model is not None


class ModelDeployer:
    def __init__(self, settings, db, registry):
        self.settings, self.db, self.registry = settings, db, registry

    def deploy(self, version):
        with ModelProvider.lock:
            previous = ModelProvider.get_snapshot()
            alias_changed = False
            try:
                bundle = load_bundle(version)
                with self.db.get_session() as session:
                    row = session.query(ModelVersion).filter_by(version=version).one_or_none()
                    if row is None:
                        raise ValueError("Version metadata is missing")
                    self.registry.promote_to_production(version=version)
                    alias_changed = True
                    session.query(ModelVersion).update({"is_production": False})
                    row.is_production = True
                    row.deployed_at = datetime.now(timezone.utc)
                ModelProvider.set_active_model(
                    bundle.pipeline, None, version, bundle.run_id, bundle
                )
                return True
            except Exception:
                logger.exception("Activation failed; serving snapshot preserved")
                if alias_changed and previous.version is not None:
                    try:
                        self.registry.promote_to_production(version=previous.version)
                    except Exception:
                        logger.exception("Alias reconciliation required on restart")
                return False

    def rollback(self, version):
        return self.deploy(version)

    def get_deployment_history(self, limit=50):
        with self.db.get_session() as session:
            rows = (
                session.query(ModelVersion).order_by(ModelVersion.version.desc()).limit(limit).all()
            )
            return [
                {column.name: getattr(row, column.name) for column in row.__table__.columns}
                for row in rows
            ]
