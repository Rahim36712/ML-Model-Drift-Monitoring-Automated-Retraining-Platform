"""MLflow registry adapter; complete serving bundles are stored separately."""

from datetime import datetime, timezone
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from src.data.database import ModelVersion


class ModelRegistry:
    def __init__(self, settings):
        self.settings = settings
        self.name = settings.mlflow.model_name
        mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
        self.client = MlflowClient(tracking_uri=settings.mlflow.tracking_uri)

    def register_model(self, run_id, name=None):
        result = mlflow.register_model(f"runs:/{run_id}/serving_pipeline", name or self.name)
        return int(result.version)

    def promote_to_production(self, name=None, version=None):
        self.client.set_registered_model_alias(name or self.name, "champion", str(version))

    def load_model(self, name=None, version=None):
        return mlflow.sklearn.load_model(f"models:/{name or self.name}/{version}")

    def get_production_model(self, name=None):
        return mlflow.sklearn.load_model(f"models:/{name or self.name}@champion")

    def record_version_in_db(
        self, db, version, run_id, metrics, is_production=False, deployed_at=None
    ):
        with db.get_session() as session:
            if is_production:
                session.query(ModelVersion).update({"is_production": False})
            row = session.query(ModelVersion).filter_by(version=version).one_or_none()
            if row is None:
                row = ModelVersion(
                    version=version, mlflow_run_id=run_id, training_date=datetime.now(timezone.utc)
                )
                session.add(row)
            for name in ["accuracy", "precision", "recall", "auc_roc"]:
                setattr(row, name, metrics.get(name))
            row.f1_score = metrics["f1"]
            row.is_production = is_production
            row.deployed_at = deployed_at
