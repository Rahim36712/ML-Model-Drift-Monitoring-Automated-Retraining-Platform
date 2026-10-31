"""Actual scikit-learn training with local MLflow tracking."""

from dataclasses import dataclass
from typing import Any
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from src.models.evaluator import ModelEvaluator


@dataclass
class TrainingResult:
    run_id: str
    metrics: dict
    model_uri: str
    model: Any = None


class ModelTrainer:
    def __init__(self, settings):
        self.settings = settings

    def train(self, X_train, y_train, X_test, y_test, params=None, feature_names=None):
        mlflow.set_tracking_uri(self.settings.mlflow.tracking_uri)
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(self.settings.mlflow.experiment_name)
        experiment_id = (
            experiment.experiment_id
            if experiment
            else client.create_experiment(
                self.settings.mlflow.experiment_name,
                artifact_location=self.settings.mlflow.artifact_location,
            )
        )
        mlflow.set_experiment(experiment_id=experiment_id)
        params = params or self.settings.model_params.model_dump()
        model = RandomForestClassifier(**params, n_jobs=2)
        with mlflow.start_run() as run:
            model.fit(X_train, y_train)
            result = ModelEvaluator().evaluate(model, X_test, y_test)
            metrics = {k: v for k, v in result.to_dict().items() if isinstance(v, (int, float))}
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            return TrainingResult(run.info.run_id, metrics, f"runs:/{run.info.run_id}/model", model)
