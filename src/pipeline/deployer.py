"""Model deployment and rollback manager.

Handles promotion of model versions, updating production flags in the local
database, and updating the in-memory active model reference in the serving layer.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable

from src.data.database import DatabaseManager, ModelVersion
from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelProvider:
    """In-memory cache for the active production model and preprocessor.

    Used by the serving layer (FastAPI) to predict with minimal latency
    without querying MLflow or database on every request.
    """

    _model: Any = None
    _preprocessor: Any = None
    _version: int | None = None
    _mlflow_run_id: str | None = None

    @classmethod
    def set_active_model(
        cls,
        model: Any,
        preprocessor: Any,
        version: int,
        mlflow_run_id: str | None = None,
    ) -> None:
        """Atomically swap the loaded production model."""
        cls._model = model
        cls._preprocessor = preprocessor
        cls._version = version
        cls._mlflow_run_id = mlflow_run_id
        logger.info(
            "ModelProvider updated in-memory model to version %d (run %s)",
            version,
            mlflow_run_id,
        )

    @classmethod
    def get_active_model(cls) -> tuple[Any, Any, int | None, str | None]:
        """Retrieve the currently loaded model, preprocessor, version and run_id."""
        return cls._model, cls._preprocessor, cls._version, cls._mlflow_run_id

    @classmethod
    def is_loaded(cls) -> bool:
        """Check if a model has been loaded."""
        return cls._model is not None and cls._preprocessor is not None


class ModelDeployer:
    """Manages model deployments, database updates, and rollbacks.

    Args:
        settings: Global configuration settings.
        db: Database manager instance.
        registry: Model registry wrapper instance.
    """

    def __init__(
        self,
        settings: Any,
        db: DatabaseManager,
        registry: ModelRegistry,
    ) -> None:
        self.settings = settings
        self.db = db
        self.registry = registry
        self._model_name: str = getattr(settings, "model_name", "FraudDetector")

    def deploy(self, version: int) -> bool:
        """Deploy a specific model version.

        Updates the production alias in MLflow, sets is_production flag in DB,
        loads the model and preprocessor into the in-memory ModelProvider.
        """
        try:
            logger.info("Deploying model version %d...", version)

            # 1. Update MLflow Alias
            self.registry.promote_to_production(self._model_name, version)

            # 2. Update Local DB Production Flag
            with self.db.get_session() as session:
                # Mark all versions as not production
                session.query(ModelVersion).update(
                    {ModelVersion.is_production: False}
                )

                # Set this version as production
                row = (
                    session.query(ModelVersion)
                    .filter(ModelVersion.version == version)
                    .one_or_none()
                )
                if row:
                    row.is_production = True
                    row.deployed_at = datetime.now(timezone.utc)
                    run_id = row.mlflow_run_id
                else:
                    # If model metadata wasn't mirrored yet (should not happen in standard flow)
                    logger.warning(
                        "Model version %d not found in local DB. Cannot fetch run_id.",
                        version,
                    )
                    run_id = None

            # 3. Load model and preprocessor
            model = self.registry.load_model(self._model_name, version)
            
            # Load preprocessor (StandardScaler)
            from src.data.loader import DataLoader
            loader = DataLoader()
            preprocessor = loader.load_preprocessor()

            # 4. Swap in-memory reference
            ModelProvider.set_active_model(model, preprocessor, version, run_id)

            logger.info("Successfully deployed version %d", version)
            return True
        except Exception:
            logger.exception("Failed to deploy model version %d", version)
            return False

    def rollback(self, version: int) -> bool:
        """Roll back the active model to a previous version."""
        logger.info("Initiating rollback to version %d", version)
        return self.deploy(version)

    def get_deployment_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Retrieve history of all model versions and their production status."""
        try:
            with self.db.get_session() as session:
                rows = (
                    session.query(ModelVersion)
                    .order_by(ModelVersion.version.desc())
                    .limit(limit)
                    .all()
                )
                
                history = []
                for row in rows:
                    history.append({
                        "version": row.version,
                        "mlflow_run_id": row.mlflow_run_id,
                        "accuracy": row.accuracy,
                        "f1_score": row.f1_score,
                        "precision": row.precision,
                        "recall": row.recall,
                        "auc_roc": row.auc_roc,
                        "training_date": row.training_date.isoformat() if row.training_date else None,
                        "is_production": row.is_production,
                        "deployed_at": row.deployed_at.isoformat() if row.deployed_at else None,
                    })
                return history
        except Exception:
            logger.exception("Failed to fetch model deployment history")
            return []
