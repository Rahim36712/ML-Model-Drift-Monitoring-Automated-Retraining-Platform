"""End-to-end automated retraining workflow.

Orchestrates: collecting recent production data, combining it with baseline data,
retraining the model, evaluating against the production champion, promoting,
deploying, and notifying.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
import json
from typing import Any

import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException

from src.data.database import DatabaseManager, RetrainingEvent, ModelVersion
from src.data.logger import PredictionLogger
from src.data.loader import DataLoader
from src.models.trainer import ModelTrainer, TrainingResult
from src.models.evaluator import ModelEvaluator, EvaluationResult
from src.models.registry import ModelRegistry
from src.pipeline.deployer import ModelDeployer
from src.alerting.alert_manager import AlertManager

logger = logging.getLogger(__name__)


class RetrainingPipeline:
    """Orchestrates the automated retraining lifecycle.

    Args:
        settings: Global configuration settings.
        db: Database manager instance.
        trainer: ModelTrainer instance.
        evaluator: ModelEvaluator instance.
        registry: ModelRegistry instance.
        deployer: ModelDeployer instance.
        alert_manager: AlertManager instance.
    """

    def __init__(
        self,
        settings: Any,
        db: DatabaseManager,
        trainer: ModelTrainer,
        evaluator: ModelEvaluator,
        registry: ModelRegistry,
        deployer: ModelDeployer,
        alert_manager: AlertManager,
    ) -> None:
        self.settings = settings
        self.db = db
        self.trainer = trainer
        self.evaluator = evaluator
        self.registry = registry
        self.deployer = deployer
        self.alert_manager = alert_manager
        self._model_name: str = getattr(settings, "model_name", "FraudDetector")

    def execute(self, trigger_reason: str) -> bool:
        """Run the retraining pipeline.

        Returns:
            True if retraining completed and successfully deployed a new model,
            False otherwise.
        """
        logger.info("Starting automated retraining pipeline. Reason: %s", trigger_reason)
        
        # 1. Audit Log: Retrieve old version and start event
        old_version: int | None = None
        old_f1: float = 0.0
        
        try:
            with self.db.get_session() as session:
                active_version = (
                    session.query(ModelVersion)
                    .filter(ModelVersion.is_production == True)
                    .one_or_none()
                )
                if active_version:
                    old_version = active_version.version
                    old_f1 = active_version.f1_score

            event = RetrainingEvent(
                timestamp=datetime.now(timezone.utc),
                trigger_reason=trigger_reason,
                old_version=old_version,
                old_f1=old_f1,
                status="STARTED",
            )
            with self.db.get_session() as session:
                session.add(event)
                session.flush()
                event_id = event.id
            logger.info("Recorded retraining event id=%d as STARTED", event_id)
        except Exception:
            logger.exception("Failed to start retraining audit log. Proceeding anyway.")
            event_id = None

        try:
            # 2. Collect production predictions with ground truth
            pred_logger = PredictionLogger(self.db)
            recent_predictions = pred_logger.get_predictions_with_ground_truth(n=1000)
            
            loader = DataLoader()
            
            # Load original raw data
            df_combined = loader.load_dataset()
            
            if recent_predictions:
                logger.info("Found %d recent predictions with ground truth for retraining.", len(recent_predictions))
                # Convert predictions to a dataframe
                prod_records = []
                feature_columns = loader.get_feature_names()
                
                for pred in recent_predictions:
                    try:
                        features = json.loads(pred.features_json)
                        # Ensure features are ordered correctly
                        record = {col: features[col] for col in feature_columns if col in features}
                        record["Class"] = pred.true_label
                        prod_records.append(record)
                    except Exception:
                        logger.exception("Failed parsing prediction ID %d features.", pred.id)
                        continue
                
                if prod_records:
                    df_prod = pd.DataFrame(prod_records)
                    # Merge with original dataset
                    df_combined = pd.concat([df_combined, df_prod], ignore_index=True)
                    logger.info("Combined dataset shape: %s", str(df_combined.shape))
            else:
                logger.info("No recent production ground truths found. Retraining on baseline dataset.")

            # 3. Preprocess combined dataset (fits a new StandardScaler and saves reference distributions)
            X_train, X_test, y_train, y_test, feature_names = loader.preprocess(df_combined)

            # 4. Train new model using trainer
            logger.info("Training candidate model...")
            train_result = self.trainer.train(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                feature_names=feature_names,
            )
            
            new_f1 = train_result.metrics.get("f1", 0.0)
            logger.info("Candidate model trained. Run ID: %s, F1: %.4f", train_result.run_id, new_f1)

            # 5. Evaluate and Compare models
            current_model = None
            try:
                current_model = self.registry.get_production_model(self._model_name)
            except MlflowException:
                logger.warning("No production model registered yet.")
            except Exception:
                logger.exception("Error fetching production model from MLflow.")

            if current_model is not None:
                # Evaluate production model on new test split
                current_metrics = self.evaluator.evaluate(current_model, X_test, y_test)
                # Evaluate new model on new test split
                new_metrics = self.evaluator.evaluate(train_result.model, X_test, y_test)
                
                comparison = self.evaluator.compare_models(current_metrics, new_metrics)
                logger.info("Comparison result: recommendation=%s, reason=%s", comparison.recommendation, comparison.reason)
                should_deploy = (comparison.recommendation == "DEPLOY")
            else:
                # No production model exists, deploy immediately
                logger.info("No current production model to compare. Proceeding to deploy new model.")
                should_deploy = True
                comparison = None

            # 6. Deployment Gate
            if should_deploy:
                # Register new model
                new_version = self.registry.register_model(train_result.run_id, self._model_name)
                
                # Mirror metadata to DB as production
                self.registry.record_version_in_db(
                    db=self.db,
                    version=new_version,
                    run_id=train_result.run_id,
                    metrics=train_result.metrics,
                    is_production=True,
                    deployed_at=datetime.now(timezone.utc),
                )
                
                # Update MLflow alias and hot-swap
                self.deployer.deploy(new_version)
                
                # Save the new training data distribution as the reference distribution
                loader.save_reference_distribution(X_train, feature_names)

                # Update RetrainingEvent audit row to COMPLETED
                if event_id is not None:
                    with self.db.get_session() as session:
                        db_event = session.get(RetrainingEvent, event_id)
                        if db_event:
                            db_event.new_version = new_version
                            db_event.new_f1 = new_f1
                            db_event.status = "COMPLETED"
                
                # Alert
                msg = f"🟢 Automated Retraining COMPLETED & DEPLOYED.\nDeployed version {new_version}.\nTrigger: {trigger_reason}"
                self.alert_manager.send_alert(
                    severity="RESOLVED",
                    drift_type="retraining",
                    metric_name="retraining_status",
                    metric_value=new_f1,
                    threshold=old_f1,
                    message=msg,
                )
                logger.info("Retraining pipeline successfully completed. Model promoted to version %d.", new_version)
                return True
            else:
                # Keep current model
                # Update RetrainingEvent audit row to REJECTED
                if event_id is not None:
                    with self.db.get_session() as session:
                        db_event = session.get(RetrainingEvent, event_id)
                        if db_event:
                            db_event.new_f1 = new_f1
                            db_event.status = "REJECTED"
                
                # Alert
                msg = f"🟡 Automated Retraining REJECTED.\nNew model did not outperform production version. Kept version {old_version}.\nReason: {comparison.reason if comparison else 'N/A'}"
                self.alert_manager.send_alert(
                    severity="WARNING",
                    drift_type="retraining",
                    metric_name="retraining_status",
                    metric_value=new_f1,
                    threshold=old_f1,
                    message=msg,
                )
                logger.info("Retraining pipeline completed. Candidate model rejected. Keeping version %s.", str(old_version))
                return False

        except Exception as e:
            logger.exception("Exception occurred during retraining pipeline execution.")
            
            # Update RetrainingEvent audit row to FAILED
            if event_id is not None:
                try:
                    with self.db.get_session() as session:
                        db_event = session.get(RetrainingEvent, event_id)
                        if db_event:
                            db_event.status = "FAILED"
                except Exception:
                    logger.exception("Failed to update retraining audit status to FAILED.")

            # Alert
            self.alert_manager.send_alert(
                severity="CRITICAL",
                drift_type="retraining",
                metric_name="retraining_status",
                metric_value=0.0,
                threshold=old_f1,
                message=f"🔴 Automated Retraining FAILED: {str(e)}"
            )
            return False
