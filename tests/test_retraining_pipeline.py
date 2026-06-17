"""Unit tests for the retraining pipeline execution orchestrator.

Mocks external services (MLflow, training, evaluation) to verify the correct
sequential execution of retraining, promotion, and deployment rules.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
import numpy as np

from src.pipeline.retrain_pipeline import RetrainingPipeline
from src.data.database import DatabaseManager, ModelVersion
from src.models.trainer import TrainingResult
from src.models.evaluator import ComparisonResult, EvaluationResult


class TestRetrainingPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = MagicMock()
        self.db = DatabaseManager("sqlite:///:memory:")
        self.db.init_db()
        
        # Insert a current production champion version in the mock database
        with self.db.get_session() as session:
            session.add(
                ModelVersion(
                    version=1,
                    mlflow_run_id="run_1",
                    accuracy=0.95,
                    f1_score=0.90,
                    precision=0.92,
                    recall=0.88,
                    auc_roc=0.96,
                    training_date=datetime.now(timezone.utc),
                    is_production=True,
                    deployed_at=datetime.now(timezone.utc)
                )
            )
            
        self.trainer = MagicMock()
        self.evaluator = MagicMock()
        self.registry = MagicMock()
        self.deployer = MagicMock()
        self.alert_manager = MagicMock()
        
        self.pipeline = RetrainingPipeline(
            settings=self.settings,
            db=self.db,
            trainer=self.trainer,
            evaluator=self.evaluator,
            registry=self.registry,
            deployer=self.deployer,
            alert_manager=self.alert_manager
        )

    @patch("src.pipeline.retrain_pipeline.DataLoader")
    @patch("src.pipeline.retrain_pipeline.PredictionLogger")
    def test_pipeline_triggers_training_and_deploys(self, mock_logger_cls, mock_loader_cls) -> None:
        # 1. Setup mock data loader
        import pandas as pd
        mock_loader = MagicMock()
        mock_loader.load_dataset.return_value = pd.DataFrame(columns=[f"V{i}" for i in range(1, 29)] + ["Amount", "Class"])
        mock_loader.preprocess.return_value = (
            np.zeros((10, 29)), np.zeros((5, 29)),
            np.zeros(10), np.zeros(5),
            [f"V{i}" for i in range(1, 29)] + ["Amount"]
        )
        mock_loader_cls.return_value = mock_loader

        # 2. Setup mock prediction logger (returns recent production data to include in retraining)
        mock_logger = MagicMock()
        mock_logger.get_predictions_with_ground_truth.return_value = []
        mock_logger_cls.return_value = mock_logger

        # 3. Setup mock training result
        self.trainer.train.return_value = TrainingResult(
            run_id="new_run_id",
            metrics={"accuracy": 0.98, "f1": 0.95, "precision": 0.96, "recall": 0.94},
            model_uri="runs:/new_run_id/model"
        )
        
        # 4. Setup mock registry (returns new version 2)
        self.registry.register_model.return_value = 2
        
        # 5. Setup mock evaluator comparison (recommend Deploys)
        self.evaluator.evaluate.return_value = EvaluationResult(
            accuracy=0.98, f1=0.95, precision=0.96, recall=0.94, auc_roc=0.98,
            confusion_matrix=[[0,0],[0,0]], classification_report_str=""
        )
        self.evaluator.compare_models.return_value = ComparisonResult(
            is_improved=True,
            metric_deltas={"f1": 0.05},
            recommendation="DEPLOY",
            reason="F1 score improved significantly."
        )

        # 6. Execute retraining
        self.pipeline.execute(trigger_reason="Concept drift detected")

        # 7. Assertions
        # Verify trainer was called
        self.trainer.train.assert_called_once()
        # Verify version was registered
        self.registry.register_model.assert_called_once()
        self.assertEqual(self.registry.register_model.call_args[0][0], "new_run_id")
        # Verify deployment was executed
        self.deployer.deploy.assert_called_once_with(2)
        # Verify database recorded the completed retraining event
        with self.db.get_session() as session:
            from src.data.database import RetrainingEvent
            events = session.query(RetrainingEvent).all()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].status, "COMPLETED")
            self.assertEqual(events[0].new_version, 2)


if __name__ == "__main__":
    unittest.main()
