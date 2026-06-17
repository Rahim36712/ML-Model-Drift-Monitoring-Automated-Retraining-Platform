"""Unit tests for the PredictionLogger and database interaction.

Tests prediction logging, batch insertion, ground-truth mapping,
and statistics computation.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
import numpy as np

from src.data.database import DatabaseManager
from src.data.logger import PredictionLogger


class TestPredictionLogger(unittest.TestCase):
    def setUp(self) -> None:
        # Create an isolated in-memory database for testing
        self.db = DatabaseManager("sqlite:///:memory:")
        self.db.init_db()
        self.logger = PredictionLogger(self.db)
        
        # Sample transaction features
        self.features = {f"V{i}": 0.1 * i for i in range(1, 29)}
        self.features["Amount"] = 100.0

    def test_log_prediction(self) -> None:
        pred_id = self.logger.log_prediction(
            features=self.features,
            predicted_label=1,
            confidence=0.95,
            model_version="1",
            latency_ms=15.5
        )
        
        self.assertGreater(pred_id, 0)
        
        # Retrieve prediction
        predictions = self.logger.get_predictions(limit=1)
        self.assertEqual(len(predictions), 1)
        
        p = predictions[0]
        self.assertEqual(p.id, pred_id)
        self.assertEqual(p.predicted_label, 1)
        self.assertEqual(p.confidence, 0.95)
        self.assertEqual(p.model_version, "1")
        self.assertEqual(p.latency_ms, 15.5)

    def test_log_ground_truth(self) -> None:
        pred_id = self.logger.log_prediction(
            features=self.features,
            predicted_label=0,
            confidence=0.10,
            model_version="1",
            latency_ms=10.0
        )
        
        # Initially true label is None
        predictions = self.logger.get_predictions(limit=1)
        self.assertIsNone(predictions[0].true_label)
        
        # Log ground truth
        success = self.logger.log_ground_truth(pred_id, 1)
        self.assertTrue(success)
        
        # Verify updated label
        predictions = self.logger.get_predictions(limit=1)
        self.assertEqual(predictions[0].true_label, 1)

    def test_log_batch_predictions(self) -> None:
        batch = [
            {
                "features": self.features,
                "predicted_label": 0,
                "confidence": 0.05,
                "model_version": "1",
                "latency_ms": 12.0
            },
            {
                "features": self.features,
                "predicted_label": 1,
                "confidence": 0.88,
                "model_version": "1",
                "latency_ms": 14.0
            }
        ]
        
        ids = self.logger.log_batch_predictions(batch)
        self.assertEqual(len(ids), 2)
        
        # Verify both exist
        predictions = self.logger.get_predictions(limit=10)
        self.assertEqual(len(predictions), 2)

    def test_get_prediction_stats(self) -> None:
        # Log some mock prediction history
        for i in range(5):
            self.logger.log_prediction(
                features=self.features,
                predicted_label=i % 2,
                confidence=0.5,
                model_version="1",
                latency_ms=20.0
            )
            
        stats = self.logger.get_prediction_stats()
        self.assertEqual(stats["total_count"], 5)
        self.assertAlmostEqual(stats["avg_latency_ms"], 20.0)


if __name__ == "__main__":
    unittest.main()
