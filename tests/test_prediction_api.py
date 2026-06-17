"""Integration tests for the FastAPI prediction and serving endpoints.

Verifies serving logic, single/batch prediction schemas, and ground-truth mapping.
"""

from __future__ import annotations

import unittest
from fastapi.testclient import TestClient
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.api.app import create_app
from src.pipeline.deployer import ModelProvider
from src.data.database import get_database


class DummyModel:
    """Mock model implementing scikit-learn classification interfaces."""
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0], dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # returns [[0.95, 0.05]] for each sample
        res = np.zeros((X.shape[0], 2))
        res[:, 0] = 0.95
        res[:, 1] = 0.05
        return res


class DummyPreprocessor:
    """Mock preprocessor implementing StandardScaler transform interfaces."""
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X.copy()


class TestPredictionAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Configure in-memory database to be used by the API instance
        cls.db = get_database("sqlite:///:memory:")
        cls.db.init_db()
        
        # Deploy a mock model inside serving cache so endpoints don't return 503
        cls.mock_model = DummyModel()
        cls.mock_preprocessor = DummyPreprocessor()
        ModelProvider.set_active_model(
            model=cls.mock_model,
            preprocessor=cls.mock_preprocessor,
            version=1,
            mlflow_run_id="mock_run_id_123"
        )
        
        # Instantiate test client
        cls.app = create_app()
        cls.client = TestClient(cls.app)

    def test_health_check(self) -> None:
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    def test_single_prediction(self) -> None:
        features = {f"V{i}": 0.0 for i in range(1, 29)}
        features["Amount"] = 50.0
        
        resp = self.client.post("/predict", json={"features": features})
        self.assertEqual(resp.status_code, 200)
        
        data = resp.json()
        self.assertIn("prediction_id", data)
        self.assertEqual(data["predicted_label"], 0)
        self.assertAlmostEqual(data["confidence"], 0.05)
        self.assertEqual(data["model_version"], "1")

    def test_batch_prediction(self) -> None:
        features = {f"V{i}": 0.0 for i in range(1, 29)}
        features["Amount"] = 50.0
        
        payload = {
            "predictions": [
                {"features": features},
                {"features": features}
            ]
        }
        
        resp = self.client.post("/predict/batch", json=payload)
        self.assertEqual(resp.status_code, 200)
        
        data = resp.json()
        self.assertEqual(len(data["predictions"]), 2)
        self.assertEqual(data["predictions"][0]["predicted_label"], 0)

    def test_submit_ground_truth(self) -> None:
        # First log a prediction to get a valid prediction_id
        features = {f"V{i}": 0.0 for i in range(1, 29)}
        features["Amount"] = 50.0
        resp = self.client.post("/predict", json={"features": features})
        pred_id = resp.json()["prediction_id"]
        
        # Submit ground truth
        gt_resp = self.client.post(f"/ground-truth/{pred_id}", json={"true_label": 0})
        self.assertEqual(gt_resp.status_code, 200)
        self.assertTrue(gt_resp.json()["success"])


if __name__ == "__main__":
    unittest.main()
