"""Real HTTP contracts, validation, batch equivalence, and authentication."""

from fastapi.testclient import TestClient
import numpy as np
import pytest
from src.api.app import create_app
from src.pipeline.deployer import ModelProvider
from src.runtime import management_token


class DummyModel:
    calls = 0

    def predict(self, X):
        return (X[:, 0] > 0).astype(int)

    def predict_proba(self, X):
        self.calls += 1
        prob = np.where(X[:, 0] > 0, 0.8, 0.05)
        return np.column_stack([1 - prob, prob])


@pytest.fixture
def client(runtime):
    with TestClient(create_app()) as client:
        ModelProvider.set_active_model(DummyModel(), None, 1, "test")
        yield client


def features(value=0):
    return {**{f"V{i}": value for i in range(1, 29)}, "Amount": 50.0}


def test_single_batch_equivalence_and_vectorization(client):
    payload = {"features": features(1)}
    single = client.post("/predict", json=payload)
    assert single.status_code == 200
    snapshot = ModelProvider.get_snapshot()
    before = snapshot.model.calls
    batch = client.post("/predict/batch", json={"predictions": [payload, payload]}).json()
    assert snapshot.model.calls - before == 1
    assert all(row["confidence"] == single.json()["confidence"] for row in batch["predictions"])
    prediction_id = single.json()["prediction_id"]
    assert client.post(f"/ground-truth/{prediction_id}", json={"true_label": 1}).status_code == 200
    assert client.get("/ready").status_code == 200
    assert client.get("/health").json()["status"] == "ok"
    assert "X-Process-Time" in single.headers


@pytest.mark.parametrize(
    "payload",
    [
        {"predictions": []},
        {"predictions": [{"features": {}}]},
        {"predictions": [{"features": features()}] * 1001},
    ],
)
def test_invalid_batches(client, payload):
    assert client.post("/predict/batch", json=payload).status_code == 422


def test_nonfinite_feature_and_management_auth(client):
    value = features()
    value["Amount"] = "NaN"
    assert client.post("/predict", json={"features": value}).status_code == 422
    assert client.post("/models/retrain").status_code == 401
    headers = {"X-MLOps-Token": management_token()}
    first = client.post("/models/retrain", headers=headers).json()
    second = client.post("/models/retrain", headers=headers).json()
    assert first["job_id"] == second["job_id"]
    assert not second["created"]
    assert client.get("/jobs/" + first["job_id"], headers=headers).status_code == 200
    assert client.get("/dashboard/snapshot", headers=headers).status_code == 200


def test_missing_model_readiness(client):
    ModelProvider.set_active_model(None, None, None)
    assert client.get("/ready").status_code == 503
    assert client.post("/predict", json={"features": features()}).status_code == 503


def test_reset_is_confirmed_and_keeps_user_predictions(client):
    headers = {"X-MLOps-Token": management_token()}
    user_id = client.post("/predict", json={"features": features()}).json()["prediction_id"]
    demo = client.post(
        "/demo/predict/batch", headers=headers, json={"predictions": [{"features": features()}]}
    ).json()
    demo_id = demo["predictions"][0]["prediction_id"]
    assert (
        client.post("/demo/reset", headers=headers, json={"confirmation": "wrong"}).status_code
        == 422
    )
    assert (
        client.post("/demo/reset", headers=headers, json={"confirmation": "RESET DEMO"}).status_code
        == 200
    )
    assert client.post(f"/ground-truth/{user_id}", json={"true_label": 0}).status_code == 200
    assert client.post(f"/ground-truth/{demo_id}", json={"true_label": 0}).status_code == 404
