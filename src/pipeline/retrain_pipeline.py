"""Real retraining with immutable holdout, version bundles, and explicit gates."""

import json
import logging
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from src.data.database import ModelVersion, RetrainingEvent
from src.data.logger import PredictionLogger
from src.models.bundle import (
    FEATURES,
    ModelBundle,
    fingerprint,
    historical_data,
    preprocessing,
    save_bundle,
    load_bundle,
)
from src.models.evaluator import promotion_gates

logger = logging.getLogger(__name__)


class RetrainingPipeline:
    def __init__(self, settings, db, trainer, evaluator, registry, deployer, alert_manager):
        self.settings, self.db = settings, db
        self.trainer, self.evaluator, self.registry = trainer, evaluator, registry
        self.deployer, self.alert_manager = deployer, alert_manager
        self.last_result = {}

    def execute(self, trigger_reason, progress=None):
        progress = progress or (lambda status, result: None)
        event_id = None
        try:
            data = historical_data()
            with self.db.get_session() as session:
                active = session.query(ModelVersion).filter_by(is_production=True).one_or_none()
                old_version = active.version if active else None
                event = RetrainingEvent(
                    trigger_reason=trigger_reason,
                    old_version=old_version or 0,
                    old_f1=active.f1_score if active else 0,
                    status="RUNNING",
                )
                session.add(event)
                session.flush()
                event_id = event.id
            X_train, y_train = data["train_x"], data["train_y"]
            # Only unique new labeled production examples enter retraining.
            records = PredictionLogger(self.db).get_predictions_with_ground_truth(n=20000)
            records.sort(key=lambda p: (p.timestamp, p.id))
            seen = set(data["holdout_hashes"]) | set(data["training_hashes"])
            prod_x, prod_y = [], []
            for row in records:
                features = json.loads(row.features_json)
                vector = np.array([features[f] for f in FEATURES], dtype=float)
                key = fingerprint(vector.reshape(1, -1))[0]
                if key in seen or not np.isfinite(vector).all():
                    continue
                seen.add(key)
                prod_x.append(vector)
                prod_y.append(row.true_label)
            recent_x = recent_y = None
            if prod_x:
                cut = max(1, int(len(prod_x) * 0.8))
                X_train = np.vstack([X_train, prod_x[:cut]])
                y_train = np.r_[y_train, prod_y[:cut]]
                remaining = np.array(prod_y[cut:])
                if (
                    len(remaining) >= 100
                    and min(np.sum(remaining == 0), np.sum(remaining == 1)) >= 5
                ):
                    recent_x, recent_y = np.array(prod_x[cut:]), remaining
            transform = preprocessing()
            train_x = transform.fit_transform(X_train)
            test_x = transform.transform(data["test_x"])
            trained = self.trainer.train(
                train_x, y_train, test_x, data["test_y"], feature_names=FEATURES
            )
            candidate = Pipeline([("preprocess", transform), ("classifier", trained.model)])
            with mlflow.start_run(run_id=trained.run_id):
                mlflow.sklearn.log_model(candidate, "serving_pipeline")
                mlflow.set_tags({"data_source": data["source"], "trigger": trigger_reason})
            candidate_historical = self.evaluator.evaluate(
                candidate, data["test_x"], data["test_y"]
            )
            gates = {
                "passed": True,
                "checks": {"initial_baseline": True},
                "selected_set": "historical",
                "historical": [candidate_historical.to_dict()],
                "recent": None,
            }
            progress("EVALUATING", {"message": "Comparing real fraud metrics on held-out raw rows"})
            if old_version:
                champion = load_bundle(old_version).pipeline
                historical = (
                    self.evaluator.evaluate(champion, data["test_x"], data["test_y"]),
                    candidate_historical,
                )
                recent = None
                if recent_x is not None:
                    recent = (
                        self.evaluator.evaluate(champion, recent_x, recent_y),
                        self.evaluator.evaluate(candidate, recent_x, recent_y),
                    )
                gates = promotion_gates(historical, recent, self.settings.promotion)
            gates["recent_evaluation_note"] = (
                "Recent holdout eligible"
                if recent_x is not None
                else "Insufficient recent evaluation data: using fixed historical holdout"
            )
            gates["training_samples"] = len(y_train)
            gates["event_id"] = event_id
            self.registry.client.log_dict(trained.run_id, gates, "evaluation/gates.json")
            self.last_result = gates
            if not gates["passed"]:
                self.last_result["message"] = (
                    "Challenger rejected by quality gates. Active champion retained."
                )
                self._finish(event_id, "REJECTED", candidate_historical.f1)
                self.alert_manager.send_alert(
                    "WARNING",
                    "retraining",
                    "quality_gate",
                    candidate_historical.f1,
                    0,
                    "Challenger rejected by quality gates; champion retained.",
                )
                return False
            version = self.registry.register_model(trained.run_id)
            self.registry.record_version_in_db(self.db, version, trained.run_id, trained.metrics)
            baseline = {
                key: getattr(candidate_historical, key)
                for key in ["accuracy", "f1", "precision", "recall"]
            }
            bundle = ModelBundle(
                version,
                trained.run_id,
                candidate,
                {name: X_train[:, i] for i, name in enumerate(FEATURES)},
                candidate.predict(data["test_x"]),
                candidate.predict_proba(data["test_x"])[:, 1],
                baseline,
                data["source"],
            )
            bundle_path = save_bundle(bundle)
            self.registry.client.log_artifact(trained.run_id, str(bundle_path), "serving_bundle")
            progress("DEPLOYING", {**gates, "version": version})
            if not self.deployer.deploy(version):
                raise RuntimeError(
                    "API did not acknowledge activation; previous champion remains active"
                )
            self.last_result = {
                **gates,
                "version": version,
                "message": "Model activated and acknowledged by serving API",
            }
            self._finish(event_id, "COMPLETED", candidate_historical.f1, version)
            self.alert_manager.send_alert(
                "RESOLVED",
                "retraining",
                "deployed",
                candidate_historical.f1,
                0,
                f"Model version {version} activated successfully.",
            )
            return True
        except Exception as exc:
            logger.exception("Retraining failed")
            self.last_result = {**self.last_result, "message": str(exc), "failed": True}
            if event_id is not None:
                self._finish(event_id, "FAILED", 0)
            return False

    def _finish(self, event_id, status, f1, version=None):
        with self.db.get_session() as session:
            event = session.get(RetrainingEvent, event_id)
            event.status, event.new_f1, event.new_version = status, f1, version
