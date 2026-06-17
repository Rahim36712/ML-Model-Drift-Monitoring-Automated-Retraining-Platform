"""Unit tests for the Retraining Decision Engine.

Validates rule triggers for automated model retraining based on drift check summaries.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.decision.retraining_engine import RetrainingDecisionEngine, RetrainingDecision
from src.monitoring.drift_manager import DriftSummary
from src.monitoring.data_drift import DataDriftResult
from src.monitoring.prediction_drift import PredictionDriftResult
from src.monitoring.concept_drift import ConceptDriftResult
from src.config.settings import DriftThresholds, DataDriftThresholds, PredictionDriftThresholds, ConceptDriftThresholds, ThresholdPair


class TestRetrainingDecisionEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.thresholds = DriftThresholds(
            data_drift=DataDriftThresholds(
                psi=ThresholdPair(warning=0.10, critical=0.25),
                kl_divergence=ThresholdPair(warning=0.5, critical=1.5)
            ),
            prediction_drift=PredictionDriftThresholds(
                hellinger=ThresholdPair(warning=0.10, critical=0.20),
                distribution_shift=ThresholdPair(warning=0.10, critical=0.20)
            ),
            concept_drift=ConceptDriftThresholds(
                accuracy_drop=ThresholdPair(warning=0.02, critical=0.05),
                f1_drop=ThresholdPair(warning=0.03, critical=0.05),
                precision_drop=ThresholdPair(warning=0.03, critical=0.05),
                recall_drop=ThresholdPair(warning=0.03, critical=0.05)
            )
        )
        self.engine = RetrainingDecisionEngine(self.thresholds, db=None)

    def test_no_drift_no_action(self) -> None:
        summary = DriftSummary(
            data_drift=DataDriftResult(
                feature_psi={"V1": 0.02}, overall_psi=0.02, feature_kl={"V1": 0.05},
                is_drifted=False, severity="none", drifted_features=[], timestamp=datetime.now(timezone.utc)
            ),
            prediction_drift=PredictionDriftResult(
                hellinger_distance=0.04, baseline_positive_rate=0.02, current_positive_rate=0.02,
                baseline_mean_confidence=0.95, current_mean_confidence=0.95, is_drifted=False, severity="none", timestamp=datetime.now(timezone.utc)
            ),
            concept_drift=ConceptDriftResult(
                current_metrics={"f1": 0.94}, baseline_metrics={"f1": 0.95}, metric_deltas={"f1": -0.01},
                is_drifted=False, degraded_metrics=[], severity="none", timestamp=datetime.now(timezone.utc)
            ),
            overall_status="healthy",
            timestamp=datetime.now(timezone.utc),
            checks_performed=["data", "prediction", "concept"]
        )
        
        decision = self.engine.evaluate(summary)
        self.assertFalse(decision.should_retrain)
        self.assertEqual(decision.action, "NO_ACTION")
        self.assertEqual(decision.urgency, "NONE")

    def test_critical_data_drift_triggers_retrain(self) -> None:
        summary = DriftSummary(
            data_drift=DataDriftResult(
                feature_psi={"V1": 0.35}, overall_psi=0.35, feature_kl={"V1": 0.8},
                is_drifted=True, severity="critical", drifted_features=["V1"], timestamp=datetime.now(timezone.utc)
            ),
            prediction_drift=None,
            concept_drift=None,
            overall_status="critical",
            timestamp=datetime.now(timezone.utc),
            checks_performed=["data"]
        )
        
        decision = self.engine.evaluate(summary)
        self.assertTrue(decision.should_retrain)
        self.assertEqual(decision.action, "RETRAIN")
        self.assertEqual(decision.urgency, "HIGH")
        self.assertIn("critical_data_drift", decision.triggered_rules)

    def test_critical_f1_drop_triggers_retrain(self) -> None:
        summary = DriftSummary(
            data_drift=None,
            prediction_drift=None,
            concept_drift=ConceptDriftResult(
                current_metrics={"f1": 0.88}, baseline_metrics={"f1": 0.95}, metric_deltas={"f1": -0.07}, # > 0.05 drop
                is_drifted=True, degraded_metrics=["f1"], severity="critical", timestamp=datetime.now(timezone.utc)
            ),
            overall_status="critical",
            timestamp=datetime.now(timezone.utc),
            checks_performed=["concept"]
        )
        
        decision = self.engine.evaluate(summary)
        self.assertTrue(decision.should_retrain)
        self.assertEqual(decision.action, "RETRAIN")
        self.assertEqual(decision.urgency, "HIGH")
        self.assertIn("f1_degradation", decision.triggered_rules)


if __name__ == "__main__":
    unittest.main()
