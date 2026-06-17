"""Unit tests for the MLOps Drift Detection Engine.

Validates feature data drift (PSI/KL), prediction distribution drift (Hellinger),
and performance metric degradation (Concept Drift).
"""

from __future__ import annotations

import unittest
import numpy as np

from src.monitoring.data_drift import DataDriftDetector
from src.monitoring.prediction_drift import PredictionDriftDetector
from src.monitoring.concept_drift import ConceptDriftDetector
from src.config.settings import DataDriftThresholds, PredictionDriftThresholds, ConceptDriftThresholds, ThresholdPair


class TestDriftDetection(unittest.TestCase):
    def setUp(self) -> None:
        # Standard warning/critical threshold setups
        self.data_thresholds = DataDriftThresholds(
            psi=ThresholdPair(warning=0.10, critical=0.25),
            kl_divergence=ThresholdPair(warning=0.5, critical=1.5)
        )
        self.pred_thresholds = PredictionDriftThresholds(
            hellinger=ThresholdPair(warning=0.10, critical=0.20),
            distribution_shift=ThresholdPair(warning=0.10, critical=0.20)
        )
        self.concept_thresholds = ConceptDriftThresholds(
            accuracy_drop=ThresholdPair(warning=0.02, critical=0.05),
            f1_drop=ThresholdPair(warning=0.03, critical=0.05),
            precision_drop=ThresholdPair(warning=0.03, critical=0.05),
            recall_drop=ThresholdPair(warning=0.03, critical=0.05)
        )

    def test_data_drift_psi_stable(self) -> None:
        np.random.seed(42)
        ref_x = np.random.normal(0, 1, 1000)
        curr_x = np.random.normal(0, 1, 1000) # no shift
        
        ref_dict = {"feature1": ref_x}
        detector = DataDriftDetector(ref_dict, self.data_thresholds)
        
        # Test PSI calculation directly
        psi = detector.calculate_psi(ref_x, curr_x)
        self.assertLess(psi, 0.10)
        
        # Test KL divergence directly
        kl = detector.calculate_kl_divergence(ref_x, curr_x)
        self.assertLess(kl, 0.5)

    def test_data_drift_psi_shifted(self) -> None:
        np.random.seed(42)
        ref_x = np.random.normal(0, 1, 1000)
        curr_x = np.random.normal(1.5, 1.2, 1000) # significant shift
        
        ref_dict = {"feature1": ref_x}
        detector = DataDriftDetector(ref_dict, self.data_thresholds)
        
        psi = detector.calculate_psi(ref_x, curr_x)
        self.assertGreater(psi, 0.25)
        
        # Verify check_drift output
        result = detector.check_drift(curr_x.reshape(-1, 1), ["feature1"])
        self.assertTrue(result.is_drifted)
        self.assertEqual(result.severity, "critical")
        self.assertIn("feature1", result.drifted_features)

    def test_prediction_drift_hellinger(self) -> None:
        np.random.seed(42)
        # Stable
        base_preds = np.random.choice([0, 1], size=500, p=[0.98, 0.02])
        base_confs = np.random.uniform(0.85, 0.99, 500)
        
        curr_preds = base_preds.copy()
        curr_confs = base_confs.copy()
        
        detector = PredictionDriftDetector(base_preds, base_confs, self.pred_thresholds)
        res = detector.check_drift(curr_preds, curr_confs)
        self.assertFalse(res.is_drifted)
        self.assertEqual(res.severity, "none")
        
        # Shifted (extreme prediction positive rate shift)
        drifted_preds = np.random.choice([0, 1], size=500, p=[0.70, 0.30]) # positive rate goes from 2% to 30%
        drifted_confs = np.random.uniform(0.50, 0.75, 500)
        
        res_drift = detector.check_drift(drifted_preds, drifted_confs)
        self.assertTrue(res_drift.is_drifted)
        self.assertEqual(res_drift.severity, "critical")

    def test_concept_drift(self) -> None:
        baseline_metrics = {"accuracy": 0.95, "f1": 0.90, "precision": 0.92, "recall": 0.88}
        detector = ConceptDriftDetector(baseline_metrics, self.concept_thresholds)
        
        # Stable predictions
        y_true = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0]) # 100% accuracy
        
        res = detector.check_drift(y_true, y_pred)
        self.assertFalse(res.is_drifted)
        
        # Shifted / degraded performance
        y_degraded = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1]) # flipped predictions, poor accuracy
        res_drift = detector.check_drift(y_true, y_degraded)
        self.assertTrue(res_drift.is_drifted)
        self.assertEqual(res_drift.severity, "critical")


if __name__ == "__main__":
    unittest.main()
