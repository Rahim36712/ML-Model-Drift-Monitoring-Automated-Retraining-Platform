#!/usr/bin/env python
"""Simulate production transaction traffic.

Loads the processed test dataset, samples transactions, checks for drift configuration,
applies synthetic drift if configured, sends prediction requests to the FastAPI serving layer,
and asynchronously submits ground-truth labels after a delay.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import requests

# Add project root to path if not already there
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import DataLoader
from src.data.drift_injector import DriftInjector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("simulate_production")

DRIFT_CONFIG_PATH = PROJECT_ROOT / "data" / "drift_config.json"
API_URL = "http://localhost:8000"


def load_drift_config() -> dict:
    """Load the current drift configuration from disk."""
    if not DRIFT_CONFIG_PATH.exists():
        return {"drift_type": None, "magnitude": 1.5, "flip_ratio": 0.20}
    try:
        with open(DRIFT_CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"drift_type": None, "magnitude": 1.5, "flip_ratio": 0.20}


def main() -> None:
    logger.info("Initializing production traffic simulator...")
    
    # 1. Load test dataset
    loader = DataLoader()
    try:
        X_test, _, y_test, _, feature_names = loader.load_processed_data()
    except Exception:
        logger.error("Processed dataset not found! Please run scripts/train_baseline.py first.")
        sys.exit(1)

    logger.info(f"Loaded {len(X_test)} candidate test samples for simulation.")
    
    # Queue for feedback loop: stores tuples of (prediction_id, true_label, submit_time)
    feedback_queue: list[tuple[int, int, float]] = []
    
    # Wait for API to be ready
    logger.info("Waiting for FastAPI serving layer to be reachable...")
    api_ready = False
    for _ in range(15):
        try:
            resp = requests.get(f"{API_URL}/health")
            if resp.status_code == 200:
                api_ready = True
                break
        except requests.RequestException:
            pass
        time.sleep(2)
        
    if not api_ready:
        logger.error("FastAPI server is not reachable at http://localhost:8000. Start it first!")
        sys.exit(1)
        
    logger.info("FastAPI serving layer is ready. Starting traffic stream.")
    
    idx = 0
    try:
        while True:
            # Check current drift config
            config = load_drift_config()
            drift_type = config.get("drift_type")
            magnitude = config.get("magnitude", 1.5)
            flip_ratio = config.get("flip_ratio", 0.20)
            
            # Select transaction sample
            sample_idx = idx % len(X_test)
            features = X_test[sample_idx].copy()
            true_label = int(y_test[sample_idx])
            
            # Apply drift if configured
            if drift_type:
                feat_matrix = features.reshape(1, -1)
                if drift_type == "feature_shift":
                    features = DriftInjector.inject_feature_shift(feat_matrix, shift_magnitude=magnitude)[0]
                elif drift_type == "scale_change":
                    features = DriftInjector.inject_scale_change(feat_matrix, scale_factor=magnitude)[0]
                elif drift_type == "noise":
                    features = DriftInjector.inject_noise(feat_matrix, noise_std=magnitude)[0]
                elif drift_type == "label_flip":
                    # Label flip is concept drift, so we flip the target label when sending feedback
                    if random.random() < flip_ratio:
                        true_label = 1 - true_label
                elif drift_type == "severe":
                    # Severe is feature shift + label flip
                    features = DriftInjector.inject_feature_shift(feat_matrix, shift_magnitude=magnitude)[0]
                    if random.random() < flip_ratio:
                        true_label = 1 - true_label
            
            # Convert features to dictionary matching schema
            features_dict = {feature_names[i]: float(features[i]) for i in range(len(feature_names))}
            
            # Send prediction request
            try:
                resp = requests.post(f"{API_URL}/predict", json={"features": features_dict})
                if resp.status_code == 200:
                    result = resp.json()
                    pred_id = result["prediction_id"]
                    pred_label = result["predicted_label"]
                    confidence = result["confidence"]
                    
                    drift_status = f" [DRIFT: {drift_type}]" if drift_type else ""
                    logger.info(
                        f"Tx {idx:04d}: Pred={pred_label} (Conf={confidence:.3f}), True={true_label}{drift_status}"
                    )
                    
                    # Queue ground truth feedback to be sent in 5 seconds
                    feedback_queue.append((pred_id, true_label, time.time() + 5.0))
                else:
                    logger.warning(f"Prediction request failed: {resp.status_code} - {resp.text}")
            except requests.RequestException as e:
                logger.error(f"Failed to connect to serving layer for prediction: {e}")
                
            # Process feedback queue (send feedback for past predictions)
            current_time = time.time()
            to_send = [item for item in feedback_queue if item[2] <= current_time]
            feedback_queue = [item for item in feedback_queue if item[2] > current_time]
            
            for pred_id, label, _ in to_send:
                if pred_id == -1:
                    continue
                try:
                    feedback_resp = requests.post(
                        f"{API_URL}/ground-truth/{pred_id}", 
                        json={"true_label": label}
                    )
                    if feedback_resp.status_code != 200:
                        logger.warning(f"Feedback submission failed: {feedback_resp.status_code} - {feedback_resp.text}")
                except requests.RequestException as e:
                    logger.error(f"Failed to connect for feedback submission: {e}")
            
            idx += 1
            # Wait 1 second between transactions
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        logger.info("Traffic simulator stopped by user.")


if __name__ == "__main__":
    main()
