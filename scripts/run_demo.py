#!/usr/bin/env python
"""Run the complete MLOps platform demo.

Ensures baseline model is trained, pre-populates the SQLite database with 
historical predictions to make dashboard charts look alive immediately,
starts the FastAPI server, the Dash dashboard, and the production traffic simulator,
and provides an interactive prompt to inject drift and trigger manual checks.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests

# Add project root to path if not already there
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings
from src.data.database import get_database, Prediction, DriftResult, ModelVersion
from src.data.loader import DataLoader
from src.data.logger import PredictionLogger
from src.models.registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("run_demo")

API_URL = "http://localhost:8000"
DASH_URL = "http://localhost:8050"


def prepopulate_database(db, n: int = 150) -> None:
    """Pre-populate the predictions table with historical entries."""
    logger.info(f"Pre-populating database with {n} historical predictions...")
    
    loader = DataLoader()
    X_train, X_test, y_train, y_test, feature_names = loader.load_processed_data()
    
    # Load settings to get registered model name
    settings = get_settings()
    registry = ModelRegistry(settings)
    
    # Get active production model version
    try:
        with db.get_session() as session:
            prod_ver = session.query(ModelVersion).filter(ModelVersion.is_production == True).first()
            version_str = str(prod_ver.version) if prod_ver else "1"
    except Exception:
        version_str = "1"

    # We will log predictions spaced over the last 2 hours
    base_time = datetime.now(timezone.utc) - timedelta(hours=2)
    records = []
    
    for i in range(n):
        sample_idx = i % len(X_test)
        features = X_test[sample_idx]
        true_label = int(y_test[sample_idx])
        
        # Make a mock prediction: 98% accuracy on normal, some false results
        pred_label = true_label if random_ratio(0.98) else (1 - true_label)
        confidence = float(0.92 if pred_label == 1 else 0.05 + 0.1 * abs(features[0]))
        latency = float(random_range(12.0, 35.0))
        
        # Features dict
        feat_dict = {feature_names[j]: float(features[j]) for j in range(len(feature_names))}
        
        timestamp = base_time + timedelta(seconds=i * 48) # space them evenly
        
        records.append(
            Prediction(
                timestamp=timestamp,
                model_version=version_str,
                features_json=json.dumps(feat_dict),
                predicted_label=pred_label,
                confidence=confidence,
                true_label=true_label,
                latency_ms=latency
            )
        )
        
    with db.get_session() as session:
        # Clear existing predictions to have a clean slate if requested,
        # but let's keep them if they exist. Here we just bulk insert.
        session.bulk_save_objects(records)
        
    logger.info("Database pre-populated successfully.")


def random_ratio(p: float) -> bool:
    import random
    return random.random() < p


def random_range(a: float, b: float) -> float:
    import random
    return random.uniform(a, b)


def main() -> None:
    logger.info("Starting MLOps Platform Demo...")
    settings = get_settings()
    
    # Clear any active drift configs
    drift_config_path = PROJECT_ROOT / "data" / "drift_config.json"
    if drift_config_path.exists():
        os.remove(drift_config_path)

    # 1. Ensure raw/creditcard.csv and baseline model exist
    db = get_database()
    db.init_db()
    
    baseline_trained = False
    try:
        with db.get_session() as session:
            prod_model = session.query(ModelVersion).filter(ModelVersion.is_production == True).first()
            if prod_model:
                baseline_trained = True
    except Exception:
        pass
        
    if not baseline_trained:
        logger.info("No baseline model version found in SQLite DB. Running baseline training script...")
        # Run train_baseline.py
        subprocess.run([sys.executable, "scripts/train_baseline.py"], check=True)
        
    # 2. Pre-populate predictions for chart visibility
    try:
        prepopulate_database(db, n=150)
    except Exception as e:
        logger.warning(f"Could not pre-populate database: {e}. Demo will start with empty logs.")

    # 3. Start FastAPI Server
    logger.info("Starting FastAPI serving app on port 8000...")
    fastapi_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.app:create_app", "--host", "127.0.0.1", "--port", "8000", "--factory"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # 4. Start Dash Dashboard
    logger.info("Starting Dash monitoring dashboard on port 8050...")
    dash_proc = subprocess.Popen(
        [sys.executable, "-m", "src.dashboard.app"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # 5. Start Production Traffic Simulator
    logger.info("Starting production traffic simulator...")
    sim_proc = subprocess.Popen(
        [sys.executable, "scripts/simulate_production.py"],
        stdout=sys.stdout, # let simulator log to console
        stderr=sys.stderr
    )
    
    logger.info("=" * 60)
    logger.info("MLOps Platform is running!")
    logger.info(f"API Endpoint:         {API_URL}")
    logger.info(f"Monitoring Dashboard: {DASH_URL}")
    logger.info("=" * 60)
    
    time.sleep(5)
    
    try:
        print("\n>>> DEMO INSTRUCTIONS <<<")
        print(f"1. Open your browser and navigate to the dashboard: {DASH_URL}")
        print("2. Notice that the status is green (✓ Healthy) and prediction volume graphs are populated.")
        print("3. Press ENTER to inject severe data + concept drift and trigger retraining.")
        input("\nPress ENTER to proceed...")
        
        # Inject Drift
        logger.info("Injecting severe drift (feature shift + flipped labels)...")
        subprocess.run([sys.executable, "scripts/inject_drift.py", "--type", "severe", "--magnitude", "2.8", "--ratio", "0.25"], check=True)
        
        logger.info("Allowing 10 seconds of drifted predictions to stream in...")
        time.sleep(10)
        
        logger.info("Triggering a manual drift check to speed up detection...")
        try:
            resp = requests.post(f"{API_URL}/drift/run")
            if resp.status_code == 200:
                summary = resp.json()
                logger.info(f"Manual Drift Check run! Overall Status: {summary['overall_status']}")
                logger.info(f"Breached features: {summary.get('data_drift', {}).get('drifted_features', [])}")
                logger.info("Automatic Retraining Pipeline triggered!")
            else:
                logger.warning(f"Manual drift check request failed: {resp.status_code} - {resp.text}")
        except Exception as e:
            logger.error(f"Error triggering manual drift check: {e}")
            
        print("\n>>> RETRAINING WATCH <<<")
        print("1. Go back to the dashboard and navigate to the 'Alerts' and 'Model Registry' tabs.")
        print("2. Observe the alerts logged in response to the drift.")
        print("3. Retraining will complete in the background shortly, registering a new model version.")
        print("4. Press ENTER when you are done to shut down the demo.")
        input("\nPress ENTER to exit...")
        
    except KeyboardInterrupt:
        logger.info("Shutting down demo...")
    finally:
        # Clean up subprocesses
        logger.info("Terminating all processes...")
        sim_proc.terminate()
        dash_proc.terminate()
        fastapi_proc.terminate()
        
        # Wait a bit for them to clean up
        sim_proc.wait()
        dash_proc.wait()
        fastapi_proc.wait()
        
        # Clear drift config
        if drift_config_path.exists():
            os.remove(drift_config_path)
            
        logger.info("Demo processes stopped successfully.")


if __name__ == "__main__":
    main()
