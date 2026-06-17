#!/usr/bin/env python
"""Train the baseline machine learning model.

Initialises the database, loads/generates the fraud-detection dataset,
preprocesses the data, saves preprocessor and reference distributions,
trains a RandomForestClassifier, logs all metrics/params/plots to MLflow,
registers the model, and deploys it to production.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
import sys
from pathlib import Path

# Add project root to path if not already there
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings
from src.data.database import get_database
from src.data.loader import DataLoader
from src.models.trainer import ModelTrainer
from src.models.registry import ModelRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("train_baseline")


def main() -> None:
    logger.info("Starting baseline model training pipeline...")
    settings = get_settings()

    # 1. Initialize SQLite Database
    logger.info("Initializing SQLite database...")
    db = get_database()
    db.init_db()
    logger.info("Database initialized successfully.")

    # 2. Load and Preprocess Data
    logger.info("Loading dataset...")
    loader = DataLoader()
    df = loader.load_dataset()
    
    logger.info("Preprocessing dataset...")
    X_train, X_test, y_train, y_test, feature_names = loader.preprocess(df)
    
    # 3. Save preprocessor and reference distribution
    logger.info("Saving reference distribution...")
    loader.save_reference_distribution(X_train, feature_names)

    # 4. Train Model with MLflow tracking
    logger.info("Training baseline RandomForest model...")
    trainer = ModelTrainer(settings)
    
    # We pass the default model parameters
    params = settings.model_params.model_dump() if hasattr(settings, "model_params") else None
    result = trainer.train(X_train, y_train, X_test, y_test, params=params, feature_names=feature_names)
    logger.info(f"Model training complete. MLflow Run ID: {result.run_id}")
    logger.info(f"Test metrics logged: {result.metrics}")

    # 5. Register Model in MLflow registry
    logger.info("Registering model with MLflow Registry...")
    registry = ModelRegistry(settings)
    version = registry.register_model(result.run_id)
    logger.info(f"Model registered as version {version}.")

    # 6. Promote Model to Production
    logger.info("Promoting model version to production alias...")
    registry.promote_to_production(version=version)
    
    # 7. Record model version in SQLite
    logger.info("Recording model version in SQLite database...")
    registry.record_version_in_db(
        db=db,
        version=version,
        run_id=result.run_id,
        metrics=result.metrics,
        is_production=True,
        deployed_at=datetime.now(timezone.utc)
    )
    
    logger.info("Baseline model training and deployment pipeline completed successfully!")


if __name__ == "__main__":
    main()
