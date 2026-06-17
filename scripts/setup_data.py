#!/usr/bin/env python
"""Set up the raw data directories and datasets.

Checks for the existence of creditcard.csv in data/raw. If not present,
it generates the synthetic fallback dataset and saves it to data/raw/creditcard.csv
so that raw data is available for retraining and analysis.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to path if not already there
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("setup_data")


def main() -> None:
    logger.info("Initializing raw data setup...")
    loader = DataLoader()
    
    csv_path = loader.raw_dir / "creditcard.csv"
    if csv_path.exists():
        logger.info(f"Kaggle dataset already exists at {csv_path}. Ready to go!")
    else:
        logger.info("Kaggle creditcard.csv not found in data/raw/. Generating synthetic fallback...")
        df = loader._generate_synthetic()
        
        # Save raw dataset so it exists
        df.to_csv(csv_path, index=False)
        logger.info(f"Synthetic dataset saved to {csv_path}.")
        
    logger.info("Raw data setup complete!")


if __name__ == "__main__":
    main()
