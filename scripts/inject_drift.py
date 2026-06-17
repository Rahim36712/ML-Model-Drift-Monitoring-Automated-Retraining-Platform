#!/usr/bin/env python
"""Inject synthetic drift into the production simulation.

Writes a configuration file to data/drift_config.json which is read by
simulate_production.py to modify feature values or target labels on the fly.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path if not already there
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("inject_drift")

DRIFT_CONFIG_PATH = PROJECT_ROOT / "data" / "drift_config.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject drift into prediction pipeline simulation.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--type",
        choices=["feature_shift", "scale_change", "noise", "label_flip", "severe"],
        help="Type of synthetic drift to inject."
    )
    group.add_argument(
        "--clear",
        action="store_true",
        help="Clear any active drift configurations."
    )
    
    parser.add_argument(
        "--magnitude",
        type=float,
        default=1.5,
        help="Magnitude of feature shift, scale factor, or noise standard deviation."
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.20,
        help="Ratio of flipped labels for concept drift (label_flip/severe)."
    )
    
    args = parser.parse_args()
    
    if args.clear:
        config = {"drift_type": None, "magnitude": 1.5, "flip_ratio": 0.20}
        logger.info("Clearing active drift configurations...")
    else:
        config = {
            "drift_type": args.type,
            "magnitude": args.magnitude,
            "flip_ratio": args.ratio
        }
        logger.info(
            f"Configuring drift injection: type={args.type}, magnitude={args.magnitude}, ratio={args.ratio}"
        )
        
    DRIFT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
        
    logger.info(f"Drift config successfully written to {DRIFT_CONFIG_PATH}.")


if __name__ == "__main__":
    main()
