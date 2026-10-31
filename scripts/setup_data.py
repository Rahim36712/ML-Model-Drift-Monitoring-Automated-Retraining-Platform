"""Create the fixed historical split inside the dedicated runtime."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.bundle import historical_data

if __name__ == "__main__":
    data = historical_data()
    print(
        f"Source: {data['source']}; training: {len(data['train_y'])}; fixed holdout: {len(data['test_y'])}"
    )
