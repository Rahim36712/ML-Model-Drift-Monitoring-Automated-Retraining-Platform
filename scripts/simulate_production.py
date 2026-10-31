"""Submit an actual inference/feedback simulation to the durable worker."""

import argparse
import sys
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.runtime import api_url, management_token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["stable", "drift"], default="stable")
    parser.add_argument("--samples", type=int, default=500)
    args = parser.parse_args()
    response = requests.post(
        api_url() + "/demo/traffic",
        headers={"X-MLOps-Token": management_token()},
        json={"scenario": args.scenario, "samples": args.samples},
        timeout=15,
    )
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()
