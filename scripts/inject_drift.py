"""Submit a labeled synthetic feature-shift scenario through the API."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.simulate_production import main

if __name__ == "__main__":
    if "--type" in sys.argv:
        position = sys.argv.index("--type")
        selected = sys.argv[position + 1]
        if selected not in ["severe", "feature_shift"]:
            raise SystemExit(
                "Supported showcase scenario: --type severe (source labels are never fabricated)."
            )
        sys.argv[position : position + 2] = ["--scenario", "drift"]
    elif "--scenario" not in sys.argv:
        sys.argv.extend(["--scenario", "drift"])
    main()
