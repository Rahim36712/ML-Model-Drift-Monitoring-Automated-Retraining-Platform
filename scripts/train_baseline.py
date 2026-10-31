"""Queue baseline training; the API and worker must be running."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config.settings import get_settings
from src.data.database import get_database
from src.jobs import JobQueue

if __name__ == "__main__":
    db = get_database(get_settings().database.url)
    db.init_db()
    job, created = JobQueue(db).enqueue("Baseline requested from CLI", "baseline")
    print(f"Job {job['id']}: {job['status']} (new={created}). Start API and worker to process it.")
