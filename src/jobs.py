"""SQLite durable job queue. A partial unique index enforces single flight."""

from datetime import datetime, timedelta, timezone
import json
import uuid
from sqlalchemy import text

ACTIVE = "'QUEUED','RUNNING','EVALUATING','DEPLOYING'"


def now():
    return datetime.now(timezone.utc).isoformat()


class JobQueue:
    def __init__(self, db):
        self.db = db
        with db.engine.begin() as connection:
            connection.execute(
                text("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
                reason TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                result_json TEXT NOT NULL DEFAULT '{}', owner TEXT,
                active_slot INTEGER NOT NULL DEFAULT 1)""")
            )
            connection.execute(
                text(
                    f"CREATE UNIQUE INDEX IF NOT EXISTS one_active_job ON jobs(active_slot) WHERE status IN ({ACTIVE})"
                )
            )
            connection.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS runtime_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
            )
            connection.execute(text("INSERT OR IGNORE INTO schema_versions VALUES (3)"))

    def enqueue(self, reason, kind="retrain"):
        with self.db.engine.connect() as connection:
            connection.execute(text("BEGIN IMMEDIATE"))
            row = (
                connection.execute(text(f"SELECT * FROM jobs WHERE status IN ({ACTIVE})"))
                .mappings()
                .first()
            )
            if row:
                connection.commit()
                return dict(row), False
            job = {
                "id": uuid.uuid4().hex,
                "kind": kind,
                "status": "QUEUED",
                "reason": reason,
                "created_at": now(),
                "updated_at": now(),
            }
            connection.execute(
                text("""INSERT INTO jobs(id,kind,status,reason,created_at,updated_at)
                VALUES(:id,:kind,:status,:reason,:created_at,:updated_at)"""),
                job,
            )
            connection.commit()
            return job, True

    def claim(self, owner):
        with self.db.engine.connect() as connection:
            connection.execute(text("BEGIN IMMEDIATE"))
            row = (
                connection.execute(
                    text("SELECT * FROM jobs WHERE status='QUEUED' ORDER BY created_at LIMIT 1")
                )
                .mappings()
                .first()
            )
            if row:
                connection.execute(
                    text(
                        "UPDATE jobs SET status='RUNNING', owner=:owner, updated_at=:time WHERE id=:id"
                    ),
                    {"owner": owner, "time": now(), "id": row["id"]},
                )
            connection.commit()
            return dict(row) if row else None

    def update(self, job_id, status, result=None):
        with self.db.engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE jobs SET status=:status, updated_at=:time, result_json=:result WHERE id=:id"
                ),
                {"status": status, "time": now(), "result": json.dumps(result or {}), "id": job_id},
            )

    def recover(self):
        with self.db.engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE retraining_events SET status='INTERRUPTED' WHERE status IN ('RUNNING','STARTED','EVALUATING','DEPLOYING')"
                )
            )
            connection.execute(
                text(
                    "UPDATE jobs SET status='INTERRUPTED', updated_at=:time, result_json=:result WHERE status IN ('RUNNING','EVALUATING','DEPLOYING')"
                ),
                {
                    "time": now(),
                    "result": json.dumps(
                        {"message": "Worker stopped before completion. Submit a new job."}
                    ),
                },
            )

    def list(self, limit=30):
        with self.db.engine.connect() as connection:
            rows = (
                connection.execute(
                    text("SELECT * FROM jobs ORDER BY created_at DESC LIMIT :limit"),
                    {"limit": limit},
                )
                .mappings()
                .all()
            )
        return [{**dict(row), "result": json.loads(row["result_json"])} for row in rows]

    def get(self, job_id):
        with self.db.engine.connect() as connection:
            row = (
                connection.execute(text("SELECT * FROM jobs WHERE id=:id"), {"id": job_id})
                .mappings()
                .first()
            )
        return {**dict(row), "result": json.loads(row["result_json"])} if row else None

    def state(self, key, value=None):
        with self.db.engine.begin() as connection:
            if value is not None:
                connection.execute(
                    text(
                        "INSERT INTO runtime_state(key,value) VALUES(:key,:value) ON CONFLICT(key) DO UPDATE SET value=excluded.value"
                    ),
                    {"key": key, "value": json.dumps(value, default=str)},
                )
            row = connection.execute(
                text("SELECT value FROM runtime_state WHERE key=:key"), {"key": key}
            ).first()
        return json.loads(row[0]) if row else None

    def cooldown(self, minutes):
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
        with self.db.engine.connect() as connection:
            return (
                connection.execute(
                    text("SELECT COUNT(*) FROM jobs WHERE kind='retrain' AND created_at>:cutoff"),
                    {"cutoff": cutoff},
                ).scalar()
                > 0
            )
