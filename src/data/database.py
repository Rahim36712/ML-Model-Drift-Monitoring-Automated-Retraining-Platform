"""SQLAlchemy 2.0 database layer with SQLite backend.

Defines the ORM models for predictions, drift results, alerts,
model versions, and retraining events.  Provides a thin
``DatabaseManager`` wrapper around engine + session lifecycle and a
module-level ``get_database()`` singleton factory.

Usage::

    from src.data.database import get_database

    db = get_database()
    db.init_db()

    with db.get_session() as session:
        session.add(Prediction(...))
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Generator

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from src.config.settings import PROJECT_ROOT, get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------
Base = declarative_base()


def _utcnow() -> datetime:
    """Return the current UTC timestamp (timezone-aware)."""
    return datetime.now(timezone.utc)


# ===================================================================
# ORM Models
# ===================================================================


class Prediction(Base):  # type: ignore[misc]
    """A single model-inference record.

    Attributes:
        id: Auto-incrementing primary key.
        timestamp: UTC time the prediction was made.
        model_version: Semantic or integer version string.
        features_json: JSON-serialised feature dict.
        predicted_label: Model output class (0 / 1).
        confidence: Predicted probability for the positive class.
        true_label: Ground-truth label (filled in asynchronously).
        latency_ms: Wall-clock inference latency in milliseconds.
    """

    __tablename__ = "predictions"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    timestamp: datetime = Column(DateTime, default=_utcnow, nullable=False)
    model_version: str = Column(String(64), nullable=False)
    features_json: str = Column(Text, nullable=False)
    predicted_label: int = Column(Integer, nullable=False)
    confidence: float = Column(Float, nullable=False)
    true_label: int | None = Column(Integer, nullable=True)
    latency_ms: float = Column(Float, nullable=False)

    # Indexes for common queries (timestamp range scans, version filters)
    __table_args__ = (
        Index("ix_predictions_timestamp", "timestamp"),
        Index("ix_predictions_model_version", "model_version"),
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction(id={self.id}, label={self.predicted_label}, "
            f"conf={self.confidence:.3f}, v={self.model_version})>"
        )


class DriftResult(Base):  # type: ignore[misc]
    """Stores a single drift-metric evaluation result.

    Attributes:
        drift_type: One of ``'data'``, ``'prediction'``, ``'concept'``.
        metric_name: E.g. ``'psi'``, ``'hellinger'``, ``'f1_drop'``.
        is_breached: Whether the metric exceeded its threshold.
    """

    __tablename__ = "drift_results"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    timestamp: datetime = Column(DateTime, default=_utcnow, nullable=False)
    drift_type: str = Column(String(32), nullable=False)
    metric_name: str = Column(String(64), nullable=False)
    metric_value: float = Column(Float, nullable=False)
    threshold: float = Column(Float, nullable=False)
    is_breached: bool = Column(Boolean, nullable=False)
    window_start: datetime = Column(DateTime, nullable=False)
    window_end: datetime = Column(DateTime, nullable=False)
    details_json: str | None = Column(Text, nullable=True)

    def __repr__(self) -> str:
        breach = "BREACHED" if self.is_breached else "ok"
        return (
            f"<DriftResult(id={self.id}, {self.drift_type}/{self.metric_name}"
            f"={self.metric_value:.4f} [{breach}])>"
        )


class Alert(Base):  # type: ignore[misc]
    """An alert dispatched through one or more notification channels.

    Attributes:
        severity: ``'WARNING'``, ``'CRITICAL'``, or ``'RESOLVED'``.
        channel: Delivery channel name (``'console'``, ``'slack'``, etc.).
    """

    __tablename__ = "alerts"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    timestamp: datetime = Column(DateTime, default=_utcnow, nullable=False)
    severity: str = Column(String(16), nullable=False)
    drift_type: str = Column(String(32), nullable=False)
    message: str = Column(Text, nullable=False)
    channel: str = Column(String(32), nullable=False)
    acknowledged: bool = Column(Boolean, default=False, nullable=False)

    def __repr__(self) -> str:
        return (
            f"<Alert(id={self.id}, severity={self.severity}, "
            f"drift={self.drift_type}, ack={self.acknowledged})>"
        )


class ModelVersion(Base):  # type: ignore[misc]
    """Metadata for a trained model version.

    Attributes:
        version: Integer version counter.
        is_production: ``True`` when this version is currently serving.
        deployed_at: UTC timestamp when promoted to production (nullable).
    """

    __tablename__ = "model_versions"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    version: int = Column(Integer, nullable=False, unique=True)
    mlflow_run_id: str = Column(String(64), nullable=False)
    accuracy: float = Column(Float, nullable=False)
    f1_score: float = Column(Float, nullable=False)
    precision: float = Column(Float, nullable=False)
    recall: float = Column(Float, nullable=False)
    auc_roc: float | None = Column(Float, nullable=True)
    training_date: datetime = Column(DateTime, nullable=False)
    is_production: bool = Column(Boolean, default=False, nullable=False)
    deployed_at: datetime | None = Column(DateTime, nullable=True)

    __table_args__ = (Index("ix_model_versions_version", "version"),)

    def __repr__(self) -> str:
        prod = " [PROD]" if self.is_production else ""
        return (
            f"<ModelVersion(v={self.version}, f1={self.f1_score:.4f}"
            f"{prod})>"
        )


class RetrainingEvent(Base):  # type: ignore[misc]
    """Audit trail for automated retraining runs.

    Attributes:
        status: ``'STARTED'``, ``'COMPLETED'``, ``'FAILED'``, or
            ``'REJECTED'``.
    """

    __tablename__ = "retraining_events"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    timestamp: datetime = Column(DateTime, default=_utcnow, nullable=False)
    trigger_reason: str = Column(Text, nullable=False)
    old_version: int = Column(Integer, nullable=False)
    new_version: int | None = Column(Integer, nullable=True)
    old_f1: float = Column(Float, nullable=False)
    new_f1: float | None = Column(Float, nullable=True)
    status: str = Column(String(16), nullable=False)

    def __repr__(self) -> str:
        return (
            f"<RetrainingEvent(id={self.id}, v{self.old_version}→"
            f"v{self.new_version}, status={self.status})>"
        )


# ===================================================================
# Database manager
# ===================================================================


class DatabaseManager:
    """Thin wrapper around SQLAlchemy engine and session lifecycle.

    Args:
        db_url: SQLAlchemy connection string.  Defaults to the value
            from ``Settings.database.url``.

    Example::

        db = DatabaseManager()
        db.init_db()
        with db.get_session() as s:
            s.query(Prediction).count()
    """

    def __init__(self, db_url: str | None = None) -> None:
        if db_url is None:
            db_url = get_settings().database.url

        self._ensure_sqlite_parent(db_url)

        self._db_url = db_url
        self._engine = create_engine(
            db_url,
            echo=False,
            future=True,
            connect_args={"check_same_thread": False}
            if db_url.startswith("sqlite")
            else {},
        )
        self._session_factory: sessionmaker[Session] = sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
        )
        logger.info("DatabaseManager initialised (url=%s).", db_url)

    # ---- public API ---------------------------------------------------

    def init_db(self) -> None:
        """Create all tables defined on ``Base`` if they don't exist."""
        Base.metadata.create_all(self._engine)
        logger.info("Database tables created / verified.")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Provide a transactional session scope.

        Commits on clean exit, rolls back on exception, and always
        closes the session.

        Yields:
            A SQLAlchemy ``Session`` instance.
        """
        session: Session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Session rolled back due to error.")
            raise
        finally:
            session.close()

    @property
    def engine(self):
        """Expose the underlying engine (useful for raw SQL / alembic)."""
        return self._engine

    @staticmethod
    def _ensure_sqlite_parent(db_url: str) -> None:
        """Create the parent directory for file-backed SQLite databases."""
        if not db_url.startswith("sqlite:///") or ":memory:" in db_url:
            return

        raw_path = db_url.removeprefix("sqlite:///")
        db_path = Path(raw_path)
        if not db_path.is_absolute():
            db_path = PROJECT_ROOT / db_path

        db_path.parent.mkdir(parents=True, exist_ok=True)


# ===================================================================
# Module-level singleton
# ===================================================================

@lru_cache(maxsize=1)
def get_database(db_url: str | None = None) -> DatabaseManager:
    """Return (and cache) the default ``DatabaseManager`` singleton.

    The database URL is read from the application settings if not provided.

    Returns:
        A ready-to-use ``DatabaseManager`` instance.
    """
    return DatabaseManager(db_url)
