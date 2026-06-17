"""Health check routes."""

from __future__ import annotations

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text

from src.data.database import DatabaseManager
from src.api.schemas import HealthResponse
from src.pipeline.deployer import ModelProvider

router = APIRouter(tags=["Health"])


def get_db_manager() -> DatabaseManager:
    """Dependency to get global DatabaseManager instance."""
    from src.data.database import get_database
    return get_database()


@router.get("/health", response_model=HealthResponse)
def health_check(db: DatabaseManager = Depends(get_db_manager)) -> dict:
    """Verify service and dependency health."""
    db_ok = False
    try:
        with db.get_session() as session:
            # Simple query check
            session.execute(text("SELECT 1"))
            db_ok = True
    except Exception:
        # Don't fail the entire response, just report db_connected=False
        pass

    _, _, version, _ = ModelProvider.get_active_model()
    version_str = str(version) if version is not None else None

    return {
        "status": "ok" if db_ok else "degraded",
        "timestamp": datetime.now(timezone.utc),
        "model_version": version_str,
        "db_connected": db_ok,
    }
