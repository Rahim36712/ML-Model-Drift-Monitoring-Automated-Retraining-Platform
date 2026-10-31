"""Portable paths and local management credentials shared by all processes."""

from functools import lru_cache
import os
from pathlib import Path
import secrets

ROOT = Path(__file__).resolve().parent.parent


def home() -> Path:
    default = (
        ROOT / "data" / "demo"
        if (ROOT / "pyproject.toml").exists()
        else Path.home() / ".driftwatch"
    )
    path = Path(os.environ.get("MLOPS_HOME", str(default))).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def sqlite_url(path: Path) -> str:
    return "sqlite:///" + path.resolve().as_posix()


@lru_cache(maxsize=1)
def management_token() -> str:
    if value := os.environ.get("MLOPS_TOKEN"):
        return value
    path = home() / ".management-token"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(secrets.token_urlsafe(32))
        if os.name != "nt":
            path.chmod(0o600)
    except FileExistsError:
        pass
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError("Management token file is empty; restart the launcher.")
    return value


def api_url() -> str:
    return os.environ.get(
        "MLOPS_API_URL", "http://127.0.0.1:" + os.environ.get("API_PORT", "8000")
    ).rstrip("/")
