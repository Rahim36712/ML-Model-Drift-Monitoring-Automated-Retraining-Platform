"""Configuration management for the MLOps Drift Monitor platform.

Provides Pydantic-based settings loaded from YAML config files with
environment variable overrides. Use ``get_settings()`` as the single
entry-point; it returns a cached singleton.
"""

from src.config.settings import Settings, get_settings

__all__: list[str] = [
    "Settings",
    "get_settings",
]
