"""FastAPI Serving and Prediction API package.

Exports:
    create_app: Factory function to construct the FastAPI application instance.
"""

from __future__ import annotations

from src.api.app import create_app

__all__ = ["create_app"]
