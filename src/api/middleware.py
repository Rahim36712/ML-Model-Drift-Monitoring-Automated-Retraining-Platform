"""Middleware stack for FastAPI application.

Includes timing tracking, request logging, and global exception handling.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class TimingMiddleware(BaseHTTPMiddleware):
    """Measures request process time and attaches a header."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        response: Response = await call_next(request)
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = f"{process_time * 1000:.2f}ms"
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Structured console logger for incoming HTTP requests."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        method = request.method
        path = request.url.path
        
        # Exclude dashboard polling endpoints from logging to prevent cluttering stdout
        silent_paths = ["/health", "/monitoring/health", "/predictions/stats"]
        is_silent = any(p in path for p in silent_paths)

        if not is_silent:
            logger.info("--> %s %s", method, path)

        start_time = time.perf_counter()
        try:
            response = await call_next(request)
            process_time = time.perf_counter() - start_time
            if not is_silent:
                logger.info(
                    "<-- %s %s | status=%d | latency=%.2fms",
                    method,
                    path,
                    response.status_code,
                    process_time * 1000,
                )
            return response
        except Exception as e:
            process_time = time.perf_counter() - start_time
            logger.exception(
                "<-- ERROR %s %s | status=500 | latency=%.2fms | error=%s",
                method,
                path,
                process_time * 1000,
                str(e),
            )
            raise


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catches all unhandled exceptions and formats a clean JSON response."""
    logger.exception("Global handler caught exception during request %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred.",
            "error_type": exc.__class__.__name__,
            "message": str(exc),
        },
    )
