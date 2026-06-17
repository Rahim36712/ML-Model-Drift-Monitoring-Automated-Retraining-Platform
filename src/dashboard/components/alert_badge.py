"""Alert badge and status indicator components.

Provides:
* ``create_alert_badge``     – a coloured pill badge for alert severity.
* ``create_status_indicator`` – a small glowing dot for health status.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html


# ── Severity → Bootstrap colour mapping ───────────────────────────────────
_SEVERITY_COLOUR: dict[str, str] = {
    "CRITICAL": "danger",
    "WARNING": "warning",
    "RESOLVED": "success",
    "INFO": "info",
}

# ── Status → dot CSS class mapping ────────────────────────────────────────
_STATUS_DOT: dict[str, str] = {
    "healthy": "status-dot status-dot-green",
    "none": "status-dot status-dot-green",
    "warning": "status-dot status-dot-yellow",
    "critical": "status-dot status-dot-red",
    "unknown": "status-dot status-dot-gray",
}


def create_alert_badge(severity: str, text: str) -> dbc.Badge:
    """Return a pill-shaped badge coloured by *severity*.

    Parameters
    ----------
    severity:
        One of ``CRITICAL``, ``WARNING``, ``RESOLVED``, or ``INFO``
        (case-insensitive).
    text:
        The human-readable label rendered inside the badge.
    """

    colour = _SEVERITY_COLOUR.get(severity.upper(), "secondary")

    return dbc.Badge(
        text,
        color=colour,
        pill=True,
        style={"fontSize": "0.75rem"},
    )


def create_status_indicator(status: str) -> html.Span:
    """Return a small coloured dot representing *status*.

    Parameters
    ----------
    status:
        One of ``healthy``, ``none``, ``warning``, ``critical``, or
        ``unknown`` (case-insensitive).  Unrecognised values fall back
        to a neutral gray dot.
    """

    dot_class = _STATUS_DOT.get(status.lower(), "status-dot status-dot-gray")

    return html.Span(className=dot_class)
