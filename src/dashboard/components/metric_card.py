"""Reusable KPI card component for the monitoring dashboard."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html


def create_metric_card(
    card_id: str,
    title: str,
    icon: str,
) -> dbc.Card:
    """Return a compact KPI card with dynamic value, delta, and status dot."""

    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(icon, className="metric-icon"),
                html.Div(title, className="metric-title"),
                html.Div("-", id=f"{card_id}-value", className="metric-value"),
                html.Div(
                    [
                        html.Span(
                            "",
                            id=f"{card_id}-delta",
                            className="metric-delta-up",
                        ),
                        html.Span(
                            "",
                            id=f"{card_id}-status",
                            className="status-dot status-dot-gray ms-2",
                        ),
                    ],
                    className="d-flex align-items-center mt-1",
                ),
            ],
            style={"padding": "1.15rem"},
        ),
        className="glass-card h-100",
    )
