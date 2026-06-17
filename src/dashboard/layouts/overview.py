"""System overview dashboard layout."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from src.dashboard.components.metric_card import create_metric_card
from src.dashboard.components.trend_chart import create_trend_chart


_PAGE_BG = "transparent"
_TEXT_PRIMARY = "#e6edf3"
_TEXT_MUTED = "#91a4b7"


def layout() -> html.Div:
    """Return the System Overview page layout."""
    return html.Div(
        style={"backgroundColor": _PAGE_BG, "minHeight": "100vh"},
        children=[
            html.H2(
                "System Overview",
                style={"color": _TEXT_PRIMARY, "marginBottom": "0.25rem"},
            ),
            html.P(
                "Real-time monitoring of model health, data drift, and prediction quality.",
                style={"color": _TEXT_MUTED, "marginBottom": "1.5rem"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        create_metric_card("total-predictions", "Total Predictions", "API"),
                        xs=12,
                        sm=6,
                        lg=3,
                    ),
                    dbc.Col(
                        create_metric_card("current-psi", "Data Drift (PSI)", "PSI"),
                        xs=12,
                        sm=6,
                        lg=3,
                    ),
                    dbc.Col(
                        create_metric_card("current-f1", "F1 Score", "F1"),
                        xs=12,
                        sm=6,
                        lg=3,
                    ),
                    dbc.Col(
                        create_metric_card("active-alerts", "Active Alerts", "ALERT"),
                        xs=12,
                        sm=6,
                        lg=3,
                    ),
                ],
                className="g-3 mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(create_trend_chart("psi-trend-chart", "PSI Trend Over Time"), xs=12, lg=6),
                    dbc.Col(create_trend_chart("f1-trend-chart", "F1 Score Trend"), xs=12, lg=6),
                ],
                className="g-3 mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        create_trend_chart("prediction-volume-chart", "Prediction Volume"),
                        xs=12,
                        lg=6,
                    ),
                    dbc.Col(
                        html.Div(
                            className="glass-card",
                            style={"padding": "1.25rem"},
                            children=[
                                html.H5(
                                    "Recent Alerts",
                                    style={"color": _TEXT_PRIMARY, "marginBottom": "1rem"},
                                ),
                                html.Div(id="recent-alerts-table"),
                            ],
                        ),
                        xs=12,
                        lg=6,
                    ),
                ],
                className="g-3 mb-4",
            ),
        ],
    )
