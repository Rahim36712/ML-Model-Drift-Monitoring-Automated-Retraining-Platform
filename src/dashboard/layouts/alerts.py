"""Alert center dashboard layout."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from src.dashboard.components.metric_card import create_metric_card
from src.dashboard.components.trend_chart import create_trend_chart


_PAGE_BG = "transparent"
_TEXT_PRIMARY = "#e6edf3"
_TEXT_MUTED = "#91a4b7"
_DROPDOWN_STYLE: dict[str, str] = {}


def layout() -> html.Div:
    """Return the Alert Center page layout."""
    return html.Div(
        style={"backgroundColor": _PAGE_BG, "minHeight": "100vh"},
        children=[
            html.H2(
                "Alert Center",
                style={"color": _TEXT_PRIMARY, "marginBottom": "0.25rem"},
            ),
            html.P(
                "Review, filter, and acknowledge drift and performance alerts.",
                style={"color": _TEXT_MUTED, "marginBottom": "1.5rem"},
            ),
            dbc.Row(
                style={"position": "relative", "zIndex": "100"},
                children=[
                    dbc.Col(
                        html.Div(
                            className="glass-card",
                            style={"padding": "1rem"},
                            children=[
                                html.H6(
                                    "Severity",
                                    style={"color": _TEXT_PRIMARY, "marginBottom": "0.5rem"},
                                ),
                                dcc.Dropdown(
                                    id="alert-severity-filter",
                                    options=[
                                        {"label": "All", "value": "All"},
                                        {"label": "WARNING", "value": "WARNING"},
                                        {"label": "CRITICAL", "value": "CRITICAL"},
                                        {"label": "RESOLVED", "value": "RESOLVED"},
                                    ],
                                    value="All",
                                    clearable=False,
                                    style=_DROPDOWN_STYLE,
                                ),
                            ],
                        ),
                        xs=12,
                        sm=6,
                        lg=3,
                    ),
                    dbc.Col(
                        html.Div(
                            className="glass-card",
                            style={"padding": "1rem"},
                            children=[
                                html.H6(
                                    "Drift Type",
                                    style={"color": _TEXT_PRIMARY, "marginBottom": "0.5rem"},
                                ),
                                dcc.Dropdown(
                                    id="alert-type-filter",
                                    options=[
                                        {"label": "All", "value": "All"},
                                        {"label": "Data", "value": "data"},
                                        {"label": "Prediction", "value": "prediction"},
                                        {"label": "Concept", "value": "concept"},
                                        {"label": "Retraining", "value": "retraining"},
                                    ],
                                    value="All",
                                    clearable=False,
                                    style=_DROPDOWN_STYLE,
                                ),
                            ],
                        ),
                        xs=12,
                        sm=6,
                        lg=3,
                    ),
                ],
                className="g-3 mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(create_metric_card("total-alerts", "Total Alerts", "ALL"), xs=12, sm=6, lg=4),
                    dbc.Col(
                        create_metric_card(
                            "active-alerts-count",
                            "Active (Unacknowledged)",
                            "LIVE",
                        ),
                        xs=12,
                        sm=6,
                        lg=4,
                    ),
                    dbc.Col(create_metric_card("resolved-alerts-count", "Resolved", "OK"), xs=12, sm=6, lg=4),
                ],
                className="g-3 mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(create_trend_chart("alert-frequency-chart", "Alert Frequency"), xs=12, lg=8),
                    dbc.Col(
                        html.Div(
                            className="glass-card",
                            style={"padding": "1.25rem"},
                            children=[
                                html.H5(
                                    "Alert Summary",
                                    style={"color": _TEXT_PRIMARY, "marginBottom": "1rem"},
                                ),
                                html.Div(id="alert-summary-content"),
                            ],
                        ),
                        xs=12,
                        lg=4,
                    ),
                ],
                className="g-3 mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            className="glass-card",
                            style={"padding": "1.25rem"},
                            children=[
                                html.H5(
                                    "Alert History",
                                    style={"color": _TEXT_PRIMARY, "marginBottom": "1rem"},
                                ),
                                html.Div(id="alert-history-table"),
                            ],
                        ),
                        xs=12,
                    ),
                ],
                className="g-3 mb-4",
            ),
        ],
    )
