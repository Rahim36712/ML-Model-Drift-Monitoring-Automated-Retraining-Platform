"""Model registry dashboard layout."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from src.dashboard.components.trend_chart import create_trend_chart


_PAGE_BG = "transparent"
_TEXT_PRIMARY = "#e6edf3"
_TEXT_MUTED = "#91a4b7"


def layout() -> html.Div:
    """Return the Model Registry page layout."""
    return html.Div(
        style={"backgroundColor": _PAGE_BG, "minHeight": "100vh"},
        children=[
            html.H2(
                "Model Registry",
                style={"color": _TEXT_PRIMARY, "marginBottom": "0.25rem"},
            ),
            html.P(
                "Manage model versions, compare metrics, and review retraining history.",
                style={"color": _TEXT_MUTED, "marginBottom": "1.5rem"},
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            id="production-model-card",
                            className="glass-card",
                            style={
                                "padding": "1.5rem",
                                "border": "1px solid rgba(34, 211, 238, 0.45)",
                            },
                            children=[
                                html.H5(
                                    "Production Model",
                                    style={"color": _TEXT_PRIMARY, "marginBottom": "1rem"},
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Span(
                                                    "Version",
                                                    style={"color": _TEXT_MUTED, "fontSize": "0.85rem"},
                                                ),
                                                html.Div(
                                                    id="prod-model-version",
                                                    style={
                                                        "color": _TEXT_PRIMARY,
                                                        "fontSize": "1.25rem",
                                                        "fontWeight": "600",
                                                    },
                                                    children="-",
                                                ),
                                            ],
                                            xs=6,
                                            lg=3,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Span(
                                                    "F1 Score",
                                                    style={"color": _TEXT_MUTED, "fontSize": "0.85rem"},
                                                ),
                                                html.Div(
                                                    id="prod-model-f1",
                                                    style={
                                                        "color": _TEXT_PRIMARY,
                                                        "fontSize": "1.25rem",
                                                        "fontWeight": "600",
                                                    },
                                                    children="-",
                                                ),
                                            ],
                                            xs=6,
                                            lg=3,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Span(
                                                    "Accuracy",
                                                    style={"color": _TEXT_MUTED, "fontSize": "0.85rem"},
                                                ),
                                                html.Div(
                                                    id="prod-model-accuracy",
                                                    style={
                                                        "color": _TEXT_PRIMARY,
                                                        "fontSize": "1.25rem",
                                                        "fontWeight": "600",
                                                    },
                                                    children="-",
                                                ),
                                            ],
                                            xs=6,
                                            lg=3,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Span(
                                                    "Deployed",
                                                    style={"color": _TEXT_MUTED, "fontSize": "0.85rem"},
                                                ),
                                                html.Div(
                                                    id="prod-model-deployed",
                                                    style={
                                                        "color": _TEXT_PRIMARY,
                                                        "fontSize": "1.25rem",
                                                        "fontWeight": "600",
                                                    },
                                                    children="-",
                                                ),
                                            ],
                                            xs=6,
                                            lg=3,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        xs=12,
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
                                    "Model Versions",
                                    style={"color": _TEXT_PRIMARY, "marginBottom": "1rem"},
                                ),
                                html.Div(id="versions-table"),
                            ],
                        ),
                        xs=12,
                    ),
                ],
                className="g-3 mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        create_trend_chart(
                            "version-metrics-chart",
                            "Version Metrics Comparison",
                        ),
                        xs=12,
                        lg=6,
                    ),
                    dbc.Col(
                        html.Div(
                            className="glass-card",
                            style={"padding": "1.25rem"},
                            children=[
                                html.H5(
                                    "Retraining History",
                                    style={"color": _TEXT_PRIMARY, "marginBottom": "1rem"},
                                ),
                                html.Div(id="retraining-history-table"),
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
