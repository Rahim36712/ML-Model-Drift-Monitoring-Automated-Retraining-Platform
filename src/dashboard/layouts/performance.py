"""Model performance dashboard layout."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from src.dashboard.components.metric_card import create_metric_card
from src.dashboard.components.trend_chart import create_trend_chart


_PAGE_BG = "transparent"
_TEXT_PRIMARY = "#e6edf3"
_TEXT_MUTED = "#91a4b7"


def layout() -> html.Div:
    """Return the Model Performance page layout."""
    return html.Div(
        style={"backgroundColor": _PAGE_BG, "minHeight": "100vh"},
        children=[
            html.H2(
                "Model Performance",
                style={"color": _TEXT_PRIMARY, "marginBottom": "0.25rem"},
            ),
            html.P(
                "Track classification metrics and ground-truth coverage over time.",
                style={"color": _TEXT_MUTED, "marginBottom": "1.5rem"},
            ),
            dbc.Row(
                [
                    dbc.Col(create_metric_card("perf-accuracy", "Accuracy", "ACC"), xs=12, sm=6, lg=3),
                    dbc.Col(create_metric_card("perf-f1", "F1 Score", "F1"), xs=12, sm=6, lg=3),
                    dbc.Col(create_metric_card("perf-precision", "Precision", "PREC"), xs=12, sm=6, lg=3),
                    dbc.Col(create_metric_card("perf-recall", "Recall", "REC"), xs=12, sm=6, lg=3),
                ],
                className="g-3 mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(create_trend_chart("metrics-trend-chart", "Performance Metrics Trend"), xs=12, lg=8),
                    dbc.Col(create_trend_chart("confusion-matrix-chart", "Confusion Matrix"), xs=12, lg=4),
                ],
                className="g-3 mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(create_trend_chart("roc-curve-chart", "ROC Curve"), xs=12, lg=6),
                    dbc.Col(
                        html.Div(
                            className="glass-card",
                            style={"padding": "1.25rem"},
                            children=[
                                html.H5(
                                    "Ground Truth Coverage",
                                    style={"color": _TEXT_PRIMARY, "marginBottom": "1rem"},
                                ),
                                html.Div(id="gt-coverage-display"),
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
