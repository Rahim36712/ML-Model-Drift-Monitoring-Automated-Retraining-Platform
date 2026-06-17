"""Prediction-drift analysis dashboard layout."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from src.dashboard.components.metric_card import create_metric_card
from src.dashboard.components.trend_chart import create_trend_chart


_PAGE_BG = "transparent"
_TEXT_PRIMARY = "#e6edf3"
_TEXT_MUTED = "#91a4b7"


def layout() -> html.Div:
    """Return the Prediction Drift Analysis page layout."""
    return html.Div(
        style={"backgroundColor": _PAGE_BG, "minHeight": "100vh"},
        children=[
            html.H2(
                "Prediction Drift Analysis",
                style={"color": _TEXT_PRIMARY, "marginBottom": "0.25rem"},
            ),
            html.P(
                "Track shifts in model output distributions and confidence scores.",
                style={"color": _TEXT_MUTED, "marginBottom": "1.5rem"},
            ),
            dbc.Row(
                [
                    dbc.Col(create_metric_card("hellinger-dist", "Hellinger Distance", "H"), xs=12, sm=6, lg=4),
                    dbc.Col(create_metric_card("positive-rate", "Positive Rate", "RATE"), xs=12, sm=6, lg=4),
                    dbc.Col(create_metric_card("mean-confidence", "Mean Confidence", "CONF"), xs=12, sm=6, lg=4),
                ],
                className="g-3 mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        create_trend_chart(
                            "confidence-distribution-chart",
                            "Confidence Score Distribution",
                        ),
                        xs=12,
                        lg=6,
                    ),
                    dbc.Col(
                        create_trend_chart(
                            "hellinger-trend-chart",
                            "Hellinger Distance Trend",
                        ),
                        xs=12,
                        lg=6,
                    ),
                ],
                className="g-3 mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        create_trend_chart(
                            "positive-rate-trend-chart",
                            "Positive Prediction Rate Trend",
                        ),
                        xs=12,
                        lg=6,
                    ),
                    dbc.Col(
                        create_trend_chart("class-balance-chart", "Prediction Class Balance"),
                        xs=12,
                        lg=6,
                    ),
                ],
                className="g-3 mb-4",
            ),
        ],
    )
