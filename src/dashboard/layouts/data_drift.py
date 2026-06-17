"""Data-drift deep-dive dashboard layout."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from src.dashboard.components.metric_card import create_metric_card
from src.dashboard.components.trend_chart import create_trend_chart


_PAGE_BG = "transparent"
_TEXT_PRIMARY = "#e6edf3"
_TEXT_MUTED = "#91a4b7"


def layout() -> html.Div:
    """Return the Data Drift Analysis page layout."""
    return html.Div(
        style={"backgroundColor": _PAGE_BG, "minHeight": "100vh"},
        children=[
            html.H2(
                "Data Drift Analysis",
                style={"color": _TEXT_PRIMARY, "marginBottom": "0.25rem"},
            ),
            html.P(
                "Analyze feature-level drift using PSI and distributional metrics.",
                style={"color": _TEXT_MUTED, "marginBottom": "1.5rem"},
            ),
            dbc.Row(
                [
                    dbc.Col(create_metric_card("max-psi", "Max PSI", "PSI"), xs=12, sm=6),
                    dbc.Col(
                        create_metric_card("drifted-features-count", "Drifted Features", "WARN"),
                        xs=12,
                        sm=6,
                    ),
                ],
                className="g-3 mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(create_trend_chart("feature-psi-heatmap", "Feature PSI Heatmap"), xs=12, lg=8),
                    dbc.Col(create_trend_chart("top-drifted-bar", "Top Drifted Features"), xs=12, lg=4),
                ],
                className="g-3 mb-4",
            ),
            dbc.Row(
                style={"position": "relative", "zIndex": "100"},
                children=[
                    dbc.Col(
                        html.Div(
                            className="glass-card",
                            style={"padding": "1.25rem"},
                            children=[
                                html.H6(
                                    "Select Feature",
                                    style={"color": _TEXT_PRIMARY, "marginBottom": "0.5rem"},
                                ),
                                dcc.Dropdown(
                                    id="feature-selector",
                                    placeholder="Select feature...",
                                    clearable=True,
                                ),
                            ],
                        ),
                        xs=12,
                        lg=3,
                    ),
                    dbc.Col(
                        create_trend_chart(
                            "feature-distribution-chart",
                            "Feature Distribution Comparison",
                        ),
                        xs=12,
                        lg=9,
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
                                    "KL Divergence Table",
                                    style={"color": _TEXT_PRIMARY, "marginBottom": "1rem"},
                                ),
                                html.Div(id="kl-table"),
                            ],
                        ),
                        xs=12,
                    ),
                ],
                className="g-3 mb-4",
            ),
        ],
    )
