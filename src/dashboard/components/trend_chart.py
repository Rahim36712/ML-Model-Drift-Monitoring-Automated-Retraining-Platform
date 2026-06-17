"""Trend chart wrapper and Plotly layout helpers.

Provides:
* ``create_trend_chart`` – dark-themed card housing a ``dcc.Graph``.
* ``get_dark_layout``    – reusable Plotly layout dict matching the
  dashboard colour palette.
* ``add_threshold_bands`` – overlay green / yellow / red zones on a
  figure to visualise warning & critical thresholds.
"""
from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html


# ---------------------------------------------------------------------------
# Chart card component
# ---------------------------------------------------------------------------
def create_trend_chart(
    chart_id: str,
    title: str,
) -> dbc.Card:
    """Return a dark-themed card wrapping a ``dcc.Graph``.

    Parameters
    ----------
    chart_id:
        The Dash component ID assigned to the inner ``dcc.Graph``.
    title:
        Text displayed in the card header.
    """

    card = dbc.Card(
        [
            dbc.CardHeader(
                html.Span(title, style={"fontWeight": 600}),
            ),
            dbc.CardBody(
                dcc.Graph(
                    id=chart_id,
                    config={
                        "displayModeBar": True,
                        "displaylogo": False,
                        "modeBarButtonsToRemove": [
                            "lasso2d",
                            "select2d",
                        ],
                    },
                    style={"height": "320px"},
                ),
            ),
        ],
        className="glass-card chart-card mb-3",
    )

    return card


# ---------------------------------------------------------------------------
# Plotly dark layout helper
# ---------------------------------------------------------------------------
def get_dark_layout(
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
) -> dict[str, Any]:
    """Return a Plotly ``layout`` dict styled for the dark dashboard.

    Colours, fonts, grid opacity, and margins are pre-configured so
    every chart shares a consistent look-and-feel.
    """

    return dict(
        title=dict(
            text=title,
            font=dict(size=14, color="#e2e8f0", family="Inter"),
            x=0.01,
            y=0.98,
        ),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            color="#e2e8f0",
            size=12,
        ),
        xaxis=dict(
            title=xaxis_title,
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.08)",
            tickfont=dict(color="#94a3b8"),
        ),
        yaxis=dict(
            title=yaxis_title,
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.08)",
            tickfont=dict(color="#94a3b8"),
        ),
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(
            font=dict(color="#94a3b8", size=11),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#1e293b",
            font_color="#e2e8f0",
            font_size=12,
            bordercolor="rgba(255,255,255,0.1)",
        ),
    )


# ---------------------------------------------------------------------------
# Threshold band helper
# ---------------------------------------------------------------------------
def add_threshold_bands(
    fig: go.Figure,
    warning_threshold: float,
    critical_threshold: float,
    y_max: float,
) -> None:
    """Add coloured rectangular zones to *fig* visualising thresholds.

    Three horizontal bands are drawn:

    * **Green zone** – 0 → *warning_threshold*
    * **Yellow zone** – *warning_threshold* → *critical_threshold*
    * **Red zone** – *critical_threshold* → *y_max*

    The bands are semi-transparent so underlying traces remain visible.

    Parameters
    ----------
    fig:
        The Plotly ``Figure`` to modify **in-place**.
    warning_threshold:
        The boundary between the green and yellow zones.
    critical_threshold:
        The boundary between the yellow and red zones.
    y_max:
        Upper limit of the red zone (typically the y-axis max).
    """

    bands: list[tuple[float, float, str]] = [
        (0, warning_threshold, "rgba(16,185,129,0.1)"),
        (warning_threshold, critical_threshold, "rgba(245,158,11,0.1)"),
        (critical_threshold, y_max, "rgba(239,68,68,0.1)"),
    ]

    for y0, y1, colour in bands:
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="y",
            x0=0,
            x1=1,
            y0=y0,
            y1=y1,
            fillcolor=colour,
            line=dict(width=0),
            layer="below",
        )
