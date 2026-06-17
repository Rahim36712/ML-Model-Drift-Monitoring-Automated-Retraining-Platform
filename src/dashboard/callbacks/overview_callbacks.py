"""Overview page callbacks — wires live data into the overview dashboard.

Registered against the Dash app instance by the main app factory.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import plotly.graph_objects as go
from dash import Input, Output, callback

from src.config.settings import get_settings
from src.data.database import get_database, DriftResult, Alert
from src.data.logger import PredictionLogger
from src.dashboard.components.trend_chart import get_dark_layout, add_threshold_bands
from src.dashboard.components.alert_badge import create_alert_badge

from sqlalchemy import desc, func

logger = logging.getLogger(__name__)


def _safe_db():
    """Return a DatabaseManager; never crash if settings are missing."""
    try:
        settings = get_settings()
        return get_database(settings.database.url)
    except Exception:
        return get_database()


# ── Metric cards ──────────────────────────────────────────────────────

@callback(
    [
        Output("total-predictions-value", "children"),
        Output("total-predictions-delta", "children"),
        Output("total-predictions-status", "style"),
    ],
    Input("interval-refresh", "n_intervals"),
)
def update_total_predictions(_n):
    try:
        db = _safe_db()
        pl = PredictionLogger(db)
        stats = pl.get_prediction_stats()
        total = stats.get("total_count", 0)
        throughput = stats.get("throughput_per_min", 0)
        delta_text = f"↑ {throughput:.1f}/min" if throughput > 0 else "—"
        return (
            f"{total:,}",
            delta_text,
            {"color": "#10b981", "width": "8px", "height": "8px", "borderRadius": "50%", "backgroundColor": "#10b981", "display": "inline-block"},
        )
    except Exception:
        return "—", "—", {"backgroundColor": "#6b7280", "width": "8px", "height": "8px", "borderRadius": "50%", "display": "inline-block"}


@callback(
    [
        Output("current-psi-value", "children"),
        Output("current-psi-delta", "children"),
        Output("current-psi-status", "style"),
    ],
    Input("interval-refresh", "n_intervals"),
)
def update_psi_card(_n):
    try:
        db = _safe_db()
        with db.get_session() as session:
            latest = (
                session.query(DriftResult)
                .filter(DriftResult.drift_type == "data", DriftResult.metric_name == "psi")
                .order_by(desc(DriftResult.timestamp))
                .first()
            )
        if latest:
            val = latest.metric_value
            color = "#10b981" if val < 0.10 else "#f59e0b" if val < 0.25 else "#ef4444"
            severity = "Stable" if val < 0.10 else "Warning" if val < 0.25 else "Critical"
            return (
                f"{val:.4f}",
                severity,
                {"backgroundColor": color, "width": "8px", "height": "8px", "borderRadius": "50%", "display": "inline-block"},
            )
        return "N/A", "No data", {"backgroundColor": "#6b7280", "width": "8px", "height": "8px", "borderRadius": "50%", "display": "inline-block"}
    except Exception:
        return "—", "—", {"backgroundColor": "#6b7280", "width": "8px", "height": "8px", "borderRadius": "50%", "display": "inline-block"}


@callback(
    [
        Output("current-f1-value", "children"),
        Output("current-f1-delta", "children"),
        Output("current-f1-status", "style"),
    ],
    Input("interval-refresh", "n_intervals"),
)
def update_f1_card(_n):
    try:
        db = _safe_db()
        with db.get_session() as session:
            latest = (
                session.query(DriftResult)
                .filter(DriftResult.drift_type == "concept")
                .order_by(desc(DriftResult.timestamp))
                .first()
            )
        if latest and latest.details_json:
            details = json.loads(latest.details_json)
            current_metrics = details.get("current_metrics", {})
            baseline_metrics = details.get("baseline_metrics", {})
            f1 = current_metrics.get("f1", 0)
            base_f1 = baseline_metrics.get("f1", 0)
            delta = f1 - base_f1
            delta_text = f"{'↑' if delta >= 0 else '↓'} {abs(delta):.4f}"
            color = "#10b981" if delta >= -0.03 else "#f59e0b" if delta >= -0.05 else "#ef4444"
            return (
                f"{f1:.4f}",
                delta_text,
                {"backgroundColor": color, "width": "8px", "height": "8px", "borderRadius": "50%", "display": "inline-block"},
            )
        return "N/A", "No data", {"backgroundColor": "#6b7280", "width": "8px", "height": "8px", "borderRadius": "50%", "display": "inline-block"}
    except Exception:
        return "—", "—", {"backgroundColor": "#6b7280", "width": "8px", "height": "8px", "borderRadius": "50%", "display": "inline-block"}


@callback(
    [
        Output("active-alerts-value", "children"),
        Output("active-alerts-delta", "children"),
        Output("active-alerts-status", "style"),
    ],
    Input("interval-refresh", "n_intervals"),
)
def update_alerts_card(_n):
    try:
        db = _safe_db()
        from src.alerting.alert_manager import AlertManager
        am = AlertManager(db)
        stats = am.get_alert_stats()
        active = stats.get("active", 0)
        total = stats.get("total", 0)
        color = "#10b981" if active == 0 else "#f59e0b" if active < 3 else "#ef4444"
        return (
            str(active),
            f"of {total} total",
            {"backgroundColor": color, "width": "8px", "height": "8px", "borderRadius": "50%", "display": "inline-block"},
        )
    except Exception:
        return "—", "—", {"backgroundColor": "#6b7280", "width": "8px", "height": "8px", "borderRadius": "50%", "display": "inline-block"}


# ── PSI Trend Chart ───────────────────────────────────────────────────

@callback(
    Output("psi-trend-chart", "figure"),
    Input("interval-refresh", "n_intervals"),
)
def update_psi_trend(_n):
    fig = go.Figure()
    layout = get_dark_layout("PSI Over Time", "Time", "PSI Value")
    try:
        db = _safe_db()
        with db.get_session() as session:
            rows = (
                session.query(DriftResult)
                .filter(DriftResult.drift_type == "data", DriftResult.metric_name == "psi")
                .order_by(DriftResult.timestamp)
                .limit(100)
                .all()
            )
        if rows:
            timestamps = [r.timestamp for r in rows]
            values = [r.metric_value for r in rows]
            fig.add_trace(go.Scatter(
                x=timestamps, y=values, mode="lines+markers",
                name="PSI", line=dict(color="#3b82f6", width=2),
                marker=dict(size=4),
            ))
            # Add threshold lines
            fig.add_hline(y=0.10, line_dash="dash", line_color="#f59e0b",
                          annotation_text="Warning (0.10)", annotation_position="top right")
            fig.add_hline(y=0.25, line_dash="dash", line_color="#ef4444",
                          annotation_text="Critical (0.25)", annotation_position="top right")
            y_max = max(values) * 1.3 if max(values) > 0.25 else 0.35
            add_threshold_bands(fig, 0.10, 0.25, y_max)
    except Exception:
        logger.exception("Failed to load PSI trend data.")

    fig.update_layout(**layout)
    return fig


# ── F1 Trend Chart ───────────────────────────────────────────────────

@callback(
    Output("f1-trend-chart", "figure"),
    Input("interval-refresh", "n_intervals"),
)
def update_f1_trend(_n):
    fig = go.Figure()
    layout = get_dark_layout("F1 Score Over Time", "Time", "F1 Score")
    try:
        db = _safe_db()
        with db.get_session() as session:
            rows = (
                session.query(DriftResult)
                .filter(DriftResult.drift_type == "concept")
                .order_by(DriftResult.timestamp)
                .limit(100)
                .all()
            )
        if rows:
            timestamps = []
            f1_values = []
            for r in rows:
                if r.details_json:
                    details = json.loads(r.details_json)
                    current = details.get("current_metrics", {})
                    if "f1" in current:
                        timestamps.append(r.timestamp)
                        f1_values.append(current["f1"])
            if timestamps:
                fig.add_trace(go.Scatter(
                    x=timestamps, y=f1_values, mode="lines+markers",
                    name="F1 Score", line=dict(color="#10b981", width=2),
                    marker=dict(size=4),
                    fill="tozeroy", fillcolor="rgba(16,185,129,0.1)",
                ))
    except Exception:
        logger.exception("Failed to load F1 trend data.")

    fig.update_layout(**layout)
    return fig


# ── Prediction Volume Chart ──────────────────────────────────────────

@callback(
    Output("prediction-volume-chart", "figure"),
    Input("interval-refresh", "n_intervals"),
)
def update_prediction_volume(_n):
    fig = go.Figure()
    layout = get_dark_layout("Prediction Volume", "Time", "Count")
    try:
        db = _safe_db()
        from src.data.database import Prediction
        with db.get_session() as session:
            rows = (
                session.query(
                    func.strftime("%Y-%m-%d %H:%M", Prediction.timestamp).label("minute"),
                    func.count(Prediction.id).label("cnt"),
                )
                .group_by("minute")
                .order_by(desc("minute"))
                .limit(60)
                .all()
            )
        if rows:
            rows = list(reversed(rows))
            minutes = [r[0] for r in rows]
            counts = [r[1] for r in rows]
            fig.add_trace(go.Bar(
                x=minutes, y=counts,
                marker_color="#3b82f6",
                opacity=0.8,
                name="Predictions",
            ))
    except Exception:
        logger.exception("Failed to load prediction volume data.")

    fig.update_layout(**layout)
    return fig


# ── Recent Alerts Table ──────────────────────────────────────────────

@callback(
    Output("recent-alerts-table", "children"),
    Input("interval-refresh", "n_intervals"),
)
def update_recent_alerts(_n):
    import dash_bootstrap_components as dbc
    from dash import html
    try:
        db = _safe_db()
        from src.alerting.alert_manager import AlertManager
        am = AlertManager(db)
        alerts = am.get_alert_history(limit=10)
        if not alerts:
            return html.P("No alerts recorded yet.", style={"color": "#94a3b8", "textAlign": "center", "padding": "20px"})

        rows = []
        for a in alerts:
            badge = create_alert_badge(a.severity, a.severity)
            rows.append(html.Tr([
                html.Td(badge),
                html.Td(a.drift_type, style={"color": "#e2e8f0"}),
                html.Td(a.message[:80] + "..." if len(a.message) > 80 else a.message, style={"color": "#94a3b8", "fontSize": "0.85rem"}),
                html.Td(a.timestamp.strftime("%H:%M:%S") if a.timestamp else "—", style={"color": "#94a3b8", "fontSize": "0.85rem"}),
            ], style={"borderBottom": "1px solid rgba(255,255,255,0.05)"}))

        table = html.Table([
            html.Thead(html.Tr([
                html.Th("Severity", style={"color": "#94a3b8", "fontWeight": "600", "padding": "8px"}),
                html.Th("Type", style={"color": "#94a3b8", "fontWeight": "600", "padding": "8px"}),
                html.Th("Message", style={"color": "#94a3b8", "fontWeight": "600", "padding": "8px"}),
                html.Th("Time", style={"color": "#94a3b8", "fontWeight": "600", "padding": "8px"}),
            ])),
            html.Tbody(rows),
        ], style={"width": "100%", "borderCollapse": "collapse"})
        return table
    except Exception:
        return html.P("Error loading alerts.", style={"color": "#ef4444"})


# ── Health Badge + Last Updated ──────────────────────────────────────

@callback(
    [
        Output("health-badge", "className"),
        Output("health-badge-text", "children"),
        Output("health-badge-text", "color"),
        Output("last-updated", "children"),
    ],
    Input("interval-refresh", "n_intervals"),
)
def update_health_badge(_n):
    now_str = datetime.now().strftime("%H:%M:%S")
    try:
        db = _safe_db()
        with db.get_session() as session:
            latest = (
                session.query(DriftResult)
                .order_by(desc(DriftResult.timestamp))
                .first()
            )
        if latest:
            if latest.is_breached:
                # If breached, determine warning vs critical
                # If there are any critical breaches, make it red (danger), otherwise yellow (warning)
                # But here we can just use warning for any drift. Let's check if severity is critical.
                color = "warning"
                dot_class = "status-dot status-dot-yellow"
                text = "Drift Warning"
                
                # Check if there's any critical alert active
                from src.alerting.alert_manager import AlertManager
                am = AlertManager(db)
                active_alerts = am.get_active_alerts()
                if any(a.severity == "CRITICAL" for a in active_alerts):
                    color = "danger"
                    dot_class = "status-dot status-dot-red"
                    text = "Drift Critical"
                    
                return dot_class, text, color, f"{now_str}"
            return "status-dot status-dot-green", "Healthy", "success", f"{now_str}"
        return "status-dot status-dot-gray", "No Data", "secondary", f"{now_str}"
    except Exception:
        return "status-dot status-dot-gray", "Unknown", "secondary", f"{now_str}"
