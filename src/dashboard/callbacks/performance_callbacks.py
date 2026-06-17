"""Performance and model registry page callbacks.

Covers the Performance page (accuracy/F1/precision/recall cards, metrics trend,
confusion matrix) and the Model Registry page (production model card, version
table, retraining history).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import plotly.graph_objects as go
from dash import Input, Output, callback, html
import dash_bootstrap_components as dbc

from src.config.settings import get_settings
from src.data.database import get_database, DriftResult, ModelVersion, RetrainingEvent, Prediction
from src.dashboard.components.trend_chart import get_dark_layout
from src.dashboard.components.alert_badge import create_alert_badge

from sqlalchemy import desc, func

logger = logging.getLogger(__name__)


def _safe_db():
    try:
        settings = get_settings()
        return get_database(settings.database.url)
    except Exception:
        return get_database()


_dot_style = lambda color: {"backgroundColor": color, "width": "8px", "height": "8px", "borderRadius": "50%", "display": "inline-block"}


# ═══════════════════════════════════════════════════════════════════════
# PERFORMANCE PAGE
# ═══════════════════════════════════════════════════════════════════════

@callback(
    [Output("perf-accuracy-value", "children"), Output("perf-accuracy-delta", "children"), Output("perf-accuracy-status", "style")],
    Input("interval-refresh", "n_intervals"),
)
def update_perf_accuracy(_n):
    return _get_concept_metric("accuracy")


@callback(
    [Output("perf-f1-value", "children"), Output("perf-f1-delta", "children"), Output("perf-f1-status", "style")],
    Input("interval-refresh", "n_intervals"),
)
def update_perf_f1(_n):
    return _get_concept_metric("f1")


@callback(
    [Output("perf-precision-value", "children"), Output("perf-precision-delta", "children"), Output("perf-precision-status", "style")],
    Input("interval-refresh", "n_intervals"),
)
def update_perf_precision(_n):
    return _get_concept_metric("precision")


@callback(
    [Output("perf-recall-value", "children"), Output("perf-recall-delta", "children"), Output("perf-recall-status", "style")],
    Input("interval-refresh", "n_intervals"),
)
def update_perf_recall(_n):
    return _get_concept_metric("recall")


def _get_concept_metric(metric_name: str):
    """Shared helper returning (value_str, delta_str, dot_style) for a concept drift metric."""
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
            current = details.get("current_metrics", {})
            baseline = details.get("baseline_metrics", {})
            val = current.get(metric_name, 0)
            base_val = baseline.get(metric_name, 0)
            delta = val - base_val
            delta_text = f"{'↑' if delta >= 0 else '↓'} {abs(delta):.4f}"
            color = "#10b981" if delta >= -0.03 else "#f59e0b" if delta >= -0.05 else "#ef4444"
            return f"{val:.4f}", delta_text, _dot_style(color)
        return "N/A", "No data", _dot_style("#6b7280")
    except Exception:
        return "—", "—", _dot_style("#6b7280")


@callback(
    Output("metrics-trend-chart", "figure"),
    Input("interval-refresh", "n_intervals"),
)
def update_metrics_trend(_n):
    fig = go.Figure()
    layout = get_dark_layout("Performance Metrics Over Time", "Time", "Score")
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
            metric_series = {"accuracy": [], "f1": [], "precision": [], "recall": []}
            timestamps = []
            for r in rows:
                if r.details_json:
                    details = json.loads(r.details_json)
                    current = details.get("current_metrics", {})
                    if current:
                        timestamps.append(r.timestamp)
                        for m in metric_series:
                            metric_series[m].append(current.get(m, 0))

            colors = {"accuracy": "#3b82f6", "f1": "#10b981", "precision": "#f59e0b", "recall": "#8b5cf6"}
            for name, values in metric_series.items():
                if values:
                    fig.add_trace(go.Scatter(
                        x=timestamps[:len(values)], y=values,
                        mode="lines+markers", name=name.capitalize(),
                        line=dict(color=colors.get(name, "#e2e8f0"), width=2),
                        marker=dict(size=3),
                    ))
    except Exception:
        logger.exception("Failed to load metrics trend.")

    fig.update_layout(**layout)
    return fig


@callback(
    Output("confusion-matrix-chart", "figure"),
    Input("interval-refresh", "n_intervals"),
)
def update_confusion_matrix(_n):
    fig = go.Figure()
    layout = get_dark_layout("Confusion Matrix", "Predicted", "Actual")
    try:
        db = _safe_db()
        with db.get_session() as session:
            preds = (
                session.query(Prediction)
                .filter(Prediction.true_label.isnot(None))
                .order_by(desc(Prediction.timestamp))
                .limit(500)
                .all()
            )
        if preds:
            from sklearn.metrics import confusion_matrix
            import numpy as np
            y_true = np.array([p.true_label for p in preds])
            y_pred = np.array([p.predicted_label for p in preds])
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            labels = ["Legitimate", "Fraud"]
            fig.add_trace(go.Heatmap(
                z=cm, x=labels, y=labels,
                colorscale=[[0, "#0f1729"], [1, "#3b82f6"]],
                text=cm, texttemplate="%{text}", textfont=dict(size=16, color="#e2e8f0"),
                hoverongaps=False,
                showscale=False,
            ))
    except Exception:
        logger.exception("Failed to build confusion matrix.")

    fig.update_layout(**layout)
    return fig


@callback(
    Output("roc-curve-chart", "figure"),
    Input("interval-refresh", "n_intervals"),
)
def update_roc_curve(_n):
    fig = go.Figure()
    layout = get_dark_layout("ROC Curve", "False Positive Rate", "True Positive Rate")
    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="#6b7280", dash="dash"), name="Random", showlegend=False,
    ))
    try:
        db = _safe_db()
        with db.get_session() as session:
            preds = (
                session.query(Prediction)
                .filter(Prediction.true_label.isnot(None))
                .order_by(desc(Prediction.timestamp))
                .limit(500)
                .all()
            )
        if preds:
            from sklearn.metrics import roc_curve, auc
            import numpy as np
            y_true = np.array([p.true_label for p in preds])
            y_scores = np.array([p.confidence for p in preds])
            if len(set(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                auc_val = auc(fpr, tpr)
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"Model (AUC={auc_val:.3f})",
                    line=dict(color="#3b82f6", width=2),
                    fill="tozeroy", fillcolor="rgba(59,130,246,0.1)",
                ))
    except Exception:
        logger.exception("Failed to build ROC curve.")

    fig.update_layout(**layout)
    return fig


@callback(
    Output("gt-coverage-display", "children"),
    Input("interval-refresh", "n_intervals"),
)
def update_gt_coverage(_n):
    try:
        db = _safe_db()
        with db.get_session() as session:
            total = session.query(func.count(Prediction.id)).scalar() or 0
            with_gt = session.query(func.count(Prediction.id)).filter(Prediction.true_label.isnot(None)).scalar() or 0

        pct = (with_gt / total * 100) if total > 0 else 0
        color = "#10b981" if pct > 50 else "#f59e0b" if pct > 20 else "#ef4444"

        return html.Div([
            html.Div(
                f"{pct:.1f}%",
                style={"fontSize": "3rem", "fontWeight": "700", "color": color, "textAlign": "center"},
            ),
            html.P(
                f"{with_gt:,} of {total:,} predictions",
                style={"color": "#94a3b8", "textAlign": "center", "marginTop": "8px"},
            ),
            html.Div(
                style={
                    "width": "100%", "height": "8px", "backgroundColor": "rgba(255,255,255,0.1)",
                    "borderRadius": "4px", "marginTop": "12px", "overflow": "hidden",
                },
                children=html.Div(style={
                    "width": f"{min(pct, 100):.1f}%", "height": "100%",
                    "backgroundColor": color, "borderRadius": "4px",
                    "transition": "width 0.5s ease",
                }),
            ),
        ], style={"padding": "20px"})
    except Exception:
        return html.P("—", style={"color": "#94a3b8", "textAlign": "center"})


# ═══════════════════════════════════════════════════════════════════════
# ALERTS PAGE
# ═══════════════════════════════════════════════════════════════════════

@callback(
    [Output("total-alerts-value", "children"), Output("total-alerts-delta", "children"), Output("total-alerts-status", "style")],
    Input("interval-refresh", "n_intervals"),
)
def update_total_alerts_card(_n):
    try:
        db = _safe_db()
        from src.alerting.alert_manager import AlertManager
        am = AlertManager(db)
        stats = am.get_alert_stats()
        return str(stats.get("total", 0)), "all time", _dot_style("#3b82f6")
    except Exception:
        return "—", "—", _dot_style("#6b7280")


@callback(
    [Output("active-alerts-count-value", "children"), Output("active-alerts-count-delta", "children"), Output("active-alerts-count-status", "style")],
    Input("interval-refresh", "n_intervals"),
)
def update_active_alerts_count(_n):
    try:
        db = _safe_db()
        from src.alerting.alert_manager import AlertManager
        am = AlertManager(db)
        stats = am.get_alert_stats()
        active = stats.get("active", 0)
        color = "#10b981" if active == 0 else "#ef4444"
        return str(active), "unacknowledged", _dot_style(color)
    except Exception:
        return "—", "—", _dot_style("#6b7280")


@callback(
    [Output("resolved-alerts-count-value", "children"), Output("resolved-alerts-count-delta", "children"), Output("resolved-alerts-count-status", "style")],
    Input("interval-refresh", "n_intervals"),
)
def update_resolved_alerts_count(_n):
    try:
        db = _safe_db()
        from src.data.database import Alert
        with db.get_session() as session:
            count = session.query(func.count(Alert.id)).filter(Alert.severity == "RESOLVED").scalar() or 0
        return str(count), "resolved", _dot_style("#10b981")
    except Exception:
        return "—", "—", _dot_style("#6b7280")


@callback(
    Output("alert-frequency-chart", "figure"),
    Input("interval-refresh", "n_intervals"),
)
def update_alert_frequency(_n):
    fig = go.Figure()
    layout = get_dark_layout("Alert Frequency", "Time", "Count")
    try:
        db = _safe_db()
        from src.data.database import Alert
        with db.get_session() as session:
            rows = (
                session.query(
                    func.strftime("%Y-%m-%d %H", Alert.timestamp).label("hour"),
                    Alert.severity,
                    func.count(Alert.id).label("cnt"),
                )
                .group_by("hour", Alert.severity)
                .order_by("hour")
                .limit(200)
                .all()
            )
        if rows:
            severity_data = {}
            for hour, severity, cnt in rows:
                if severity not in severity_data:
                    severity_data[severity] = {"hours": [], "counts": []}
                severity_data[severity]["hours"].append(hour)
                severity_data[severity]["counts"].append(cnt)

            colors = {"CRITICAL": "#ef4444", "WARNING": "#f59e0b", "RESOLVED": "#10b981"}
            for sev, data in severity_data.items():
                fig.add_trace(go.Bar(
                    x=data["hours"], y=data["counts"],
                    name=sev, marker_color=colors.get(sev, "#3b82f6"),
                    opacity=0.85,
                ))
            fig.update_layout(barmode="stack")
    except Exception:
        pass

    fig.update_layout(**layout)
    return fig


@callback(
    Output("alert-summary-content", "children"),
    Input("interval-refresh", "n_intervals"),
)
def update_alert_summary(_n):
    try:
        db = _safe_db()
        from src.alerting.alert_manager import AlertManager
        am = AlertManager(db)
        stats = am.get_alert_stats()
        by_severity = stats.get("by_severity", {})
        by_type = stats.get("by_type", {})

        items = []
        items.append(html.H6("By Severity", style={"color": "#e2e8f0", "marginBottom": "8px"}))
        for sev, cnt in by_severity.items():
            badge = create_alert_badge(sev, f"{sev}: {cnt}")
            items.append(html.Div(badge, style={"marginBottom": "4px"}))

        items.append(html.Hr(style={"borderColor": "rgba(255,255,255,0.1)", "margin": "12px 0"}))
        items.append(html.H6("By Type", style={"color": "#e2e8f0", "marginBottom": "8px"}))
        for typ, cnt in by_type.items():
            items.append(html.P(f"{typ}: {cnt}", style={"color": "#94a3b8", "margin": "2px 0", "fontSize": "0.9rem"}))

        return html.Div(items, style={"padding": "12px"})
    except Exception:
        return html.P("Error loading summary.", style={"color": "#ef4444"})


@callback(
    Output("alert-history-table", "children"),
    [Input("alert-severity-filter", "value"), Input("alert-type-filter", "value"), Input("interval-refresh", "n_intervals")],
)
def update_alert_history_table(severity_filter, type_filter, _n):
    try:
        db = _safe_db()
        from src.data.database import Alert
        with db.get_session() as session:
            query = session.query(Alert).order_by(desc(Alert.timestamp))
            if severity_filter and severity_filter != "All":
                query = query.filter(Alert.severity == severity_filter)
            if type_filter and type_filter != "All":
                query = query.filter(Alert.drift_type == type_filter)
            alerts = query.limit(50).all()

        if not alerts:
            return html.P("No alerts match the selected filters.", style={"color": "#94a3b8", "textAlign": "center", "padding": "20px"})

        rows = []
        for a in alerts:
            badge = create_alert_badge(a.severity, a.severity)
            ack_icon = "✓" if a.acknowledged else "✗"
            ack_color = "#10b981" if a.acknowledged else "#ef4444"
            rows.append(html.Tr([
                html.Td(a.timestamp.strftime("%Y-%m-%d %H:%M:%S") if a.timestamp else "—", style={"color": "#94a3b8", "padding": "8px", "fontSize": "0.85rem"}),
                html.Td(badge),
                html.Td(a.drift_type, style={"color": "#e2e8f0", "padding": "8px"}),
                html.Td(a.message[:100] if a.message else "—", style={"color": "#94a3b8", "padding": "8px", "fontSize": "0.85rem"}),
                html.Td(ack_icon, style={"color": ack_color, "textAlign": "center", "fontSize": "1.1rem"}),
            ], style={"borderBottom": "1px solid rgba(255,255,255,0.05)"}))

        return html.Table([
            html.Thead(html.Tr([
                html.Th("Time", style={"color": "#94a3b8", "padding": "8px", "fontWeight": "600"}),
                html.Th("Severity", style={"color": "#94a3b8", "padding": "8px", "fontWeight": "600"}),
                html.Th("Type", style={"color": "#94a3b8", "padding": "8px", "fontWeight": "600"}),
                html.Th("Message", style={"color": "#94a3b8", "padding": "8px", "fontWeight": "600"}),
                html.Th("Ack", style={"color": "#94a3b8", "padding": "8px", "fontWeight": "600", "textAlign": "center"}),
            ])),
            html.Tbody(rows),
        ], style={"width": "100%", "borderCollapse": "collapse"})
    except Exception:
        return html.P("Error loading alert history.", style={"color": "#ef4444"})


# ═══════════════════════════════════════════════════════════════════════
# MODEL REGISTRY PAGE
# ═══════════════════════════════════════════════════════════════════════

@callback(
    Output("production-model-card", "children"),
    Input("interval-refresh", "n_intervals"),
)
def update_production_model_card(_n):
    try:
        db = _safe_db()
        with db.get_session() as session:
            prod = (
                session.query(ModelVersion)
                .filter(ModelVersion.is_production == True)
                .one_or_none()
            )
        if prod:
            return html.Div([
                html.Div([
                    html.Span("🏆 ", style={"fontSize": "1.5rem"}),
                    html.Span("Production Model", style={"color": "#3b82f6", "fontWeight": "700", "fontSize": "1.2rem"}),
                ], style={"marginBottom": "16px"}),
                dbc.Row([
                    dbc.Col([
                        html.P("Version", style={"color": "#94a3b8", "fontSize": "0.8rem", "margin": "0"}),
                        html.P(f"v{prod.version}", style={"color": "#e2e8f0", "fontSize": "1.5rem", "fontWeight": "700", "margin": "0"}),
                    ], width=3),
                    dbc.Col([
                        html.P("F1 Score", style={"color": "#94a3b8", "fontSize": "0.8rem", "margin": "0"}),
                        html.P(f"{prod.f1_score:.4f}", style={"color": "#10b981", "fontSize": "1.5rem", "fontWeight": "700", "margin": "0"}),
                    ], width=3),
                    dbc.Col([
                        html.P("Accuracy", style={"color": "#94a3b8", "fontSize": "0.8rem", "margin": "0"}),
                        html.P(f"{prod.accuracy:.4f}", style={"color": "#3b82f6", "fontSize": "1.5rem", "fontWeight": "700", "margin": "0"}),
                    ], width=3),
                    dbc.Col([
                        html.P("Deployed", style={"color": "#94a3b8", "fontSize": "0.8rem", "margin": "0"}),
                        html.P(
                            prod.deployed_at.strftime("%Y-%m-%d %H:%M") if prod.deployed_at else "—",
                            style={"color": "#e2e8f0", "fontSize": "1rem", "fontWeight": "500", "margin": "0"},
                        ),
                    ], width=3),
                ]),
            ], style={"padding": "20px"})
        return html.P("No production model deployed.", style={"color": "#94a3b8", "textAlign": "center", "padding": "20px"})
    except Exception:
        return html.P("Error loading model info.", style={"color": "#ef4444"})


@callback(
    Output("versions-table", "children"),
    Input("interval-refresh", "n_intervals"),
)
def update_versions_table(_n):
    try:
        db = _safe_db()
        with db.get_session() as session:
            versions = (
                session.query(ModelVersion)
                .order_by(desc(ModelVersion.version))
                .limit(20)
                .all()
            )
        if not versions:
            return html.P("No model versions registered.", style={"color": "#94a3b8", "textAlign": "center", "padding": "20px"})

        rows = []
        for v in versions:
            prod_badge = dbc.Badge("PRODUCTION", color="primary", pill=True) if v.is_production else ""
            rows.append(html.Tr([
                html.Td(f"v{v.version}", style={"color": "#e2e8f0", "padding": "8px", "fontWeight": "600"}),
                html.Td(prod_badge),
                html.Td(f"{v.f1_score:.4f}", style={"color": "#10b981", "padding": "8px"}),
                html.Td(f"{v.accuracy:.4f}", style={"color": "#3b82f6", "padding": "8px"}),
                html.Td(f"{v.precision:.4f}" if v.precision else "—", style={"color": "#f59e0b", "padding": "8px"}),
                html.Td(f"{v.recall:.4f}" if v.recall else "—", style={"color": "#8b5cf6", "padding": "8px"}),
                html.Td(
                    v.training_date.strftime("%Y-%m-%d %H:%M") if v.training_date else "—",
                    style={"color": "#94a3b8", "padding": "8px", "fontSize": "0.85rem"},
                ),
            ], style={"borderBottom": "1px solid rgba(255,255,255,0.05)"}))

        return html.Table([
            html.Thead(html.Tr([
                html.Th("Version", style={"color": "#94a3b8", "padding": "8px", "fontWeight": "600"}),
                html.Th("Status", style={"color": "#94a3b8", "padding": "8px", "fontWeight": "600"}),
                html.Th("F1", style={"color": "#94a3b8", "padding": "8px", "fontWeight": "600"}),
                html.Th("Accuracy", style={"color": "#94a3b8", "padding": "8px", "fontWeight": "600"}),
                html.Th("Precision", style={"color": "#94a3b8", "padding": "8px", "fontWeight": "600"}),
                html.Th("Recall", style={"color": "#94a3b8", "padding": "8px", "fontWeight": "600"}),
                html.Th("Trained", style={"color": "#94a3b8", "padding": "8px", "fontWeight": "600"}),
            ])),
            html.Tbody(rows),
        ], style={"width": "100%", "borderCollapse": "collapse"})
    except Exception:
        return html.P("Error loading versions.", style={"color": "#ef4444"})


@callback(
    Output("version-metrics-chart", "figure"),
    Input("interval-refresh", "n_intervals"),
)
def update_version_metrics_chart(_n):
    fig = go.Figure()
    layout = get_dark_layout("Model Version Metrics Comparison", "Version", "Score")
    try:
        db = _safe_db()
        with db.get_session() as session:
            versions = (
                session.query(ModelVersion)
                .order_by(ModelVersion.version)
                .all()
            )
        if versions:
            v_labels = [f"v{v.version}" for v in versions]
            metrics = {
                "F1": ([v.f1_score for v in versions], "#10b981"),
                "Accuracy": ([v.accuracy for v in versions], "#3b82f6"),
                "Precision": ([v.precision for v in versions], "#f59e0b"),
                "Recall": ([v.recall for v in versions], "#8b5cf6"),
            }
            for name, (vals, color) in metrics.items():
                fig.add_trace(go.Bar(
                    x=v_labels, y=vals, name=name,
                    marker_color=color, opacity=0.85,
                ))
            fig.update_layout(barmode="group")
    except Exception:
        pass

    fig.update_layout(**layout)
    return fig


@callback(
    Output("retraining-history-table", "children"),
    Input("interval-refresh", "n_intervals"),
)
def update_retraining_history(_n):
    try:
        db = _safe_db()
        with db.get_session() as session:
            events = (
                session.query(RetrainingEvent)
                .order_by(desc(RetrainingEvent.timestamp))
                .limit(20)
                .all()
            )
        if not events:
            return html.P("No retraining events recorded.", style={"color": "#94a3b8", "textAlign": "center", "padding": "20px"})

        status_colors = {"COMPLETED": "#10b981", "FAILED": "#ef4444", "REJECTED": "#f59e0b", "STARTED": "#3b82f6"}
        rows = []
        for e in events:
            color = status_colors.get(e.status, "#6b7280")
            rows.append(html.Tr([
                html.Td(
                    e.timestamp.strftime("%Y-%m-%d %H:%M") if e.timestamp else "—",
                    style={"color": "#94a3b8", "padding": "8px", "fontSize": "0.85rem"},
                ),
                html.Td(
                    dbc.Badge(e.status, style={"backgroundColor": color}, pill=True),
                ),
                html.Td(f"v{e.old_version}" if e.old_version else "—", style={"color": "#e2e8f0", "padding": "8px"}),
                html.Td(f"v{e.new_version}" if e.new_version else "—", style={"color": "#e2e8f0", "padding": "8px"}),
                html.Td(f"{e.old_f1:.4f}" if e.old_f1 else "—", style={"color": "#94a3b8", "padding": "8px"}),
                html.Td(f"{e.new_f1:.4f}" if e.new_f1 else "—", style={"color": "#94a3b8", "padding": "8px"}),
                html.Td(
                    e.trigger_reason[:60] + "..." if e.trigger_reason and len(e.trigger_reason) > 60 else (e.trigger_reason or "—"),
                    style={"color": "#94a3b8", "padding": "8px", "fontSize": "0.85rem"},
                ),
            ], style={"borderBottom": "1px solid rgba(255,255,255,0.05)"}))

        return html.Table([
            html.Thead(html.Tr([
                html.Th("Time", style={"color": "#94a3b8", "padding": "8px", "fontWeight": "600"}),
                html.Th("Status", style={"color": "#94a3b8", "padding": "8px", "fontWeight": "600"}),
                html.Th("From", style={"color": "#94a3b8", "padding": "8px", "fontWeight": "600"}),
                html.Th("To", style={"color": "#94a3b8", "padding": "8px", "fontWeight": "600"}),
                html.Th("Old F1", style={"color": "#94a3b8", "padding": "8px", "fontWeight": "600"}),
                html.Th("New F1", style={"color": "#94a3b8", "padding": "8px", "fontWeight": "600"}),
                html.Th("Reason", style={"color": "#94a3b8", "padding": "8px", "fontWeight": "600"}),
            ])),
            html.Tbody(rows),
        ], style={"width": "100%", "borderCollapse": "collapse"})
    except Exception:
        return html.P("Error loading retraining history.", style={"color": "#ef4444"})
