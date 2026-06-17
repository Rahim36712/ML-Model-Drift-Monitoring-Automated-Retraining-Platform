"""Drift analysis page callbacks — data drift and prediction drift charts.

Registered automatically when imported by the Dash app factory.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback, html
import dash_bootstrap_components as dbc

from src.config.settings import get_settings
from src.data.database import get_database, DriftResult
from src.dashboard.components.trend_chart import get_dark_layout, add_threshold_bands
from src.dashboard.components.alert_badge import create_alert_badge

from sqlalchemy import desc

logger = logging.getLogger(__name__)


def _safe_db():
    try:
        settings = get_settings()
        return get_database(settings.database.url)
    except Exception:
        return get_database()


# ═══════════════════════════════════════════════════════════════════════
# DATA DRIFT PAGE CALLBACKS
# ═══════════════════════════════════════════════════════════════════════

@callback(
    [
        Output("max-psi-value", "children"),
        Output("max-psi-delta", "children"),
        Output("max-psi-status", "style"),
    ],
    Input("interval-refresh", "n_intervals"),
)
def update_max_psi(_n):
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
            label = "Stable" if val < 0.10 else "Moderate" if val < 0.25 else "Significant"
            return f"{val:.4f}", label, {"backgroundColor": color, "width": "8px", "height": "8px", "borderRadius": "50%", "display": "inline-block"}
        return "N/A", "No data", {"backgroundColor": "#6b7280", "width": "8px", "height": "8px", "borderRadius": "50%", "display": "inline-block"}
    except Exception:
        return "—", "—", {"backgroundColor": "#6b7280", "width": "8px", "height": "8px", "borderRadius": "50%", "display": "inline-block"}


@callback(
    [
        Output("drifted-features-count-value", "children"),
        Output("drifted-features-count-delta", "children"),
        Output("drifted-features-count-status", "style"),
    ],
    Input("interval-refresh", "n_intervals"),
)
def update_drifted_features_count(_n):
    try:
        db = _safe_db()
        with db.get_session() as session:
            latest = (
                session.query(DriftResult)
                .filter(DriftResult.drift_type == "data", DriftResult.metric_name == "psi")
                .order_by(desc(DriftResult.timestamp))
                .first()
            )
        if latest and latest.details_json:
            details = json.loads(latest.details_json)
            drifted = details.get("drifted_features", [])
            count = len(drifted)
            color = "#10b981" if count == 0 else "#f59e0b" if count < 5 else "#ef4444"
            return str(count), f"of 29 features", {"backgroundColor": color, "width": "8px", "height": "8px", "borderRadius": "50%", "display": "inline-block"}
        return "0", "No data", {"backgroundColor": "#6b7280", "width": "8px", "height": "8px", "borderRadius": "50%", "display": "inline-block"}
    except Exception:
        return "—", "—", {"backgroundColor": "#6b7280", "width": "8px", "height": "8px", "borderRadius": "50%", "display": "inline-block"}


@callback(
    Output("feature-psi-heatmap", "figure"),
    Input("interval-refresh", "n_intervals"),
)
def update_feature_psi_heatmap(_n):
    fig = go.Figure()
    layout = get_dark_layout("Feature PSI Over Time", "Check Time", "Feature")
    try:
        db = _safe_db()
        with db.get_session() as session:
            rows = (
                session.query(DriftResult)
                .filter(DriftResult.drift_type == "data", DriftResult.metric_name == "psi")
                .order_by(DriftResult.timestamp)
                .limit(50)
                .all()
            )

        if rows:
            # Build heatmap data: rows are features, columns are timestamps
            timestamps = []
            all_features = set()
            psi_data = []

            for r in rows:
                if r.details_json:
                    details = json.loads(r.details_json)
                    feature_psi = details.get("feature_psi", {})
                    if feature_psi:
                        timestamps.append(r.timestamp.strftime("%H:%M:%S"))
                        psi_data.append(feature_psi)
                        all_features.update(feature_psi.keys())

            if timestamps and all_features:
                features_sorted = sorted(all_features)[:15]  # Top 15 for readability
                z = []
                for feat in features_sorted:
                    z.append([d.get(feat, 0) for d in psi_data])

                fig.add_trace(go.Heatmap(
                    z=z,
                    x=timestamps,
                    y=features_sorted,
                    colorscale=[[0, "#0f1729"], [0.4, "#3b82f6"], [0.7, "#f59e0b"], [1, "#ef4444"]],
                    colorbar=dict(title="PSI", tickfont=dict(color="#e2e8f0")),
                    hoverongaps=False,
                ))
    except Exception:
        logger.exception("Failed to load feature PSI heatmap.")

    fig.update_layout(**layout)
    return fig


@callback(
    Output("top-drifted-bar", "figure"),
    Input("interval-refresh", "n_intervals"),
)
def update_top_drifted_bar(_n):
    fig = go.Figure()
    layout = get_dark_layout("Top Drifted Features", "PSI Value", "Feature")
    try:
        db = _safe_db()
        with db.get_session() as session:
            latest = (
                session.query(DriftResult)
                .filter(DriftResult.drift_type == "data", DriftResult.metric_name == "psi")
                .order_by(desc(DriftResult.timestamp))
                .first()
            )
        if latest and latest.details_json:
            details = json.loads(latest.details_json)
            feature_psi = details.get("feature_psi", {})
            if feature_psi:
                sorted_features = sorted(feature_psi.items(), key=lambda x: x[1], reverse=True)[:10]
                names = [f[0] for f in sorted_features]
                vals = [f[1] for f in sorted_features]
                colors = ["#ef4444" if v > 0.25 else "#f59e0b" if v > 0.10 else "#10b981" for v in vals]

                fig.add_trace(go.Bar(
                    x=vals, y=names, orientation="h",
                    marker_color=colors,
                ))
                fig.add_vline(x=0.10, line_dash="dash", line_color="#f59e0b")
                fig.add_vline(x=0.25, line_dash="dash", line_color="#ef4444")
    except Exception:
        logger.exception("Failed to load top drifted features.")

    fig.update_layout(**layout)
    return fig


@callback(
    Output("feature-selector", "options"),
    Input("interval-refresh", "n_intervals"),
)
def populate_feature_dropdown(_n):
    try:
        db = _safe_db()
        with db.get_session() as session:
            latest = (
                session.query(DriftResult)
                .filter(DriftResult.drift_type == "data", DriftResult.metric_name == "psi")
                .order_by(desc(DriftResult.timestamp))
                .first()
            )
        if latest and latest.details_json:
            details = json.loads(latest.details_json)
            feature_psi = details.get("feature_psi", {})
            return [{"label": f"{k} (PSI={v:.4f})", "value": k} for k, v in sorted(feature_psi.items())]
    except Exception:
        pass
    return [{"label": f"V{i}", "value": f"V{i}"} for i in range(1, 29)] + [{"label": "Amount", "value": "Amount"}]


@callback(
    Output("feature-distribution-chart", "figure"),
    [Input("feature-selector", "value"), Input("interval-refresh", "n_intervals")],
)
def update_feature_distribution(selected_feature, _n):
    fig = go.Figure()
    title = f"Distribution: {selected_feature}" if selected_feature else "Select a feature"
    layout = get_dark_layout(title, "Value", "Density")
    # Placeholder — real implementation would compare reference vs current distributions
    if not selected_feature:
        fig.update_layout(**layout)
        return fig

    try:
        from src.data.loader import DataLoader
        loader = DataLoader()
        ref_data = loader.load_reference_distribution()
        if selected_feature in ref_data:
            ref_values = ref_data[selected_feature]
            fig.add_trace(go.Histogram(
                x=ref_values, name="Reference", opacity=0.6,
                marker_color="#3b82f6", nbinsx=50,
                histnorm="probability density",
            ))
    except Exception:
        pass

    fig.update_layout(**layout)
    return fig


@callback(
    Output("kl-table", "children"),
    Input("interval-refresh", "n_intervals"),
)
def update_kl_table(_n):
    try:
        db = _safe_db()
        with db.get_session() as session:
            latest = (
                session.query(DriftResult)
                .filter(DriftResult.drift_type == "data", DriftResult.metric_name == "psi")
                .order_by(desc(DriftResult.timestamp))
                .first()
            )
        if latest and latest.details_json:
            details = json.loads(latest.details_json)
            feature_psi = details.get("feature_psi", {})
            if not feature_psi:
                return html.P("No PSI data available.", style={"color": "#94a3b8"})

            rows = []
            for feat, psi_val in sorted(feature_psi.items(), key=lambda x: x[1], reverse=True):
                status_color = "#10b981" if psi_val < 0.10 else "#f59e0b" if psi_val < 0.25 else "#ef4444"
                rows.append(html.Tr([
                    html.Td(feat, style={"color": "#e2e8f0", "padding": "6px 12px"}),
                    html.Td(f"{psi_val:.4f}", style={"color": status_color, "padding": "6px 12px", "fontWeight": "600"}),
                    html.Td("●", style={"color": status_color, "fontSize": "1.2rem", "textAlign": "center"}),
                ], style={"borderBottom": "1px solid rgba(255,255,255,0.05)"}))

            return html.Table([
                html.Thead(html.Tr([
                    html.Th("Feature", style={"color": "#94a3b8", "padding": "8px 12px", "fontWeight": "600"}),
                    html.Th("PSI", style={"color": "#94a3b8", "padding": "8px 12px", "fontWeight": "600"}),
                    html.Th("Status", style={"color": "#94a3b8", "padding": "8px 12px", "fontWeight": "600", "textAlign": "center"}),
                ])),
                html.Tbody(rows),
            ], style={"width": "100%", "borderCollapse": "collapse"})
        return html.P("No drift data recorded yet.", style={"color": "#94a3b8", "textAlign": "center", "padding": "20px"})
    except Exception:
        return html.P("Error loading data.", style={"color": "#ef4444"})


# ═══════════════════════════════════════════════════════════════════════
# PREDICTION DRIFT PAGE CALLBACKS
# ═══════════════════════════════════════════════════════════════════════

@callback(
    [
        Output("hellinger-dist-value", "children"),
        Output("hellinger-dist-delta", "children"),
        Output("hellinger-dist-status", "style"),
    ],
    Input("interval-refresh", "n_intervals"),
)
def update_hellinger_card(_n):
    try:
        db = _safe_db()
        with db.get_session() as session:
            latest = (
                session.query(DriftResult)
                .filter(DriftResult.drift_type == "prediction", DriftResult.metric_name == "hellinger")
                .order_by(desc(DriftResult.timestamp))
                .first()
            )
        if latest:
            val = latest.metric_value
            color = "#10b981" if val < 0.10 else "#f59e0b" if val < 0.20 else "#ef4444"
            label = "Stable" if val < 0.10 else "Moderate" if val < 0.20 else "Significant"
            return f"{val:.4f}", label, {"backgroundColor": color, "width": "8px", "height": "8px", "borderRadius": "50%", "display": "inline-block"}
        return "N/A", "No data", {"backgroundColor": "#6b7280", "width": "8px", "height": "8px", "borderRadius": "50%", "display": "inline-block"}
    except Exception:
        return "—", "—", {"backgroundColor": "#6b7280", "width": "8px", "height": "8px", "borderRadius": "50%", "display": "inline-block"}


@callback(
    [
        Output("positive-rate-value", "children"),
        Output("positive-rate-delta", "children"),
        Output("positive-rate-status", "style"),
    ],
    Input("interval-refresh", "n_intervals"),
)
def update_positive_rate_card(_n):
    try:
        db = _safe_db()
        with db.get_session() as session:
            latest = (
                session.query(DriftResult)
                .filter(DriftResult.drift_type == "prediction", DriftResult.metric_name == "hellinger")
                .order_by(desc(DriftResult.timestamp))
                .first()
            )
        if latest and latest.details_json:
            details = json.loads(latest.details_json)
            rate = details.get("current_positive_rate", 0)
            baseline = details.get("baseline_positive_rate", 0)
            delta = rate - baseline
            color = "#10b981" if abs(delta) < 0.02 else "#f59e0b" if abs(delta) < 0.05 else "#ef4444"
            return f"{rate:.2%}", f"{'↑' if delta >= 0 else '↓'} {abs(delta):.2%} from baseline", {"backgroundColor": color, "width": "8px", "height": "8px", "borderRadius": "50%", "display": "inline-block"}
        return "N/A", "No data", {"backgroundColor": "#6b7280", "width": "8px", "height": "8px", "borderRadius": "50%", "display": "inline-block"}
    except Exception:
        return "—", "—", {"backgroundColor": "#6b7280", "width": "8px", "height": "8px", "borderRadius": "50%", "display": "inline-block"}


@callback(
    [
        Output("mean-confidence-value", "children"),
        Output("mean-confidence-delta", "children"),
        Output("mean-confidence-status", "style"),
    ],
    Input("interval-refresh", "n_intervals"),
)
def update_mean_confidence_card(_n):
    try:
        db = _safe_db()
        with db.get_session() as session:
            latest = (
                session.query(DriftResult)
                .filter(DriftResult.drift_type == "prediction", DriftResult.metric_name == "hellinger")
                .order_by(desc(DriftResult.timestamp))
                .first()
            )
        if latest and latest.details_json:
            details = json.loads(latest.details_json)
            conf = details.get("current_mean_confidence", 0)
            baseline = details.get("baseline_mean_confidence", 0)
            delta = conf - baseline
            color = "#10b981" if abs(delta) < 0.05 else "#f59e0b"
            return f"{conf:.4f}", f"{'↑' if delta >= 0 else '↓'} {abs(delta):.4f}", {"backgroundColor": color, "width": "8px", "height": "8px", "borderRadius": "50%", "display": "inline-block"}
        return "N/A", "No data", {"backgroundColor": "#6b7280", "width": "8px", "height": "8px", "borderRadius": "50%", "display": "inline-block"}
    except Exception:
        return "—", "—", {"backgroundColor": "#6b7280", "width": "8px", "height": "8px", "borderRadius": "50%", "display": "inline-block"}


@callback(
    Output("hellinger-trend-chart", "figure"),
    Input("interval-refresh", "n_intervals"),
)
def update_hellinger_trend(_n):
    fig = go.Figure()
    layout = get_dark_layout("Hellinger Distance Over Time", "Time", "Distance")
    try:
        db = _safe_db()
        with db.get_session() as session:
            rows = (
                session.query(DriftResult)
                .filter(DriftResult.drift_type == "prediction", DriftResult.metric_name == "hellinger")
                .order_by(DriftResult.timestamp)
                .limit(100)
                .all()
            )
        if rows:
            timestamps = [r.timestamp for r in rows]
            values = [r.metric_value for r in rows]
            fig.add_trace(go.Scatter(
                x=timestamps, y=values, mode="lines+markers",
                name="Hellinger", line=dict(color="#8b5cf6", width=2),
                marker=dict(size=4),
            ))
            fig.add_hline(y=0.10, line_dash="dash", line_color="#f59e0b")
            fig.add_hline(y=0.20, line_dash="dash", line_color="#ef4444")
    except Exception:
        logger.exception("Failed to load Hellinger trend.")

    fig.update_layout(**layout)
    return fig


@callback(
    Output("confidence-distribution-chart", "figure"),
    Input("interval-refresh", "n_intervals"),
)
def update_confidence_distribution(_n):
    fig = go.Figure()
    layout = get_dark_layout("Confidence Score Distribution", "Confidence", "Count")
    try:
        db = _safe_db()
        from src.data.database import Prediction
        with db.get_session() as session:
            rows = (
                session.query(Prediction.confidence)
                .order_by(desc(Prediction.timestamp))
                .limit(500)
                .all()
            )
        if rows:
            confidences = [r[0] for r in rows]
            fig.add_trace(go.Histogram(
                x=confidences, nbinsx=50, name="Current",
                marker_color="#3b82f6", opacity=0.7,
            ))
    except Exception:
        pass

    fig.update_layout(**layout)
    return fig


@callback(
    Output("positive-rate-trend-chart", "figure"),
    Input("interval-refresh", "n_intervals"),
)
def update_positive_rate_trend(_n):
    fig = go.Figure()
    layout = get_dark_layout("Positive Prediction Rate", "Time", "Rate")
    try:
        db = _safe_db()
        with db.get_session() as session:
            rows = (
                session.query(DriftResult)
                .filter(DriftResult.drift_type == "prediction", DriftResult.metric_name == "hellinger")
                .order_by(DriftResult.timestamp)
                .limit(100)
                .all()
            )
        if rows:
            timestamps = []
            rates = []
            for r in rows:
                if r.details_json:
                    details = json.loads(r.details_json)
                    rate = details.get("current_positive_rate")
                    if rate is not None:
                        timestamps.append(r.timestamp)
                        rates.append(rate)
            if timestamps:
                fig.add_trace(go.Scatter(
                    x=timestamps, y=rates, mode="lines+markers",
                    name="Positive Rate", line=dict(color="#f59e0b", width=2),
                    marker=dict(size=4),
                ))
    except Exception:
        pass

    fig.update_layout(**layout)
    return fig


@callback(
    Output("class-balance-chart", "figure"),
    Input("interval-refresh", "n_intervals"),
)
def update_class_balance(_n):
    fig = go.Figure()
    layout = get_dark_layout("Prediction Class Balance", "", "")
    try:
        db = _safe_db()
        from src.data.database import Prediction
        from sqlalchemy import func as sqlfunc
        with db.get_session() as session:
            counts = (
                session.query(Prediction.predicted_label, sqlfunc.count(Prediction.id))
                .group_by(Prediction.predicted_label)
                .all()
            )
        if counts:
            labels = ["Legitimate" if c[0] == 0 else "Fraud" for c in counts]
            values = [c[1] for c in counts]
            fig.add_trace(go.Pie(
                labels=labels, values=values,
                hole=0.5,
                marker=dict(colors=["#3b82f6", "#ef4444"]),
                textfont=dict(color="#e2e8f0"),
            ))
    except Exception:
        pass

    fig.update_layout(**layout)
    fig.update_layout(showlegend=True, legend=dict(font=dict(color="#e2e8f0")))
    return fig
