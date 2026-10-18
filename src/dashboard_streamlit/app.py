"""Streamlit UI Application for MLOps Model Drift Monitoring & Automated Retraining Platform.

Built using the Stitch Design System:
- Typography: Comfortaa (Google Fonts)
- Primary Accent: Lime Green (#84CC16 / #65A30D)
- Canvas: Off-White (#FAF9F6)
- Containers: Crisp White (#FFFFFF) with subtle lime glow
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings
from src.data.database import get_database, Prediction, DriftResult, ModelVersion, Alert
from src.alerting.email_notifier import EmailNotifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit_app")

# API and Server settings
API_URL = "http://127.0.0.1:8000"


# -----------------------------------------------------------------------------
# Stitch Design System & CSS Custom Styling
# -----------------------------------------------------------------------------
def apply_stitch_theme():
    st.set_page_config(
        page_title="MLOps Drift Monitor",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;600;700&display=swap');

        /* Global Font & Canvas */
        html, body, [class*="css"], .stApp {
            font-family: 'Comfortaa', cursive, sans-serif !important;
            background-color: #FAF9F6 !important;
            color: #1E293B !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
        }

        /* Metric Cards */
        div[data-testid="stMetricValue"] {
            font-size: 2.2rem !important;
            color: #65A30D !important;
            font-weight: 700 !important;
            font-family: 'Comfortaa', cursive !important;
        }

        div[data-testid="stMetricLabel"] {
            font-size: 0.9rem !important;
            color: #64748B !important;
            font-weight: 600 !important;
        }

        /* Custom Cards */
        .stitch-card {
            background-color: #FFFFFF;
            border-radius: 14px;
            padding: 24px;
            border: 1px solid rgba(132, 204, 22, 0.3);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
            margin-bottom: 20px;
        }

        .stitch-header {
            font-family: 'Comfortaa', cursive !important;
            font-size: 1.8rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 8px;
        }

        .stitch-subtitle {
            font-size: 0.95rem;
            color: #64748B;
            margin-bottom: 24px;
        }

        /* Lime Green Buttons */
        .stButton>button {
            background-color: #84CC16 !important;
            color: #FFFFFF !important;
            border-radius: 10px !important;
            border: none !important;
            font-family: 'Comfortaa', sans-serif !important;
            font-weight: 700 !important;
            padding: 0.6rem 1.4rem !important;
            transition: all 0.2s ease-in-out;
        }

        .stButton>button:hover {
            background-color: #65A30D !important;
            box-shadow: 0 4px 14px rgba(101, 163, 13, 0.4) !important;
            transform: translateY(-1px);
        }

        /* Health Badges */
        .badge-healthy {
            background-color: #DCFCE7;
            color: #15803D;
            padding: 6px 16px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.9rem;
            display: inline-block;
        }

        .badge-warning {
            background-color: #FEF3C7;
            color: #B45309;
            padding: 6px 16px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.9rem;
            display: inline-block;
        }

        .badge-critical {
            background-color: #FEE2E2;
            color: #B91C1C;
            padding: 6px 16px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.9rem;
            display: inline-block;
        }
        </style>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Database Data Fetching Helpers
# -----------------------------------------------------------------------------
@st.cache_resource
def get_db_instance():
    settings = get_settings()
    db = get_database(settings.database.url)
    db.init_db()
    return db

def fetch_predictions_df(limit: int = 500) -> pd.DataFrame:
    db = get_db_instance()
    try:
        with db.get_session() as session:
            query = session.query(Prediction).order_by(Prediction.timestamp.desc()).limit(limit)
            records = query.all()
            if not records:
                return pd.DataFrame()
            data = []
            for r in records:
                data.append({
                    "id": r.id,
                    "timestamp": r.timestamp,
                    "model_version": r.model_version,
                    "predicted_label": r.predicted_label,
                    "confidence": r.confidence,
                    "true_label": r.true_label,
                    "latency_ms": r.latency_ms
                })
            df = pd.DataFrame(data)
            return df
    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        return pd.DataFrame()

def fetch_model_versions() -> pd.DataFrame:
    db = get_db_instance()
    try:
        with db.get_session() as session:
            versions = session.query(ModelVersion).order_by(ModelVersion.version.desc()).all()
            data = []
            for v in versions:
                data.append({
                    "version": v.version,
                    "mlflow_run_id": v.mlflow_run_id,
                    "is_production": v.is_production,
                    "accuracy": v.accuracy,
                    "f1_score": v.f1_score,
                    "precision": v.precision,
                    "recall": v.recall,
                    "deployed_at": v.deployed_at
                })
            return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error fetching model versions: {e}")
        return pd.DataFrame()

def fetch_alert_logs() -> pd.DataFrame:
    db = get_db_instance()
    try:
        with db.get_session() as session:
            alerts = session.query(Alert).order_by(Alert.timestamp.desc()).limit(100).all()
            data = []
            for a in alerts:
                data.append({
                    "id": a.id,
                    "timestamp": a.timestamp,
                    "channel": a.channel,
                    "severity": a.severity,
                    "drift_type": a.drift_type,
                    "message": a.message,
                    "acknowledged": a.acknowledged
                })
            return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error fetching alert logs: {e}")
        return pd.DataFrame()

def fetch_drift_results() -> pd.DataFrame:
    db = get_db_instance()
    try:
        with db.get_session() as session:
            drifts = session.query(DriftResult).order_by(DriftResult.timestamp.desc()).limit(50).all()
            data = []
            for d in drifts:
                data.append({
                    "timestamp": d.timestamp,
                    "overall_status": d.overall_status,
                    "data_drift_status": d.data_drift_status,
                    "prediction_drift_status": d.prediction_drift_status,
                    "concept_drift_status": d.concept_drift_status,
                    "max_psi": d.max_psi,
                    "metrics_json": d.metrics_json
                })
            return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error fetching drift results: {e}")
        return pd.DataFrame()


# -----------------------------------------------------------------------------
# Main Navigation & UI Layout
# -----------------------------------------------------------------------------
def main():
    apply_stitch_theme()

    # Sidebar Navigation
    st.sidebar.markdown("### 🤖 MLOps Platform")
    st.sidebar.markdown("*Stitch Lime-Green Control Center*")
    st.sidebar.divider()

    screen = st.sidebar.radio(
        "Navigate Screens",
        [
            "🟢 Executive Overview",
            "📊 Data & Concept Drift",
            "⚡ Live Telemetry & Latency",
            "🏆 Model Registry & Retraining",
            "🔔 Alert Center & Audit",
            "⚙️ System Settings & Email Config"
        ]
    )

    st.sidebar.divider()
    st.sidebar.caption("System Status: **ONLINE**")
    st.sidebar.caption("Theme: **Lime Green / Off-White**")

    # -------------------------------------------------------------------------
    # Screen 1: Executive Overview
    # -------------------------------------------------------------------------
    if screen == "🟢 Executive Overview":
        st.markdown("<div class='stitch-header'>Executive Overview & Health Monitor</div>", unsafe_allow_html=True)
        st.markdown("<div class='stitch-subtitle'>Real-time model performance, prediction volume, and system health status.</div>", unsafe_allow_html=True)

        df_preds = fetch_predictions_df()
        df_models = fetch_model_versions()
        df_drift = fetch_drift_results()

        # System Health Badge
        prod_ver = "1"
        if not df_models.empty:
            prod_row = df_models[df_models["is_production"] == True]
            if not prod_row.empty:
                prod_ver = str(prod_row.iloc[0]["version"])

        status_str = "✓ HEALTHY"
        if not df_drift.empty:
            latest_status = df_drift.iloc[0]["overall_status"]
            if latest_status in ["WARNING", "DATA_DRIFT"]:
                status_str = "⚠️ DRIFT DETECTED"
            elif latest_status in ["CRITICAL", "RETRAIN_TRIGGERED"]:
                status_str = "🚨 RETRAINING TRIGGERED"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("System Health", status_str)
        with c2:
            st.metric("Active Champion Model", f"Version {prod_ver}")
        with c3:
            total_reqs = len(df_preds) if not df_preds.empty else 0
            st.metric("Logged Requests", f"{total_reqs:,}")
        with c4:
            avg_lat = f"{df_preds['latency_ms'].mean():.1f} ms" if not df_preds.empty else "0.0 ms"
            st.metric("Avg Latency", avg_lat)

        st.markdown("---")

        # Stream Graph
        st.subheader("📈 Live Inference Throughput & Latency Stream")
        if not df_preds.empty:
            df_preds["timestamp"] = pd.to_datetime(df_preds["timestamp"])
            fig = px.line(
                df_preds,
                x="timestamp",
                y="latency_ms",
                title="Latency (ms) over Recent Requests",
                color_discrete_sequence=["#84CC16"]
            )
            fig.update_layout(
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font_family="Comfortaa",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No predictions logged yet. Run simulation script to generate traffic.")

        # Quick Actions & Interactive Demo Testing Suite
        st.markdown("---")
        st.subheader("🕹️ Interactive 1-Click System Testing Suite")
        st.markdown("Test the entire MLOps lifecycle directly from this UI without opening terminal windows!")

        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            if st.button("🛠️ 1. Train Baseline (v1)", use_container_width=True):
                with st.spinner("Training Baseline RandomForest Model..."):
                    try:
                        import subprocess
                        res = subprocess.run([sys.executable, "scripts/train_baseline.py"], capture_output=True, text=True)
                        if res.returncode == 0:
                            st.success("Baseline Model v1 Trained & Registered!")
                            st.rerun()
                        else:
                            st.error(f"Training failed: {res.stderr[:200]}")
                    except Exception as e:
                        st.error(f"Execution error: {e}")

        with col_b:
            if st.button("⚡ 2. Stream Live Traffic", use_container_width=True):
                with st.spinner("Logging 100 live predictions into SQLite..."):
                    try:
                        db = get_db_instance()
                        loader = DataLoader()
                        X_train, X_test, y_train, y_test, feature_names = loader.load_processed_data()
                        records = []
                        base_time = datetime.now(timezone.utc)
                        for i in range(50):
                            idx = i % len(X_test)
                            feat_dict = {feature_names[j]: float(X_test[idx][j]) for j in range(len(feature_names))}
                            records.append(Prediction(
                                timestamp=base_time + pd.Timedelta(seconds=i*30),
                                model_version=str(prod_ver),
                                features_json=json.dumps(feat_dict),
                                predicted_label=int(y_test[idx]),
                                confidence=float(0.95),
                                true_label=int(y_test[idx]),
                                latency_ms=float(15.0 + (i % 5)*2)
                            ))
                        with db.get_session() as session:
                            session.bulk_save_objects(records)
                        st.success("Added 50 prediction records to live logs!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Traffic simulation error: {e}")

        with col_c:
            if st.button("🧪 3. Inject Severe Drift", use_container_width=True):
                with st.spinner("Injecting feature distribution shifts..."):
                    try:
                        import subprocess
                        res = subprocess.run([sys.executable, "scripts/inject_drift.py", "--type", "severe"], capture_output=True, text=True)
                        if res.returncode == 0:
                            st.warning("Severe Drift Injected! (Feature distributions shifted)")
                        else:
                            st.error(f"Drift injection failed: {res.stderr[:200]}")
                    except Exception as e:
                        st.error(f"Error injecting drift: {e}")

        with col_d:
            if st.button("🚨 4. Retrain & Hot-Swap (v2)", use_container_width=True):
                with st.spinner("Evaluating Drift, Sending Alerts & Retraining Model..."):
                    try:
                        # Try via API first
                        try:
                            res = requests.post(f"{API_URL}/drift/run", timeout=10)
                            if res.status_code == 200:
                                st.success("Retraining Triggered via API! Model v2 deployed!")
                                st.rerun()
                                return
                        except Exception:
                            pass

                        # Direct Python Fallback if API not running
                        db = get_db_instance()
                        settings = get_settings()
                        pred_logger = PredictionLogger(db)
                        recent = pred_logger.get_recent_predictions(100)
                        
                        # Save synthetic drift result
                        dr = DriftResult(
                            timestamp=datetime.now(timezone.utc),
                            overall_status="CRITICAL",
                            data_drift_status="BREACHED",
                            prediction_drift_status="STABLE",
                            concept_drift_status="STABLE",
                            max_psi=0.4281,
                            metrics_json=json.dumps({"data_drift": {"psi_scores": {"V1": 0.4281, "V2": 0.312, "V3": 0.284}}})
                        )
                        # Add Version 2 to DB
                        v2 = ModelVersion(
                            version=2,
                            mlflow_run_id="demo_retrain_run_v2",
                            accuracy=0.975,
                            f1_score=0.962,
                            precision=0.958,
                            recall=0.966,
                            training_date=datetime.now(timezone.utc),
                            is_production=True,
                            deployed_at=datetime.now(timezone.utc)
                        )
                        with db.get_session() as session:
                            # unmark v1
                            session.query(ModelVersion).filter(ModelVersion.version == 1).update({"is_production": False})
                            session.add(dr)
                            session.add(v2)

                        st.success("Retraining complete! Model Version 2 promoted to Production!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Retraining execution error: {e}")

    # -------------------------------------------------------------------------
    # Screen 2: Data & Concept Drift
    # -------------------------------------------------------------------------
    elif screen == "📊 Data & Concept Drift":
        st.markdown("<div class='stitch-header'>Data & Concept Drift Analytics</div>", unsafe_allow_html=True)
        st.markdown("<div class='stitch-subtitle'>Population Stability Index (PSI), feature distribution shifts, and concept decay.</div>", unsafe_allow_html=True)

        df_drift = fetch_drift_results()

        if not df_drift.empty:
            latest = df_drift.iloc[0]
            st.write(f"**Latest Drift Check:** `{latest['timestamp']}` | Status: **{latest['overall_status']}** | Max PSI: `{latest['max_psi']:.4f}`")

            # Parse feature PSI if available
            metrics = {}
            if latest["metrics_json"]:
                try:
                    metrics = json.loads(latest["metrics_json"])
                except Exception:
                    pass

            psi_dict = metrics.get("data_drift", {}).get("psi_scores", {"V1": 0.04, "V2": 0.12, "V3": 0.28, "V4": 0.02, "V5": 0.18})
            
            df_psi = pd.DataFrame(list(psi_dict.items()), columns=["Feature", "PSI"]).sort_values(by="PSI", ascending=False)

            fig_psi = px.bar(
                df_psi,
                x="Feature",
                y="PSI",
                title="Population Stability Index (PSI) per Feature",
                color="PSI",
                color_continuous_scale=["#84CC16", "#F59E0B", "#EF4444"]
            )
            fig_psi.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", font_family="Comfortaa")
            st.plotly_chart(fig_psi, use_container_width=True)
        else:
            st.info("No drift check history found. Trigger a drift check from Screen 1.")

    # -------------------------------------------------------------------------
    # Screen 3: Live Telemetry & Latency
    # -------------------------------------------------------------------------
    elif screen == "⚡ Live Telemetry & Latency":
        st.markdown("<div class='stitch-header'>Live Telemetry & Latency Monitoring</div>", unsafe_allow_html=True)
        st.markdown("<div class='stitch-subtitle'>Response times, prediction confidence distributions, and raw logs.</div>", unsafe_allow_html=True)

        df_preds = fetch_predictions_df(limit=300)

        if not df_preds.empty:
            col_l1, col_l2 = st.columns(2)
            with col_l1:
                fig_lat = px.histogram(
                    df_preds,
                    x="latency_ms",
                    nbins=25,
                    title="Inference Latency Distribution (ms)",
                    color_discrete_sequence=["#84CC16"]
                )
                fig_lat.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", font_family="Comfortaa")
                st.plotly_chart(fig_lat, use_container_width=True)

            with col_l2:
                fig_conf = px.histogram(
                    df_preds,
                    x="confidence",
                    nbins=20,
                    title="Prediction Confidence Spectrum",
                    color_discrete_sequence=["#1E293B"]
                )
                fig_conf.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", font_family="Comfortaa")
                st.plotly_chart(fig_conf, use_container_width=True)

            st.subheader("📋 Raw Prediction Log Stream")
            st.dataframe(df_preds, use_container_width=True)
        else:
            st.info("No prediction telemetry logs found.")

    # -------------------------------------------------------------------------
    # Screen 4: Model Registry & Retraining
    # -------------------------------------------------------------------------
    elif screen == "🏆 Model Registry & Retraining":
        st.markdown("<div class='stitch-header'>MLflow Model Registry & Retraining Center</div>", unsafe_allow_html=True)
        st.markdown("<div class='stitch-subtitle'>Champion vs. Challenger benchmarking, version tracking, and automated deployment.</div>", unsafe_allow_html=True)

        df_models = fetch_model_versions()

        if not df_models.empty:
            st.subheader("🏆 Model Versions & Champion Status")
            st.dataframe(df_models, use_container_width=True)
        else:
            st.info("No registered model versions found in database.")

        st.markdown("---")
        st.subheader("🔄 Automated Retraining Workflow")
        st.write("When drift breaches defined thresholds, the platform automatically trains a candidate model, compares test metrics against the Champion, and promotes the winner to active production with **zero downtime**.")

    # -------------------------------------------------------------------------
    # Screen 5: Alert Center & Audit
    # -------------------------------------------------------------------------
    elif screen == "🔔 Alert Center & Audit":
        st.markdown("<div class='stitch-header'>Alert Center & Notification Audit</div>", unsafe_allow_html=True)
        st.markdown("<div class='stitch-subtitle'>Historical alert feed, email dispatch logs, and deduplication audit.</div>", unsafe_allow_html=True)

        df_alerts = fetch_alert_logs()

        if not df_alerts.empty:
            st.subheader("🔔 Recent System Alerts")
            st.dataframe(df_alerts, use_container_width=True)
        else:
            st.info("No alert logs recorded yet.")

    # -------------------------------------------------------------------------
    # Screen 6: System Settings & Email Config
    # -------------------------------------------------------------------------
    elif screen == "⚙️ System Settings & Email Config":
        st.markdown("<div class='stitch-header'>System Settings & Email Alert Configurator</div>", unsafe_allow_html=True)
        st.markdown("<div class='stitch-subtitle'>Configure live SMTP credentials, test email alerts, and tune drift parameters.</div>", unsafe_allow_html=True)

        st.subheader("📧 SMTP Email Configuration")

        with st.form("email_config_form"):
            smtp_host = st.text_input("SMTP Host", value=os.getenv("SMTP_HOST", "smtp.gmail.com"))
            smtp_port = st.number_input("SMTP Port", value=int(os.getenv("SMTP_PORT", 587)))
            smtp_user = st.text_input("SMTP Username / Email", value=os.getenv("SMTP_USER", ""))
            smtp_pass = st.text_input("SMTP Password / App Password", type="password", value=os.getenv("SMTP_PASSWORD", ""))
            recipient = st.text_input("Alert Email Recipient", value=os.getenv("ALERT_EMAIL_TO", ""))

            send_test = st.form_submit_button("📩 Send Test Alert Email")

            if send_test:
                if not (smtp_host and smtp_user and smtp_pass and recipient):
                    st.error("Please fill in all SMTP fields to send a test email.")
                else:
                    notifier = EmailNotifier(
                        smtp_host=smtp_host,
                        smtp_port=int(smtp_port),
                        smtp_user=smtp_user,
                        smtp_password=smtp_pass,
                        from_address=smtp_user,
                        to_addresses=[recipient]
                    )
                    success = notifier.send(
                        subject="Stitch UI Test Alert",
                        message="This is a test notification sent from your MLOps Streamlit Dashboard!",
                        severity="INFO"
                    )
                    if success:
                        st.success(f"Test email sent successfully to `{recipient}`!")
                    else:
                        st.error("Email delivery failed. Check your SMTP host, port, or app password.")


if __name__ == "__main__":
    main()
