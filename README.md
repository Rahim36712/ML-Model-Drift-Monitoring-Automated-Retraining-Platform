# 🤖 Self-Healing MLOps Platform: Model Drift Monitoring & Automated Retraining

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Plotly Dash](https://img.shields.io/badge/Plotly_Dash-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://dash.plotly.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> **An enterprise-ready, production-grade MLOps system that monitors machine learning models in real-time, detects when they start failing or degrading (drift), alerts teams instantly, and automatically retrains & hot-swaps better models with ZERO downtime.**

---

## 💡 What Problem Does This Solve? (In Simple Words)

When you deploy an AI/ML model to production (like a Fraud Detection model), it works great on day one. But over time, the real world changes:
- 💳 **User behavior shifts** (e.g., spending habits during holidays).
- 📈 **Data inputs change** (e.g., new app versions send different scale numbers).
- 📉 **Model accuracy drops quietly** because it was trained on old patterns.

This phenomenon is called **Model Drift**. In traditional systems, model drift goes unnoticed for months, costing businesses millions in bad decisions.

### 🛡️ The Solution: A Self-Healing System
This project provides an **autonomous autopilot** for live machine learning models. It constantly watches live prediction traffic, measures statistical health, triggers real-time alerts on Slack/Email when anomalies appear, trains a new "challenger" model in the background, and seamlessly replaces the live model without restarting your server!

---

## ✨ Key Features at a Glance

| Feature | Easy Explanation | Tech Used |
| :--- | :--- | :--- |
| ⚡ **Zero-Latency Serving** | High-throughput prediction endpoints with in-memory model caching so live requests stay under 5ms. | FastAPI, Uvicorn, Python memory cache |
| 📡 **Live Telemetry Logging** | Automatically logs every incoming prediction request, output confidence, features, and latency into a database. | SQLAlchemy, SQLite |
| 📊 **3D Drift Radar** | Detects 3 types of drift: **Data shifts** (PSI/KL), **Prediction score shifts** (Hellinger), & **Accuracy drops** (Concept drift). | SciPy, NumPy, scikit-learn |
| 🔔 **Smart Multi-Channel Alerts** | Sends rich alerts to Slack, Email, and Console with a 30-minute anti-spam window so your inbox isn't flooded. | Slack Webhooks, SMTP HTML Mail |
| 🤖 **Autonomous Retraining** | Automatically triggers background retraining whenever model health drops below safety thresholds. | MLflow, Threading, scikit-learn |
| 🥊 **Champion vs. Challenger** | Benchmark the new retrained model against the active live model. Only promotes the winner! | MLflow Model Registry |
| 🔄 **Zero-Downtime Hot-Swap** | Replaces the live model in memory instantly without taking down the serving server or losing requests. | In-Memory Atomic Swap |
| 💻 **Real-Time Dash Control Center** | A dark-themed web dashboard for real-time monitoring of metrics, drift charts, and model history. | Plotly Dash, Dash Bootstrap Components |
| 🧪 **Built-in Drift Simulator** | Interactive tool to artificially inject feature noise or label flips and watch the system self-heal live! | Python CLI Automation |

---

## 🏗️ System Architecture & Continuous Feedback Loop

```mermaid
graph TD
    Client[📱 Live Client Requests] -->|1. POST /predict| API[⚡ FastAPI Serving Layer :8000]
    API -->|2. Fast Lookup| Cache[🧠 In-Memory Model Cache]
    API -->|3. Log Telemetry| DB[(🗄️ SQLite Database)]
    
    Feedback[🏷️ Ground-Truth Labels] -->|POST /ground-truth/{id}| API
    
    Scheduler[⏰ APScheduler / Cron] -->|Trigger Routine Check| DriftEngine[🔍 Drift Monitoring Engine]
    DB -->|Fetch Sliding Window| DriftEngine
    
    DriftEngine -->|Data Drift| PSI[📊 Feature Drift: PSI & KL]
    DriftEngine -->|Prediction Drift| Hellinger[📈 Prediction Drift: Hellinger]
    DriftEngine -->|Concept Drift| Accuracy[📉 Concept Drift: F1/Recall Drop]
    
    DriftEngine -->|Record History| DB
    DriftEngine -->|Evaluate Rules| DecisionEngine[🧠 Retraining Decision Engine]
    
    DecisionEngine -->|Threshold Breached| AlertMgr[🔔 Alert Manager]
    AlertMgr -->|Dispatch| Slack[💬 Slack Webhook]
    AlertMgr -->|Dispatch| Email[📧 Email Notifications]
    AlertMgr -->|Dispatch| Terminal[🖥️ Console Logs]
    
    DecisionEngine -->|Trigger Auto-Retrain| Pipeline[🔄 Retraining Pipeline]
    Pipeline -->|Fit New Model| Trainer[🤖 scikit-learn Trainer]
    Trainer -->|Log Artifacts & Metrics| MLflow[(📦 MLflow Model Registry)]
    
    Pipeline -->|Champion vs. Challenger| Evaluator[⚖️ Model Evaluator]
    Evaluator -->|Challenger Wins!| Registry[🏷️ MLflow Registry Promoter]
    
    Registry -->|Promote to Production| Deployer[🚀 Hot-Swap Deployer]
    Deployer -->|Atomic In-Memory Replacement| Cache
    
    Dash[📊 Dash Dashboard :8050] -->|Read Stats & Charts| DB
```

---

## ⚡ Quick Start Guide (Run in 2 Minutes!)

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Rahim36712/ML-Model-Drift-Monitoring-Automated-Retraining-Platform.git
cd ML-Model-Drift-Monitoring-Automated-Retraining-Platform

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Zero-Dependency Interactive Demo 🚀

Experience the complete self-healing lifecycle with **a single command**:

```bash
python scripts/run_demo.py
```

What this script automatically does for you:
1. 🏋️ Trains initial Baseline Model (V1) & deploys it.
2. 🗄️ Pre-populates historical transaction logs for rich charts.
3. ⚡ Launches **FastAPI Server** at `http://localhost:8000` (Docs at `/docs`).
4. 📊 Launches **Dash Monitoring Dashboard** at `http://localhost:8050`.
5. 🚗 Starts a **Live Traffic Simulator** streaming transactions.
6. 🧪 Prompts you interactively to **Inject Data Drift** and watch the platform detect drift, alert, and auto-retrain live!

---

## 🔬 Real-Data Production Setup (Kaggle Dataset)

If you want to run this platform using the real-world **Kaggle Credit Card Fraud Dataset**:

1. Download `creditcard.csv` from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Place the file inside `data/raw/creditcard.csv`.
3. Execute the full production pipeline:

```bash
# Step 1: Train Baseline Model V1 & register in MLflow
python scripts/train_baseline.py

# Step 2: Start live FastAPI inference server (Port 8000)
python -m uvicorn src.api.app:create_app --host 127.0.0.1 --port 8000 --factory

# Step 3: Start Dash Web Dashboard (Port 8050)
python -m src.dashboard.app

# Step 4: Start production traffic simulator
python scripts/simulate_production.py
```

---

## 🧠 How Drift Detection & Auto-Healing Work

### 1. 📊 Feature Data Drift (PSI & KL Divergence)
- **Population Stability Index (PSI):** Measures shift between training baseline feature distribution and current production traffic (sliding window of 500 samples).
  - `PSI < 0.1` 🟢 Normal (No Drift)
  - `0.1 <= PSI <= 0.25` 🟡 Moderate Drift (Warning Alert)
  - `PSI > 0.25` 🔴 Significant Drift (Triggers Automated Retraining)

### 2. 📈 Prediction Drift (Hellinger Distance)
- Measures changes in predicted probability distributions (confidence scores) before ground-truth labels arrive.
  - `Hellinger > 0.20` 🟡 Model behavior shifted drastically. Flags model for review.

### 3. 📉 Concept Drift (F1-Score & Accuracy Drop)
- When delayed ground-truth feedback labels (`/ground-truth/{id}`) arrive, the platform compares predicted labels against real labels.
  - `F1 Drop > 5%` 🔴 Model quality degraded! Triggers immediate auto-retraining.

---

## 🥊 Champion vs. Challenger Retraining Gating

When drift triggers retraining, the system doesn't blindly trust the new model:

1. **Merge & Train:** Combines current baseline data with recent drifted transactions and trains a new Random Forest model.
2. **Benchmark:** Evaluates the new model (**Challenger**) against the live active model (**Champion**) on a fresh holdout evaluation dataset.
3. **Promotion Rule:**
   $$\text{F1}_{\text{Challenger}} > \text{F1}_{\text{Champion}} \quad \text{AND} \quad \text{Accuracy}_{\text{Challenger}} \ge \text{Accuracy}_{\text{Champion}} - 1\%$$
4. **Atomic Hot-Swap:** If passed, MLflow registers the model, assigns the `"production"` alias, and updates the in-memory FastAPI server instantly without interrupting live requests!

---

## 📊 Interactive Web Dashboard Overview

The **Plotly Dash** control center runs on `http://localhost:8050` and provides 5 dedicated views:

- 📈 **Overview Dashboard:** Live KPI cards (Total predictions, drift status, active model version, alert counter) + real-time latency & volume charts.
- 🔍 **Data Drift Radar:** Interactive PSI bar charts across all features (V1–V28 & Amount) and distribution overlays.
- 🎯 **Model Performance & Concept Drift:** F1, Precision, Recall, Accuracy, and ROC AUC metrics over time as feedback labels arrive.
- 🔔 **Alert Center:** Historical log of warnings, critical alerts, and resolutions across Slack, Email, and Console.
- 📦 **Model Registry:** Audit trail of all MLflow champion/challenger model versions, retraining triggers, and hot-swap events.

---

## 🛠️ Repository Structure

```
├── configs/                     # YAML Configuration files
│   ├── base_config.yaml         # Serving, database, MLflow, and model hyperparameters
│   ├── drift_thresholds.yaml    # Alert & retraining thresholds (PSI, Hellinger, F1 drop)
│   └── alerting_config.yaml     # Slack webhook & SMTP email settings
├── docs/                        # Specifications & Developer Guides
│   ├── architecture.md          # Detailed diagrams & component layout
│   ├── operational_guide.md     # Production operational runbook
│   └── manual_validation.md     # Step-by-step human testing checklist
├── scripts/                     # Operational & Demo Automation
│   ├── run_demo.py              # 🚀 1-Click interactive demo launcher
│   ├── train_baseline.py        # Initializes DB & trains baseline Model V1
│   ├── simulate_production.py   # Streams real-time transaction traffic
│   ├── inject_drift.py          # Interactive drift injection tool
│   └── setup_data.py            # Dataset preparation utility
├── src/                         # Core Python Package
│   ├── alerting/                # Multi-channel alert dispatchers (Console, Slack, Email)
│   ├── api/                     # FastAPI endpoints, middleware & validation schemas
│   ├── config/                  # Pydantic Settings & YAML loader
│   ├── dashboard/               # Plotly Dash web application & callbacks
│   ├── data/                    # Data loaders, SQLite ORM, & drift injectors
│   ├── decision/                # Rule-based decision engine for retraining triggers
│   ├── models/                  # Trainer, Champion/Challenger Evaluator, MLflow Registry
│   └── pipeline/                # Retraining background workflow & atomic deployer
└── tests/                       # Complete Unit & Integration Test Suite
```

---

## ⚙️ Environment Variables (`.env`)

Copy `.env.example` to `.env` to configure your environment:

```env
# MLflow Experiment Tracking
MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db

# Serving Layer
API_HOST=0.0.0.0
API_PORT=8000

# Dashboard
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8050

# Slack Integration (Optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Email Alerts (Optional SMTP)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL_TO=alerts@company.com
```

---

## 🧪 Running Tests

Validate math algorithms, FastAPI endpoints, SQLite database loggers, and retraining triggers with `pytest`:

```bash
pytest tests/ -v
```

---

## ☁️ Enterprise Scale-Up Roadmap

To transition this platform from local execution to high-scale enterprise cloud infrastructure:

| Component | Local Implementation | Enterprise Cloud Upgrade |
| :--- | :--- | :--- |
| **Telemetry Database** | Local SQLite | AWS RDS PostgreSQL / Cloud SQL |
| **Model Registry** | Local SQLite/MLflow file store | AWS S3 backend + MLflow Server on Kubernetes (EKS/ECS) |
| **Workflow Engine** | In-process APScheduler & Threading | Apache Airflow / Prefect / Kubeflow DAGs |
| **API Serving Layer** | Single FastAPI instance | Gunicorn + Uvicorn workers behind AWS ALB / Nginx |
| **Observability** | Plotly Dash Dashboard | Prometheus + Grafana dashboards + Datadog telemetry |

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

<p align="center">
  Made with ❤️ for MLOps Engineers, Data Scientists, and AI System Builders.
</p>
