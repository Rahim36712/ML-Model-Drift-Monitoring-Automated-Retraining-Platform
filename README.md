# 🤖 Self-Healing MLOps Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Plotly Dash](https://img.shields.io/badge/Plotly_Dash-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://dash.plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> A production-ready MLOps system that monitors machine learning models in real-time, detects data & accuracy drift, alerts stakeholders, and automatically retrains and hot-swaps new models with **zero downtime**.

---

## ⚡ What It Does

- 📡 **Real-Time Telemetry & Inference:** Low-latency model serving via FastAPI with in-memory caching and SQLite log tracking.
- 📊 **3D Drift Monitoring:** Detects **Data Drift** (PSI & KL), **Prediction Drift** (Hellinger Distance), and **Concept Drift** (F1/Accuracy drops).
- 🔔 **Multi-Channel Alerting:** Instant notifications on Slack, Email, and Console with anti-spam deduplication.
- 🔄 **Self-Healing Retraining:** Automatically retrains models in the background when drift thresholds are breached, benchmarks **Champion vs. Challenger**, and hot-swaps the winner live.
- 📈 **Interactive Control Center:** Dark-themed Plotly Dash dashboard for live monitoring and model registry tracking.

---

## 🏗️ System Flow

```
┌─────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ Live Client │ ────► │ FastAPI Server  │ ────► │ SQLite Database │
└─────────────┘       └────────┬────────┘       └─────────────────┘
                               │
                               ▼ (Periodic Drift Check)
                      ┌─────────────────┐
                      │  Drift Engine   │ (PSI / Hellinger / Concept Drift)
                      └────────┬────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
 ┌──────────────────────┐              ┌──────────────────────┐
 │  Alerts (Slack/Email)│              │ Retrain & Evaluate   │
 └──────────────────────┘              └──────────┬───────────┘
                                                  │ (MLflow)
                                                  ▼
                                       ┌──────────────────────┐
                                       │ Zero-Downtime Swap   │
                                       └──────────────────────┘
```

---

## 🚀 Quick Start (Interactive Demo)

Run the entire self-healing pipeline with a single command:

```bash
# Clone repository
git clone https://github.com/Rahim36712/ML-Model-Drift-Monitoring-Automated-Retraining-Platform.git
cd ML-Model-Drift-Monitoring-Automated-Retraining-Platform

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run demo orchestrator
python scripts/run_demo.py
```

- 📊 **Dashboard:** `http://localhost:8050`
- ⚡ **API Documentation:** `http://localhost:8000/docs`

---

## 🧪 Real-Data Setup (Kaggle Dataset)

To run with real data, download `creditcard.csv` from Kaggle into `data/raw/creditcard.csv`, then run:

```bash
# 1. Train baseline model
python scripts/train_baseline.py

# 2. Start serving API
python -m uvicorn src.api.app:create_app --host 127.0.0.1 --port 8000 --factory

# 3. Start monitoring dashboard
python -m src.dashboard.app

# 4. Stream production traffic
python scripts/simulate_production.py
```

---

## 📂 Project Overview

```
├── configs/          # YAML configuration files (thresholds, serving, alerts)
├── docs/             # Architecture diagrams & developer guides
├── scripts/          # Automation scripts (demo, traffic simulation, drift injection)
├── src/
│   ├── alerting/     # Slack, Email & Console notification dispatchers
│   ├── api/          # FastAPI inference endpoints & telemetry middleware
│   ├── dashboard/    # Plotly Dash web application
│   ├── data/         # SQLite ORM, data loader & drift injectors
│   ├── decision/     # Retraining trigger rules engine
│   ├── models/       # Model trainer, evaluator & MLflow registry
│   └── pipeline/     # Background retraining pipeline & deployer
└── tests/            # Test suite for math, API, and retraining logic
```

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for details.
