# MLOps Model Drift Monitor & Automated Retraining Platform

A production-grade, continuous machine learning lifecycle platform that serves predictions, logs inference telemetry, monitors feature and concept drift, alerts stakeholders, and triggers automated model retraining pipelines when performance degrades.

---

## 🚀 Key Features

* **Real-time API Serving:** Low-latency FastAPI endpoint serving RandomForest inferences.
* **Prediction Telemetry Logging:** Unified SQLite logs tracking prediction inputs, confidence scores, latencies, and asynchronous true label feedback.
* **Sliding-Window Drift Detection:**
  * **Data Drift:** Population Stability Index (PSI) & Kullback-Leibler (KL) Divergence.
  * **Prediction Drift:** Hellinger Distance on classification confidence distributions.
  * **Concept Drift:** Degradation in Accuracy, F1-Score, Precision, and Recall.
* **Interactive Alerting:** Dispatches warnings and critical notifications to stdout, Slack hooks, and SMTP email.
* **Self-Healing Retraining Loop:** Automated retraining combining historical and current drifted data, logging models and Plotly confusion matrices to MLflow, comparing candidates, and promoting/hot-swapping production models.
* **Modern Monitoring Dashboard:** Responsive dark-themed Plotly Dash UI showcasing metrics, heatmaps, alerts, and model histories.

---

## 🛠️ Installation & Setup

1. **Clone the repository and enter workspace:**
   ```bash
   cd MLOPS
   ```

2. **Set up virtual environment & install requirements:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Copy `.env.example` to `.env` and fill in optional Slack/SMTP details if needed:
   ```bash
   cp .env.example .env
   ```

---

## 🎮 Running the Interactive Demo

The platform provides a complete orchestration demo that sets up raw datasets, trains the baseline model, pre-populates prediction histories, starts FastAPI and Dash processes, runs transaction traffic, and allows you to inject drift interactively.

To launch the demo:
```bash
python scripts/run_demo.py
```

### What happens in the demo:
1. **Initial Setup:** Database is initialized, and `scripts/train_baseline.py` runs (training the baseline model, registering version 1, and setting it to production).
2. **Pre-population:** Database is pre-seeded with 150 historical predictions to fill the dashboard charts immediately.
3. **Services Start:** FastAPI Serving API starts on `http://localhost:8000`, Dash Dashboard starts on `http://localhost:8050`, and a simulated transaction traffic stream begins.
4. **Drift Injection:** The prompt will wait for you to press **ENTER**. When pressed, it injects severe data and concept drift, runs a manual drift check, alerts the system, and launches the background retraining pipeline which promotes a new, corrected model version.

---

## 🧪 Running Unit Tests

We have implemented a comprehensive test suite (16 tests) covering data logging, drift mathematics, API schemas, decision engines, and pipeline orchestrations:

To run all tests:
```bash
pytest tests/
```

---

## 📁 Repository Structure

```
├── configs/                  # YAML configurations (model, database, thresholds, alerts)
├── data/                     # Raw, processed, and reference distributions
├── docs/                     # Platform architecture diagrams and documentation
├── scripts/                  # Demo execution and training automation
│   ├── train_baseline.py     # Initial model trainer
│   ├── setup_data.py         # Directory and synthetic dataset generator
│   ├── simulate_production.py# Real-time transaction sender
│   ├── inject_drift.py       # Configures synthetic drift
│   └── run_demo.py           # Unified platform orchestrator
├── src/                      # Source modules
│   ├── alerting/             # Slack, Email, and Console alert managers
│   ├── api/                  # FastAPI app, routers, and schemas
│   ├── config/               # Settings parser and env overrides
│   ├── dashboard/            # Plotly Dash web UI (layouts, callbacks)
│   ├── data/                 # Logging, loading, and drift injection
│   ├── decision/             # Retraining rule engine
│   ├── models/               # Trainer, evaluator, and MLflow registry
│   └── pipeline/             # Retraining pipeline and model deployer
└── tests/                    # Comprehensive unit and integration test suite
```
