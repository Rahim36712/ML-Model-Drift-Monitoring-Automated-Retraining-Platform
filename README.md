# MLOps Model Drift Monitor & Automated Retraining Platform

A production-grade, continuous machine learning lifecycle platform designed to serve predictions, log inference telemetry, monitor feature/prediction/concept drift, alert stakeholders, and trigger automated retraining loops when model quality degrades.

This repository serves as a complete blueprint for an end-to-end self-healing machine learning system in production.

---

## 🚀 System Architecture & Data Flow

The platform is designed around a continuous feedback loop: **Serve ➔ Log ➔ Monitor ➔ Decide ➔ Retrain ➔ Evaluate ➔ Deploy ➔ Observe.**

```mermaid
graph TD
    Client[Transaction Client] -->|1. HTTP POST /predict| API[FastAPI Serving Layer :8000]
    API -->|2. Query Cache| ModelProvider[Model Cache]
    API -->|3. Log telemetry| SQLite[(predictions.db)]
    
    ClientFeedback[Client Feedback Label] -->|HTTP POST /ground-truth/{id}| API
    
    Scheduler[APScheduler / Cron] -->|Trigger Check| DriftManager[Drift Manager]
    SQLite -->|Load sliding window data| DriftManager
    
    DriftManager -->|Compute| DataDrift[Data Drift: PSI & KL]
    DriftManager -->|Compute| PredDrift[Prediction Drift: Hellinger]
    DriftManager -->|Compute| ConceptDrift[Concept Drift: Metrics Drop]
    
    DriftManager -->|Write historical stats| SQLite
    DriftManager -->|Evaluate rules| RetrainEngine[Retraining Decision Engine]
    
    RetrainEngine -->|Drift Alert| AlertManager[Alert Manager]
    AlertManager -->|Dispatch| Console[Console stdout]
    AlertManager -->|Dispatch| Slack[Slack Webhook]
    AlertManager -->|Dispatch| Email[Email Notifier]
    
    RetrainEngine -->|Retrain Trigger| RetrainPipeline[Retraining Pipeline]
    RetrainPipeline -->|Train Model| ModelTrainer[Model Trainer]
    ModelTrainer -->|Log run/metrics/plots| MLflow[(MLflow Registry)]
    
    RetrainPipeline -->|Register & Compare| Evaluator[Model Evaluator]
    Evaluator -->|Champion/Challenger Check| Registry[MLflow Registry Wrapper]
    
    Registry -->|Promote to Production| ModelDeployer[Model Deployer]
    ModelDeployer -->|Atomic Hot-Swap| ModelProvider
    
    Dash[Dash Dashboard :8050] -->|Visualise metrics & history| SQLite
```

### 1. Ingestion & Serving Layer (FastAPI)
* **High-Throughput Inferences:** A FastAPI service handles single (`/predict`) and batch (`/predict/batch`) prediction queries.
* **Low-Latency Cache (`ModelProvider`):** Active models and preprocessors (StandardScaler) are cached in-memory. Telemetry lookup or model re-loading is skipped during predictions.
* **Telemetry Logger (`PredictionLogger`):** Inputs (serialized as JSON), outputs, confidences, model versions, and latency measurements are logged to SQLite immediately.

### 2. Monitoring & Drift Detection Engine
Specialized detectors evaluate recent transaction windows (sliding window size: 500) against training baseline distributions:
* **Covariate Data Drift:** Computes Population Stability Index (PSI) and Kullback-Leibler (KL) Divergence for features (V1–V28 and Amount) to identify changes in user demographics, spend distributions, or data pipelines.
* **Prediction Drift:** Measures shifts in model confidence scores and classification boundaries using Hellinger Distance to detect changes in prediction behavior before target labels are available.
* **Concept Drift:** Calculates metrics (Accuracy, F1-Score, Precision, Recall) using asynchronous ground-truth labels submitted through the `/ground-truth` feedback route.

### 3. Alerting & Decision Engine
* **Alert Manager:** Formats alerts with distinct severities (`WARNING`, `CRITICAL`, `RESOLVED`) and routes them to console, Slack channels, and emails. Features a 30-minute deduplication window to prevent alert fatigue.
* **Decision Engine:** Evaluates computed drift states against rule-based criteria:
  * **Urgency HIGH:** PSI > 0.25 OR F1-score drop > 5% ➔ Triggers retraining pipeline.
  * **Urgency MEDIUM:** Hellinger distance > 0.20 ➔ Flags model for manual review.

### 4. Retraining & Deployer Pipeline
* **Background Retraining:** Gathers recent drifted transactions with ground-truth labels, merges them with the baseline dataset, fits a new Random Forest model, and logs it to MLflow.
* **Champion-Challenger Gating:** Evaluates the new model (challenger) against the active production model (champion) on a fresh test split.
* **Promotion & Deploy:** If the challenger is superior (higher F1 without accuracy degradation), it is registered, promoted to the `"production"` alias in MLflow, and hot-swapped into the FastAPI serving cache atomically without downtime.

---

## 🛠️ Technology Stack & Rationale

* **Python 3.11+:** Chosen for native compatibility with machine learning libraries and MLOps tooling.
* **FastAPI + Uvicorn:** Standard for low-latency, asynchronous ML model serving; provides automatic OpenAPI (Swagger) documentation.
* **scikit-learn:** Chosen for baseline modelling (Random Forest Classifier) and standard dataset processing.
* **MLflow:** Manages experiment tracking, version control, parameters, artifacts, and alias promotion stage gates.
* **Plotly Dash:** Python-native web application framework used to build the monitoring GUI without introducing a React/JS stack.
* **SQLite + SQLAlchemy ORM:** Zero-configuration local database storing prediction logs, alerts, drift trends, and retraining audits.
* **APScheduler:** Lightweight scheduler running inside the serving process to trigger periodic drift evaluations.
* **Pydantic Settings:** Parses YAML configurations with type enforcement and environment variable overrides.

---

## 📂 Repository Structure

```
├── configs/                     # YAML configuration files
│   ├── base_config.yaml         # Serving, database, MLflow, and model parameters
│   ├── drift_thresholds.yaml    # Warning/Critical thresholds for PSI, Hellinger, and F1
│   └── alerting_config.yaml     # SMTP and Slack alert channels details
├── docs/                        # Architecture and specifications
│   ├── architecture.md          # Component layouts and Mermaid diagrams
│   ├── operational_guide.md     # Setup paths (synthetic vs. real data)
│   ├── manual_validation.md     # Step-by-step checklist for human testing
│   └── project_specification.md # Database schemas and component details
├── scripts/                     # Operational automation scripts
│   ├── setup_data.py            # Generates/validates raw csv inputs
│   ├── train_baseline.py        # Initializes DB and deploys production version 1
│   ├── simulate_production.py   # Streams transaction queries and ground-truth labels
│   ├── inject_drift.py          # Interactive script to trigger data/concept drift
│   └── run_demo.py              # Orchestrates serving, UI, traffic, and checks
├── src/                         # Source package
│   ├── alerting/                # Dispatchers for console, Slack webhooks, and SMTP
│   │   ├── alert_manager.py     # Dispatches alerts with deduplication
│   │   ├── slack_notifier.py    # Formats Slack Block Kit payloads
│   │   └── email_notifier.py    # SMTP HTML mail dispatcher
│   ├── api/                     # FastAPI endpoint logic
│   │   ├── app.py               # Lifespan startup, middleware, and scheduler jobs
│   │   ├── middleware.py        # Request timing and structured logs middleware
│   │   ├── schemas.py           # Single/Batch prediction and ground truth validators
│   │   └── routes/              # Health, models, monitoring, and prediction routers
│   ├── config/                  # Configuration loaders
│   │   └── settings.py          # Pydantic Settings overrides parser
│   ├── dashboard/               # Plotly Dash Web Application
│   │   ├── app.py               # Router, CSS layout templates, and index injections
│   │   ├── callbacks/           # Callback files for overview, drift, and performance
│   │   ├── components/          # Navbar, metric cards, badges, and Plotly templates
│   │   └── layouts/             # Overview, drift, performance, alerts, and model pages
│   ├── data/                    # Data managers
│   │   ├── loader.py            # Raw data scaling and reference saving helpers
│   │   └── drift_injector.py    # Injectors for feature shift, scale, noise, and label flip
│   ├── decision/                # Rule-based decision managers
│   │   └── retraining_engine.py # Evaluation rules to flag or trigger retraining
│   ├── models/                  # ML models lifecycle
│   │   ├── trainer.py           # RandomForest training and MLflow artifact logger
│   │   ├── evaluator.py         # Champion/Candidate score comparisons
│   │   └── registry.py          # MLflow Model Registry metadata mapping
│   └── pipeline/                # Workflows
│       ├── retrain_pipeline.py  # Background thread retraining pipeline
│       └── deployer.py          # In-memory ModelProvider cache hot-swapper
└── tests/                       # Unit & integration testing package
    ├── test_data_logger.py      # Telemetry logger tests
    ├── test_drift_detection.py  # PSI, KL, Hellinger, and performance degradation math
    ├── test_decision_engine.py  # Urgency check rules tests
    ├── test_prediction_api.py   # TestClient route validation (predict, batch, health)
    └── test_retraining_pipeline.py # Mock-based retrain workflow triggers tests
```

---

## 🗄️ SQLite Database Schema

The platform maintains five key tables inside the local SQLite database (`data/predictions.db`):

### 1. `predictions`
Tracks every transaction.
```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    model_version VARCHAR NOT NULL,
    features_json TEXT NOT NULL,         -- JSON string of all input features
    predicted_label INTEGER NOT NULL,    -- 0 = Legitimate, 1 = Fraud
    confidence FLOAT NOT NULL,           -- Probability for positive class
    true_label INTEGER DEFAULT NULL,     -- Injected asynchronously via feedback API
    latency_ms FLOAT NOT NULL            -- serving latency in ms
);
```

### 2. `drift_results`
Stores computed metrics to build dashboard historical trends.
```sql
CREATE TABLE drift_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    drift_type VARCHAR NOT NULL,         -- 'data', 'prediction', or 'concept'
    metric_name VARCHAR NOT NULL,        -- 'psi', 'hellinger', 'f1_drop', etc.
    metric_value FLOAT NOT NULL,
    threshold FLOAT NOT NULL,
    is_breached BOOLEAN NOT NULL,
    window_start DATETIME NOT NULL,
    window_end DATETIME NOT NULL,
    details_json TEXT DEFAULT NULL       -- Detailed metrics per feature as JSON
);
```

### 3. `alerts`
Audits dispatched notifications.
```sql
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    severity VARCHAR NOT NULL,           -- 'WARNING', 'CRITICAL', or 'RESOLVED'
    drift_type VARCHAR NOT NULL,
    message TEXT NOT NULL,
    channel VARCHAR NOT NULL,            -- 'console', 'slack', or 'email'
    acknowledged BOOLEAN DEFAULT FALSE
);
```

### 4. `model_versions`
Local index of all models logged in MLflow registry.
```sql
CREATE TABLE model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version INTEGER NOT NULL,
    mlflow_run_id VARCHAR NOT NULL,
    accuracy FLOAT NOT NULL,
    f1_score FLOAT NOT NULL,
    precision FLOAT NOT NULL,
    recall FLOAT NOT NULL,
    auc_roc FLOAT,
    training_date DATETIME NOT NULL,
    is_production BOOLEAN DEFAULT FALSE,
    deployed_at DATETIME DEFAULT NULL
);
```

### 5. `retraining_events`
Audit trail of retraining execution history.
```sql
CREATE TABLE retraining_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    trigger_reason TEXT NOT NULL,
    old_version INTEGER,
    new_version INTEGER DEFAULT NULL,
    old_f1 FLOAT,
    new_f1 FLOAT DEFAULT NULL,
    status VARCHAR NOT NULL             -- 'STARTED', 'COMPLETED', 'FAILED', or 'REJECTED'
);
```

---

## ⚙️ Environment Configuration

Copy the template `.env.example` to `.env` to override configuration properties.

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db

# API Host & Port (Serving Layer)
API_HOST=0.0.0.0
API_PORT=8000

# Dashboard Host & Port (Plotly Dash)
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8050

# Slack Integration (Optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR_WORKSPACE_ID/YOUR_CHANNEL_ID/YOUR_TOKEN

# SMTP Email Alert Details (Optional)
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
ALERT_EMAIL_TO=
```

---

## 🚀 Setup & Execution

### 1. Installation
Ensure Python 3.11+ is installed.

```bash
# Clone the repository
git clone https://github.com/Rahim36712/ML-Model-Drift-Monitoring-Automated-Retraining-Platform.git
cd ML-Model-Drift-Monitoring-Automated-Retraining-Platform

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Path A: Synthetic Demo (Zero-Dependency Setup)
The demo trains a baseline model, seeds historical records, starts prediction traffic, injects drift, alerts, and retrains automatically.

```bash
# Run the demo orchestrator
python scripts/run_demo.py
```
* **Dashboard:** Open `http://localhost:8050`
* **Serving API:** Check docs at `http://localhost:8000/docs`

### 3. Path B: Real-Data Production Setup
To use the actual Kaggle dataset:
1. Download `creditcard.csv` from [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Save it to `data/raw/creditcard.csv`.
3. Execute baseline training and start services:

```bash
# Train baseline model (version 1)
python scripts/train_baseline.py

# Start serving API (port 8000)
python -m uvicorn src.api.app:create_app --host 127.0.0.1 --port 8000 --factory

# Start dashboard (port 8050)
python -m src.dashboard.app

# Start traffic simulator
python scripts/simulate_production.py
```

### 4. Running Unit Tests
Validate mathematics, API routes, database loggers, and decision rules:
```bash
pytest tests/ -v
```

---

## 🔒 Production Deployment Blueprint

When transitioning from this local architecture to an enterprise environment, implement the following architectural upgrades:

```
┌────────────────────────────────────────────────────────────────────────┐
│                              PRODUCTION                                │
├───────────────────┬────────────────────────────────────────────────────┤
│ Component         │ Production Upgrade                                 │
├───────────────────┼────────────────────────────────────────────────────┤
│ Telemetry Store   │ Migrate SQLite database to PostgreSQL/RDS.          │
│ Model Registry    │ Host MLflow Tracking Server on AWS ECS/EKS using    │
│                   │ S3 backend storage.                                │
│ Retrain Engine    │ Move periodic Cron/APScheduler to Apache Airflow    │
│                   │ or Prefect DAG workflows.                           │
│ API Cluster       │ Deploy FastAPI container with Gunicorn/Uvicorn     │
│                   │ behind Nginx / ALB load balancer.                  │
│ Observability     │ Send telemetry metrics to Prometheus & Grafana.    │
└───────────────────┴────────────────────────────────────────────────────┘
```

---

## 🛠️ Known Issues & Future Work
* **Concept Drift Delay:** Concept drift relies on ground-truth labels. In real credit card transactions, chargebacks and fraud reports can take 30–90 days to appear. Future work should implement an *asymmetric delayed feedback evaluator* to simulate real-world label arrival times.
* **Model Explainability (SHAP):** Adding SHAP or LIME outputs to the prediction logger would allow the dashboard to display *why* the model is drifting (e.g. shifts in feature importances over time).
