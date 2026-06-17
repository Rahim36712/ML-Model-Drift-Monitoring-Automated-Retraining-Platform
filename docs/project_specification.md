# Project Specification: ML Model Drift Monitor & Automated Retraining Platform

This document outlines the scope, file directory layout, component purposes, and operational metrics of the MLOps platform.

---

## 1. Project Objectives

The platform solves the silent degradation problem of machine learning models in production:
* **Prediction Telemetry:** Tracks predictions, latencies, confidence metrics, and inputs.
* **Continuous Auditing:** Computes data drift (PSI/KL), prediction drift (Hellinger), and concept drift (accuracy/F1 drop) in a sliding window.
* **Proactive Alerting:** Dispatches severity-rated alerts to Slack, Email, and logs.
* **Closed-Loop Retraining:** Detects degradation, retrains models on the combined historical and drifted data, evaluates candidates, and hot-swaps the active model atomically.

---

## 2. Directory & File List

```
├── configs/                     # YAML configuration files
│   ├── base_config.yaml         # Serving, database, MLflow, and model parameters
│   ├── drift_thresholds.yaml    # Warning/Critical thresholds for PSI, Hellinger, and F1
│   └── alerting_config.yaml     # SMTP and Slack alert channels details
├── docs/                        # Architecture and specifications
│   ├── architecture.md          # Component layouts and Mermaid diagrams
│   └── project_specification.md # Project listing and file scopes
├── scripts/                     # Operational automation scripts
│   ├── setup_data.py            # Generates/validates raw csv inputs
│   ├── train_baseline.py        # Initializes DB and deploys production version 1
│   ├── simulate_production.py   # Streams transaction queries and ground-truth labels
│   ├── inject_drift.py          # Interactive script to trigger data/concept drift
│   └── run_demo.py              # Orchestrates serving, UI, traffic, and checks
├── src/                         # Source package
│   ├── alerting/                # Dispatchers for console, Slack webhooks, and SMTP
│   │   ├── alert_manager.py     # Dispatches alerts with a 30-minute deduplication window
│   │   ├── slack_notifier.py    # Formats Slack Block Kit JSON payloads
│   │   └── email_notifier.py    # Constructs HTML templates and sends via TLS
│   ├── api/                     # FastAPI endpoint logic
│   │   ├── app.py               # Lifespan startup, middleware, and scheduler jobs
│   │   ├── middleware.py        # Request timing and structured logs middleware
│   │   ├── schemas.py           # Single/Batch prediction and ground truth validators
│   │   └── routes/              # Health, models, monitoring, and prediction routers
│   ├── config/                  # Configuration loaders
│   │   └── settings.py          # Pydantic Settings parser with environment variable overrides
│   ├── dashboard/               # Plotly Dash Web Application
│   │   ├── app.py               # Router, CSS layout templates, and index injections
│   │   ├── callbacks/           # Callback files for overview, drift, and performance
│   │   ├── components/          # Navbar, metric cards, badges, and Plotly templates
│   │   └── layouts/             # Overview, drift, performance, alerts, and model pages
│   ├── data/                    # Data managers
│   │   ├── loader.py            # Raw data scaling, reference saving, and loader helpers
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
├── tests/                       # Unit & integration testing package
│   ├── test_data_logger.py      # Telemetry logger tests
│   ├── test_drift_detection.py  # PSI, KL, Hellinger, and performance degradation math
│   ├── test_decision_engine.py  # Urgency check rules tests
│   ├── test_prediction_api.py   # TestClient route validation (predict, batch, health)
│   └── test_retraining_pipeline.py # Mock-based retrain workflow triggers tests
```

---

## 3. SQLite Database Schema

### Table: `predictions`
Tracks every transaction. Features are serialized as JSON.
* `id` (Integer PK), `timestamp` (DateTime), `model_version` (String), `features_json` (Text), `predicted_label` (Integer), `confidence` (Float), `true_label` (Integer Nullable), `latency_ms` (Float)

### Table: `drift_results`
Stores computed metrics to build dashboard historical trends.
* `id` (Integer PK), `timestamp` (DateTime), `drift_type` (String), `metric_name` (String), `metric_value` (Float), `threshold` (Float), `is_breached` (Boolean), `window_start` (DateTime), `window_end` (DateTime), `details_json` (Text)

### Table: `alerts`
Audits dispatched notifications.
* `id` (Integer PK), `timestamp` (DateTime), `severity` (String), `drift_type` (String), `message` (Text), `channel` (String), `acknowledged` (Boolean)

### Table: `model_versions`
Local index of all models logged in MLflow registry.
* `id` (Integer PK), `version` (Integer), `mlflow_run_id` (String), `accuracy` (Float), `f1_score` (Float), `precision` (Float), `recall` (Float), `auc_roc` (Float), `training_date` (DateTime), `is_production` (Boolean), `deployed_at` (DateTime Nullable)

### Table: `retraining_events`
Audit trail of retraining execution history.
* `id` (Integer PK), `timestamp` (DateTime), `trigger_reason` (Text), `old_version` (Integer), `new_version` (Integer Nullable), `old_f1` (Float), `new_f1` (Float Nullable), `status` (String)
