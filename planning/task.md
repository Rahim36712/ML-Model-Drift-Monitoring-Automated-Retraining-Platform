# MLOps Platform - Task Tracker

## Phase 1: Foundation & Data Infrastructure
- [x] Project setup (pyproject.toml, requirements.txt, .gitignore, .env.example)
- [x] Configuration system (configs/*.yaml + src/config/settings.py)
- [x] Database schema & logger (src/data/database.py, src/data/logger.py)
- [x] Data loader & preprocessor (src/data/loader.py)
- [x] Drift injector (src/data/drift_injector.py)

## Phase 2: Model Training & MLflow
- [x] Model trainer (src/models/trainer.py)
- [x] Model evaluator (src/models/evaluator.py)
- [x] Model registry wrapper (src/models/registry.py)
- [ ] Baseline training script (scripts/train_baseline.py)

## Phase 3: Drift Detection Engine
- [/] Data drift detector - PSI & KL (src/monitoring/data_drift.py)
- [/] Prediction drift detector - Hellinger (src/monitoring/prediction_drift.py)
- [/] Concept drift detector (src/monitoring/concept_drift.py)
- [/] Drift manager & scheduler (src/monitoring/drift_manager.py)

## Phase 4: Alerting System
- [/] Alert manager (src/alerting/alert_manager.py)
- [/] Slack notifier (src/alerting/slack_notifier.py)
- [/] Email notifier (src/alerting/email_notifier.py)

## Phase 5: Retraining Decision Engine
- [ ] Decision engine (src/decision/retraining_engine.py)

## Phase 6: Retraining Pipeline
- [ ] Retraining pipeline (src/pipeline/retrain_pipeline.py)
- [ ] Model deployer (src/pipeline/deployer.py)

## Phase 7: FastAPI Prediction API
- [ ] API app factory (src/api/app.py)
- [ ] Schemas (src/api/schemas.py)
- [ ] Middleware (src/api/middleware.py)
- [ ] Prediction routes (src/api/routes/predictions.py)
- [ ] Monitoring routes (src/api/routes/monitoring.py)
- [ ] Model routes (src/api/routes/models.py)
- [ ] Health route (src/api/routes/health.py)

## Phase 8: Plotly Dash Dashboard
- [ ] Dashboard app factory (src/dashboard/app.py)
- [ ] Navbar component (src/dashboard/components/navbar.py)
- [ ] Metric card component (src/dashboard/components/metric_card.py)
- [ ] Trend chart component (src/dashboard/components/trend_chart.py)
- [ ] Alert badge component (src/dashboard/components/alert_badge.py)
- [ ] Overview page (src/dashboard/layouts/overview.py)
- [ ] Data drift page (src/dashboard/layouts/data_drift.py)
- [ ] Prediction drift page (src/dashboard/layouts/prediction_drift.py)
- [ ] Performance page (src/dashboard/layouts/performance.py)
- [ ] Alerts page (src/dashboard/layouts/alerts.py)
- [ ] Model registry page (src/dashboard/layouts/model_registry.py)
- [ ] Dashboard callbacks (src/dashboard/callbacks/*.py)

## Phase 9: Demo & Simulation
- [ ] Setup data script (scripts/setup_data.py)
- [ ] Simulate production script (scripts/simulate_production.py)
- [ ] Inject drift script (scripts/inject_drift.py)
- [ ] Run demo script (scripts/run_demo.py)

## Phase 10: Testing & Documentation
- [ ] Test drift detection (tests/test_drift_detection.py)
- [ ] Test prediction API (tests/test_prediction_api.py)
- [ ] Test retraining pipeline (tests/test_retraining_pipeline.py)
- [ ] Test decision engine (tests/test_decision_engine.py)
- [ ] Test data logger (tests/test_data_logger.py)
- [ ] README.md
- [ ] Architecture docs (docs/architecture.md)
