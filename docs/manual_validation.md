# Manual Validation Checklist

This document contains step-by-step test cases that must be **manually verified** by a human operator. Each test has clear pass/fail criteria.

> [!NOTE]
> Run through these tests **after** completing the operational setup (either Path A or Path B from the Operational Guide). Run the automated tests first with `pytest tests`; this checklist covers things that **cannot be fully automated**.

---

## Pre-Test Setup

Before starting, choose your path:

- **Quick Demo (Path A):** Run `python scripts/run_demo.py` — everything starts automatically
- **Manual Setup (Path B):** Start each service individually (see Operational Guide)

---

## Test 1: Environment & Dependencies

| # | Check | How to Verify | Pass Criteria |
|---|---|---|---|
| 1.1 | Python version | `python --version` | 3.11 or higher |
| 1.2 | Dependencies installed | `pip list \| findstr mlflow` | mlflow, scikit-learn, dash, fastapi all present |
| 1.3 | Config files exist | Check `configs/` directory | `base_config.yaml`, `drift_thresholds.yaml`, `alerting_config.yaml` all present |

**Result:** ☐ PASS / ☐ FAIL

---

## Test 2: Baseline Model Training

> Run this test if you used Path B. Skip if using `run_demo.py` (it trains automatically).

| # | Check | How to Verify | Pass Criteria |
|---|---|---|---|
| 2.1 | Training completes | `python scripts/train_baseline.py` | Script finishes without errors, prints metrics |
| 2.2 | MLflow artifacts | Open `mlruns/` directory | Run directory exists with metrics, params, and artifacts subdirectories |
| 2.3 | Reference distributions | Check `data/reference/` | `.npy` files exist (one per feature + combined) |
| 2.4 | SQLite model record | Open `data/predictions.db` with any SQLite viewer | `model_versions` table has 1 row with `is_production = 1` |
| 2.5 | Model metrics reasonable | Read console output | F1 > 0.70, Accuracy > 0.95 (synthetic) or F1 > 0.80 (real data) |

**Result:** ☐ PASS / ☐ FAIL

---

## Test 3: FastAPI Serving Layer

Start the API if not already running:
```bash
python -m uvicorn src.api.app:create_app --host 127.0.0.1 --port 8000 --factory
```

| # | Check | How to Verify | Pass Criteria |
|---|---|---|---|
| 3.1 | Service health endpoint | Open `http://localhost:8000/health` in browser | Returns JSON with `status` equal to `ok` or `degraded` |
| 3.2 | API docs | Open `http://localhost:8000/docs` | Swagger UI loads with all endpoints listed |
| 3.3 | Single prediction | Send a POST request (see below) | Returns JSON with `prediction_id`, `predicted_label` (0 or 1), `confidence` (0.0–1.0) |
| 3.4 | Prediction logged | Check `http://localhost:8000/predictions/stats` | `total_count` incremented by 1 |
| 3.5 | Ground truth feedback | POST to `/ground-truth/{id}` with `{"true_label": 0}` | Returns 200 OK |
| 3.6 | Model info | GET `http://localhost:8000/models/current` | Returns model version and production status |

**Test 3.3 — Sample prediction request:**
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"features\": {\"V1\": -1.36, \"V2\": -0.07, \"V3\": 2.54, \"V4\": 1.38, \"V5\": -0.34, \"V6\": 0.46, \"V7\": 0.24, \"V8\": 0.10, \"V9\": 0.60, \"V10\": 0.07, \"V11\": -0.55, \"V12\": -0.62, \"V13\": -0.99, \"V14\": -0.31, \"V15\": 1.47, \"V16\": -0.47, \"V17\": 0.21, \"V18\": 0.03, \"V19\": 0.40, \"V20\": 0.25, \"V21\": -0.02, \"V22\": 0.28, \"V23\": -0.11, \"V24\": 0.07, \"V25\": 0.13, \"V26\": -0.19, \"V27\": 0.13, \"V28\": -0.02, \"Amount\": 149.62}}"
```

**Result:** ☐ PASS / ☐ FAIL

---

## Test 4: Dash Monitoring Dashboard

Start the dashboard if not already running:
```bash
python -m src.dashboard.app
```

| # | Check | How to Verify | Pass Criteria |
|---|---|---|---|
| 4.1 | Dashboard loads | Open `http://localhost:8050` | Dark-themed dashboard renders without errors |
| 4.2 | Navigation works | Click each tab in the navbar | All 6 tabs (Overview, Data Drift, Prediction Drift, Performance, Alerts, Model Registry) load without errors |
| 4.3 | Overview metrics | View Overview tab | Shows prediction count, fraud rate, model accuracy, latency — values update every 5 seconds |
| 4.4 | Auto-refresh | Wait 10 seconds on Overview tab | Charts/metrics visibly refresh with new data (if simulator is running) |
| 4.5 | Timestamp accuracy | Check timestamps on charts | Current time matches (within a few seconds) the latest data point |
| 4.6 | Data drift tab | Navigate to Data Drift tab | Feature PSI values displayed, heatmap or bar chart renders |
| 4.7 | Model registry tab | Navigate to Model Registry tab | Shows at least version 1 with "Production" badge |

**Result:** ☐ PASS / ☐ FAIL

---

## Test 5: Production Traffic Simulation

| # | Check | How to Verify | Pass Criteria |
|---|---|---|---|
| 5.1 | Simulator starts | `python scripts/simulate_production.py` | Logs `FastAPI serving layer is ready. Starting traffic stream.` |
| 5.2 | Predictions stream | Watch console output | Logs like `Tx 0001: Pred=0 (Conf=0.95), True=0` appear every ~1 second |
| 5.3 | Dashboard updates | Check dashboard Overview tab | Prediction count and charts increase over time |
| 5.4 | Ground truth feedback | Wait 10+ seconds, then check SQLite | `predictions` table has rows where `true_label` is NOT NULL |

**Result:** ☐ PASS / ☐ FAIL

---

## Test 6: Drift Injection & Detection

| # | Check | How to Verify | Pass Criteria |
|---|---|---|---|
| 6.1 | Inject feature drift | `python scripts/inject_drift.py --type feature_shift --magnitude 2.0` | Prints `Drift config successfully written` |
| 6.2 | Simulator reflects drift | Watch simulator console | Logs show `[DRIFT: feature_shift]` appended to each transaction |
| 6.3 | Manual drift check | `curl -X POST http://localhost:8000/drift/run` | Returns JSON with `overall_status` of `"WARNING"` or `"CRITICAL"` |
| 6.4 | Dashboard drift tab | Check Data Drift tab | PSI values elevated (orange/red indicators), drifted features listed |
| 6.5 | Inject severe drift | `python scripts/inject_drift.py --type severe --magnitude 2.8 --ratio 0.25` | Config written successfully |
| 6.6 | Alert generated | Check dashboard Alerts tab | At least one WARNING or CRITICAL alert logged |
| 6.7 | Clear drift | `python scripts/inject_drift.py --clear` | Drift config reset, simulator returns to normal |
| 6.8 | Drift resolves | Wait 1-2 minutes, trigger manual drift check again | PSI values return to normal range |

**Result:** ☐ PASS / ☐ FAIL

---

## Test 7: Automated Retraining Pipeline

> [!IMPORTANT]
> This test requires severe drift to be injected and enough drifted predictions to accumulate (~50+ with ground truth).

| # | Check | How to Verify | Pass Criteria |
|---|---|---|---|
| 7.1 | Inject severe drift | `python scripts/inject_drift.py --type severe --magnitude 2.8 --ratio 0.25` | Config written |
| 7.2 | Wait for accumulation | Let simulator run for ~60 seconds | At least 50 predictions with ground truth in database |
| 7.3 | Trigger drift check | `curl -X POST http://localhost:8000/drift/run` | Returns with `overall_status: "CRITICAL"` |
| 7.4 | Retraining triggered | Watch API console logs | Messages about retraining pipeline starting |
| 7.5 | New model registered | Check dashboard Model Registry tab | Version 2 appears (or version N+1) |
| 7.6 | Champion/Challenger | Check console logs | Log entry showing model comparison result (DEPLOY or KEEP) |
| 7.7 | Model hot-swap | GET `http://localhost:8000/models/current` | If new model was better: version number incremented |
| 7.8 | Retraining event | Check SQLite `retraining_events` table | Row with `status = 'COMPLETED'` and correct version numbers |

**Result:** ☐ PASS / ☐ FAIL

---

## Test 8: Alert Channels

### 8a. Console Alerts (Always On)

| # | Check | How to Verify | Pass Criteria |
|---|---|---|---|
| 8a.1 | Console output | Trigger a drift check after drift injection | Alert text printed to terminal/console with severity and message |
| 8a.2 | Alert in database | Check dashboard Alerts tab or SQLite `alerts` table | Alert row exists with `channel = 'console'` |

### 8b. Slack Alerts (Optional)

> Skip this section if you have not configured Slack integration.

| # | Check | How to Verify | Pass Criteria |
|---|---|---|---|
| 8b.1 | Slack config | Set `slack.enabled: true` in `configs/alerting_config.yaml` and `SLACK_WEBHOOK_URL` in `.env` | No startup errors |
| 8b.2 | Slack message | Inject drift → trigger drift check | Message appears in configured Slack channel |
| 8b.3 | Message format | Read the Slack message | Contains severity, drift type, metric values, and timestamp |

### 8c. Email Alerts (Optional)

> Skip this section if you have not configured email integration.

| # | Check | How to Verify | Pass Criteria |
|---|---|---|---|
| 8c.1 | Email config | Set `email.enabled: true` in `configs/alerting_config.yaml` and SMTP vars in `.env` | No startup errors |
| 8c.2 | Email received | Inject drift → trigger drift check | Email arrives at configured `to_addresses` |
| 8c.3 | Email content | Read the email | Contains subject with severity, body with drift details |

**Result:** ☐ PASS / ☐ FAIL

---

## Test 9: Data Persistence & Recovery

| # | Check | How to Verify | Pass Criteria |
|---|---|---|---|
| 9.1 | Stop all services | Ctrl+C all running processes | All processes stop cleanly |
| 9.2 | Restart API | `python -m uvicorn src.api.app:create_app --host 127.0.0.1 --port 8000 --factory` | API starts, loads existing production model from registry |
| 9.3 | Historical data intact | GET `http://localhost:8000/predictions/stats` | `total_count` matches pre-restart value |
| 9.4 | Dashboard recovery | Restart `python -m src.dashboard.app` | Charts display historical data from before the restart |
| 9.5 | MLflow data intact | Check `mlruns/` directory | All previous experiment runs and model versions still accessible |

**Result:** ☐ PASS / ☐ FAIL

---

## Test 10: Edge Cases & Error Handling

| # | Check | How to Verify | Pass Criteria |
|---|---|---|---|
| 10.1 | Invalid features | POST `/predict` with missing features | Returns 422 with validation error, doesn't crash |
| 10.2 | Empty request body | POST `/predict` with `{}` | Returns 422, server continues running |
| 10.3 | Invalid ground truth ID | POST `/ground-truth/99999` with `{"true_label": 0}` | Returns 404 or appropriate error |
| 10.4 | Drift check with no data | Clear DB → POST `/drift/run` | Returns graceful response (not enough data), no crash |
| 10.5 | Concurrent requests | Send 10 simultaneous prediction requests | All return valid responses, no deadlocks |

**Sample test for 10.1:**
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"features\": {\"V1\": 1.0}}"
```

**Sample test for 10.5 (PowerShell):**
```powershell
1..10 | ForEach-Object -Parallel {
    $body = '{"features": {"V1":-1.36,"V2":-0.07,"V3":2.54,"V4":1.38,"V5":-0.34,"V6":0.46,"V7":0.24,"V8":0.10,"V9":0.60,"V10":0.07,"V11":-0.55,"V12":-0.62,"V13":-0.99,"V14":-0.31,"V15":1.47,"V16":-0.47,"V17":0.21,"V18":0.03,"V19":0.40,"V20":0.25,"V21":-0.02,"V22":0.28,"V23":-0.11,"V24":0.07,"V25":0.13,"V26":-0.19,"V27":0.13,"V28":-0.02,"Amount":149.62}}'
    Invoke-RestMethod -Method Post -Uri "http://localhost:8000/predict" -Body $body -ContentType "application/json"
} -ThrottleLimit 10
```

**Result:** ☐ PASS / ☐ FAIL

---

## Summary Scorecard

| Test Group | Pass? |
|---|---|
| 1. Environment & Dependencies | ☐ |
| 2. Baseline Model Training | ☐ |
| 3. FastAPI Serving Layer | ☐ |
| 4. Dash Monitoring Dashboard | ☐ |
| 5. Production Traffic Simulation | ☐ |
| 6. Drift Injection & Detection | ☐ |
| 7. Automated Retraining Pipeline | ☐ |
| 8. Alert Channels | ☐ |
| 9. Data Persistence & Recovery | ☐ |
| 10. Edge Cases & Error Handling | ☐ |

**Overall Result:** ______ / 10 test groups passed

> [!TIP]
> If all 10 test groups pass, the platform is fully validated and ready for production use with real data.
