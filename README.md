# DriftWatch

A local ML observability showcase for credit-card fraud detection. Follow real predictions,
feature and prediction drift, labeled performance, retraining, quality gates, and model rollback
in one light-themed dashboard.

The default dataset and demo scenarios are **synthetic**. Metrics are computed from actual
scikit-learn inference and source labels; a rejected challenger is shown as rejected.
This is a single-machine portfolio application, not a distributed production service.

![DriftWatch dashboard](docs/screenshots/overview-desktop.png)

## Run on Windows

Install Python **3.11 or 3.12**, then open PowerShell in this repository:

```powershell
.\start-demo.ps1
```

If script execution is restricted, use the manual commands below. The launcher creates a
virtual environment, installs `requirements.lock`, trains the first baseline, and starts all
services. First startup needs internet for dependencies and may take several minutes.
Subsequent runs reuse the persisted model. Press **Ctrl+C** to stop the supervised services.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --timeout 120 -r requirements.lock
.\.venv\Scripts\python.exe -m scripts.run_demo
```

For Linux/macOS, create a Python 3.11/3.12 virtual environment, install the same lock file,
and run `python -m scripts.run_demo`. Use `--no-browser` for terminal-only startup.

## Run with Docker

```sh
docker compose up --build
```

The Compose service supervises the same API, worker, dashboard, and MLflow UI. A named volume
preserves runtime data. All published ports bind to `127.0.0.1`. Stop with `docker compose down`;
that keeps the volume. Startup errors and occupied ports are reported explicitly.

- Dashboard: [localhost:8050](http://127.0.0.1:8050)
- API documentation: [localhost:8000/docs](http://127.0.0.1:8000/docs)
- MLflow: [localhost:5000](http://127.0.0.1:5000)

## Present the project

1. Open **Overview**. Confirm API ready, worker online, and a champion version.
2. Click **Generate 500 predictions**. Wait for its real inference job to complete.
3. Click **Simulate feature shift**. Five input features shift; source labels remain attached.
4. Click **Run drift check**. Inspect feature PSI/KL, prediction drift, label coverage, and decisions.
5. Open **Models & Retraining**. Follow the queued job and inspect champion/challenger metrics.
6. Explain why the gates accepted or rejected the candidate. Promotion is never guaranteed.
7. If another version was accepted, demonstrate rollback. Inspect the audit and alert records.

See [the presentation walkthrough](docs/showcase_walkthrough.md) and
[the architecture](docs/architecture.md). A normal synthetic run is fast enough for a short
presentation; initial dependency installation should be completed before the event.

## Real compatible data and configuration

The feature contract is `V1`–`V28`, `Amount`, and binary target `Class`.
Set `MLOPS_DATASET_PATH` to a trusted compatible CSV **before the first training**. Use a new
`MLOPS_HOME` for each dataset so the historical holdout remains fixed. The source CSV is never
modified. Missing/nonfinite features, conflicting duplicate rows, and inadequate class samples
are rejected with an actionable error. Arbitrary datasets and custom model uploads are outside scope.

Runtime state defaults to `data/demo/`, separate from the CSV: telemetry and job databases,
MLflow tracking, immutable serving bundles, the fixed split, logs, and a local management token.
Paths are anchored to the repository or explicit environment settings, not the shell's directory.
Settings displays thresholds and notification status without returning credentials to the browser.
Edit the YAML configurations or `.env` and restart to change them.

The demo reset in Settings clears only records tagged `demo`. It keeps user API predictions,
source data, models, and audit history. Prediction retention is 30 days / 50,000 records;
job history and model versions remain available.

## API essentials

| Endpoint | Purpose |
| --- | --- |
| `GET /health`, `GET /ready` | Liveness/database access and loaded-model readiness |
| `POST /predict`, `POST /predict/batch` | Single and vectorized batch inference (1–1,000 records) |
| `POST /ground-truth/{id}`, `POST /ground-truth/batch` | Attach delayed binary source labels |
| `GET /predictions`, `GET /predictions/stats` | Telemetry and summary statistics |
| `POST /drift/run`, `GET /drift/latest`, `GET /drift/history` | Monitoring |
| `POST /models/retrain`, `GET /jobs/{id}` | Queue and follow durable training work |
| `GET /models/current`, `GET /models/versions` | Champion and registry metadata |
| `POST /models/rollback/{version}` | Activate a complete previous serving bundle |
| `GET /alerts`, `POST /alerts/{id}/acknowledge` | Notification audit and acknowledgment |
| `GET /dashboard/snapshot` | Shared monitoring snapshot for the GUI |

Management operations require `X-MLOps-Token`, read server-side from `MLOPS_TOKEN` or the
generated `MLOPS_HOME/.management-token`. Dash attaches it on the server. Swagger permits
entering this header on protected operations. Do not expose these local services publicly;
MLflow and the Dash workspace are intended for trusted local access.

## Quality and failure behavior

- Split historical raw data once; fit preprocessing on training data only.
- Each immutable version includes its pipeline, schema, raw-feature references, baseline
  predictions, fraud metrics, and data source. Only trusted locally generated bundles are loaded.
- Reserve the newest 20% of unique new labeled production rows for recent evaluation.
  Use that set only with at least 100 records and five of each class; otherwise use the fixed holdout.
- Automatic promotion needs F1 improvement ≥0.005 on the selected set, historical F1 drop ≤0.01,
  and recall drop ≤0.02 on both eligible evaluation sets.
- Rejected/failed candidates preserve the champion. Activation is acknowledged by the serving API;
  an OS lock prevents multiple API owners. Atomic in-memory snapshots prevent mixed versions.
- SQLite jobs survive restarts. Interrupted training is explicitly marked; duplicate submissions
  reuse the active job. Monitoring has minimum samples and an automatic retraining cooldown.
- Missing/stale monitoring, insufficient labels, offline workers, and notification failures stay visible.

## Verification

```sh
python -m pip check
python -m ruff check src scripts tests --select E9,F63,F7,F82
python -m pytest -q
python -m build
```

CI checks Python 3.11/3.12 on Windows/Linux and exercises a real Docker lifecycle and restart.
See the [latest verification evidence](docs/validation/2026-10-31/README.md),
[browser validation and screenshots](docs/manual_validation.md), and the draft PR's checks.

With the application running, verify the complete lifecycle from another terminal using the
same environment configuration:

```sh
python -m scripts.smoke_test --report lifecycle.json
```

This submits 500 synthetic predictions with labels, checks drift, and follows real challenger
evaluation. Rejection is a valid result when the quality gates fail. For a read-only readiness
check, use `python -m scripts.smoke_test --check-only`.
