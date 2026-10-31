# Observed validation — 2026-10-27

This is the initial verification record. See [the 2026-10-31 follow-up](validation/2026-10-31/README.md) for compatibility and CI follow-up results.

## Environment and results

Executed on Windows with Python 3.12.14 in a newly populated repository virtual environment.
Dependency downloads initially timed out; the resolved PyPI wheels were downloaded with range
requests, checked against their published SHA-256 values, and installed locally. The Windows
launcher was subsequently executed successfully against the locked installed environment.

| Check | Observed result |
| --- | --- |
| `python -m pytest -q` | **36 passed**, 23 warnings, 79.20 seconds |
| `python -m pip check` | No broken requirements |
| `python -m ruff check src scripts tests --select E9,F63,F7,F82` | Passed |
| `git diff --check` | Passed |
| `python -m build --no-isolation` | Wheel and source distribution built |
| Built wheel installed outside source tree | Config defaults, launcher imports, and dashboard CSS passed |
| Distribution content inspection | Required assets included; management token and telemetry excluded |
| `start-demo.ps1 -NoBrowser` | API, worker, dashboard, and MLflow UI started |
| Stop supervised platform | Ports 8000, 8050, and 5000 closed |
| Restart existing runtime | Champion version 1 and 1,000 predictions preserved; worker online |
| Second launcher while running | Refused before starting services with occupied-port explanation |
| MLflow UI | HTTP 200 |
| PowerShell AST parsing | No syntax errors |
| Compose YAML | Valid; every published port binds to loopback |

Warnings originate in third-party MLflow/Pydantic/Starlette deprecations. They are reported rather
than hidden. The lint command checks syntax/runtime name errors; it is not a claim of a full
security audit or exhaustive style enforcement.

## Real lifecycle exercised

1. Created the deterministic 10,000-row synthetic dataset and fixed historical split.
2. Trained and registered a real RandomForest baseline, logged its complete serving pipeline and
   bundle in local MLflow, and received activation acknowledgment from the serving API.
3. Used the dashboard to generate **500 stable predictions**, with source labels submitted through
   the ground-truth API. Verified prediction counts, label coverage, and measured latency.
4. Generated **500 feature-shifted predictions** through the same inference and feedback paths.
5. Ran all three drift checks on the latest 500 samples. Observed critical drift, PSI approximately
   13.282 and Hellinger distance approximately 0.404, with no failed detector checks.
6. Followed an actual worker retraining job. The candidate improved recent-set F1 from 0.000 to
   approximately 0.286 but reduced historical F1 from approximately 0.154 to 0.080 and recall from
   approximately 0.083 to 0.042. The gates **rejected** it; version 1 stayed active. These measurements
   describe this synthetic scenario and are not promised scores for other data.
7. Confirmed durable state and serving readiness after restarting the complete platform.

Live QA caught a decision-timestamp JSON serialization failure that the initial tests missed. It
was corrected and covered by state round-trip and real monitoring regression checks before the
final passing suite. Older development logs/audit entries remain historical evidence; the current
implementation does not fabricate successful recovery outcomes.

## Automated scenarios covered

- Real MLflow training, pipeline reload, immutable historical split, and holdout exclusion from feedback training.
- Actual rejected challenger; failed activation cannot become completed; rollback reloads the complete bundle.
- Immutable snapshots remain internally consistent during concurrent reads and updates.
- Single/batch equivalence, one vectorized probability call, invalid/nonfinite/missing features,
  empty/oversized batches, delayed labels, unloaded-model readiness, and management authentication.
- Concurrent duplicate job submissions and claims; interrupted-job recovery.
- Constant/tail/empty/nonfinite histogram inputs; signed F1 changes and promotion regression gates.
- Notification delivery failure and persistent deduplication.
- New-database and legacy-table migration, including preservation of existing records.
- Demo reset requires the confirmation phrase and preserves ordinary API predictions.
- All empty GUI pages serialize; local assets and legacy dashboard redirects work.
- Latest-100 trend selection; monitoring decisions containing timestamps persist correctly.

## Browser QA

Verified all six pages at the default desktop viewport and the Overview at **390 × 844**.
Feature selection updated the chart. Alert search narrowed the records; acknowledgment changed
its status. CSV export downloaded an actual 1,000-record explorer extract. The reset dialog was
opened and canceled, preserving the visible demo data. Pause/resume controls and responsive
navigation were inspected. Keyboard selection worked in the feature dropdown.

Screenshots are in [screenshots](screenshots/):

- [Overview — desktop](screenshots/overview-desktop.png)
- [Overview — mobile](screenshots/overview-mobile.png)
- [Drift analysis](screenshots/drift-desktop.png)
- [Predictions](screenshots/predictions-desktop.png)
- [Models and quality gates](screenshots/models-desktop.png)
- [Alert acknowledgment](screenshots/alerts-desktop.png)
- [Settings](screenshots/settings-desktop.png)

## Explicit verification limits

- Docker is not installed on this host. The image was **not built or run locally**. Compose syntax
  and bindings were checked; the committed CI workflow defines a Linux build/start/restart check.
- Python 3.11 and Linux were **not executed locally**. Resolved package metadata accepts Python
  3.11; CI defines Windows/Linux × Python 3.11/3.12. That workflow has not run remotely because
  these changes have not been pushed.
- SMTP and Slack were disabled for the live demo. Failures/deduplication were tested with mocks;
  real external delivery was not attempted.
- A real external creditcard.csv was not supplied. The compatible CSV validation path was tested
  with generated frames, not a user-provided Kaggle download.
- The launcher install step was verified with dependencies already installed after the initial
  network workaround; a second clean environment using only normal pip downloads was not timed.

This release is for a trusted local showcase. It does not claim distributed serving, public-hosting
security, guaranteed model improvement, or universal zero-downtime deployment.
