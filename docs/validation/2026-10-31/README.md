# Verification follow-up — 2026-10-31

## Completed locally

- Windows / Python 3.12.14: **40 tests passed**, 23 upstream warnings, 96.23 seconds.
- The new smoke check exercised all four services, 500 actual predictions with labels,
  all three drift detectors, a real challenger evaluation, and serving-version/gate consistency.
- The challenger was rejected by its quality gates; champion version 1 was preserved.
- Restart verification preserved version 1 and all 500 retained predictions, with the worker online.
- See [lifecycle evidence](python312-lifecycle.json) and [restart evidence](python312-restart.json).
- Python 3.11.9 is installed in an isolated package directory. Locked dependencies pass
  `pip check`, and NumPy, pandas, SciPy, and MLflow import successfully. Its tests are in progress.

## Changes made during follow-up

- Startup now waits for the dashboard, MLflow UI, and a worker heartbeat before announcing readiness.
- Startup verification can be cancelled promptly; Linux shutdown tolerates child exit races and
  terminates a surviving process group after a shutdown timeout.
- Windows setup selects Python 3.12 or 3.11 explicitly instead of whichever `py -3` resolves to.
- Added `python -m scripts.smoke_test` for full lifecycle verification and a read-only
  `--check-only --expect-snapshot <report>` mode for restart validation.
- Docker CI uses real lifecycle checks and bounded polling in place of fixed sleep delays,
  and uploads JSON reports plus service logs as evidence. Manual workflow dispatch is enabled.
- Converted two documentation files from Windows text encoding to UTF-8.

## Remaining remote verification

Docker is not installed here, and the Windows Subsystem for Linux and Virtual Machine Platform
features are disabled. Linux and Docker execution must therefore run through GitHub Actions.
The user approved publishing a separate branch and draft PR; main will remain unchanged.
Results will be recorded here after the remote checks complete.

No external SMTP/Slack delivery was attempted, and no real creditcard.csv was supplied.
