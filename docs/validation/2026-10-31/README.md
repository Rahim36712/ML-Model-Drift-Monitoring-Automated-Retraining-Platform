# Verification follow-up — 2026-10-31

## Completed locally

- Windows / Python 3.12.14: **40 tests passed**, 23 upstream warnings, 96.23 seconds.
- The new smoke check exercised all four services, 500 actual predictions with labels,
  all three drift detectors, a real challenger evaluation, and serving-version/gate consistency.
- The challenger was rejected by its quality gates; champion version 1 was preserved.
- Restart verification preserved version 1 and all 500 retained predictions, with the worker online.
- See [lifecycle evidence](python312-lifecycle.json) and [restart evidence](python312-restart.json).
- Python 3.11.9 is installed in an isolated package directory. Locked dependencies pass
  `pip check`, and NumPy, pandas, SciPy, and MLflow import successfully. **40 tests passed**,
  17 upstream warnings, 81.14 seconds. See [test evidence](python-test-results.json).
- Python 3.11 also passed the real four-service lifecycle and restart checks: 500 predictions,
  all three detectors, a rejected challenger, champion version 1 preserved, and an online worker.
  See [3.11 lifecycle](python311-lifecycle.json) and [3.11 restart](python311-restart.json).
- The source distribution and wheel build successfully. Dependency consistency and critical
  Ruff checks pass. Verification services were shut down cleanly afterward.

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

## Completed through GitHub Actions

[Run 41230438047](https://github.com/Rahim36712/ML-Model-Drift-Monitoring-Automated-Retraining-Platform/actions/runs/41230438047)
completed successfully for code commit `4ae16bfb0fccb8acd7380481c87849731772fefb`.
All five jobs passed:

| Environment | Tests | Other checks |
| --- | --- | --- |
| Ubuntu / Python 3.11 | 40 passed, 17 warnings | Locked dependency install, pip check, Ruff, wheel and sdist |
| Ubuntu / Python 3.12 | 40 passed, 23 warnings | Locked dependency install, pip check, Ruff, wheel and sdist |
| Windows / Python 3.11 | 40 passed, 17 warnings | Locked dependency install, pip check, Ruff, wheel and sdist |
| Windows / Python 3.12 | 40 passed, 23 warnings | Locked dependency install, pip check, Ruff, wheel and sdist |
| Docker on Ubuntu | Lifecycle and restart passed | Compose validation, image build, four-service readiness, real inference, drift and retraining |

CI creates a fresh virtual environment before installing the lock. This also resolves the first
Windows CI attempt's conflict with the hosted runner's preinstalled pipx package; application
dependencies were not changed to accommodate unrelated runner software.

Docker generated 500 labeled predictions, completed all three drift checks, rejected the real
challenger on quality gates, and preserved champion version 1 and all 500 records across restart.
See [Docker lifecycle](docker-lifecycle.json), [Docker restart](docker-restart.json), and
[machine-readable CI job results](github-actions.json). Service logs are retained in that run's
`docker-lifecycle-evidence` artifact. Docker is not installed on this local Windows host;
the container verification ran on GitHub's Ubuntu runner.

The changes are published on `codex/showcase-ready-verification` in
[draft PR #2](https://github.com/Rahim36712/ML-Model-Drift-Monitoring-Automated-Retraining-Platform/pull/2).
Main was not modified. Subsequent documentation-only commits can be checked in the PR's Checks tab.

## Remaining limits

- No external SMTP/Slack delivery was attempted. Failure handling and deduplication have automated tests.
- No real creditcard.csv was supplied; live scenarios use explicitly identified synthetic data.
- CI uses clean virtual environments, but this was not tested on a newly installed Windows operating system.
- This release supports one local serving owner. Public hosting and distributed serving remain out of scope.
- Upstream deprecation warnings are visible in the test/build logs; all required checks pass.
