# Initial repository review

Historical findings before the v2 implementation. See the current README and manual validation for the replacement architecture and observed verification results.

Reviewed on 2026-10-27 against cloned commit `dac3f2c`.

## What this project does

This is a credit-card fraud monitoring prototype with 29 input features (`V1`–`V28`
and `Amount`). It is not yet a general platform for arbitrary models or datasets.
It supports a real `creditcard.csv`, with a synthetic dataset fallback.

The code contains:

- FastAPI single and batch prediction routes, prediction history/statistics, and
  delayed ground-truth submission at `/ground-truth/{prediction_id}`.
- Feature drift with PSI/KL, prediction drift with Hellinger distance, and
  performance degradation checks against labeled predictions.
- A five-minute scheduled monitoring job, configurable thresholds, and a
  retraining decision engine.
- A retraining workflow intended to train a challenger, evaluate it, register it
  in MLflow, and replace the in-memory model.
- SQLite telemetry, alerts, model metadata, and retraining audit records.
- Console, Slack, and email notification implementations.
- A six-page Dash monitoring interface plus a separate Streamlit interface.

These are implementation capabilities and intentions, not a claim that the full
system currently runs from a fresh clone. The missing modules below prevent that.

## Fix first: correctness and reproducibility

| Priority | Finding and evidence | Recommended change |
| --- | --- | --- |
| P0 | `src/models/` is absent. The API, pipeline, training script, and tests import its trainer, evaluator, and registry modules. `git check-ignore src/models/trainer.py` confirms that the unanchored `models/` ignore pattern matches source code too. | Recover the original source modules and commit them. Anchor the artifact ignore to `/models/`. Do not replace the missing implementation with mocks merely to make startup pass. |
| P0 | Streamlit's retraining fallback writes a hardcoded version 2 and displays deployment success without training or swapping a model. Several queries use nonexistent `DriftResult` attributes, including `overall_status` and `metrics_json`; the actual schema stores individual metrics and `details_json`. | Route operator actions through the backend, display accepted/running/completed/rejected/failed states, and align the UI with the database/API schemas. Remove fabricated success paths. |
| P1 | `DataLoader.preprocess` fits the scaler before the train/test split and overwrites a shared scaler file before the candidate is accepted. The pipeline evaluates the champion using the candidate's transformed inputs. Rollback also loads the shared scaler. | Split first; fit only on training data. Bundle the scaler and model as an immutable versioned pipeline. Evaluate each version using its own preprocessing on the same untouched raw holdout. Keep candidate artifacts isolated until promotion. |
| P1 | Training reconstructs a new split from the combined historical and production dataset. A previously used holdout can enter later training splits. | Reserve an immutable evaluation dataset, supplement it with a recent labeled evaluation window, and deduplicate production events against training inputs. |
| P1 | `RetrainingPipeline.execute` ignores the Boolean result of `deployer.deploy` and can mark a failed deployment completed. Database production status is changed before a successful serving swap. | Check deployment results, record failure, and coordinate registry/DB/serving changes with recovery and rollback. Enforce a single production version. |
| P1 | Data drift compares raw logged features with reference arrays saved from preprocessed training data, including scaled Amount. The in-memory detectors retain old references after retraining. | Compare matching feature representations and reload all reference metrics/distributions after a successful promotion. |
| P1 | PSI/KL histogram bins end at the reference minimum/maximum. Out-of-range current values are discarded; all-outside samples can divide by zero. Constant references can produce no usable bins. | Include tail buckets, handle constant/empty/nonfinite inputs, normalize safely, and add boundary-case tests. |
| P1 | The decision engine uses `abs(f1_delta)` as degradation, so an F1 improvement can also trigger retraining. Startup prediction baselines may be all-zero labels and constant confidence. | Use signed degradation and versioned real baseline predictions. Report insufficient baseline data explicitly. |
| P1 | API management routes have no authentication. Scheduling and the active model cache are process-local. Multiple API workers can independently schedule training and serve different versions. | Authenticate management operations, restrict CORS, run one durable monitoring/training worker with a job lock, and coordinate model reloads across serving workers. |
| P2 | README endpoint names differ from implemented routes, Streamlit is absent from dependency declarations, and the build backend is `setuptools.backends._legacy:_Backend`. | Generate API docs from OpenAPI, choose a supported dashboard, declare optional UI dependencies, use `setuptools.build_meta`, and verify fresh installs in CI with locked dependencies. |

## Performance optimizations after correctness

1. Vectorize batch inference: currently the batch endpoint calls transform,
   predict, and predict_proba once per row. Validate the batch once, build one
   matrix, and call each model operation once per batch.
2. Consolidate dashboard reads: every refresh triggers multiple callbacks that
   independently fetch overlapping statistics. Use one cached snapshot per
   refresh, shared by the relevant charts and cards.
3. Keep telemetry costs bounded: add retention, time-range filtering, and
   preaggregated volume/latency metrics. Measure SQLite contention under load;
   move to PostgreSQL when concurrent writes and operational needs justify it.
4. Run training outside serving processes with durable job state, bounded
   concurrency, cancellation, retry policy, and resource limits.
5. Cache reference histogram edges/counts per model version. Recompute current
   histogram counts per window instead of rebuilding static reference statistics.
6. Measure p50/p95/p99 request latency, model inference time, logging time,
   throughput, label coverage, monitoring age, and training duration separately.
   The current inference timer does not measure complete HTTP response latency.

## GUI changes made locally

The primary Dash interface now includes:

- A pause/resume auto-refresh switch and visible configured refresh interval.
- A neutral initial health badge and explicit UTC dashboard refresh time.
- Overview guidance explaining PSI, ground-truth dependence, and the limits of
  the drift badge as a health signal.
- Empty-state annotations for overview charts and persistent chart zoom across
  refreshes.
- The newest 100 PSI/F1 observations, displayed in chronological order, instead
  of the oldest 100.
- Keyboard focus indicators and reduced-motion support.
- Removal of the unused duplicate HTML/CSS theme definition.

## Next GUI iteration

Keep Dash as the primary interface until the Streamlit schema and action flows
are repaired. Use one visual system instead of maintaining two divergent UIs.

Add a compact summary for active model version, API readiness, latest monitoring
window, sample count, label coverage, and last successful check. Show stale data
and unavailable data explicitly. Add time/model/feature filters and threshold
values read from configuration. Provide champion/challenger comparisons and a
training timeline with honest job status. Keep demo actions and synthetic records
visibly labeled so operators can distinguish them from actual inference traffic.

## Validation

- `python -m compileall -q src tests`: passed using the bundled Python runtime.
- `git diff --check`: passed.
- A source import audit confirmed missing `src.models.trainer`,
  `src.models.evaluator`, and `src.models.registry` modules.
- Added `tests/test_dashboard_overview.py` to exercise a real in-memory database
  with 105 measurements and verify that both trend charts show the latest 100.
- Runtime tests and browser rendering remain unverified: dependency installation
  stalled on wheel downloads, including after retrying the public package index.
  The pytest command failed because pytest was not installed. Interrupted the
  stalled installs; the partial local `.venv` is ignored by Git.

The README's historical test badge is not evidence that the fresh clone passes.
API and retraining tests also depend on the missing `src.models` modules.
