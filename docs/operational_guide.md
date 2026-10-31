# Operations

Use the root README for current launch commands. The supported GUI is Dash; the earlier Streamlit
prototype is retired. Default ports: API 8000, dashboard 8050, MLflow 5000. Use one serving worker.

Logs are under `MLOPS_HOME/logs`. An occupied port stops launch before any service starts. A missing
champion bundle makes readiness fail; inspect the API log and restore the matching trusted bundle
or use a fresh runtime for a new demonstration. Do not replace a scaler separately from its model.

If a worker stops, its heartbeat becomes offline. Restart the supervised platform: interrupted jobs
are marked and can be resubmitted. A completed prediction request is distinct from healthy monitoring;
check monitoring age and label coverage. Notification failures are recorded without pretending delivery.

Back up the whole runtime while services are stopped. Keep the local management token private.
To use a different dataset, point MLOPS_HOME at a new dedicated directory; never reuse a historical
holdout created from another dataset. Changing quality gates requires a restart.

The demo reset removes only generated predictions. Source CSVs, API predictions, version bundles,
and model/audit records are retained. External email/Slack channels are disabled until configured.
