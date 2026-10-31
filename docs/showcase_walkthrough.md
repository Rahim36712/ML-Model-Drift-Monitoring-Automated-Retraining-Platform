# Five-minute presentation

## Before the event

Start the application once with internet access to install dependencies and train the baseline.
Restart to confirm persistence. Open the dashboard and MLflow in advance. Keep the API documentation
available if the audience wants the prediction contract. Reset generated demo telemetry from Settings
when no job is active; this preserves the model and prior audit history.

## Talk track

**0:00 — The problem.** A fraud classifier can become unreliable when incoming data changes.
DriftWatch makes changes measurable and model replacement conditional on evidence.

**0:45 — Real inference.** Generate traffic. Explain fraud probability, labels, inference latency,
end-to-end application timing, and label coverage. Synthetic records are explicitly tagged.

**1:30 — A distribution shift.** Introduce drift, then run the check. Open Drift Analysis and identify
the shifted features. PSI/KL track features; Hellinger tracks prediction probabilities; F1 needs labels.
A distribution shift alone is not proof of worse business performance.

**2:30 — A controlled response.** Follow the worker job in Models & Retraining. Show fixed historical
and eligible recent evaluation results. The challenger must improve F1 without exceeding regression
limits. A rejected candidate demonstrates the safety gate just as honestly as a promoted candidate.

**3:45 — Recovery and audit.** When there is a previous accepted version, show rollback. Inspect alerts,
delivery status, and model history. Disable auto-refresh to inspect a chart without losing its zoom.

**4:30 — Engineering choices.** Describe immutable model/scaler bundles, training outside serving,
durable jobs, consistent snapshots, and the single-machine deployment boundary. Mention the automated
tests and distinguish locally executed checks from CI configurations.

Do not promise promotion, a specific score, production readiness, or zero mistakes. Explain the
measured outcome visible in the current run.
