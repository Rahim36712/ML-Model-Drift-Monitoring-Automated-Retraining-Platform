"""Exercise a running showcase and emit evidence for startup/restart checks.

Run inside the same runtime as the platform so the management token stays local.
The full check adds labeled synthetic traffic; --check-only is read-only.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import requests

from src.config.settings import get_settings
from src.runtime import api_url, management_token

TERMINAL = {"COMPLETED", "REJECTED", "FAILED", "INTERRUPTED"}


def wait_for(description, operation, timeout=180, interval=0.5, cancelled=None):
    deadline = time.monotonic() + timeout
    last_error = "not ready"
    while time.monotonic() < deadline:
        if cancelled and cancelled():
            raise RuntimeError("Verification cancelled")
        try:
            result = operation()
            if result:
                return result
        except (requests.RequestException, ValueError) as exc:
            last_error = str(exc)
        time.sleep(interval)
    raise RuntimeError(f"Timed out waiting for {description}: {last_error}")


def check_persistence(expected, observed):
    if observed["active_version"] != expected["active_version"]:
        raise RuntimeError("Restart changed the active champion version")
    if observed["prediction_count"] < expected["prediction_count"]:
        raise RuntimeError("Restart lost retained prediction records")


def verify(base_url, dashboard_url, mlflow_url, timeout=180, check_only=False, cancelled=None):
    session = requests.Session()
    session.headers["X-MLOps-Token"] = management_token()

    def request(method, path, **kwargs):
        response = session.request(method, base_url.rstrip("/") + path, timeout=30, **kwargs)
        response.raise_for_status()
        return response.json()

    def ready():
        status = request("GET", "/ready")
        return status if status.get("ready") and status.get("db_connected") else None

    wait_for("serving readiness", ready, timeout, cancelled=cancelled)
    for label, url in [("dashboard", dashboard_url + "/_dash-layout"), ("MLflow", mlflow_url)]:

        def available(target=url):
            response = requests.get(target, timeout=5)
            return response.status_code == 200

        wait_for(label, available, timeout, cancelled=cancelled)

    def worker_ready():
        value = request("GET", "/dashboard/snapshot", params={"hours": 720})
        return value if value.get("worker_online") else None

    snapshot = wait_for("worker heartbeat", worker_ready, timeout, cancelled=cancelled)
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "services_ready": True}
    if not check_only:

        def idle():
            return not any(job["status"] not in TERMINAL for job in request("GET", "/jobs"))

        wait_for("idle worker", idle, timeout)
        submitted = request("POST", "/demo/traffic", json={"scenario": "drift", "samples": 500})

        def finished(job_id):
            value = request("GET", "/jobs/" + job_id)
            return value if value["status"] in TERMINAL else None

        traffic = wait_for(
            "real inference and feedback", lambda: finished(submitted["job_id"]), timeout
        )
        if traffic["status"] != "COMPLETED" or traffic["result"].get("processed") != 500:
            raise RuntimeError("Traffic job failed: " + json.dumps(traffic["result"]))
        monitoring = request("POST", "/drift/run")
        if monitoring.get("failed_checks") or len(monitoring.get("checks_performed", [])) != 3:
            raise RuntimeError("Not all monitoring detectors completed")
        if monitoring["overall_status"] != "critical":
            raise RuntimeError("The seeded severe feature shift was not detected")
        job_id = monitoring.get("job_id")
        if not job_id:
            job_id = request("POST", "/models/retrain")["job_id"]
        training = wait_for("challenger evaluation", lambda: finished(job_id), timeout)
        if training["status"] not in {"COMPLETED", "REJECTED"}:
            raise RuntimeError("Training failed: " + json.dumps(training["result"]))
        if "passed" not in training["result"]:
            raise RuntimeError("Training finished without recorded quality gates")
        if (training["status"] == "COMPLETED") != bool(training["result"]["passed"]):
            raise RuntimeError("Job outcome contradicts its quality gates")
        actual_version = request("GET", "/ready")["model_version"]
        expected_version = (
            training["result"]["version"]
            if training["status"] == "COMPLETED"
            else snapshot["active_version"]
        )
        if actual_version != expected_version:
            raise RuntimeError("Serving version contradicts the recorded training outcome")
        report.update(
            traffic_job=submitted["job_id"],
            training_job=job_id,
            training_outcome=training["status"],
            gate_passed=training["result"]["passed"],
            drift_status=monitoring["overall_status"],
            detector_checks=monitoring["checks_performed"],
        )
        # Read after the snapshot cache expires, rather than testing stale counts.
        previous = snapshot["stats"]["total"]
        snapshot = wait_for(
            "prediction snapshot",
            lambda: (
                value
                if (value := worker_ready()) and value["stats"]["total"] >= previous + 500
                else None
            ),
            timeout,
        )
    report.update(
        active_version=request("GET", "/ready")["model_version"],
        prediction_count=snapshot["stats"]["total"],
        worker_online=snapshot["worker_online"],
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--expect-snapshot", type=Path)
    args = parser.parse_args()
    settings = get_settings()
    value = verify(
        api_url(),
        f"http://127.0.0.1:{settings.dashboard.port}",
        "http://127.0.0.1:5000",
        args.timeout,
        args.check_only,
    )
    if args.expect_snapshot:
        check_persistence(json.loads(args.expect_snapshot.read_text(encoding="utf-8")), value)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
