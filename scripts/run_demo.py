"""Start and supervise the local API, worker, dashboard, and MLflow UI."""

import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.runtime import home, management_token
from src.config.settings import get_settings
from src.data.database import get_database
from src.jobs import JobQueue


def check_port(port):
    with socket.socket() as connection:
        try:
            connection.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(
                f"Port {port} is occupied. Stop the other service before launching."
            ) from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    for port in [settings.api.port, settings.dashboard.port, 5000]:
        check_port(port)
    management_token()
    db = get_database(settings.database.url)
    db.init_db()
    JobQueue(db)
    logs = home() / "logs"
    logs.mkdir(exist_ok=True)
    processes, streams = [], []
    stop = False

    def on_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, on_stop)
    signal.signal(signal.SIGTERM, on_stop)

    def launch(name, command):
        stream = (logs / (name + ".log")).open("a", encoding="utf-8")
        streams.append(stream)
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "false",
            "MLOPS_API_URL": f"http://127.0.0.1:{settings.api.port}",
        }
        process = subprocess.Popen(
            [sys.executable, *command],
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        processes.append((name, process))

    def wait_http(path, seconds=180, base_url=None):
        end = time.monotonic() + seconds
        while time.monotonic() < end and not stop:
            for name, process in processes:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} stopped. Inspect {logs / (name + '.log')}")
            try:
                response = requests.get(
                    (base_url or f"http://127.0.0.1:{settings.api.port}") + path, timeout=2
                )
                if response.ok:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)
        raise RuntimeError(f"Startup timed out waiting for {path}; inspect {logs}")

    try:
        launch(
            "api",
            [
                "-m",
                "uvicorn",
                "src.api.app:create_app",
                "--factory",
                "--host",
                settings.api.host,
                "--port",
                str(settings.api.port),
                "--workers",
                "1",
            ],
        )
        wait_http("/health")
        launch("worker", ["-m", "src.worker"])
        if not requests.get(f"http://127.0.0.1:{settings.api.port}/ready", timeout=5).ok:
            job, _ = JobQueue(db).enqueue("Initial baseline training", "baseline")
            print("Training initial baseline. Logs:", logs, flush=True)
            wait_http("/ready", seconds=600)
        launch("dashboard", ["-m", "src.dashboard.app"])
        launch(
            "mlflow",
            [
                "-m",
                "mlflow",
                "ui",
                "--backend-store-uri",
                settings.mlflow.tracking_uri,
                "--host",
                os.environ.get("MLFLOW_HOST", "127.0.0.1"),
                "--port",
                "5000",
            ],
        )
        wait_http("/_dash-layout", base_url=f"http://127.0.0.1:{settings.dashboard.port}")
        wait_http("/", base_url="http://127.0.0.1:5000")
        from scripts.smoke_test import verify

        verify(
            f"http://127.0.0.1:{settings.api.port}",
            f"http://127.0.0.1:{settings.dashboard.port}",
            "http://127.0.0.1:5000",
            timeout=180,
            check_only=True,
            cancelled=lambda: stop,
        )
        print(
            f"DriftWatch is running. Dashboard: http://127.0.0.1:{settings.dashboard.port}",
            flush=True,
        )
        print(
            f"API: http://127.0.0.1:{settings.api.port}/docs | MLflow: http://127.0.0.1:5000",
            flush=True,
        )
        print("Press Ctrl+C to stop all services. Logs:", logs, flush=True)
        if not args.no_browser:
            import webbrowser

            webbrowser.open(f"http://127.0.0.1:{settings.dashboard.port}")
        while not stop:
            for name, process in processes:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited. Inspect {logs / (name + '.log')}")
            time.sleep(0.5)
    finally:
        for _, process in reversed(processes):
            if process.poll() is not None:
                continue
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True
                )
            else:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for _, process in reversed(processes):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    process.kill()
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait()
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    main()
