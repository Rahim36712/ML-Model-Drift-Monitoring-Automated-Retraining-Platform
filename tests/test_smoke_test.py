"""Verification must fail clearly instead of announcing a false ready state."""

import pytest
import requests
from scripts.smoke_test import check_persistence, wait_for


def test_wait_for_retries_unavailable_service():
    attempts = []

    def operation():
        attempts.append(1)
        if len(attempts) < 3:
            raise requests.ConnectionError("not listening")
        return {"ready": True}

    assert wait_for("service", operation, timeout=1, interval=0)["ready"]
    assert len(attempts) == 3


def test_wait_for_can_cancel_startup_immediately():
    with pytest.raises(RuntimeError, match="cancelled"):
        wait_for("service", lambda: None, timeout=60, cancelled=lambda: True)


def test_wait_for_times_out_instead_of_claiming_readiness():
    with pytest.raises(RuntimeError, match="Timed out waiting for worker"):
        wait_for("worker", lambda: None, timeout=0)


def test_restart_requires_same_model_and_retained_records():
    expected = {"active_version": 2, "prediction_count": 500}
    check_persistence(expected, {"active_version": 2, "prediction_count": 501})
    with pytest.raises(RuntimeError, match="champion"):
        check_persistence(expected, {"active_version": 1, "prediction_count": 501})
    with pytest.raises(RuntimeError, match="prediction"):
        check_persistence(expected, {"active_version": 2, "prediction_count": 499})
