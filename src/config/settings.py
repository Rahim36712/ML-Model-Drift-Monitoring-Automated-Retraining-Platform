"""Pydantic v2 settings loaded from YAML config files with env-var overrides.

Config files resolved relative to ``PROJECT_ROOT``:
    - configs/base_config.yaml
    - configs/drift_thresholds.yaml
    - configs/alerting_config.yaml

Environment variables (loaded via python-dotenv from a ``.env`` file at
the project root) can override any YAML value.  The mapping follows the
flat naming convention documented on each sub-model.

Usage::

    from src.config import get_settings

    settings = get_settings()
    print(settings.database.url)
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root: two parents up from  src/config/settings.py  → repo root
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# YAML loader helper
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return its contents as a dictionary.

    Args:
        path: Absolute or relative ``pathlib.Path`` to a YAML file.

    Returns:
        Parsed YAML contents.  Returns an empty dict when the file is
        empty or missing.

    Raises:
        FileNotFoundError: If *path* does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if data is None:
        logger.warning("Config file %s is empty – using defaults.", path)
        return {}

    return data


# ===================================================================
# Drift-threshold models
# ===================================================================

class ThresholdPair(BaseModel):
    """A warning / critical threshold pair for a single drift metric."""

    warning: float
    critical: float


class DataDriftThresholds(BaseModel):
    """Thresholds for feature-level data-drift detection."""

    psi: ThresholdPair
    kl_divergence: ThresholdPair


class PredictionDriftThresholds(BaseModel):
    """Thresholds for prediction-distribution drift detection."""

    hellinger: ThresholdPair
    distribution_shift: ThresholdPair


class ConceptDriftThresholds(BaseModel):
    """Thresholds for concept-drift (performance degradation) detection."""

    accuracy_drop: ThresholdPair
    f1_drop: ThresholdPair
    precision_drop: ThresholdPair
    recall_drop: ThresholdPair


class DriftThresholds(BaseModel):
    """Aggregated drift thresholds for all drift families."""

    data_drift: DataDriftThresholds
    prediction_drift: PredictionDriftThresholds
    concept_drift: ConceptDriftThresholds


# ===================================================================
# Infrastructure / service models
# ===================================================================

class DatabaseSettings(BaseModel):
    """Database connection configuration."""

    url: str = "sqlite:///data/predictions.db"


class ProjectSettings(BaseModel):
    """Project metadata."""

    name: str = "MLOps Drift Monitor"
    version: str = "1.0.0"


class MLflowSettings(BaseModel):
    """MLflow tracking and model-registry configuration."""

    tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    experiment_name: str = "drift-monitor"
    artifact_location: str = "./models"
    model_name: str = "FraudDetector"


class ModelParams(BaseModel):
    """Hyper-parameters for the production model."""

    n_estimators: int = 200
    max_depth: int = 15
    min_samples_split: int = 5
    class_weight: str = "balanced"
    random_state: int = 42


class MonitoringSettings(BaseModel):
    """Monitoring loop configuration."""

    check_interval_minutes: int = 5
    window_size: int = 500
    min_samples: int = 100


class APISettings(BaseModel):
    """FastAPI service binding configuration."""

    host: str = "0.0.0.0"
    port: int = 8000


class DashboardSettings(BaseModel):
    """Dash dashboard binding and refresh configuration."""

    host: str = "0.0.0.0"
    port: int = 8050
    refresh_interval_ms: int = 5000


# ===================================================================
# Alerting models
# ===================================================================

class ConsoleAlertSettings(BaseModel):
    """Console (stdout/logging) alert channel."""

    enabled: bool = True
    log_level: str = "INFO"


class SlackAlertSettings(BaseModel):
    """Slack webhook alert channel."""

    enabled: bool = False
    webhook_url: str = ""
    channel: str = "#ml-alerts"
    username: str = "MLOps Monitor"
    icon_emoji: str = ":robot_face:"


class EmailAlertSettings(BaseModel):
    """Email (SMTP) alert channel."""

    enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    from_address: str = "mlops-monitor@example.com"
    to_addresses: list[str] = Field(default_factory=list)


class DeduplicationSettings(BaseModel):
    """Alert deduplication window."""

    window_minutes: int = 30


class AlertingSettings(BaseModel):
    """Aggregated alerting configuration across all channels."""

    console: ConsoleAlertSettings = Field(default_factory=ConsoleAlertSettings)
    slack: SlackAlertSettings = Field(default_factory=SlackAlertSettings)
    email: EmailAlertSettings = Field(default_factory=EmailAlertSettings)
    deduplication: DeduplicationSettings = Field(
        default_factory=DeduplicationSettings,
    )


# ===================================================================
# Top-level settings
# ===================================================================

class Settings(BaseModel):
    """Root settings object that aggregates every configuration section.

    Instantiated once via :func:`get_settings` and cached for the
    lifetime of the process.
    """

    project: ProjectSettings = Field(default_factory=ProjectSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    model_params: ModelParams = Field(default_factory=ModelParams)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    api: APISettings = Field(default_factory=APISettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)
    drift_thresholds: DriftThresholds | None = None
    alerting: AlertingSettings = Field(default_factory=AlertingSettings)


# ===================================================================
# Env-var override helpers
# ===================================================================

def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Overlay select environment variables onto the raw config dict.

    Only a curated set of env vars is supported so that we don't
    accidentally leak arbitrary env state into configuration.

    Supported env vars:
        - ``DATABASE_URL``       → database.url
        - ``MLFLOW_TRACKING_URI`` → mlflow.tracking_uri
        - ``SLACK_WEBHOOK_URL``  → alerting.slack.webhook_url
        - ``SMTP_HOST``          → alerting.email.smtp_host
        - ``SMTP_PORT``          → alerting.email.smtp_port
        - ``SMTP_USER``          → alerting.email.smtp_user
        - ``SMTP_PASSWORD``      → alerting.email.smtp_password
    """
    env_map: list[tuple[str, list[str]]] = [
        ("DATABASE_URL", ["database", "url"]),
        ("MLFLOW_TRACKING_URI", ["mlflow", "tracking_uri"]),
        ("SLACK_WEBHOOK_URL", ["alerting", "slack", "webhook_url"]),
        ("SMTP_HOST", ["alerting", "email", "smtp_host"]),
        ("SMTP_PORT", ["alerting", "email", "smtp_port"]),
        ("SMTP_USER", ["alerting", "email", "smtp_user"]),
        ("SMTP_PASSWORD", ["alerting", "email", "smtp_password"]),
        ("API_HOST", ["api", "host"]),
        ("API_PORT", ["api", "port"]),
        ("DASHBOARD_HOST", ["dashboard", "host"]),
        ("DASHBOARD_PORT", ["dashboard", "port"]),
    ]

    for env_key, path in env_map:
        value = os.environ.get(env_key)
        if value is not None:
            _nested_set(raw, path, value)
            logger.info("Env override applied: %s", env_key)

    if recipients := os.environ.get("ALERT_EMAIL_TO"):
        parsed = [item.strip() for item in recipients.split(",") if item.strip()]
        _nested_set(raw, ["alerting", "email", "to_addresses"], parsed)
        logger.info("Env override applied: ALERT_EMAIL_TO")

    if os.environ.get("SLACK_WEBHOOK_URL"):
        _nested_set(raw, ["alerting", "slack", "enabled"], True)

    if os.environ.get("SMTP_HOST"):
        _nested_set(raw, ["alerting", "email", "enabled"], True)

    return raw


def _nested_set(data: dict[str, Any], keys: list[str], value: Any) -> None:
    """Set a value in a nested dict, creating intermediate dicts as needed."""
    for key in keys[:-1]:
        data = data.setdefault(key, {})
    data[keys[-1]] = value


# ===================================================================
# Factory
# ===================================================================

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and return the application ``Settings`` singleton.

    Merges the three YAML configuration files located under
    ``PROJECT_ROOT/configs/`` and applies environment-variable overrides.

    Returns:
        Fully-validated, immutable ``Settings`` instance.

    Raises:
        FileNotFoundError: If any required config file is missing.
        pydantic.ValidationError: If the merged config violates the schema.
    """
    # Load .env file (if present) so env vars are available for overrides
    dotenv_path = PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)
        logger.info("Loaded .env from %s", dotenv_path)

    # --- YAML files ---------------------------------------------------
    base_cfg = _load_yaml(PROJECT_ROOT / "configs" / "base_config.yaml")
    drift_cfg = _load_yaml(PROJECT_ROOT / "configs" / "drift_thresholds.yaml")
    alert_cfg = _load_yaml(PROJECT_ROOT / "configs" / "alerting_config.yaml")

    # --- Merge into a single dict matching Settings schema ------------
    merged: dict[str, Any] = {
        "project": base_cfg.get("project", {}),
        "database": base_cfg.get("database", {}),
        "mlflow": base_cfg.get("mlflow", {}),
        "model_params": base_cfg.get("model", {}).get("params", {}),
        "monitoring": base_cfg.get("monitoring", {}),
        "api": base_cfg.get("api", {}),
        "dashboard": base_cfg.get("dashboard", {}),
        "drift_thresholds": drift_cfg,
        "alerting": alert_cfg.get("alerting", {}),
    }

    # --- Env-var overrides --------------------------------------------
    merged = _apply_env_overrides(merged)

    settings = Settings.model_validate(merged)
    logger.info(
        "Settings loaded successfully (db=%s, mlflow=%s).",
        settings.database.url,
        settings.mlflow.tracking_uri,
    )
    return settings
