"""Alerting subsystem for the MLOps monitoring platform.

Exports:
    AlertManager  — central alert dispatcher (dedup, persist, route).
    SlackNotifier — Slack incoming-webhook notifier.
    EmailNotifier — SMTP / HTML email notifier.
"""

from .alert_manager import AlertManager
from .email_notifier import EmailNotifier
from .slack_notifier import SlackNotifier

__all__: list[str] = [
    "AlertManager",
    "SlackNotifier",
    "EmailNotifier",
]
