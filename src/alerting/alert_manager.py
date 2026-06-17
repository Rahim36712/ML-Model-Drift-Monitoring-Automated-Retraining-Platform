"""Central alert dispatcher for the MLOps monitoring platform.

``AlertManager`` is the single entry-point for raising, deduplicating,
persisting, and routing drift alerts.  It delegates delivery to one or
more *notifier* back-ends (console, Slack, email) based on the active
``AlertingSettings``.

Usage::

    from src.alerting import AlertManager
    from src.data.database import get_database
    from src.config.settings import get_settings

    db = get_database()
    mgr = AlertManager(db, settings=get_settings().alerting)
    mgr.send_alert(
        severity="CRITICAL",
        drift_type="data",
        metric_name="psi",
        metric_value=0.32,
        threshold=0.25,
    )
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from sqlalchemy import func

from src.data.database import Alert

from .email_notifier import EmailNotifier
from .slack_notifier import SlackNotifier

if TYPE_CHECKING:
    from src.config.settings import AlertingSettings
    from src.data.database import DatabaseManager

logger = logging.getLogger(__name__)

# Emoji mapping for formatted alert strings
_SEVERITY_EMOJIS: dict[str, str] = {
    "CRITICAL": "🔴",
    "WARNING":  "⚠️",
    "RESOLVED": "✅",
}


class AlertManager:
    """Orchestrates alert creation, deduplication, and multi-channel dispatch.

    Args:
        db: ``DatabaseManager`` instance used for persistence.
        settings: Optional ``AlertingSettings`` controlling which
            notification channels are enabled.  Falls back to safe
            defaults (console-only) when ``None``.
    """

    def __init__(
        self,
        db: DatabaseManager,
        settings: AlertingSettings | None = None,
    ) -> None:
        self._db = db
        self._settings = settings

        # Deduplication window (minutes)
        self._dedup_window_minutes: int = (
            settings.deduplication.window_minutes
            if settings and settings.deduplication
            else 30
        )

        # --- Notifier registry -----------------------------------------
        self._slack_notifier: SlackNotifier | None = None
        self._email_notifier: EmailNotifier | None = None

        # Console is always enabled (uses the standard logger).
        if settings and settings.slack.enabled and settings.slack.webhook_url:
            self._slack_notifier = SlackNotifier(
                webhook_url=settings.slack.webhook_url,
                channel=settings.slack.channel,
                username=settings.slack.username,
            )
            logger.info("Slack notifier enabled (channel=%s).", settings.slack.channel)

        if settings and settings.email.enabled and settings.email.smtp_host:
            self._email_notifier = EmailNotifier(
                smtp_host=settings.email.smtp_host,
                smtp_port=settings.email.smtp_port,
                smtp_user=settings.email.smtp_user,
                smtp_password=settings.email.smtp_password,
                from_address=settings.email.from_address,
                to_addresses=settings.email.to_addresses,
            )
            logger.info("Email notifier enabled (host=%s).", settings.email.smtp_host)

        logger.info(
            "AlertManager initialised (dedup_window=%d min, slack=%s, email=%s).",
            self._dedup_window_minutes,
            self._slack_notifier is not None,
            self._email_notifier is not None,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_alert(
        self,
        severity: str,
        drift_type: str,
        metric_name: str,
        metric_value: float,
        threshold: float,
        message: str | None = None,
    ) -> None:
        """Format, persist, deduplicate, and route an alert.

        Args:
            severity: ``CRITICAL``, ``WARNING``, or ``RESOLVED``.
            drift_type: The drift family (``data``, ``prediction``,
                ``concept``).
            metric_name: Specific metric that triggered the alert
                (``psi``, ``hellinger``, ``f1_drop``, …).
            metric_value: Observed metric value.
            threshold: The threshold that was breached.
            message: Optional override for the formatted alert body.
        """
        formatted = message or self._format_alert_message(
            severity, drift_type, metric_name, metric_value, threshold,
        )

        # 1. Persist in the database
        self._store_alert(severity, drift_type, formatted)

        # 2. Deduplication check
        if self._is_duplicate(drift_type, metric_name):
            logger.info(
                "Alert suppressed (dedup): %s/%s within %d-min window.",
                drift_type,
                metric_name,
                self._dedup_window_minutes,
            )
            return

        # 3. Route to enabled channels
        self._route_to_channels(formatted, severity, drift_type)

    def get_active_alerts(self, limit: int = 20) -> list[Alert]:
        """Return the most recent unacknowledged alerts.

        Args:
            limit: Maximum number of alerts to return.

        Returns:
            List of ``Alert`` ORM instances ordered newest-first.
        """
        with self._db.get_session() as session:
            return (
                session.query(Alert)
                .filter(Alert.acknowledged.is_(False))
                .order_by(Alert.timestamp.desc())
                .limit(limit)
                .all()
            )

    def get_alert_history(self, limit: int = 50) -> list[Alert]:
        """Return recent alerts regardless of acknowledgement state.

        Args:
            limit: Maximum number of alerts to return.

        Returns:
            List of ``Alert`` ORM instances ordered newest-first.
        """
        with self._db.get_session() as session:
            return (
                session.query(Alert)
                .order_by(Alert.timestamp.desc())
                .limit(limit)
                .all()
            )

    def acknowledge_alert(self, alert_id: int) -> bool:
        """Mark a specific alert as acknowledged.

        Args:
            alert_id: Primary key of the alert row.

        Returns:
            ``True`` if the alert was found and updated, ``False``
            otherwise.
        """
        try:
            with self._db.get_session() as session:
                alert: Alert | None = session.get(Alert, alert_id)
                if alert is None:
                    logger.warning("Alert id=%d not found for acknowledgement.", alert_id)
                    return False
                alert.acknowledged = True
                logger.info("Alert id=%d acknowledged.", alert_id)
                return True
        except Exception:  # noqa: BLE001
            logger.exception("Error acknowledging alert id=%d.", alert_id)
            return False

    def get_alert_stats(self) -> dict:
        """Compute summary statistics for the alert table.

        Returns:
            Dictionary with keys:
            - ``total``: Total number of alerts ever recorded.
            - ``active``: Unacknowledged alert count.
            - ``by_severity``: ``{severity: count}`` mapping.
            - ``by_type``: ``{drift_type: count}`` mapping.
        """
        try:
            with self._db.get_session() as session:
                total: int = session.query(func.count(Alert.id)).scalar() or 0
                active: int = (
                    session.query(func.count(Alert.id))
                    .filter(Alert.acknowledged.is_(False))
                    .scalar()
                    or 0
                )

                by_severity: dict[str, int] = dict(
                    session.query(Alert.severity, func.count(Alert.id))
                    .group_by(Alert.severity)
                    .all()
                )

                by_type: dict[str, int] = dict(
                    session.query(Alert.drift_type, func.count(Alert.id))
                    .group_by(Alert.drift_type)
                    .all()
                )

            return {
                "total": total,
                "active": active,
                "by_severity": by_severity,
                "by_type": by_type,
            }
        except Exception:  # noqa: BLE001
            logger.exception("Error computing alert statistics.")
            return {"total": 0, "active": 0, "by_severity": {}, "by_type": {}}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_alert_message(
        self,
        severity: str,
        drift_type: str,
        metric_name: str,
        metric_value: float,
        threshold: float,
    ) -> str:
        """Build a human-readable alert string with box-drawing decoration.

        Args:
            severity: Alert severity level.
            drift_type: Drift family identifier.
            metric_name: The metric that triggered the alert.
            metric_value: Observed value.
            threshold: The threshold that was breached.

        Returns:
            Multi-line formatted alert string.
        """
        emoji = _SEVERITY_EMOJIS.get(severity.upper(), "ℹ️")
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        breach = "EXCEEDED" if metric_value > threshold else "OK"

        header = f"{emoji} {severity.upper()} — {drift_type.upper()} DRIFT"
        lines = [
            "┌─────────────────────────────────────────────┐",
            f"│  {header:<43s}│",
            "├─────────────────────────────────────────────┤",
            f"│  Metric   : {metric_name:<31s}│",
            f"│  Value    : {metric_value:<31.6f}│",
            f"│  Threshold: {threshold:<31.6f}│",
            f"│  Status   : {breach:<31s}│",
            f"│  Time     : {timestamp:<31s}│",
            "└─────────────────────────────────────────────┘",
        ]
        return "\n".join(lines)

    def _is_duplicate(self, drift_type: str, metric_name: str) -> bool:
        """Check whether the same alert was dispatched within the dedup window.

        Args:
            drift_type: Drift family identifier.
            metric_name: Metric name to match against the message body.

        Returns:
            ``True`` if a matching unacknowledged alert exists within
            the deduplication window.
        """
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(
                minutes=self._dedup_window_minutes,
            )
            with self._db.get_session() as session:
                count: int = (
                    session.query(func.count(Alert.id))
                    .filter(
                        Alert.drift_type == drift_type,
                        Alert.message.contains(metric_name),
                        Alert.timestamp >= cutoff,
                    )
                    .scalar()
                    or 0
                )
                # count > 1 because the current alert has already been stored
                return count > 1
        except Exception:  # noqa: BLE001
            logger.exception("Deduplication check failed – allowing alert through.")
            return False

    def _store_alert(
        self,
        severity: str,
        drift_type: str,
        message: str,
    ) -> None:
        """Persist an alert row in the database.

        Args:
            severity: Alert severity level.
            drift_type: Drift family identifier.
            message: Formatted alert body.
        """
        try:
            with self._db.get_session() as session:
                alert = Alert(
                    severity=severity.upper(),
                    drift_type=drift_type,
                    message=message,
                    channel="all",
                    acknowledged=False,
                )
                session.add(alert)
            logger.debug("Alert stored in database (severity=%s, type=%s).", severity, drift_type)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to store alert in database.")

    def _route_to_channels(
        self,
        message: str,
        severity: str,
        drift_type: str,
    ) -> None:
        """Dispatch the alert to all enabled notification channels.

        Args:
            message: Formatted alert body.
            severity: Alert severity level.
            drift_type: Drift family identifier.
        """
        # Console (always on)
        log_method = logger.critical if severity.upper() == "CRITICAL" else logger.warning
        log_method("\n%s", message)

        # Slack
        if self._slack_notifier is not None:
            try:
                self._slack_notifier.send(message, severity=severity)
            except Exception:  # noqa: BLE001
                logger.exception("Slack notification dispatch failed.")

        # Email
        if self._email_notifier is not None:
            try:
                subject = f"{drift_type.upper()} Drift — {severity.upper()}"
                self._email_notifier.send(subject, message, severity=severity)
            except Exception:  # noqa: BLE001
                logger.exception("Email notification dispatch failed.")
