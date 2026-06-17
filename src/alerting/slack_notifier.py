"""Slack webhook notifier for MLOps drift alerts.

Posts richly-formatted Block Kit messages to a Slack channel via an
incoming webhook URL.  Colour-coded sidebar indicates severity.

Design constraint:
    All public methods catch exceptions internally and return ``False``
    on failure so that a misconfigured Slack integration can never crash
    the monitoring pipeline.

Example::

    notifier = SlackNotifier(
        webhook_url="https://hooks.slack.com/services/YOUR_WORKSPACE/YOUR_CHANNEL/YOUR_TOKEN",
    )
    notifier.send("PSI exceeded critical threshold (0.32)", severity="CRITICAL")
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Slack sidebar colour map (hex)
_SEVERITY_COLOURS: dict[str, str] = {
    "CRITICAL": "#E74C3C",   # red
    "WARNING":  "#F39C12",   # yellow / amber
    "RESOLVED": "#2ECC71",   # green
    "INFO":     "#3498DB",   # blue
}


class SlackNotifier:
    """Sends alert messages to Slack via an incoming webhook.

    Args:
        webhook_url: Full Slack incoming-webhook URL.
        channel: Target channel override (e.g. ``#ml-alerts``).
        username: Display name for the bot post.
    """

    def __init__(
        self,
        webhook_url: str,
        channel: str = "#ml-alerts",
        username: str = "MLOps Monitor",
    ) -> None:
        self._webhook_url: str = webhook_url
        self._channel: str = channel
        self._username: str = username
        logger.info(
            "SlackNotifier initialised (channel=%s, username=%s).",
            channel,
            username,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, message: str, severity: str = "INFO") -> bool:
        """Post a formatted alert message to Slack.

        Args:
            message: The alert body text.
            severity: One of ``CRITICAL``, ``WARNING``, ``RESOLVED``,
                or ``INFO``.

        Returns:
            ``True`` if the webhook accepted the payload, ``False``
            otherwise.  Errors are logged but never raised.
        """
        try:
            payload = self._build_payload(message, severity)
            response = requests.post(
                self._webhook_url,
                json=payload,
                timeout=10,
            )
            if response.status_code == 200 and response.text == "ok":
                logger.info("Slack alert sent successfully (severity=%s).", severity)
                return True

            logger.warning(
                "Slack webhook returned non-OK: status=%d body=%s",
                response.status_code,
                response.text[:200],
            )
            return False

        except requests.exceptions.Timeout:
            logger.error("Slack webhook timed out after 10 s.")
            return False
        except requests.exceptions.ConnectionError:
            logger.error("Could not connect to Slack webhook URL.")
            return False
        except Exception:  # noqa: BLE001
            logger.exception("Unexpected error sending Slack alert.")
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_payload(self, message: str, severity: str) -> dict[str, Any]:
        """Build a Slack Block Kit payload with colour-coded attachment.

        The layout consists of:
        1. A header block with the severity label.
        2. A divider.
        3. A section block containing the alert message.
        4. A context block with the UTC timestamp.

        Args:
            message: Alert body text.
            severity: Severity level string.

        Returns:
            Dictionary suitable for ``requests.post(json=...)``.
        """
        colour = _SEVERITY_COLOURS.get(severity.upper(), _SEVERITY_COLOURS["INFO"])
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        severity_emoji_map: dict[str, str] = {
            "CRITICAL": "🔴",
            "WARNING": "⚠️",
            "RESOLVED": "✅",
            "INFO": "ℹ️",
        }
        emoji = severity_emoji_map.get(severity.upper(), "ℹ️")

        blocks: list[dict[str, Any]] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} MLOps Alert — {severity.upper()}",
                    "emoji": True,
                },
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message,
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"📅 {timestamp}",
                    },
                ],
            },
        ]

        return {
            "channel": self._channel,
            "username": self._username,
            "icon_emoji": ":robot_face:",
            "attachments": [
                {
                    "color": colour,
                    "blocks": blocks,
                },
            ],
        }
