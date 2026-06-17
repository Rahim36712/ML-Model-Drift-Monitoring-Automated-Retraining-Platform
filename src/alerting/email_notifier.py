"""Email SMTP notifier for MLOps drift alerts.

Sends HTML-formatted alert emails through an SMTP relay with STARTTLS.
Each email includes a colour-coded header bar matching the severity level.

Design constraint:
    All public methods catch exceptions internally and return ``False``
    on failure so that a misconfigured email integration can never crash
    the monitoring pipeline.

Example::

    notifier = EmailNotifier(
        smtp_host="smtp.gmail.com",
        smtp_port=587,
        smtp_user="bot@example.com",
        smtp_password="app-password",
        from_address="mlops@example.com",
        to_addresses=["team@example.com"],
    )
    notifier.send("PSI Breach", "PSI exceeded 0.32", severity="CRITICAL")
"""

from __future__ import annotations

import logging
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

# Header-bar background colour per severity
_SEVERITY_COLOURS: dict[str, str] = {
    "CRITICAL": "#E74C3C",
    "WARNING":  "#F39C12",
    "RESOLVED": "#2ECC71",
    "INFO":     "#3498DB",
}


class EmailNotifier:
    """Sends alert emails via SMTP with HTML formatting.

    Args:
        smtp_host: SMTP server hostname.
        smtp_port: SMTP server port (typically 587 for STARTTLS).
        smtp_user: Username for SMTP authentication.
        smtp_password: Password / app-password for SMTP authentication.
        from_address: The ``From:`` header value.
        to_addresses: List of recipient email addresses.
    """

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        smtp_user: str,
        smtp_password: str,
        from_address: str,
        to_addresses: list[str],
    ) -> None:
        self._smtp_host: str = smtp_host
        self._smtp_port: int = smtp_port
        self._smtp_user: str = smtp_user
        self._smtp_password: str = smtp_password
        self._from_address: str = from_address
        self._to_addresses: list[str] = to_addresses
        logger.info(
            "EmailNotifier initialised (host=%s, port=%d, recipients=%d).",
            smtp_host,
            smtp_port,
            len(to_addresses),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(
        self,
        subject: str,
        message: str,
        severity: str = "INFO",
    ) -> bool:
        """Send an HTML-formatted alert email.

        Args:
            subject: Email subject line.
            message: Plain-text alert body (rendered inside an HTML
                template).
            severity: One of ``CRITICAL``, ``WARNING``, ``RESOLVED``,
                or ``INFO``.

        Returns:
            ``True`` if the email was accepted by the SMTP server,
            ``False`` otherwise.  Errors are logged but never raised.
        """
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[MLOps {severity.upper()}] {subject}"
            msg["From"] = self._from_address
            msg["To"] = ", ".join(self._to_addresses)

            html_body = self._build_html(message, severity)
            msg.attach(MIMEText(message, "plain", "utf-8"))
            msg.attach(MIMEText(html_body, "html", "utf-8"))

            with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(self._smtp_user, self._smtp_password)
                server.sendmail(
                    self._from_address,
                    self._to_addresses,
                    msg.as_string(),
                )

            logger.info(
                "Email alert sent successfully (subject=%r, recipients=%d).",
                subject,
                len(self._to_addresses),
            )
            return True

        except smtplib.SMTPAuthenticationError:
            logger.error("SMTP authentication failed for user %s.", self._smtp_user)
            return False
        except smtplib.SMTPConnectError:
            logger.error(
                "Could not connect to SMTP server %s:%d.",
                self._smtp_host,
                self._smtp_port,
            )
            return False
        except TimeoutError:
            logger.error("SMTP connection timed out.")
            return False
        except Exception:  # noqa: BLE001
            logger.exception("Unexpected error sending email alert.")
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_html(self, message: str, severity: str) -> str:
        """Render an HTML email template with a severity-coloured header.

        Args:
            message: Plain-text alert body.
            severity: Severity level string.

        Returns:
            Complete HTML document string.
        """
        colour = _SEVERITY_COLOURS.get(severity.upper(), _SEVERITY_COLOURS["INFO"])
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Escape basic HTML entities in the message body
        safe_message = (
            message
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>\n")
        )

        return f"""\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;font-family:Arial,Helvetica,sans-serif;
             background-color:#f4f4f4;">
  <table width="100%" cellpadding="0" cellspacing="0"
         style="max-width:600px;margin:20px auto;background:#ffffff;
                border-radius:8px;overflow:hidden;
                box-shadow:0 2px 8px rgba(0,0,0,0.1);">
    <!-- Colour bar header -->
    <tr>
      <td style="background-color:{colour};padding:16px 24px;
                 color:#ffffff;font-size:20px;font-weight:bold;">
        MLOps Alert &mdash; {severity.upper()}
      </td>
    </tr>
    <!-- Body -->
    <tr>
      <td style="padding:24px;font-size:14px;color:#333333;
                 line-height:1.6;">
        {safe_message}
      </td>
    </tr>
    <!-- Footer -->
    <tr>
      <td style="padding:12px 24px;font-size:12px;color:#999999;
                 border-top:1px solid #eeeeee;">
        Sent by MLOps Monitoring Platform &bull; {timestamp}
      </td>
    </tr>
  </table>
</body>
</html>"""
