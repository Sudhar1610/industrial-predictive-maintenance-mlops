"""Stage 2/3 `Alerter`: posts MS Teams adaptive cards via incoming webhook.

Uses the plain `requests` library against the webhook URL rather than a
Teams SDK, since the office/plant network in Stage 2/3 has no package
registry access and this keeps the dependency surface to one already-
vendored HTTP client.
"""

from __future__ import annotations

import time

import requests
from loguru import logger

from pdm.alerting.base import Alert, Alerter, AlertSeverity
from pdm.config.schemas import SEVERITY_ORDER, TeamsWebhookAlertConfig

_SEVERITY_LABEL = {
    AlertSeverity.INFO: "INFO",
    AlertSeverity.WARNING: "WARNING",
    AlertSeverity.CRITICAL: "CRITICAL",
}


def _build_adaptive_card(alert: Alert, theme_color: str) -> dict:
    facts = [{"title": k, "value": str(v)} for k, v in alert.details.items()]
    facts.append({"title": "Source", "value": alert.source})
    facts.append({"title": "Time (UTC)", "value": alert.timestamp.isoformat()})

    card_body = [
        {
            "type": "TextBlock",
            "text": f"[{_SEVERITY_LABEL[alert.severity]}] {alert.title}",
            "weight": "Bolder",
            "size": "Medium",
            "wrap": True,
        },
        {"type": "TextBlock", "text": alert.message, "wrap": True},
        {"type": "FactSet", "facts": facts},
    ]
    adaptive_card = {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.4",
        "body": card_body,
    }
    return {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": adaptive_card,
            }
        ],
        # themeColor is ignored by the adaptive-card renderer but kept
        # for MS Teams connector clients that still read it.
        "themeColor": theme_color,
    }


class TeamsAlerter(Alerter):
    def __init__(self, config: TeamsWebhookAlertConfig) -> None:
        self._config = config

    def send(self, alert: Alert) -> bool:
        if SEVERITY_ORDER[alert.severity.value] < SEVERITY_ORDER[self._config.min_severity]:
            return False

        payload = _build_adaptive_card(alert, self._config.card_theme_color)

        for attempt in range(1, self._config.retry_attempts + 1):
            try:
                response = requests.post(
                    self._config.webhook_url, json=payload, timeout=self._config.timeout_s
                )
                response.raise_for_status()
                return True
            except requests.RequestException as exc:
                logger.warning(
                    "Teams alert delivery failed (attempt {}/{}): {}",
                    attempt,
                    self._config.retry_attempts,
                    exc,
                )
                if attempt < self._config.retry_attempts:
                    time.sleep(min(2**attempt, 10))

        logger.error("Teams alert delivery failed after {} attempts.", self._config.retry_attempts)
        return False
