"""Tests for pdm.alerting: console/teams channels, severity filtering, factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from pdm.alerting.base import Alert, AlertSeverity
from pdm.alerting.console_alerter import ConsoleAlerter
from pdm.alerting.factory import get_alerter
from pdm.alerting.teams_alerter import TeamsAlerter, _build_adaptive_card
from pdm.config.schemas import (
    AlertingConfig,
    AlertTriggerConfig,
    ConsoleAlertConfig,
    TeamsWebhookAlertConfig,
)


def _alert(severity: AlertSeverity = AlertSeverity.WARNING) -> Alert:
    return Alert(
        title="Drift detected",
        message="3/14 sensor columns drifted.",
        severity=severity,
        source="drift_monitor",
        details={"share_of_drifted_columns": "0.21"},
    )


class TestConsoleAlerter:
    def test_sends_alert_at_or_above_min_severity(self):
        alerter = ConsoleAlerter(ConsoleAlertConfig(min_severity="info"))
        assert alerter.send(_alert(AlertSeverity.WARNING)) is True

    def test_suppresses_alert_below_min_severity(self):
        alerter = ConsoleAlerter(ConsoleAlertConfig(min_severity="critical"))
        assert alerter.send(_alert(AlertSeverity.WARNING)) is False

    def test_exact_min_severity_still_sends(self):
        alerter = ConsoleAlerter(ConsoleAlertConfig(min_severity="warning"))
        assert alerter.send(_alert(AlertSeverity.WARNING)) is True


class TestTeamsAlerter:
    def _config(self, **overrides) -> TeamsWebhookAlertConfig:
        defaults = dict(
            webhook_url="https://example.com/webhook",
            min_severity="warning",
            retry_attempts=2,
            timeout_s=1,
        )
        defaults.update(overrides)
        return TeamsWebhookAlertConfig(**defaults)

    def test_adaptive_card_structure(self):
        card = _build_adaptive_card(_alert(), theme_color="FF0000")
        assert card["type"] == "message"
        content = card["attachments"][0]["content"]
        assert content["type"] == "AdaptiveCard"
        assert any("Drift detected" in block.get("text", "") for block in content["body"])

    @patch("pdm.alerting.teams_alerter.requests.post")
    def test_send_success(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200, raise_for_status=lambda: None)
        alerter = TeamsAlerter(self._config())
        assert alerter.send(_alert()) is True
        mock_post.assert_called_once()

    def test_below_min_severity_never_calls_requests(self):
        with patch("pdm.alerting.teams_alerter.requests.post") as mock_post:
            alerter = TeamsAlerter(self._config(min_severity="critical"))
            assert alerter.send(_alert(AlertSeverity.WARNING)) is False
            mock_post.assert_not_called()

    @patch("pdm.alerting.teams_alerter.time.sleep", return_value=None)
    @patch("pdm.alerting.teams_alerter.requests.post")
    def test_retries_then_fails_gracefully(self, mock_post, _mock_sleep):
        mock_post.side_effect = requests.ConnectionError("network down")
        alerter = TeamsAlerter(self._config(retry_attempts=3))
        result = alerter.send(_alert())
        assert result is False
        assert mock_post.call_count == 3

    @patch("pdm.alerting.teams_alerter.time.sleep", return_value=None)
    @patch("pdm.alerting.teams_alerter.requests.post")
    def test_recovers_after_transient_failure(self, mock_post, _mock_sleep):
        mock_post.side_effect = [
            requests.ConnectionError("transient"),
            MagicMock(status_code=200, raise_for_status=lambda: None),
        ]
        alerter = TeamsAlerter(self._config(retry_attempts=3))
        assert alerter.send(_alert()) is True
        assert mock_post.call_count == 2


class TestAlertingFactory:
    def _full_config(self, active_channel: str) -> AlertingConfig:
        return AlertingConfig(
            active_channel=active_channel,
            console=ConsoleAlertConfig(),
            teams_webhook=TeamsWebhookAlertConfig(webhook_url="https://example.com/hook"),
            triggers=AlertTriggerConfig(),
        )

    def test_returns_console_alerter(self):
        assert isinstance(get_alerter(self._full_config("console")), ConsoleAlerter)

    def test_returns_teams_alerter(self):
        assert isinstance(get_alerter(self._full_config("teams_webhook")), TeamsAlerter)
