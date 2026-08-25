"""Builds the active `Alerter` from `alerting_config.yaml`."""

from __future__ import annotations

from pdm.alerting.base import Alerter
from pdm.alerting.console_alerter import ConsoleAlerter
from pdm.alerting.teams_alerter import TeamsAlerter
from pdm.config.schemas import AlertingConfig


def get_alerter(config: AlertingConfig) -> Alerter:
    """Instantiate the `Alerter` implementation named by
    `config.active_channel`."""
    if config.active_channel == "console":
        return ConsoleAlerter(config.console)
    if config.active_channel == "teams_webhook":
        return TeamsAlerter(config.teams_webhook)
    raise ValueError(f"Unknown active_channel: {config.active_channel!r}")
