"""Stage 1 `Alerter`: logs alerts to the console via loguru.

No network, no external service -- this exists so the full monitoring/
serving alert-firing code path is exercisable (and testable) from day
one, without needing MS Teams configured yet.
"""

from __future__ import annotations

from loguru import logger

from pdm.alerting.base import Alert, Alerter, AlertSeverity
from pdm.config.schemas import SEVERITY_ORDER, ConsoleAlertConfig

_LOG_FUNCS = {
    AlertSeverity.INFO: logger.info,
    AlertSeverity.WARNING: logger.warning,
    AlertSeverity.CRITICAL: logger.critical,
}


class ConsoleAlerter(Alerter):
    def __init__(self, config: ConsoleAlertConfig) -> None:
        self._config = config

    def send(self, alert: Alert) -> bool:
        if SEVERITY_ORDER[alert.severity.value] < SEVERITY_ORDER[self._config.min_severity]:
            return False
        log_fn = _LOG_FUNCS[alert.severity]
        log_fn("[{}] {} -- {}", alert.source, alert.title, alert.message)
        for key, value in alert.details.items():
            log_fn("    {}: {}", key, value)
        return True
