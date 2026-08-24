"""The `Alerter` abstraction.

Stage 1 prints alerts to the console; Stage 2/3 post MS Teams adaptive
cards via webhook. `configs/alerting_config.yaml`'s `active_channel`
picks the implementation; `pdm.monitoring` and `pdm.serving` only ever
call `Alerter.send`, never a concrete channel class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """A single alert-worthy event, channel-agnostic."""

    title: str
    message: str
    severity: AlertSeverity
    source: str  # e.g. "drift_monitor", "serving_api", "training_pipeline"
    details: dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class Alerter(ABC):
    """Abstract alert channel."""

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Deliver `alert`. Returns True on confirmed delivery, False on
        a handled failure (e.g. webhook timeout) -- callers should log a
        False return but must never let alerting failures crash the
        caller (fail safe in serving/monitoring paths)."""
        raise NotImplementedError
