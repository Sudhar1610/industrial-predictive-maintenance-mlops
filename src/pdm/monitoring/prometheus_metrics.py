"""Prometheus metrics for the serving API and monitoring jobs.

A dedicated `CollectorRegistry` (rather than the global default registry)
so tests can import this module repeatedly without "duplicate metric"
registration errors, and so a future multi-process deployment can wire
up `multiprocess_mode` cleanly if needed.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest

REGISTRY = CollectorRegistry()

PREDICTIONS_TOTAL = Counter(
    "pdm_predictions_total",
    "Total predictions served",
    ["model_backend", "will_fail"],
    registry=REGISTRY,
)

PREDICTION_LATENCY_SECONDS = Histogram(
    "pdm_prediction_latency_seconds",
    "Prediction request latency in seconds",
    ["model_backend"],
    registry=REGISTRY,
)

FAILURE_PROBABILITY = Histogram(
    "pdm_failure_probability",
    "Distribution of predicted failure probabilities",
    buckets=[i / 10 for i in range(11)],
    registry=REGISTRY,
)

DRIFT_BREACHES_TOTAL = Counter(
    "pdm_drift_breaches_total",
    "Total data-drift breaches detected by the scheduled drift job",
    registry=REGISTRY,
)


def record_prediction(model_backend: str, will_fail: bool, failure_probability: float, latency_seconds: float) -> None:
    PREDICTIONS_TOTAL.labels(model_backend=model_backend, will_fail=str(will_fail)).inc()
    PREDICTION_LATENCY_SECONDS.labels(model_backend=model_backend).observe(latency_seconds)
    FAILURE_PROBABILITY.observe(failure_probability)


def record_drift_breach() -> None:
    DRIFT_BREACHES_TOTAL.inc()


def render_latest() -> bytes:
    """Render all metrics in the Prometheus text exposition format, for
    a `/metrics` endpoint."""
    return generate_latest(REGISTRY)
