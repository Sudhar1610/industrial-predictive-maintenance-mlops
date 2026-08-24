"""Scheduled data-drift check: compares recently-scored sensor readings
(from `PredictionLogger`) against the training-time reference
distribution, and fires an alert through `pdm.alerting` on breach.

A plain function, not a long-running service -- callers (a Prefect
deployment schedule, cron, or a manual run) decide the cadence. This is
what keeps "no extra services unless we choose to run them" true on
Stage 3.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from pdm.alerting.base import Alert, AlertSeverity
from pdm.alerting.factory import get_alerter
from pdm.config.schemas import AlertingConfig, FeatureEngineeringConfig
from pdm.evaluation.drift import compute_data_drift
from pdm.monitoring.prediction_logger import PredictionLogger
from pdm.monitoring.prometheus_metrics import record_drift_breach

MIN_CURRENT_ROWS = 30


def run_drift_check(
    reference_df: pd.DataFrame,
    feature_config: FeatureEngineeringConfig,
    alerting_config: AlertingConfig,
    prediction_logger: PredictionLogger | None = None,
    min_current_rows: int = MIN_CURRENT_ROWS,
) -> dict | None:
    """Returns the drift result dict (see `pdm.evaluation.drift.compute_data_drift`),
    or `None` if the check was skipped (drift alerting disabled, or not
    enough recently-logged predictions yet to compare against).
    """
    if not alerting_config.triggers.drift_breach:
        logger.info("Drift alerting disabled via config; skipping drift check.")
        return None

    prediction_logger = prediction_logger or PredictionLogger()
    current_df = prediction_logger.read_recent()
    if len(current_df) < min_current_rows:
        logger.info(
            "Only {} recent predictions logged (need >= {}); skipping drift check.",
            len(current_df),
            min_current_rows,
        )
        return None

    current_sensors = pd.json_normalize(current_df["latest_reading"])
    sensor_columns = [
        c
        for c in feature_config.sensor_columns
        if c in current_sensors.columns and c in reference_df.columns
    ]
    if not sensor_columns:
        logger.warning(
            "No overlapping sensor columns between reference data and recent predictions; "
            "skipping drift check."
        )
        return None

    result = compute_data_drift(reference_df, current_sensors, columns=sensor_columns)

    if result["dataset_drift"]:
        record_drift_breach()
        alerter = get_alerter(alerting_config)
        alerter.send(
            Alert(
                title="Data drift detected",
                message=(
                    f"{result['number_of_drifted_columns']} of {len(sensor_columns)} "
                    f"monitored sensor columns have drifted from the training distribution."
                ),
                severity=AlertSeverity.WARNING,
                source="drift_monitor",
                details={
                    "share_of_drifted_columns": f"{result['share_of_drifted_columns']:.2f}",
                    "columns_checked": ", ".join(sensor_columns),
                },
            )
        )

    return result
