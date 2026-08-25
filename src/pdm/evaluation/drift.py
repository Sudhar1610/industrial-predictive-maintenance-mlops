"""Data-drift detection (Evidently), run against a fixed reference
distribution (the training set) and a current window of scored data.

`pdm.monitoring.drift_job` schedules this to run periodically against
recent prediction-logged data; a drift breach fires an alert through
`pdm.alerting`. Running it here as a plain function (not a class) keeps
it trivially callable from a Prefect task, a cron-style script, or a test.
"""

from __future__ import annotations

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from loguru import logger


def compute_data_drift(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, columns: list[str]
) -> dict:
    """Run Evidently's `DataDriftPreset` comparing `current_df` against
    `reference_df` on `columns`. Returns the report as a plain dict
    (JSON-serializable) plus a top-level `dataset_drift` bool.
    """
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df[columns],
        current_data=current_df[columns],
    )
    result = report.as_dict()
    dataset_drift = result["metrics"][0]["result"]["dataset_drift"]
    if dataset_drift:
        logger.warning(
            "Data drift detected: {}/{} columns drifted.",
            result["metrics"][0]["result"]["number_of_drifted_columns"],
            result["metrics"][0]["result"]["number_of_columns"],
        )
    return {
        "dataset_drift": dataset_drift,
        "share_of_drifted_columns": result["metrics"][0]["result"]["share_of_drifted_columns"],
        "number_of_drifted_columns": result["metrics"][0]["result"]["number_of_drifted_columns"],
        "full_report": result,
    }
