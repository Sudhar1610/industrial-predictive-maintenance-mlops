"""Tests for pdm.monitoring: prediction logging, drift job, Prometheus metrics."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from pdm.alerting.base import AlertSeverity
from pdm.config.schemas import (
    AlertingConfig,
    AlertTriggerConfig,
    ConsoleAlertConfig,
    FeatureEngineeringConfig,
    TeamsWebhookAlertConfig,
)
from pdm.monitoring.drift_job import run_drift_check
from pdm.monitoring.prediction_logger import PredictionLogger
from pdm.monitoring.prometheus_metrics import (
    PREDICTIONS_TOTAL,
    record_drift_breach,
    record_prediction,
    render_latest,
)
from pdm.serving.schemas import PredictionResponse, PredictRequest, SensorReading


def _request_response_pair(
    unit_id: int, sensor_1_value: float
) -> tuple[PredictRequest, PredictionResponse]:
    request = PredictRequest(
        unit_id=unit_id,
        history=[SensorReading(cycle=1, values={"sensor_1": sensor_1_value})],
    )
    response = PredictionResponse(
        unit_id=unit_id,
        cycle=1,
        failure_probability=0.1,
        will_fail=False,
        remaining_useful_life=100.0,
        model_backend="xgboost",
    )
    return request, response


class TestPredictionLogger:
    def test_read_recent_empty_when_no_log_file(self, tmp_path: Path):
        logger_ = PredictionLogger(log_path=tmp_path / "predictions.jsonl")
        df = logger_.read_recent()
        assert df.empty

    def test_log_then_read_back(self, tmp_path: Path):
        logger_ = PredictionLogger(log_path=tmp_path / "predictions.jsonl")
        request, response = _request_response_pair(unit_id=1, sensor_1_value=42.0)
        logger_.log(request, response)

        df = logger_.read_recent()
        assert len(df) == 1
        assert df.iloc[0]["unit_id"] == 1
        assert df.iloc[0]["latest_reading"] == {"sensor_1": 42.0}

    def test_read_recent_respects_n(self, tmp_path: Path):
        logger_ = PredictionLogger(log_path=tmp_path / "predictions.jsonl")
        for i in range(5):
            request, response = _request_response_pair(unit_id=i, sensor_1_value=float(i))
            logger_.log(request, response)

        df = logger_.read_recent(n=2)
        assert len(df) == 2
        assert list(df["unit_id"]) == [3, 4]


class TestPrometheusMetrics:
    def test_record_prediction_increments_counter(self):
        before = PREDICTIONS_TOTAL.labels(model_backend="xgboost", will_fail="False")._value.get()
        record_prediction(
            model_backend="xgboost", will_fail=False, failure_probability=0.2, latency_seconds=0.01
        )
        after = PREDICTIONS_TOTAL.labels(model_backend="xgboost", will_fail="False")._value.get()
        assert after == before + 1

    def test_render_latest_contains_metric_names(self):
        record_prediction(
            model_backend="lstm", will_fail=True, failure_probability=0.9, latency_seconds=0.05
        )
        output = render_latest().decode("utf-8")
        assert "pdm_predictions_total" in output
        assert "pdm_prediction_latency_seconds" in output

    def test_record_drift_breach_increments_counter(self):
        from pdm.monitoring.prometheus_metrics import DRIFT_BREACHES_TOTAL

        before = DRIFT_BREACHES_TOTAL._value.get()
        record_drift_breach()
        after = DRIFT_BREACHES_TOTAL._value.get()
        assert after == before + 1


class TestDriftJob:
    def _alerting_config(self, drift_breach: bool = True) -> AlertingConfig:
        return AlertingConfig(
            active_channel="console",
            console=ConsoleAlertConfig(min_severity="info"),
            teams_webhook=TeamsWebhookAlertConfig(webhook_url="https://example.com/hook"),
            triggers=AlertTriggerConfig(drift_breach=drift_breach),
        )

    def _feature_config(self) -> FeatureEngineeringConfig:
        return FeatureEngineeringConfig(
            rolling_windows=[3],
            lag_steps=[1],
            degradation_slope_window=3,
            sensor_columns=["sensor_1", "sensor_2"],
        )

    def test_skips_when_drift_alerting_disabled(self, tmp_path: Path):
        logger_ = PredictionLogger(log_path=tmp_path / "predictions.jsonl")
        reference_df = pd.DataFrame({"sensor_1": [1.0] * 50, "sensor_2": [2.0] * 50})
        result = run_drift_check(
            reference_df, self._feature_config(), self._alerting_config(drift_breach=False), logger_
        )
        assert result is None

    def test_skips_when_not_enough_recent_predictions(self, tmp_path: Path):
        logger_ = PredictionLogger(log_path=tmp_path / "predictions.jsonl")
        request, response = _request_response_pair(unit_id=1, sensor_1_value=1.0)
        logger_.log(request, response)  # only 1 logged row

        reference_df = pd.DataFrame({"sensor_1": [1.0] * 50, "sensor_2": [2.0] * 50})
        result = run_drift_check(
            reference_df,
            self._feature_config(),
            self._alerting_config(),
            logger_,
            min_current_rows=30,
        )
        assert result is None

    def test_fires_alert_on_drift_breach(self, tmp_path: Path):
        logger_ = PredictionLogger(log_path=tmp_path / "predictions.jsonl")
        rng = np.random.default_rng(0)
        # Recent predictions carry a heavily shifted sensor_1 distribution.
        for i in range(40):
            request = PredictRequest(
                unit_id=i,
                history=[
                    SensorReading(
                        cycle=1,
                        values={
                            "sensor_1": float(rng.normal(50, 1)),
                            "sensor_2": float(rng.normal(2, 1)),
                        },
                    )
                ],
            )
            response = PredictionResponse(
                unit_id=i,
                cycle=1,
                failure_probability=0.1,
                will_fail=False,
                remaining_useful_life=100.0,
                model_backend="xgboost",
            )
            logger_.log(request, response)

        reference_df = pd.DataFrame(
            {"sensor_1": rng.normal(0, 1, 200), "sensor_2": rng.normal(2, 1, 200)}
        )

        with patch("pdm.monitoring.drift_job.get_alerter") as mock_get_alerter:
            mock_alerter = mock_get_alerter.return_value
            result = run_drift_check(
                reference_df,
                self._feature_config(),
                self._alerting_config(),
                logger_,
                min_current_rows=30,
            )

        assert result is not None
        assert result["dataset_drift"] is True
        mock_alerter.send.assert_called_once()
        sent_alert = mock_alerter.send.call_args[0][0]
        assert sent_alert.severity == AlertSeverity.WARNING
        assert sent_alert.source == "drift_monitor"
