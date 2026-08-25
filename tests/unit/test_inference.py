"""Tests for pdm.serving.inference: request -> model input shaping."""

from __future__ import annotations

import numpy as np

from pdm.config.schemas import ModelConfig
from pdm.serving.inference import build_model_input, request_to_dataframe
from pdm.serving.schemas import PredictRequest, SensorReading


def _model_config(active_model: str) -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "active_model": active_model,
            "feature_engineering": {
                "rolling_windows": [3],
                "lag_steps": [1],
                "degradation_slope_window": 3,
                "sensor_columns": ["sensor_1"],
            },
            "classification": {"failure_horizon_cycles": 30},
            "regression": {},
            "xgboost": {"n_estimators": 5, "max_depth": 2},
            "sklearn_rf": {"n_estimators": 5, "max_depth": 2},
            "lstm": {"hidden_size": 4, "num_layers": 1, "sequence_length": 3, "epochs": 1},
            "training": {},
            "registry": {
                "tracking_uri": "sqlite:///x.db",
                "artifact_location": "x",
                "model_name": "x",
                "local_fallback_path": "x",
            },
        }
    )


def _request(n_cycles: int = 5) -> PredictRequest:
    history = [
        SensorReading(cycle=c, values={"sensor_1": 100.0 + c, "op_setting_1": 0.01})
        for c in range(1, n_cycles + 1)
    ]
    return PredictRequest(unit_id=7, history=history)


class TestRequestToDataframe:
    def test_shape_and_sort_order(self):
        request = _request()
        df = request_to_dataframe(request)
        assert list(df["cycle"]) == [1, 2, 3, 4, 5]
        assert (df["unit_id"] == 7).all()


class TestBuildModelInput:
    def test_tabular_backend_returns_2d(self):
        X, latest_cycle = build_model_input(_request(), _model_config("xgboost"))
        assert X.ndim == 2
        assert X.shape[0] == 1
        assert latest_cycle == 5

    def test_lstm_backend_returns_3d(self):
        X, latest_cycle = build_model_input(_request(), _model_config("lstm"))
        assert X.ndim == 3
        assert X.shape[0] == 1
        assert X.shape[1] == 3  # sequence_length
        assert latest_cycle == 5

    def test_short_history_still_works(self):
        # Only 1 cycle -- rolling/lag windows warm up gracefully (see
        # pdm.features), so a brand-new unit can still be scored.
        X, latest_cycle = build_model_input(_request(n_cycles=1), _model_config("xgboost"))
        assert X.shape[0] == 1
        assert latest_cycle == 1
        assert not np.isnan(X).any()
