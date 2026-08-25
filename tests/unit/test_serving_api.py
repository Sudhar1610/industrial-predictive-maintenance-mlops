"""Tests for the FastAPI serving layer's HTTP contract.

The `lifespan` startup hook (which loads the Production model from the
registry) is deliberately bypassed here by setting `pdm.serving.app.state`
directly -- these tests exercise routing, request validation, and
response shaping against a known model, not registry wiring (that's
covered by tests/unit/test_registry.py and the integration suite).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pdm.config.schemas import SklearnRfConfig
from pdm.models.sklearn_models import SklearnRfModel
from pdm.serving import app as app_module


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    test_client = TestClient(app_module.app)
    yield test_client
    app_module.state.model = None
    app_module.state.model_backend = None


def _history_payload(n_cycles: int = 5) -> dict:
    return {
        "unit_id": 1,
        "history": [
            {"cycle": c, "values": {"sensor_1": 100.0 + c, "op_setting_1": 0.01}}
            for c in range(1, n_cycles + 1)
        ],
    }


def _fit_and_install_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fit a real SklearnRfModel against whatever feature width the
    default test payload actually produces, then install it directly
    into app state (bypassing the registry-backed lifespan hook)."""
    from pdm.config.schemas import ModelConfig
    from pdm.serving.inference import build_model_input
    from pdm.serving.schemas import PredictRequest

    model_config = ModelConfig.model_validate(
        {
            "active_model": "sklearn_rf",
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
            "lstm": {"hidden_size": 4, "num_layers": 1, "epochs": 1},
            "training": {},
            "registry": {
                "tracking_uri": "sqlite:///x.db",
                "artifact_location": "x",
                "model_name": "x",
                "local_fallback_path": "x",
            },
        }
    )
    request = PredictRequest.model_validate(_history_payload())
    X, _ = build_model_input(request, model_config)

    rng = np.random.default_rng(0)
    n_features = X.shape[1]
    X_train = rng.normal(size=(50, n_features))
    y_class = rng.integers(0, 2, 50)
    y_reg = rng.normal(50, 10, 50)
    model = SklearnRfModel(SklearnRfConfig(n_estimators=5, max_depth=2))
    model.fit(X_train, y_class, y_reg)

    monkeypatch.setattr(app_module.settings, "get_model_config", lambda: model_config)
    app_module.state.model = model
    app_module.state.model_backend = "sklearn_rf"


class TestPredictEndpoint:
    def test_predict_returns_expected_shape(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        _fit_and_install_model(monkeypatch)
        response = client.post("/predict", json=_history_payload())
        assert response.status_code == 200
        body = response.json()
        assert body["unit_id"] == 1
        assert body["cycle"] == 5
        assert 0.0 <= body["failure_probability"] <= 1.0
        assert isinstance(body["will_fail"], bool)
        assert body["model_backend"] == "sklearn_rf"

    def test_predict_without_model_returns_503(self, client: TestClient):
        response = client.post("/predict", json=_history_payload())
        assert response.status_code == 503

    def test_predict_rejects_empty_history(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        _fit_and_install_model(monkeypatch)
        payload = _history_payload()
        payload["history"] = []
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_rejects_missing_unit_id(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        _fit_and_install_model(monkeypatch)
        payload = _history_payload()
        del payload["unit_id"]
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


class TestPredictBatchEndpoint:
    def test_batch_returns_one_prediction_per_unit(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        _fit_and_install_model(monkeypatch)
        payload = {"units": [_history_payload(), _history_payload()]}
        response = client.post("/predict-batch", json=payload)
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 2


class TestHealthEndpoint:
    def test_health_degraded_when_no_model(self, client: TestClient):
        with patch("pdm.serving.app.get_datasource") as mock_get_datasource:
            mock_get_datasource.return_value.health_check.return_value = True
            response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "degraded"
        assert response.json()["model_loaded"] is False

    def test_health_ok_when_model_and_datasource_ready(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        _fit_and_install_model(monkeypatch)
        with patch("pdm.serving.app.get_datasource") as mock_get_datasource:
            mock_get_datasource.return_value.health_check.return_value = True
            response = client.get("/health")
        assert response.json()["status"] == "ok"
        assert response.json()["model_backend"] == "sklearn_rf"
