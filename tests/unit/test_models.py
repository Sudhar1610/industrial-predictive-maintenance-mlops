"""Tests for pdm.models: each backend against the same synthetic data,
plus round-trip save/load and factory dispatch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pdm.config.schemas import LstmConfig, ModelConfig, SklearnRfConfig, XgboostConfig
from pdm.models.base import Model, PredictionResult
from pdm.models.factory import get_model, load_model
from pdm.models.lstm_model import LstmModel
from pdm.models.sklearn_models import SklearnRfModel
from pdm.models.xgboost_models import XgboostModel


@pytest.fixture
def tabular_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n, n_features = 200, 5
    X = rng.normal(size=(n, n_features))
    y_class = (X[:, 0] > 0).astype(int)
    y_reg = X[:, 1] * 10 + 50
    return X, y_class, y_reg


@pytest.fixture
def sequence_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n, seq_len, n_features = 40, 5, 3
    X = rng.normal(size=(n, seq_len, n_features))
    y_class = rng.integers(0, 2, size=n)
    y_reg = rng.normal(loc=50, scale=5, size=n)
    return X, y_class, y_reg


class TestSklearnRfModel:
    def test_fit_predict_roundtrip(self, tabular_data):
        X, y_class, y_reg = tabular_data
        model = SklearnRfModel(SklearnRfConfig(n_estimators=10, max_depth=3))
        model.fit(X, y_class, y_reg)
        result = model.predict(X)
        assert isinstance(result, PredictionResult)
        assert result.failure_probability.shape == (len(X),)
        assert result.remaining_useful_life.shape == (len(X),)

    def test_predict_before_fit_raises(self, tabular_data):
        X, _, _ = tabular_data
        model = SklearnRfModel(SklearnRfConfig(n_estimators=10, max_depth=3))
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_save_load_roundtrip(self, tabular_data, tmp_path: Path):
        X, y_class, y_reg = tabular_data
        model = SklearnRfModel(SklearnRfConfig(n_estimators=10, max_depth=3))
        model.fit(X, y_class, y_reg)
        before = model.predict(X)

        model.save(tmp_path)
        loaded = SklearnRfModel.load(tmp_path)
        after = loaded.predict(X)

        np.testing.assert_allclose(before.failure_probability, after.failure_probability)
        np.testing.assert_allclose(before.remaining_useful_life, after.remaining_useful_life)


class TestXgboostModel:
    def test_fit_predict_roundtrip(self, tabular_data):
        X, y_class, y_reg = tabular_data
        model = XgboostModel(XgboostConfig(n_estimators=10, max_depth=3))
        model.fit(X, y_class, y_reg)
        result = model.predict(X)
        assert result.failure_probability.shape == (len(X),)

    def test_save_load_roundtrip(self, tabular_data, tmp_path: Path):
        X, y_class, y_reg = tabular_data
        model = XgboostModel(XgboostConfig(n_estimators=10, max_depth=3))
        model.fit(X, y_class, y_reg)
        before = model.predict(X)

        model.save(tmp_path)
        loaded = XgboostModel.load(tmp_path)
        after = loaded.predict(X)

        np.testing.assert_allclose(before.failure_probability, after.failure_probability, rtol=1e-5)


class TestLstmModel:
    def test_fit_predict_roundtrip(self, sequence_data):
        X, y_class, y_reg = sequence_data
        config = LstmConfig(hidden_size=8, num_layers=1, epochs=2, batch_size=8)
        model = LstmModel(config)
        model.fit(X, y_class, y_reg)
        result = model.predict(X)
        assert result.failure_probability.shape == (len(X),)
        assert result.remaining_useful_life.shape == (len(X),)

    def test_rejects_2d_input(self, tabular_data):
        X, y_class, y_reg = tabular_data
        model = LstmModel(LstmConfig(epochs=1))
        with pytest.raises(ValueError, match="3D"):
            model.fit(X, y_class, y_reg)

    def test_save_load_roundtrip(self, sequence_data, tmp_path: Path):
        X, y_class, y_reg = sequence_data
        config = LstmConfig(hidden_size=8, num_layers=1, epochs=2, batch_size=8)
        model = LstmModel(config)
        model.fit(X, y_class, y_reg)
        before = model.predict(X)

        model.save(tmp_path)
        loaded = LstmModel.load(tmp_path)
        after = loaded.predict(X)

        np.testing.assert_allclose(before.failure_probability, after.failure_probability, rtol=1e-5)


class TestModelFactory:
    def _model_config(self, active_model: str) -> ModelConfig:
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
                "xgboost": {"n_estimators": 10, "max_depth": 3},
                "sklearn_rf": {"n_estimators": 10, "max_depth": 3},
                "lstm": {"hidden_size": 8, "num_layers": 1, "epochs": 1},
                "training": {},
                "registry": {
                    "tracking_uri": "sqlite:///x.db",
                    "artifact_location": "x",
                    "model_name": "x",
                    "local_fallback_path": "x",
                },
            }
        )

    @pytest.mark.parametrize(
        ("active_model", "expected_cls"),
        [("sklearn_rf", SklearnRfModel), ("xgboost", XgboostModel), ("lstm", LstmModel)],
    )
    def test_get_model_dispatches_correctly(self, active_model, expected_cls):
        model = get_model(self._model_config(active_model))
        assert isinstance(model, expected_cls)
        assert isinstance(model, Model)

    def test_load_model_reads_backend_from_metadata(self, tabular_data, tmp_path: Path):
        X, y_class, y_reg = tabular_data
        model = SklearnRfModel(SklearnRfConfig(n_estimators=5, max_depth=2))
        model.fit(X, y_class, y_reg)
        model.save(tmp_path)

        loaded = load_model(tmp_path)
        assert isinstance(loaded, SklearnRfModel)

    def test_load_model_missing_metadata_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path)
