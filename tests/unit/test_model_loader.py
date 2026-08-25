"""Tests for pdm.serving.model_loader: registry-first, local-fallback-second."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pdm.config.schemas import ModelConfig, RegistryConfig, SklearnRfConfig
from pdm.models.sklearn_models import SklearnRfModel
from pdm.registry.local_artifact import save_local_fallback
from pdm.registry.mlflow_registry import MlflowModelRegistry
from pdm.serving.model_loader import load_production_model


def _model_config(registry_config: RegistryConfig) -> ModelConfig:
    return ModelConfig.model_validate(
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
            "registry": registry_config.model_dump(),
        }
    )


@pytest.fixture
def trained_model() -> SklearnRfModel:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 4))
    model = SklearnRfModel(SklearnRfConfig(n_estimators=5, max_depth=2))
    model.fit(X, (X[:, 0] > 0).astype(int), X[:, 1] * 5)
    return model


class TestLoadProductionModel:
    def test_raises_when_nothing_available(self, tmp_path: Path):
        registry_config = RegistryConfig(
            tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
            artifact_location=str(tmp_path / "artifacts"),
            model_name="test_model",
            local_fallback_path=str(tmp_path / "fallback"),
        )
        with pytest.raises(RuntimeError, match="No Production model available"):
            load_production_model(_model_config(registry_config))

    def test_falls_back_to_local_artifact_when_no_mlflow_production(
        self, trained_model: SklearnRfModel, tmp_path: Path
    ):
        registry_config = RegistryConfig(
            tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
            artifact_location=str(tmp_path / "artifacts"),
            model_name="test_model",
            local_fallback_path=str(tmp_path / "fallback"),
        )
        save_local_fallback(trained_model, registry_config)

        loaded = load_production_model(_model_config(registry_config))
        X = np.random.default_rng(1).normal(size=(5, 4))
        np.testing.assert_allclose(
            trained_model.predict(X).failure_probability, loaded.predict(X).failure_probability
        )

    def test_prefers_mlflow_registry_over_local_fallback(
        self, trained_model: SklearnRfModel, tmp_path: Path
    ):
        registry_config = RegistryConfig(
            tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
            artifact_location=str(tmp_path / "artifacts"),
            model_name="test_model",
            local_fallback_path=str(tmp_path / "fallback"),
        )
        registry = MlflowModelRegistry(registry_config)
        version = registry.log_training_run(trained_model, params={}, metrics={"f1": 0.9})
        registry.transition_stage(version, "Production")

        # A different, untrained-looking artifact sits in the fallback
        # path too -- if the loader picked this up instead, predictions
        # would differ from the MLflow-registered model.
        rng = np.random.default_rng(99)
        decoy = SklearnRfModel(SklearnRfConfig(n_estimators=5, max_depth=2))
        decoy.fit(rng.normal(size=(30, 4)), rng.integers(0, 2, 30), rng.normal(size=30))
        save_local_fallback(decoy, registry_config)

        loaded = load_production_model(_model_config(registry_config))
        X = np.random.default_rng(1).normal(size=(5, 4))
        np.testing.assert_allclose(
            trained_model.predict(X).failure_probability, loaded.predict(X).failure_probability
        )
