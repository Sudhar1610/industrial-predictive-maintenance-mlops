"""Tests for scripts/validate_and_promote_model.py: the CI model-validation gate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from validate_and_promote_model import _regressed, validate_and_promote  # noqa: E402

from pdm.config import settings
from pdm.config.schemas import ModelConfig, RegistryConfig, SklearnRfConfig
from pdm.models.sklearn_models import SklearnRfModel
from pdm.registry.mlflow_registry import MlflowModelRegistry


@pytest.fixture
def registry_config(tmp_path: Path) -> RegistryConfig:
    return RegistryConfig(
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_location=str(tmp_path / "artifacts"),
        model_name="gate_test_model",
        local_fallback_path=str(tmp_path / "fallback"),
    )


@pytest.fixture
def model_config(registry_config: RegistryConfig) -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "active_model": "sklearn_rf",
            "feature_engineering": {
                "rolling_windows": [3],
                "lag_steps": [1],
                "degradation_slope_window": 3,
                "sensor_columns": ["sensor_1"],
            },
            "classification": {"failure_horizon_cycles": 30, "target_metric": "f1"},
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


class TestRegressed:
    def test_higher_is_better_metric_flags_drop(self):
        assert _regressed("f1", candidate=0.80, baseline=0.90, tolerance=0.02) is True

    def test_higher_is_better_within_tolerance_ok(self):
        assert _regressed("f1", candidate=0.895, baseline=0.90, tolerance=0.02) is False

    def test_lower_is_better_metric_flags_increase(self):
        assert _regressed("rmse", candidate=20.0, baseline=15.0, tolerance=0.02) is True

    def test_lower_is_better_within_tolerance_ok(self):
        assert _regressed("rmse", candidate=15.2, baseline=15.0, tolerance=0.02) is False

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown guardrail metric"):
            _regressed("mystery_metric", 1.0, 1.0, 0.02)


class TestValidateAndPromote:
    def test_no_staging_version_is_noop(
        self, model_config: ModelConfig, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings, "get_model_config", lambda: model_config)
        assert validate_and_promote() == 0

    def test_promotes_when_no_existing_production(
        self,
        trained_model: SklearnRfModel,
        model_config: ModelConfig,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(settings, "get_model_config", lambda: model_config)
        registry = MlflowModelRegistry(model_config.registry)
        version = registry.log_training_run(trained_model, params={}, metrics={"f1": 0.85})
        registry.transition_stage(version, "Staging")

        assert validate_and_promote() == 0
        production = registry.get_production_version()
        assert production is not None
        assert str(production.version) == version

    def test_blocks_when_candidate_regresses(
        self,
        trained_model: SklearnRfModel,
        model_config: ModelConfig,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(settings, "get_model_config", lambda: model_config)
        registry = MlflowModelRegistry(model_config.registry)

        v1 = registry.log_training_run(trained_model, params={}, metrics={"f1": 0.90})
        registry.transition_stage(v1, "Production")

        v2 = registry.log_training_run(trained_model, params={}, metrics={"f1": 0.70})
        registry.transition_stage(v2, "Staging")

        assert validate_and_promote() == 1
        production = registry.get_production_version()
        assert str(production.version) == v1  # unchanged

    def test_promotes_when_candidate_improves(
        self,
        trained_model: SklearnRfModel,
        model_config: ModelConfig,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(settings, "get_model_config", lambda: model_config)
        registry = MlflowModelRegistry(model_config.registry)

        v1 = registry.log_training_run(trained_model, params={}, metrics={"f1": 0.80})
        registry.transition_stage(v1, "Production")

        v2 = registry.log_training_run(trained_model, params={}, metrics={"f1": 0.92})
        registry.transition_stage(v2, "Staging")

        assert validate_and_promote() == 0
        production = registry.get_production_version()
        assert str(production.version) == v2
