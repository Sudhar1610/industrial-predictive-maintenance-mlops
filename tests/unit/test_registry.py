"""Tests for pdm.registry: local MLflow registry and the no-server
local-artifact fallback.

`mlflow_registry_config` points at a temp sqlite db + temp artifact dir
per test, so tests never touch a shared `mlflow/` directory and can run
in parallel / repeatedly with no leftover state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pdm.config.schemas import RegistryConfig, SklearnRfConfig
from pdm.models.sklearn_models import SklearnRfModel
from pdm.registry.local_artifact import (
    load_local_fallback,
    local_fallback_exists,
    save_local_fallback,
)
from pdm.registry.mlflow_registry import MlflowModelRegistry


@pytest.fixture
def trained_model() -> SklearnRfModel:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 4))
    y_class = (X[:, 0] > 0).astype(int)
    y_reg = X[:, 1] * 5 + 20
    model = SklearnRfModel(SklearnRfConfig(n_estimators=5, max_depth=2))
    model.fit(X, y_class, y_reg)
    return model


@pytest.fixture
def registry_config(tmp_path: Path) -> RegistryConfig:
    return RegistryConfig(
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_location=str(tmp_path / "artifacts"),
        model_name="test_pdm_model",
        local_fallback_path=str(tmp_path / "fallback"),
    )


class TestLocalArtifactFallback:
    def test_not_exists_before_save(self, registry_config: RegistryConfig):
        assert local_fallback_exists(registry_config) is False

    def test_save_then_load_roundtrip(
        self, trained_model: SklearnRfModel, registry_config: RegistryConfig
    ):
        save_local_fallback(trained_model, registry_config)
        assert local_fallback_exists(registry_config) is True

        loaded = load_local_fallback(registry_config)
        X = np.random.default_rng(2).normal(size=(5, 4))
        before = trained_model.predict(X)
        after = loaded.predict(X)
        np.testing.assert_allclose(before.failure_probability, after.failure_probability)

    def test_load_missing_raises(self, registry_config: RegistryConfig):
        with pytest.raises(FileNotFoundError):
            load_local_fallback(registry_config)


class TestMlflowModelRegistry:
    def test_creates_registered_model_on_init(self, registry_config: RegistryConfig):
        registry = MlflowModelRegistry(registry_config)
        assert registry._client.get_registered_model(registry_config.model_name) is not None

    def test_works_when_sqlite_parent_dir_does_not_exist_yet(self, tmp_path: Path):
        # Regression test: a fresh checkout has no mlflow/ directory at
        # all (it's gitignored). sqlite can't create its own parent dir,
        # so MlflowModelRegistry must create it first -- this reproduces
        # that exact "unable to open database file" failure mode.
        nested = tmp_path / "does" / "not" / "exist" / "mlflow.db"
        assert not nested.parent.exists()
        config = RegistryConfig(
            tracking_uri=f"sqlite:///{nested}",
            artifact_location=str(tmp_path / "artifacts"),
            model_name="fresh_checkout_model",
            local_fallback_path=str(tmp_path / "fallback"),
        )
        registry = MlflowModelRegistry(config)
        assert registry._client.get_registered_model(config.model_name) is not None

    def test_log_training_run_registers_version(
        self, trained_model: SklearnRfModel, registry_config: RegistryConfig
    ):
        registry = MlflowModelRegistry(registry_config)
        version = registry.log_training_run(
            trained_model, params={"n_estimators": 5}, metrics={"f1": 0.9, "rmse": 12.3}
        )
        assert version == "1"
        assert registry.get_version_metrics(version) == {"f1": 0.9, "rmse": 12.3}

    def test_no_production_version_initially(self, registry_config: RegistryConfig):
        registry = MlflowModelRegistry(registry_config)
        assert registry.get_production_version() is None

    def test_stage_transition_and_load_production(
        self, trained_model: SklearnRfModel, registry_config: RegistryConfig
    ):
        registry = MlflowModelRegistry(registry_config)
        version = registry.log_training_run(
            trained_model, params={"n_estimators": 5}, metrics={"f1": 0.9}
        )
        registry.transition_stage(version, "Staging")
        assert registry.get_production_version() is None

        registry.transition_stage(version, "Production")
        production = registry.get_production_version()
        assert production is not None
        assert str(production.version) == version

        loaded = registry.load_production_model()
        X = np.random.default_rng(3).normal(size=(5, 4))
        before = trained_model.predict(X)
        after = loaded.predict(X)
        np.testing.assert_allclose(before.failure_probability, after.failure_probability)

    def test_promoting_new_version_archives_old_production(
        self, trained_model: SklearnRfModel, registry_config: RegistryConfig
    ):
        registry = MlflowModelRegistry(registry_config)
        v1 = registry.log_training_run(trained_model, params={}, metrics={"f1": 0.8})
        registry.transition_stage(v1, "Production")

        v2 = registry.log_training_run(trained_model, params={}, metrics={"f1": 0.95})
        registry.transition_stage(v2, "Production")

        production = registry.get_production_version()
        assert str(production.version) == v2
        v1_info = registry._client.get_model_version(registry_config.model_name, v1)
        assert v1_info.current_stage == "Archived"
