"""End-to-end test of the Prefect training pipeline: ingest -> validate
-> feature -> train -> evaluate -> register, run against synthetic
C-MAPSS-shaped data and a temp MLflow registry.

This is an integration test (not a unit test) because it exercises the
real Prefect flow engine and a real (temp, sqlite-backed) MLflow
registry rather than mocking either -- proving the pipeline stages
actually compose, not just that each stage works in isolation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pdm.config.schemas import DatasourceConfig, ModelConfig
from pdm.pipelines.training_flow import training_flow
from pdm.registry.mlflow_registry import MlflowModelRegistry


def _datasource_config(cmapss_csv_dir: Path) -> DatasourceConfig:
    column_names = [
        "unit_id",
        "cycle",
        "op_setting_1",
        "op_setting_2",
        "op_setting_3",
    ] + [f"sensor_{i}" for i in range(1, 22)]
    return DatasourceConfig.model_validate(
        {
            "active_source": "csv",
            "csv": {
                "train_path": str(cmapss_csv_dir / "train_FD001.txt"),
                "test_path": str(cmapss_csv_dir / "test_FD001.txt"),
                "rul_path": str(cmapss_csv_dir / "RUL_FD001.txt"),
                "column_names": column_names,
            },
            "sql": {
                "driver": "postgresql",
                "host": "h",
                "port": 5432,
                "database": "d",
                "username": "u",
                "password": "p",
                "query_training": "SELECT 1",
                "query_latest": "SELECT 1",
            },
            "mongodb": {
                "uri": "mongodb://localhost",
                "database": "d",
                "collection_training": "c",
                "collection_latest": "c",
            },
            "validation": {"fail_on_violation": True},
        }
    )


def _model_config(tmp_path: Path, active_model: str = "xgboost") -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "active_model": active_model,
            "feature_engineering": {
                "rolling_windows": [3],
                "lag_steps": [1],
                "degradation_slope_window": 3,
                "sensor_columns": [f"sensor_{i}" for i in (2, 3, 4, 7, 11)],
            },
            "classification": {"failure_horizon_cycles": 20},
            "regression": {},
            "xgboost": {"n_estimators": 10, "max_depth": 3},
            "sklearn_rf": {"n_estimators": 10, "max_depth": 3},
            "lstm": {"hidden_size": 4, "num_layers": 1, "epochs": 1, "sequence_length": 5},
            "training": {"test_split": 0.25, "seed": 42},
            "registry": {
                "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
                "artifact_location": str(tmp_path / "artifacts"),
                "model_name": "pipeline_test_model",
                "local_fallback_path": str(tmp_path / "fallback"),
            },
        }
    )


@pytest.fixture(autouse=True)
def _patch_config(monkeypatch: pytest.MonkeyPatch, cmapss_csv_dir: Path, tmp_path: Path):
    from pdm.config import settings

    datasource_config = _datasource_config(cmapss_csv_dir)
    model_config = _model_config(tmp_path)
    monkeypatch.setattr(settings, "get_datasource_config", lambda: datasource_config)
    monkeypatch.setattr(settings, "get_model_config", lambda: model_config)
    return model_config


class TestTrainingFlowEndToEnd:
    def test_flow_registers_a_staging_version(self, _patch_config: ModelConfig):
        version = training_flow()
        assert version == "1"

        registry = MlflowModelRegistry(_patch_config.registry)
        model_version = registry._client.get_model_version(
            _patch_config.registry.model_name, version
        )
        assert model_version.current_stage == "Staging"

    def test_flow_logs_both_classification_and_regression_metrics(self, _patch_config: ModelConfig):
        version = training_flow()
        registry = MlflowModelRegistry(_patch_config.registry)
        metrics = registry.get_version_metrics(version)

        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert "rmse" in metrics
        assert "nasa_score" in metrics

    def test_running_flow_twice_creates_two_versions(self, _patch_config: ModelConfig):
        v1 = training_flow()
        v2 = training_flow()
        assert v1 != v2
