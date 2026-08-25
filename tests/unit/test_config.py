"""Tests for pdm.config: YAML loading, env-var interpolation, validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdm.config import settings


@pytest.fixture(autouse=True)
def _reset_config_cache():
    settings.clear_config_cache()
    yield
    settings.clear_config_cache()


def test_get_datasource_config_loads_real_repo_config():
    config = settings.get_datasource_config()
    assert config.active_source in {"csv", "sql", "mongodb"}
    assert config.csv.train_path == "data/raw/train_FD001.txt"


def test_get_alerting_config_loads_real_repo_config():
    config = settings.get_alerting_config()
    assert config.active_channel in {"console", "teams_webhook"}


def test_get_model_config_loads_real_repo_config():
    config = settings.get_model_config()
    assert config.active_model in {"sklearn_rf", "xgboost", "lstm"}
    assert config.registry.model_name == "pdm_turbofan"


def test_env_var_interpolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "datasource_config.yaml").write_text(
        """
active_source: sql
csv:
  train_path: "x"
  test_path: "x"
  rul_path: "x"
  column_names: ["unit_id"]
sql:
  driver: "postgresql"
  host: "${TEST_SQL_HOST}"
  port: 5432
  database: "db"
  username: "user"
  password: "${TEST_SQL_PASSWORD}"
  query_training: "SELECT 1"
  query_latest: "SELECT 1"
mongodb:
  uri: "mongodb://localhost"
  database: "db"
  collection_training: "c"
  collection_latest: "c"
validation:
  fail_on_violation: true
"""
    )
    monkeypatch.setenv("TEST_SQL_HOST", "real-host.example.com")
    monkeypatch.setenv("TEST_SQL_PASSWORD", "s3cr3t")
    monkeypatch.setenv("PDM_CONFIG_DIR", str(config_dir))
    monkeypatch.setattr(settings, "CONFIG_DIR", config_dir)
    settings.clear_config_cache()

    config = settings.get_datasource_config()
    assert config.sql.host == "real-host.example.com"
    assert config.sql.password == "s3cr3t"


def test_missing_env_var_resolves_to_empty_string(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TOTALLY_UNSET_VAR", raising=False)
    result = settings._interpolate_env_vars("${TOTALLY_UNSET_VAR}")
    assert result == ""
