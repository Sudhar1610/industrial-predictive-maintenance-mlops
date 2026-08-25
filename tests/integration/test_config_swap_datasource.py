"""Proves the central architectural claim of this project: promoting
between datasource backends is a config-only change.

This test writes real `datasource_config.yaml` files to a temp config
directory (exactly the file a human would edit by hand per
docs/STAGES.md), points `pdm.config.settings` at that directory, and
shows that `pdm.data.factory.get_datasource` returns a different
concrete `DataSource` implementation purely as a function of the YAML's
`active_source` field -- with the exact same calling code
(`get_datasource(settings.get_datasource_config())`) used unmodified in
all three cases. No test-only branching, no mocking of the factory
itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pdm.config import settings
from pdm.data.csv_source import CsvDataSource
from pdm.data.factory import get_datasource
from pdm.data.mongo_source import MongoDataSource
from pdm.data.sql_source import SqlDataSource

_DATASOURCE_YAML_TEMPLATE = """
active_source: {active_source}

csv:
  train_path: "{csv_dir}/train_FD001.txt"
  test_path: "{csv_dir}/test_FD001.txt"
  rul_path: "{csv_dir}/RUL_FD001.txt"
  column_names: [unit_id, cycle, op_setting_1, op_setting_2, op_setting_3, sensor_1]

sql:
  driver: "postgresql"
  host: "${{TEST_SQL_HOST}}"
  port: 5432
  database: "telemetry"
  username: "svc_pdm"
  password: "${{TEST_SQL_PASSWORD}}"
  query_training: "SELECT * FROM readings"
  query_latest: "SELECT * FROM readings WHERE unit_id = :unit_id"

mongodb:
  uri: "mongodb://plant-mongo:27017"
  database: "machine_telemetry"
  collection_training: "sensor_history"
  collection_latest: "sensor_stream"

validation:
  fail_on_violation: true
"""


def _write_datasource_config(config_dir: Path, active_source: str, csv_dir: Path) -> None:
    """Writes only `datasource_config.yaml` -- the other two YAMLs are
    loaded from the real repo config dir via a fallback, since this test
    is only exercising the datasource swap."""
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "datasource_config.yaml").write_text(
        _DATASOURCE_YAML_TEMPLATE.format(active_source=active_source, csv_dir=csv_dir)
    )


@pytest.fixture(autouse=True)
def _reset_config_cache():
    settings.clear_config_cache()
    yield
    settings.clear_config_cache()


class TestConfigOnlyDatasourceSwap:
    @pytest.mark.parametrize(
        ("active_source", "expected_class"),
        [
            ("csv", CsvDataSource),
            ("sql", SqlDataSource),
            ("mongodb", MongoDataSource),
        ],
    )
    def test_active_source_alone_determines_datasource_class(
        self,
        active_source: str,
        expected_class: type,
        tmp_path: Path,
        cmapss_csv_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        config_dir = tmp_path / "configs"
        _write_datasource_config(config_dir, active_source, cmapss_csv_dir)
        monkeypatch.setattr(settings, "CONFIG_DIR", config_dir)
        settings.clear_config_cache()

        # This is the SAME two-line call every real caller in the
        # codebase uses (pdm.pipelines.training_flow, pdm.serving.app's
        # /health check, etc.) -- nothing here is test-only wiring.
        config = settings.get_datasource_config()
        source = get_datasource(config)

        assert isinstance(source, expected_class)

    def test_swapping_active_source_mid_process_changes_the_class(
        self, tmp_path: Path, cmapss_csv_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Simulates exactly what a human does to promote Stage 1 -> 2:
        edit one line in datasource_config.yaml, nothing else."""
        config_dir = tmp_path / "configs"
        monkeypatch.setattr(settings, "CONFIG_DIR", config_dir)

        _write_datasource_config(config_dir, "csv", cmapss_csv_dir)
        settings.clear_config_cache()
        stage1_source = get_datasource(settings.get_datasource_config())
        assert isinstance(stage1_source, CsvDataSource)

        _write_datasource_config(config_dir, "sql", cmapss_csv_dir)
        settings.clear_config_cache()
        stage2_source = get_datasource(settings.get_datasource_config())
        assert isinstance(stage2_source, SqlDataSource)

    def test_csv_source_from_config_swap_can_actually_read_data(
        self, tmp_path: Path, cmapss_csv_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Not just isinstance-correct -- the config-driven CsvDataSource
        actually functions end-to-end against real files on disk."""
        config_dir = tmp_path / "configs"
        _write_datasource_config(config_dir, "csv", cmapss_csv_dir)
        monkeypatch.setattr(settings, "CONFIG_DIR", config_dir)
        settings.clear_config_cache()

        source = get_datasource(settings.get_datasource_config())
        assert source.health_check() is True
        df = source.fetch_training_data()
        assert not df.empty
        assert "unit_id" in df.columns
