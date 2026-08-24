"""Tests for pdm.data: DataSource ABC implementations, validation, factory."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera.errors
import pytest

from pdm.config.schemas import CsvSourceConfig, DatasourceConfig
from pdm.data.csv_source import CsvDataSource
from pdm.data.factory import get_datasource
from pdm.data.validation import validate_sensor_readings

CMAPSS_COLUMNS = [
    "unit_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]


def _csv_config(cmapss_csv_dir: Path) -> CsvSourceConfig:
    return CsvSourceConfig(
        train_path=str(cmapss_csv_dir / "train_FD001.txt"),
        test_path=str(cmapss_csv_dir / "test_FD001.txt"),
        rul_path=str(cmapss_csv_dir / "RUL_FD001.txt"),
        column_names=CMAPSS_COLUMNS,
    )


class TestCsvDataSource:
    def test_fetch_training_data_shape(self, cmapss_csv_dir: Path, synthetic_cmapss_df: pd.DataFrame):
        source = CsvDataSource(_csv_config(cmapss_csv_dir))
        df = source.fetch_training_data()
        assert list(df.columns) == CMAPSS_COLUMNS
        assert len(df) == len(synthetic_cmapss_df)
        assert df["unit_id"].nunique() == 5

    def test_fetch_test_data_and_rul_labels(self, cmapss_csv_dir: Path):
        source = CsvDataSource(_csv_config(cmapss_csv_dir))
        test_df = source.fetch_test_data()
        rul_df = source.fetch_rul_labels()
        assert test_df["unit_id"].nunique() == 5
        assert len(rul_df) == 5
        assert "RUL" in rul_df.columns

    def test_fetch_latest_all_units(self, cmapss_csv_dir: Path):
        source = CsvDataSource(_csv_config(cmapss_csv_dir))
        latest = source.fetch_latest()
        assert len(latest) == 5  # one row per unit
        full = source.fetch_training_data()
        for uid in latest["unit_id"]:
            expected_max_cycle = full[full["unit_id"] == uid]["cycle"].max()
            actual = latest.loc[latest["unit_id"] == uid, "cycle"].iloc[0]
            assert actual == expected_max_cycle

    def test_fetch_latest_single_unit(self, cmapss_csv_dir: Path):
        source = CsvDataSource(_csv_config(cmapss_csv_dir))
        latest = source.fetch_latest(unit_id=1)
        assert len(latest) == 1
        assert latest["unit_id"].iloc[0] == 1

    def test_fetch_latest_unknown_unit_raises(self, cmapss_csv_dir: Path):
        source = CsvDataSource(_csv_config(cmapss_csv_dir))
        with pytest.raises(ValueError, match="unit_id=999"):
            source.fetch_latest(unit_id=999)

    def test_health_check_true_when_file_exists(self, cmapss_csv_dir: Path):
        source = CsvDataSource(_csv_config(cmapss_csv_dir))
        assert source.health_check() is True

    def test_health_check_false_when_file_missing(self, tmp_path: Path):
        config = CsvSourceConfig(
            train_path=str(tmp_path / "nope.txt"),
            test_path=str(tmp_path / "nope.txt"),
            rul_path=str(tmp_path / "nope.txt"),
            column_names=CMAPSS_COLUMNS,
        )
        source = CsvDataSource(config)
        assert source.health_check() is False

    def test_missing_file_raises_filenotfound(self, tmp_path: Path):
        config = CsvSourceConfig(
            train_path=str(tmp_path / "nope.txt"),
            test_path=str(tmp_path / "nope.txt"),
            rul_path=str(tmp_path / "nope.txt"),
            column_names=CMAPSS_COLUMNS,
        )
        source = CsvDataSource(config)
        with pytest.raises(FileNotFoundError):
            source.fetch_training_data()


class TestValidation:
    def test_valid_data_passes(self, synthetic_cmapss_df: pd.DataFrame):
        validated = validate_sensor_readings(synthetic_cmapss_df)
        assert len(validated) == len(synthetic_cmapss_df)

    def test_duplicate_unit_cycle_fails(self, synthetic_cmapss_df: pd.DataFrame):
        dup = pd.concat([synthetic_cmapss_df, synthetic_cmapss_df.iloc[[0]]], ignore_index=True)
        with pytest.raises(pandera.errors.SchemaError):
            validate_sensor_readings(dup)

    def test_non_numeric_sensor_column_fails(self, synthetic_cmapss_df: pd.DataFrame):
        bad = synthetic_cmapss_df.copy()
        bad["sensor_1"] = "not-a-number"
        with pytest.raises(pandera.errors.SchemaError):
            validate_sensor_readings(bad)

    def test_fail_on_violation_false_returns_data_unchanged(self, synthetic_cmapss_df: pd.DataFrame):
        dup = pd.concat([synthetic_cmapss_df, synthetic_cmapss_df.iloc[[0]]], ignore_index=True)
        result = validate_sensor_readings(dup, fail_on_violation=False)
        assert len(result) == len(dup)


class TestDatasourceFactory:
    def _full_config(self, cmapss_csv_dir: Path) -> DatasourceConfig:
        return DatasourceConfig.model_validate(
            {
                "active_source": "csv",
                "csv": _csv_config(cmapss_csv_dir).model_dump(),
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

    def test_factory_returns_csv_source_for_csv_config(self, cmapss_csv_dir: Path):
        config = self._full_config(cmapss_csv_dir)
        source = get_datasource(config)
        assert isinstance(source, CsvDataSource)

    def test_factory_returns_sql_source_for_sql_config(self, cmapss_csv_dir: Path):
        config = self._full_config(cmapss_csv_dir)
        config.active_source = "sql"
        from pdm.data.sql_source import SqlDataSource

        source = get_datasource(config)
        assert isinstance(source, SqlDataSource)

    def test_factory_returns_mongo_source_for_mongodb_config(self, cmapss_csv_dir: Path):
        config = self._full_config(cmapss_csv_dir)
        config.active_source = "mongodb"
        from pdm.data.mongo_source import MongoDataSource

        source = get_datasource(config)
        assert isinstance(source, MongoDataSource)
