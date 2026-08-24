"""Builds the active `DataSource` from `datasource_config.yaml`.

This is the single switch point for Stage 1 -> Stage 2 -> Stage 3
promotion: every caller asks this factory for "the datasource" and gets
back whichever backend `active_source` names, without ever importing a
concrete `*DataSource` class directly.
"""

from __future__ import annotations

from pdm.config.schemas import DatasourceConfig
from pdm.data.base import DataSource
from pdm.data.csv_source import CsvDataSource
from pdm.data.mongo_source import MongoDataSource
from pdm.data.sql_source import SqlDataSource


def get_datasource(config: DatasourceConfig) -> DataSource:
    """Instantiate the `DataSource` implementation named by
    `config.active_source`. Raises `ValueError` for an unrecognized value
    (should be unreachable -- `DatasourceConfig.active_source` is a
    `Literal`, so pydantic already rejects anything else at config-load
    time)."""
    if config.active_source == "csv":
        return CsvDataSource(config.csv)
    if config.active_source == "sql":
        return SqlDataSource(config.sql)
    if config.active_source == "mongodb":
        return MongoDataSource(config.mongodb)
    raise ValueError(f"Unknown active_source: {config.active_source!r}")
