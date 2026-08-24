"""Unit tests for SqlDataSource / MongoDataSource with mocked backends.

These never touch a real database -- that's deliberate: Stage 1 CI has no
SQL/Mongo server available, so these tests verify the *contract*
(interface conformance, query construction, health-check failure
handling) rather than real connectivity. Real connectivity is verified
manually against the office PWS during Stage 2 promotion.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pdm.config.schemas import MongoSourceConfig, SqlSourceConfig
from pdm.data.base import DataSource
from pdm.data.mongo_source import MongoDataSource
from pdm.data.sql_source import SqlDataSource, _build_connection_url


@pytest.fixture
def sql_config() -> SqlSourceConfig:
    return SqlSourceConfig(
        driver="postgresql",
        host="db.example.com",
        port=5432,
        database="telemetry",
        username="svc_pdm",
        password="secret",
        query_training="SELECT * FROM readings",
        query_latest="SELECT * FROM readings WHERE unit_id = :unit_id",
    )


@pytest.fixture
def mongo_config() -> MongoSourceConfig:
    return MongoSourceConfig(
        uri="mongodb://db.example.com:27017",
        database="machine_telemetry",
        collection_training="sensor_history",
        collection_latest="sensor_stream",
    )


class TestSqlDataSource:
    def test_is_a_datasource(self, sql_config: SqlSourceConfig):
        assert isinstance(SqlDataSource(sql_config), DataSource)

    def test_connection_url_omits_odbc_driver_when_unset(self, sql_config: SqlSourceConfig):
        url = _build_connection_url(sql_config)
        assert url == "postgresql://svc_pdm:secret@db.example.com:5432/telemetry"

    def test_connection_url_includes_odbc_driver_when_set(self, sql_config: SqlSourceConfig):
        sql_config.odbc_driver = "ODBC Driver 17 for SQL Server"
        url = _build_connection_url(sql_config)
        assert "driver=ODBC+Driver+17+for+SQL+Server" in url

    @patch("pdm.data.sql_source.pd.read_sql")
    @patch("pdm.data.sql_source.create_engine")
    def test_fetch_training_data_runs_configured_query(
        self, mock_create_engine, mock_read_sql, sql_config: SqlSourceConfig
    ):
        mock_read_sql.return_value = pd.DataFrame({"unit_id": [1], "cycle": [1]})
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        source = SqlDataSource(sql_config)
        df = source.fetch_training_data()

        assert not df.empty
        mock_read_sql.assert_called_once()

    @patch("pdm.data.sql_source.create_engine")
    def test_health_check_returns_false_on_exception(
        self, mock_create_engine, sql_config: SqlSourceConfig
    ):
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = ConnectionError("down")
        mock_create_engine.return_value = mock_engine

        source = SqlDataSource(sql_config)
        assert source.health_check() is False


class TestMongoDataSource:
    def test_is_a_datasource(self, mongo_config: MongoSourceConfig):
        assert isinstance(MongoDataSource(mongo_config), DataSource)

    @patch("pdm.data.mongo_source.MongoClient")
    def test_fetch_training_data_queries_configured_collection(
        self, mock_mongo_client, mongo_config: MongoSourceConfig
    ):
        mock_collection = MagicMock()
        mock_collection.find.return_value = [{"unit_id": 1, "cycle": 1}]
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        mock_client_instance = MagicMock()
        mock_client_instance.__getitem__.return_value = mock_db
        mock_mongo_client.return_value = mock_client_instance

        source = MongoDataSource(mongo_config)
        df = source.fetch_training_data()

        assert len(df) == 1
        mock_db.__getitem__.assert_called_with(mongo_config.collection_training)

    @patch("pdm.data.mongo_source.MongoClient")
    def test_health_check_returns_false_on_exception(
        self, mock_mongo_client, mongo_config: MongoSourceConfig
    ):
        mock_db = MagicMock()
        mock_db.command.side_effect = ConnectionError("down")
        mock_client_instance = MagicMock()
        mock_client_instance.__getitem__.return_value = mock_db
        mock_mongo_client.return_value = mock_client_instance

        source = MongoDataSource(mongo_config)
        assert source.health_check() is False
