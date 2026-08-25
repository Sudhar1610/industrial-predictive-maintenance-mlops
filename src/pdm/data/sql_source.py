"""Stage 2/3 `DataSource`: reads real machine telemetry from SQL.

Same interface as `CsvDataSource` -- `pdm.features`, `pdm.models`, and
`pdm.pipelines` call `fetch_training_data`/`fetch_latest` without knowing
or caring that rows now come from a SQL Server/Postgres query instead of
a CSV file. Swapping this in is a one-line change to
`datasource_config.yaml` (`active_source: sql`), never a code change.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from pdm.config.schemas import SqlSourceConfig
from pdm.data.base import DataSource


def _build_connection_url(config: SqlSourceConfig) -> str:
    driver_suffix = f"?driver={config.odbc_driver.replace(' ', '+')}" if config.odbc_driver else ""
    return (
        f"{config.driver}://{config.username}:{config.password}"
        f"@{config.host}:{config.port}/{config.database}{driver_suffix}"
    )


class SqlDataSource(DataSource):
    """Reads training/inference data from a SQL database via SQLAlchemy.

    Connection is lazy: no network call happens until the first query, so
    constructing this object (e.g. during config validation or dependency
    injection wiring) never requires a live database.
    """

    def __init__(self, config: SqlSourceConfig) -> None:
        self._config = config
        self._engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(
                _build_connection_url(self._config),
                connect_args={"timeout": self._config.connection_timeout_s}
                if "mssql" in self._config.driver
                else {},
                pool_pre_ping=True,
            )
        return self._engine

    def fetch_training_data(self) -> pd.DataFrame:
        logger.info("Querying SQL training data from {}", self._config.host)
        with self.engine.connect() as conn:
            return pd.read_sql(text(self._config.query_training), conn)

    def fetch_latest(self, unit_id: str | int | None = None) -> pd.DataFrame:
        logger.info("Querying SQL latest reading(s), unit_id={}", unit_id)
        with self.engine.connect() as conn:
            if unit_id is not None:
                return pd.read_sql(
                    text(self._config.query_latest), conn, params={"unit_id": unit_id}
                )
            return pd.read_sql(text(self._config.query_training), conn)

    def health_check(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:  # noqa: BLE001 - health check must never raise
            logger.error("SQL health check failed: {}", exc)
            return False
