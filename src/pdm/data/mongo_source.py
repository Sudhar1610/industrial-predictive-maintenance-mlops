"""Stage 2/3 `DataSource`: reads real machine telemetry from MongoDB.

Same interface contract as `CsvDataSource`/`SqlDataSource` -- see
`sql_source.py`'s module docstring for why that matters.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger
from pymongo import MongoClient
from pymongo.database import Database

from pdm.config.schemas import MongoSourceConfig
from pdm.data.base import DataSource


class MongoDataSource(DataSource):
    """Reads training/inference data from MongoDB collections.

    Connection is lazy, same rationale as `SqlDataSource`.
    """

    def __init__(self, config: MongoSourceConfig) -> None:
        self._config = config
        self._client: MongoClient | None = None

    @property
    def _db(self) -> Database:
        if self._client is None:
            self._client = MongoClient(
                self._config.uri,
                serverSelectionTimeoutMS=self._config.connection_timeout_ms,
            )
        return self._client[self._config.database]

    def fetch_training_data(self) -> pd.DataFrame:
        logger.info("Querying MongoDB training collection: {}", self._config.collection_training)
        cursor = self._db[self._config.collection_training].find({}, {"_id": 0})
        return pd.DataFrame(list(cursor))

    def fetch_latest(self, unit_id: str | int | None = None) -> pd.DataFrame:
        logger.info("Querying MongoDB latest stream, unit_id={}", unit_id)
        query = {} if unit_id is None else {"unit_id": unit_id}
        cursor = self._db[self._config.collection_latest].find(query, {"_id": 0})
        return pd.DataFrame(list(cursor))

    def health_check(self) -> bool:
        try:
            self._db.command("ping")
            return True
        except Exception as exc:  # noqa: BLE001 - health check must never raise
            logger.error("MongoDB health check failed: {}", exc)
            return False
