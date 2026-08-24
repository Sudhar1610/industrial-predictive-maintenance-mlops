"""The `DataSource` abstraction.

Every place this application reads sensor telemetry -- training or
inference -- goes through this interface. Stage 1 implements it with a
`CsvDataSource`; Stage 2/3 swap in `SqlDataSource` or `MongoDataSource`
via `configs/datasource_config.yaml`. No calling code branches on which
concrete class is in use, so promoting stages never requires touching
`pdm.features`, `pdm.models`, or `pdm.pipelines`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    """Abstract source of sensor telemetry.

    Implementations must return data in the same tidy shape regardless of
    backend: one row per (unit_id, cycle), with operational settings and
    sensor readings as columns. `pdm.data.validation` enforces this shape
    identically for every implementation, so a schema violation in a real
    SQL/Mongo feed fails the same way a malformed CSV would.
    """

    @abstractmethod
    def fetch_training_data(self) -> pd.DataFrame:
        """Return the full historical dataset used to train models:
        multiple units run to failure (or truncated), each with a
        cycle-indexed sensor history."""
        raise NotImplementedError

    @abstractmethod
    def fetch_latest(self, unit_id: str | int | None = None) -> pd.DataFrame:
        """Return the most recent sensor reading(s) available for
        inference. `unit_id=None` means "all units currently reporting"
        (used by scheduled batch scoring jobs); a specific `unit_id`
        scopes to one asset (used by the single-prediction API path)."""
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the underlying source is reachable right now.
        Used by the FastAPI `/health` endpoint and by the training
        pipeline's ingest step to fail loud before doing any work."""
        raise NotImplementedError
