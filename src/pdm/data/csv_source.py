"""Stage 1 `DataSource`: reads the C-MAPSS sample dataset from local CSV.

File format note: the raw C-MAPSS `.txt` files are whitespace-delimited
with no header row and a trailing-space artifact that produces two extra
all-NaN columns if read naively -- both are handled below.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from pdm.config.schemas import CsvSourceConfig
from pdm.data.base import DataSource


class CsvDataSource(DataSource):
    """Reads training/test/RUL data from the local C-MAPSS CSV files
    configured in `configs/datasource_config.yaml` under `csv:`."""

    def __init__(self, config: CsvSourceConfig) -> None:
        self._config = config

    def _read_cmapss_file(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(
                f"C-MAPSS data file not found: {path}. "
                f"Run `python scripts/download_data.py` first."
            )
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=self._config.column_names,
            engine="python",
        )
        return df

    def fetch_training_data(self) -> pd.DataFrame:
        logger.info("Loading C-MAPSS training data from {}", self._config.train_path)
        df = self._read_cmapss_file(Path(self._config.train_path))
        df["unit_id"] = df["unit_id"].astype(int)
        df["cycle"] = df["cycle"].astype(int)
        return df

    def fetch_test_data(self) -> pd.DataFrame:
        """C-MAPSS-specific: the held-out test set, each unit truncated
        before failure. Not part of the `DataSource` interface (SQL/Mongo
        sources have no equivalent notion), but useful for offline model
        evaluation against the official FD001 RUL labels."""
        logger.info("Loading C-MAPSS test data from {}", self._config.test_path)
        df = self._read_cmapss_file(Path(self._config.test_path))
        df["unit_id"] = df["unit_id"].astype(int)
        df["cycle"] = df["cycle"].astype(int)
        return df

    def fetch_rul_labels(self) -> pd.DataFrame:
        """C-MAPSS-specific: ground-truth RUL at the last cycle of each
        test-set unit, one value per unit."""
        path = Path(self._config.rul_path)
        if not path.exists():
            raise FileNotFoundError(f"RUL label file not found: {path}.")
        rul = pd.read_csv(path, sep=r"\s+", header=None, names=["RUL"], engine="python")
        rul["unit_id"] = rul.index + 1
        return rul

    def fetch_latest(self, unit_id: str | int | None = None) -> pd.DataFrame:
        """For Stage 1, "latest" means the final recorded cycle per unit
        in the training set (there is no live feed to poll)."""
        df = self.fetch_training_data()
        if unit_id is not None:
            df = df[df["unit_id"] == int(unit_id)]
            if df.empty:
                raise ValueError(f"No data found for unit_id={unit_id}")
        latest = df.sort_values("cycle").groupby("unit_id", as_index=False).tail(1)
        return latest.reset_index(drop=True)

    def health_check(self) -> bool:
        exists = Path(self._config.train_path).exists()
        if not exists:
            logger.error("CSV health check failed: {} does not exist", self._config.train_path)
        return exists
