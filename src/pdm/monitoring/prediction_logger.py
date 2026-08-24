"""Logs every prediction to a local, append-only JSON-lines file.

This is the audit trail AND the "current data" source `drift_job` reads
from -- a local file rather than an external logging service, so it
works identically whether the process is a dev laptop, the office PWS,
or the fully offline IIOT server.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from pdm.serving.schemas import PredictionResponse, PredictRequest

DEFAULT_LOG_PATH = Path("logs/predictions.jsonl")


class PredictionLogger:
    def __init__(self, log_path: Path = DEFAULT_LOG_PATH) -> None:
        self._log_path = log_path
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, request: PredictRequest, response: PredictionResponse) -> None:
        """Append one record combining the request's latest sensor
        reading with the model's output for it."""
        record = {
            "logged_at": datetime.now(UTC).isoformat(),
            "unit_id": response.unit_id,
            "cycle": response.cycle,
            "failure_probability": response.failure_probability,
            "will_fail": response.will_fail,
            "remaining_useful_life": response.remaining_useful_life,
            "model_backend": response.model_backend,
            "latest_reading": request.history[-1].values,
        }
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def read_recent(self, n: int | None = None) -> pd.DataFrame:
        """Return logged predictions as a DataFrame, most recent `n`
        rows if given, else everything. Returns an empty DataFrame if
        nothing has been logged yet."""
        if not self._log_path.exists():
            return pd.DataFrame()
        lines = [
            line for line in self._log_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        if not lines:
            return pd.DataFrame()
        records = [json.loads(line) for line in lines]
        df = pd.DataFrame(records)
        return df.tail(n) if n is not None else df
