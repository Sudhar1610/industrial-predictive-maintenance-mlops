"""Bridges a `PredictRequest` into the exact feature matrix shape each
model backend expects, reusing the same `pdm.features` pipeline used at
training time -- this is what keeps train/serve feature parity structural
rather than a thing that can silently drift.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from pdm.config.schemas import ModelConfig
from pdm.features.build import build_features, get_feature_columns
from pdm.features.sequences import build_sequences
from pdm.serving.schemas import PredictRequest


def request_to_dataframe(request: PredictRequest) -> pd.DataFrame:
    """Flatten a `PredictRequest`'s history into a tidy DataFrame with
    the same shape `pdm.data.DataSource` implementations produce."""
    rows = [
        {"unit_id": request.unit_id, "cycle": reading.cycle, **reading.values}
        for reading in request.history
    ]
    return pd.DataFrame(rows).sort_values("cycle").reset_index(drop=True)


def build_model_input(
    request: PredictRequest, config: ModelConfig
) -> tuple[npt.NDArray[np.float64], int]:
    """Return `(X, latest_cycle)` where `X` is shaped correctly for the
    currently active model backend and represents a prediction for the
    most recent cycle in `request.history`.

    Tabular backends (`sklearn_rf`, `xgboost`) get a single engineered
    feature row, shape `(1, n_features)`. The `lstm` backend gets a
    single sequence window, shape `(1, sequence_length, n_features)`.
    """
    df = request_to_dataframe(request)
    engineered = build_features(df, config.feature_engineering)
    latest_cycle = int(engineered["cycle"].iloc[-1])

    if config.active_model == "lstm":
        feature_columns = get_feature_columns(engineered)
        X, _, _ = build_sequences(engineered, feature_columns, config.lstm.sequence_length)
        return X[-1:], latest_cycle

    feature_columns = get_feature_columns(engineered)
    X = engineered[feature_columns].to_numpy(dtype=np.float64)
    return X[-1:], latest_cycle
