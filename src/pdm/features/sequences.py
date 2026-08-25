"""Sliding-window sequence construction, for the LSTM model backend.

`sklearn`/`XGBoost` consume one engineered feature row per prediction.
The LSTM instead consumes a `(sequence_length, n_features)` window per
prediction, so it can learn temporal degradation patterns directly rather
than relying on the rolling/lag/slope features to encode them by hand.
This module bridges the same engineered tabular DataFrame produced by
`pdm.features.build` into that windowed shape.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd


def build_sequences(
    df: pd.DataFrame, feature_columns: list[str], sequence_length: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_], npt.NDArray[np.float64]]:
    """For each unit, produce one `(sequence_length, n_features)` window
    ending at each cycle. Units with fewer than `sequence_length` cycles
    so far are left-padded by repeating their first row, so every unit
    contributes a sequence starting from its very first observed cycle
    (important for early-life predictions, where waiting for a full
    window would mean never scoring a new asset until day `sequence_length`).

    Returns `(X, y_classification, y_regression)`. If `df` has no
    `will_fail`/`RUL` columns (e.g. an inference-time frame), the label
    arrays are empty.
    """
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    has_labels = "will_fail" in df.columns and "RUL" in df.columns

    sequences: list[npt.NDArray[np.float64]] = []
    y_class: list[int] = []
    y_reg: list[float] = []

    for _, group in df.groupby("unit_id", sort=False):
        values = group[feature_columns].to_numpy(dtype=np.float64)
        n = len(values)
        for i in range(n):
            start = i - sequence_length + 1
            if start < 0:
                pad = np.repeat(values[[0]], -start, axis=0)
                window = np.concatenate([pad, values[: i + 1]], axis=0)
            else:
                window = values[start : i + 1]
            sequences.append(window)
            if has_labels:
                y_class.append(int(group["will_fail"].iloc[i]))
                y_reg.append(float(group["RUL"].iloc[i]))

    X = np.stack(sequences) if sequences else np.empty((0, sequence_length, len(feature_columns)))
    return (
        X,
        np.array(y_class, dtype=np.int_),
        np.array(y_reg, dtype=np.float64),
    )
