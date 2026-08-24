"""Lag features: sensor value N cycles ago, per unit."""

from __future__ import annotations

import pandas as pd


def add_lag_features(df: pd.DataFrame, sensor_columns: list[str], lag_steps: list[int]) -> pd.DataFrame:
    """Add `{sensor}_lag_{n}` columns: the sensor's value `n` cycles
    earlier in the same unit's trajectory. Rows before a unit's `n`-th
    cycle have no history yet, so the lag is back-filled with that row's
    own current value (equivalent to assuming "no change yet observed"
    rather than injecting NaN into early-life predictions).
    """
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    grouped = df.groupby("unit_id", sort=False)

    new_columns: dict[str, pd.Series] = {}
    for lag in lag_steps:
        for col in sensor_columns:
            shifted = grouped[col].shift(lag)
            new_columns[f"{col}_lag_{lag}"] = shifted.fillna(df[col])

    return pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
