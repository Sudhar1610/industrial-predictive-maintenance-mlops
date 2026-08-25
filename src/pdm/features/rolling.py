"""Rolling-window statistical features.

Every function here operates per-unit (grouped by `unit_id`, ordered by
`cycle`) so that a rolling window never leaks information across two
different engines' trajectories.
"""

from __future__ import annotations

import pandas as pd


def add_rolling_features(
    df: pd.DataFrame, sensor_columns: list[str], windows: list[int]
) -> pd.DataFrame:
    """Add rolling mean and std columns for each `(sensor, window)` pair.

    New columns are named `{sensor}_roll_mean_{window}` and
    `{sensor}_roll_std_{window}`. The first `window - 1` rows of each
    unit's trajectory get a std of 0.0 (not NaN) via `min_periods=1`, so
    downstream models never see NaNs from window warm-up.
    """
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    grouped = df.groupby("unit_id", sort=False)

    new_columns: dict[str, pd.Series] = {}
    for window in windows:
        for col in sensor_columns:
            rolling = grouped[col].rolling(window=window, min_periods=1)
            new_columns[f"{col}_roll_mean_{window}"] = rolling.mean().reset_index(
                level=0, drop=True
            )
            new_columns[f"{col}_roll_std_{window}"] = (
                rolling.std().reset_index(level=0, drop=True).fillna(0.0)
            )

    return pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
