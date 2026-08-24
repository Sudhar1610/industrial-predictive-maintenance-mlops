"""Degradation-trend features: rolling linear slope of each sensor.

A rising or falling slope over the recent window is a direct proxy for
"this sensor is drifting," which is the core signal a degrading turbofan
component produces before failure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_slope(values: np.ndarray) -> float:
    """OLS slope of `values` against an evenly-spaced index, using the
    closed-form covariance/variance formula (cheaper than np.polyfit when
    called once per row via a rolling window)."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    y_mean = values.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    numer = ((x - x_mean) * (values - y_mean)).sum()
    return float(numer / denom)


def add_degradation_slope_features(
    df: pd.DataFrame, sensor_columns: list[str], window: int
) -> pd.DataFrame:
    """Add `{sensor}_slope_{window}` columns: the rolling OLS slope of
    each sensor over the last `window` cycles within the same unit."""
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    grouped = df.groupby("unit_id", sort=False)

    new_columns: dict[str, pd.Series] = {}
    for col in sensor_columns:
        slope = grouped[col].rolling(window=window, min_periods=2).apply(_rolling_slope, raw=True)
        new_columns[f"{col}_slope_{window}"] = slope.reset_index(level=0, drop=True).fillna(0.0)

    return pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
