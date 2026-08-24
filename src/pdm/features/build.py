"""Orchestrates the full feature-engineering pipeline stage.

This is the single function the training pipeline and the serving layer
both call, so feature parity between train-time and inference-time is
structural rather than something that has to be kept in sync by hand.
"""

from __future__ import annotations

import pandas as pd

from pdm.config.schemas import FeatureEngineeringConfig
from pdm.features.degradation import add_degradation_slope_features
from pdm.features.lags import add_lag_features
from pdm.features.rolling import add_rolling_features


def build_features(df: pd.DataFrame, config: FeatureEngineeringConfig) -> pd.DataFrame:
    """Apply rolling, lag, and degradation-slope feature engineering, in
    that order, to raw sensor readings. Does not add labels -- see
    `pdm.features.labels` -- since inference-time frames have no labels
    to add.
    """
    df = add_rolling_features(df, config.sensor_columns, config.rolling_windows)
    df = add_lag_features(df, config.sensor_columns, config.lag_steps)
    df = add_degradation_slope_features(
        df, config.sensor_columns, config.degradation_slope_window
    )
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the columns of an engineered DataFrame that are model
    inputs -- i.e. everything except identifiers and labels. Used to
    build the `X` matrix consistently at both train and inference time.
    """
    excluded = {"unit_id", "cycle", "RUL", "will_fail"}
    return [c for c in df.columns if c not in excluded]
