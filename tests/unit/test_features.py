"""Tests for pdm.features: rolling, lag, degradation, labels, build."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pdm.config.schemas import FeatureEngineeringConfig
from pdm.features.build import build_features, get_feature_columns
from pdm.features.degradation import add_degradation_slope_features
from pdm.features.labels import add_rul_and_failure_labels
from pdm.features.lags import add_lag_features
from pdm.features.rolling import add_rolling_features

SENSOR_COLS = ["sensor_1", "sensor_2"]


@pytest.fixture
def two_unit_df() -> pd.DataFrame:
    unit1 = pd.DataFrame(
        {
            "unit_id": 1,
            "cycle": range(1, 11),
            "sensor_1": np.arange(10, dtype=float),
            "sensor_2": np.linspace(100, 110, 10),
        }
    )
    unit2 = pd.DataFrame(
        {
            "unit_id": 2,
            "cycle": range(1, 6),
            "sensor_1": np.arange(5, dtype=float) * 2,
            "sensor_2": np.linspace(200, 210, 5),
        }
    )
    return pd.concat([unit1, unit2], ignore_index=True)


class TestRolling:
    def test_adds_expected_columns(self, two_unit_df: pd.DataFrame):
        result = add_rolling_features(two_unit_df, SENSOR_COLS, windows=[3])
        assert "sensor_1_roll_mean_3" in result.columns
        assert "sensor_1_roll_std_3" in result.columns

    def test_no_nans_produced(self, two_unit_df: pd.DataFrame):
        result = add_rolling_features(two_unit_df, SENSOR_COLS, windows=[3, 5])
        new_cols = [c for c in result.columns if "_roll_" in c]
        assert not result[new_cols].isna().any().any()

    def test_does_not_leak_across_units(self, two_unit_df: pd.DataFrame):
        result = add_rolling_features(two_unit_df, SENSOR_COLS, windows=[3])
        # unit 2's first row rolling mean must equal its own first value,
        # not blended with unit 1's tail.
        first_row_unit2 = result[result["unit_id"] == 2].iloc[0]
        assert first_row_unit2["sensor_1_roll_mean_3"] == first_row_unit2["sensor_1"]


class TestLags:
    def test_lag_matches_shifted_value(self, two_unit_df: pd.DataFrame):
        result = add_lag_features(two_unit_df, SENSOR_COLS, lag_steps=[1])
        unit1 = result[result["unit_id"] == 1].reset_index(drop=True)
        assert unit1["sensor_1_lag_1"].iloc[2] == unit1["sensor_1"].iloc[1]

    def test_first_row_backfilled_not_nan(self, two_unit_df: pd.DataFrame):
        result = add_lag_features(two_unit_df, SENSOR_COLS, lag_steps=[1, 2])
        new_cols = [c for c in result.columns if "_lag_" in c]
        assert not result[new_cols].isna().any().any()
        unit1 = result[result["unit_id"] == 1].reset_index(drop=True)
        assert unit1["sensor_1_lag_1"].iloc[0] == unit1["sensor_1"].iloc[0]


class TestDegradationSlope:
    def test_positive_trend_gives_positive_slope(self, two_unit_df: pd.DataFrame):
        result = add_degradation_slope_features(two_unit_df, ["sensor_1"], window=5)
        unit1 = result[result["unit_id"] == 1].reset_index(drop=True)
        assert unit1["sensor_1_slope_5"].iloc[-1] > 0

    def test_no_nans_produced(self, two_unit_df: pd.DataFrame):
        result = add_degradation_slope_features(two_unit_df, SENSOR_COLS, window=4)
        new_cols = [c for c in result.columns if "_slope_" in c]
        assert not result[new_cols].isna().any().any()


class TestLabels:
    def test_rul_zero_at_last_cycle(self, two_unit_df: pd.DataFrame):
        result = add_rul_and_failure_labels(two_unit_df, failure_horizon_cycles=3)
        unit1 = result[result["unit_id"] == 1]
        assert unit1.loc[unit1["cycle"] == 10, "RUL"].iloc[0] == 0

    def test_will_fail_flag_near_end_of_life(self, two_unit_df: pd.DataFrame):
        result = add_rul_and_failure_labels(two_unit_df, failure_horizon_cycles=3)
        unit1 = result[result["unit_id"] == 1].sort_values("cycle")
        assert unit1["will_fail"].iloc[-1] == 1  # RUL=0
        assert unit1["will_fail"].iloc[0] == 0  # RUL=9

    def test_rul_cap_applied(self, two_unit_df: pd.DataFrame):
        result = add_rul_and_failure_labels(two_unit_df, failure_horizon_cycles=3, rul_cap=5)
        assert result["RUL"].max() <= 5


class TestBuildFeatures:
    def test_build_features_end_to_end(self, two_unit_df: pd.DataFrame):
        config = FeatureEngineeringConfig(
            rolling_windows=[3],
            lag_steps=[1],
            degradation_slope_window=3,
            sensor_columns=SENSOR_COLS,
        )
        engineered = build_features(two_unit_df, config)
        labeled = add_rul_and_failure_labels(engineered, failure_horizon_cycles=3)
        feature_cols = get_feature_columns(labeled)

        assert "unit_id" not in feature_cols
        assert "RUL" not in feature_cols
        assert "will_fail" not in feature_cols
        assert not labeled[feature_cols].isna().any().any()
