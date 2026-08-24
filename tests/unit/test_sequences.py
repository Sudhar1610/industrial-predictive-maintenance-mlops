"""Tests for pdm.features.sequences: sliding-window construction for the LSTM backend."""

from __future__ import annotations

import pandas as pd

from pdm.features.sequences import build_sequences


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 1, 2],
            "sensor_1": [10.0, 20.0, 30.0, 100.0, 200.0],
            "RUL": [2, 1, 0, 1, 0],
            "will_fail": [0, 0, 1, 0, 1],
        }
    )


class TestBuildSequences:
    def test_output_shape(self):
        X, y_class, y_reg = build_sequences(_df(), ["sensor_1"], sequence_length=3)
        assert X.shape == (5, 3, 1)
        assert y_class.shape == (5,)
        assert y_reg.shape == (5,)

    def test_short_trajectory_left_padded(self):
        X, _, _ = build_sequences(_df(), ["sensor_1"], sequence_length=3)
        # unit 1's first row (cycle=1): window should be [10, 10, 10] (padded).
        first_window = X[0]
        assert (first_window == 10.0).all()

    def test_full_window_uses_real_history(self):
        X, _, _ = build_sequences(_df(), ["sensor_1"], sequence_length=3)
        # unit 1's third row (cycle=3): window is the real [10, 20, 30].
        third_window = X[2].flatten()
        assert list(third_window) == [10.0, 20.0, 30.0]

    def test_no_labels_when_absent(self):
        df = _df().drop(columns=["RUL", "will_fail"])
        X, y_class, y_reg = build_sequences(df, ["sensor_1"], sequence_length=3)
        assert X.shape == (5, 3, 1)
        assert y_class.shape == (0,)
        assert y_reg.shape == (0,)

    def test_does_not_leak_across_units(self):
        X, _, _ = build_sequences(_df(), ["sensor_1"], sequence_length=3)
        # unit 2's first row (index 3, cycle=1): window padded with its
        # OWN first value (100), never unit 1's tail (30).
        unit2_first_window = X[3].flatten()
        assert list(unit2_first_window) == [100.0, 100.0, 100.0]
