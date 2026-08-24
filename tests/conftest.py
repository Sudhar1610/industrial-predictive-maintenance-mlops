"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CMAPSS_COLUMNS = [
    "unit_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]


def _make_unit_trajectory(unit_id: int, n_cycles: int, rng: np.random.Generator) -> pd.DataFrame:
    """Build one synthetic engine's run-to-failure trajectory: sensor
    values drift monotonically to simulate degradation, with noise."""
    cycles = np.arange(1, n_cycles + 1)
    degradation = cycles / n_cycles
    data = {
        "unit_id": unit_id,
        "cycle": cycles,
        "op_setting_1": rng.normal(0, 0.01, n_cycles),
        "op_setting_2": rng.normal(0, 0.01, n_cycles),
        "op_setting_3": np.full(n_cycles, 100.0),
    }
    for i in range(1, 22):
        base = 500 + i * 10
        drift = base * 0.1 * degradation
        noise = rng.normal(0, 0.5, n_cycles)
        data[f"sensor_{i}"] = base + drift + noise
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_cmapss_df() -> pd.DataFrame:
    """A small, deterministic, C-MAPSS-shaped DataFrame (5 units, random
    lifetimes) for fast unit tests that don't need the real dataset."""
    rng = np.random.default_rng(42)
    units = [_make_unit_trajectory(uid, rng.integers(50, 150), rng) for uid in range(1, 6)]
    return pd.concat(units, ignore_index=True)


@pytest.fixture
def cmapss_csv_dir(tmp_path: Path, synthetic_cmapss_df: pd.DataFrame) -> Path:
    """Write `synthetic_cmapss_df` out as train/test/RUL files in the raw
    C-MAPSS whitespace-delimited, no-header format, in a temp dir."""
    train_path = tmp_path / "train_FD001.txt"
    synthetic_cmapss_df.to_csv(train_path, sep=" ", header=False, index=False)

    # Build a matching, truncated "test" set + RUL labels.
    test_frames = []
    ruls = []
    for uid, group in synthetic_cmapss_df.groupby("unit_id"):
        truncated = group.iloc[: max(1, len(group) - 10)]
        test_frames.append(truncated)
        ruls.append(len(group) - len(truncated))
    test_df = pd.concat(test_frames, ignore_index=True)
    test_path = tmp_path / "test_FD001.txt"
    test_df.to_csv(test_path, sep=" ", header=False, index=False)

    rul_path = tmp_path / "RUL_FD001.txt"
    pd.Series(ruls).to_csv(rul_path, header=False, index=False)

    return tmp_path
