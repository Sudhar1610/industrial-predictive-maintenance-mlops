"""Label construction for the two prediction heads.

Both labels are derived purely from `unit_id`/`cycle` -- they need no
sensor values -- so this stays a separate, trivially-testable module
rather than being buried inside feature engineering.
"""

from __future__ import annotations

import pandas as pd


def add_rul_and_failure_labels(
    df: pd.DataFrame, *, failure_horizon_cycles: int, rul_cap: int | None = None
) -> pd.DataFrame:
    """Add `RUL` (remaining useful life, in cycles, at each row) and
    `will_fail` (1 if `RUL <= failure_horizon_cycles` else 0).

    Assumes `df` contains complete run-to-failure trajectories (i.e. the
    training set, where the last recorded cycle per unit *is* the failure
    point) -- this is NOT valid for a truncated/live inference frame,
    which has no known failure point and gets its RUL from the model's
    regression head instead.

    `rul_cap` applies the standard C-MAPSS piecewise-linear RUL
    convention: RUL is capped at a max value for early-life cycles, since
    degradation is not yet observable and an unbounded linear RUL target
    just adds label noise a model can't learn from.
    """
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    max_cycle_per_unit = df.groupby("unit_id")["cycle"].transform("max")
    rul = max_cycle_per_unit - df["cycle"]
    if rul_cap is not None:
        rul = rul.clip(upper=rul_cap)

    df = df.copy()
    df["RUL"] = rul
    df["will_fail"] = (df["RUL"] <= failure_horizon_cycles).astype(int)
    return df
