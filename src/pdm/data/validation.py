"""Shared data-validation schema (Pandera).

This schema runs identically against DataFrames returned by
`CsvDataSource`, `SqlDataSource`, and `MongoDataSource`. Because
`DataSource.fetch_training_data`/`fetch_latest` all return the same tidy
shape, one schema catches malformed data regardless of which backend
produced it -- a real SQL/Mongo feed is validated exactly as strictly as
the sample CSV, so nothing "only breaks in Stage 2."

Validation failure raises `pandera.errors.SchemaError`; pipelines let this
propagate (fail loud). Serving code catches it and returns HTTP 422
instead of crashing the process (fail safe).
"""

from __future__ import annotations

import pandas as pd
import pandera as pa
from pandera.typing import Series


class SensorReadingSchema(pa.DataFrameModel):
    """One row = one (unit_id, cycle) sensor reading."""

    unit_id: Series[int] = pa.Field(ge=0)
    cycle: Series[int] = pa.Field(ge=1)
    op_setting_1: Series[float] = pa.Field(nullable=False)
    op_setting_2: Series[float] = pa.Field(nullable=False)
    op_setting_3: Series[float] = pa.Field(nullable=False)

    class Config:
        # Extra sensor_N columns are allowed and validated for dtype only
        # (not enumerated here) via the coerce/check below, so this
        # schema stays valid for FD001..FD004-style variants that carry
        # a different sensor count.
        strict = False
        coerce = True


def validate_sensor_readings(df: pd.DataFrame, *, fail_on_violation: bool = True) -> pd.DataFrame:
    """Validate `df` against `SensorReadingSchema` plus generic sensor-
    column checks (numeric dtype, no fully-null sensor column, no
    duplicate (unit_id, cycle) pairs).

    Raises `pandera.errors.SchemaError` if `fail_on_violation` is True and
    validation fails. If False, logs and returns `df` unchanged (used only
    for exploratory/dev scripts, never in pipelines or serving).
    """
    from loguru import logger

    try:
        validated = SensorReadingSchema.validate(df, lazy=True)

        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
        if not sensor_cols:
            raise pa.errors.SchemaError(
                SensorReadingSchema, df, "No sensor_* columns found in data."
            )
        non_numeric = [c for c in sensor_cols if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            raise pa.errors.SchemaError(
                SensorReadingSchema, df, f"Non-numeric sensor columns: {non_numeric}"
            )
        fully_null = [c for c in sensor_cols if df[c].isna().all()]
        if fully_null:
            raise pa.errors.SchemaError(
                SensorReadingSchema, df, f"Fully-null sensor columns: {fully_null}"
            )
        dup_key = df.duplicated(subset=["unit_id", "cycle"]).sum()
        if dup_key > 0:
            raise pa.errors.SchemaError(
                SensorReadingSchema, df, f"{dup_key} duplicate (unit_id, cycle) rows found."
            )

        return validated
    except pa.errors.SchemaError:
        if fail_on_violation:
            raise
        logger.warning("Sensor data failed validation but fail_on_violation=False; continuing.")
        return df
