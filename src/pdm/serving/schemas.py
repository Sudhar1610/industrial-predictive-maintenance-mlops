"""Pydantic request/response models for the FastAPI serving layer.

`SensorReading.values` is a free-form `{column_name: value}` mapping
rather than 21 hardcoded `sensor_N` fields, deliberately: the sensor set
is config-driven (`datasource_config.yaml`'s `csv.column_names`, or a
SQL/Mongo schema in Stage 2/3) and can differ across C-MAPSS subsets or
real machines. Hardcoding sensor fields here would mean a config-only
Stage 1->2 promotion could still force an API schema change, which is
exactly the kind of coupling this project's architecture is meant to
avoid.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SensorReading(BaseModel):
    """One cycle's worth of operational settings + sensor values for a
    single unit."""

    cycle: int = Field(..., ge=1)
    values: dict[str, float] = Field(
        ..., description="Column name -> value, e.g. op_setting_1, sensor_2, ..."
    )


class PredictRequest(BaseModel):
    """A recent, cycle-ordered (oldest first) history for one unit,
    long enough to cover the largest configured rolling/lag/sequence
    window -- `docs/runbook.md` documents the minimum history length for
    the currently active model config."""

    model_config = ConfigDict(protected_namespaces=())

    unit_id: int = Field(..., ge=0)
    history: list[SensorReading] = Field(..., min_length=1)


class BatchPredictRequest(BaseModel):
    units: list[PredictRequest] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    unit_id: int
    cycle: int
    failure_probability: float
    will_fail: bool
    remaining_useful_life: float
    model_backend: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str  # "ok" | "degraded"
    model_loaded: bool
    model_backend: str | None = None
    datasource_reachable: bool
