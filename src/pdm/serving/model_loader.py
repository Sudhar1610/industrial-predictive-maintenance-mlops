"""Loads the Production model for serving, preferring the MLflow
registry and transparently falling back to the local artifact directory.

Both paths return a `Model` built from the exact same on-disk format
(see `pdm.models.base.Model.save`/`load`), so which one actually served
a given process is invisible to `pdm.serving.app` -- this is what lets
Stage 3 run with no MLflow tracking store deployed at all.
"""

from __future__ import annotations

from loguru import logger

from pdm.config.schemas import ModelConfig
from pdm.models.base import Model
from pdm.registry.local_artifact import load_local_fallback, local_fallback_exists
from pdm.registry.mlflow_registry import MlflowModelRegistry


def load_production_model(config: ModelConfig) -> Model:
    """Load the current Production model. Raises `RuntimeError` if
    neither the MLflow registry nor the local fallback artifact has one
    -- callers (the FastAPI startup hook) should let this fail loud at
    startup rather than serve with no model.
    """
    try:
        registry = MlflowModelRegistry(config.registry)
        return registry.load_production_model()
    except Exception as exc:  # noqa: BLE001 - any registry failure falls back
        logger.warning(
            "MLflow registry unavailable or has no Production model ({}); "
            "falling back to local artifact.",
            exc,
        )

    if local_fallback_exists(config.registry):
        return load_local_fallback(config.registry)

    raise RuntimeError(
        "No Production model available from the MLflow registry or the local "
        "fallback artifact. Train and promote a model first (see docs/STAGES.md)."
    )
