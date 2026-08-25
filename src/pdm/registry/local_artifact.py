"""Stage 3 fallback registry: load/save a model directly from a local
artifact directory, with no MLflow server, sqlite database, or network
access required at inference time.

This is the path `pdm.serving.model_loader` falls back to when the
MLflow-backed registry is unavailable or intentionally not run as a
service on the production IIOT server -- see `model_config.yaml`'s
`registry.local_fallback_path`. It reads/writes the exact same
`Model.save()`/`load()` format as `pdm.registry.mlflow_registry`, so a
model promoted to Production there can be copied here (or re-saved here
directly) with no format conversion.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from pdm.config.schemas import RegistryConfig
from pdm.models.base import Model
from pdm.models.factory import load_model


def save_local_fallback(model: Model, config: RegistryConfig) -> Path:
    """Write `model`'s artifacts to `config.local_fallback_path`."""
    path = Path(config.local_fallback_path)
    model.save(path)
    logger.info("Saved local fallback model artifact to {}", path)
    return path


def load_local_fallback(config: RegistryConfig) -> Model:
    """Load the model at `config.local_fallback_path`. Raises
    `FileNotFoundError` if nothing has been promoted there yet."""
    path = Path(config.local_fallback_path)
    if not local_fallback_exists(config):
        raise FileNotFoundError(
            f"No local fallback model artifact found at {path}. "
            f"Promote a model there first (see docs/STAGES.md)."
        )
    return load_model(path)


def local_fallback_exists(config: RegistryConfig) -> bool:
    return (Path(config.local_fallback_path) / "metadata.json").exists()
