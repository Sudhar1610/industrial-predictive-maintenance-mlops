"""Builds the active `Model` from `model_config.yaml`, and loads a saved
model back without needing to know which backend produced it.
"""

from __future__ import annotations

import json
from pathlib import Path

from pdm.config.schemas import ModelConfig
from pdm.models.base import Model
from pdm.models.lstm_model import LstmModel
from pdm.models.sklearn_models import SklearnRfModel
from pdm.models.xgboost_models import XgboostModel

_METADATA_FILENAME = "metadata.json"

_BACKEND_CLASSES: dict[str, type[Model]] = {
    "sklearn_rf": SklearnRfModel,
    "xgboost": XgboostModel,
    "lstm": LstmModel,
}


def get_model(config: ModelConfig) -> Model:
    """Instantiate the untrained `Model` implementation named by
    `config.active_model`."""
    if config.active_model == "sklearn_rf":
        return SklearnRfModel(config.sklearn_rf, config.classification.decision_threshold)
    if config.active_model == "xgboost":
        return XgboostModel(config.xgboost, config.classification.decision_threshold)
    if config.active_model == "lstm":
        return LstmModel(config.lstm)
    raise ValueError(f"Unknown active_model: {config.active_model!r}")


def load_model(path: Path) -> Model:
    """Load a previously-saved model from `path` without the caller
    needing to know which backend trained it -- the backend name is
    read back from the artifact's own `metadata.json`, written by every
    `Model.save()` implementation. This is what lets `pdm.registry` and
    `pdm.serving` load a Production model generically."""
    metadata_path = path / _METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"No {_METADATA_FILENAME} found at {path}")
    backend = json.loads(metadata_path.read_text())["backend"]
    model_cls = _BACKEND_CLASSES.get(backend)
    if model_cls is None:
        raise ValueError(f"Unknown model backend in metadata: {backend!r}")
    return model_cls.load(path)
