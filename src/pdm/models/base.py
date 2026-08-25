"""The `Model` abstraction.

Wraps the two prediction heads (failure classification, RUL regression)
behind one interface so `sklearn`/`XGBoost` baselines and the PyTorch LSTM
are interchangeable from the training pipeline's point of view, and so
the registry/serving layers can load and call any of them identically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class PredictionResult:
    """Output of a single `Model.predict` call, covering both heads."""

    failure_probability: npt.NDArray[np.float64]
    will_fail: npt.NDArray[np.bool_]
    remaining_useful_life: npt.NDArray[np.float64]


class Model(ABC):
    """Abstract dual-head predictive-maintenance model.

    `fit`/`predict` operate on already-engineered feature matrices (see
    `pdm.features`), not raw sensor data -- feature engineering is a
    pipeline stage shared by every model backend, not something each
    model implementation duplicates.
    """

    @abstractmethod
    def fit(
        self,
        X: npt.NDArray[np.float64],
        y_classification: npt.NDArray[np.int_],
        y_regression: npt.NDArray[np.float64],
    ) -> None:
        """Train both heads on the same feature matrix `X`."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: npt.NDArray[np.float64]) -> PredictionResult:
        """Run inference for both heads on feature matrix `X`."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize model weights + any backend-specific state to
        `path` (a directory). Must be loadable by `load()` with no
        external services (e.g. no MLflow server, no network) running --
        this is what makes Stage 3 offline inference possible."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> Model:
        """Deserialize a model previously written by `save()`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Hyperparameters used to build this model instance, for MLflow
        param logging."""
        raise NotImplementedError
