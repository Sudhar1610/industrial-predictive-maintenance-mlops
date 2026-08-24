"""Random Forest baseline: one classifier head, one regressor head."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from pdm.config.schemas import SklearnRfConfig
from pdm.models.base import Model, PredictionResult

_WEIGHTS_FILENAME = "model.pkl"
_METADATA_FILENAME = "metadata.json"


class SklearnRfModel(Model):
    """Dual-head model backed by two independent `RandomForest*`
    estimators trained on the same feature matrix."""

    def __init__(self, config: SklearnRfConfig, decision_threshold: float = 0.5) -> None:
        self._config = config
        self._decision_threshold = decision_threshold
        self._classifier = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=config.random_state,
            n_jobs=-1,
        )
        self._regressor = RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=config.random_state,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y_classification: npt.NDArray[np.int_],
        y_regression: npt.NDArray[np.float64],
    ) -> None:
        self._classifier.fit(X, y_classification)
        self._regressor.fit(X, y_regression)
        self._fitted = True

    def predict(self, X: npt.NDArray[np.float64]) -> PredictionResult:
        if not self._fitted:
            raise RuntimeError("SklearnRfModel.predict called before fit().")
        proba = self._classifier.predict_proba(X)[:, 1]
        rul = self._regressor.predict(X)
        return PredictionResult(
            failure_probability=proba,
            will_fail=proba >= self._decision_threshold,
            remaining_useful_life=rul,
        )

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"classifier": self._classifier, "regressor": self._regressor},
            path / _WEIGHTS_FILENAME,
        )
        (path / _METADATA_FILENAME).write_text(
            json.dumps(
                {
                    "backend": "sklearn_rf",
                    "params": self.params,
                    "decision_threshold": self._decision_threshold,
                }
            )
        )

    @classmethod
    def load(cls, path: Path) -> "SklearnRfModel":
        metadata = json.loads((path / _METADATA_FILENAME).read_text())
        instance = cls(
            SklearnRfConfig(**metadata["params"]),
            decision_threshold=metadata["decision_threshold"],
        )
        weights = joblib.load(path / _WEIGHTS_FILENAME)
        instance._classifier = weights["classifier"]
        instance._regressor = weights["regressor"]
        instance._fitted = True
        return instance

    @property
    def params(self) -> dict[str, Any]:
        return self._config.model_dump()
