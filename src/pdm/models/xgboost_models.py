"""XGBoost backend: one classifier head, one regressor head."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import xgboost as xgb

from pdm.config.schemas import XgboostConfig
from pdm.models.base import Model, PredictionResult

_CLASSIFIER_FILENAME = "classifier.json"
_REGRESSOR_FILENAME = "regressor.json"
_METADATA_FILENAME = "metadata.json"


class XgboostModel(Model):
    """Dual-head model backed by `XGBClassifier` + `XGBRegressor`.

    Weights are saved via XGBoost's native JSON format rather than
    pickle, which keeps model artifacts portable across XGBoost versions
    and avoids pickle's arbitrary-code-execution surface for a file that
    Stage 3 will eventually copy onto a production server.
    """

    def __init__(self, config: XgboostConfig, decision_threshold: float = 0.5) -> None:
        self._config = config
        self._decision_threshold = decision_threshold
        self._classifier = xgb.XGBClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            random_state=config.random_state,
            eval_metric="logloss",
        )
        self._regressor = xgb.XGBRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            random_state=config.random_state,
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
            raise RuntimeError("XgboostModel.predict called before fit().")
        proba = self._classifier.predict_proba(X)[:, 1]
        rul = self._regressor.predict(X)
        return PredictionResult(
            failure_probability=proba,
            will_fail=proba >= self._decision_threshold,
            remaining_useful_life=rul,
        )

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._classifier.save_model(str(path / _CLASSIFIER_FILENAME))
        self._regressor.save_model(str(path / _REGRESSOR_FILENAME))
        (path / _METADATA_FILENAME).write_text(
            json.dumps(
                {
                    "backend": "xgboost",
                    "params": self.params,
                    "decision_threshold": self._decision_threshold,
                }
            )
        )

    @classmethod
    def load(cls, path: Path) -> XgboostModel:
        metadata = json.loads((path / _METADATA_FILENAME).read_text())
        instance = cls(
            XgboostConfig(**metadata["params"]),
            decision_threshold=metadata["decision_threshold"],
        )
        instance._classifier.load_model(str(path / _CLASSIFIER_FILENAME))
        instance._regressor.load_model(str(path / _REGRESSOR_FILENAME))
        instance._fitted = True
        return instance

    @property
    def params(self) -> dict[str, Any]:
        return self._config.model_dump()
