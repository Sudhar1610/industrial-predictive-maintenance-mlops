"""PyTorch LSTM backend: shared recurrent encoder, two output heads.

Unlike the tabular `sklearn`/`XGBoost` backends, this model consumes 3D
sequences shaped `(n_samples, sequence_length, n_features)` -- see
`pdm.features.sequences.build_sequences`. CPU-only by design: the target
Stage 3 IIOT server has no GPU, so training and inference must both be
fast and correct on CPU alone, and this module never calls `.cuda()`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pdm.config.schemas import LstmConfig
from pdm.models.base import Model, PredictionResult

_WEIGHTS_FILENAME = "model_state.pt"
_METADATA_FILENAME = "metadata.json"


class _DualHeadLstm(nn.Module):
    """Shared LSTM encoder feeding two independent linear heads: a
    failure-classification logit and a RUL regression value."""

    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classification_head = nn.Linear(hidden_size, 1)
        self.regression_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size), final layer's last timestep
        class_logit = self.classification_head(last_hidden).squeeze(-1)
        rul = self.regression_head(last_hidden).squeeze(-1)
        return class_logit, rul


class LstmModel(Model):
    """Dual-head sequence model. `X` passed to `fit`/`predict` must be
    3D: `(n_samples, sequence_length, n_features)`.
    """

    def __init__(self, config: LstmConfig) -> None:
        self._config = config
        self._net: _DualHeadLstm | None = None
        self._n_features: int | None = None
        torch.manual_seed(config.random_state)

    def _build_net(self, n_features: int) -> _DualHeadLstm:
        return _DualHeadLstm(
            n_features=n_features,
            hidden_size=self._config.hidden_size,
            num_layers=self._config.num_layers,
            dropout=self._config.dropout,
        )

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y_classification: npt.NDArray[np.int_],
        y_regression: npt.NDArray[np.float64],
    ) -> None:
        if X.ndim != 3:
            raise ValueError(
                f"LstmModel expects 3D input (n_samples, sequence_length, n_features), got shape {X.shape}"
            )
        self._n_features = X.shape[2]
        self._net = self._build_net(self._n_features)

        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y_classification, dtype=torch.float32),
            torch.tensor(y_regression, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self._config.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self._config.learning_rate)
        classification_loss_fn = nn.BCEWithLogitsLoss()
        regression_loss_fn = nn.MSELoss()

        self._net.train()
        for epoch in range(self._config.epochs):
            epoch_loss = 0.0
            for xb, yb_class, yb_reg in loader:
                optimizer.zero_grad()
                class_logit, rul_pred = self._net(xb)
                loss = classification_loss_fn(class_logit, yb_class) + regression_loss_fn(
                    rul_pred, yb_reg
                )
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % max(1, self._config.epochs // 5) == 0:
                logger.debug(
                    "LSTM epoch {}/{}: loss={:.4f}", epoch + 1, self._config.epochs, epoch_loss
                )

    def predict(self, X: npt.NDArray[np.float64]) -> PredictionResult:
        if self._net is None:
            raise RuntimeError("LstmModel.predict called before fit().")
        self._net.eval()
        with torch.no_grad():
            class_logit, rul_pred = self._net(torch.tensor(X, dtype=torch.float32))
            proba = torch.sigmoid(class_logit).numpy()
        return PredictionResult(
            failure_probability=proba,
            will_fail=proba >= 0.5,
            remaining_useful_life=rul_pred.numpy(),
        )

    def save(self, path: Path) -> None:
        if self._net is None:
            raise RuntimeError("LstmModel.save called before fit().")
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), path / _WEIGHTS_FILENAME)
        (path / _METADATA_FILENAME).write_text(
            json.dumps({"backend": "lstm", "params": self.params, "n_features": self._n_features})
        )

    @classmethod
    def load(cls, path: Path) -> "LstmModel":
        metadata = json.loads((path / _METADATA_FILENAME).read_text())
        instance = cls(LstmConfig(**metadata["params"]))
        instance._n_features = metadata["n_features"]
        instance._net = instance._build_net(metadata["n_features"])
        state_dict = torch.load(path / _WEIGHTS_FILENAME, map_location="cpu", weights_only=True)
        instance._net.load_state_dict(state_dict)
        instance._net.eval()
        return instance

    @property
    def params(self) -> dict[str, Any]:
        return self._config.model_dump()
