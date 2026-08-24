"""Metrics for both prediction heads, plus the NASA C-MAPSS RUL score.

The NASA scoring function is included alongside standard regression
metrics (RMSE/MAE) because it is the benchmark competitions and papers
report for this dataset, and because it is asymmetric in a way plain RMSE
is not: predicting a shorter RUL than reality (an early warning) is
penalized far less than predicting a longer RUL than reality (a missed
failure) -- which mirrors the real cost asymmetry of predictive
maintenance.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: npt.NDArray[np.int_],
    y_pred: npt.NDArray[np.bool_],
    y_proba: npt.NDArray[np.float64],
) -> dict[str, float]:
    """Accuracy, precision, recall, F1, and ROC-AUC for the
    failure-classification head."""
    y_pred_int = y_pred.astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred_int),
        "precision": precision_score(y_true, y_pred_int, zero_division=0),
        "recall": recall_score(y_true, y_pred_int, zero_division=0),
        "f1": f1_score(y_true, y_pred_int, zero_division=0),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    else:
        metrics["roc_auc"] = float("nan")
    return {k: float(v) for k, v in metrics.items()}


def regression_metrics(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> dict[str, float]:
    """RMSE, MAE, R^2, and the NASA scoring function for the RUL
    regression head."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "nasa_score": nasa_scoring_function(y_true, y_pred),
    }


def nasa_scoring_function(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> float:
    """The asymmetric scoring function from the original C-MAPSS
    benchmark papers (Saxena et al., 2008).

    For each sample, `d = predicted_RUL - true_RUL`:
      - late prediction (d > 0, i.e. predicted RUL is optimistic /
        failure comes sooner than predicted): score += exp(d/10) - 1
      - early prediction (d <= 0): score += exp(-d/13) - 1

    Lower is better; the exponential means a badly-late prediction is
    penalized far more steeply than an equally-wrong early one.
    """
    d = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    scores = np.where(d > 0, np.exp(d / 10) - 1, np.exp(-d / 13) - 1)
    return float(np.sum(scores))
