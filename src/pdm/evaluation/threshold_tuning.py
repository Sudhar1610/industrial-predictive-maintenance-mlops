"""Decision-threshold tuning for the failure-classification head.

`model_config.yaml`'s `classification.decision_threshold` is a sane
default (0.5); this module lets the training pipeline instead pick the
threshold that maximizes a chosen metric on a held-out validation split,
which matters here because false negatives (missing a real failure) and
false positives (an unnecessary maintenance call) have very different
real-world costs.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.metrics import f1_score, precision_score, recall_score

_METRIC_FUNCTIONS = {
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score,
}


def find_best_threshold(
    y_true: npt.NDArray[np.int_],
    y_proba: npt.NDArray[np.float64],
    metric: str = "f1",
    n_thresholds: int = 100,
) -> tuple[float, float]:
    """Sweep `n_thresholds` evenly-spaced cutoffs in (0, 1) and return
    `(best_threshold, best_metric_value)` for the requested `metric`.
    """
    if metric not in _METRIC_FUNCTIONS:
        raise ValueError(f"Unknown metric {metric!r}; choose from {list(_METRIC_FUNCTIONS)}")
    metric_fn = _METRIC_FUNCTIONS[metric]

    best_threshold, best_score = 0.5, -1.0
    for threshold in np.linspace(0.01, 0.99, n_thresholds):
        y_pred = (y_proba >= threshold).astype(int)
        score = metric_fn(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)

    return best_threshold, best_score
