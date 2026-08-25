"""Tests for pdm.evaluation: metrics, threshold tuning, drift detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pdm.evaluation.drift import compute_data_drift
from pdm.evaluation.metrics import (
    classification_metrics,
    nasa_scoring_function,
    regression_metrics,
)
from pdm.evaluation.threshold_tuning import find_best_threshold


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([False, True, True, False])
        y_proba = np.array([0.1, 0.9, 0.8, 0.2])
        metrics = classification_metrics(y_true, y_pred, y_proba)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["roc_auc"] == 1.0

    def test_single_class_roc_auc_is_nan(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([False, False, False])
        y_proba = np.array([0.1, 0.2, 0.1])
        metrics = classification_metrics(y_true, y_pred, y_proba)
        assert np.isnan(metrics["roc_auc"])


class TestRegressionMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        metrics = regression_metrics(y_true, y_pred)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["nasa_score"] == pytest.approx(0.0)

    def test_rmse_matches_known_value(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        metrics = regression_metrics(y_true, y_pred)
        assert metrics["rmse"] == pytest.approx(np.sqrt((9 + 16) / 2))


class TestNasaScoringFunction:
    def test_zero_for_perfect_prediction(self):
        assert nasa_scoring_function(np.array([50.0]), np.array([50.0])) == pytest.approx(0.0)

    def test_late_prediction_penalized_more_than_early(self):
        # Same absolute error (10 cycles), opposite direction.
        late_penalty = nasa_scoring_function(np.array([50.0]), np.array([60.0]))  # d=+10
        early_penalty = nasa_scoring_function(np.array([50.0]), np.array([40.0]))  # d=-10
        assert late_penalty > early_penalty

    def test_symmetric_functions_match_formula(self):
        # d = +10 -> exp(1) - 1
        assert nasa_scoring_function(np.array([50.0]), np.array([60.0])) == pytest.approx(
            np.exp(1.0) - 1
        )
        # d = -10 -> exp(10/13) - 1
        assert nasa_scoring_function(np.array([50.0]), np.array([40.0])) == pytest.approx(
            np.exp(10 / 13) - 1
        )


class TestThresholdTuning:
    def test_finds_reasonable_threshold_for_separable_data(self):
        rng = np.random.default_rng(0)
        y_true = np.array([0] * 100 + [1] * 100)
        y_proba = np.concatenate([rng.uniform(0, 0.3, 100), rng.uniform(0.7, 1.0, 100)])
        threshold, score = find_best_threshold(y_true, y_proba, metric="f1")
        assert 0.3 < threshold < 0.7
        assert score > 0.95

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            find_best_threshold(np.array([0, 1]), np.array([0.1, 0.9]), metric="bogus")


class TestDataDrift:
    def test_no_drift_when_distributions_match(self):
        rng = np.random.default_rng(0)
        ref = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(5, 1, 200)})
        cur = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(5, 1, 200)})
        result = compute_data_drift(ref, cur, columns=["a", "b"])
        assert result["dataset_drift"] is False

    def test_drift_detected_on_shifted_distribution(self):
        rng = np.random.default_rng(0)
        ref = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(5, 1, 200)})
        cur = pd.DataFrame({"a": rng.normal(10, 1, 200), "b": rng.normal(5, 1, 200)})
        result = compute_data_drift(ref, cur, columns=["a", "b"])
        assert result["dataset_drift"] is True
        assert result["number_of_drifted_columns"] >= 1
