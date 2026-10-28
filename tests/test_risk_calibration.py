"""Unit tests for Risk Threshold Calibration and Optimization (Milestone 9)."""

import numpy as np
import pytest
from backend.risk.baseline_model import BaselineRiskModel
from backend.risk.evaluator import RiskEvaluator, find_optimal_threshold
from backend.risk.schemas import RiskCategory, RiskExample


def test_find_optimal_threshold_f1():
    # Construct synthetic ground truth and probabilities
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.85, 0.75, 0.70, 0.65, 0.3, 0.2, 0.1])

    # Threshold around 0.66-0.70 separates positive 0.70 from negative 0.65
    best_t, best_f1 = find_optimal_threshold(y_true, y_prob, criterion="f1")
    assert 0.65 <= best_t <= 0.72
    assert best_f1 >= 0.90


def test_find_optimal_threshold_min_fpr():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_prob = np.array([0.95, 0.90, 0.85, 0.80, 0.70, 0.50, 0.30, 0.10])

    # Threshold around 0.71-0.80 separates positive 0.80 from negative 0.70
    best_t, best_score = find_optimal_threshold(y_true, y_prob, criterion="min_fpr", min_recall=1.0)
    assert 0.70 <= best_t <= 0.81
    assert best_score >= 0.95  # 1 - fpr == 1.0


def test_model_calibrate_and_predict(tmp_path):
    model = BaselineRiskModel(model_type="random_forest", random_state=42)
    # Train dummy model
    X_train = np.random.RandomState(42).randn(20, 15)
    y_train = np.array([1] * 10 + [0] * 10)
    model.fit(X_train, y_train)

    assert model.optimal_threshold == 0.5

    # Set custom optimal threshold
    model.optimal_threshold = 0.78
    model_path = tmp_path / "model.joblib"
    model.save(model_path)

    loaded = BaselineRiskModel.load(model_path)
    assert loaded.optimal_threshold == 0.78
