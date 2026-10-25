"""Unit tests for BaselineRiskModel (LogisticRegression, RandomForest, GradientBoosting)."""

import numpy as np
import pytest
from backend.risk.baseline_model import BaselineRiskModel, map_probability_to_score
from backend.risk.schemas import RiskCategory, RiskExample, RiskThresholds


def test_probability_to_score_mapping():
    assert map_probability_to_score(0.0) == 1.0
    assert map_probability_to_score(0.5) == 5.5
    assert map_probability_to_score(1.0) == 10.0
    # Clamping
    assert map_probability_to_score(-0.2) == 1.0
    assert map_probability_to_score(1.5) == 10.0


def test_risk_thresholds_categorization():
    thresholds = RiskThresholds()
    assert thresholds.categorize(1.0) == RiskCategory.LOW
    assert thresholds.categorize(3.5) == RiskCategory.LOW
    assert thresholds.categorize(5.0) == RiskCategory.MEDIUM
    assert thresholds.categorize(7.5) == RiskCategory.HIGH
    assert thresholds.categorize(9.5) == RiskCategory.CRITICAL


def test_baseline_fit_and_predict():
    np.random.seed(42)
    X = np.random.randn(20, 15)
    y = np.array([1] * 10 + [0] * 10)

    for model_type in ("logistic_regression", "random_forest", "gradient_boosting"):
        model = BaselineRiskModel(model_type=model_type)
        model.fit(X, y)
        assert model.is_fitted

        probs = model.predict_proba(X)
        assert len(probs) == 20
        assert all(0.0 <= p <= 1.0 for p in probs)

        importances = model.get_feature_importances()
        assert len(importances) == 15


def test_model_save_and_load(tmp_path):
    np.random.seed(42)
    X = np.random.randn(20, 15)
    y = np.array([1] * 10 + [0] * 10)

    model = BaselineRiskModel(model_type="random_forest")
    model.fit(X, y)

    save_path = tmp_path / "model.joblib"
    model.save(save_path)
    assert save_path.exists()

    loaded = BaselineRiskModel.load(save_path)
    assert loaded.is_fitted
    assert loaded.name == model.name
    probs = loaded.predict_proba(X)
    assert len(probs) == 20


def test_predict_example():
    np.random.seed(42)
    X = np.random.randn(20, 15)
    y = np.array([1] * 10 + [0] * 10)
    model = BaselineRiskModel(model_type="logistic_regression")
    model.fit(X, y)

    ex = RiskExample(
        example_id="test_ex",
        repository="repo",
        commit_sha="c1",
        function_name="Flask.run",
        file_path="src/flask/app.py",
        language="python",
        code="def run(): return 1",
        timestamp="2026-01-01T00:00:00Z",
        timestamp_epoch=100.0,
        label=1,
        label_rationale="test",
    )
    pred = model.predict_example(ex)
    assert pred.function_name == "Flask.run"
    assert 0.0 <= pred.risk_probability <= 1.0
    assert 1.0 <= pred.risk_score <= 10.0
    assert pred.risk_category in (RiskCategory.LOW, RiskCategory.MEDIUM, RiskCategory.HIGH, RiskCategory.CRITICAL)
