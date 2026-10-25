"""Integration test for the complete CodeGhost risk pipeline:
Function -> Vector Retrieval -> Feature Extraction -> Model Prediction -> Risk Score
"""

import numpy as np
import pytest
from backend.models.schemas import CodeUnit, CodeUnitType
from backend.retrieval.embedder import MockEmbedder
from backend.retrieval.retriever import FunctionRetriever
from backend.retrieval.vector_store import QdrantVectorStore
from backend.risk.baseline_model import BaselineRiskModel
from backend.risk.evaluator import RiskEvaluator
from backend.risk.features import CodeRiskFeatureExtractor
from backend.risk.schemas import RiskCategory, RiskExample, RiskThresholds


@pytest.fixture
def mock_retriever():
    """Sets up an in-memory vector store with deterministic MockEmbedder."""
    store = QdrantVectorStore(storage_path=":memory:")
    embedder = MockEmbedder(dimension=128)
    collection = "integration_test_collection"
    store.create_collection(collection, vector_size=128, recreate=True)

    # Insert a sample historical bug-fix vector
    vector = embedder.embed_code("def run(): host.partition(':')")
    payload = {
        "repository": "https://github.com/pallets/flask",
        "commit_sha": "hist_commit_01",
        "parent_sha": "hist_parent_01",
        "file_path": "src/flask/app.py",
        "function_name": "Flask.run",
        "commit_message": "Fix .partition(:) IPv6 splitting",
        "buggy_code": "def run(): host.partition(':')",
        "fixed_code": "def run(): urlsplit(host)",
        "language": "python",
        "diff": "--- a\n+++ b",
        "detector_score": 0.8,
        "detector_classification": "BUG_FIX_CANDIDATE",
        "timestamp": "2026-01-01T00:00:00Z",
    }
    store.insert_batch(collection, [vector], [payload], ["11111111-1111-1111-1111-111111111111"])

    return FunctionRetriever(vector_store=store, embedder=embedder, collection_name=collection)


def test_end_to_end_risk_pipeline(mock_retriever):
    # 1. Target function to evaluate
    target_unit = CodeUnit(
        repository="https://github.com/pallets/flask",
        commit_hash="current_commit",
        file_path="src/flask/app.py",
        unit_type=CodeUnitType.METHOD,
        name="Flask.run",
        start_line=1,
        end_line=15,
        source_code="""
def run(self, host=None, port=None, debug=None):
    if host:
        host, _, port = host.partition(":")
    return True
""",
        language="python",
    )

    # 2. Vector Retrieval (Milestone 3)
    matches = mock_retriever.retrieve_for_code_unit(target_unit, top_k=3)
    assert len(matches) >= 1
    assert matches[0].function_name == "Flask.run"
    assert matches[0].similarity_score > 0.0

    # 3. Feature Extraction (Milestone 4 Part C)
    feature_extractor = CodeRiskFeatureExtractor()
    features = feature_extractor.extract_features(
        code=target_unit.source_code,
        language="python",
        retrieved_matches=matches,
        function_name=target_unit.name,
        file_path=target_unit.file_path,
    )
    assert features["top1_similarity"] > 0.0
    assert features["cyclomatic_complexity"] >= 2.0

    # 4. Train a Baseline Model on synthetic labeled data
    np.random.seed(42)
    X_train = np.random.randn(30, 15)
    y_train = np.array([1] * 15 + [0] * 15)
    baseline = BaselineRiskModel(model_type="random_forest", random_state=42)
    baseline.fit(X_train, y_train)

    # 5. Predict on target function
    pred = baseline.predict_function(
        code=target_unit.source_code,
        function_name=target_unit.name,
        file_path=target_unit.file_path,
        retrieved_matches=matches,
    )

    # 6. Verify predictions and outputs
    assert pred.function_name == "Flask.run"
    assert pred.file_path == "src/flask/app.py"
    assert 0.0 <= pred.risk_probability <= 1.0
    assert 1.0 <= pred.risk_score <= 10.0
    assert pred.risk_category in (RiskCategory.LOW, RiskCategory.MEDIUM, RiskCategory.HIGH, RiskCategory.CRITICAL)
    assert len(pred.top_retrieved_matches) >= 1
    assert "top_feature_importances" in pred.explanation_context


def test_evaluator_metrics_computation():
    evaluator = RiskEvaluator()
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3])

    metrics = evaluator.evaluate_predictions(
        y_true=y_true,
        y_prob=y_prob,
        model_name="TestModel",
        split_name="test",
        decision_threshold=0.5,
    )

    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.roc_auc == 1.0
    assert metrics.false_positive_rate == 0.0
    assert metrics.confusion_matrix == [[3, 0], [0, 3]]
    assert metrics.score_distribution["CRITICAL"] >= 1
    assert metrics.score_distribution["LOW"] >= 2
