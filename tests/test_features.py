"""Unit tests for CodeRiskFeatureExtractor and AST complexity computation."""

import pytest
from backend.retrieval.schemas import RetrievedMatch
from backend.risk.features import CodeRiskFeatureExtractor, FEATURE_NAMES
from backend.risk.schemas import RiskExample


def test_code_ast_features_extraction():
    extractor = CodeRiskFeatureExtractor()
    code = """
def process_data(items, strict=False):
    results = []
    if strict:
        if not items:
            raise ValueError("Empty items")
    for item in items:
        try:
            if item > 0:
                results.append(item * 2)
        except Exception:
            pass
    return results
"""
    feats = extractor.extract_features(code, language="python")
    assert len(feats) == len(FEATURE_NAMES)
    assert feats["lines_of_code"] >= 10
    assert feats["num_parameters"] >= 1
    assert feats["num_branches"] >= 2
    assert feats["num_loops"] >= 1
    assert feats["has_exception_handling"] == 1.0
    assert feats["cyclomatic_complexity"] >= 4.0
    assert feats["max_nesting_depth"] >= 2.0


def test_retrieval_features_extraction():
    extractor = CodeRiskFeatureExtractor()
    matches = [
        RetrievedMatch(
            similarity_score=0.92,
            repository="repo",
            commit_sha="c1",
            parent_sha="p1",
            file_path="app.py",
            function_name="fn1",
            commit_message="Fix bug",
            buggy_code="pass",
            fixed_code="return",
            diff="diff",
            detector_score=0.8,
            detector_classification="BUG_FIX_CANDIDATE",
            timestamp="2026-01-01T00:00:00+00:00",
        ),
        RetrievedMatch(
            similarity_score=0.88,
            repository="repo",
            commit_sha="c2",
            parent_sha="p2",
            file_path="app.py",
            function_name="fn2",
            commit_message="Fix regression",
            buggy_code="pass",
            fixed_code="return",
            diff="diff",
            detector_score=0.7,
            detector_classification="BUG_FIX_CANDIDATE",
            timestamp="2026-01-02T00:00:00+00:00",
        ),
    ]

    feats = extractor.extract_features("def test(): pass", retrieved_matches=matches)
    assert feats["top1_similarity"] == 0.92
    assert feats["top3_max_similarity"] == 0.92
    assert round(feats["topk_avg_similarity"], 2) == 0.90
    assert feats["num_strong_matches"] == 2.0
    assert feats["similarity_std"] > 0.0


def test_feature_matrix_extraction():
    extractor = CodeRiskFeatureExtractor()
    examples = [
        RiskExample(
            example_id="1",
            repository="r",
            commit_sha="c1",
            function_name="f1",
            file_path="f1.py",
            language="python",
            code="def f1(): return 1",
            timestamp="2026-01-01T00:00:00Z",
            timestamp_epoch=1.0,
            label=1,
            label_rationale="test",
        ),
        RiskExample(
            example_id="2",
            repository="r",
            commit_sha="c2",
            function_name="f2",
            file_path="f2.py",
            language="python",
            code="def f2(): return 2",
            timestamp="2026-01-02T00:00:00Z",
            timestamp_epoch=2.0,
            label=0,
            label_rationale="test",
        ),
    ]

    X, y, names = extractor.extract_dataset_matrix(examples)
    assert X.shape == (2, len(FEATURE_NAMES))
    assert y.tolist() == [1, 0]
    assert names == FEATURE_NAMES
