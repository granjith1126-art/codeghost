"""Unit tests for CodeGhost Reasoning & Explainability (Milestone 5)."""

import pytest
from backend.reasoning.explainer import ReasoningEngine
from backend.reasoning.schemas import ComplexityFactor, HistoricalPrecedent, RiskExplanation
from backend.retrieval.schemas import RetrievedMatch
from backend.risk.schemas import RiskCategory, RiskPrediction


@pytest.fixture
def sample_high_risk_prediction():
    """Sample high-risk prediction mimicking Flask.run or similar network function."""
    match = RetrievedMatch(
        similarity_score=0.942,
        repository="https://github.com/pallets/flask",
        commit_sha="05e9c6bd6bc9b39b34383c27ff77f805a5dfb0aa",
        parent_sha="7203feab2ba2b9cb0c6b1a20ee4db82bfdd8e04b",
        file_path="src/flask/app.py",
        function_name="Flask.run",
        commit_message="Fix `.partition(':')` usage on potential IPv6 addresses (#6096)",
        buggy_code="host, _, port = server_name.partition(':')",
        fixed_code="if server_name.startswith('['): ...",
        diff="- host, _, port = server_name.partition(':')\n+ if server_name.startswith('['): ...",
        detector_score=0.8,
        detector_classification="BUG_FIX_CANDIDATE",
        timestamp="2024-05-01T12:00:00Z",
    )

    return RiskPrediction(
        function_name="Flask.run",
        file_path="src/flask/app.py",
        risk_probability=0.92,
        risk_score=9.3,
        risk_category=RiskCategory.CRITICAL,
        is_risky=True,
        model_name="NumpyRandomForest",
        features_used={
            "lines_of_code": 45,
            "cyclomatic_complexity": 9,
            "max_nesting_depth": 3,
            "has_exception_handling": 0,
            "num_branches": 8,
            "top1_similarity": 0.942,
        },
        top_retrieved_matches=[match],
        reasoning_summary="High similarity to commit 05e9c6bd and high branch complexity.",
    )


@pytest.fixture
def sample_code_snippet():
    return """def run(self, host=None, port=None, debug=None, **options):
    server_name = options.get("SERVER_NAME")
    if server_name:
        host, _, port = server_name.partition(":")
        port = int(port)
    if debug:
        for attempt in range(3):
            if attempt > 1:
                if host == "localhost":
                    options["ssl"] = True
    return serve(host, port, **options)
"""


def test_explain_high_risk(sample_high_risk_prediction, sample_code_snippet):
    """Test generating explanation for a high-risk prediction."""
    engine = ReasoningEngine(provider="deterministic")
    explanation = engine.explain(
        prediction=sample_high_risk_prediction,
        code=sample_code_snippet,
        file_path="src/flask/app.py",
        language="python",
    )

    assert isinstance(explanation, RiskExplanation)
    assert explanation.function_name == "Flask.run"
    assert explanation.risk_category == RiskCategory.CRITICAL
    assert explanation.risk_score == 9.3
    assert explanation.is_risky is True

    # Check complexity factors
    assert len(explanation.complexity_factors) >= 2
    factor_names = [f.name for f in explanation.complexity_factors]
    assert "Cyclomatic Complexity" in factor_names
    assert "Nesting Depth" in factor_names

    # Check historical precedents
    assert len(explanation.historical_precedents) == 1
    prec = explanation.historical_precedents[0]
    assert "05e9c6bd" in prec.commit_sha
    assert prec.similarity_score == 0.942

    # Check vulnerable patterns & mitigations
    assert len(explanation.vulnerable_patterns) >= 1
    assert any("partition" in p or "IPv6" in p for p in explanation.vulnerable_patterns)
    assert len(explanation.mitigation_strategies) >= 1


def test_explain_low_risk():
    """Test generating explanation for a simple, low-risk function."""
    prediction = RiskPrediction(
        function_name="simple_add",
        file_path="src/utils.py",
        risk_probability=0.08,
        risk_score=1.7,
        risk_category=RiskCategory.LOW,
        is_risky=False,
        model_name="NumpyLogisticRegression",
        features_used={
            "lines_of_code": 3,
            "cyclomatic_complexity": 1,
            "max_nesting_depth": 0,
            "has_exception_handling": 0,
            "num_branches": 0,
            "top1_similarity": 0.15,
        },
        top_retrieved_matches=[],
    )

    code = "def simple_add(a: int, b: int) -> int:\n    return a + b\n"
    engine = ReasoningEngine(provider="deterministic")
    explanation = engine.explain(prediction, code, file_path="src/utils.py")

    assert explanation.risk_category == RiskCategory.LOW
    assert explanation.is_risky is False
    assert "LOW risk" in explanation.summary
    assert len(explanation.historical_precedents) == 0


def test_vulnerable_pattern_detection():
    """Test pattern detection on integer conversion and partition."""
    code = """def parse(val):
    h, _, p = val.partition(":")
    return int(p)
"""
    engine = ReasoningEngine(provider="deterministic")
    patterns = engine._identify_vulnerable_patterns(code, "python", [])
    assert any("delimiter split" in p or "colon" in p for p in patterns)
    assert any("int()" in p for p in patterns)
