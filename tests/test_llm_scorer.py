"""Unit tests for LLM Scorer (Prompt generation, JSON parsing, validation, MockLLMScorer)."""

import pytest
from backend.retrieval.schemas import RetrievedMatch
from backend.risk.llm_scorer import MockLLMScorer


def test_mock_llm_prompt_building():
    scorer = MockLLMScorer()
    matches = [
        RetrievedMatch(
            similarity_score=0.95,
            repository="repo",
            commit_sha="c12345678",
            parent_sha="p1",
            file_path="src/flask/app.py",
            function_name="Flask.run",
            commit_message="Fix IPv6 address splitting",
            buggy_code="def run(): host.partition(':')",
            fixed_code="def run(): urlsplit(host)",
            diff="--- a\n+++ b",
            detector_score=0.8,
            detector_classification="BUG_FIX_CANDIDATE",
            timestamp="2026-01-01T00:00:00Z",
        )
    ]

    prompt = scorer.build_prompt(
        code="def run(): host.partition(':')",
        function_name="Flask.run",
        file_path="src/flask/app.py",
        retrieved_matches=matches,
        language="python",
    )
    assert "TARGET FUNCTION" in prompt
    assert "HISTORICAL EXAMPLE 1" in prompt
    assert "Fix IPv6 address splitting" in prompt
    assert "OUTPUT INSTRUCTION" in prompt


def test_mock_llm_parse_valid_json():
    scorer = MockLLMScorer()
    valid_json_text = """
    ```json
    {
        "risk_probability": 0.85,
        "risk_score": 8.6,
        "classification": "HIGH",
        "reasoning_summary": "Function shares vulnerable string splitting pattern with commit c1234567."
    }
    ```
    """
    parsed = scorer.parse_and_validate_response(valid_json_text)
    assert parsed.risk_probability == 0.85
    assert parsed.risk_score == 8.6
    assert parsed.classification == "HIGH"
    assert "shares vulnerable" in parsed.reasoning_summary


def test_mock_llm_parse_invalid_json():
    scorer = MockLLMScorer()
    invalid_text = "I think this function has an 80% chance of being buggy."
    with pytest.raises(ValueError, match="not valid JSON"):
        scorer.parse_and_validate_response(invalid_text)


def test_mock_llm_predict_high_risk():
    scorer = MockLLMScorer()
    matches = [
        RetrievedMatch(
            similarity_score=0.96,
            repository="repo",
            commit_sha="c1",
            parent_sha="p1",
            file_path="app.py",
            function_name="run",
            commit_message="Fix bug",
            buggy_code="pass",
            fixed_code="return",
            diff="diff",
            detector_score=0.8,
            detector_classification="BUG_FIX_CANDIDATE",
            timestamp="2026-01-01T00:00:00Z",
        )
    ]
    pred = scorer.predict(
        code="def run(): pass",
        function_name="run",
        file_path="app.py",
        retrieved_matches=matches,
    )
    assert pred.risk_probability >= 0.8
    assert pred.risk_score >= 8.0
    assert pred.is_risky is True


def test_mock_llm_predict_low_risk():
    scorer = MockLLMScorer()
    # No matches
    pred = scorer.predict(
        code="def add(a, b): return a + b",
        function_name="add",
        file_path="math.py",
        retrieved_matches=[],
    )
    assert pred.risk_probability < 0.3
    assert pred.risk_score < 4.0
    assert pred.is_risky is False
