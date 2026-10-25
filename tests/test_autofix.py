"""Unit tests for CodeGhost Automated Fix Generation & Test Sandbox (Milestone 6)."""

import pytest
from backend.autofix.generator import FixGenerator
from backend.autofix.sandbox import TestSandbox
from backend.autofix.schemas import FixCandidate, FixStrategy, VerifiedFix
from backend.reasoning.schemas import HistoricalPrecedent, RiskCategory, RiskExplanation


@pytest.fixture
def sample_explanation():
    prec = HistoricalPrecedent(
        commit_sha="05e9c6bd6bc9b39b34383c27ff77f805a5dfb0aa",
        similarity_score=0.95,
        commit_message="Fix `.partition(':')` usage on potential IPv6 addresses (#6096)",
        file_path="src/flask/app.py",
        function_name="Flask.run",
        relevance_reason="Shared string partitioning on host:port",
    )
    return RiskExplanation(
        function_name="Flask.run",
        file_path="src/flask/app.py",
        risk_score=9.5,
        risk_category=RiskCategory.CRITICAL,
        is_risky=True,
        summary="High risk host partitioning",
        detailed_analysis="Function splits host:port without handling bracketed IPv6.",
        historical_precedents=[prec],
        provider="deterministic",
        model_name="test",
    )


def test_generate_historical_analogy_fix(sample_explanation):
    """Test adapting a historical bug fix for host:port partitioning."""
    code = """def run(self, host=None, port=None):
    server_name = "127.0.0.1:5000"
    host, _, port = server_name.partition(":")
    return host, port
"""
    generator = FixGenerator()
    fixes = generator.generate_fixes("Flask.run", code, "src/flask/app.py", sample_explanation)

    assert len(fixes) >= 1
    analogy_fix = next((f for f in fixes if f.strategy == FixStrategy.HISTORICAL_ANALOGY), None)
    assert analogy_fix is not None
    assert "startswith('[')" in analogy_fix.fixed_code
    assert analogy_fix.confidence >= 0.90
    assert "--- a/src/flask/app.py" in analogy_fix.diff


def test_generate_defensive_and_exception_fixes():
    """Test generating defensive dict.get() and exception shield for int casts."""
    code = """def process_config(cfg):
    port = int(cfg["PORT"])
    return port
"""
    generator = FixGenerator()
    fixes = generator.generate_fixes("process_config", code, "config.py")

    assert len(fixes) >= 1
    # Check that exception shield was generated
    exc_fix = next((f for f in fixes if f.strategy == FixStrategy.EXCEPTION_SHIELD), None)
    assert exc_fix is not None
    assert "except (ValueError, TypeError):" in exc_fix.fixed_code

    # Check that defensive dict get was generated
    guard_fix = next((f for f in fixes if f.strategy == FixStrategy.DEFENSIVE_GUARD), None)
    assert guard_fix is not None
    assert "cfg.get(" in guard_fix.fixed_code


def test_syntax_validation():
    """Ensure that malformed code is caught and rejected."""
    generator = FixGenerator()
    assert generator._validate_syntax("def valid(): return 42", "python") is True
    assert generator._validate_syntax("def invalid(: return 42", "python") is False


def test_sandbox_verification():
    """Test executing candidate fix inside isolated temporary test sandbox."""
    candidate = FixCandidate(
        fix_id="fix_test_001",
        function_name="calculate_total",
        file_path="math_utils.py",
        strategy=FixStrategy.DEFENSIVE_GUARD,
        description="Safe addition",
        original_code="def calculate_total(a, b): return a + b\n",
        fixed_code="def calculate_total(a=0, b=0): return (a or 0) + (b or 0)\n",
        diff="--- a/math_utils.py\n+++ b/math_utils.py\n",
        confidence=0.9,
    )

    sandbox = TestSandbox(timeout_seconds=15)
    verified = sandbox.verify_fix(candidate)

    assert isinstance(verified, VerifiedFix)
    assert verified.syntax_valid is True
    assert verified.is_verified is True
    assert verified.sandbox_result is not None
    assert verified.sandbox_result.success is True
    assert verified.sandbox_result.tests_passed >= 1


def test_sandbox_rejects_syntax_error():
    """Ensure sandbox immediately rejects syntactically broken candidate."""
    candidate = FixCandidate(
        fix_id="fix_broken",
        function_name="broken",
        file_path="broken.py",
        strategy=FixStrategy.DEFENSIVE_GUARD,
        description="Broken syntax",
        original_code="def broken(): return 1\n",
        fixed_code="def broken( return 1\n",
        diff="",
        confidence=0.5,
    )

    sandbox = TestSandbox()
    verified = sandbox.verify_fix(candidate)
    assert verified.syntax_valid is False
    assert verified.is_verified is False
    assert "syntax error" in verified.verification_summary.lower()
