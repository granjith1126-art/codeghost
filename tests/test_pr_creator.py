"""Unit tests for CodeGhost GitHub PR Creation (Milestone 7)."""

import os
import tempfile
import pytest
from backend.autofix.schemas import FixCandidate, FixStrategy, SandboxExecutionResult, VerifiedFix
from backend.pr.creator import PRCreator
from backend.pr.formatter import PRFormatter
from backend.pr.schemas import PRDraft, PRResult
from backend.reasoning.schemas import HistoricalPrecedent, RiskCategory, RiskExplanation


@pytest.fixture
def sample_verified_fix():
    candidate = FixCandidate(
        fix_id="fix_ipv6_harden",
        function_name="Flask.run",
        file_path="src/flask/app.py",
        strategy=FixStrategy.HISTORICAL_ANALOGY,
        description="Handles bracketed IPv6 addresses before colon partitioning.",
        original_code="host, _, port = server_name.partition(':')",
        fixed_code="if server_name.startswith('['):\n    host, _, port = server_name[1:].partition(']:')\nelse:\n    host, _, port = server_name.partition(':')",
        diff="--- a/src/flask/app.py\n+++ b/src/flask/app.py\n@@ -1 +1,4 @@\n-host, _, port = server_name.partition(':')\n+if server_name.startswith('['):\n+    host, _, port = server_name[1:].partition(']:')\n+else:\n+    host, _, port = server_name.partition(':')",
        confidence=0.95,
    )

    sb_result = SandboxExecutionResult(
        success=True,
        tests_run=12,
        tests_passed=12,
        tests_failed=0,
        tests_errored=0,
        exit_code=0,
        stdout="12 passed in 0.42s",
        duration_seconds=0.42,
    )

    return VerifiedFix(
        candidate=candidate,
        syntax_valid=True,
        sandbox_result=sb_result,
        is_verified=True,
        verification_summary="Verified: Fix passed 12/12 sandbox tests.",
    )


@pytest.fixture
def sample_explanation():
    prec = HistoricalPrecedent(
        commit_sha="05e9c6bd6bc9b39b34383c27ff77f805a5dfb0aa",
        similarity_score=0.942,
        commit_message="Fix `.partition(':')` usage on potential IPv6 addresses (#6096)",
        file_path="src/flask/app.py",
        function_name="Flask.run",
        relevance_reason="Identical host partitioning pattern",
    )
    return RiskExplanation(
        function_name="Flask.run",
        file_path="src/flask/app.py",
        risk_score=9.3,
        risk_category=RiskCategory.CRITICAL,
        is_risky=True,
        summary="Critical risk host partitioning without IPv6 checks.",
        detailed_analysis="Function splits host:port without handling bracketed IPv6 notation.",
        historical_precedents=[prec],
        provider="deterministic",
        model_name="test",
    )


def test_pr_formatter(sample_verified_fix, sample_explanation):
    """Test generating complete PR draft with markdown formatting."""
    formatter = PRFormatter()
    draft = formatter.format_draft(
        verified_fix=sample_verified_fix,
        explanation=sample_explanation,
        repo_identifier="https://github.com/pallets/flask",
        base_branch="main",
    )

    assert isinstance(draft, PRDraft)
    assert "Flask.run" in draft.title
    assert "codeghost/fix-flask_run" in draft.branch_name
    assert draft.risk_score == 9.3
    assert draft.risk_category == "CRITICAL"
    assert draft.is_verified is True

    # Check Markdown body sections
    body = draft.body_markdown
    assert "🛡️ CodeGhost Automated Fix" in body
    assert "Executive Summary" in body
    assert "Historical Repository Precedents" in body
    assert "05e9c6bd" in body
    assert "Isolated Sandbox Verification" in body
    assert "Reviewer Checklist" in body


def test_pr_creator_dry_run(sample_verified_fix, sample_explanation):
    """Test PRCreator dry run serializing patch file to disk."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        creator = PRCreator()
        result = creator.create_pr(
            verified_fix=sample_verified_fix,
            explanation=sample_explanation,
            repo_url_or_path="https://github.com/pallets/flask",
            dry_run=True,
            patches_dir=tmp_dir,
        )

        assert isinstance(result, PRResult)
        assert result.success is True
        assert result.dry_run is True
        assert result.patch_file_path is not None
        assert os.path.exists(result.patch_file_path)

        with open(result.patch_file_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "CodeGhost Pull Request Patch" in content
            assert "host, _, port" in content


def test_parse_owner_repo():
    """Test repository URL parsing."""
    creator = PRCreator()
    assert creator._parse_owner_repo("https://github.com/pallets/flask") == ("pallets", "flask")
    assert creator._parse_owner_repo("https://github.com/pallets/flask.git") == ("pallets", "flask")
    assert creator._parse_owner_repo("git@github.com:pallets/flask.git") == ("pallets", "flask")
    assert creator._parse_owner_repo("pallets/flask") == ("pallets", "flask")
    assert creator._parse_owner_repo("invalid_repo_string") is None
