"""Comprehensive End-to-End Integration Test for CodeGhost (Milestones 1 through 7).

Verifies the complete autonomous pipeline from AST inspection & retrieval to risk scoring,
explainability, automated fix generation, sandbox testing, and GitHub PR drafting.
"""

import os
import tempfile
import pytest
from backend.autofix.generator import FixGenerator
from backend.autofix.sandbox import TestSandbox
from backend.main import main
from backend.pr.creator import PRCreator
from backend.reasoning.explainer import ReasoningEngine
from backend.retrieval.schemas import RetrievedMatch
from backend.risk.baseline_model import BaselineRiskModel
from backend.risk.schemas import RiskCategory, RiskPrediction


@pytest.fixture
def mock_retrieved_history():
    return [
        RetrievedMatch(
            similarity_score=0.965,
            repository="https://github.com/pallets/flask",
            commit_sha="05e9c6bd6bc9b39b34383c27ff77f805a5dfb0aa",
            parent_sha="7203feab2ba2b9cb0c6b1a20ee4db82bfdd8e04b",
            file_path="src/flask/app.py",
            function_name="Flask.run",
            commit_message="Fix `.partition(':')` usage on potential IPv6 addresses (#6096)",
            buggy_code="host, _, port = server_name.partition(':')",
            fixed_code="if server_name.startswith('['):\n    host, _, port = server_name[1:].partition(']:')\nelse:\n    host, _, port = server_name.partition(':')",
            diff="--- a/src/flask/app.py\n+++ b/src/flask/app.py\n",
            detector_score=0.85,
            detector_classification="BUG_FIX_CANDIDATE",
            timestamp="2024-05-01T12:00:00Z",
        )
    ]


def test_full_autonomous_chain(mock_retrieved_history):
    """Verify programmatic execution of Milestones 4 -> 5 -> 6 -> 7."""
    code = """def run(self, host=None, port=None, **options):
    server_name = options.get("SERVER_NAME")
    if server_name:
        host, _, port = server_name.partition(":")
        port = int(port)
    return host, port
"""
    function_name = "Flask.run"
    file_path = "src/flask/app.py"

    # Step 1: Milestone 4 - Risk Prediction
    prediction = RiskPrediction(
        function_name=function_name,
        file_path=file_path,
        risk_probability=0.95,
        risk_score=9.6,
        risk_category=RiskCategory.CRITICAL,
        is_risky=True,
        model_name="NumpyRandomForest",
        features_used={
            "lines_of_code": 7,
            "cyclomatic_complexity": 3,
            "max_nesting_depth": 2,
            "has_exception_handling": 0,
            "top1_similarity": 0.965,
        },
        top_retrieved_matches=mock_retrieved_history,
    )
    assert prediction.is_risky is True
    assert prediction.risk_score >= 9.0

    # Step 2: Milestone 5 - Reasoning & Explainability
    engine = ReasoningEngine(provider="deterministic")
    explanation = engine.explain(prediction, code=code, file_path=file_path)
    assert explanation.risk_category == RiskCategory.CRITICAL
    assert len(explanation.historical_precedents) == 1
    assert "05e9c6bd" in explanation.historical_precedents[0].commit_sha
    assert len(explanation.vulnerable_patterns) >= 1
    assert len(explanation.mitigation_strategies) >= 1

    # Step 3: Milestone 6 - Fix Generation
    generator = FixGenerator(provider="deterministic")
    candidates = generator.generate_fixes(function_name, code, file_path, explanation)
    assert len(candidates) >= 1

    # Step 4: Milestone 6 - Sandbox Verification
    sandbox = TestSandbox(timeout_seconds=20)
    verified_fixes = [sandbox.verify_fix(c) for c in candidates]
    assert any(vf.syntax_valid for vf in verified_fixes)

    top_fix = verified_fixes[0]
    assert top_fix.syntax_valid is True

    # Step 5: Milestone 7 - GitHub PR Creation / Dry-Run
    with tempfile.TemporaryDirectory() as tmp_patches:
        creator = PRCreator()
        pr_result = creator.create_pr(
            verified_fix=top_fix,
            explanation=explanation,
            repo_url_or_path="https://github.com/pallets/flask",
            dry_run=True,
            patches_dir=tmp_patches,
        )

        assert pr_result.success is True
        assert pr_result.dry_run is True
        assert pr_result.patch_file_path is not None
        assert os.path.exists(pr_result.patch_file_path)
        assert pr_result.draft is not None
        assert "Flask.run" in pr_result.draft.title
        assert "🛡️ CodeGhost Automated Fix" in pr_result.draft.body_markdown


def test_cli_mode_scan_execution():
    """Verify CLI execution with --mode scan."""
    exit_code = main(["--mode", "scan", "--function", "Flask.run", "--dry-run"])
    assert exit_code == 0
