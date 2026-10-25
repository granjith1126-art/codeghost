"""Deterministic End-to-End Integration Test for CodeGhost.

Constructs an isolated, self-contained Git repository with a reproducible bug,
mines history, retrieves past precedents, scores risk, synthesizes fixes,
and validates PR draft generation.
"""

import os
import shutil
import tempfile
from pathlib import Path
import git
import pytest

from backend.autofix.generator import FixGenerator
from backend.autofix.sandbox import TestSandbox
from backend.history.extractor import HistoryExtractor
from backend.pr.creator import PRCreator
from backend.reasoning.explainer import ReasoningEngine
from backend.risk.baseline_model import BaselineRiskModel
from backend.risk.schemas import RiskCategory, RiskPrediction


@pytest.fixture
def controlled_git_repo():
    """Create a controlled git repo fixture with reproducible commits."""
    temp_dir = tempfile.mkdtemp(prefix="codeghost_e2e_fixture_")
    repo = git.Repo.init(temp_dir)
    
    app_file = os.path.join(temp_dir, "calculator.py")
    test_file = os.path.join(temp_dir, "test_calculator.py")

    # Commit 1: Initial implementation
    with open(app_file, "w", encoding="utf-8") as f:
        f.write("def divide(a, b):\n    return a / b\n")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("from calculator import divide\ndef test_divide():\n    assert divide(10, 2) == 5\n")

    repo.index.add(["calculator.py", "test_calculator.py"])
    repo.index.commit("Initial commit: basic division")

    # Commit 2: Buggy commit (causes regression on negative or string input)
    with open(app_file, "w", encoding="utf-8") as f:
        f.write("def divide(a, b):\n    # Risky unshielded conversion\n    val = int(a) / int(b)\n    return val\n")
    repo.index.add(["calculator.py"])
    repo.index.commit("feat: add integer cast in divide")

    # Commit 3: Fix commit
    with open(app_file, "w", encoding="utf-8") as f:
        f.write("def divide(a, b):\n    if b == 0:\n        raise ZeroDivisionError('Division by zero')\n    return float(a) / float(b)\n")
    repo.index.add(["calculator.py"])
    repo.index.commit("fix: prevent zero division and cast safely (#101)")

    yield temp_dir

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_e2e_history_mining(controlled_git_repo):
    """Test history extractor on the controlled fixture."""
    extractor = HistoryExtractor(base_storage_dir=Path(controlled_git_repo))
    res = extractor.extract_history(repo_url_or_path=controlled_git_repo, max_commits=10)
    assert res.commits_scanned >= 3
    assert res.bug_fix_pairs_extracted >= 1


def test_e2e_autonomous_cycle(controlled_git_repo):
    """Test full cycle: prediction -> reasoning -> fix -> sandbox -> PR."""
    code = "def divide(a, b):\n    val = int(a) / int(b)\n    return val\n"
    
    # 1. Prediction
    pred = RiskPrediction(
        function_name="divide",
        file_path="calculator.py",
        risk_probability=0.88,
        risk_score=8.9,
        risk_category=RiskCategory.HIGH,
        is_risky=True,
        model_name="NumpyRandomForest",
        features_used={"cyclomatic_complexity": 4, "lines_of_code": 4},
        top_retrieved_matches=[],
    )

    # 2. Reasoning
    engine = ReasoningEngine(provider="deterministic")
    exp = engine.explain(pred, code=code, file_path="calculator.py")
    assert exp.risk_category == RiskCategory.HIGH

    # 3. Fix Generation
    gen = FixGenerator(provider="deterministic")
    fixes = gen.generate_fixes("divide", code, "calculator.py", exp)
    assert len(fixes) >= 1

    # 4. Sandbox Verification
    sandbox = TestSandbox(timeout_seconds=15)
    verified = [sandbox.verify_fix(f) for f in fixes]
    assert any(vf.syntax_valid for vf in verified)

    # 5. PR Creator Dry-Run
    creator = PRCreator()
    pr_res = creator.create_pr(verified[0], exp, repo_url_or_path=controlled_git_repo, dry_run=True)
    assert pr_res.success is True
    assert pr_res.patch_file_path is not None
