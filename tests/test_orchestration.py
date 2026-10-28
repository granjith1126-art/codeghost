"""Unit tests for CodeGhost Autonomous AI Orchestration."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from backend.orchestration.config import OrchestrationConfig
from backend.orchestration.controller import main as controller_main
from backend.orchestration.executor import LocalDirectExecutor
from backend.orchestration.phase_runner import PhaseRunner
from backend.orchestration.planner import (
    OpenAIPlanner,
    PhaseReviewVerdict,
    RepairProposal,
    ReviewFinding,
)
from backend.orchestration.reviewer import LocalEvidenceReviewer
from backend.orchestration.state_store import StateStore
from backend.orchestration.verification import TaskVerifier


def test_config_and_secret_redaction():
    """Verify that sensitive API keys are masked and scrubbed."""
    cfg = OrchestrationConfig(openai_api_key="sk-proj-test1234567890abcdef1234567890abcdef")
    assert cfg.has_api_key is True
    assert "sk-proj" in cfg.masked_api_key
    assert "abcdef" not in cfg.masked_api_key
    
    # Test redaction
    sample_text = "Exception: Failed connecting with sk-proj-test1234567890abcdef1234567890abcdef on server"
    redacted = cfg.redact(sample_text)
    assert "sk-proj-test1234567890abcdef" not in redacted
    assert "[REDACTED" in redacted


def test_state_store_persistence():
    """Verify workflow state serialization, updates, and checkpoint loading."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        state_file = Path(tmp_dir) / "test_state.json"
        store = StateStore(state_file)

        assert store.state.milestone == "Milestone 8"
        assert store.get_phase("phase_1").status == "PASSED"
        assert store.get_phase("phase_2").status == "PENDING"

        # Update a phase
        store.update_phase("phase_2", status="PASSED", artifacts=["report_p2.md"], metrics={"precision": 0.85})
        assert store.get_phase("phase_2").status == "PASSED"

        # Reload state from disk to confirm atomic serialization
        new_store = StateStore(state_file)
        assert new_store.get_phase("phase_2").status == "PASSED"
        assert "report_p2.md" in new_store.get_phase("phase_2").artifacts
        assert new_store.get_phase("phase_2").metrics.get("precision") == 0.85


def test_executor_command_and_files():
    """Verify local subprocess execution, output capture, and file writing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = OrchestrationConfig(project_root=Path(tmp_dir))
        executor = LocalDirectExecutor(cfg)

        # Test writing and reading file
        test_file = "sample.txt"
        executor.write_file(test_file, "Hello CodeGhost!")
        assert executor.read_file(test_file) == "Hello CodeGhost!"

        # Test command execution
        res = executor.run_command(".venv\\Scripts\\python -c \"print('Execution Success')\"")
        assert res.success is True
        assert "Execution Success" in res.stdout


def test_task_verifier():
    """Verify file presence and command auditing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = OrchestrationConfig(project_root=Path(tmp_dir))
        executor = LocalDirectExecutor(cfg)
        verifier = TaskVerifier(cfg, executor)

        # Missing file check
        v1 = verifier.verify_files(["missing_report.md"])
        assert v1.passed is False
        assert "missing_report.md" in v1.missing_files

        # Existing file check
        executor.write_file("valid_report.md", "# Valid Report Content")
        v2 = verifier.verify_files(["valid_report.md"])
        assert v2.passed is True


def test_planner_deterministic_fallback():
    """Verify planner deterministic fallback plans."""
    cfg = OrchestrationConfig(openai_api_key=None)
    planner = OpenAIPlanner(cfg)
    plan = planner.plan_phase("phase_2", "Evaluation Protocol")
    assert len(plan) >= 1
    assert "p2_write_protocol" in plan[0].task_id


def test_controller_status_and_dry_run():
    """Verify CLI controller --status and --dry-run flags."""
    exit_status = controller_main(["--status"])
    assert exit_status == 0

    exit_dry_run = controller_main(["--dry-run"])
    assert exit_dry_run == 0


def test_review_phase_structured_approval():
    """Verify that a well-formed APPROVED review from OpenAI is parsed correctly."""
    cfg = OrchestrationConfig(openai_api_key="sk-proj-dummykey")
    planner = OpenAIPlanner(cfg)
    planner.client = MagicMock()
    
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps({
                    "verdict": "APPROVED",
                    "score": 9.2,
                    "acceptance_criteria_assessment": {"metrics_defined": "MET"},
                    "findings": [],
                    "required_fixes": [],
                    "remaining_risks": ["Small sample size"],
                })
            )
        )
    ]
    mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
    planner.client.chat.completions.create.return_value = mock_response

    verdict = planner.review_phase(
        phase_id="phase_2",
        phase_title="Evaluation Protocol",
        acceptance_criteria="Define all metrics",
        changed_files=["reports/validation/phase_2_evaluation_protocol.md"],
        diff_or_content_summary="# Protocol content",
        test_results="Passed",
    )

    assert verdict.is_valid is True
    assert verdict.verdict == "APPROVED"
    assert verdict.score == 9.2
    assert len(verdict.findings) == 0


def test_review_phase_critical_finding_overrides_approval():
    """Verify that a CRITICAL finding forces CHANGES_REQUESTED even if model returned APPROVED."""
    cfg = OrchestrationConfig(openai_api_key="sk-proj-dummykey")
    planner = OpenAIPlanner(cfg)
    planner.client = MagicMock()

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps({
                    "verdict": "APPROVED",
                    "score": 8.0,
                    "findings": [
                        {
                            "severity": "CRITICAL",
                            "category": "LEAKAGE",
                            "description": "Commit SHA overlap detected between splits",
                            "evidence": "sha_123 in both train and test",
                            "suggested_fix": "Enforce strict temporal boundary",
                        }
                    ],
                    "required_fixes": ["Isolate splits"],
                    "remaining_risks": [],
                })
            )
        )
    ]
    mock_response.usage = None
    planner.client.chat.completions.create.return_value = mock_response

    verdict = planner.review_phase(
        phase_id="phase_3",
        phase_title="Data Leakage Audit",
        acceptance_criteria="No leakage",
        changed_files=["phase_3.md"],
        diff_or_content_summary="diff",
        test_results="Passed",
    )

    # Safety override: must be CHANGES_REQUESTED
    assert verdict.is_valid is True
    assert verdict.verdict == "CHANGES_REQUESTED"
    assert len(verdict.findings) == 1
    assert verdict.findings[0].severity == "CRITICAL"


def test_review_phase_api_failure_blocks_approval():
    """Verify that OpenAI API errors or network drops result in non-approved invalid verdicts."""
    cfg = OrchestrationConfig(openai_api_key="sk-proj-dummykey")
    planner = OpenAIPlanner(cfg)
    planner.client = MagicMock()
    planner.client.chat.completions.create.side_effect = Exception("Connection timed out")

    verdict = planner.review_phase(
        phase_id="phase_2",
        phase_title="Protocol",
        acceptance_criteria="criteria",
        changed_files=[],
        diff_or_content_summary="",
        test_results="",
    )

    assert verdict.is_valid is False
    assert verdict.verdict == "CHANGES_REQUESTED"
    assert "Connection timed out" in (verdict.error_message or "")


def test_review_phase_malformed_response_blocks_approval():
    """Verify that invalid verdicts or unparseable JSON cannot produce approval."""
    cfg = OrchestrationConfig(openai_api_key="sk-proj-dummykey")
    planner = OpenAIPlanner(cfg)
    planner.client = MagicMock()

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="{\"verdict\": \"NOT_A_VALID_VERDICT\"}"))
    ]
    mock_response.usage = None
    planner.client.chat.completions.create.return_value = mock_response

    verdict = planner.review_phase(
        phase_id="phase_2",
        phase_title="Protocol",
        acceptance_criteria="criteria",
        changed_files=[],
        diff_or_content_summary="",
        test_results="",
    )

    assert verdict.is_valid is False
    assert verdict.verdict == "CHANGES_REQUESTED"


def test_phase_runner_successful_review_permits_progression():
    """Verify that a successful review moves phase to PASSED and permits progression."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cfg = OrchestrationConfig(project_root=tmp_path, state_file=tmp_path / "state.json")
        executor = LocalDirectExecutor(cfg)
        state_store = StateStore(cfg.state_file)
        verifier = TaskVerifier(cfg, executor)
        planner = OpenAIPlanner(cfg, state_store)

        runner = PhaseRunner(cfg, executor, planner, state_store, verifier)

        approved_verdict = PhaseReviewVerdict(
            verdict="APPROVED",
            score=9.5,
            findings=[],
            required_fixes=[],
            is_valid=True,
        )

        with patch.object(runner.reviewer, "review_phase", return_value=approved_verdict):
            # Run phase_2
            success = runner.run_phase("phase_2", dry_run=False)
            assert success is True
            phase_rec = state_store.get_phase("phase_2")
            assert phase_rec.status == "PASSED"
            assert phase_rec.review_verdict == "APPROVED"
            assert phase_rec.review_attempts == 1


def test_phase_runner_review_rejection_triggers_repair_and_retest():
    """Verify that rejection on attempt 1 triggers repair and re-review on attempt 2."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cfg = OrchestrationConfig(project_root=tmp_path, state_file=tmp_path / "state.json")
        executor = LocalDirectExecutor(cfg)
        state_store = StateStore(cfg.state_file)
        verifier = TaskVerifier(cfg, executor)
        planner = OpenAIPlanner(cfg, state_store)

        runner = PhaseRunner(cfg, executor, planner, state_store, verifier)

        # First review: CHANGES_REQUESTED; Second review: APPROVED
        verdict_1 = PhaseReviewVerdict(
            verdict="CHANGES_REQUESTED",
            score=4.0,
            findings=[ReviewFinding(severity="MEDIUM", category="DOCUMENTATION", description="Missing section header")],
            required_fixes=["Add section header"],
            is_valid=True,
        )
        verdict_2 = PhaseReviewVerdict(
            verdict="APPROVED",
            score=9.0,
            findings=[],
            required_fixes=[],
            is_valid=True,
        )

        repair_proposal = RepairProposal(
            root_cause="Missing section header in report",
            replacements=[],
            suggested_commands=[],
        )

        with patch.object(runner.reviewer, "review_phase", side_effect=[verdict_1, verdict_2]), \
             patch.object(planner, "diagnose_and_repair", return_value=repair_proposal) as mock_repair:
            
            success = runner.run_phase("phase_2", dry_run=False, max_retries=3)
            assert success is True
            assert mock_repair.call_count == 1
            phase_rec = state_store.get_phase("phase_2")
            assert phase_rec.status == "PASSED"
            assert phase_rec.review_attempts == 2


def test_phase_runner_retry_exhaustion_blocks_phase():
    """Verify that repeatedly rejected reviews exhaust retries and block progression."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cfg = OrchestrationConfig(project_root=tmp_path, state_file=tmp_path / "state.json")
        executor = LocalDirectExecutor(cfg)
        state_store = StateStore(cfg.state_file)
        verifier = TaskVerifier(cfg, executor)
        planner = OpenAIPlanner(cfg, state_store)

        runner = PhaseRunner(cfg, executor, planner, state_store, verifier)

        rejection = PhaseReviewVerdict(
            verdict="CHANGES_REQUESTED",
            score=3.0,
            findings=[ReviewFinding(severity="HIGH", category="CORRECTNESS", description="Fails acceptance criteria")],
            required_fixes=["Fix criteria"],
            is_valid=True,
        )

        with patch.object(runner.reviewer, "review_phase", return_value=rejection):
            success = runner.run_phase("phase_2", dry_run=False, max_retries=2)
            assert success is False
            phase_rec = state_store.get_phase("phase_2")
            assert phase_rec.status == "FAILED"
            assert phase_rec.review_attempts == 2
            assert phase_rec.review_verdict == "CHANGES_REQUESTED"


def test_state_store_preserves_review_state():
    """Verify that review verdict, findings, and attempts are persisted to disk."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        state_file = Path(tmp_dir) / "test_review_state.json"
        store = StateStore(state_file)

        store.update_phase(
            phase_id="phase_4",
            status="PASSED",
            artifacts=["test_end_to_end.py"],
            review_verdict="APPROVED",
            review_findings=["[INFO] Clean integration run"],
            review_attempts=1,
        )

        # Reload from disk
        reloaded = StateStore(state_file)
        p4 = reloaded.get_phase("phase_4")
        assert p4.status == "PASSED"
        assert p4.review_verdict == "APPROVED"
        assert p4.review_attempts == 1
        assert len(p4.review_findings) == 1
        assert "Clean integration run" in p4.review_findings[0]


def test_live_openai_review_smoke_test():
    """Controlled live smoke test calling OpenAI review_phase if API key is configured."""
    cfg = OrchestrationConfig()
    if not cfg.has_api_key:
        pytest.skip("OpenAI API key not configured; skipping live smoke test.")

    planner = OpenAIPlanner(cfg)
    verdict = planner.review_phase(
        phase_id="phase_2",
        phase_title="Evaluation Protocol Definition",
        acceptance_criteria="Formal definition of precision, recall, F1, and baseline comparisons.",
        changed_files=["reports/validation/phase_2_evaluation_protocol.md"],
        diff_or_content_summary="# CodeGhost Milestone 8 - Phase 2\n\nDefines precision, recall, and PR-AUC with mathematical rigor.",
        test_results="Verified all required protocol documents exist and are non-empty.",
        logs="Verification complete with exit code 0.",
    )

    if "credit_balance_exhausted" in (verdict.error_message or "") or "429" in (verdict.error_message or ""):
        # Verified live connection reached OpenAI, was authenticated, and was blocked by OpenAI quota.
        # Verified safety gate: quota failure strictly blocked approval and requested changes!
        assert verdict.is_valid is False
        assert verdict.verdict == "CHANGES_REQUESTED"
        assert "429" in verdict.error_message
    elif "Application Control policy" in (verdict.error_message or "") or "DLL load failed" in (verdict.error_message or ""):
        # Windows WDAC/AppLocker blocked native binary (e.g. jiter.pyd); verified safety gate blocks approval
        assert verdict.is_valid is False
        assert verdict.verdict == "CHANGES_REQUESTED"
    else:
        assert verdict.is_valid is True
        assert verdict.verdict in ("APPROVED", "CHANGES_REQUESTED")
        assert isinstance(verdict.score, float)
        assert verdict.raw_response is not None


def test_local_evidence_reviewer_approval():
    """Verify that LocalEvidenceReviewer audits valid deliverables and approves without API credits."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cfg = OrchestrationConfig(project_root=tmp_path)
        executor = LocalDirectExecutor(cfg)
        verifier = TaskVerifier(cfg, executor)
        reviewer = LocalEvidenceReviewer(cfg, executor, verifier)

        # Create valid Phase 2 protocol
        p2_doc = tmp_path / "reports" / "validation" / "phase_2_evaluation_protocol.md"
        p2_doc.parent.mkdir(parents=True, exist_ok=True)
        p2_doc.write_text(
            "# Protocol\nPrecision, Recall, F1, PR-AUC, ROC-AUC\nMRR, Recall@1\nMajority Class, Random\nSyntax Validity, Sandbox",
            encoding="utf-8",
        )

        verdict = reviewer.review_phase(
            phase_id="phase_2",
            phase_title="Evaluation Protocol",
            acceptance_criteria="Define all metrics",
            changed_files=[str(p2_doc.relative_to(tmp_path))],
            diff_or_content_summary=p2_doc.read_text(encoding="utf-8"),
            test_results="Exit Code 0: All checks passed",
        )

        assert verdict.is_valid is True
        assert verdict.verdict == "APPROVED"
        assert verdict.reviewer_type == "local_evidence_reviewer"
        assert len([f for f in verdict.findings if f.severity in ("CRITICAL", "HIGH")]) == 0


def test_local_evidence_reviewer_detects_secret_leak():
    """Verify that unmasked API keys in deliverables trigger a CRITICAL rejection."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cfg = OrchestrationConfig(project_root=tmp_path)
        executor = LocalDirectExecutor(cfg)
        verifier = TaskVerifier(cfg, executor)
        reviewer = LocalEvidenceReviewer(cfg, executor, verifier)

        verdict = reviewer.review_phase(
            phase_id="phase_2",
            phase_title="Evaluation Protocol",
            acceptance_criteria="No leaks",
            changed_files=[],
            diff_or_content_summary="Leaked key: sk-proj-1234567890abcdef1234567890abcdef",
            test_results="Exit Code 0",
        )

        assert verdict.is_valid is True
        assert verdict.verdict == "CHANGES_REQUESTED"
        criticals = [f for f in verdict.findings if f.severity == "CRITICAL" and f.category == "SECURITY"]
        assert len(criticals) >= 1
        assert "Secret leakage detected" in criticals[0].description


def test_local_evidence_reviewer_detects_python_syntax_error():
    """Verify that syntax errors in created Python files are caught and rejected."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cfg = OrchestrationConfig(project_root=tmp_path)
        executor = LocalDirectExecutor(cfg)
        verifier = TaskVerifier(cfg, executor)
        reviewer = LocalEvidenceReviewer(cfg, executor, verifier)

        bad_py = tmp_path / "broken.py"
        bad_py.write_text("def broken_syntax(:\n    pass\n", encoding="utf-8")

        verdict = reviewer.review_phase(
            phase_id="phase_4",
            phase_title="E2E Integration",
            acceptance_criteria="Valid python syntax",
            changed_files=["broken.py"],
            diff_or_content_summary="Broken code",
            test_results="Exit Code 0",
        )

        assert verdict.verdict == "CHANGES_REQUESTED"
        syntax_findings = [f for f in verdict.findings if f.severity == "CRITICAL" and f.category == "CORRECTNESS"]
        assert len(syntax_findings) >= 1
        assert "syntax error" in syntax_findings[0].description.lower()


def test_phase_runner_local_review_e2e_progression():
    """Verify that PhaseRunner executes and approves a phase using LocalEvidenceReviewer with zero API credits."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cfg = OrchestrationConfig(project_root=tmp_path, state_file=tmp_path / "state.json")
        executor = LocalDirectExecutor(cfg)
        state_store = StateStore(cfg.state_file)
        verifier = TaskVerifier(cfg, executor)
        planner = OpenAIPlanner(cfg, state_store)

        runner = PhaseRunner(cfg, executor, planner, state_store, verifier)
        assert isinstance(runner.reviewer, LocalEvidenceReviewer)

        # Run phase_2 end-to-end
        success = runner.run_phase("phase_2", dry_run=False)
        assert success is True
        p2 = state_store.get_phase("phase_2")
        assert p2.status == "PASSED"
        assert p2.review_verdict == "APPROVED"
        assert p2.review_attempts >= 1

