"""Unit tests for CodeGhost Autonomous AI Orchestration."""

import tempfile
from pathlib import Path
import pytest

from backend.orchestration.config import OrchestrationConfig
from backend.orchestration.controller import main as controller_main
from backend.orchestration.executor import LocalDirectExecutor
from backend.orchestration.planner import OpenAIPlanner
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
