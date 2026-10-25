"""Independent Task and Phase Verification Engine.

Enforces the core rule: The controller must independently verify completion;
a model-generated statement of success is never accepted as proof.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from backend.orchestration.config import OrchestrationConfig
from backend.orchestration.executor import BaseExecutor, ExecutionResult


@dataclass
class VerificationResult:
    passed: bool
    reason: str
    missing_files: List[str]
    failed_command_output: Optional[str] = None


class TaskVerifier:
    """Independent auditor verifying task and phase acceptance criteria."""

    def __init__(self, config: OrchestrationConfig, executor: BaseExecutor):
        self.config = config
        self.executor = executor
        self.project_root = config.project_root

    def verify_files(self, required_files: List[str]) -> VerificationResult:
        """Verify that all required files exist on disk and are non-empty."""
        missing = []
        for rf in required_files:
            full_path = self.project_root / rf if not Path(rf).is_absolute() else Path(rf)
            if not full_path.exists():
                missing.append(rf)
            elif full_path.stat().st_size == 0:
                missing.append(f"{rf} (empty file)")

        if missing:
            return VerificationResult(
                passed=False,
                reason=f"Missing or empty required files: {', '.join(missing)}",
                missing_files=missing,
            )
        return VerificationResult(passed=True, reason="All required files exist and are populated.", missing_files=[])

    def verify_command(self, test_command: str) -> VerificationResult:
        """Execute a verification command and ensure exit_code == 0."""
        res: ExecutionResult = self.executor.run_command(test_command)
        if not res.success:
            return VerificationResult(
                passed=False,
                reason=f"Verification command failed (exit code {res.exit_code}): {res.command}",
                missing_files=[],
                failed_command_output=f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}",
            )
        return VerificationResult(passed=True, reason=f"Verification command passed: {res.command}", missing_files=[])
