"""Modular Execution Interfaces for CodeGhost Orchestration.

Provides the verified execution adapter for running shell commands, executing tests,
editing project files, and inspecting Git state with least privilege and timeout safety.
"""

import os
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from backend.orchestration.config import OrchestrationConfig


@dataclass
class ExecutionResult:
    """Result of a command executed by an executor."""
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    success: bool = False

    def __post_init__(self):
        self.success = (self.exit_code == 0 and not self.timed_out)


@dataclass
class PatchResult:
    """Result of modifying or patching a file."""
    file_path: str
    success: bool
    diff: str = ""
    error_message: Optional[str] = None


class BaseExecutor(ABC):
    """Abstract interface defining required capabilities for any coding-task executor."""

    @abstractmethod
    def run_command(self, command: str, cwd: Optional[Path] = None, timeout: Optional[float] = None, env: Optional[Dict[str, str]] = None) -> ExecutionResult:
        """Execute a command string and return stdout/stderr/exit code."""
        pass

    @abstractmethod
    def apply_patch(self, file_path: str, target: str, replacement: str) -> PatchResult:
        """Replace target string with replacement in file."""
        pass

    @abstractmethod
    def write_file(self, file_path: str, content: str) -> bool:
        """Write text content to file."""
        pass

    @abstractmethod
    def read_file(self, file_path: str) -> str:
        """Read text content from file."""
        pass


class LocalDirectExecutor(BaseExecutor):
    """Verified local subprocess and file-system executor."""

    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self.project_root = config.project_root

    def run_command(self, command: str, cwd: Optional[Path] = None, timeout: Optional[float] = None, env: Optional[Dict[str, str]] = None) -> ExecutionResult:
        """Execute shell or python command via subprocess with timeout and output capture."""
        work_dir = str(cwd or self.project_root)
        to = timeout or self.config.per_command_timeout_seconds
        
        # Merge environment safely
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        
        # Replace .venv/Scripts/python with sys.executable for rock-solid portability
        normalized_cmd = command.replace(".venv\\Scripts\\python", f'"{sys.executable}"')
        normalized_cmd = normalized_cmd.replace(".venv/Scripts/python", f'"{sys.executable}"')

        start_time = time.time()
        try:
            # Execute command using system shell (cmd.exe on Windows) to avoid PowerShell quoting & AMSI issues
            proc = subprocess.run(
                normalized_cmd,
                shell=True,
                cwd=work_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=to,
                env=run_env,
            )
            duration = time.time() - start_time
            stdout_clean = self.config.redact(proc.stdout or "")
            stderr_clean = self.config.redact(proc.stderr or "")
            return ExecutionResult(
                command=self.config.redact(command),
                exit_code=proc.returncode,
                stdout=stdout_clean,
                stderr=stderr_clean,
                duration_seconds=duration,
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return ExecutionResult(
                command=self.config.redact(command),
                exit_code=-1,
                stdout="",
                stderr=f"Command timed out after {to} seconds.",
                duration_seconds=duration,
                timed_out=True,
            )
        except Exception as e:
            duration = time.time() - start_time
            return ExecutionResult(
                command=self.config.redact(command),
                exit_code=-1,
                stdout="",
                stderr=f"Subprocess invocation error: {self.config.redact(str(e))}",
                duration_seconds=duration,
                timed_out=False,
            )

    def write_file(self, file_path: str, content: str) -> bool:
        """Write content to file with directory creation and secret redaction."""
        full_path = self.project_root / file_path if not os.path.isabs(file_path) else Path(file_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure we never write the active API key to disk
        safe_content = self.config.redact(content)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(safe_content)
        return True

    def read_file(self, file_path: str) -> str:
        """Read content from file."""
        full_path = self.project_root / file_path if not os.path.isabs(file_path) else Path(file_path)
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def apply_patch(self, file_path: str, target: str, replacement: str) -> PatchResult:
        """Perform exact contiguous text substitution in file."""
        full_path = self.project_root / file_path if not os.path.isabs(file_path) else Path(file_path)
        if not full_path.exists():
            return PatchResult(file_path, success=False, error_message="File does not exist")

        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            original = f.read()

        if target not in original:
            return PatchResult(file_path, success=False, error_message="Target string not found in file")

        # Replace first occurrence
        modified = original.replace(target, replacement, 1)
        safe_modified = self.config.redact(modified)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(safe_modified)

        return PatchResult(file_path, success=True, diff=f"- {target[:60]}...\n+ {replacement[:60]}...")


class AntigravitySDKExecutor(BaseExecutor):
    """Adapter for future Google Antigravity Python SDK integration.
    
    Adheres strictly to project rules: reports honestly that SDK is not
    currently active/installed in this runtime rather than faking execution.
    """

    def __init__(self, config: OrchestrationConfig):
        self.config = config

    def _check_sdk(self):
        try:
            import google.antigravity  # type: ignore
            return True
        except ImportError:
            raise NotImplementedError(
                "Antigravity Python SDK ('google-antigravity') is not installed in the local virtual environment. "
                "Use LocalDirectExecutor for verified local execution."
            )

    def run_command(self, command: str, cwd: Optional[Path] = None, timeout: Optional[float] = None, env: Optional[Dict[str, str]] = None) -> ExecutionResult:
        self._check_sdk()
        raise NotImplementedError("Antigravity SDK execution adapter is a pending extension.")

    def apply_patch(self, file_path: str, target: str, replacement: str) -> PatchResult:
        self._check_sdk()
        raise NotImplementedError("Antigravity SDK execution adapter is a pending extension.")

    def write_file(self, file_path: str, content: str) -> bool:
        self._check_sdk()
        raise NotImplementedError("Antigravity SDK execution adapter is a pending extension.")

    def read_file(self, file_path: str) -> str:
        self._check_sdk()
        raise NotImplementedError("Antigravity SDK execution adapter is a pending extension.")
