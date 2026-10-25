"""Pydantic schemas and models for CodeGhost Automated Fix Generation & Sandbox Verification (Milestone 6)."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class FixStrategy(str, Enum):
    """Strategies utilized to generate candidate bug fixes."""
    HISTORICAL_ANALOGY = "historical_analogy"
    DEFENSIVE_GUARD = "defensive_guard"
    EXCEPTION_SHIELD = "exception_shield"
    BOUNDS_CHECK = "bounds_check"
    LLM_SYNTHESIZED = "llm_synthesized"


class FixCandidate(BaseModel):
    """An individual candidate bug fix with code diff and explanation."""
    fix_id: str = Field(..., description="Unique deterministic identifier for the fix candidate")
    function_name: str = Field(..., description="Target function or method name")
    file_path: str = Field(..., description="Target source file path")
    strategy: FixStrategy = Field(..., description="Strategy utilized to generate this fix")
    description: str = Field(..., description="Concise explanation of what the fix alters")
    original_code: str = Field(..., description="Original vulnerable/risky function source code")
    fixed_code: str = Field(..., description="Proposed fixed function source code")
    diff: str = Field(..., description="Unified diff string representing the fix")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for this candidate fix")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context and metrics")

    model_config = ConfigDict(use_enum_values=True)


class SandboxExecutionResult(BaseModel):
    """Execution outcome from testing a candidate fix in an isolated sandbox."""
    success: bool = Field(..., description="True if sandbox tests executed and passed without regression")
    tests_run: int = Field(0, description="Total number of tests executed")
    tests_passed: int = Field(0, description="Number of passed tests")
    tests_failed: int = Field(0, description="Number of failed tests")
    tests_errored: int = Field(0, description="Number of test collection or execution errors")
    exit_code: int = Field(0, description="Subprocess exit code")
    stdout: str = Field("", description="Subprocess stdout log excerpt")
    stderr: str = Field("", description="Subprocess stderr log excerpt")
    duration_seconds: float = Field(0.0, description="Execution duration in seconds")
    error_message: Optional[str] = Field(None, description="Error message if sandbox setup or execution failed")

    model_config = ConfigDict(use_enum_values=True)


class VerifiedFix(BaseModel):
    """A verified fix candidate bundled with syntax and test execution validation."""
    candidate: FixCandidate
    syntax_valid: bool = Field(..., description="True if fixed code parses without AST syntax errors")
    sandbox_result: Optional[SandboxExecutionResult] = Field(None, description="Sandbox test execution outcome")
    is_verified: bool = Field(..., description="True if syntax is valid and sandbox tests passed")
    verification_summary: str = Field(..., description="Summary of verification status and outcomes")

    model_config = ConfigDict(use_enum_values=True)
