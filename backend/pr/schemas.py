"""Pydantic schemas and models for CodeGhost GitHub PR Creation (Milestone 7)."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class PRDraft(BaseModel):
    """Complete Pull Request payload including title, branch, markdown body, and diff."""
    title: str = Field(..., description="Pull request title (e.g. 'fix(app): harden Flask.run host parsing against IPv6 regressions')")
    branch_name: str = Field(..., description="Git branch name (e.g. 'codeghost/fix-flask-run-a1b2c3d4')")
    commit_message: str = Field(..., description="Git commit message adhering to Conventional Commits")
    body_markdown: str = Field(..., description="Rich Markdown description with risk analysis, historical evidence, diff, and review checklist")
    diff: str = Field(..., description="Unified diff of the proposed fix")
    target_repo: str = Field(..., description="Target repository URL or owner/name")
    base_branch: str = Field("main", description="Target base branch to merge into")
    risk_score: float = Field(..., description="Calibrated risk score of the original function")
    risk_category: str = Field(..., description="Categorical risk tier: LOW, MEDIUM, HIGH, CRITICAL")
    is_verified: bool = Field(..., description="True if verified in isolated test sandbox")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata and telemetry")

    model_config = ConfigDict(use_enum_values=True)


class PRResult(BaseModel):
    """Result of Pull Request creation or dry-run assembly."""
    success: bool = Field(..., description="True if PR draft or remote PR creation succeeded")
    pr_url: Optional[str] = Field(None, description="URL of created GitHub Pull Request if live")
    pr_number: Optional[int] = Field(None, description="Pull Request issue number on GitHub")
    branch_created: bool = Field(False, description="True if git branch was created locally or remotely")
    patch_file_path: Optional[str] = Field(None, description="Local path to generated .patch file")
    dry_run: bool = Field(True, description="True if executed in dry-run/preview mode")
    message: str = Field(..., description="Summary status message")
    draft: Optional[PRDraft] = Field(None, description="The PRDraft object assembled")

    model_config = ConfigDict(use_enum_values=True)
