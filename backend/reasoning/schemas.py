"""Pydantic schemas and data models for CodeGhost Reasoning & Explainability (Milestone 5)."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from backend.risk.schemas import RiskCategory


class HistoricalPrecedent(BaseModel):
    """Contextual evidence linking current code to a historical bug-fix."""
    commit_sha: str = Field(..., description="Commit SHA of historical fix")
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    commit_message: str = Field(..., description="Historical bug-fix commit message")
    file_path: str = Field(..., description="File path of historical code")
    function_name: str = Field(..., description="Historical function or method name")
    buggy_snippet: Optional[str] = Field(None, description="Excerpt of historical buggy code")
    fixed_snippet: Optional[str] = Field(None, description="Excerpt of historical fixed code")
    diff_summary: Optional[str] = Field(None, description="Summary of changes applied in the historical fix")
    relevance_reason: str = Field(..., description="Why this historical fix is relevant to the current function")

    model_config = ConfigDict(use_enum_values=True)


class ComplexityFactor(BaseModel):
    """AST or structural complexity metric contributing to failure risk."""
    name: str = Field(..., description="Factor name (e.g. Cyclomatic Complexity, Max Nesting Depth)")
    value: Any = Field(..., description="Observed numeric or categorical value")
    risk_impact: str = Field(..., description="'HIGH', 'MEDIUM', or 'LOW'")
    description: str = Field(..., description="Human explanation of why this factor increases risk")

    model_config = ConfigDict(use_enum_values=True)


class RiskExplanation(BaseModel):
    """Comprehensive, human-understandable explanation of a function's risk."""
    function_name: str = Field(..., description="Target function or method evaluated")
    file_path: str = Field(..., description="Target file path")
    risk_score: float = Field(..., ge=1.0, le=10.0, description="Calibrated risk score (1.0 - 10.0)")
    risk_category: RiskCategory = Field(..., description="Risk category: LOW, MEDIUM, HIGH, CRITICAL")
    is_risky: bool = Field(..., description="Binary risk decision")
    summary: str = Field(..., description="Executive summary of the risk assessment")
    detailed_analysis: str = Field(..., description="In-depth analysis of code logic and failure vectors")
    complexity_factors: List[ComplexityFactor] = Field(default_factory=list, description="Structural risk drivers")
    historical_precedents: List[HistoricalPrecedent] = Field(default_factory=list, description="Similar historical bug fixes")
    vulnerable_patterns: List[str] = Field(default_factory=list, description="Specific identified anti-patterns or hazards")
    mitigation_strategies: List[str] = Field(default_factory=list, description="Prescribed concrete mitigation steps")
    provider: str = Field("deterministic", description="Explanation engine provider (e.g. 'deterministic', 'gemini', 'openai')")
    model_name: str = Field(..., description="Name of underlying reasoning model or engine")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context and metrics")

    model_config = ConfigDict(use_enum_values=True)
