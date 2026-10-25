"""Pydantic schemas and data models for CodeGhost Risk Scoring (Milestone 4)."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from backend.retrieval.schemas import RetrievedMatch


class RiskCategory(str, Enum):
    """Calibrated categorical risk levels."""
    LOW = "LOW"            # 1.0 - 3.9
    MEDIUM = "MEDIUM"      # 4.0 - 6.9
    HIGH = "HIGH"          # 7.0 - 8.9
    CRITICAL = "CRITICAL"  # 9.0 - 10.0


class RiskThresholds(BaseModel):
    """Configurable numerical score thresholds for categorical risk tiers."""
    low_max: float = 3.9
    medium_max: float = 6.9
    high_max: float = 8.9
    critical_max: float = 10.0

    def categorize(self, score: float) -> RiskCategory:
        """Categorize a numerical risk score (1.0 to 10.0) into a RiskCategory."""
        clamped_score = max(1.0, min(10.0, score))
        if clamped_score <= self.low_max:
            return RiskCategory.LOW
        elif clamped_score <= self.medium_max:
            return RiskCategory.MEDIUM
        elif clamped_score <= self.high_max:
            return RiskCategory.HIGH
        else:
            return RiskCategory.CRITICAL


class RiskExample(BaseModel):
    """A labeled training or evaluation example for risk scoring."""
    example_id: str = Field(..., description="Unique deterministic identifier")
    repository: str = Field(..., description="Repository URL or identifier")
    commit_sha: str = Field(..., description="Commit SHA associated with this code state")
    parent_sha: Optional[str] = Field(None, description="Parent commit SHA if known")
    function_name: str = Field(..., description="Qualified or simple function/method name")
    file_path: str = Field(..., description="Relative file path")
    language: str = Field(..., description="Programming language (e.g. python)")
    code: str = Field(..., description="Function source code to be evaluated for risk")
    timestamp: str = Field(..., description="ISO 8601 commit timestamp")
    timestamp_epoch: float = Field(..., description="Epoch seconds for exact chronological ordering")
    label: int = Field(..., description="1 = POSITIVE (buggy historical state), 0 = NEGATIVE (stable non-buggy state)")
    label_rationale: str = Field(..., description="Explanation of how the label was determined")
    retrieved_matches: List[RetrievedMatch] = Field(default_factory=list, description="Historical bug examples retrieved from vector DB")
    similarity_scores: List[float] = Field(default_factory=list, description="Extracted similarity scores from retrieved matches")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context (signals, diff stats, AST metrics)")

    model_config = ConfigDict(use_enum_values=True)


class RiskPrediction(BaseModel):
    """Predicted risk outcome for a function."""
    function_name: str
    file_path: str
    risk_probability: float = Field(..., ge=0.0, le=1.0, description="Calibrated risk probability (0.0 to 1.0)")
    risk_score: float = Field(..., ge=1.0, le=10.0, description="Mapped risk score from 1.0 to 10.0")
    risk_category: RiskCategory
    is_risky: bool = Field(..., description="Binary classification (True if predicted risky)")
    model_name: str = Field(..., description="Model identifier (e.g. 'LogisticRegression', 'Qwen2.5-Coder-7B')")
    features_used: Optional[Dict[str, float]] = Field(default=None, description="Feature values used by tabular models")
    top_retrieved_matches: List[RetrievedMatch] = Field(default_factory=list, description="Top retrieved historical matches used as evidence")
    explanation_context: Dict[str, Any] = Field(default_factory=dict, description="Preserved evidence for Milestone 5 explainability")
    reasoning_summary: Optional[str] = Field(None, description="Optional brief model reasoning")

    model_config = ConfigDict(use_enum_values=True)


class RiskEvaluationMetrics(BaseModel):
    """Comprehensive evaluation metrics for risk classification models."""
    model_name: str
    split_name: str = Field(..., description="Data split evaluated: 'test', 'validation', or 'train'")
    total_samples: int
    positive_samples: int
    negative_samples: int
    class_balance_ratio: float = Field(..., description="Ratio of positive samples: positive / total")
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    false_positive_rate: float = 0.0
    confusion_matrix: List[List[int]] = Field(default_factory=list, description="[[TN, FP], [FN, TP]]")
    score_distribution: Dict[str, int] = Field(default_factory=dict, description="Count per RiskCategory")
    threshold: float = 0.5
    latency_ms: float = 0.0

    model_config = ConfigDict(use_enum_values=True)


class DatasetSplits(BaseModel):
    """Chronologically separated dataset splits."""
    train: List[RiskExample] = Field(default_factory=list)
    val: List[RiskExample] = Field(default_factory=list)
    test: List[RiskExample] = Field(default_factory=list)
    train_cutoff_timestamp: Optional[str] = None
    val_cutoff_timestamp: Optional[str] = None
