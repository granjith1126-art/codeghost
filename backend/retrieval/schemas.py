"""Pydantic schemas for CodeGhost Function Embeddings & Vector Retrieval (Milestone 3)."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class RepresentationType(str, Enum):
    """Strategies for constructing the text to embed."""
    CODE_ONLY = "code_only"
    CODE_AND_METADATA = "code_and_metadata"
    CODE_AND_CONTEXT = "code_and_context"


class RetrievalQuery(BaseModel):
    """Query input for finding historically similar bug-fix examples."""
    code: str = Field(..., description="Current function source code")
    repository: Optional[str] = Field(None, description="Repository identifier to enforce repository isolation")
    language: Optional[str] = Field(None, description="Programming language")
    file_path: Optional[str] = Field(None, description="Relative file path")
    function_name: Optional[str] = Field(None, description="Function or method name")
    class_name: Optional[str] = Field(None, description="Enclosing class name")
    timestamp: Optional[str] = Field(None, description="Commit timestamp for temporal cutoff")
    exclude_commit_sha: Optional[str] = Field(None, description="Commit SHA to exclude from results to prevent leakage")
    representation_type: RepresentationType = Field(
        default=RepresentationType.CODE_AND_METADATA,
        description="Representation strategy to use for embedding the query",
    )

    model_config = ConfigDict(use_enum_values=True)


class RetrievedMatch(BaseModel):
    """A retrieved historical bug-fix example with similarity score and context."""
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    repository: str
    commit_sha: str
    parent_sha: str
    file_path: str
    function_name: str
    class_name: Optional[str] = None
    commit_message: str
    buggy_code: str
    fixed_code: str
    diff: str
    detector_score: float
    detector_classification: str
    timestamp: str
    matched_text: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IndexingStats(BaseModel):
    """Summary metrics of a vector indexing run."""
    dataset_records: int = 0
    valid_records: int = 0
    embedded: int = 0
    indexed: int = 0
    failed: int = 0
    vector_dimension: int = 0
    collection_name: str = ""
    elapsed_seconds: float = 0.0


class EvaluationResult(BaseModel):
    """Retrieval evaluation metrics comparing algorithms or representations."""
    method_name: str
    representation: str
    total_queries: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    avg_latency_ms: float = 0.0
