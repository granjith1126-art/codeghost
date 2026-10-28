"""Pydantic schemas and data models for CodeGhost."""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field


class CodeUnitType(str, Enum):
    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    CLASS = "class"
    METHOD = "method"


class CodeUnit(BaseModel):
    """Represents an extracted function or class structured unit."""
    repository: str = Field(..., description="Repository URL or identifier")
    commit_hash: str = Field(..., description="Commit SHA or version identifier")
    file_path: str = Field(..., description="Relative file path from repository root")
    unit_type: CodeUnitType = Field(..., description="Type of code unit: function, method, class, or async_function")
    name: str = Field(..., description="Name of the function or class (or ClassName.method_name)")
    start_line: int = Field(..., description="1-indexed starting line number in source file")
    end_line: int = Field(..., description="1-indexed ending line number in source file")
    source_code: str = Field(..., description="Exact raw source code of the unit")
    language: str = Field(..., description="Source code language (e.g. python, javascript)")
    parent_name: Optional[str] = Field(None, description="Enclosing class or parent scope name if applicable")
    parameters: Optional[List[str]] = Field(default_factory=list, description="List of parameter names for functions/methods")
    docstring: Optional[str] = Field(None, description="Extracted docstring or leading documentation comment")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional extensible metadata for future milestones")

    model_config = ConfigDict(use_enum_values=True)


class LanguageStat(BaseModel):
    """Statistics for a single programming language detected in repository."""
    language: str
    file_count: int
    total_bytes: int
    percentage: float


class IngestionError(BaseModel):
    """Details of an error encountered during ingestion/parsing."""
    file_path: str
    error_type: str
    message: str


class IngestionResult(BaseModel):
    """Comprehensive summary of a repository ingestion run."""
    repository_url: str
    commit_hash: str
    local_path: str
    files_discovered: int
    files_parsed: int
    unsupported_files: int
    functions_extracted: int
    classes_extracted: int
    errors: List[IngestionError] = Field(default_factory=list)
    languages_detected: Dict[str, int] = Field(default_factory=dict)
    output_file: Optional[str] = None


# ==========================================
# Milestone 2: Historical Bug Extraction Schemas
# ==========================================

class BugFixClassification(str, Enum):
    """Heuristic classification categories for commit messages."""
    BUG_FIX_CANDIDATE = "BUG_FIX_CANDIDATE"
    UNCERTAIN = "UNCERTAIN"
    NON_BUG_FIX = "NON_BUG_FIX"


class BugDetectionEvidence(BaseModel):
    """Evidence and scores produced by the heuristic bug detector."""
    score: float = Field(..., description="Heuristic score between 0.0 and 1.0")
    classification: BugFixClassification = Field(..., description="Candidate classification")
    matched_signals: List[str] = Field(default_factory=list, description="List of matched keywords/regex patterns")
    explanation: str = Field(..., description="Human-readable breakdown of the score calculation")

    model_config = ConfigDict(use_enum_values=True)


class CommitFileDiff(BaseModel):
    """Diff details for an individual file changed in a commit."""
    file_path: str = Field(..., description="Current relative path of the file")
    old_file_path: Optional[str] = Field(None, description="Original path if renamed")
    change_type: str = Field("M", description="Git change type: A (added), M (modified), D (deleted), R (renamed)")
    additions: int = Field(0, description="Lines added")
    deletions: int = Field(0, description="Lines deleted")
    diff_text: str = Field("", description="Unified diff text for this file")
    old_line_ranges: List[Tuple[int, int]] = Field(default_factory=list, description="Ranges (start, end) changed in parent file")
    new_line_ranges: List[Tuple[int, int]] = Field(default_factory=list, description="Ranges (start, end) changed in commit file")
    is_binary: bool = Field(False, description="Whether the file is binary")


class CommitRecord(BaseModel):
    """Raw record of an extracted commit with bug detection evidence."""
    repository: str
    commit_sha: str
    parent_shas: List[str] = Field(default_factory=list)
    author_name: str
    author_email: str
    timestamp: str = Field(..., description="ISO 8601 formatted timestamp")
    commit_message: str
    is_merge: bool = False
    changed_files: List[str] = Field(default_factory=list)
    file_diffs: List[CommitFileDiff] = Field(default_factory=list)
    total_additions: int = 0
    total_deletions: int = 0
    detection_result: Optional[BugDetectionEvidence] = None


class BugFixPair(BaseModel):
    """A complete extracted historical bug-fix example (BUGGY FUNCTION -> FIXED FUNCTION)."""
    repository: str = Field(..., description="Repository URL or identifier")
    commit_sha: str = Field(..., description="Fix commit SHA")
    parent_sha: str = Field(..., description="Buggy parent commit SHA")
    commit_message: str = Field(..., description="Original commit message")
    author: str = Field(..., description="Commit author")
    timestamp: str = Field(..., description="ISO 8601 commit timestamp")
    file_path: str = Field(..., description="Relative file path")
    language: str = Field(..., description="Programming language")
    code_unit_type: str = Field(..., description="function, method, or class")
    class_name: Optional[str] = Field(None, description="Parent class name if method")
    function_name: str = Field(..., description="Qualified or simple function/method name")
    buggy_code: str = Field(..., description="Complete source code of the unit in the parent commit")
    fixed_code: str = Field(..., description="Complete source code of the unit in the fix commit")
    diff: str = Field(..., description="Unified diff of the code unit itself")
    additions: int = Field(0, description="Lines added within the code unit")
    deletions: int = Field(0, description="Lines deleted within the code unit")
    detector_score: float = Field(..., description="Heuristic bug detector confidence score")
    detector_signals: List[str] = Field(default_factory=list, description="Signals matched in the commit message")
    detector_classification: str = Field(..., description="Heuristic classification")
    semantic_classification: Optional[str] = Field(None, description="AST semantic classification (SEMANTIC_CODE_CHANGE, COMMENT_ONLY, etc.)")
    extraction_status: str = Field("success", description="Extraction status (success, partial, unchanged)")
    error_message: Optional[str] = Field(None, description="Error message if extraction had warnings")

    model_config = ConfigDict(use_enum_values=True)


class HistoryExtractionSummary(BaseModel):
    """Summary metrics of a historical bug extraction run."""
    repository_url: str
    commits_scanned: int = 0
    bug_fix_candidates: int = 0
    uncertain_commits: int = 0
    non_bug_commits: int = 0
    files_changed: int = 0
    functions_changed: int = 0
    bug_fix_pairs_extracted: int = 0
    non_semantic_pairs_filtered: int = 0
    extraction_failures: int = 0
    raw_commits_file: Optional[str] = None
    processed_pairs_file: Optional[str] = None
