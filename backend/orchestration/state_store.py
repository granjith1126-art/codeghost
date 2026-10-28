"""Persistent Workflow State Store and Checkpoint Manager.

Serializes workflow execution state to disk after every phase transition,
enabling safe resume, auditability, and progress tracking.
"""

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TaskRecord:
    task_id: str
    description: str
    status: str = "PENDING"  # PENDING, IN_PROGRESS, PASSED, FAILED, SKIPPED
    attempts: int = 0
    commands_run: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    verification_output: Optional[str] = None


@dataclass
class PhaseRecord:
    phase_id: str
    title: str
    status: str = "PENDING"  # PENDING, IN_PROGRESS, PASSED, PASSED_WITH_LIMITATIONS, BLOCKED, FAILED
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: float = 0.0
    tasks: List[TaskRecord] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    blockers: List[str] = field(default_factory=list)
    review_verdict: Optional[str] = None
    review_findings: List[str] = field(default_factory=list)
    review_attempts: int = 0


@dataclass
class WorkflowState:
    milestone: str = "Milestone 8"
    active_phase: str = "phase_1"
    status: str = "IN_PROGRESS"  # IN_PROGRESS, COMPLETED, BLOCKED, FAILED
    started_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    updated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    phases: Dict[str, PhaseRecord] = field(default_factory=dict)
    
    # Telemetry and budgets
    api_requests_count: int = 0
    prompt_tokens_used: int = 0
    completion_tokens_used: int = 0
    estimated_cost_usd: float = 0.0
    git_commit_sha: Optional[str] = None


class StateStore:
    """Manages persistent JSON workflow checkpoints with atomic file writes."""

    def __init__(self, state_file_path: Path):
        self.state_file_path = Path(state_file_path)
        self.state_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.state: WorkflowState = self._load_or_create()

    def _load_or_create(self) -> WorkflowState:
        if self.state_file_path.exists():
            try:
                with open(self.state_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Reconstruct phases
                phases = {}
                valid_keys = {f.name for f in fields(PhaseRecord)}
                for pid, pdata in data.get("phases", {}).items():
                    tasks = [TaskRecord(**t) for t in pdata.get("tasks", [])]
                    pdata_copy = dict(pdata)
                    pdata_copy["tasks"] = tasks
                    filtered_pdata = {k: v for k, v in pdata_copy.items() if k in valid_keys}
                    phases[pid] = PhaseRecord(**filtered_pdata)
                
                data["phases"] = phases
                return WorkflowState(**data)
            except Exception:
                pass
        
        # Initialize default phases
        if "m9" in str(self.state_file_path).lower():
            default_phases = {
                "phase_9_1": PhaseRecord("phase_9_1", "AST Semantic Diff Analyzer", artifacts=["backend/history/semantic_diff.py", "tests/test_semantic_diff.py"]),
                "phase_9_2": PhaseRecord("phase_9_2", "Threshold Calibration & Optimization", artifacts=["backend/risk/evaluator.py", "tests/test_risk_calibration.py"]),
                "phase_9_3": PhaseRecord("phase_9_3", "History Extractor Integration", artifacts=["backend/history/extractor.py", "backend/main.py"]),
                "phase_9_4": PhaseRecord("phase_9_4", "Milestone 9 Validation & Final Report", artifacts=["reports/validation/milestone_9_report.md", "README.md"]),
            }
            return WorkflowState(milestone="Milestone 9", active_phase="phase_9_1", phases=default_phases)

        # Initialize default Milestone 8 phases (1 through 10)
        default_phases = {
            "phase_1": PhaseRecord("phase_1", "Repository Baseline & Audit", status="PASSED", review_verdict="APPROVED", artifacts=["reports/validation/initial_audit.md"]),
            "phase_2": PhaseRecord("phase_2", "Evaluation Protocol Definition"),
            "phase_3": PhaseRecord("phase_3", "Data Leakage & Dataset Audit"),
            "phase_4": PhaseRecord("phase_4", "End-to-End Integration Test"),
            "phase_5": PhaseRecord("phase_5", "Risk Scoring Benchmark"),
            "phase_6": PhaseRecord("phase_6", "Retrieval & Explainability Benchmark"),
            "phase_7": PhaseRecord("phase_7", "Automated Fix & PR Dispatch Benchmark"),
            "phase_8": PhaseRecord("phase_8", "Integration Defect Fixes"),
            "phase_9": PhaseRecord("phase_9", "Reproducible Benchmark Runner"),
            "phase_10": PhaseRecord("phase_10", "Final Validation Report"),
        }
        return WorkflowState(phases=default_phases)

    def save(self) -> None:
        """Atomic write to disk to prevent corrupt states."""
        self.state.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        data = asdict(self.state)
        
        dir_name = self.state_file_path.parent
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
            json.dump(data, tf, indent=2)
            temp_name = tf.name
        
        os.replace(temp_name, self.state_file_path)

    def record_usage(self, prompt_tokens: int, completion_tokens: int, cost_estimate: float = 0.0) -> None:
        """Track API usage statistics."""
        self.state.api_requests_count += 1
        self.state.prompt_tokens_used += prompt_tokens
        self.state.completion_tokens_used += completion_tokens
        self.state.estimated_cost_usd += cost_estimate
        self.save()

    def get_phase(self, phase_id: str) -> PhaseRecord:
        """Retrieve phase record."""
        return self.state.phases[phase_id]

    def update_phase(
        self,
        phase_id: str,
        status: str,
        artifacts: Optional[List[str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        blockers: Optional[List[str]] = None,
        review_verdict: Optional[str] = None,
        review_findings: Optional[List[str]] = None,
        review_attempts: Optional[int] = None,
    ) -> None:
        """Update phase state and persist."""
        phase = self.state.phases[phase_id]
        phase.status = status
        if artifacts:
            phase.artifacts.extend([a for a in artifacts if a not in phase.artifacts])
        if metrics:
            phase.metrics.update(metrics)
        if blockers:
            phase.blockers.extend([b for b in blockers if b not in phase.blockers])
        if review_verdict is not None:
            phase.review_verdict = review_verdict
        if review_findings is not None:
            phase.review_findings = review_findings
        if review_attempts is not None:
            phase.review_attempts = review_attempts
        self.save()

    def get_next_incomplete_phase(self) -> Optional[str]:
        """Find the earliest phase that is not fully PASSED with an APPROVED review."""
        for pid, p in self.state.phases.items():
            if p.status not in ("PASSED", "PASSED_WITH_LIMITATIONS"):
                return pid
            if getattr(p, "review_verdict", None) != "APPROVED":
                return pid
        return None
