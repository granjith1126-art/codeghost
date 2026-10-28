"""OpenAI-Powered Task Planner and Diagnostic Engine for CodeGhost.

Constructs structured task plans, diagnoses test failures, and proposes bounded
code repairs using OpenAI's API with deterministic fallback support.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import openai

from backend.orchestration.config import OrchestrationConfig
from backend.orchestration.state_store import StateStore


@dataclass
class PlannedTask:
    task_id: str
    description: str
    command: Optional[str] = None
    target_files: List[str] = field(default_factory=list)
    acceptance_criteria: str = ""
    is_critical: bool = True


@dataclass
class ReviewFinding:
    severity: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"
    category: str  # "CORRECTNESS", "LEAKAGE", "PERFORMANCE", "SECURITY", "DOCUMENTATION"
    description: str
    evidence: str = ""
    suggested_fix: str = ""


@dataclass
class PhaseReviewVerdict:
    verdict: str  # "APPROVED" or "CHANGES_REQUESTED"
    score: float = 0.0
    acceptance_criteria_assessment: Dict[str, str] = field(default_factory=dict)
    findings: List[ReviewFinding] = field(default_factory=list)
    required_fixes: List[str] = field(default_factory=list)
    remaining_risks: List[str] = field(default_factory=list)
    raw_response: Optional[str] = None
    is_valid: bool = True
    reviewer_type: str = "openai"
    error_message: Optional[str] = None


@dataclass
class RepairProposal:
    root_cause: str
    proposed_files: List[str] = field(default_factory=list)
    patch_instructions: Dict[str, str] = field(default_factory=dict)  # file_path -> instructions or code
    suggested_commands: List[str] = field(default_factory=list)
    replacements: List[Dict[str, str]] = field(default_factory=list)  # list of {"file_path": ..., "target": ..., "replacement": ...}


class OpenAIPlanner:
    """Manages OpenAI model interactions for planning and diagnosis."""

    def __init__(self, config: OrchestrationConfig, state_store: Optional[StateStore] = None):
        self.config = config
        self.state_store = state_store
        self.client = openai.OpenAI(api_key=config.openai_api_key) if config.has_api_key else None

    def plan_phase(self, phase_id: str, phase_title: str, context: Optional[str] = None) -> List[PlannedTask]:
        """Generate structured task list for a phase."""
        if not self.client or not self.config.has_api_key:
            return self._get_deterministic_phase_plan(phase_id)

        prompt = f"""You are the lead architect for CodeGhost. Formulate a concise, concrete task plan for:
Phase: {phase_id} - {phase_title}
Context: {context or 'Execute Phase according to CodeGhost Milestone 8 specifications.'}

Return ONLY valid JSON with this exact schema:
{{
  "tasks": [
    {{
      "task_id": "string",
      "description": "string",
      "command": "string or null",
      "target_files": ["string"],
      "acceptance_criteria": "string",
      "is_critical": true
    }}
  ]
}}"""

        try:
            resp = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "You are CodeGhost's lead autonomous validation planner. Output strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            # Track usage
            if self.state_store and resp.usage:
                self.state_store.record_usage(
                    prompt_tokens=resp.usage.prompt_tokens,
                    completion_tokens=resp.usage.completion_tokens,
                    cost_estimate=(resp.usage.prompt_tokens * 0.000005) + (resp.usage.completion_tokens * 0.000015),
                )

            data = json.loads(resp.choices[0].message.content or "{}")
            tasks = []
            for t in data.get("tasks", []):
                tasks.append(PlannedTask(**t))
            return tasks or self._get_deterministic_phase_plan(phase_id)
        except Exception:
            return self._get_deterministic_phase_plan(phase_id)

    def review_phase(
        self,
        phase_id: str,
        phase_title: str,
        acceptance_criteria: str,
        changed_files: List[str],
        diff_or_content_summary: str,
        test_results: str,
        logs: str = "",
    ) -> PhaseReviewVerdict:
        """Perform genuine, independent code review using the OpenAI API.
        
        Evaluates artifacts, test results, logs, and acceptance criteria.
        Returns a structured PhaseReviewVerdict (APPROVED or CHANGES_REQUESTED).
        Never treats API failures or malformed responses as approval.
        """
        if not self.client or not self.config.has_api_key:
            return PhaseReviewVerdict(
                verdict="CHANGES_REQUESTED",
                score=0.0,
                is_valid=False,
                error_message="OpenAI API client is not configured or API key is missing. Genuine review requires active API configuration.",
            )

        prompt = f"""You are CodeGhost's lead independent validation reviewer.
Your responsibility is to strictly audit phase deliverables, code changes, test results, and compliance with specifications.

Phase: {phase_id} - {phase_title}
Acceptance Criteria:
{acceptance_criteria}

Changed Files:
{json.dumps(changed_files)}

Deliverables / Code / Diff Summary:
{self.config.redact(diff_or_content_summary[:8000])}

Test Execution Results:
{self.config.redact(test_results[:4000])}

Execution Logs:
{self.config.redact(logs[:2000])}

Audit Instructions:
1. Verify each acceptance criterion.
2. Ensure tests executed and passed without syntax or runtime failures.
3. Check for data leakage, unvalidated assumptions, or regressions.
4. If ALL criteria are satisfied and tests pass with no critical defects, verdict MUST be 'APPROVED'.
5. If any test fails, critical requirement is missing, or severe risk exists, verdict MUST be 'CHANGES_REQUESTED'.

Return ONLY valid JSON matching this schema:
{{
  "verdict": "APPROVED" | "CHANGES_REQUESTED",
  "score": 8.5,
  "acceptance_criteria_assessment": {{
    "criterion_name": "MET"
  }},
  "findings": [
    {{
      "severity": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "INFO",
      "category": "CORRECTNESS" | "LEAKAGE" | "PERFORMANCE" | "SECURITY" | "DOCUMENTATION",
      "description": "string",
      "evidence": "string",
      "suggested_fix": "string"
    }}
  ],
  "required_fixes": ["string"],
  "remaining_risks": ["string"]
}}"""

        try:
            resp = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "You are CodeGhost's autonomous independent validation reviewer. Be rigorous, objective, and return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            if self.state_store and resp.usage:
                self.state_store.record_usage(
                    prompt_tokens=resp.usage.prompt_tokens,
                    completion_tokens=resp.usage.completion_tokens,
                    cost_estimate=(resp.usage.prompt_tokens * 0.000005) + (resp.usage.completion_tokens * 0.000015),
                )

            raw_text = resp.choices[0].message.content or "{}"
            data = json.loads(raw_text)

            # Strict validation of parsed response
            verdict_raw = str(data.get("verdict", "")).strip().upper()
            if verdict_raw not in ("APPROVED", "CHANGES_REQUESTED"):
                return PhaseReviewVerdict(
                    verdict="CHANGES_REQUESTED",
                    score=float(data.get("score", 0.0)),
                    is_valid=False,
                    error_message=f"Invalid review verdict received: '{verdict_raw}'. Must be APPROVED or CHANGES_REQUESTED.",
                    raw_response=raw_text,
                )

            findings = []
            for f in data.get("findings", []):
                findings.append(
                    ReviewFinding(
                        severity=str(f.get("severity", "MEDIUM")).upper(),
                        category=str(f.get("category", "CORRECTNESS")).upper(),
                        description=f.get("description", ""),
                        evidence=f.get("evidence", ""),
                        suggested_fix=f.get("suggested_fix", ""),
                    )
                )

            # Safety gate: If any finding is CRITICAL or HIGH, verdict cannot be APPROVED
            has_critical = any(f.severity in ("CRITICAL", "HIGH") for f in findings)
            final_verdict = "CHANGES_REQUESTED" if has_critical else verdict_raw

            return PhaseReviewVerdict(
                verdict=final_verdict,
                score=float(data.get("score", 7.0 if final_verdict == "APPROVED" else 4.0)),
                acceptance_criteria_assessment=data.get("acceptance_criteria_assessment", {}),
                findings=findings,
                required_fixes=data.get("required_fixes", []),
                remaining_risks=data.get("remaining_risks", []),
                raw_response=raw_text,
                is_valid=True,
            )
        except Exception as e:
            return PhaseReviewVerdict(
                verdict="CHANGES_REQUESTED",
                score=0.0,
                is_valid=False,
                error_message=f"Review API exception: {self.config.redact(str(e))}",
            )

    def diagnose_and_repair(
        self,
        task_description: str,
        command: str = "",
        error_output: str = "",
        findings: Optional[List[str]] = None,
        target_files: Optional[List[str]] = None,
    ) -> RepairProposal:
        """Diagnose a command failure or review rejection and propose a concrete repair action."""
        if not self.client or not self.config.has_api_key:
            return RepairProposal(
                root_cause="Automated offline diagnosis: inspect error output and rerun test.",
                suggested_commands=[command] if command else [],
                proposed_files=target_files or [],
            )

        context_findings = "\n".join(f"- {f}" for f in (findings or []))
        prompt = f"""Task / Phase Action '{task_description}' requires repair.
Command: {self.config.redact(command)}
Error / Review Output:
{self.config.redact(error_output[:2500])}

Specific Findings to Address:
{self.config.redact(context_findings or 'None provided')}

Target Files:
{json.dumps(target_files or [])}

Analyze the issues, determine root cause, and propose safe, targeted text replacements and commands to verify the fix.
Return ONLY valid JSON with this schema:
{{
  "root_cause": "concise explanation",
  "proposed_files": ["files to update"],
  "replacements": [
    {{
      "file_path": "path/to/file",
      "target": "exact string to replace",
      "replacement": "replacement text"
    }}
  ],
  "suggested_commands": ["commands to run to verify fix"]
}}"""

        try:
            resp = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "You are CodeGhost's debugging AI. Output strict JSON with safe, contiguous text replacements."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            if self.state_store and resp.usage:
                self.state_store.record_usage(resp.usage.prompt_tokens, resp.usage.completion_tokens)

            data = json.loads(resp.choices[0].message.content or "{}")
            return RepairProposal(
                root_cause=data.get("root_cause", "Unspecified error"),
                proposed_files=data.get("proposed_files", target_files or []),
                suggested_commands=data.get("suggested_commands", [command] if command else []),
                replacements=data.get("replacements", []),
            )
        except Exception as e:
            return RepairProposal(
                root_cause=f"Diagnosis API error: {self.config.redact(str(e))}",
                suggested_commands=[command] if command else [],
                proposed_files=target_files or [],
            )

    def _get_deterministic_phase_plan(self, phase_id: str) -> List[PlannedTask]:
        """Pre-verified deterministic fallback plans for Milestone 8 phases."""
        plans = {
            "phase_2": [
                PlannedTask(
                    task_id="p2_write_protocol",
                    description="Write formal evaluation protocol for risk, retrieval, fixes, and pipeline",
                    target_files=["reports/validation/phase_2_evaluation_protocol.md"],
                    acceptance_criteria="File reports/validation/phase_2_evaluation_protocol.md must exist and define required metrics",
                ),
            ],
            "phase_3": [
                PlannedTask(
                    task_id="p3_leakage_audit",
                    description="Audit historical data splits for temporal leakage and investigate 100% recall root cause",
                    target_files=["reports/validation/phase_3_data_audit.md"],
                    acceptance_criteria="Audit document reports/validation/phase_3_data_audit.md must exist with empirical leakage findings",
                ),
            ],
            "phase_4": [
                PlannedTask(
                    task_id="p4_create_e2e_integration_test",
                    description="Build end-to-end integration test with deterministic Git repository fixture",
                    command=".venv\\Scripts\\python -m pytest tests/integration/test_end_to_end.py -v",
                    target_files=["tests/integration/test_end_to_end.py", "reports/validation/phase_4_e2e.md"],
                    acceptance_criteria="Integration tests pass and reports/validation/phase_4_e2e.md is generated",
                ),
            ],
            "phase_5": [
                PlannedTask(
                    task_id="p5_risk_benchmark",
                    description="Evaluate pure NumPy Random Forest baseline on untouched test split and record metrics",
                    command=".venv\\Scripts\\python -m backend.main --mode risk --model baseline --baseline-type random_forest",
                    target_files=["reports/validation/phase_5_risk_benchmark.md", "reports/validation/phase_5_risk_metrics.json"],
                    acceptance_criteria="Benchmark metrics JSON and Markdown report generated",
                ),
            ],
            "phase_6": [
                PlannedTask(
                    task_id="p6_retrieval_benchmark",
                    description="Benchmark BGE vector retrieval vs BM25 baseline and audit explainability fidelity",
                    target_files=["reports/validation/phase_6_retrieval_explainability.md"],
                    acceptance_criteria="Report reports/validation/phase_6_retrieval_explainability.md generated",
                ),
            ],
            "phase_7": [
                PlannedTask(
                    task_id="p7_autofix_benchmark",
                    description="Benchmark multi-strategy patch generation, syntax validity, and sandbox verification",
                    target_files=["reports/validation/phase_7_autofix_pr_benchmark.md"],
                    acceptance_criteria="Report reports/validation/phase_7_autofix_pr_benchmark.md generated",
                ),
            ],
            "phase_8": [
                PlannedTask(
                    task_id="p8_defect_fixes",
                    description="Verify integration defects, add regression tests, and record defect log",
                    target_files=["reports/validation/phase_8_fixes.md", "reports/validation/defect_log.md"],
                    acceptance_criteria="Defect log and regression tests verified",
                ),
            ],
            "phase_9": [
                PlannedTask(
                    task_id="p9_benchmark_runner",
                    description="Implement and verify standalone benchmark runner with manifest output",
                    command=".venv\\Scripts\\python -m backend.validation.run_all --output reports/validation/run_latest",
                    target_files=["backend/validation/run_all.py", "reports/validation/phase_9_benchmark_runner.md"],
                    acceptance_criteria="Benchmark runner executes and generates run manifest",
                ),
            ],
            "phase_10": [
                PlannedTask(
                    task_id="p10_final_report",
                    description="Assemble comprehensive final validation report, metrics JSON, and run manifest",
                    target_files=["reports/validation/milestone_8_final_report.md", "reports/validation/milestone_8_metrics.json", "reports/validation/milestone_8_run_manifest.json"],
                    acceptance_criteria="All final validation deliverables exist and summarize all verified findings",
                ),
            ],
        }
        return plans.get(phase_id, [PlannedTask(task_id=f"{phase_id}_task_1", description=f"Execute {phase_id}")])
