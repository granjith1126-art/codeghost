# CodeGhost Autonomous AI Orchestration & Milestone 8 Execution Plan

**Document Path**: `reports/validation/ai_orchestration_plan.md`  
**Created**: 2026-10-25T13:58:00+05:30  
**Git Baseline Commit**: `55e5f2fcedc60f919f71682f62710d70ea67a2b9`  
**Target Milestone**: Milestone 8 (Phases 2 through 10)  

---

## 1. Executive Mission & Orchestration Strategy

CodeGhost has implemented seven architectural layers (Ingestion, Bug Mining, Vector Embeddings/RAG, Risk Scoring, Explainability, Automated Fix Sandbox, and GitHub PR Dispatch). Milestone 8 validates this system end-to-end, evaluates actual performance, audits data leakage, discovers and repairs integration defects, and establishes reproducible benchmark results.

To accomplish this autonomously without requiring continuous manual intervention, this plan establishes:
1. **OpenAI Planning & Diagnostic Engine**: Uses official OpenAI SDK (`openai>=1.0.0`) with configurable model selection (`OPENAI_MODEL`, defaulting to `gpt-4o`) to formulate task plans, analyze failed test outputs, and propose bounded code repairs.
2. **Independent Controller Verification**: The controller never takes model claims of success on faith. Every task must satisfy independent test execution and artifact presence checks.
3. **Resumable State Management**: Workflow state is serialized to JSON checkpoints after every meaningful phase transition, enabling clean recovery after interruption.
4. **Strict Secret Redaction**: All API keys, tokens, and sensitive strings are masked in logs, reports, and error messages.

---

## 2. Investigation of Execution Interfaces

In accordance with Section 4 of the project specifications, we inspected the host environment to determine the actual execution interfaces available:

### A. Antigravity IDE & Environment Inspection
1. **Antigravity Host Application**: The current session is running inside the Antigravity desktop IDE. Antigravity does not expose an unauthenticated external local daemon socket, HTTP REST listener, or background RPC port for external command-line scripts (such as `python -m backend.orchestration.controller`) to drive IDE UI sessions or remote tool dispatch.
2. **Antigravity Python SDK (`google-antigravity`)**: The `google-antigravity` package is documented as a public Python SDK on PyPI for programmatic agent leasing. However, it is not installed in the active virtual environment and requires platform-specific native binaries.
3. **Honesty & Transparency Rule**: We will NOT invent fake Antigravity CLI commands or claim that external OpenAI API calls automatically control the Antigravity IDE.

### B. Locally Verified Alternative Execution Mechanism
To enable fully autonomous, reliable, and verifiable task execution, the controller uses a **Local Direct Code & Subprocess Executor (`LocalDirectExecutor`)**:
- **Capabilities**:
  - File reading, atomic writing, and patch application with automatic syntax validation and rollback on error.
  - Subprocess execution (`python -m pytest`, git commands, benchmark scripts) with strict timeouts, environment variable isolation, and stdout/stderr capture.
  - Git working-tree inspection, diff computation, and status validation.
  - Test result parsing and structured metric collection.
- **Modularity**: All executor interactions adhere to an abstract `BaseExecutor` interface (`backend/orchestration/executor.py`). An `AntigravitySDKExecutor` adapter is provided as an extensible plug-in when the SDK runtime becomes available.

---

## 3. OpenAI API Integration & Security

### A. Secret Protection & Configuration
1. **Zero Hardcoded Secrets**: The API key is never committed to Git, hardcoded in source code, or repeated in generated reports.
2. **Gitignore Protection**: `.env` is verified in `.gitignore` and excluded from source control.
3. **Template Documentation**: `.env.example` contains placeholder tokens only.
4. **Live Connectivity Verification**: Verified real connectivity via `client.models.list()`.
5. **Masking & Redaction**: The `SecretRedactor` filter scrubs all patterns matching `sk-...`, `Bearer ...`, or sensitive tokens into `[REDACTED_SECRET]`.
6. **Billing Notice**: Documented that OpenAI API usage requires pay-as-you-go platform credits and is billed separately from ChatGPT consumer subscriptions.

### B. Safety & Cost Budgets
Default safety limits configured in `backend/orchestration/config.py`:
- `MAX_REQUESTS_PER_RUN`: 50
- `MAX_TOTAL_TOKENS`: 250,000
- `MAX_COST_USD`: $5.00
- `PER_COMMAND_TIMEOUT_SECONDS`: 60
- `MAX_REPAIR_ATTEMPTS_PER_TASK`: 3
- `MAX_CONSECUTIVE_FAILURES`: 3

---

## 4. Orchestration Architecture (`backend/orchestration/`)

```text
backend/orchestration/
├── __init__.py
├── config.py           # Configuration, environment loading, safety & cost budgets
├── logging_utils.py    # Structured logging with automatic secret redaction
├── state_store.py      # Resumable JSON state store and checkpoint manager
├── executor.py         # BaseExecutor, LocalDirectExecutor, AntigravitySDKExecutor adapter
├── planner.py          # OpenAI task planner with structured JSON schema
├── verification.py     # Independent verification engine (test execution, artifact checks)
├── phase_runner.py     # Milestone 8 phase manager (Phases 2 through 10)
├── reporter.py         # Markdown and JSON report generator
└── controller.py       # Main CLI entrypoint (--status, --milestone 8, --resume, --dry-run, --report)
```

---

## 5. Milestone 8 Phase Execution Sequence

| Phase | Title | Primary Deliverable | Acceptance Criteria |
| :---: | :--- | :--- | :--- |
| **Phase 1** | Baseline Audit | `reports/validation/initial_audit.md` | **COMPLETED** (Commit `55e5f2f`, 67 tests passing) |
| **Phase 2** | Evaluation Protocol | `reports/validation/phase_2_evaluation_protocol.md` | Formal mathematical definitions of risk, retrieval, patch, and pipeline metrics |
| **Phase 3** | Data Leakage Audit | `reports/validation/phase_3_data_audit.md` | Investigation of temporal split, function overlap, post-fix leakage, and 100% recall root cause |
| **Phase 4** | E2E Integration Test | `tests/integration/test_end_to_end.py` + `reports/validation/phase_4_e2e.md` | Deterministic local Git fixture exercising all 7 layers with passing and failing cases |
| **Phase 5** | Risk Scoring Benchmark | `reports/validation/phase_5_risk_benchmark.md` + metrics JSON | Pure NumPy Random Forest evaluated on held-out test split with confusion matrix and baseline comparison |
| **Phase 6** | Retrieval & Explainability | `reports/validation/phase_6_retrieval_explainability.md` | BGE vector retrieval vs BM25 lexical baseline, latency, and explanation groundedness |
| **Phase 7** | Automated Fix Benchmark | `reports/validation/phase_7_autofix_pr_benchmark.md` | Multi-strategy patch generation, syntax validity, sandbox verification rate, PR draft generation |
| **Phase 8** | Integration Defect Fixes | `reports/validation/phase_8_fixes.md` + `defect_log.md` | Prioritized defect resolution, targeted regression tests, zero regression on 67 existing tests |
| **Phase 9** | Benchmark Runner | `backend/validation/run_all.py` + `phase_9_benchmark_runner.md` | Standalone CLI runner with repeatable seeds, manifest output, and non-overwriting runs |
| **Phase 10**| Final Report | `reports/validation/milestone_8_final_report.md` + metrics & run manifests | Comprehensive summary with clear status labels (PASS, PASS WITH LIMITATIONS, BLOCKED) |

---

## 6. Verification & Self-Testing Plan for the Orchestrator

Before orchestrating changes to the core codebase, the orchestration system itself will be validated with targeted unit tests in `tests/test_orchestration.py`:
1. `test_secret_redaction`: Verifies keys and tokens are never written to logs or strings.
2. `test_state_persistence_and_resume`: Verifies state serialization and safe resume.
3. `test_executor_command_execution`: Verifies subprocess execution, timeout handling, and exit code capture.
4. `test_independent_verifier`: Confirms that tasks with failing tests are rejected regardless of LLM claims.
5. `test_budget_limits`: Verifies execution halts when token/request budgets are exceeded.
6. `test_dry_run_mode`: Verifies plans can be simulated without disk or network mutations.
