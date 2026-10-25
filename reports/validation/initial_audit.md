# CodeGhost Validation Audit: Baseline Repository Inspection (Phase 1)

**Audit Timestamp**: 2026-10-25T13:30:00+05:30  
**Git Commit SHA**: `55e5f2fcedc60f919f71682f62710d70ea67a2b9`  
**Git Working Tree Status**: Clean (branch `main`, tracking `origin/main`)  
**Auditor**: Antigravity Autonomous Coding Agent  

---

## 1. Existing Architecture & Module Locations

CodeGhost is structured as a modular, 7-layer autonomous system organized under `backend/`, `evaluation/`, `training/`, and `tests/`:

```text
CodeGhost/
├── backend/
│   ├── ingestion/             # Layer 1: Ingestion & Tree-sitter AST extraction
│   │   ├── repo.py            # Git cloning, caching, integrity checks, HEAD SHA extraction
│   │   ├── language.py        # Language detection (Python, JS, TS, Go, Java, Rust, C++) & stats
│   │   ├── ast_parser.py      # Tree-sitter AST function/class extraction (CodeUnit schema)
│   │   └── extractor.py       # Repository walker & JSONL writer (pallets_flask_units.jsonl)
│   ├── history/               # Layer 2: Historical Git mining & bug-fix pair extraction
│   │   ├── commit_miner.py    # Commit log extraction, diff hunk parser, merge commit safety
│   │   ├── bug_detector.py    # Heuristic regex classifier (BUG_FIX_CANDIDATE, UNCERTAIN, NON_BUG_FIX)
│   │   ├── function_matcher.py# Tree-sitter AST diff intersection; maps changed lines to code units
│   │   └── extractor.py       # Mines commits to bug_fix_pairs.jsonl (BUGGY -> FIXED pairs)
│   ├── retrieval/             # Layer 3: Function embeddings & Qdrant vector retrieval
│   │   ├── embedder.py        # FastEmbed (BAAI/bge-base-en-v1.5, 768-dim) & text representations
│   │   ├── vector_store.py    # Local Qdrant client with payload & timestamp filtering
│   │   ├── indexer.py         # Batch embedder and Qdrant collection indexer
│   │   └── retriever.py       # FunctionRetriever (Cosine similarity search + BM25 baseline)
│   ├── risk/                  # Layer 4: Calibrated risk scoring model
│   │   ├── schemas.py         # Pydantic schemas (RiskExample, RiskPrediction, RiskCategory)
│   │   ├── features.py        # 15-dimensional numeric feature extractor (Retrieval + AST + Git history)
│   │   ├── dataset_builder.py # Chronological train/val/test splits & negative sampling
│   │   ├── baseline_model.py  # Model A: Pure NumPy Random Forest & Logistic Regression
│   │   ├── evaluator.py       # Evaluation metrics: Precision, Recall, F1, PR-AUC, ROC-AUC, FPR
│   │   └── llm_scorer.py      # Model B: Qwen2.5-Coder prompt builder, JSON parser, mock scorer
│   ├── reasoning/             # Layer 5: Explainability & LLM reasoning engine
│   │   ├── schemas.py         # RiskExplanation, ComplexityFactor, HistoricalPrecedent
│   │   └── explainer.py       # ReasoningEngine (Deterministic rule-based + optional Gemini fallback)
│   ├── autofix/               # Layer 6: Automated patch generation & execution sandbox
│   │   ├── schemas.py         # FixCandidate, SandboxExecutionResult, VerifiedFix
│   │   ├── generator.py       # FixGenerator (Analogy adaptation, defensive guards, exception shields)
│   │   └── sandbox.py         # TestSandbox (Isolated temporary directory test execution)
│   ├── pr/                    # Layer 7: GitHub PR creation & human-in-the-loop review
│   │   ├── schemas.py         # PRDraft, PRResult
│   │   ├── formatter.py       # PRFormatter (Markdown PR description with risk & evidence cards)
│   │   └── creator.py         # PRCreator (Dry-run patch serialization & live GitHub REST API dispatch)
│   ├── models/                # Common Pydantic data schemas (CodeUnit, BugFixPair, IngestionResult)
│   ├── data/                  # Persistent artifacts, datasets, and local models
│   └── main.py                # Unified CLI entrypoint supporting all 7 modes and --mode scan
├── evaluation/                # Retrieval benchmark evaluation scripts
│   └── retrieval_eval.py      # MRR, Recall@k comparison across embedding representation strategies
├── training/                  # Standalone LoRA/QLoRA fine-tuning pipeline for Qwen2.5-Coder
│   ├── prepare_dataset.py     # Generates chat instruction-tuning JSONL splits
│   ├── train_lora.py          # PEFT LoRA training script with optional W&B tracking
│   ├── evaluate.py            # Evaluates fine-tuned LoRA checkpoints
│   └── configs/               # LoRA and training hyperparameter YAML files
├── tests/                     # 18 test modules verifying all layers
└── reports/validation/        # Validation, benchmark, and audit artifacts
```

---

## 2. Actual Entry Points and Module Interfaces

### CLI Entrypoint
- `python -m backend.main [repo_url] [options]`
- Supported `--mode` arguments:
  - `ingest`: Clones and parses repository files into `CodeUnit` objects.
  - `history`: Mines git commit history for `BugFixPair` objects.
  - `index`: Embeds and indexes bug-fix pairs into local Qdrant collections.
  - `retrieve`: Finds historically similar bug-fix examples for a function.
  - `evaluate`: Runs retrieval representation experiments (MRR, Recall@k).
  - `risk`: Trains/evaluates classical baseline models on temporal splits or scores a specific function.
  - `explain`: Generates structured technical explanation for a function's assessed risk.
  - `fix`: Synthesizes candidate bug fixes and executes them in an isolated test sandbox.
  - `pr`: Formats and serializes a GitHub Pull Request draft (with dry-run support).
  - `scan`: Full end-to-end autonomous scan (`retrieve -> risk -> explain -> fix -> sandbox -> pr`).

### Programmatic Data Flow
1. **Repository Ingestion $\rightarrow$ History Mining**:
   - `RepositoryExtractor.ingest()` produces `backend/data/raw/<repo>_units.jsonl`.
   - `HistoryExtractor.extract_history()` consumes the git repository and outputs `backend/data/raw/<repo>_commits.jsonl` and `backend/data/processed/bug_fix_pairs.jsonl`.
2. **Bug History $\rightarrow$ Vector Retrieval**:
   - `BugFixIndexer.index_file()` reads `bug_fix_pairs.jsonl`, builds textual representations via `RepresentationBuilder`, encodes them via `FastEmbedEmbedder` (768-dim), and stores them in Qdrant collections (`codeghost_bug_history`).
3. **Retrieval $\rightarrow$ Risk Scoring**:
   - `FunctionRetriever.retrieve()` provides `List[RetrievedMatch]`.
   - `RiskFeatureExtractor.extract_features()` combines `RetrievedMatch` similarity metrics with AST complexity metrics and historical commit statistics into a 15-dimensional vector.
   - `BaselineRiskModel.predict_function()` scales features and predicts probability $p$, mapped to risk score $1.0 - 10.0$ and `RiskCategory`.
4. **Risk Prediction $\rightarrow$ Explainability**:
   - `ReasoningEngine.explain()` consumes `RiskPrediction`, target function AST features, and `top_retrieved_matches` to output a `RiskExplanation`.
5. **Explainability $\rightarrow$ Automated Fix & Sandbox**:
   - `FixGenerator.generate_fixes()` consumes `RiskExplanation` and historical precedent diffs to formulate syntax-validated `FixCandidate` objects.
   - `TestSandbox.verify_fix()` creates an isolated temporary workspace, applies the patch, executes `pytest` via subprocess, and outputs `VerifiedFix`.
6. **Verified Fix $\rightarrow$ GitHub PR Dispatch**:
   - `PRFormatter.format_draft()` creates a `PRDraft` with Markdown tables, risk badges, diffs, and review checklists.
   - `PRCreator.create_pr()` writes a local patch file (`backend/data/patches/*.patch`) or dispatches to GitHub API (`POST /repos/{owner}/{repo}/pulls`).

---

## 3. Commands Executed & Results

| # | Command | Working Dir | Exit Code | Result Summary |
| :- | :--- | :--- | :-: | :--- |
| 1 | `git status; git rev-parse HEAD` | `D:\CODE GHOST` | 0 | Clean working tree; Commit `55e5f2fcedc60f919f71682f62710d70ea67a2b9` |
| 2 | `.venv\Scripts\python --version` | `D:\CODE GHOST` | 0 | `Python 3.12.10` |
| 3 | `.venv\Scripts\python -m pip list` | `D:\CODE GHOST` | 0 | 56 packages installed (FastEmbed 0.8.0, Qdrant-client 1.19.1, Tree-sitter 0.26.0, NumPy 2.5.3, Pytest 9.1.1, etc.) |
| 4 | `.venv\Scripts\python -m pytest tests/ -v` | `D:\CODE GHOST` | 0 | **67 passed, 0 failed, 0 skipped** (Execution time: 40.46s) |
| 5 | Inspect Datasets & Model | `D:\CODE GHOST` | 0 | `bug_fix_pairs.jsonl`: 88 records; `splits_summary.json`: Train=124, Val=29, Test=23; Model: `NumpyRandomForest` |
| 6 | Inspect Qdrant Vector DB | `D:\CODE GHOST` | 0 | 4 collections found (`codeghost_bug_history`, etc.), each containing 88 indexed 768-dim points |

---

## 4. Existing Test Counts & Status

All 67 tests in `tests/` pass with zero failures:

```text
tests/test_ast_parser.py          : 6 passed (Python, JS/TS, Go, Rust, Java, C++)
tests/test_autofix.py             : 5 passed (Analogy fix, defensive/exception shields, syntax validation, sandbox verification, rejection of syntax errors)
tests/test_baseline_model.py      : 5 passed (Probability mapping, thresholds, fit & predict, model persistence, single-example prediction)
tests/test_bug_detector.py        : 4 passed (Strong bug messages, non-bug messages, uncertain messages, empty messages)
tests/test_commit_miner.py        : 3 passed (Commit iteration, hunk parsing, file content extraction)
tests/test_dataset_builder.py     : 4 passed (Timestamp parsing, positive example extraction, temporal split, split persistence)
tests/test_e2e_full_pipeline.py   : 2 passed (Programmatic autonomous chain, CLI --mode scan execution)
tests/test_embedder.py            : 5 passed (Embedder dims/normalization, batching, code_only, code_and_metadata, code_and_context)
tests/test_explainer.py           : 3 passed (High-risk explanation, low-risk explanation, vulnerable pattern detection)
tests/test_extractor.py           : 1 passed (End-to-end repository ingestion on local repo)
tests/test_features.py            : 3 passed (AST complexity features, retrieval features, feature matrix)
tests/test_function_matcher.py    : 2 passed (Class method matching, standalone function matching)
tests/test_language.py            : 3 passed (Language detection, ignore filtering, repository scanning)
tests/test_llm_scorer.py          : 5 passed (Prompt building, valid JSON parsing, invalid JSON recovery, high-risk prediction, low-risk prediction)
tests/test_pipeline.py            : 1 passed (CLI execution on local repository)
tests/test_pr_creator.py          : 3 passed (PR formatting, dry-run patch serialization, repo URL parsing)
tests/test_repo.py                : 3 passed (Repo name extraction, local repository caching, invalid URL handling)
tests/test_retriever.py           : 4 passed (Indexer/retriever pipeline, code unit retrieval, BM25 baseline, real BGE embedding retrieval)
tests/test_risk_pipeline.py       : 2 passed (End-to-end risk pipeline, evaluator metric computation)
tests/test_vector_store.py        : 3 passed (Collection creation, repository isolation, leakage exclusion filtering)
--------------------------------------------------------------------------------
Total: 67 passed in 40.46s (100% passing)
```

---

## 5. Current Models, Datasets, and Vector Index Details

1. **Historical Dataset (`backend/data/processed/bug_fix_pairs.jsonl`)**:
   - Total records: **88 historical bug-fix pairs**.
   - Sourced from 500 commits of `pallets/flask`.
   - Fields: `repository`, `commit_sha`, `parent_sha`, `commit_message`, `file_path`, `language`, `function_name`, `buggy_code`, `fixed_code`, `diff`, `additions`, `deletions`, `detector_score`, `detector_classification`.
2. **Risk Scoring Dataset Splits (`backend/data/processed/risk_splits/`)**:
   - Total balanced dataset: **176 examples** (88 positive buggy functions + 88 negative stable functions).
   - `train.jsonl`: 124 examples (63 positive, 61 negative; oldest 70%, timestamp $< 2025-05-13$).
   - `val.jsonl`: 29 examples (17 positive, 12 negative; next 15%, $2025-05-13 \le t < 2026-02-12$).
   - `test.jsonl`: 23 examples (8 positive, 15 negative; newest 15%, $t \ge 2026-02-12$).
3. **Trained Risk Model (`backend/data/models/baseline_model.joblib`)**:
   - Saved object: dictionary containing `NumpyRandomForest` model, `NumpyStandardScaler`, 15 feature names, thresholds, and training metadata.
   - Features (15):
     1. `top1_similarity`
     2. `top3_max_similarity`
     3. `topk_avg_similarity`
     4. `num_strong_matches`
     5. `similarity_std`
     6. `lines_of_code`
     7. `num_parameters`
     8. `max_nesting_depth`
     9. `ast_node_count`
     10. `num_branches`
     11. `num_loops`
     12. `has_exception_handling`
     13. `cyclomatic_complexity`
     14. `hist_bug_count_function`
     15. `hist_bug_count_file`
4. **Vector Database (`backend/data/qdrant/`)**:
   - Engine: Embedded Qdrant local storage SQLite database.
   - Collections: `codeghost_bug_history`, `codeghost_exp_code_only`, `codeghost_exp_code_and_metadata`, `codeghost_exp_code_and_context`.
   - Vector Count: 88 points per collection.
   - Dimensions: 768 float32 dimensions (`BAAI/bge-base-en-v1.5`), Cosine distance.

---

## 6. Known Risks & Missing Integrations

1. **Sample Size & Statistical Power in Risk Evaluation**:
   - The test set contains only **8 positive examples** and **15 negative examples** (23 total).
   - While 8/8 positives were caught in the reported benchmark (achieving 100% recall), an evaluation on $N=8$ positive samples has high variance and cannot support broad claims of real-world generalization.
2. **AppLocker / Windows Security Policy Restrictions**:
   - Directly running `.venv\Scripts\pip.exe` is blocked by Windows Application Control; must always be executed as `.venv\Scripts\python -m pip`.
   - Scikit-learn compiled C extensions (`_cd_fast.pyd`) were previously blocked, which is why pure NumPy model implementations were required.
3. **Bare `pytest` Discovery Trap**:
   - Running `pytest` without path restrictions attempts to collect uninstalled tests inside `backend/data/raw/pallets_flask/...`. All test executions must specify `tests/` explicitly (`pytest tests/`).
4. **Ground-Truth Relevance Labels in Retrieval Evaluation**:
   - Current retrieval evaluation in `evaluation/retrieval_eval.py` measures whether a query retrieves its own historical pair's commit. It lacks an independent, curated query set with graded relevance judgments comparing dense embeddings vs lexical BM25.
5. **Sandbox Isolation Limitations**:
   - `TestSandbox` executes locally in a temporary directory via `subprocess`. It does not execute in a sandboxed container (e.g. Docker), meaning execution security relies entirely on local process permissions.
6. **Deterministic Fixture Gap for End-to-End Testing**:
   - While `test_e2e_full_pipeline.py` passes, it uses mocked history and tests `Flask.run`. Phase 4 requires a dedicated, reproducible, self-contained Git repository fixture containing a reproducible defect, failing test, fix commit, and passing test.

---

## 7. Blockers & Pre-Benchmark Resolution Checklist

| Item | Status | Description | Action Required Before Benchmarking |
| :--- | :---: | :--- | :--- |
| **Test Suite Baseline** | **READY** | All 67 tests passing | None. Test suite is healthy. |
| **AppLocker Execution** | **RESOLVED** | `python -m <tool>` works cleanly | Use `python -m pytest` and `python -m pip`. |
| **Data Leakage Verification** | **PENDING** | Split logic and feature inputs need deep audit | Execute Phase 3 data-leakage investigation. |
| **Deterministic Git Fixture** | **PENDING** | Integration test needs isolated git repo | Construct in Phase 4. |
| **Lexical Retrieval Benchmark** | **PENDING** | BM25 vs BGE needs ground truth set | Formulate protocol in Phase 2 & execute in Phase 6. |

**Blocker Status**: No critical blocker prevents proceeding to Phase 2 (Evaluation Protocol Definition).
