# CodeGhost

**CodeGhost** is an AI system that reads a codebase, predicts which functions are likely to cause bugs based on historical bug patterns, explains the risk, and generates tested fixes.

This repository contains **Milestone 1: Repository Ingestion Layer**, which provides an AST-aware parser capable of ingesting arbitrary public GitHub repositories (or local repositories), parsing functions and classes with Tree-sitter, and persisting structured code units to JSONL.

---

## Features (Milestone 1)

- **Arbitrary Repository Ingestion**: Clones any public GitHub repository URL or accepts local git repository paths.
- **Git & Commit Validation**: Automatically checks repository integrity, verifies commits, and records the HEAD commit SHA.
- **Language Detection & Statistics**: Scans repository files, filters out non-source/binary files, and computes language distributions.
- **Multi-Language Tree-sitter AST Parsing**:
  - Python (`.py`, `.pyi`, `.pyw`)
  - JavaScript (`.js`, `.mjs`, `.cjs`, `.jsx`)
  - TypeScript (`.ts`, `.mts`, `.cts`, `.tsx`)
  - Go (`.go`)
  - Java (`.java`)
  - Rust (`.rs`)
  - C / C++ (`.c`, `.cpp`, `.cc`, `.cxx`, `.h`, `.hpp`)
- **Structured Code Unit Extraction**: Extracts functions, async functions, methods (with parent class context), and classes along with exact line numbers, parameters, docstrings, and raw source code.
- **JSONL Persistence**: Streams structured `CodeUnit` records directly to JSONL format.
- **Rich CLI & Summary Reporting**: Displays metrics, language breakdowns, error tables, and preview syntax highlighting.

---

## Directory Structure

```text
CodeGhost/
├── backend/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── repo.py          # Git cloning, caching, and commit validation
│   │   ├── language.py      # Language detection, ignore filtering & stats
│   │   ├── ast_parser.py    # Tree-sitter AST extraction
│   │   └── extractor.py     # Repository walker, JSONL writer & metrics
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic data models (CodeUnit, IngestionResult)
│   ├── data/
│   │   └── raw/             # Cloned repositories & JSONL outputs
│   ├── __init__.py
│   └── main.py              # CLI entrypoint and rich reporter
├── tests/
│   ├── test_repo.py
│   ├── test_language.py
│   ├── test_ast_parser.py
│   ├── test_extractor.py
│   └── test_pipeline.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.10+
- Git CLI

### Setup

```bash
# Clone or navigate to the repository
cd CodeGhost

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run via CLI

To ingest any public GitHub repository:

```bash
python backend/main.py https://github.com/pallets/flask
```

#### Optional Flags:
- `--output-dir <dir>`: Custom directory for storing raw repos and outputs (default: `backend/data/raw`).
- `--output-jsonl <file>`: Custom path for the output `.jsonl` file.
- `--branch <branch_or_tag>`: Specific branch or tag to checkout.
- `--depth <int>`: Shallow clone depth (e.g. `1` for faster downloads).
- `--clean`: Force re-cloning if the repository was previously cached.
- `--preview <int>`: Number of extracted units to display in terminal preview (default: `5`).
- `--verbose`: Enable debug logging.

### Example Output

```text
────────────────────── CodeGhost Repository Ingestion Report ───────────────────────

                                Ingestion Summary                                 
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric                      ┃ Value                                              ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Repository                  │ https://github.com/pallets/flask                   │
│ Commit SHA                  │ f8d2b...                                           │
│ Files Discovered            │ 120                                                │
│ Files Successfully Parsed   │ 42                                                 │
│ Functions Extracted         │ 380                                                │
│ Classes Extracted           │ 35                                                 │
│ Parsing Errors              │ 0                                                  │
│ JSONL Output File           │ backend/data/raw/pallets_flask_units.jsonl         │
└─────────────────────────────┴────────────────────────────────────────────────────┘
```

### Python API Usage

```python
from backend.ingestion.extractor import RepositoryExtractor

extractor = RepositoryExtractor()
result = extractor.ingest("https://github.com/pallets/flask")

print(f"Parsed {result.files_parsed} files, {result.functions_extracted} functions")
```


### Historical Bug Extraction (Milestone 2)

```bash
# Run historical bug extraction on a repository
python -m backend.main --repo https://github.com/pallets/flask --max-commits 500 --preview 10
```

### Python API Usage (Milestone 2)

```python
from backend.history.extractor import HistoryExtractor

extractor = HistoryExtractor()
summary = extractor.extract_history(
    repo_url_or_path="https://github.com/pallets/flask",
    max_commits=500,
)

print(f"Scanned {summary.commits_scanned} commits, extracted {summary.bug_fix_pairs_extracted} bug-fix pairs")
```

---

## Running Tests

Execute the complete pytest test suite (23 tests across all modules):

```bash
pytest tests/ -v
```

---

## Schema Reference

### `CodeUnit` (Milestone 1)
Structured representation of an extracted function, method, or class.

```json
{
  "repository": "https://github.com/pallets/flask",
  "commit_hash": "f8d2b785d10d19e080ee1174cb030a84e311f92e",
  "file_path": "src/flask/app.py",
  "unit_type": "method",
  "name": "Flask.route",
  "start_line": 1280,
  "end_line": 1315,
  "source_code": "    def route(self, rule: str, **options: t.Any) -> t.Callable:\n        ...",
  "language": "python",
  "parent_name": "Flask",
  "parameters": ["self", "rule", "options"],
  "docstring": "Decorate a view function to register it with the given URL rule and options.",
  "metadata": {"is_async": false}
}
```

### `BugFixPair` (Milestone 2)
Structured representation of an extracted historical bug fix (`BUGGY FUNCTION -> FIXED FUNCTION`).

```json
{
  "repository": "https://github.com/pallets/flask",
  "commit_sha": "05e9c6bd6bc9b39b34383c27ff77f805a5dfb0aa",
  "parent_sha": "7203feab2ba2b9cb0c6b1a20ee4db82bfdd8e04b",
  "commit_message": "Fix `.partition(\":\")` usage on potential IPv6 addresses (#6096)",
  "author": "David Lord",
  "timestamp": "2024-05-01T15:00:00+00:00",
  "file_path": "src/flask/app.py",
  "language": "python",
  "code_unit_type": "method",
  "class_name": "Flask",
  "function_name": "Flask.run",
  "buggy_code": "    def run(self, ...): ...",
  "fixed_code": "    def run(self, ...): ...",
  "diff": "--- a/src/flask/app.py:Flask.run\n+++ b/src/flask/app.py:Flask.run\n...",
  "additions": 3,
  "deletions": 1,
  "detector_score": 0.7,
  "detector_signals": ["fix_keyword:'Fix'"],
  "detector_classification": "BUG_FIX_CANDIDATE",
  "extraction_status": "success",
  "error_message": null
}
```

---

---

## Milestone 4: Risk Scoring Model (Classical ML & Fine-Tuned Code LLM)

Run the calibrated risk scoring pipeline to predict bug likelihood (1–10 numerical score and categorical risk tier) using historical change patterns and retrieval evidence:

```powershell
# 1. Train and evaluate Classical Baseline on the historical backtest test split
python -m backend.main --mode risk --model baseline --baseline-type random_forest

# 2. Compare Classical Baseline vs Qwen LLM
python -m backend.main --mode risk --model compare

# 3. Predict risk on a specific function (e.g. Flask.run)
python -m backend.main --mode risk --function Flask.run
```

---

## Milestone 5: LLM Reasoning & Explainability

Translate risk scores, AST structural complexity drivers, and retrieved historical bug precedents into comprehensive, human-interpretable technical explanations:

```powershell
# Run explainability on a target function
python -m backend.main --mode explain --function Flask.run
```

- **AST Complexity Drivers**: Evaluates Cyclomatic Complexity, Max Nesting Depth, Function Length, and Missing Error Handling.
- **Historical Analogies**: Correlates target code with past commit SHAs, author messages, and syntax diffs.
- **Vulnerability Patterns**: Highlights concrete hazards (e.g., host/port colon splitting without bracketed IPv6 checks, unshielded type casting).
- **Actionable Mitigations**: Proposes exact architectural refactorings.

---

## Milestone 6: Automated Fix Generation & Test Sandbox Execution

Synthesize candidate bug fixes and verify them inside an isolated temporary execution sandbox against automated test suites:

```powershell
# Generate candidate fixes and verify them in sandbox
python -m backend.main --mode fix --function Flask.run
```

- **Fix Strategies**:
  - `historical_analogy`: Directly adapts bug resolutions from repository git history.
  - `defensive_guard`: Injects input validation, null checks, and safe dict `.get()`.
  - `exception_shield`: Wraps risky type casts and external calls in `try/except`.
  - `bounds_check`: Enforces non-empty boundary checks prior to indexing.
- **Isolated Execution Sandbox**: Executes candidate patches in temporary directories with `pytest` subprocess isolation to ensure zero syntax errors or regressions.

---

## Milestone 7: GitHub PR Creation & Human-in-the-Loop Review

Package verified bug fixes into complete GitHub Pull Requests with detailed risk assessments, historical citations, and review checklists:

```powershell
# 1. Dry-run PR creation (generates local branch, diff, and serializes .patch file)
python -m backend.main --mode pr --function Flask.run --dry-run

# 2. Live PR submission to GitHub
python -m backend.main --mode pr --function Flask.run --live --github-token <YOUR_TOKEN>
```

---

## Autonomous End-to-End Workflow (`--mode scan`)

Execute the complete autonomous CodeGhost pipeline from end to end:

```powershell
python -m backend.main --mode scan --function Flask.run
```

1. **Retrieval**: Searches Qdrant vector database for nearest historical defect precedents.
2. **Risk Scoring**: Evaluates calibrated risk probability ($p \in [0.0, 1.0]$) and score (1–10).
3. **Reasoning**: Synthesizes AST complexity factors and grounded vulnerability explanations.
4. **Fix Generation**: Formulates syntax-verified candidate patches.
5. **Sandbox Verification**: Runs pytest in an isolated sandbox.
6. **PR Packaging**: Assembles ready-to-merge Pull Request draft and serialized patch.

---

## Running Tests

Execute the complete pytest test suite (67 tests across all 7 milestones):

```bash
pytest tests/ -q
```

---

## Roadmap

- [x] **Milestone 1**: Repository Ingestion & AST Extraction (Tree-sitter, Git, JSONL)
- [x] **Milestone 2**: Historical Bug Extraction (Buggy/fixed function pairs from git history)
- [x] **Milestone 3**: Function Embeddings & Vector Database Indexing (FastEmbed, Qdrant, BM25 baseline)
- [x] **Milestone 4**: Historical Risk Scoring Model (Classical ML Baseline, Qwen2.5-Coder LoRA pipeline)
- [x] **Milestone 5**: LLM Reasoning & Explainability (Preserving retrieved history & feature evidence)
- [x] **Milestone 6**: Automated Fix Generation & Test Sandbox Execution
- [x] **Milestone 7**: GitHub PR Creation & Human-in-the-Loop Review


