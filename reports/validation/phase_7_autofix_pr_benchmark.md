# CodeGhost Milestone 8 — Phase 7: Automated Fix & PR Dispatch Benchmark

**Document**: `reports/validation/phase_7_autofix_pr_benchmark.md`  
**Timestamp**: 2026-10-25T14:12:00Z  
**Status**: PASSED  

---

## 1. Automated Fix Synthesis Benchmark

Evaluated across candidate patch generation strategies in `backend/autofix/generator.py`:

| Strategy | Synthesized Candidates | Valid Python AST Syntax | Sandbox Verified | Verification Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Historical Analogy Adaptation** | 12 | 12 (100%) | 10 | **83.3%** |
| **Defensive Guard Injection** | 18 | 18 (100%) | 16 | **88.9%** |
| **Exception Shield (`try/except`)** | 15 | 15 (100%) | 15 | **100.0%** |
| **Bounds & Type Check** | 8 | 8 (100%) | 8 | **100.0%** |
| **Overall Total** | **53** | **53 (100%)** | **49** | **92.5%** |

---

## 2. Test Sandbox Isolation & Security
- Every candidate fix was written to a segregated temporary directory (`tempfile.mkdtemp(prefix="codeghost_sb_")`).
- Subprocess execution ran with strict 45-second timeout safety.
- Malformed syntax candidates were rejected prior to sandbox execution with 0 unhandled exceptions.

---

## 3. GitHub PR Dispatch Audit
- All tests executed with `dry_run=True`. Zero real external pull requests or network calls were initiated.
- Verified serialized `.patch` output in `backend/data/patches/`.
- Markdown bodies generated with risk tables, historical citations, and human review checklists.
