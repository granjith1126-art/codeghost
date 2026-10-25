# CodeGhost Milestone 8 — Phase 3: Data Leakage & Dataset Audit

**Document**: `reports/validation/phase_3_data_audit.md`  
**Timestamp**: 2026-10-25T14:02:00Z  
**Audit Target**: `backend/data/processed/risk_splits/`  
**Status**: PASSED WITH LIMITATIONS  

---

## 1. Executive Summary & Investigation of Reported 100% Recall

In prior backtests, the Random Forest baseline reportedly achieved **100% recall (8/8 positives caught)** on the test split. This audit investigated the exact root cause of this result to determine whether it stemmed from data leakage or genuine signal:

### Key Findings:
1. **Commit Lineage Isolation**:
   - `train` commit SHAs count: 89
   - `test` commit SHAs count: 20
   - Direct Commit SHA Overlap between Train and Test: **0** (Zero overlap).
   - Direct Commit SHA Overlap between Val and Test: **0** (Zero overlap).
2. **Temporal Boundary Enforcement**:
   - Split 1 (Train): Oldest 70% of history (124 samples, cutoff: `2025-05-13T01:30:27Z`).
   - Split 2 (Val): Subsequent 15% (29 samples, cutoff: `2026-02-12T21:07:50Z`).
   - Split 3 (Test): Most recent 15% (23 samples, timestamps >= 2026-02-12).
   - Vector search queries during backtesting strictly enforced `max_timestamp < sample.timestamp` and `exclude_commit_sha = sample.commit_sha`, preventing lookahead leakage.
3. **Root Cause of 100% Recall**:
   - The test split contains only **8 positive buggy functions**.
   - These 8 functions are high-complexity functions (`Flask.run`, `Flask.open_resource`, etc.) with high cyclomatic complexity (>= 9) and frequent historical defect lineage.
   - The Random Forest easily separates them from the 15 negative stable functions.
   - **Conclusion**: The 100% recall is mathematically authentic on this partition, **BUT statistically inconclusive** because N_pos = 8 has wide 95% confidence intervals ([63.1%, 100%] by Wilson score interval). It must not be cited as proof of general 100% production recall.

---

## 2. Function Overlap Analysis
- Total unique functions in Train: 84
- Total unique functions in Test: 20
- Overlapping function names between Train and Test: **1** (Flask.make_response).
- **Audit Note**: Core framework methods (like `Flask.run` or `Flask.route`) evolve across multiple commits over a 10-year Git history. When evaluating a new commit modifying `Flask.run`, prior fixes to `Flask.run` are valid historical evidence as long as they occurred strictly in the past.

---

## 3. Label Noise Inspection
- Heuristic classification in `backend/history/bug_detector.py` uses keyword weights (`fix`, `bug`, `issue`).
- Review of 88 pairs indicates ~6% of pairs involved type-annotation refinements or small docstring corrections that coincided with bug fixes.
- Recommendation: In Milestone 9, incorporate AST-only diff filtering to exclude non-semantic changes.
