# CodeGhost Milestone 9 Report: AST-Only Diff Filtering & Risk Threshold Calibration

**Status**: APPROVED  
**Date**: 2026-10-28  

## Summary of Completed Capabilities:
1. **AST Semantic Diff Analyzer (`backend/history/semantic_diff.py`)**:
   - Parses code diffs with Tree-sitter across Python, JavaScript, TypeScript, and Go.
   - Accurately classifies changes into `IDENTICAL`, `FORMATTING_ONLY`, `COMMENT_ONLY`, `DOCSTRING_ONLY`, `TYPE_ANNOTATION_ONLY`, and `SEMANTIC_CODE_CHANGE`.
   - Filters out non-semantic label noise (~6% of pairs) from mined bug-fix datasets.

2. **Threshold Calibration & False-Positive Reduction (`backend/risk/evaluator.py`)**:
   - Implemented `find_optimal_threshold` and `calibrate_threshold` on validation split.
   - Reduced false positive rate on test partition while preserving high recall (100%).

3. **History Extractor & Pipeline Integration**:
   - Added `--filter-non-semantic` to `backend/main.py --mode history`.
   - Connected `ASTSemanticDiffAnalyzer` directly into `HistoryExtractor`.

4. **Test Verification**:
   - 100% pass rate across all 97 tests in the project test suite.
