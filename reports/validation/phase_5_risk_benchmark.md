# CodeGhost Milestone 8 — Phase 5: Risk Scoring Benchmark Report

**Document**: `reports/validation/phase_5_risk_benchmark.md`  
**Timestamp**: 2026-10-25T14:07:00Z  
**Model**: `RandomForest (Pure NumPy, 100 trees, max_depth=5)`  
**Status**: PASSED WITH LIMITATIONS  

---

## 1. Benchmark Results on Untouched Chronological Test Partition

Evaluated on the held-out temporal **Test Split** (23 samples: 8 positive buggy functions, 15 negative stable functions):

| Metric | Random Forest (CodeGhost) | Majority-Class Baseline | Heuristic LOC Baseline |
| :--- | :---: | :---: | :---: |
| **Recall (Sensitivity)** | **1.0000 (8/8)** | 0.0000 | 0.6250 |
| **Precision** | **0.3636** | 0.0000 | 0.3125 |
| **F1-Score** | **0.5333** | 0.0000 | 0.4167 |
| **ROC-AUC** | **0.8417** | 0.5000 | 0.6125 |
| **PR-AUC** | **0.8063** | 0.3478 | 0.4350 |
| **False Positive Rate** | 0.9333 (14/15) | **0.0000** | 0.7333 |
| **Inference Latency** | **1.48 ms** | <0.01 ms | <0.01 ms |

---

## 2. Confusion Matrix

```text
                  Predicted Safe (0)   Predicted Risky (1)
True Safe (0)             1                     14
True Risky (1)            0                      8
```

- **Zero Missed Defects**: The model caught 100% of the true bugs in the unseen future test split.
- **High False Positive Rate**: 14 of 15 stable functions were flagged as risky due to shared complexity features with framework entrypoints.
- **Wilson 95% Confidence Interval**:
  - Precision: `[18.0%, 59.2%]`
  - Recall: `[63.1%, 100.0%]`
