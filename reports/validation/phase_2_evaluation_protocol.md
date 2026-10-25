# CodeGhost Milestone 8 — Phase 2: Reproducible Evaluation Protocol

**Document**: `reports/validation/phase_2_evaluation_protocol.md`  
**Timestamp**: 2026-10-25T14:00:00Z  
**Author**: CodeGhost Autonomous Orchestrator  
**Status**: APPROVED  

---

## 1. Objectives & Scope
This evaluation protocol defines the formal mathematical metrics, acceptance criteria, dataset partitions, and ground-truth validation rules across all four core analytical subsystems:
1. **Risk Scoring Engine**
2. **Historical Bug Retrieval (RAG)**
3. **Automated Fix Generation & Test Sandbox**
4. **End-to-End Autonomous Pipeline**

---

## 2. Evaluation Subsystem A: Risk Scoring Model

### Evaluation Question
> *"Does CodeGhost accurately predict which repository functions are historically prone to regression faults before a bug is introduced?"*

### Primary Target Metric Suite
Given true labels $y_i \in \{0, 1\}$ (where $1 = \text{Positive/Buggy}$, $0 = \text{Negative/Stable}$) and predicted probabilities $\hat{p}_i \in [0.0, 1.0]$:

1. **Confusion Matrix**:
   $$\begin{pmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{pmatrix}$$
2. **Precision**: $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$
3. **Recall (Sensitivity)**: $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$
4. **F1-Score**: $\text{F}_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
5. **False Positive Rate (FPR)**: $\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$
6. **PR-AUC (Precision-Recall Area Under Curve)**:
   $$\text{PR-AUC} = \sum_{k=1}^N (\text{Recall}_k - \text{Recall}_{k-1}) \cdot \text{Precision}_k$$
   *Crucial for imbalanced code defect detection datasets.*
7. **ROC-AUC**:
   Calculated using Mann-Whitney U statistic:
   $$\text{ROC-AUC} = \frac{\sum_{i \in \text{Pos}} \sum_{j \in \text{Neg}} \mathbb{I}(\hat{p}_i > \hat{p}_j) + 0.5 \cdot \mathbb{I}(\hat{p}_i = \hat{p}_j)}{|\text{Pos}| \cdot |\text{Neg}|}$$
   *Requires both positive and negative samples in the partition.*

### Baseline Comparisons
Every evaluation must be benchmarked against:
- **Majority Class Baseline**: Always predicts the majority label (Class 0).
- **Random Baseline**: Uniform random sampling $\hat{p} \sim \mathcal{U}(0, 1)$.
- **Lexical/Heuristic Complexity Baseline**: Risk proportional to Cyclomatic Complexity alone.

---

## 3. Evaluation Subsystem B: Historical Retrieval (RAG)

### Evaluation Question
> *"Given a target function, does dense vector search retrieve genuinely analogous past bug-fix commits rather than superficially similar code?"*

### Metrics
1. **Recall@k** ($k \in \{1, 3, 5\}$):
   $$\text{Recall@}k = \frac{\mathbb{I}(\text{rank}(\text{relevant}) \le k)}{1}$$
2. **Mean Reciprocal Rank (MRR)**:
   $$\text{MRR} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\text{rank}_q}$$
3. **Baseline Comparison**:
   Comparison against BM25 lexical keyword matching over the exact same query set.

---

## 4. Evaluation Subsystem C: Automated Patch Verification

### Evaluation Question
> *"Do synthesized patches resolve identified defects without introducing syntax errors or test regressions?"*

### Metrics
- **Syntax Validity Rate**: $\frac{N_{\text{valid\_ast}}}{N_{\text{total\_candidates}}}$
- **Sandbox Test Pass Rate**: $\frac{N_{\text{passed\_tests}}}{N_{\text{attempted\_tests}}}$
- **Regression Rate**: Percentage of candidate patches that break previously passing tests.
- **Verified Fix Rate**: Percentage of candidates where syntax is valid AND all sandbox tests pass.

---

## 5. Evaluation Subsystem D: Full Pipeline

### Metrics
- **End-to-End Success Rate**: $\frac{N_{\text{clean\_pr\_payloads}}}{N_{\text{scanned\_functions}}}$
- **Latency Breakdown**: Ingestion (s), Mining (s), Vector Query (ms), Risk Inference (ms), Patch & Sandbox (s).
