# CodeGhost Milestone 8 — Phase 6: Retrieval & Explainability Benchmark

**Document**: `reports/validation/phase_6_retrieval_explainability.md`  
**Timestamp**: 2026-10-25T14:10:00Z  
**Embedding Model**: `BAAI/bge-base-en-v1.5` (768 dimensions, Normalized Cosine Distance)  
**Vector Database**: `Qdrant` (Embedded SQLite)  
**Status**: PASSED WITH LIMITATIONS  

---

## 1. Historical Retrieval Benchmark (Dense BGE vs BM25 Lexical)

Evaluated across the 88 historical bug-fix pairs in `backend/data/processed/bug_fix_pairs.jsonl`:

| Retrieval Method | Recall@1 | Recall@3 | Recall@5 | MRR | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dense BGE (`code_and_metadata`)** | **0.8636** | **0.9545** | **0.9773** | **0.9125** | 18.4 ms |
| **Dense BGE (`code_only`)** | 0.7727 | 0.8864 | 0.9205 | 0.8350 | 16.2 ms |
| **BM25 Lexical Baseline** | 0.5909 | 0.7273 | 0.7955 | 0.6680 | **4.1 ms** |

### Findings:
1. Dense vector embeddings with function metadata outperform pure keyword search by **+27.3% on Recall@1** and **+0.245 on MRR**.
2. Including file paths, class names, and commit signals in the embedded text (`code_and_metadata`) significantly improves discrimination between identically named helper functions.

---

## 2. Explainability Layer Fidelity Audit

Audited across high-, medium-, and low-risk generated explanations (`ReasoningEngine`):
- **Evidence Provenance**: 100% of historical precedents cited in generated explanations map directly to real retrieved vector search points.
- **AST Metric Alignment**: Cyclomatic complexity and nesting depth claims strictly match AST parser metrics.
- **Vulnerability Pattern Detection**: Accurately detects colon-splitting hazards without IPv6 bracket handling, unshielded `int()` casts, and direct dictionary subscripting.
