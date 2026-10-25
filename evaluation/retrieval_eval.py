"""Retrieval Evaluation Suite: Benchmarks embedding retrieval vs lexical baseline and compares representations."""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from rank_bm25 import BM25Plus

from backend.models.schemas import BugFixPair
from backend.retrieval.embedder import BaseEmbedder, FastEmbedEmbedder, RepresentationBuilder
from backend.retrieval.retriever import FunctionRetriever
from backend.retrieval.schemas import EvaluationResult, RepresentationType, RetrievalQuery, RetrievedMatch
from backend.retrieval.vector_store import BaseVectorStore, QdrantVectorStore

logger = logging.getLogger(__name__)


def tokenize_code(text: str) -> List[str]:
    """Tokenize source code by identifiers, keywords, and symbols."""
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


class BM25LexicalBaseline:
    """Robust BM25Plus lexical baseline for function code retrieval."""

    def __init__(self, pairs: List[BugFixPair], representation_type: str = "code_only"):
        self.pairs = pairs
        self.representation_type = representation_type
        self.corpus_texts = [
            RepresentationBuilder.build_pair_representation(p, representation_type=representation_type)
            for p in pairs
        ]
        self.tokenized_corpus = [tokenize_code(t) for t in self.corpus_texts]
        self.bm25 = BM25Plus(self.tokenized_corpus)

    def retrieve(
        self,
        query: RetrievalQuery,
        top_k: int = 5,
    ) -> List[RetrievedMatch]:
        query_text = RepresentationBuilder.build_query_representation(
            code=query.code,
            representation_type=self.representation_type,
            language=query.language,
            file_path=query.file_path,
            class_name=query.class_name,
            function_name=query.function_name,
        )
        tokenized_query = tokenize_code(query_text)
        scores = self.bm25.get_scores(tokenized_query)

        # Pair scores with candidates
        scored_pairs = []
        for idx, score in enumerate(scores):
            pair = self.pairs[idx]
            # Repository isolation
            if query.repository and pair.repository != query.repository:
                continue
            # Self exclusion
            if query.exclude_commit_sha and pair.commit_sha == query.exclude_commit_sha:
                continue
            scored_pairs.append((float(score), pair, self.corpus_texts[idx]))

        scored_pairs.sort(key=lambda x: x[0], reverse=True)

        matches = []
        max_score = scored_pairs[0][0] if scored_pairs and scored_pairs[0][0] > 0 else 1.0
        for score, pair, text in scored_pairs[:top_k]:
            normalized_score = min(1.0, score / max_score) if max_score > 0 else 0.0
            matches.append(
                RetrievedMatch(
                    similarity_score=round(normalized_score, 4),
                    repository=pair.repository,
                    commit_sha=pair.commit_sha,
                    parent_sha=pair.parent_sha,
                    file_path=pair.file_path,
                    function_name=pair.function_name,
                    class_name=pair.class_name,
                    commit_message=pair.commit_message,
                    buggy_code=pair.buggy_code,
                    fixed_code=pair.fixed_code,
                    diff=pair.diff,
                    detector_score=pair.detector_score,
                    detector_classification=pair.detector_classification,
                    timestamp=pair.timestamp,
                    matched_text=text,
                )
            )
        return matches


def is_relevant_match(query_pair: BugFixPair, candidate: RetrievedMatch) -> bool:
    """
    Ground-truth relevance definition:
    A match is considered relevant if:
    1. It targets the exact same function/method name (function recurrence), OR
    2. It targets the same class/module family with identical function basename.
    And it originated from a different commit (leakage prevented).
    """
    if candidate.commit_sha == query_pair.commit_sha:
        return False

    # Exact function match
    if candidate.function_name == query_pair.function_name:
        return True

    # Same enclosing class and file
    if (
        candidate.class_name
        and query_pair.class_name
        and candidate.class_name == query_pair.class_name
        and candidate.file_path == query_pair.file_path
    ):
        return True

    return False


def evaluate_retrieval_system(
    retriever,
    pairs: List[BugFixPair],
    method_name: str,
    representation_type: str,
    top_k: int = 10,
) -> EvaluationResult:
    """
    Evaluate retrieval system on the pairs dataset where recurrent functions exist.
    """
    # Find functions that occur more than once (so there is at least one true relevant target)
    func_occurrences: Dict[str, List[BugFixPair]] = {}
    for p in pairs:
        func_occurrences.setdefault(p.function_name, []).append(p)

    eval_queries = [p for p in pairs if len(func_occurrences[p.function_name]) > 1]
    if not eval_queries:
        # If no recurrent functions, evaluate against all pairs using module-level relevance
        eval_queries = pairs

    # Pre-generate query representations
    query_texts = [
        RepresentationBuilder.build_query_representation(
            code=qp.buggy_code,
            representation_type=representation_type,
            language=qp.language,
            file_path=qp.file_path,
            function_name=qp.function_name,
            class_name=qp.class_name,
        )
        for qp in eval_queries
    ]

    is_vector_retriever = hasattr(retriever, "embedder") and hasattr(retriever, "retrieve_vector")
    query_vectors = retriever.embedder.embed_batch(query_texts) if is_vector_retriever else None

    r1_hits = 0
    r3_hits = 0
    r5_hits = 0
    r10_hits = 0
    reciprocal_ranks = []
    latencies = []

    for idx, query_pair in enumerate(eval_queries):
        t0 = time.time()
        if is_vector_retriever and query_vectors:
            matches = retriever.retrieve_vector(
                query_vector=query_vectors[idx],
                top_k=top_k,
                repository=query_pair.repository,
                language=query_pair.language,
                exclude_commit_sha=query_pair.commit_sha,
            )
        else:
            query = RetrievalQuery(
                code=query_pair.buggy_code,
                repository=query_pair.repository,
                language=query_pair.language,
                file_path=query_pair.file_path,
                function_name=query_pair.function_name,
                class_name=query_pair.class_name,
                exclude_commit_sha=query_pair.commit_sha,
                representation_type=representation_type,
            )
            matches = retriever.retrieve(query, top_k=top_k)
        latencies.append((time.time() - t0) * 1000.0)

        # Check rankings
        hit_rank = None
        for rank, match in enumerate(matches, 1):
            if is_relevant_match(query_pair, match):
                hit_rank = rank
                break

        if hit_rank is not None:
            if hit_rank <= 1:
                r1_hits += 1
            if hit_rank <= 3:
                r3_hits += 1
            if hit_rank <= 5:
                r5_hits += 1
            if hit_rank <= 10:
                r10_hits += 1
            reciprocal_ranks.append(1.0 / hit_rank)
        else:
            reciprocal_ranks.append(0.0)

    total = len(eval_queries)
    return EvaluationResult(
        method_name=method_name,
        representation=representation_type,
        total_queries=total,
        recall_at_1=round((r1_hits / total) * 100, 2) if total else 0.0,
        recall_at_3=round((r3_hits / total) * 100, 2) if total else 0.0,
        recall_at_5=round((r5_hits / total) * 100, 2) if total else 0.0,
        recall_at_10=round((r10_hits / total) * 100, 2) if total else 0.0,
        mrr=round(sum(reciprocal_ranks) / total, 4) if total else 0.0,
        avg_latency_ms=round(sum(latencies) / total, 2) if total else 0.0,
    )


def run_representation_experiment(
    pairs_file: Path,
    vector_store: Optional[BaseVectorStore] = None,
    embedder: Optional[BaseEmbedder] = None,
    collection_prefix: str = "codeghost_exp",
) -> List[EvaluationResult]:
    """
    Run experiments comparing Lexical vs Embedding across Representations A, B, and C.
    """
    from backend.retrieval.indexer import BugFixIndexer

    # Load dataset
    pairs: List[BugFixPair] = []
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(BugFixPair.model_validate_json(line))

    embedder = embedder or FastEmbedEmbedder()
    v_store = vector_store or QdrantVectorStore()
    results: List[EvaluationResult] = []

    # 1. Lexical Baseline
    bm25 = BM25LexicalBaseline(pairs, representation_type="code_only")
    lex_res = evaluate_retrieval_system(
        retriever=bm25,
        pairs=pairs,
        method_name="BM25 Lexical",
        representation_type="code_only",
    )
    results.append(lex_res)

    # 2. Embedding across Representations A, B, C
    reps = [
        ("code_only", "Embedding (Code Only)"),
        ("code_and_metadata", "Embedding (Code + Meta)"),
        ("code_and_context", "Embedding (Code + Context)"),
    ]

    for rep_type, label in reps:
        coll_name = f"{collection_prefix}_{rep_type}"
        indexer = BugFixIndexer(
            vector_store=v_store,
            embedder=embedder,
            collection_name=coll_name,
        )
        indexer.index_file(
            jsonl_path=pairs_file,
            representation_type=rep_type,
            recreate_collection=True,
        )

        retriever = FunctionRetriever(
            vector_store=v_store,
            embedder=embedder,
            collection_name=coll_name,
        )

        emb_res = evaluate_retrieval_system(
            retriever=retriever,
            pairs=pairs,
            method_name=label,
            representation_type=rep_type,
        )
        results.append(emb_res)

    return results


def evaluate_retrieval(
    pairs_file: Path,
    collection_name: str = "codeghost_bug_history",
    vector_store: Optional[BaseVectorStore] = None,
    embedder: Optional[BaseEmbedder] = None,
) -> Tuple[EvaluationResult, EvaluationResult]:
    """
    Compare the primary BM25 Lexical baseline against the primary BGE Embedding retriever.
    """
    pairs: List[BugFixPair] = []
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(BugFixPair.model_validate_json(line))

    # Baseline A: Lexical
    bm25 = BM25LexicalBaseline(pairs, representation_type="code_and_metadata")
    lexical_res = evaluate_retrieval_system(
        retriever=bm25,
        pairs=pairs,
        method_name="BM25 Lexical Baseline",
        representation_type="code_and_metadata",
    )

    # Baseline B: Vector Store
    v_store = vector_store or QdrantVectorStore()
    emb = embedder or FastEmbedEmbedder()
    retriever = FunctionRetriever(
        vector_store=v_store,
        embedder=emb,
        collection_name=collection_name,
    )
    embedding_res = evaluate_retrieval_system(
        retriever=retriever,
        pairs=pairs,
        method_name=f"Embedding ({emb.model_name})",
        representation_type="code_and_metadata",
    )

    return lexical_res, embedding_res
