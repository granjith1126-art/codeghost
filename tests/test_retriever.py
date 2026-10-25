"""End-to-end tests for BugFixIndexer, FunctionRetriever, and BM25 baseline."""

import json
from pathlib import Path
import pytest

from backend.models.schemas import BugFixPair, CodeUnit, CodeUnitType
from backend.retrieval.embedder import MockEmbedder
from backend.retrieval.indexer import BugFixIndexer
from backend.retrieval.retriever import FunctionRetriever
from backend.retrieval.schemas import RepresentationType, RetrievalQuery
from backend.retrieval.vector_store import QdrantVectorStore
from evaluation.retrieval_eval import BM25LexicalBaseline


@pytest.fixture
def mock_dataset(tmp_path: Path):
    dataset_file = tmp_path / "test_pairs.jsonl"
    pairs = [
        BugFixPair(
            repository="test_org/test_repo",
            commit_sha="commit_01",
            parent_sha="parent_01",
            commit_message="Fix IPv6 address parsing bug",
            author="Author 1",
            timestamp="2026-01-01T12:00:00Z",
            file_path="src/network.py",
            language="python",
            code_unit_type="function",
            class_name=None,
            function_name="parse_ip",
            buggy_code="def parse_ip(host):\n    return host.split(':')[0]",
            fixed_code="def parse_ip(host):\n    return urlsplit(f'//{host}').hostname",
            diff="--- a\n+++ b",
            detector_score=0.75,
            detector_signals=["fix_keyword:'Fix'"],
            detector_classification="BUG_FIX_CANDIDATE",
        ),
        BugFixPair(
            repository="test_org/test_repo",
            commit_sha="commit_02",
            parent_sha="parent_02",
            commit_message="Fix database connection pool timeout crash",
            author="Author 2",
            timestamp="2026-01-02T12:00:00Z",
            file_path="src/db.py",
            language="python",
            code_unit_type="method",
            class_name="DatabasePool",
            function_name="DatabasePool.connect",
            buggy_code="def connect(self):\n    return self.pool.get(timeout=0)",
            fixed_code="def connect(self):\n    return self.pool.get(timeout=30)",
            diff="--- a\n+++ b",
            detector_score=0.85,
            detector_signals=["fix_keyword:'Fix'", "crash_keyword:'crash'"],
            detector_classification="BUG_FIX_CANDIDATE",
        ),
    ]

    with open(dataset_file, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(p.model_dump_json() + "\n")

    return dataset_file, pairs


def test_indexer_and_retriever_pipeline(mock_dataset):
    dataset_file, original_pairs = mock_dataset
    vector_store = QdrantVectorStore(storage_path=":memory:")
    embedder = MockEmbedder(dimension=64)

    indexer = BugFixIndexer(
        vector_store=vector_store,
        embedder=embedder,
        collection_name="test_pipeline_collection",
    )

    stats = indexer.index_file(
        jsonl_path=dataset_file,
        representation_type=RepresentationType.CODE_AND_METADATA,
        recreate_collection=True,
    )

    assert stats.dataset_records == 2
    assert stats.indexed == 2
    assert stats.failed == 0

    retriever = FunctionRetriever(
        vector_store=vector_store,
        embedder=embedder,
        collection_name="test_pipeline_collection",
    )

    # Query for IP parsing function
    query = RetrievalQuery(
        code="def parse_ip(host):\n    return host.split(':')[0]",
        repository="test_org/test_repo",
        language="python",
        file_path="src/network.py",
        function_name="parse_ip",
    )

    matches = retriever.retrieve(query=query, top_k=2)
    assert len(matches) == 2
    top_match = matches[0]
    assert top_match.function_name == "parse_ip"
    assert "urlsplit" in top_match.fixed_code


def test_retriever_from_code_unit(mock_dataset):
    dataset_file, _ = mock_dataset
    vector_store = QdrantVectorStore(storage_path=":memory:")
    embedder = MockEmbedder(dimension=64)

    indexer = BugFixIndexer(vector_store=vector_store, embedder=embedder, collection_name="unit_test_coll")
    indexer.index_file(dataset_file)

    retriever = FunctionRetriever(vector_store=vector_store, embedder=embedder, collection_name="unit_test_coll")

    code_unit = CodeUnit(
        repository="test_org/test_repo",
        commit_hash="new_commit_sha",
        file_path="src/db.py",
        unit_type=CodeUnitType.METHOD,
        name="DatabasePool.connect",
        start_line=10,
        end_line=20,
        source_code="def connect(self):\n    return self.pool.get(timeout=0)",
        language="python",
        parent_name="DatabasePool",
    )

    matches = retriever.retrieve_for_code_unit(code_unit, top_k=1)
    assert len(matches) == 1
    assert matches[0].function_name == "DatabasePool.connect"


def test_bm25_baseline(mock_dataset):
    _, pairs = mock_dataset
    bm25 = BM25LexicalBaseline(pairs=pairs)

    query = RetrievalQuery(
        code="def parse_ip(host): host.split(':')",
        repository="test_org/test_repo",
        function_name="parse_ip",
    )

    matches = bm25.retrieve(query=query, top_k=1)
    assert len(matches) == 1
    assert matches[0].function_name == "parse_ip"


def test_real_bge_embedding_and_retrieval(mock_dataset):
    """Integration test with the real BAAI/bge-base-en-v1.5 model."""
    dataset_file, _ = mock_dataset
    from backend.retrieval.embedder import FastEmbedEmbedder

    bge_embedder = FastEmbedEmbedder(model_name="BAAI/bge-base-en-v1.5")
    assert bge_embedder.dimension == 768

    vector_store = QdrantVectorStore(storage_path=":memory:")
    indexer = BugFixIndexer(
        vector_store=vector_store,
        embedder=bge_embedder,
        collection_name="bge_real_test",
    )
    stats = indexer.index_file(dataset_file, recreate_collection=True)
    assert stats.indexed == 2
    assert stats.vector_dimension == 768

    retriever = FunctionRetriever(
        vector_store=vector_store,
        embedder=bge_embedder,
        collection_name="bge_real_test",
    )

    query = RetrievalQuery(
        code="def parse_ip(host):\n    return host.split(':')[0]",
        repository="test_org/test_repo",
        function_name="parse_ip",
    )
    matches = retriever.retrieve(query, top_k=1)
    assert len(matches) == 1
    assert matches[0].function_name == "parse_ip"
    assert matches[0].similarity_score > 0.5

