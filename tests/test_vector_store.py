"""Unit tests for QdrantVectorStore in-memory functionality and filtering."""

import pytest
from backend.retrieval.vector_store import QdrantVectorStore


@pytest.fixture
def memory_store():
    store = QdrantVectorStore(storage_path=":memory:")
    return store


def test_collection_creation_and_insertion(memory_store: QdrantVectorStore):
    collection = "test_collection"
    dim = 4

    memory_store.create_collection(collection, vector_size=dim, recreate=True)
    assert memory_store.collection_exists(collection)

    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    payloads = [
        {
            "repository": "repo_a",
            "commit_sha": "c1",
            "file_path": "a.py",
            "function_name": "func_a",
            "commit_message": "Fix bug 1",
            "buggy_code": "def a(): pass",
            "fixed_code": "def a(): return 1",
            "diff": "...",
            "detector_score": 0.7,
            "detector_classification": "BUG_FIX_CANDIDATE",
            "timestamp": "2026-01-01T00:00:00Z",
        },
        {
            "repository": "repo_a",
            "commit_sha": "c2",
            "file_path": "b.py",
            "function_name": "func_b",
            "commit_message": "Fix bug 2",
            "buggy_code": "def b(): pass",
            "fixed_code": "def b(): return 2",
            "diff": "...",
            "detector_score": 0.8,
            "detector_classification": "BUG_FIX_CANDIDATE",
            "timestamp": "2026-01-02T00:00:00Z",
        },
        {
            "repository": "repo_b",
            "commit_sha": "c3",
            "file_path": "c.py",
            "function_name": "func_c",
            "commit_message": "Fix bug 3",
            "buggy_code": "def c(): pass",
            "fixed_code": "def c(): return 3",
            "diff": "...",
            "detector_score": 0.9,
            "detector_classification": "BUG_FIX_CANDIDATE",
            "timestamp": "2026-01-03T00:00:00Z",
        },
    ]
    ids = [
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
        "33333333-3333-3333-3333-333333333333",
    ]

    inserted = memory_store.insert_batch(collection, vectors, payloads, ids)
    assert inserted == 3
    assert memory_store.count(collection) == 3


def test_repository_isolation_filtering(memory_store: QdrantVectorStore):
    collection = "isolation_test"
    dim = 2
    memory_store.create_collection(collection, vector_size=dim, recreate=True)

    vectors = [[1.0, 0.0], [1.0, 0.0]]
    payloads = [
        {"repository": "repo_alpha", "commit_sha": "sha_1", "function_name": "alpha_fn"},
        {"repository": "repo_beta", "commit_sha": "sha_2", "function_name": "beta_fn"},
    ]
    ids = [
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    ]
    memory_store.insert_batch(collection, vectors, payloads, ids)

    # Query with repo_alpha filter
    results_alpha = memory_store.search(
        collection_name=collection,
        query_vector=[1.0, 0.0],
        repository="repo_alpha",
    )
    assert len(results_alpha) == 1
    assert results_alpha[0].repository == "repo_alpha"
    assert results_alpha[0].function_name == "alpha_fn"

    # Query with repo_beta filter
    results_beta = memory_store.search(
        collection_name=collection,
        query_vector=[1.0, 0.0],
        repository="repo_beta",
    )
    assert len(results_beta) == 1
    assert results_beta[0].repository == "repo_beta"


def test_leakage_exclusion_filter(memory_store: QdrantVectorStore):
    collection = "leakage_test"
    dim = 2
    memory_store.create_collection(collection, vector_size=dim, recreate=True)

    vectors = [[1.0, 0.0], [0.9, 0.1]]
    payloads = [
        {"repository": "repo_x", "commit_sha": "target_commit", "function_name": "same_fn"},
        {"repository": "repo_x", "commit_sha": "other_commit", "function_name": "prior_fn"},
    ]
    ids = [
        "11111111-0000-0000-0000-000000000000",
        "22222222-0000-0000-0000-000000000000",
    ]
    memory_store.insert_batch(collection, vectors, payloads, ids)

    # Exclude target_commit to simulate testing
    results = memory_store.search(
        collection_name=collection,
        query_vector=[1.0, 0.0],
        repository="repo_x",
        exclude_commit_sha="target_commit",
    )
    assert len(results) == 1
    assert results[0].commit_sha == "other_commit"
