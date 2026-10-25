"""Unit tests for embedder models and code representation builder."""

import pytest
from backend.models.schemas import BugFixPair
from backend.retrieval.embedder import MockEmbedder, RepresentationBuilder
from backend.retrieval.schemas import RepresentationType


@pytest.fixture
def mock_embedder():
    return MockEmbedder(dimension=128)


def test_mock_embedder_dimension_and_normalization(mock_embedder: MockEmbedder):
    text = "def calculate_sum(a, b):\n    return a + b"
    vec = mock_embedder.embed_code(text)

    assert len(vec) == 128
    # Verify unit-normalized vector
    norm = sum(x * x for x in vec)
    assert abs(norm - 1.0) < 1e-4


def test_mock_embedder_batch(mock_embedder: MockEmbedder):
    texts = [
        "def func_a(): pass",
        "def func_b(): return 1",
        "def func_c(): return 2",
    ]
    vectors = mock_embedder.embed_batch(texts)
    assert len(vectors) == 3
    for v in vectors:
        assert len(v) == 128


def test_representation_builder_code_only():
    code = "def sample():\n    return 42"
    rep = RepresentationBuilder.build_query_representation(
        code=code,
        representation_type=RepresentationType.CODE_ONLY,
        language="python",
        file_path="app.py",
    )
    assert rep == code


def test_representation_builder_code_and_metadata():
    code = "def sample():\n    return 42"
    rep = RepresentationBuilder.build_query_representation(
        code=code,
        representation_type=RepresentationType.CODE_AND_METADATA,
        language="python",
        file_path="src/app.py",
        class_name="Server",
        function_name="Server.sample",
    )
    assert "Language: python" in rep
    assert "File: src/app.py" in rep
    assert "Class: Server" in rep
    assert "Function: Server.sample" in rep
    assert "Code:\ndef sample():\n    return 42" in rep


def test_representation_builder_code_and_context():
    pair = BugFixPair(
        repository="test_repo",
        commit_sha="sha_fix_1",
        parent_sha="sha_bug_1",
        commit_message="Fix memory leak in buffer pool",
        author="Dev",
        timestamp="2026-01-01T00:00:00Z",
        file_path="pool.py",
        language="python",
        code_unit_type="function",
        class_name=None,
        function_name="acquire_buffer",
        buggy_code="def acquire_buffer(): return None",
        fixed_code="def acquire_buffer(): return []",
        diff="--- a\n+++ b",
        detector_score=0.75,
        detector_signals=["fix_keyword:'Fix'"],
        detector_classification="BUG_FIX_CANDIDATE",
    )

    rep = RepresentationBuilder.build_pair_representation(
        pair=pair,
        representation_type=RepresentationType.CODE_AND_CONTEXT,
        target_version="buggy",
    )

    assert "Language: python" in rep
    assert "File: pool.py" in rep
    assert "Function: acquire_buffer" in rep
    assert "Bug Context: Fix memory leak in buffer pool" in rep
    assert "Buggy Code:\ndef acquire_buffer(): return None" in rep
