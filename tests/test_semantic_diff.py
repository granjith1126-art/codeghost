"""Unit tests for ASTSemanticDiffAnalyzer (Milestone 9)."""

import pytest
from backend.history.semantic_diff import ASTSemanticDiffAnalyzer, SemanticDiffResult


@pytest.fixture
def analyzer():
    return ASTSemanticDiffAnalyzer()


def test_identical_code(analyzer):
    code = "def add(a, b):\n    return a + b\n"
    res = analyzer.analyze(code, code, language="python")
    assert res.is_semantic is False
    assert res.change_type == "IDENTICAL"


def test_formatting_only_change(analyzer):
    code_a = "def add(a, b):\n    return a + b\n"
    code_b = "def add(a,   b):\n\n    return  a + b\n"
    res = analyzer.analyze(code_a, code_b, language="python")
    assert res.is_semantic is False
    assert res.change_type == "FORMATTING_ONLY"


def test_comment_only_change_python(analyzer):
    code_a = "def compute(x):\n    return x * 2\n"
    code_b = "def compute(x):\n    # Multiplies input by two\n    return x * 2\n"
    res = analyzer.analyze(code_a, code_b, language="python")
    assert res.is_semantic is False
    assert res.change_type in ("COMMENT_ONLY", "FORMATTING_ONLY")


def test_docstring_only_change_python(analyzer):
    code_a = 'def helper():\n    """Old docstring."""\n    return 42\n'
    code_b = 'def helper():\n    """Updated comprehensive docstring."""\n    return 42\n'
    res = analyzer.analyze(code_a, code_b, language="python")
    assert res.is_semantic is False
    assert res.change_type == "DOCSTRING_ONLY"


def test_comment_only_change_javascript(analyzer):
    code_a = "function greet(name) {\n    return 'Hello ' + name;\n}"
    code_b = "function greet(name) {\n    // Added greeting comment\n    return 'Hello ' + name;\n}"
    res = analyzer.analyze(code_a, code_b, language="javascript")
    assert res.is_semantic is False
    assert res.change_type == "COMMENT_ONLY"


def test_semantic_logic_change_operator(analyzer):
    code_a = "def check(a, b):\n    return a > b\n"
    code_b = "def check(a, b):\n    return a >= b\n"
    res = analyzer.analyze(code_a, code_b, language="python")
    assert res.is_semantic is True
    assert res.change_type == "SEMANTIC_CODE_CHANGE"


def test_semantic_logic_change_guard_injection(analyzer):
    code_a = "def divide(a, b):\n    return a / b\n"
    code_b = "def divide(a, b):\n    if b == 0:\n        raise ZeroDivisionError('Division by zero')\n    return a / b\n"
    res = analyzer.analyze(code_a, code_b, language="python")
    assert res.is_semantic is True
    assert res.change_type == "SEMANTIC_CODE_CHANGE"
    assert res.ast_diff_summary is not None
    assert "if" in res.ast_diff_summary.lower() or "added" in res.ast_diff_summary.lower()


def test_semantic_change_in_go(analyzer):
    code_a = "func sum(a int, b int) int {\n    return a + b\n}"
    code_b = "func sum(a int, b int) int {\n    if a < 0 {\n        return 0\n    }\n    return a + b\n}"
    res = analyzer.analyze(code_a, code_b, language="go")
    assert res.is_semantic is True
    assert res.change_type == "SEMANTIC_CODE_CHANGE"


def test_history_extractor_filters_non_semantic(tmp_path):
    """Test HistoryExtractor filtering out non-semantic docstring fix commits."""
    import git
    from backend.history.extractor import HistoryExtractor

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    repo = git.Repo.init(repo_dir)

    file_p = repo_dir / "mod.py"
    with open(file_p, "w", encoding="utf-8") as f:
        f.write('def func():\n    """Initial docstring."""\n    return 100\n')
    repo.index.add(["mod.py"])
    repo.index.commit("Initial commit")

    # Non-semantic commit: fix docstring typo
    with open(file_p, "w", encoding="utf-8") as f:
        f.write('def func():\n    """Updated docstring with typo fix."""\n    return 100\n')
    repo.index.add(["mod.py"])
    repo.index.commit("fix: docstring typo in func")

    extractor = HistoryExtractor(base_storage_dir=tmp_path / "storage")
    summary = extractor.extract_history(
        repo_url_or_path=str(repo_dir),
        max_commits=5,
        filter_non_semantic=True,
    )

    assert summary.commits_scanned >= 2
    assert summary.non_semantic_pairs_filtered == 1
    assert summary.bug_fix_pairs_extracted == 0

