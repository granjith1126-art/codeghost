"""Tests for repository extractor and JSONL persistence."""

import json
from pathlib import Path
import git
import pytest

from backend.ingestion.extractor import RepositoryExtractor
from backend.models.schemas import CodeUnit, CodeUnitType


def test_extractor_end_to_end(tmp_path: Path):
    # Setup mock git repo
    repo_dir = tmp_path / "sample_project"
    repo_dir.mkdir()
    repo = git.Repo.init(repo_dir)

    # Create Python file
    py_file = repo_dir / "math_utils.py"
    py_file.write_text(
        '''
class Calculator:
    """A simple calculator class."""
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

def multiply(a: int, b: int) -> int:
    return a * b
''',
        encoding="utf-8",
    )

    # Create JavaScript file
    js_file = repo_dir / "web.js"
    js_file.write_text(
        '''
function handleRequest(req) {
    return true;
}
''',
        encoding="utf-8",
    )

    # Create unsupported text file
    txt_file = repo_dir / "notes.txt"
    txt_file.write_text("Some notes", encoding="utf-8")

    repo.index.add(["math_utils.py", "web.js", "notes.txt"])
    author = git.Actor("Test Author", "test@example.com")
    commit = repo.index.commit("Initial commit", author=author, committer=author)

    output_jsonl = tmp_path / "output.jsonl"
    storage_dir = tmp_path / "storage"

    extractor = RepositoryExtractor(base_storage_dir=storage_dir)
    result = extractor.ingest(
        repo_url_or_path=str(repo_dir),
        output_jsonl_path=output_jsonl,
    )

    assert result.commit_hash == commit.hexsha
    assert result.files_discovered == 3
    assert result.files_parsed == 2
    assert result.unsupported_files == 1
    assert result.classes_extracted == 1  # Calculator
    assert result.functions_extracted == 3  # Calculator.add, multiply, handleRequest
    assert len(result.errors) == 0

    # Verify JSONL content
    assert output_jsonl.exists()
    units = []
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            units.append(CodeUnit.model_validate_json(line))

    assert len(units) == 4
    names = [u.name for u in units]
    assert "Calculator" in names
    assert "Calculator.add" in names
    assert "multiply" in names
    assert "handleRequest" in names
