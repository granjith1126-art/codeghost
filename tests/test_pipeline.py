"""Integration and CLI tests for CodeGhost."""

from pathlib import Path
import git
import pytest

from backend.main import main


def test_cli_execution_on_local_repo(tmp_path: Path):
    # Setup mock git repo
    repo_dir = tmp_path / "cli_test_repo"
    repo_dir.mkdir()
    repo = git.Repo.init(repo_dir)

    py_file = repo_dir / "core.py"
    py_file.write_text("def test_func():\n    return 42\n", encoding="utf-8")
    repo.index.add(["core.py"])
    author = git.Actor("CLI Tester", "cli@example.com")
    repo.index.commit("CLI commit", author=author, committer=author)

    output_dir = tmp_path / "cli_storage"
    exit_code = main([
        str(repo_dir),
        "--output-dir", str(output_dir),
        "--preview", "2",
    ])

    assert exit_code == 0
