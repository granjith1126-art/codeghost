"""Tests for repository cloning and management."""

import os
from pathlib import Path
import git
import pytest

from backend.ingestion.repo import RepositoryManager, RepoContext


def test_extract_repo_name():
    assert RepositoryManager.extract_repo_name("https://github.com/pallets/flask") == "pallets_flask"
    assert RepositoryManager.extract_repo_name("https://github.com/pallets/flask.git") == "pallets_flask"
    assert RepositoryManager.extract_repo_name("git@github.com:pallets/flask.git") == "pallets_flask"
    assert RepositoryManager.extract_repo_name("https://github.com/torvalds/linux") == "torvalds_linux"


def test_local_repo_management(tmp_path: Path):
    # Create a mock git repository in a temp directory
    mock_repo_dir = tmp_path / "mock_repo"
    mock_repo_dir.mkdir()
    repo = git.Repo.init(mock_repo_dir)

    # Add a file and commit
    test_file = mock_repo_dir / "app.py"
    test_file.write_text("def main(): pass\n", encoding="utf-8")
    repo.index.add(["app.py"])
    author = git.Actor("Test Author", "author@example.com")
    commit = repo.index.commit("Initial commit", author=author, committer=author)

    manager = RepositoryManager(base_storage_dir=tmp_path / "storage")
    context = manager.clone_or_update(str(mock_repo_dir))

    assert context.repo_name == "mock_repo"
    assert context.local_path == mock_repo_dir.resolve()
    assert context.commit_hash == commit.hexsha
    assert context.branch in ("master", "main", None)


def test_invalid_repo_url(tmp_path: Path):
    manager = RepositoryManager(base_storage_dir=tmp_path / "storage")
    with pytest.raises(ValueError, match="Failed to clone repository|not a valid git repository"):
        manager.clone_or_update("https://github.com/nonexistent-org-12345/nonexistent-repo-99999.git")
