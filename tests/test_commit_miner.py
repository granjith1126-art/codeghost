"""Tests for CommitMiner: Git commit history, diffs, merge commits, and file changes."""

from pathlib import Path
import git
import pytest

from backend.history.commit_miner import CommitMiner


@pytest.fixture
def mock_repo(tmp_path: Path):
    """Create a temporary local git repository with multiple commits including renames and merges."""
    repo_dir = tmp_path / "test_git_repo"
    repo_dir.mkdir()
    repo = git.Repo.init(repo_dir)

    author = git.Actor("Test Committer", "committer@example.com")

    # Commit 1: Initial commit
    f1 = repo_dir / "app.py"
    f1.write_text("def hello():\n    return 'hello'\n", encoding="utf-8")
    repo.index.add(["app.py"])
    c1 = repo.index.commit("Initial commit", author=author, committer=author)

    # Commit 2: Modify app.py and add helper.py
    f1.write_text("def hello():\n    return 'hello world'\n", encoding="utf-8")
    f2 = repo_dir / "helper.py"
    f2.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    repo.index.add(["app.py", "helper.py"])
    c2 = repo.index.commit("Fix typo in hello and add helper", author=author, committer=author)

    # Commit 3: Rename helper.py -> utils.py
    f2.unlink()
    f3 = repo_dir / "utils.py"
    f3.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    repo.index.remove(["helper.py"])
    repo.index.add(["utils.py"])
    c3 = repo.index.commit("Rename helper.py to utils.py", author=author, committer=author)

    # Commit 4: Create a branch, make a change, and merge back
    main_branch = repo.active_branch
    feature_branch = repo.create_head("feature-branch")
    feature_branch.checkout()

    f4 = repo_dir / "extra.py"
    f4.write_text("EXTRA = 100\n", encoding="utf-8")
    repo.index.add(["extra.py"])
    repo.index.commit("Add extra module in branch", author=author, committer=author)

    main_branch.checkout()
    f1.write_text("def hello():\n    # greeting\n    return 'hello world'\n", encoding="utf-8")
    repo.index.add(["app.py"])
    repo.index.commit("Add comment to app.py on main", author=author, committer=author)

    # Merge
    repo.git.merge("feature-branch", "--no-ff", "-m", "Merge branch 'feature-branch'")

    return repo_dir


def test_commit_miner_iteration(mock_repo: Path):
    miner = CommitMiner(mock_repo)
    commits = list(miner.mine_commits("test_repo", max_commits=10))

    assert len(commits) >= 4

    # Latest commit should be the merge commit
    latest = commits[0]
    assert latest.is_merge is True
    assert len(latest.parent_shas) == 2


def test_parse_hunk_ranges():
    diff_text = """@@ -10,5 +10,7 @@
-old line
+new line 1
+new line 2
@@ -30 +32,3 @@
"""
    old_ranges, new_ranges, additions, deletions = CommitMiner.parse_hunk_ranges(diff_text)

    assert len(old_ranges) == 2
    assert old_ranges[0] == (10, 14)
    assert new_ranges[0] == (10, 16)
    assert old_ranges[1] == (30, 30)
    assert new_ranges[1] == (32, 34)
    assert additions == 2
    assert deletions == 1


def test_get_file_content(mock_repo: Path):
    miner = CommitMiner(mock_repo)
    head_commit = miner.repo.head.commit

    content = miner.get_file_content(head_commit, "app.py")
    assert content is not None
    assert "def hello():" in content

    # Non-existent file
    missing = miner.get_file_content(head_commit, "non_existent.py")
    assert missing is None
