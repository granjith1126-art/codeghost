"""Commit history extraction and Git diff mining."""

import datetime
import logging
import re
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import git
from git.exc import GitCommandError

from backend.models.schemas import CommitFileDiff, CommitRecord

logger = logging.getLogger(__name__)

HUNK_HEADER_REGEX = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")


class CommitMiner:
    """Extracts commit history, file diffs, and modified line ranges using GitPython."""

    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path).resolve()
        try:
            self.repo = git.Repo(self.repo_path)
        except Exception as e:
            logger.error("Failed to load Git repository at %s: %s", self.repo_path, e)
            raise ValueError(f"Invalid Git repository at {self.repo_path}: {e}") from e

    @staticmethod
    def parse_hunk_ranges(diff_text: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], int, int]:
        """
        Parse a unified diff text to extract changed line ranges and addition/deletion counts.

        Returns:
            Tuple of:
            - old_line_ranges: list of (start_line, end_line) in the parent file
            - new_line_ranges: list of (start_line, end_line) in the commit file
            - additions: total lines added
            - deletions: total lines deleted
        """
        old_ranges: List[Tuple[int, int]] = []
        new_ranges: List[Tuple[int, int]] = []
        additions = 0
        deletions = 0

        for line in diff_text.splitlines():
            if line.startswith("@@"):
                match = HUNK_HEADER_REGEX.match(line)
                if match:
                    old_start = int(match.group(1))
                    old_count = int(match.group(2)) if match.group(2) is not None else 1
                    new_start = int(match.group(3))
                    new_count = int(match.group(4)) if match.group(4) is not None else 1

                    if old_count > 0:
                        old_ranges.append((old_start, old_start + old_count - 1))
                    else:
                        old_ranges.append((old_start, old_start))

                    if new_count > 0:
                        new_ranges.append((new_start, new_start + new_count - 1))
                    else:
                        new_ranges.append((new_start, new_start))
            elif line.startswith("+") and not line.startswith("+++"):
                additions += 1
            elif line.startswith("-") and not line.startswith("---"):
                deletions += 1

        return old_ranges, new_ranges, additions, deletions

    def get_file_content(self, commit: git.Commit, file_path: str) -> Optional[str]:
        """Safely retrieve the string content of a file at a specific commit."""
        try:
            blob = commit.tree[file_path]
            # Verify not binary
            if getattr(blob, "is_binary", False):
                return None
            data = blob.data_stream.read()
            # Check for null bytes (heuristic for binary)
            if b"\x00" in data[:8192]:
                return None
            return data.decode("utf-8", errors="replace")
        except (KeyError, IndexError, ValueError, OSError):
            return None

    def mine_commits(
        self,
        repository_url: str,
        max_commits: Optional[int] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        rev: str = "HEAD",
    ) -> Iterator[CommitRecord]:
        """
        Iterate over repository commits and extract diff details.

        Args:
            repository_url: Identifier or remote URL of the repository.
            max_commits: Maximum number of commits to scan.
            since: Date string for lower bound (e.g. '2023-01-01').
            until: Date string for upper bound.
            rev: Git revision/branch to start iteration from.

        Yields:
            CommitRecord objects for processed commits.
        """
        iter_kwargs = {"rev": rev}
        if max_commits is not None:
            iter_kwargs["max_count"] = max_commits
        if since is not None:
            iter_kwargs["since"] = since
        if until is not None:
            iter_kwargs["until"] = until

        logger.info(
            "Mining commits for '%s' (rev=%s, max_commits=%s, since=%s, until=%s)",
            repository_url,
            rev,
            max_commits,
            since,
            until,
        )

        try:
            commits = list(self.repo.iter_commits(**iter_kwargs))
        except GitCommandError as e:
            logger.error("Git error while reading commits: %s", e)
            return

        for commit in commits:
            parent_shas = [p.hexsha for p in commit.parents]
            is_merge = len(parent_shas) > 1

            dt = datetime.datetime.fromtimestamp(commit.committed_date, datetime.timezone.utc)
            timestamp_str = dt.isoformat()

            # Handle initial root commit (no parents)
            if not commit.parents:
                logger.debug("Skipping root commit %s (no parent to diff against)", commit.hexsha)
                yield CommitRecord(
                    repository=repository_url,
                    commit_sha=commit.hexsha,
                    parent_shas=[],
                    author_name=commit.author.name or "Unknown",
                    author_email=commit.author.email or "Unknown",
                    timestamp=timestamp_str,
                    commit_message=commit.message.strip(),
                    is_merge=False,
                    changed_files=[],
                    file_diffs=[],
                    total_additions=0,
                    total_deletions=0,
                )
                continue

            # For merge commits, compare against primary parent (mainline)
            primary_parent = commit.parents[0]

            file_diffs: List[CommitFileDiff] = []
            changed_files: List[str] = []
            total_additions = 0
            total_deletions = 0

            try:
                raw_diffs = primary_parent.diff(commit, create_patch=True)
            except Exception as e:
                logger.warning("Failed to compute diff for commit %s: %s", commit.hexsha, e)
                continue

            for d in raw_diffs:
                change_type = d.change_type or "M"  # 'A', 'D', 'R', 'M', etc.
                file_path = d.b_path or d.a_path
                old_file_path = d.a_path if change_type == "R" else None

                if not file_path:
                    continue

                changed_files.append(file_path)

                # Decode patch text
                diff_bytes = d.diff
                if isinstance(diff_bytes, bytes):
                    # Check for binary content
                    if b"\x00" in diff_bytes[:8192] or b"Binary files" in diff_bytes:
                        file_diffs.append(
                            CommitFileDiff(
                                file_path=file_path,
                                old_file_path=old_file_path,
                                change_type=change_type,
                                is_binary=True,
                            )
                        )
                        continue
                    diff_text = diff_bytes.decode("utf-8", errors="replace")
                elif isinstance(diff_bytes, str):
                    diff_text = diff_bytes
                else:
                    diff_text = ""

                old_ranges, new_ranges, adds, dels = self.parse_hunk_ranges(diff_text)
                total_additions += adds
                total_deletions += dels

                file_diffs.append(
                    CommitFileDiff(
                        file_path=file_path,
                        old_file_path=old_file_path,
                        change_type=change_type,
                        additions=adds,
                        deletions=dels,
                        diff_text=diff_text,
                        old_line_ranges=old_ranges,
                        new_line_ranges=new_ranges,
                        is_binary=False,
                    )
                )

            yield CommitRecord(
                repository=repository_url,
                commit_sha=commit.hexsha,
                parent_shas=parent_shas,
                author_name=commit.author.name or "Unknown",
                author_email=commit.author.email or "Unknown",
                timestamp=timestamp_str,
                commit_message=commit.message.strip(),
                is_merge=is_merge,
                changed_files=changed_files,
                file_diffs=file_diffs,
                total_additions=total_additions,
                total_deletions=total_deletions,
            )
