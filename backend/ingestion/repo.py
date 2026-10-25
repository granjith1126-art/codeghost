"""Repository cloning, validation, and metadata extraction."""

import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import git
from git.exc import GitCommandError, InvalidGitRepositoryError, NoSuchPathError

logger = logging.getLogger(__name__)


@dataclass
class RepoContext:
    """Metadata and local context for a managed git repository."""
    repo_url: str
    repo_name: str
    local_path: Path
    commit_hash: str
    branch: Optional[str] = None


class RepositoryManager:
    """Handles cloning, caching, validation, and metadata extraction for git repositories."""

    def __init__(self, base_storage_dir: Optional[Path] = None):
        if base_storage_dir is None:
            self.base_storage_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
        else:
            self.base_storage_dir = Path(base_storage_dir)
        self.base_storage_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def extract_repo_name(repo_url: str) -> str:
        """Extract a sanitized repository name (e.g., 'pallets/flask' or 'flask') from a URL or path."""
        # Handle local filesystem paths
        if os.path.exists(repo_url) or repo_url.startswith("file://"):
            clean_path = repo_url.replace("file://", "").rstrip("/\\")
            return Path(clean_path).name or "local_repo"

        # Handle SSH style git@ URLs e.g. git@github.com:pallets/flask.git
        if "@" in repo_url and ":" in repo_url:
            repo_url = repo_url.split(":", 1)[1]

        # Handle git/HTTP URLs
        cleaned = re.sub(r"\.git$", "", repo_url.strip())
        parsed = urlparse(cleaned)
        path = (parsed.path or parsed.netloc).strip("/")
        if path:
            return path.replace("/", "_")
        return "unknown_repo"

    def clone_or_update(
        self,
        repo_url_or_path: str,
        target_dir: Optional[Path] = None,
        branch: Optional[str] = None,
        depth: Optional[int] = None,
        force_clean: bool = False,
    ) -> RepoContext:
        """
        Clone a remote git repository or validate an existing local repository.

        Args:
            repo_url_or_path: Public git repository URL or local folder path.
            target_dir: Destination directory. If None, clones to base_storage_dir / repo_name.
            branch: Specific branch or tag to checkout.
            depth: Shallow clone depth (e.g. 1 for faster download).
            force_clean: If True, removes existing local destination before cloning.

        Returns:
            RepoContext containing repo details and HEAD commit hash.

        Raises:
            ValueError: If URL/path is invalid or cloning fails validation.
            GitCommandError: If git command fails.
        """
        repo_name = self.extract_repo_name(repo_url_or_path)

        # Check if input is an existing local directory
        is_local_dir = os.path.isdir(repo_url_or_path)

        if is_local_dir:
            local_path = Path(repo_url_or_path).resolve()
            logger.info("Using local repository directory: %s", local_path)
            try:
                repo = git.Repo(local_path)
            except (InvalidGitRepositoryError, NoSuchPathError) as e:
                logger.error("Directory %s is not a valid git repository: %s", local_path, e)
                raise ValueError(f"Directory {local_path} is not a valid git repository") from e

            commit_hash = repo.head.commit.hexsha
            active_branch = repo.active_branch.name if not repo.head.is_detached else None
            return RepoContext(
                repo_url=str(local_path),
                repo_name=repo_name,
                local_path=local_path,
                commit_hash=commit_hash,
                branch=active_branch,
            )

        # Handle remote git clone
        if target_dir is None:
            local_path = (self.base_storage_dir / repo_name).resolve()
        else:
            local_path = Path(target_dir).resolve()

        if force_clean and local_path.exists():
            logger.warning("Force cleaning existing directory: %s", local_path)
            shutil.rmtree(local_path, ignore_errors=True)

        repo: Optional[git.Repo] = None

        # Check if already cloned
        if local_path.exists() and (local_path / ".git").exists():
            try:
                repo = git.Repo(local_path)
                logger.info("Reusing existing cloned repository at: %s", local_path)
                # Attempt a fast-forward fetch / pull if online, otherwise use local head
                try:
                    origin = repo.remotes.origin
                    origin.fetch()
                except Exception as fetch_err:
                    logger.warning("Remote fetch skipped (%s), using local HEAD", fetch_err)
            except Exception as e:
                logger.warning("Existing directory corrupt (%s), re-cloning...", e)
                shutil.rmtree(local_path, ignore_errors=True)
                repo = None

        if repo is None:
            logger.info("Cloning repository '%s' into '%s'...", repo_url_or_path, local_path)
            clone_kwargs = {}
            if depth:
                clone_kwargs["depth"] = depth
            if branch:
                clone_kwargs["branch"] = branch

            try:
                repo = git.Repo.clone_from(
                    url=repo_url_or_path,
                    to_path=local_path,
                    **clone_kwargs,
                )
            except GitCommandError as e:
                logger.error("Git clone failed for %s: %s", repo_url_or_path, e)
                raise ValueError(f"Failed to clone repository '{repo_url_or_path}': {e.stderr or e}") from e

        # Validate git repository state
        try:
            commit_hash = repo.head.commit.hexsha
            logger.info("Validated repository. HEAD commit: %s", commit_hash)
        except Exception as e:
            logger.error("Failed to read HEAD commit from repository %s: %s", local_path, e)
            raise ValueError(f"Corrupt or empty repository at {local_path}: {e}") from e

        active_branch = None
        try:
            if not repo.head.is_detached:
                active_branch = repo.active_branch.name
        except TypeError:
            active_branch = None

        return RepoContext(
            repo_url=repo_url_or_path,
            repo_name=repo_name,
            local_path=local_path,
            commit_hash=commit_hash,
            branch=active_branch,
        )
