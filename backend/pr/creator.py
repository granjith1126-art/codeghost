"""GitHub Pull Request Creation and Dispatch Engine for CodeGhost (Milestone 7).

Manages git branch generation, patch artifact serialization, and GitHub REST API
dispatch with dry-run support for human review.
"""

import json
import os
import re
import urllib.error
import urllib.request
from typing import Optional

from backend.autofix.schemas import VerifiedFix
from backend.pr.formatter import PRFormatter
from backend.pr.schemas import PRDraft, PRResult
from backend.reasoning.schemas import RiskExplanation


class PRCreator:
    """Manages the creation and submission of Pull Requests for verified bug fixes."""

    def __init__(self, formatter: Optional[PRFormatter] = None):
        self.formatter = formatter or PRFormatter()

    def create_pr(
        self,
        verified_fix: VerifiedFix,
        explanation: RiskExplanation,
        repo_url_or_path: str = "https://github.com/pallets/flask",
        base_branch: str = "main",
        dry_run: bool = True,
        github_token: Optional[str] = None,
        patches_dir: str = "backend/data/patches",
    ) -> PRResult:
        """Create a pull request draft or publish live to GitHub.

        Args:
            verified_fix: VerifiedFix candidate.
            explanation: RiskExplanation from Milestone 5.
            repo_url_or_path: GitHub repository URL or local repository path.
            base_branch: Base branch to merge into.
            dry_run: If True, serializes patch and preview without making external API calls.
            github_token: Optional GitHub Personal Access Token for live PR creation.
            patches_dir: Directory where dry-run patch artifacts are stored.

        Returns:
            PRResult with status, URL/number (if live), or patch path (if dry-run).
        """
        token = github_token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

        # Format PR draft
        draft = self.formatter.format_draft(
            verified_fix=verified_fix,
            explanation=explanation,
            repo_identifier=repo_url_or_path,
            base_branch=base_branch,
        )

        # Ensure patches directory exists
        os.makedirs(patches_dir, exist_ok=True)
        safe_branch_name = draft.branch_name.replace("/", "_")
        patch_file_path = os.path.join(patches_dir, f"{safe_branch_name}.patch")

        with open(patch_file_path, "w", encoding="utf-8") as f:
            f.write(f"# CodeGhost Pull Request Patch: {draft.title}\n")
            f.write(f"# Target Branch: {draft.base_branch}\n")
            f.write(f"# Risk Score: {draft.risk_score} ({draft.risk_category})\n\n")
            f.write(draft.diff)

        # If dry-run requested or no token provided, return dry-run result
        if dry_run or not token:
            return PRResult(
                success=True,
                pr_url=None,
                pr_number=None,
                branch_created=False,
                patch_file_path=patch_file_path,
                dry_run=True,
                message=(
                    f"Dry-run PR draft prepared successfully for `{verified_fix.candidate.function_name}`. "
                    f"Branch: `{draft.branch_name}`. Patch saved to `{patch_file_path}`."
                ),
                draft=draft,
            )

        # Remote GitHub API PR submission
        owner_repo = self._parse_owner_repo(repo_url_or_path)
        if not owner_repo:
            return PRResult(
                success=False,
                dry_run=False,
                patch_file_path=patch_file_path,
                message=f"Unable to parse GitHub owner/repository from: '{repo_url_or_path}'",
                draft=draft,
            )

        owner, repo = owner_repo
        api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "CodeGhost-PR-Agent",
            "Content-Type": "application/json",
        }

        payload = {
            "title": draft.title,
            "head": draft.branch_name,
            "base": draft.base_branch,
            "body": draft.body_markdown,
            "maintainer_can_modify": True,
        }

        try:
            req = urllib.request.Request(
                api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                pr_url = resp_data.get("html_url")
                pr_num = resp_data.get("number")
                return PRResult(
                    success=True,
                    pr_url=pr_url,
                    pr_number=pr_num,
                    branch_created=True,
                    patch_file_path=patch_file_path,
                    dry_run=False,
                    message=f"Pull Request #{pr_num} successfully opened on GitHub: {pr_url}",
                    draft=draft,
                )
        except urllib.error.HTTPError as err:
            err_body = err.read().decode("utf-8", errors="ignore")
            return PRResult(
                success=False,
                dry_run=False,
                patch_file_path=patch_file_path,
                message=f"GitHub API HTTP error {err.code}: {err.reason} - {err_body[:200]}",
                draft=draft,
            )
        except Exception as exc:
            return PRResult(
                success=False,
                dry_run=False,
                patch_file_path=patch_file_path,
                message=f"Failed to submit PR to GitHub: {str(exc)}",
                draft=draft,
            )

    def _parse_owner_repo(self, repo_url_or_path: str) -> Optional[tuple]:
        """Extract (owner, repo) tuple from GitHub URL or string."""
        clean = repo_url_or_path.strip().rstrip("/")
        if clean.endswith(".git"):
            clean = clean[:-4]

        # e.g. https://github.com/pallets/flask
        match = re.search(r"github\.com[/:]([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.-]+)", clean)
        if match:
            return match.group(1), match.group(2)

        # e.g. pallets/flask
        if "/" in clean and len(clean.split("/")) == 2:
            parts = clean.split("/")
            return parts[0], parts[1]

        return None
