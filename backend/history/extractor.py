"""Historical bug extraction orchestrator: mines Git history, detects bug-fix candidates, and pairs AST units."""

import json
import logging
from pathlib import Path
from typing import Callable, List, Optional

from backend.history.bug_detector import BugDetector
from backend.history.commit_miner import CommitMiner
from backend.history.function_matcher import FunctionMatcher
from backend.ingestion.repo import RepoContext, RepositoryManager
from backend.models.schemas import (
    BugFixClassification,
    BugFixPair,
    CommitRecord,
    HistoryExtractionSummary,
)

logger = logging.getLogger(__name__)


class HistoryExtractor:
    """Orchestrates commit mining, heuristic bug detection, and AST-level bug-fix extraction."""

    def __init__(
        self,
        base_storage_dir: Optional[Path] = None,
        processed_storage_dir: Optional[Path] = None,
        repo_manager: Optional[RepositoryManager] = None,
        bug_detector: Optional[BugDetector] = None,
        function_matcher: Optional[FunctionMatcher] = None,
    ):
        self.repo_manager = repo_manager or RepositoryManager(base_storage_dir=base_storage_dir)
        self.base_storage_dir = self.repo_manager.base_storage_dir

        if processed_storage_dir is None:
            self.processed_storage_dir = (
                Path(__file__).resolve().parent.parent / "data" / "processed"
            )
        else:
            self.processed_storage_dir = Path(processed_storage_dir).resolve()

        self.processed_storage_dir.mkdir(parents=True, exist_ok=True)
        self.base_storage_dir.mkdir(parents=True, exist_ok=True)

        self.bug_detector = bug_detector or BugDetector()
        self.function_matcher = function_matcher or FunctionMatcher()

    def extract_history(
        self,
        repo_url_or_path: str,
        max_commits: Optional[int] = 500,
        since: Optional[str] = None,
        until: Optional[str] = None,
        branch: Optional[str] = None,
        raw_commits_path: Optional[Path] = None,
        processed_pairs_path: Optional[Path] = None,
        process_uncertain: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> HistoryExtractionSummary:
        """
        Run historical bug extraction on a repository.

        Args:
            repo_url_or_path: Git remote URL or local folder path.
            max_commits: Maximum commits to scan.
            since: Lower date limit.
            until: Upper date limit.
            branch: Branch or rev to iterate from.
            raw_commits_path: Destination for raw commit JSONL.
            processed_pairs_path: Destination for BugFixPair JSONL.
            process_uncertain: If True, also attempts function extraction on UNCERTAIN commits.
            progress_callback: Optional callback(current, total, msg) for UI updates.

        Returns:
            HistoryExtractionSummary with metrics.
        """
        logger.info("Initializing historical bug extraction for: %s", repo_url_or_path)

        # 1. Obtain and validate repository
        repo_ctx: RepoContext = self.repo_manager.clone_or_update(
            repo_url_or_path=repo_url_or_path,
            branch=branch,
        )

        repo_root = repo_ctx.local_path.resolve()
        repo_name = repo_ctx.repo_name

        # 2. Setup output destinations
        if raw_commits_path is None:
            raw_commits_path = self.base_storage_dir / f"{repo_name}_commits.jsonl"
        else:
            raw_commits_path = Path(raw_commits_path).resolve()
        raw_commits_path.parent.mkdir(parents=True, exist_ok=True)

        if processed_pairs_path is None:
            processed_pairs_path = self.processed_storage_dir / "bug_fix_pairs.jsonl"
        else:
            processed_pairs_path = Path(processed_pairs_path).resolve()
        processed_pairs_path.parent.mkdir(parents=True, exist_ok=True)

        miner = CommitMiner(repo_root)

        commits_scanned = 0
        bug_candidates = 0
        uncertain_count = 0
        non_bug_count = 0
        files_changed_total = 0
        functions_changed_total = 0
        pairs_extracted = 0
        failures = 0

        # Open file handles for streaming output
        with open(raw_commits_path, "w", encoding="utf-8") as raw_f, open(
            processed_pairs_path, "a", encoding="utf-8"
        ) as proc_f:

            commit_stream = miner.mine_commits(
                repository_url=repo_ctx.repo_url,
                max_commits=max_commits,
                since=since,
                until=until,
                rev=branch or "HEAD",
            )

            for commit_record in commit_stream:
                commits_scanned += 1

                # Step A: Run heuristic bug detection on commit message
                evidence = self.bug_detector.analyze(commit_record.commit_message)
                commit_record.detection_result = evidence

                class_val = (
                    evidence.classification.value
                    if hasattr(evidence.classification, "value")
                    else str(evidence.classification)
                )

                if class_val == BugFixClassification.BUG_FIX_CANDIDATE.value:
                    bug_candidates += 1
                elif class_val == BugFixClassification.UNCERTAIN.value:
                    uncertain_count += 1
                else:
                    non_bug_count += 1

                # Save raw commit record
                raw_f.write(commit_record.model_dump_json() + "\n")

                files_changed_total += len(commit_record.changed_files)

                if progress_callback:
                    progress_callback(
                        commits_scanned,
                        max_commits or commits_scanned,
                        f"Commit {commit_record.commit_sha[:7]}: {class_val}",
                    )

                # Step B: If candidate (or uncertain when enabled), extract functions
                should_extract = (
                    class_val == BugFixClassification.BUG_FIX_CANDIDATE.value
                    or (process_uncertain and class_val == BugFixClassification.UNCERTAIN.value)
                )

                if not should_extract or not commit_record.parent_shas:
                    continue

                try:
                    primary_parent = miner.repo.commit(commit_record.parent_shas[0])
                    fix_commit = miner.repo.commit(commit_record.commit_sha)
                except Exception as e:
                    logger.warning("Could not load Git commit objects for %s: %s", commit_record.commit_sha, e)
                    failures += 1
                    continue

                for file_diff in commit_record.file_diffs:
                    if file_diff.is_binary or file_diff.change_type in ("A", "D"):
                        continue

                    old_path = file_diff.old_file_path or file_diff.file_path
                    new_path = file_diff.file_path

                    old_content = miner.get_file_content(primary_parent, old_path)
                    new_content = miner.get_file_content(fix_commit, new_path)

                    if not old_content or not new_content:
                        continue

                    try:
                        pairs = self.function_matcher.match_commit_changes(
                            commit_record=commit_record,
                            file_diff=file_diff,
                            old_content=old_content,
                            new_content=new_content,
                            detection_evidence=evidence,
                        )

                        for pair in pairs:
                            proc_f.write(pair.model_dump_json() + "\n")
                            pairs_extracted += 1
                            functions_changed_total += 1

                    except Exception as err:
                        logger.error(
                            "Error matching functions for file %s in commit %s: %s",
                            file_diff.file_path,
                            commit_record.commit_sha,
                            err,
                            exc_info=True,
                        )
                        failures += 1

        logger.info(
            "Historical extraction finished: %d commits scanned (%d bug candidates, %d uncertain), "
            "%d bug-fix pairs extracted (%d failures). Saved to %s and %s",
            commits_scanned,
            bug_candidates,
            uncertain_count,
            pairs_extracted,
            failures,
            raw_commits_path,
            processed_pairs_path,
        )

        return HistoryExtractionSummary(
            repository_url=repo_ctx.repo_url,
            commits_scanned=commits_scanned,
            bug_fix_candidates=bug_candidates,
            uncertain_commits=uncertain_count,
            non_bug_commits=non_bug_count,
            files_changed=files_changed_total,
            functions_changed=functions_changed_total,
            bug_fix_pairs_extracted=pairs_extracted,
            extraction_failures=failures,
            raw_commits_file=str(raw_commits_path),
            processed_pairs_file=str(processed_pairs_path),
        )
