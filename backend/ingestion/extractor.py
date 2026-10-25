"""Repository extraction orchestrator: walks files, parses ASTs, and writes JSONL."""

import json
import logging
from pathlib import Path
from typing import Callable, List, Optional

from backend.ingestion.ast_parser import ASTParserManager, ASTParsingError
from backend.ingestion.language import LanguageDetector
from backend.ingestion.repo import RepoContext, RepositoryManager
from backend.models.schemas import (
    CodeUnit,
    CodeUnitType,
    IngestionError,
    IngestionResult,
)

logger = logging.getLogger(__name__)


class RepositoryExtractor:
    """Orchestrates the cloning, language scanning, AST parsing, and JSONL persistence."""

    def __init__(
        self,
        base_storage_dir: Optional[Path] = None,
        repo_manager: Optional[RepositoryManager] = None,
        language_detector: Optional[LanguageDetector] = None,
        ast_parser: Optional[ASTParserManager] = None,
    ):
        self.repo_manager = repo_manager or RepositoryManager(base_storage_dir=base_storage_dir)
        self.language_detector = language_detector or LanguageDetector()
        self.ast_parser = ast_parser or ASTParserManager()
        self.base_storage_dir = self.repo_manager.base_storage_dir

    def ingest(
        self,
        repo_url_or_path: str,
        output_jsonl_path: Optional[Path] = None,
        branch: Optional[str] = None,
        depth: Optional[int] = None,
        force_clean: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> IngestionResult:
        """
        Run the complete ingestion pipeline for a given repository.

        Args:
            repo_url_or_path: Git URL or local folder path.
            output_jsonl_path: Custom path for output JSONL. Defaults to raw/<repo_name>_units.jsonl.
            branch: Git branch/tag to checkout.
            depth: Shallow clone depth.
            force_clean: If True, re-clones repository even if cached.
            progress_callback: Optional callback(current, total, current_file) for progress tracking.

        Returns:
            IngestionResult summarizing the ingestion run.
        """
        logger.info("Starting ingestion for: %s", repo_url_or_path)

        # 1. Clone / validate repository
        repo_ctx: RepoContext = self.repo_manager.clone_or_update(
            repo_url_or_path=repo_url_or_path,
            branch=branch,
            depth=depth,
            force_clean=force_clean,
        )

        repo_root = repo_ctx.local_path.resolve()
        repo_name = repo_ctx.repo_name
        commit_hash = repo_ctx.commit_hash

        # 2. Determine output JSONL destination
        if output_jsonl_path is None:
            output_jsonl_path = self.base_storage_dir / f"{repo_name}_units.jsonl"
        else:
            output_jsonl_path = Path(output_jsonl_path).resolve()
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        # 3. Discover files and detect languages
        supported_files, unsupported_files, language_stats = self.language_detector.scan_repository(repo_root)

        total_files = len(supported_files)
        files_parsed = 0
        functions_extracted = 0
        classes_extracted = 0
        errors: List[IngestionError] = []

        logger.info(
            "Found %d supported files and %d unsupported files in %s",
            total_files,
            len(unsupported_files),
            repo_root,
        )

        # 4. Parse supported files and write to JSONL
        with open(output_jsonl_path, "w", encoding="utf-8") as jsonl_file:
            for idx, file_path in enumerate(supported_files, start=1):
                rel_path = file_path.relative_to(repo_root).as_posix()
                language = self.language_detector.detect_language(file_path)

                if progress_callback:
                    progress_callback(idx, total_files, rel_path)

                if not language:
                    continue

                # Read source code
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        source_code = f.read()
                except Exception as read_err:
                    err_msg = f"Failed to read file: {read_err}"
                    logger.error("Read error for %s: %s", rel_path, read_err)
                    errors.append(
                        IngestionError(
                            file_path=rel_path,
                            error_type="FileReadError",
                            message=err_msg,
                        )
                    )
                    continue

                # Parse AST and extract units
                try:
                    units: List[CodeUnit] = self.ast_parser.extract_units(
                        source_code=source_code,
                        language=language,
                        file_path=rel_path,
                        repository=repo_ctx.repo_url,
                        commit_hash=commit_hash,
                    )

                    for unit in units:
                        if unit.unit_type in (
                            CodeUnitType.FUNCTION,
                            CodeUnitType.ASYNC_FUNCTION,
                            CodeUnitType.METHOD,
                        ):
                            functions_extracted += 1
                        elif unit.unit_type == CodeUnitType.CLASS:
                            classes_extracted += 1

                        jsonl_file.write(unit.model_dump_json() + "\n")

                    files_parsed += 1

                except ASTParsingError as ast_err:
                    err_msg = f"AST Parsing failed: {ast_err}"
                    logger.warning("AST error on %s: %s", rel_path, ast_err)
                    errors.append(
                        IngestionError(
                            file_path=rel_path,
                            error_type="ASTParsingError",
                            message=err_msg,
                        )
                    )
                except Exception as unk_err:
                    err_msg = f"Unexpected error while parsing: {unk_err}"
                    logger.error("Unexpected error on %s: %s", rel_path, unk_err, exc_info=True)
                    errors.append(
                        IngestionError(
                            file_path=rel_path,
                            error_type="UnexpectedError",
                            message=err_msg,
                        )
                    )

        logger.info(
            "Ingestion completed for %s: %d files parsed, %d functions, %d classes, %d errors. Saved to %s",
            repo_name,
            files_parsed,
            functions_extracted,
            classes_extracted,
            len(errors),
            output_jsonl_path,
        )

        languages_detected_counts = {
            lang: stat.file_count for lang, stat in language_stats.items()
        }

        return IngestionResult(
            repository_url=repo_ctx.repo_url,
            commit_hash=commit_hash,
            local_path=str(repo_root),
            files_discovered=total_files + len(unsupported_files),
            files_parsed=files_parsed,
            unsupported_files=len(unsupported_files),
            functions_extracted=functions_extracted,
            classes_extracted=classes_extracted,
            errors=errors,
            languages_detected=languages_detected_counts,
            output_file=str(output_jsonl_path),
        )
