"""Matches Git diff changes to Tree-sitter AST functions and classes to generate BugFixPairs."""

import difflib
import logging
from typing import List, Optional, Set, Tuple

from backend.ingestion.ast_parser import ASTParserManager
from backend.ingestion.language import LanguageDetector
from backend.models.schemas import (
    BugDetectionEvidence,
    BugFixPair,
    CodeUnit,
    CodeUnitType,
    CommitFileDiff,
    CommitRecord,
)

logger = logging.getLogger(__name__)


def ranges_overlap(start1: int, end1: int, start2: int, end2: int) -> bool:
    """Check whether two 1-indexed inclusive line intervals overlap."""
    return max(start1, start2) <= min(end1, end2)


class FunctionMatcher:
    """Matches modified line ranges in Git diffs to AST code units before and after a commit."""

    def __init__(
        self,
        ast_parser: Optional[ASTParserManager] = None,
        language_detector: Optional[LanguageDetector] = None,
    ):
        self.ast_parser = ast_parser or ASTParserManager()
        self.language_detector = language_detector or LanguageDetector()

    def _find_overlapping_units(
        self, units: List[CodeUnit], line_ranges: List[Tuple[int, int]]
    ) -> List[CodeUnit]:
        """
        Identify units that intersect modified line ranges, prioritizing smallest units
        (methods and functions over enclosing classes).
        """
        if not line_ranges or not units:
            return []

        overlapping: List[CodeUnit] = []
        for unit in units:
            for r_start, r_end in line_ranges:
                if ranges_overlap(unit.start_line, unit.end_line, r_start, r_end):
                    overlapping.append(unit)
                    break

        # Prioritize methods/functions over enclosing classes
        # If any method of class C was touched, exclude class C itself
        methods_and_funcs = [
            u for u in overlapping
            if u.unit_type in (CodeUnitType.FUNCTION, CodeUnitType.ASYNC_FUNCTION, CodeUnitType.METHOD)
        ]

        if methods_and_funcs:
            # Only keep the granular function/method units
            return methods_and_funcs

        return overlapping

    @staticmethod
    def compute_unit_diff(
        old_code: str, new_code: str, unit_name: str, file_path: str
    ) -> Tuple[str, int, int]:
        """
        Generate a unified diff and compute additions/deletions for a single code unit.
        """
        old_lines = old_code.splitlines(keepends=True)
        new_lines = new_code.splitlines(keepends=True)

        diff_iter = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{file_path}:{unit_name}",
            tofile=f"b/{file_path}:{unit_name}",
        )
        diff_text = "".join(diff_iter)

        adds = 0
        dels = 0
        for line in diff_text.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                adds += 1
            elif line.startswith("-") and not line.startswith("---"):
                dels += 1

        return diff_text, adds, dels

    def match_commit_changes(
        self,
        commit_record: CommitRecord,
        file_diff: CommitFileDiff,
        old_content: Optional[str],
        new_content: Optional[str],
        detection_evidence: BugDetectionEvidence,
    ) -> List[BugFixPair]:
        """
        Extract BugFixPair instances for a modified source file in a commit.

        Args:
            commit_record: The commit record metadata.
            file_diff: Diff details for the specific file.
            old_content: Full source code at parent commit (None if added).
            new_content: Full source code at fix commit (None if deleted).
            detection_evidence: Heuristic evidence for the commit.

        Returns:
            List of BugFixPair objects.
        """
        # Skip additions or deletions (we need both buggy and fixed versions)
        if file_diff.change_type in ("A", "D") or not old_content or not new_content:
            return []

        # Detect language
        from pathlib import Path
        file_path = file_diff.file_path
        language = self.language_detector.detect_language(Path(file_path))
        if not language or not self.ast_parser.is_language_supported(language):
            return []

        parent_sha = commit_record.parent_shas[0] if commit_record.parent_shas else "unknown"

        # Parse ASTs
        try:
            old_units = self.ast_parser.extract_units(
                source_code=old_content,
                language=language,
                file_path=file_diff.old_file_path or file_path,
                repository=commit_record.repository,
                commit_hash=parent_sha,
            )
        except Exception as e:
            logger.warning("Failed to parse old AST for %s at %s: %s", file_path, parent_sha, e)
            return []

        try:
            new_units = self.ast_parser.extract_units(
                source_code=new_content,
                language=language,
                file_path=file_path,
                repository=commit_record.repository,
                commit_hash=commit_record.commit_sha,
            )
        except Exception as e:
            logger.warning("Failed to parse new AST for %s at %s: %s", file_path, commit_record.commit_sha, e)
            return []

        # Find units overlapping changed lines
        modified_old_units = self._find_overlapping_units(old_units, file_diff.old_line_ranges)
        modified_new_units = self._find_overlapping_units(new_units, file_diff.new_line_ranges)

        new_units_by_name = {u.name: u for u in new_units}
        pairs: List[BugFixPair] = []
        processed_names: Set[str] = set()

        for old_u in modified_old_units:
            if old_u.name in processed_names:
                continue

            # Attempt to match with new unit by name
            matching_new = new_units_by_name.get(old_u.name)

            # Fallback: if not found by name, check if any modified new unit is at the same location
            if matching_new is None and len(modified_new_units) == 1 and len(modified_old_units) == 1:
                matching_new = modified_new_units[0]

            if matching_new is None:
                continue

            # Verify that the function code actually changed
            if old_u.source_code.strip() == matching_new.source_code.strip():
                continue

            diff_text, adds, dels = self.compute_unit_diff(
                old_u.source_code, matching_new.source_code, old_u.name, file_path
            )

            processed_names.add(old_u.name)

            pairs.append(
                BugFixPair(
                    repository=commit_record.repository,
                    commit_sha=commit_record.commit_sha,
                    parent_sha=parent_sha,
                    commit_message=commit_record.commit_message,
                    author=commit_record.author_name,
                    timestamp=commit_record.timestamp,
                    file_path=file_path,
                    language=language,
                    code_unit_type=str(old_u.unit_type),
                    class_name=old_u.parent_name,
                    function_name=old_u.name,
                    buggy_code=old_u.source_code,
                    fixed_code=matching_new.source_code,
                    diff=diff_text,
                    additions=adds,
                    deletions=dels,
                    detector_score=detection_evidence.score,
                    detector_signals=detection_evidence.matched_signals,
                    detector_classification=(
                        detection_evidence.classification.value
                        if hasattr(detection_evidence.classification, "value")
                        else str(detection_evidence.classification)
                    ),
                    extraction_status="success",
                    error_message=None,
                )
            )

        return pairs
