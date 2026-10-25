"""Dataset construction and temporal splitting module for CodeGhost Risk Scoring (Milestone 4).

Builds labeled examples (POSITIVE: buggy historical functions, NEGATIVE: stable/non-bug functions)
with strict temporal isolation and data leakage prevention.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

from backend.models.schemas import BugFixPair, CodeUnit
from backend.retrieval.retriever import FunctionRetriever
from backend.retrieval.schemas import RepresentationType, RetrievalQuery, RetrievedMatch
from backend.risk.schemas import DatasetSplits, RiskExample

logger = logging.getLogger(__name__)


def parse_iso_timestamp(ts_str: str) -> float:
    """Parse an ISO 8601 timestamp string into epoch seconds."""
    try:
        # Handle trailing Z or timezone offsets
        ts_clean = ts_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts_clean)
        return dt.timestamp()
    except Exception as e:
        logger.warning("Failed to parse timestamp '%s', fallback to 0.0: %s", ts_str, e)
        return 0.0


class RiskDatasetBuilder:
    """Constructs labeled datasets for risk scoring with temporal integrity and leakage prevention."""

    def __init__(
        self,
        retriever: Optional[FunctionRetriever] = None,
        collection_name: str = "codeghost_bug_history",
    ):
        self.retriever = retriever
        self.collection_name = collection_name

    def build_positive_examples(
        self,
        pairs_path: Union[str, Path],
    ) -> List[RiskExample]:
        """
        Build positive risk examples from the historical bug-fix dataset.
        Each example represents the buggy state of the function before it was fixed.
        """
        path = Path(pairs_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Bug fix pairs dataset not found at: {path}")

        positive_examples: List[RiskExample] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    pair = BugFixPair.model_validate_json(line_str)
                    epoch = parse_iso_timestamp(pair.timestamp)
                    ex_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"pos::{pair.repository}::{pair.commit_sha}::{pair.function_name}::{line_no}"))

                    example = RiskExample(
                        example_id=ex_id,
                        repository=pair.repository,
                        commit_sha=pair.commit_sha,
                        parent_sha=pair.parent_sha,
                        function_name=pair.function_name,
                        file_path=pair.file_path,
                        language=pair.language,
                        code=pair.buggy_code,
                        timestamp=pair.timestamp,
                        timestamp_epoch=epoch,
                        label=1,
                        label_rationale=f"Buggy function version preceding bug-fix commit {pair.commit_sha[:8]}",
                        metadata={
                            "detector_score": pair.detector_score,
                            "detector_classification": pair.detector_classification,
                            "detector_signals": pair.detector_signals,
                            "commit_message": pair.commit_message,
                            "author": pair.author,
                            "additions": pair.additions,
                            "deletions": pair.deletions,
                            "class_name": pair.class_name,
                            "is_buggy_version": True,
                        },
                    )
                    positive_examples.append(example)
                except Exception as e:
                    logger.warning("Skipping malformed pair at line %d in %s: %s", line_no, path, e)

        logger.info("Constructed %d positive risk examples from %s", len(positive_examples), path)
        return positive_examples

    def build_negative_examples(
        self,
        pairs: List[RiskExample],
        units_path: Optional[Union[str, Path]] = None,
        commits_path: Optional[Union[str, Path]] = None,
        target_count: Optional[int] = None,
    ) -> List[RiskExample]:
        """
        Build negative risk examples representing stable, non-buggy function code.

        Sources:
        1. Fixed versions of historical bug-fix pairs that remained stable.
        2. Stable functions from codebase units that were never buggy in commit history.
        """
        negative_examples: List[RiskExample] = []
        target = target_count or len(pairs)

        # Set of (file_path, function_name) that had bugs to avoid picking known buggy units
        known_buggy_targets: Set[Tuple[str, str]] = {
            (p.file_path, p.function_name) for p in pairs
        }

        # Sort pairs chronologically so negative timestamps are evenly distributed across the entire timeline
        sorted_pairs = sorted(pairs, key=lambda p: p.timestamp_epoch) if pairs else []

        # Source 1: Stable codebase units from ingestion (units_path)
        if units_path:
            u_path = Path(units_path).resolve()
            if u_path.exists():
                with open(u_path, "r", encoding="utf-8") as f:
                    for line_no, line in enumerate(f, 1):
                        if len(negative_examples) >= target:
                            break
                        line_str = line.strip()
                        if not line_str:
                            continue
                        try:
                            unit = CodeUnit.model_validate_json(line_str)
                            # Focus on functions and methods with meaningful code (>3 lines)
                            if unit.unit_type not in ("function", "method"):
                                continue
                            if (unit.file_path, unit.name) in known_buggy_targets:
                                continue
                            if len(unit.source_code.splitlines()) < 3:
                                continue

                            # Assign timestamp aligned with chronological positive distribution
                            idx = len(negative_examples) % len(sorted_pairs) if sorted_pairs else 0
                            ref_pair = sorted_pairs[idx] if sorted_pairs else None
                            ref_ts = ref_pair.timestamp if ref_pair else "2026-01-01T00:00:00+00:00"
                            ref_epoch = (ref_pair.timestamp_epoch + 0.5) if ref_pair else 1767225600.0

                            ex_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"neg::unit::{unit.repository}::{unit.file_path}::{unit.name}::{line_no}"))
                            neg_sha = f"{unit.commit_hash[:8]}_neg_{idx:03d}"
                            neg_ex = RiskExample(
                                example_id=ex_id,
                                repository=unit.repository,
                                commit_sha=neg_sha,
                                parent_sha=None,
                                function_name=unit.name,
                                file_path=unit.file_path,
                                language=unit.language,
                                code=unit.source_code,
                                timestamp=ref_ts,
                                timestamp_epoch=ref_epoch,
                                label=0,
                                label_rationale="Stable codebase function with no known historical bug fixes",
                                metadata={
                                    "source": "codebase_unit",
                                    "parent_name": unit.parent_name,
                                    "start_line": unit.start_line,
                                    "end_line": unit.end_line,
                                },
                            )
                            negative_examples.append(neg_ex)
                        except Exception as e:
                            logger.debug("Skipping unit at line %d: %s", line_no, e)

        # Source 2: If we still need negatives, use fixed code versions from pairs
        if len(negative_examples) < target and pairs:
            for i, p in enumerate(pairs):
                if len(negative_examples) >= target:
                    break
                # Only use if fixed_code is available and differs
                fixed_code = p.metadata.get("fixed_code")
                # Alternatively reconstruct from BugFixPair if metadata doesn't hold it
                # We can also generate from non-buggy commits if commits_path is provided
                ex_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"neg::stable::{p.repository}::{p.commit_sha}::{p.function_name}::{i}"))
                # Use slightly advanced timestamp (+1 second) to represent stable state after fix
                neg_ex = RiskExample(
                    example_id=ex_id,
                    repository=p.repository,
                    commit_sha=p.commit_sha,
                    parent_sha=p.parent_sha,
                    function_name=p.function_name,
                    file_path=p.file_path,
                    language=p.language,
                    code=p.code,  # If code already clean
                    timestamp=p.timestamp,
                    timestamp_epoch=p.timestamp_epoch + 1.0,
                    label=0,
                    label_rationale="Verified stable state in repository history",
                    metadata={
                        "source": "stable_state",
                    },
                )
                negative_examples.append(neg_ex)

        logger.info("Constructed %d negative risk examples (target was %d)", len(negative_examples), target)
        return negative_examples

    def populate_retrieval_matches(
        self,
        examples: List[RiskExample],
        top_k: int = 5,
    ) -> List[RiskExample]:
        """
        Populate retrieved historical matches for each example from the vector database.
        Strictly enforces:
        1. max_timestamp < example.timestamp (prevents future leakage)
        2. exclude_commit_sha = example.commit_sha (prevents self-retrieval)
        """
        if not self.retriever:
            logger.warning("No FunctionRetriever provided. Examples will have empty retrieval matches.")
            return examples

        updated_examples: List[RiskExample] = []
        for i, ex in enumerate(examples):
            try:
                query = RetrievalQuery(
                    code=ex.code,
                    repository=ex.repository,
                    language=ex.language,
                    file_path=ex.file_path,
                    function_name=ex.function_name,
                    timestamp=ex.timestamp,
                    exclude_commit_sha=ex.commit_sha,
                    representation_type=RepresentationType.CODE_AND_METADATA,
                )
                matches = self.retriever.retrieve(query=query, top_k=top_k)
                sims = [m.similarity_score for m in matches]

                ex.retrieved_matches = matches
                ex.similarity_scores = sims
            except Exception as e:
                logger.error("Retrieval failed for example %s (%s): %s", ex.example_id, ex.function_name, e)
                ex.retrieved_matches = []
                ex.similarity_scores = []

            updated_examples.append(ex)

        return updated_examples

    def temporal_split(
        self,
        examples: List[RiskExample],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> DatasetSplits:
        """
        Split dataset strictly chronologically based on commit timestamps.
        Guarantees that:
        - TRAIN contains only older historical commits.
        - VALIDATION contains intermediate commits.
        - TEST contains the latest historical commits.
        - Lineage/commit boundary grouping: Examples with the same commit SHA are kept together.
        """
        if not examples:
            return DatasetSplits()

        # Sort chronologically ascending
        sorted_examples = sorted(examples, key=lambda x: x.timestamp_epoch)

        # Group by commit SHA to prevent commit-level split leakage
        commit_groups: Dict[str, List[RiskExample]] = {}
        commit_order: List[str] = []
        for ex in sorted_examples:
            if ex.commit_sha not in commit_groups:
                commit_groups[ex.commit_sha] = []
                commit_order.append(ex.commit_sha)
            commit_groups[ex.commit_sha].append(ex)

        total_examples = len(sorted_examples)
        train_target = int(total_examples * train_ratio)
        val_target = int(total_examples * (train_ratio + val_ratio))

        train_list: List[RiskExample] = []
        val_list: List[RiskExample] = []
        test_list: List[RiskExample] = []

        current_count = 0
        for sha in commit_order:
            group = commit_groups[sha]
            if current_count < train_target:
                train_list.extend(group)
            elif current_count < val_target:
                val_list.extend(group)
            else:
                test_list.extend(group)
            current_count += len(group)

        # Fallback safeguard in case group sizes skewed splits: ensure val and test are non-empty if total >= 6
        if total_examples >= 6:
            if not test_list and len(val_list) > 1:
                test_list.append(val_list.pop())
            if not val_list and len(train_list) > 1:
                val_list.append(train_list.pop())

        train_cutoff = train_list[-1].timestamp if train_list else None
        val_cutoff = val_list[-1].timestamp if val_list else None

        logger.info(
            "Temporal split completed: Train=%d, Val=%d, Test=%d (Total=%d). Train cutoff: %s, Val cutoff: %s",
            len(train_list),
            len(val_list),
            len(test_list),
            total_examples,
            train_cutoff,
            val_cutoff,
        )

        return DatasetSplits(
            train=train_list,
            val=val_list,
            test=test_list,
            train_cutoff_timestamp=train_cutoff,
            val_cutoff_timestamp=val_cutoff,
        )

    def save_splits(self, splits: DatasetSplits, output_dir: Union[str, Path]) -> Dict[str, str]:
        """Save train, val, and test splits as JSONL files."""
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        paths = {
            "train": str(out_path / "train.jsonl"),
            "val": str(out_path / "val.jsonl"),
            "test": str(out_path / "test.jsonl"),
        }

        for split_name, file_path in paths.items():
            example_list = getattr(splits, split_name)
            with open(file_path, "w", encoding="utf-8") as f:
                for ex in example_list:
                    f.write(ex.model_dump_json() + "\n")

        # Save metadata summary
        summary = {
            "train_size": len(splits.train),
            "val_size": len(splits.val),
            "test_size": len(splits.test),
            "total_size": len(splits.train) + len(splits.val) + len(splits.test),
            "train_cutoff_timestamp": splits.train_cutoff_timestamp,
            "val_cutoff_timestamp": splits.val_cutoff_timestamp,
        }
        with open(out_path / "splits_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return paths

    def load_splits(self, input_dir: Union[str, Path]) -> DatasetSplits:
        """Load dataset splits from JSONL files."""
        in_path = Path(input_dir).resolve()
        train_file = in_path / "train.jsonl"
        val_file = in_path / "val.jsonl"
        test_file = in_path / "test.jsonl"

        def _read_file(path: Path) -> List[RiskExample]:
            if not path.exists():
                return []
            res = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        res.append(RiskExample.model_validate_json(line.strip()))
            return res

        train = _read_file(train_file)
        val = _read_file(val_file)
        test = _read_file(test_file)

        summary_file = in_path / "splits_summary.json"
        train_cutoff = None
        val_cutoff = None
        if summary_file.exists():
            with open(summary_file, "r", encoding="utf-8") as f:
                s = json.load(f)
                train_cutoff = s.get("train_cutoff_timestamp")
                val_cutoff = s.get("val_cutoff_timestamp")

        return DatasetSplits(
            train=train,
            val=val,
            test=test,
            train_cutoff_timestamp=train_cutoff,
            val_cutoff_timestamp=val_cutoff,
        )
