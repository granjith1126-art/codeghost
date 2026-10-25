"""Indexing pipeline: Reads BugFixPair JSONL, builds representations, embeds, and stores in Qdrant."""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Callable, List, Optional, Union

from backend.models.schemas import BugFixPair
from backend.retrieval.embedder import BaseEmbedder, FastEmbedEmbedder, RepresentationBuilder
from backend.retrieval.schemas import IndexingStats, RepresentationType
from backend.retrieval.vector_store import BaseVectorStore, QdrantVectorStore

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "codeghost_bug_history"


class BugFixIndexer:
    """Indexes historical bug-fix examples into a vector database."""

    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        embedder: Optional[BaseEmbedder] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        self.embedder = embedder or FastEmbedEmbedder()
        self.vector_store = vector_store or QdrantVectorStore()
        self.collection_name = collection_name

    @staticmethod
    def generate_point_id(pair: BugFixPair) -> str:
        """Generate a deterministic UUID from the unique tuple of a bug-fix pair."""
        unique_key = f"{pair.repository}::{pair.commit_sha}::{pair.file_path}::{pair.function_name}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_key))

    def index_file(
        self,
        jsonl_path: Union[str, Path],
        representation_type: Union[RepresentationType, str] = RepresentationType.CODE_AND_METADATA,
        recreate_collection: bool = True,
        batch_size: int = 32,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> IndexingStats:
        """
        Read bug-fix pairs from JSONL and index into the vector store.

        Args:
            jsonl_path: Path to bug_fix_pairs.jsonl
            representation_type: Representation format (code_only, code_and_metadata, code_and_context)
            recreate_collection: Whether to recreate the collection before indexing
            batch_size: Batch size for embedding and insertion
            progress_callback: Optional callback(completed, total, message)

        Returns:
            IndexingStats with summary metrics.
        """
        start_time = time.time()
        jsonl_file = Path(jsonl_path).resolve()
        if not jsonl_file.exists():
            raise FileNotFoundError(f"Bug fix pairs dataset not found at: {jsonl_file}")

        logger.info(
            "Starting indexing for '%s' into collection '%s' (representation=%s, recreate=%s)",
            jsonl_file,
            self.collection_name,
            representation_type,
            recreate_collection,
        )

        # Ensure collection exists
        self.vector_store.create_collection(
            collection_name=self.collection_name,
            vector_size=self.embedder.dimension,
            recreate=recreate_collection,
        )

        dataset_records = 0
        valid_pairs: List[BugFixPair] = []
        failed_count = 0

        # Read and validate JSONL
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line_str = line.strip()
                if not line_str:
                    continue
                dataset_records += 1
                try:
                    pair = BugFixPair.model_validate_json(line_str)
                    valid_pairs.append(pair)
                except Exception as e:
                    logger.warning("Malformed record at line %d in %s: %s", line_no, jsonl_file, e)
                    failed_count += 1

        total_valid = len(valid_pairs)
        logger.info("Validated %d records (%d failed parsing)", total_valid, failed_count)

        embedded_count = 0
        indexed_count = 0

        # Process in batches
        for i in range(0, total_valid, batch_size):
            batch_pairs = valid_pairs[i : i + batch_size]

            # Build representation texts to embed
            texts_to_embed = [
                RepresentationBuilder.build_pair_representation(
                    pair=p,
                    representation_type=representation_type,
                    target_version="buggy",
                )
                for p in batch_pairs
            ]

            # Generate embeddings
            try:
                embeddings = self.embedder.embed_batch(texts_to_embed)
                embedded_count += len(embeddings)
            except Exception as e:
                logger.error("Failed to generate embeddings for batch %d-%d: %s", i, i + len(batch_pairs), e)
                failed_count += len(batch_pairs)
                continue

            # Build Qdrant payloads
            payloads = []
            ids = []
            for pair, rep_text in zip(batch_pairs, texts_to_embed):
                ids.append(self.generate_point_id(pair))
                payloads.append(
                    {
                        "repository": pair.repository,
                        "commit_sha": pair.commit_sha,
                        "parent_sha": pair.parent_sha,
                        "file_path": pair.file_path,
                        "language": pair.language,
                        "class_name": pair.class_name,
                        "function_name": pair.function_name,
                        "commit_message": pair.commit_message,
                        "detector_classification": pair.detector_classification,
                        "detector_score": pair.detector_score,
                        "buggy_code": pair.buggy_code,
                        "fixed_code": pair.fixed_code,
                        "diff": pair.diff,
                        "timestamp": pair.timestamp,
                        "representation_type": str(representation_type),
                        "indexed_text": rep_text,
                    }
                )

            # Insert into vector store
            try:
                inserted = self.vector_store.insert_batch(
                    collection_name=self.collection_name,
                    vectors=embeddings,
                    payloads=payloads,
                    ids=ids,
                )
                indexed_count += inserted
            except Exception as e:
                logger.error("Failed to insert batch into Qdrant: %s", e)
                failed_count += len(batch_pairs)

            if progress_callback:
                progress_callback(
                    indexed_count,
                    total_valid,
                    f"Indexed {indexed_count}/{total_valid} functions",
                )

        elapsed = round(time.time() - start_time, 2)
        logger.info(
            "Indexing complete: %d indexed, %d failed out of %d in %.2f seconds",
            indexed_count,
            failed_count,
            dataset_records,
            elapsed,
        )

        return IndexingStats(
            dataset_records=dataset_records,
            valid_records=total_valid,
            embedded=embedded_count,
            indexed=indexed_count,
            failed=failed_count,
            vector_dimension=self.embedder.dimension,
            collection_name=self.collection_name,
            elapsed_seconds=elapsed,
        )
