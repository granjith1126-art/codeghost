"""Vector Database abstraction and Qdrant implementation for CodeGhost."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from qdrant_client import QdrantClient
from qdrant_client.http import models

from backend.retrieval.schemas import RetrievedMatch

logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """Abstract base class for CodeGhost vector storage and retrieval."""

    @abstractmethod
    def create_collection(self, collection_name: str, vector_size: int, recreate: bool = False) -> None:
        """Create or recreate a vector collection with the specified vector dimension."""
        pass

    @abstractmethod
    def insert_batch(
        self,
        collection_name: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: List[str],
    ) -> int:
        """Insert a batch of vectors with associated metadata payloads."""
        pass

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        repository: Optional[str] = None,
        language: Optional[str] = None,
        exclude_commit_sha: Optional[str] = None,
        max_timestamp: Optional[str] = None,
    ) -> List[RetrievedMatch]:
        """Perform similarity search with repository isolation and leakage filters."""
        pass

    @abstractmethod
    def count(self, collection_name: str) -> int:
        """Get total number of points in the collection."""
        pass


class QdrantVectorStore(BaseVectorStore):
    """Local or embedded Qdrant vector database implementation."""

    def __init__(self, storage_path: Optional[Union[str, Path]] = None):
        """
        Initialize Qdrant client.

        Args:
            storage_path: Local folder path for persistent storage, or ':memory:' for transient in-memory store.
        """
        if storage_path == ":memory:":
            self.client = QdrantClient(location=":memory:")
            self.storage_path = ":memory:"
            logger.info("Initialized in-memory QdrantVectorStore")
        elif storage_path:
            p = Path(storage_path).resolve()
            p.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(p))
            self.storage_path = str(p)
            logger.info("Initialized local persistent QdrantVectorStore at: %s", self.storage_path)
        else:
            default_path = Path(__file__).resolve().parent.parent / "data" / "qdrant"
            default_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(default_path))
            self.storage_path = str(default_path)
            logger.info("Initialized default local QdrantVectorStore at: %s", self.storage_path)

    def collection_exists(self, collection_name: str) -> bool:
        collections = self.client.get_collections().collections
        return any(c.name == collection_name for c in collections)

    def create_collection(self, collection_name: str, vector_size: int, recreate: bool = False) -> None:
        exists = self.collection_exists(collection_name)
        if exists:
            if recreate:
                logger.info("Recreating Qdrant collection '%s'", collection_name)
                self.client.delete_collection(collection_name)
            else:
                logger.debug("Collection '%s' already exists", collection_name)
                return

        logger.info("Creating Qdrant collection '%s' (dimension=%d)", collection_name, vector_size)
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def insert_batch(
        self,
        collection_name: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: List[str],
    ) -> int:
        if not vectors or len(vectors) != len(payloads) or len(vectors) != len(ids):
            raise ValueError("Vectors, payloads, and ids must have the same non-empty length")

        points = [
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
            for point_id, vector, payload in zip(ids, vectors, payloads)
        ]

        self.client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
        return len(points)

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        repository: Optional[str] = None,
        language: Optional[str] = None,
        exclude_commit_sha: Optional[str] = None,
        max_timestamp: Optional[str] = None,
    ) -> List[RetrievedMatch]:
        if not self.collection_exists(collection_name):
            logger.warning("Search failed: collection '%s' does not exist", collection_name)
            return []

        must_conditions = []
        must_not_conditions = []

        # Repository isolation filter
        if repository:
            must_conditions.append(
                models.FieldCondition(
                    key="repository",
                    match=models.MatchValue(value=repository),
                )
            )

        # Language filter
        if language:
            must_conditions.append(
                models.FieldCondition(
                    key="language",
                    match=models.MatchValue(value=language),
                )
            )

        # Temporal cutoff filter (avoid future leakage in evaluation)
        if max_timestamp:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(max_timestamp.replace("Z", "+00:00"))
                must_conditions.append(
                    models.FieldCondition(
                        key="timestamp",
                        range=models.DatetimeRange(lt=dt),
                    )
                )
            except Exception as e:
                logger.debug("Failed to construct DatetimeRange filter: %s", e)

        # Self-exclusion filter (cannot retrieve the exact same commit under evaluation)
        if exclude_commit_sha:
            must_not_conditions.append(
                models.FieldCondition(
                    key="commit_sha",
                    match=models.MatchValue(value=exclude_commit_sha),
                )
            )

        query_filter = None
        if must_conditions or must_not_conditions:
            query_filter = models.Filter(
                must=must_conditions if must_conditions else None,
                must_not=must_not_conditions if must_not_conditions else None,
            )

        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        results: List[RetrievedMatch] = []
        for scored_point in response.points:
            p = scored_point.payload or {}
            results.append(
                RetrievedMatch(
                    similarity_score=round(float(scored_point.score), 4),
                    repository=p.get("repository", "unknown"),
                    commit_sha=p.get("commit_sha", "unknown"),
                    parent_sha=p.get("parent_sha", "unknown"),
                    file_path=p.get("file_path", "unknown"),
                    function_name=p.get("function_name", "unknown"),
                    class_name=p.get("class_name"),
                    commit_message=p.get("commit_message", ""),
                    buggy_code=p.get("buggy_code", ""),
                    fixed_code=p.get("fixed_code", ""),
                    diff=p.get("diff", ""),
                    detector_score=float(p.get("detector_score", 0.0)),
                    detector_classification=p.get("detector_classification", "UNKNOWN"),
                    timestamp=p.get("timestamp", ""),
                    matched_text=p.get("indexed_text"),
                    metadata=p.get("metadata", {}),
                )
            )

        return results

    def count(self, collection_name: str) -> int:
        if not self.collection_exists(collection_name):
            return 0
        res = self.client.count(collection_name)
        return res.count
