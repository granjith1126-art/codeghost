"""Function Retriever: Given a target function, retrieves top-k historically similar bug-fix pairs."""

import logging
from typing import List, Optional, Union

from backend.models.schemas import CodeUnit
from backend.retrieval.embedder import BaseEmbedder, FastEmbedEmbedder, RepresentationBuilder
from backend.retrieval.schemas import RepresentationType, RetrievalQuery, RetrievedMatch
from backend.retrieval.vector_store import BaseVectorStore, QdrantVectorStore

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "codeghost_bug_history"


class FunctionRetriever:
    """Retrieves historically similar bug-fix examples for a given function or code unit."""

    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        embedder: Optional[BaseEmbedder] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        self.embedder = embedder or FastEmbedEmbedder()
        self.vector_store = vector_store or QdrantVectorStore()
        self.collection_name = collection_name

    def retrieve(
        self,
        query: RetrievalQuery,
        top_k: int = 5,
    ) -> List[RetrievedMatch]:
        """
        Execute similarity search for a query with repository isolation and leakage prevention.

        Args:
            query: RetrievalQuery containing code, repository, filters, and representation type.
            top_k: Number of most similar historical matches to retrieve.

        Returns:
            List of RetrievedMatch sorted by similarity score descending.
        """
        # Format text to embed according to requested representation
        query_text = RepresentationBuilder.build_query_representation(
            code=query.code,
            representation_type=query.representation_type,
            language=query.language,
            file_path=query.file_path,
            class_name=query.class_name,
            function_name=query.function_name,
        )

        # Generate query vector
        query_vector = self.embedder.embed_code(query_text)

        # Query vector store with isolation and leakage filters
        matches = self.vector_store.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            repository=query.repository,
            language=query.language,
            exclude_commit_sha=query.exclude_commit_sha,
            max_timestamp=query.timestamp,
        )

        return matches

    def retrieve_vector(
        self,
        query_vector: List[float],
        top_k: int = 5,
        repository: Optional[str] = None,
        language: Optional[str] = None,
        exclude_commit_sha: Optional[str] = None,
        max_timestamp: Optional[str] = None,
    ) -> List[RetrievedMatch]:
        """Execute similarity search with a precomputed query vector."""
        return self.vector_store.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            repository=repository,
            language=language,
            exclude_commit_sha=exclude_commit_sha,
            max_timestamp=max_timestamp,
        )

    def retrieve_for_code_unit(
        self,
        unit: CodeUnit,
        top_k: int = 5,
        exclude_commit_sha: Optional[str] = None,
        max_timestamp: Optional[str] = None,
        representation_type: Union[RepresentationType, str] = RepresentationType.CODE_AND_METADATA,
    ) -> List[RetrievedMatch]:
        """
        Convenience method to retrieve matches directly from an extracted CodeUnit.
        """
        query = RetrievalQuery(
            code=unit.source_code,
            repository=unit.repository,
            language=unit.language,
            file_path=unit.file_path,
            function_name=unit.name,
            class_name=unit.parent_name,
            timestamp=max_timestamp,
            exclude_commit_sha=exclude_commit_sha or unit.commit_hash,
            representation_type=representation_type,
        )
        return self.retrieve(query=query, top_k=top_k)
