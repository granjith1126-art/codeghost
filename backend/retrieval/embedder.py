"""Embedding layer and code representation builders for CodeGhost."""

import hashlib
import logging
import math
from abc import ABC, abstractmethod
from typing import List, Optional, Union

from backend.models.schemas import BugFixPair, CodeUnit
from backend.retrieval.schemas import RepresentationType

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract base class for all CodeGhost embedding models."""

    @abstractmethod
    def embed_code(self, text: str) -> List[float]:
        """Generate an embedding vector for a single code snippet or representation."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name or identifier of the embedding model."""
        pass


class FastEmbedEmbedder(BaseEmbedder):
    """
    Embedding implementation using FastEmbed (ONNX Runtime).
    Defaults to BAAI/bge-base-en-v1.5 (768 dimensions).
    Can be replaced by other models supported by FastEmbed or CodeBERT.
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self._model_name = model_name
        self._dimension = 768 if "base" in model_name else 384
        self._model = None

    def _get_model(self):
        if self._model is None:
            logger.info("Initializing FastEmbed model: %s", self._model_name)
            try:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name=self._model_name)
            except Exception as e:
                logger.error("Failed to load FastEmbed model %s: %s", self._model_name, e)
                raise RuntimeError(f"Could not initialize FastEmbed with {self._model_name}: {e}") from e
        return self._model

    def embed_code(self, text: str) -> List[float]:
        model = self._get_model()
        vectors = list(model.embed([text]))
        if not vectors:
            return [0.0] * self.dimension
        return vectors[0].tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._get_model()
        embeddings = list(model.embed(texts))
        return [e.tolist() for e in embeddings]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name


class MockEmbedder(BaseEmbedder):
    """
    Fast deterministic pseudo-embedding generator for testing without downloading models.
    Produces unit-normalized vectors derived from MD5 hashes of words in the text.
    """

    def __init__(self, dimension: int = 128, model_name: str = "mock-embedder-v1"):
        self._dimension = dimension
        self._model_name = model_name

    def embed_code(self, text: str) -> List[float]:
        tokens = text.lower().split()
        vec = [0.0] * self._dimension
        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dimension
            val = ((h >> 8) % 1000) / 1000.0 - 0.5
            vec[idx] += val

        # Normalize to unit length
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-9:
            vec = [x / norm for x in vec]
        else:
            vec[0] = 1.0
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_code(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name


class RepresentationBuilder:
    """Constructs structured representation strings for function code units and bug-fix pairs."""

    @staticmethod
    def build_query_representation(
        code: str,
        representation_type: Union[RepresentationType, str] = RepresentationType.CODE_AND_METADATA,
        language: Optional[str] = None,
        file_path: Optional[str] = None,
        class_name: Optional[str] = None,
        function_name: Optional[str] = None,
    ) -> str:
        rep_val = representation_type.value if hasattr(representation_type, "value") else str(representation_type)

        if rep_val == RepresentationType.CODE_ONLY.value:
            return code.strip()

        # Build metadata header
        header_lines = []
        if language:
            header_lines.append(f"Language: {language}")
        if file_path:
            header_lines.append(f"File: {file_path}")
        if class_name:
            header_lines.append(f"Class: {class_name}")
        if function_name:
            header_lines.append(f"Function: {function_name}")

        header = "\n".join(header_lines)

        return f"{header}\n\nCode:\n{code.strip()}"

    @staticmethod
    def build_pair_representation(
        pair: BugFixPair,
        representation_type: Union[RepresentationType, str] = RepresentationType.CODE_AND_METADATA,
        target_version: str = "buggy",
    ) -> str:
        rep_val = representation_type.value if hasattr(representation_type, "value") else str(representation_type)
        code = pair.buggy_code if target_version == "buggy" else pair.fixed_code

        if rep_val == RepresentationType.CODE_ONLY.value:
            return code.strip()

        header_lines = [
            f"Language: {pair.language}",
            f"File: {pair.file_path}",
        ]
        if pair.class_name:
            header_lines.append(f"Class: {pair.class_name}")
        header_lines.append(f"Function: {pair.function_name}")

        if rep_val == RepresentationType.CODE_AND_CONTEXT.value and pair.commit_message:
            header_lines.append(f"Bug Context: {pair.commit_message.splitlines()[0]}")

        header = "\n".join(header_lines)
        code_label = "Buggy Code" if target_version == "buggy" else "Fixed Code"

        return f"{header}\n\n{code_label}:\n{code.strip()}"
