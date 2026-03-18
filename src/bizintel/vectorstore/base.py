"""
Abstract base class for vector stores + shared models + factory function.

All backends (ChromaDB, FAISS, etc.) implement VectorStoreBase so the
rest of the codebase can swap backends via a single config change.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel, ConfigDict

from bizintel.config.settings import VECTOR_STORE_BACKEND, EMBEDDING_BATCH_SIZE, TOP_K

logger = logging.getLogger(__name__)


# ── Search result model ──────────────────────────────────────────────────


class SearchResult(BaseModel):
    """A single result from a similarity search."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    text: str
    metadata: dict
    distance: float


# ── Abstract base ────────────────────────────────────────────────────────


class VectorStoreBase(ABC):
    """
    Common interface that every vector-store backend must implement.

    This enables the Strategy Pattern — swap implementations via config
    without touching business logic.
    """

    @property
    @abstractmethod
    def count(self) -> int:
        """Number of documents in the store."""

    @abstractmethod
    def add(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict],
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        """Add documents + embeddings + metadata (batched)."""

    @abstractmethod
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = TOP_K,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """Return top-K most similar documents."""

    @abstractmethod
    def reset(self) -> None:
        """Delete all documents from the store."""

    @abstractmethod
    def get_all_documents(
        self,
        batch_size: int = 5000,
    ) -> tuple[list[str], list[str], list[dict]]:
        """
        Return ALL (doc_ids, texts, metadatas) from the store.

        Used to build auxiliary indices (e.g. BM25) at startup.
        """


# ── Factory ──────────────────────────────────────────────────────────────


def create_vector_store(backend: str = VECTOR_STORE_BACKEND) -> VectorStoreBase:
    """
    Factory function — creates the right backend based on config.

    Parameters
    ----------
    backend : str
        "chroma" or "faiss" (default from settings.py).

    Returns
    -------
    VectorStoreBase
    """
    backend = backend.lower().strip()

    if backend == "chroma":
        from bizintel.vectorstore.chroma_store import ChromaStore
        return ChromaStore()

    if backend == "faiss":
        from bizintel.vectorstore.faiss_store import FAISSStore
        return FAISSStore()

    raise ValueError(
        f"Unknown vector store backend: '{backend}'. "
        f"Supported: 'chroma', 'faiss'."
    )
