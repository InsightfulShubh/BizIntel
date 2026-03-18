"""
ChromaDB backend for the vector store.

Stores vectors, documents, and metadata all in one system.
Supports metadata filtering out of the box.
"""

from __future__ import annotations

import logging
import math
import time

import chromadb
import numpy as np

from bizintel.config.settings import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_BATCH_SIZE,
    TOP_K,
)
from bizintel.vectorstore.base import VectorStoreBase, SearchResult

logger = logging.getLogger(__name__)


class ChromaStore(VectorStoreBase):
    """ChromaDB-backed vector store with cosine similarity."""

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ) -> None:
        persist_path = persist_dir or str(CHROMA_PERSIST_DIR)

        logger.info("Initializing ChromaDB at: %s", persist_path)
        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Collection '%s' ready — %d existing documents",
            collection_name,
            self._collection.count(),
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def add(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict],
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        total = len(texts)
        n_batches = math.ceil(total / batch_size)

        logger.info(
            "ChromaDB: adding %d documents in %d batches",
            total, n_batches,
        )

        overall_start = time.perf_counter()

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, total)
            ids = [f"doc_{j}" for j in range(start_idx, end_idx)]

            batch_start = time.perf_counter()

            self._collection.add(
                ids=ids,
                documents=texts[start_idx:end_idx],
                embeddings=embeddings[start_idx:end_idx].tolist(),
                metadatas=metadatas[start_idx:end_idx],
            )

            batch_elapsed = time.perf_counter() - batch_start
            logger.info(
                "ChromaDB batch %d/%d — %d docs in %.2fs",
                i + 1, n_batches, end_idx - start_idx, batch_elapsed,
            )

        overall_elapsed = time.perf_counter() - overall_start
        logger.info(
            "ChromaDB: all %d documents stored in %.2fs — total: %d",
            total, overall_elapsed, self._collection.count(),
        )

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = TOP_K,
        where: dict | None = None,
    ) -> list[SearchResult]:
        query_args: dict = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if where:
            query_args["where"] = where

        raw = self._collection.query(**query_args)

        # ChromaDB returns lists-of-lists (one per query); we sent 1 query
        ids = raw["ids"][0]
        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]

        return [
            SearchResult(doc_id=did, text=text, metadata=meta, distance=dist)
            for did, text, meta, dist in zip(ids, documents, metadatas, distances)
        ]

    def reset(self) -> None:
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' reset — 0 documents", name)

    def get_all_documents(
        self,
        batch_size: int = 5000,
    ) -> tuple[list[str], list[str], list[dict]]:
        """
        Fetch ALL documents from ChromaDB in batches.

        Returns (doc_ids, texts, metadatas) — each a list of equal length.
        """
        total = self._collection.count()
        logger.info("Fetching all %d documents from ChromaDB…", total)

        all_ids: list[str] = []
        all_texts: list[str] = []
        all_metadatas: list[dict] = []

        for offset in range(0, total, batch_size):
            batch = self._collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            all_ids.extend(batch["ids"])
            all_texts.extend(batch["documents"])
            all_metadatas.extend(batch["metadatas"])
            logger.info(
                "  fetched %d / %d documents",
                len(all_ids), total,
            )

        return all_ids, all_texts, all_metadatas
