"""
FAISS backend for the vector store.

FAISS handles vectors only — text and metadata are stored in a JSON
sidecar file alongside the FAISS index.

Note: FAISS does NOT support metadata filtering natively. The `where`
parameter in query() is applied as a post-filter on the retrieved results.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import faiss
import numpy as np

from bizintel.config.settings import (
    FAISS_INDEX_DIR,
    FAISS_INDEX_FILENAME,
    FAISS_DOCSTORE_FILENAME,
    EMBEDDING_BATCH_SIZE,
    TOP_K,
)
from bizintel.vectorstore.base import VectorStoreBase, SearchResult

logger = logging.getLogger(__name__)


class FAISSStore(VectorStoreBase):
    """
    FAISS-backed vector store with JSON sidecar for text + metadata.

    Uses IndexFlatIP (inner-product / cosine on normalised vectors)
    for exact nearest-neighbour search.
    """

    def __init__(self, index_dir: str | None = None) -> None:
        self._dir = Path(index_dir or str(FAISS_INDEX_DIR))
        self._dir.mkdir(parents=True, exist_ok=True)

        self._index_path = self._dir / FAISS_INDEX_FILENAME
        self._docstore_path = self._dir / FAISS_DOCSTORE_FILENAME

        # Load existing index + docstore, or start fresh
        if self._index_path.exists() and self._docstore_path.exists():
            logger.info("Loading existing FAISS index from: %s", self._dir)
            self._index = faiss.read_index(str(self._index_path))
            self._docstore = self._load_docstore()
            logger.info("FAISS index loaded — %d vectors", self._index.ntotal)
        else:
            logger.info("Creating new FAISS index at: %s", self._dir)
            # Dimension set to 0 — will be initialised on first add()
            self._index: faiss.Index | None = None
            self._docstore: dict[str, dict] = {}

    @property
    def count(self) -> int:
        if self._index is None:
            return 0
        return self._index.ntotal

    # ── Persistence helpers ──────────────────────────────────────────

    def _save_index(self) -> None:
        if self._index is not None:
            faiss.write_index(self._index, str(self._index_path))

    def _save_docstore(self) -> None:
        with open(self._docstore_path, "w", encoding="utf-8") as f:
            json.dump(self._docstore, f, ensure_ascii=False)

    def _load_docstore(self) -> dict[str, dict]:
        with open(self._docstore_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _persist(self) -> None:
        self._save_index()
        self._save_docstore()

    # ── Batched add ──────────────────────────────────────────────────

    def add(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict],
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        total = len(texts)
        n_batches = math.ceil(total / batch_size)
        dim = embeddings.shape[1]

        # Initialise index on first add
        if self._index is None:
            # Normalise vectors → use inner-product = cosine similarity
            self._index = faiss.IndexFlatIP(dim)
            logger.info("FAISS index created — dimension: %d", dim)

        logger.info(
            "FAISS: adding %d documents in %d batches",
            total, n_batches,
        )

        overall_start = time.perf_counter()

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, total)

            batch_vectors = embeddings[start_idx:end_idx].copy()

            # L2-normalise so inner-product = cosine similarity
            faiss.normalize_L2(batch_vectors)
            self._index.add(batch_vectors)

            # Store text + metadata in docstore
            for j in range(start_idx, end_idx):
                doc_id = f"doc_{j}"
                self._docstore[doc_id] = {
                    "text": texts[j],
                    "metadata": metadatas[j],
                }

            batch_elapsed = time.perf_counter() - overall_start
            logger.info(
                "FAISS batch %d/%d — %d docs (%.2fs elapsed)",
                i + 1, n_batches, end_idx - start_idx, batch_elapsed,
            )

        # Persist index + docstore to disk
        self._persist()

        overall_elapsed = time.perf_counter() - overall_start
        logger.info(
            "FAISS: all %d documents stored in %.2fs — total: %d",
            total, overall_elapsed, self._index.ntotal,
        )

    # ── Query ────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = TOP_K,
        where: dict | None = None,
    ) -> list[SearchResult]:
        if self._index is None or self._index.ntotal == 0:
            logger.warning("FAISS index is empty — returning no results")
            return []

        # Normalise query vector (must match add-time normalisation)
        query_vec = query_embedding.copy().reshape(1, -1).astype("float32")
        faiss.normalize_L2(query_vec)

        # Over-fetch if filtering, since post-filter may discard some
        fetch_k = top_k * 3 if where else top_k

        distances, indices = self._index.search(query_vec, fetch_k)
        distances = distances[0]  # single query
        indices = indices[0]

        results: list[SearchResult] = []

        for dist, idx in zip(distances, indices):
            if idx == -1:
                continue  # FAISS returns -1 for empty slots

            doc_id = f"doc_{idx}"
            entry = self._docstore.get(doc_id)
            if entry is None:
                continue

            # Post-filter by metadata (FAISS has no native filtering)
            if where and not self._matches_filter(entry["metadata"], where):
                continue

            results.append(
                SearchResult(
                    doc_id=doc_id,
                    text=entry["text"],
                    metadata=entry["metadata"],
                    distance=float(dist),
                )
            )

            if len(results) >= top_k:
                break

        return results

    @staticmethod
    def _matches_filter(metadata: dict, where: dict) -> bool:
        """
        Simple metadata filter matching.

        Supports exact match: {"source": "YC"}
        Does NOT support ChromaDB-style operators ($gte, $lte, etc.) yet.
        """
        for key, value in where.items():
            if metadata.get(key) != value:
                return False
        return True

    # ── Reset ────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._index = None
        self._docstore = {}

        # Remove files from disk
        if self._index_path.exists():
            self._index_path.unlink()
        if self._docstore_path.exists():
            self._docstore_path.unlink()

        logger.info("FAISS index reset — 0 documents")
