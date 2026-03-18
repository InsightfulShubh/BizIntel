"""
Retriever — encodes a user query and fetches top-K documents from the
vector store, optionally using hybrid search (semantic + BM25) and
cross-encoder reranking.

Uses dependency injection: embedder, store, reranker, and bm25_index are
passed in, not created internally.
"""

from __future__ import annotations

import logging

from bizintel.config.settings import (
    TOP_K,
    RERANK_ENABLED,
    RERANK_TOP_K_INITIAL,
    HYBRID_SEARCH_ENABLED,
    BM25_TOP_K,
    RRF_WEIGHT_SEMANTIC,
    RRF_WEIGHT_BM25,
)
from bizintel.embeddings.embedder import StartupEmbedder
from bizintel.vectorstore.base import VectorStoreBase, SearchResult

logger = logging.getLogger(__name__)


class StartupRetriever:
    """Encodes a query and retrieves similar documents from the vector store."""

    def __init__(
        self,
        embedder: StartupEmbedder,
        store: VectorStoreBase,
        reranker=None,
        bm25_index=None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._reranker = reranker
        self._bm25 = bm25_index

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """
        Encode the query and return the top-K most relevant startup documents.

        Pipeline (when all components enabled):
            1. Semantic search (ChromaDB) → top 20
            2. BM25 keyword search         → top 20
            3. RRF fusion (merge + dedup)  → top 20
            4. Cross-encoder reranker      → top 5

        Parameters
        ----------
        query : str
            Natural language query from the user.
        top_k : int
            Number of documents to return to the caller.
        where : dict | None
            Optional metadata filter (e.g. {"source": "YC"}).

        Returns
        -------
        list[SearchResult]
            Ranked by relevance (best first).
        """
        logger.info("Retrieving top-%d for: '%s'", top_k, query[:80])

        use_reranker = RERANK_ENABLED and self._reranker is not None
        use_hybrid = HYBRID_SEARCH_ENABLED and self._bm25 is not None

        # ── Step 1: Semantic search ──────────────────────────────────
        semantic_k = RERANK_TOP_K_INITIAL if (use_reranker or use_hybrid) else top_k
        query_embedding = self._embedder.encode_single(query)
        semantic_results = self._store.query(
            query_embedding, top_k=semantic_k, where=where,
        )

        # ── Step 2: BM25 keyword search (if enabled) ────────────────
        if use_hybrid:
            from bizintel.search.fusion import reciprocal_rank_fusion

            bm25_results = self._bm25.search(query, top_k=BM25_TOP_K)

            # ── Step 3: Weighted RRF merge ────────────────────────────
            candidates = reciprocal_rank_fusion(
                ranked_lists=[semantic_results, bm25_results],
                weights=[RRF_WEIGHT_SEMANTIC, RRF_WEIGHT_BM25],
                top_k=RERANK_TOP_K_INITIAL,
            )
        else:
            candidates = semantic_results

        # ── Step 4: Rerank (if enabled) ──────────────────────────────
        if use_reranker and len(candidates) > top_k:
            results = self._reranker.rerank(query, candidates, top_k=top_k)
        else:
            results = candidates[:top_k]

        logger.info(
            "Retrieved %d results (hybrid=%s, reranked=%s)",
            len(results), use_hybrid, use_reranker,
        )
        return results
