"""
Retriever — encodes a user query and fetches top-K documents from the
vector store, optionally using hybrid search (semantic + BM25) and
cross-encoder reranking.

Uses dependency injection: embedder, store, reranker, and bm25_index are
passed in, not created internally.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

from bizintel.config.settings import (
    TOP_K,
    RERANK_ENABLED,
    RERANK_TOP_K_INITIAL,
    HYBRID_SEARCH_ENABLED,
    BM25_TOP_K,
    RRF_WEIGHT_SEMANTIC,
    RRF_WEIGHT_BM25,
    GUARDRAILS_ENABLED,
    CONFIDENCE_THRESHOLD_SOFT,
    CONFIDENCE_THRESHOLD_HARD,
)
from bizintel.embeddings.embedder import StartupEmbedder
from bizintel.vectorstore.base import VectorStoreBase, SearchResult

logger = logging.getLogger(__name__)


class RetrievalResult(NamedTuple):
    """Output of the retriever — documents + confidence metadata."""

    documents: list[SearchResult]
    confidence: str        # "high" | "low" | "none"
    best_score: float      # highest reranker score (0.0 if no reranker)
    mean_score: float      # mean reranker score across kept docs


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
    ) -> RetrievalResult:
        """
        Encode the query and return the top-K most relevant startup documents.

        Pipeline (when all components enabled):
            1. Semantic search (ChromaDB) → top 20
            2. BM25 keyword search         → top 20
            3. RRF fusion (merge + dedup)  → top 20
            4. Cross-encoder reranker      → top 5
            5. Confidence check            → high / low / none

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
        RetrievalResult
            Named tuple of (documents, confidence, best_score, mean_score).
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
        reranker_scores: list[float] = []

        if use_reranker and len(candidates) > top_k:
            reranked = self._reranker.rerank(query, candidates, top_k=top_k)
            results = reranked.documents
            reranker_scores = reranked.scores
        else:
            results = candidates[:top_k]

        # ── Step 5: Compute confidence ───────────────────────────────
        if reranker_scores:
            best_score = reranker_scores[0]
            mean_score = sum(reranker_scores) / len(reranker_scores)
        else:
            # No reranker — fall back to optimistic (we can't measure)
            best_score = 1.0
            mean_score = 1.0

        # Classify confidence (only when guardrails are enabled)
        if not GUARDRAILS_ENABLED:
            confidence = "high"
        elif best_score < CONFIDENCE_THRESHOLD_HARD:
            confidence = "none"
        elif best_score < CONFIDENCE_THRESHOLD_SOFT:
            confidence = "low"
        else:
            confidence = "high"

        logger.info(
            "Retrieved %d results (hybrid=%s, reranked=%s) | "
            "confidence=%s  best_score=%.3f  mean_score=%.3f",
            len(results), use_hybrid, use_reranker,
            confidence, best_score, mean_score,
        )

        return RetrievalResult(
            documents=results,
            confidence=confidence,
            best_score=best_score,
            mean_score=mean_score,
        )
