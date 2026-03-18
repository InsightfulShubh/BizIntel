"""
Cross-Encoder Reranker — rescores candidate documents against the query.

The bi-encoder (SentenceTransformer) retrieves an initial broad set of
candidates from the vector store.  The cross-encoder then reads each
(query, document) **pair** jointly through all transformer layers to
produce a much more accurate relevance score.

Pipeline position:
    ChromaDB (top-20)  →  CrossEncoderReranker  →  top-5 (sent to LLM)

Model: ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (~22 MB, very fast)
"""

from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

from bizintel.config.settings import RERANKER_MODEL_NAME
from bizintel.vectorstore.base import SearchResult

logger = logging.getLogger(__name__)


class StartupReranker:
    """
    Reranks a list of SearchResult objects using a cross-encoder model.

    Usage::

        reranker = StartupReranker()
        best_5 = reranker.rerank(query, retrieved_docs, top_k=5)
    """

    def __init__(self, model_name: str = RERANKER_MODEL_NAME) -> None:
        logger.info("Loading cross-encoder reranker: %s", model_name)
        self._model = CrossEncoder(model_name)
        logger.info("Reranker loaded successfully")

    def rerank(
        self,
        query: str,
        retrieved_docs: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Score each candidate against the query and return the best *top_k*.

        Parameters
        ----------
        query : str
            The (already expanded) search query.
        retrieved_docs : list[SearchResult]
            Broad candidate set from the vector store (e.g. 20 results).
        top_k : int
            Number of results to keep after reranking.

        Returns
        -------
        list[SearchResult]
            The *top_k* retrieved_docs sorted by cross-encoder score (best first).
        """
        if not retrieved_docs:
            return []

        if len(retrieved_docs) <= top_k:
            logger.debug(
                "Candidate count (%d) ≤ top_k (%d) — skipping rerank",
                len(retrieved_docs), top_k,
            )
            return retrieved_docs

        # Build (query, document_text) pairs for the cross-encoder
        pairs = [(query, c.text) for c in retrieved_docs]

        # Score all pairs in a single batch
        scores = self._model.predict(pairs)

        # Pair each candidate with its cross-encoder score
        scored = sorted(
            zip(retrieved_docs, scores),
            key=lambda x: float(x[1]),
            reverse=True,           # highest score = most relevant
        )

        reranked = [candidate for candidate, _score in scored[:top_k]]

        logger.info(
            "Reranked %d → %d  |  best=%.3f  worst-kept=%.3f",
            len(retrieved_docs), len(reranked),
            float(scored[0][1]), float(scored[top_k - 1][1]),
        )

        return reranked
