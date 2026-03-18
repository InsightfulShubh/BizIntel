"""
BM25 Keyword Search Index — complements vector (semantic) search.

Builds an in-memory BM25Okapi index from all document texts stored in the
vector store.  At query time, tokenises the query and returns top-K docs
ranked by keyword relevance.

Used in hybrid search:
    Semantic (ChromaDB) ──┐
                          ├── RRF merge → Reranker → top-5 final
    BM25 (this module) ───┘
"""

from __future__ import annotations

import logging
import re
import time

import numpy as np
from rank_bm25 import BM25Okapi

from bizintel.config.settings import BM25_TOP_K
from bizintel.vectorstore.base import SearchResult

logger = logging.getLogger(__name__)

# Simple tokeniser — lowercase, keep alphanumeric tokens ≥ 2 chars
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def _tokenise(text: str) -> list[str]:
    """Lowercase + split into alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """
    In-memory BM25 keyword index over startup documents.

    Build once at startup, then call ``search()`` per query.

    Parameters
    ----------
    doc_ids : list[str]
        Document IDs matching the vector store (e.g. "doc_0", "doc_1", …).
    texts : list[str]
        Full document texts (same order as *doc_ids*).
    metadatas : list[dict]
        Metadata dicts (same order).
    """

    def __init__(
        self,
        doc_ids: list[str],
        texts: list[str],
        metadatas: list[dict],
    ) -> None:
        assert len(doc_ids) == len(texts) == len(metadatas), (
            "doc_ids, texts, and metadatas must have the same length"
        )

        self._doc_ids = doc_ids
        self._texts = texts
        self._metadatas = metadatas

        logger.info("Building BM25 index over %d documents…", len(texts))
        start = time.perf_counter()

        tokenised = [_tokenise(t) for t in texts]
        self._bm25 = BM25Okapi(tokenised)

        elapsed = time.perf_counter() - start
        logger.info("BM25 index built in %.2fs", elapsed)

    @property
    def count(self) -> int:
        return len(self._doc_ids)

    def search(
        self,
        query: str,
        top_k: int = BM25_TOP_K,
    ) -> list[SearchResult]:
        """
        Score all documents against the query using BM25 and return *top_k*.

        Parameters
        ----------
        query : str
            Natural language query (will be tokenised internally).
        top_k : int
            Number of results to return.

        Returns
        -------
        list[SearchResult]
            Ranked by BM25 score (best first).  The ``distance`` field
            stores the **negative** BM25 score so that lower distance =
            better match, staying consistent with vector search semantics.
        """
        tokens = _tokenise(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)          # ndarray, len = corpus size
        top_indices = np.argsort(scores)[::-1][:top_k]   # descending

        results: list[SearchResult] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                break  # no point returning zero-score docs
            results.append(
                SearchResult(
                    doc_id=self._doc_ids[idx],
                    text=self._texts[idx],
                    metadata=self._metadatas[idx],
                    distance=-score,  # negative so lower = better (like cosine)
                )
            )

        logger.info("BM25 search returned %d results (top score=%.2f)",
                     len(results), -results[0].distance if results else 0)
        return results
