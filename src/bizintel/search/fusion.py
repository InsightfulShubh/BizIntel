"""
Weighted Reciprocal Rank Fusion (RRF) — merges ranked lists from different
search systems with configurable per-list weights.

Given two ranked lists (e.g. semantic + BM25), RRF assigns each document a
combined score:

    RRF_score(doc) = Σ  weight_i / (k + rank_in_list_i)

Documents appearing in **both** lists get boosted (summed contributions).
The constant *k* (default 60) dampens the influence of rank position.
Weights allow semantic search to dominate while BM25 still contributes.

Reference: Cormack, Clarke, Büttcher — "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods" (SIGIR 2009).
"""

from __future__ import annotations

import logging
from collections import defaultdict

from bizintel.config.settings import RRF_K
from bizintel.vectorstore.base import SearchResult

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    weights: list[float] | None = None,
    k: int = RRF_K,
    top_k: int = 20,
) -> list[SearchResult]:
    """
    Merge multiple ranked SearchResult lists using weighted RRF.

    Parameters
    ----------
    ranked_lists : list[list[SearchResult]]
        Two or more ranked lists (best-first order).
    weights : list[float] | None
        Per-list weight (same length as *ranked_lists*).
        Higher weight = more influence on the final ranking.
        If None, all lists are weighted equally (1.0).
    k : int
        Smoothing constant (default 60).
    top_k : int
        Number of results to return after fusion.

    Returns
    -------
    list[SearchResult]
        Merged list sorted by weighted RRF score (best first).
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    assert len(weights) == len(ranked_lists), (
        f"weights ({len(weights)}) must match ranked_lists ({len(ranked_lists)})"
    )

    # Accumulate weighted RRF scores keyed by doc_id
    rrf_scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, SearchResult] = {}  # keep the first occurrence

    for weight, ranked_list in zip(weights, ranked_lists):
        for rank, result in enumerate(ranked_list, start=1):
            rrf_scores[result.doc_id] += weight / (k + rank)
            if result.doc_id not in doc_map:
                doc_map[result.doc_id] = result

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

    merged = [doc_map[did] for did in sorted_ids]

    logger.info(
        "RRF merged %d lists (weights=%s) → %d unique docs → top %d returned",
        len(ranked_lists),
        [f"{w:.1f}" for w in weights],
        len(rrf_scores),
        len(merged),
    )

    return merged
