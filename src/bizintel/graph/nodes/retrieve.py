"""Retrieve node — runs the hybrid retrieval pipeline (semantic + BM25 + reranker)."""

from __future__ import annotations

import logging

from bizintel.config.settings import TOP_K
from bizintel.rag.retriever import StartupRetriever

logger = logging.getLogger(__name__)


def make_retrieve_node(retriever: StartupRetriever):
    """Factory: returns a retrieve node that closes over the retriever."""

    def retrieve_node(state) -> dict:
        """Retrieve relevant startup documents for the expanded query."""
        expanded_query = state.expanded_query

        result = retriever.retrieve(query=expanded_query, top_k=TOP_K)

        source_docs = [
            {
                "doc_id": doc.doc_id,
                "text": doc.text,
                "metadata": doc.metadata,
                "distance": doc.distance,
            }
            for doc in result.documents
        ]

        logger.info(
            "Retrieved %d docs | best_score=%.3f mean_score=%.3f",
            len(source_docs), result.best_score, result.mean_score,
        )

        return {
            "source_docs": source_docs,
            "best_score": result.best_score,
            "mean_score": result.mean_score,
        }

    return retrieve_node
