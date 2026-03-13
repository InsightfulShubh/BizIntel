"""
Retriever — encodes a user query and fetches top-K documents from the
vector store.

Uses dependency injection: embedder and store are passed in, not created
internally. This makes the retriever testable and decoupled.
"""

from __future__ import annotations

import logging

from bizintel.config.settings import TOP_K
from bizintel.embeddings.embedder import StartupEmbedder
from bizintel.vectorstore.base import VectorStoreBase, SearchResult

logger = logging.getLogger(__name__)


class StartupRetriever:
    """Encodes a query and retrieves similar documents from the vector store."""

    def __init__(
        self,
        embedder: StartupEmbedder,
        store: VectorStoreBase,
    ) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """
        Encode the query and return the top-K most relevant startup documents.

        Parameters
        ----------
        query : str
            Natural language query from the user.
        top_k : int
            Number of documents to retrieve.
        where : dict | None
            Optional metadata filter (e.g. {"source": "YC"}).

        Returns
        -------
        list[SearchResult]
            Ranked by similarity (best first).
        """
        logger.info("Retrieving top-%d for: '%s'", top_k, query[:80])

        query_embedding = self._embedder.encode_single(query)
        results = self._store.query(query_embedding, top_k=top_k, where=where)

        logger.info("Retrieved %d results", len(results))
        return results
