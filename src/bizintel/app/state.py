"""
App state — session state initialisation + cached resource loaders.

Streamlit re-runs the entire script on every interaction, so:
  - @st.cache_resource → expensive one-time loads (model, vector store)
  - st.session_state   → per-session mutable state (chat history, settings)
"""

from __future__ import annotations

import logging

import streamlit as st

from bizintel.config.settings import (
    ANALYSIS_TYPES,
    DEFAULT_ANALYSIS_TYPE,
    VECTOR_STORE_BACKEND,
    TOP_K,
    RERANK_ENABLED,
    HYBRID_SEARCH_ENABLED,
)
from bizintel.embeddings.embedder import StartupEmbedder
from bizintel.vectorstore.base import VectorStoreBase, create_vector_store
from bizintel.rag.retriever import StartupRetriever
from bizintel.rag.chain import BizIntelChain

logger = logging.getLogger(__name__)


# ── Cached resource loaders (run once, persist across reruns) ────────────


@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedder() -> StartupEmbedder:
    """Load the SentenceTransformer model (once per app lifetime)."""
    return StartupEmbedder()


@st.cache_resource(show_spinner="Connecting to vector store…")
def load_vector_store(backend: str = VECTOR_STORE_BACKEND) -> VectorStoreBase:
    """Connect to the persisted vector store (once per app lifetime)."""
    store = create_vector_store(backend)
    logger.info("Vector store loaded — %d documents", store.count)
    return store


@st.cache_resource(show_spinner="Loading reranker model…")
def load_reranker():
    """Load the cross-encoder reranker (once per app lifetime)."""
    if not RERANK_ENABLED:
        return None
    from bizintel.rag.reranker import StartupReranker
    return StartupReranker()


@st.cache_resource(show_spinner="Building keyword search index…")
def load_bm25_index(_store: VectorStoreBase):
    """Build BM25 index from all documents in the vector store."""
    if not HYBRID_SEARCH_ENABLED:
        return None
    from bizintel.search.bm25_search import BM25Index
    doc_ids, texts, metadatas = _store.get_all_documents()
    return BM25Index(doc_ids, texts, metadatas)


@st.cache_resource(show_spinner="Initialising BizIntel engine…")
def load_chain(
    _embedder: StartupEmbedder,
    _store: VectorStoreBase,
) -> BizIntelChain:
    """Build the full RAG chain (once per app lifetime)."""
    reranker = load_reranker()
    bm25_index = load_bm25_index(_store)
    retriever = StartupRetriever(
        _embedder, _store, reranker=reranker, bm25_index=bm25_index,
    )
    return BizIntelChain(retriever)


# ── Session state initialisation ─────────────────────────────────────────


def init_session_state() -> None:
    """Ensure all required session_state keys exist with defaults."""
    defaults = {
        "messages": [],                          # chat history
        "analysis_type": DEFAULT_ANALYSIS_TYPE,  # sidebar dropdown
        "source_filter": "All",                  # sidebar filter
        "top_k": TOP_K,                          # sidebar slider
        "processing": False,                     # loading guard
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
