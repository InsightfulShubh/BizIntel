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
    VECTOR_STORE_BACKEND,
    TOP_K,
    RERANK_ENABLED,
    HYBRID_SEARCH_ENABLED,
    CONVERSATIONS_DB_PATH,
    WEB_SEARCH_ENABLED,
)
from bizintel.embeddings.embedder import StartupEmbedder
from bizintel.vectorstore.base import VectorStoreBase, create_vector_store
from bizintel.rag.retriever import StartupRetriever
from bizintel.config.llm_client import get_llm_client
from bizintel.graph.builder import build_graph

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
def load_graph(
    _embedder: StartupEmbedder,
    _store: VectorStoreBase,
):
    """Build the LangGraph pipeline with SQLite checkpointer."""
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver

    reranker = load_reranker()
    bm25_index = load_bm25_index(_store)
    retriever = StartupRetriever(
        _embedder, _store, reranker=reranker, bm25_index=bm25_index,
    )
    llm_client = get_llm_client()
    conn = sqlite3.connect(str(CONVERSATIONS_DB_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)

    tavily_client = None
    if WEB_SEARCH_ENABLED:
        import os
        from tavily import TavilyClient
        api_key = os.environ.get("TAVILY_API_KEY")
        if api_key:
            tavily_client = TavilyClient(api_key=api_key)
            logger.info("Tavily web-search client initialised")
        else:
            logger.warning("WEB_SEARCH_ENABLED=True but TAVILY_API_KEY not set — disabling web search")

    return build_graph(
        retriever=retriever,
        llm_client=llm_client,
        checkpointer=checkpointer,
        tavily_client=tavily_client,
    )


# ── Session state initialisation ─────────────────────────────────────────


def init_session_state() -> None:
    """Ensure all required session_state keys exist with defaults."""
    defaults = {
        "messages": [],                          # chat history
        "source_filter": "All",                  # sidebar filter
        "top_k": TOP_K,                          # sidebar slider
        "processing": False,                     # loading guard
        "thread_id": None,                       # conversation thread for checkpointer
        "_hitl_pending": False,                  # HITL: waiting for user approval
        "_hitl_data": None,                      # HITL: interrupt payload (web results)
        "_hitl_resume": None,                    # HITL: resume value after approval
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
