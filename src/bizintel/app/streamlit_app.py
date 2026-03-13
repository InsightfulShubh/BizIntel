"""
BizIntel — Streamlit UI entry point.

Run:
    uv run streamlit run src/bizintel/app/streamlit_app.py

Architecture:
    ┌──────────────┐   ┌──────────────────────────────────────┐
    │   Sidebar     │   │  Main area                           │
    │  ─────────── │   │  ┌──────────────────────────────────┐│
    │  Analysis Type│   │  │  Chat history (scrollable)       ││
    │  Data Source  │   │  │  - User messages                 ││
    │  Top-K slider │   │  │  - AI responses + source cards   ││
    │  DB Stats     │   │  └──────────────────────────────────┘│
    │  Clear button │   │  ┌──────────────────────────────────┐│
    └──────────────┘   │  │  Chat input (fixed bottom)       ││
                        │  └──────────────────────────────────┘│
                        └──────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import time

import streamlit as st

from bizintel.app.state import (
    init_session_state,
    load_embedder,
    load_vector_store,
    load_chain,
)
from bizintel.app.components import (
    render_header,
    render_sidebar,
    render_welcome,
    render_chat_history,
    render_sources,
    build_where_filter,
)

# ── Logging ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s",
)
logger = logging.getLogger(__name__)

# ── Page config (must be first Streamlit call) ───────────────────────────

st.set_page_config(
    page_title="BizIntel — Startup Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Initialise state + load resources ────────────────────────────────────

init_session_state()

embedder = load_embedder()
store = load_vector_store()
chain = load_chain(embedder, store)

doc_count = store.count


# ── Render UI ────────────────────────────────────────────────────────────

render_header()
render_sidebar(doc_count=doc_count)


# ── Guard: no data indexed yet ───────────────────────────────────────────

if doc_count == 0:
    st.warning(
        "⚠️ **No startups indexed yet.**  \n"
        "Run the batch embedding script first:  \n"
        "```\n"
        "uv run python scripts/batch_embed.py\n"
        "```",
        icon="🔧",
    )
    st.stop()


# ── Query handler (defined before use) ───────────────────────────────────


def _process_query(query: str) -> None:
    """Run the RAG chain and render the response."""

    # Show user message
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(query)

    # Show assistant response
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Analyzing startups…"):
            start = time.perf_counter()

            try:
                result = chain.analyze(
                    query=query,
                    analysis_type=st.session_state.analysis_type,
                    top_k=st.session_state.top_k,
                    where=build_where_filter(),
                )
                elapsed = time.perf_counter() - start

                # Display the answer
                st.markdown(result["answer"])

                # Display source documents
                if result["sources"]:
                    render_sources(result["sources"])

                # Footer with metadata
                st.caption(
                    f"⏱️ {elapsed:.1f}s · "
                    f"📋 {result['analysis_type']} · "
                    f"🤖 {result['model']} · "
                    f"📚 {len(result['sources'])} sources"
                )

                # Save to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })

            except Exception as e:
                error_msg = f"❌ **Error:** {str(e)}"
                st.error(error_msg)
                logger.exception("Chain error: %s", e)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })


# ── Main chat area ───────────────────────────────────────────────────────

# Show welcome screen or chat history
if not st.session_state.messages:
    render_welcome()
else:
    render_chat_history()


# ── Handle user input ────────────────────────────────────────────────────

user_input = st.chat_input(
    placeholder="Ask about startups… e.g., 'Find AI healthcare startups in the US'",
)

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    _process_query(user_input)

# Handle pending query from quick-action buttons
pending = st.session_state.pop("_pending_query", None)
if pending:
    _process_query(pending)
