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
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

# Load .env from project root (two levels up from this file)
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from bizintel.app.state import (
    init_session_state,
    load_embedder,
    load_vector_store,
    load_graph,
)
from bizintel.app.components import (
    render_header,
    render_sidebar,
    render_welcome,
    render_chat_history,
    render_sources,
    build_where_filter,
    format_confidence_badge,
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

# Assign a unique thread_id for this Streamlit session (persists across reruns)
if st.session_state.thread_id is None:
    st.session_state.thread_id = str(uuid.uuid4())

embedder = load_embedder()
store = load_vector_store()
graph = load_graph(embedder, store)

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
        "uv run python -m bizintel.pipeline.batch_embed --reset\n"
        "```",
        icon="🔧",
    )
    st.stop()


# ── Query handler (defined before use) ───────────────────────────────────


# Friendly labels shown in the streaming status widget
_NODE_LABELS: dict[str, str] = {
    "classify": "🏷️ Classifying query…",
    "expand_query": "🔍 Expanding search terms…",
    "retrieve": "📚 Retrieving relevant startups…",
    "confidence_gate": "🎯 Checking relevance confidence…",
    "generate_similar": "✍️ Generating similar-startup analysis…",
    "generate_swot": "✍️ Generating SWOT analysis…",
    "generate_competitor": "✍️ Generating competitor analysis…",
    "generate_comparison": "✍️ Generating comparison analysis…",
    "generate_ecosystem": "✍️ Generating ecosystem analysis…",
    "validate": "✅ Validating answer quality…",
    "rewrite": "🔄 Rewriting query…",
    "refuse": "🚫 Not enough context — refusing…",
    "web_search": "🌐 Searching the web for more info…",
    "web_review": "🧑 Reviewing web results…",
    "record_turn": "💾 Saving conversation turn…",
}


def _stream_graph(input_or_command, config: dict) -> tuple[dict, bool]:
    """Stream the graph and return (accumulated_result, was_interrupted).

    Works for both initial invocations (input_or_command is a dict like
    {"user_query": ...}) and HITL resumes (input_or_command is a Command).
    """
    result: dict = {}
    with st.status("🧠 Thinking…", expanded=True) as status:
        for chunk in graph.stream(
            input_or_command,
            config=config,
            stream_mode="updates",
        ):
            for node_name, node_output in chunk.items():
                # Skip interrupt metadata — not a regular node output
                if node_name == "__interrupt" or not isinstance(node_output, dict):
                    continue
                label = _NODE_LABELS.get(node_name, f"⚙️ {node_name}…")
                status.update(label=label)
                st.write(label)
                result.update(node_output)

        # Check if graph paused at an interrupt
        graph_state = graph.get_state(config)
        pending_interrupts = [
            intr
            for task in (graph_state.tasks or [])
            for intr in (task.interrupts or [])
        ]

        if pending_interrupts:
            status.update(label="🧑 Waiting for your review…", state="running", expanded=False)
            return result, True

        # Merge full graph state into result so fields set by earlier nodes
        # (e.g. confidence, best_score from web_search) are not lost on resume
        final_values = graph_state.values
        if isinstance(final_values, dict):
            for key in ("confidence", "best_score", "analysis_type", "source_docs", "web_searched"):
                if key not in result and key in final_values:
                    result[key] = final_values[key]

        status.update(label="✅ Done", state="complete", expanded=False)
        return result, False


def _render_answer(result: dict, elapsed: float) -> None:
    """Render the final answer, sources, and footer inside a chat bubble."""
    st.markdown(result.get("answer", ""))

    source_docs = result.get("source_docs", [])
    if source_docs:
        render_sources(source_docs)

    confidence = result.get("confidence", "high")
    best_score = result.get("best_score", 0.0)
    badge_html = format_confidence_badge(confidence, best_score)

    st.markdown(
        f"⏱️ {elapsed:.1f}s · "
        f"📋 {result.get('analysis_type', '')} · "
        f"📚 {len(source_docs)} sources · "
        f"{badge_html}",
        unsafe_allow_html=True,
    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result.get("answer", ""),
        "sources": source_docs,
        "confidence": confidence,
        "best_score": best_score,
    })


def _process_query(query: str) -> None:
    """Run the RAG chain via streaming and render progress + response."""

    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🧠"):
        start = time.perf_counter()
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        try:
            result, interrupted = _stream_graph({"user_query": query}, config)

            if interrupted:
                # Graph paused at web_review — save state for HITL UI
                graph_state = graph.get_state(config)
                interrupt_data = graph_state.tasks[0].interrupts[0].value
                st.session_state._hitl_pending = True
                st.session_state._hitl_data = interrupt_data
                st.info("🌐 Web search completed. Please review the results below.")
                st.rerun()  # rerun so _render_hitl_review() renders the review UI

            elapsed = time.perf_counter() - start
            _render_answer(result, elapsed)

        except Exception as e:
            error_msg = f"❌ **Error:** {str(e)}"
            st.error(error_msg)
            logger.exception("Graph error: %s", e)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
            })


def _render_hitl_review() -> None:
    """Show web review UI when graph is paused at HITL interrupt."""
    data = st.session_state._hitl_data
    if not data:
        return

    results = data.get("results", [])

    st.markdown(
        '<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); '
        'border-radius: 12px; padding: 24px; margin-bottom: 16px;">'
        '<h3 style="margin-top: 0; color: #e0e0e0;">🌐 Review Web Search Results</h3>'
        '<p style="color: #a0a0a0; margin-bottom: 0;">I searched the web since the local database didn\'t have enough info. '
        'Select which sources to use for the analysis:</p></div>',
        unsafe_allow_html=True,
    )

    # Result cards with checkboxes
    selected = []
    for r in results:
        idx = r["index"]
        title = r.get("title", f"Result {idx + 1}")
        url = r.get("url", "")
        snippet = r.get("snippet", "")

        with st.container():
            col_check, col_content = st.columns([0.05, 0.95])
            with col_check:
                checked = st.checkbox("", value=True, key=f"hitl_check_{idx}", label_visibility="collapsed")
            with col_content:
                st.markdown(
                    f'<div style="background: #1e1e2e; border-left: 3px solid {"#4A90D9" if checked else "#555"}; '
                    f'border-radius: 8px; padding: 12px 16px; margin-bottom: 4px; '
                    f'opacity: {"1" if checked else "0.5"};">'
                    f'<strong style="color: #e0e0e0; font-size: 0.95rem;">{title}</strong><br>'
                    f'<span style="color: #4A90D9; font-size: 0.8rem;">{url}</span><br>'
                    f'<span style="color: #a0a0a0; font-size: 0.85rem;">{snippet}</span></div>',
                    unsafe_allow_html=True,
                )
            if checked:
                selected.append(idx)

    st.markdown("")  # spacer
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("✅ Use Selected", type="primary", use_container_width=True):
            st.session_state._hitl_resume = selected
            st.rerun()

    with col2:
        if st.button("📋 Use All", use_container_width=True):
            st.session_state._hitl_resume = "all"
            st.rerun()

    with col3:
        count = len(selected)
        st.markdown(
            f'<div style="text-align: center; padding: 8px; color: #a0a0a0;">'
            f'{count}/{len(results)} selected</div>',
            unsafe_allow_html=True,
        )


def _resume_graph(resume_value) -> None:
    """Resume the graph after HITL approval and display the answer."""
    # Clear HITL state
    st.session_state._hitl_pending = False
    st.session_state._hitl_data = None
    st.session_state._hitl_resume = None

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    start = time.perf_counter()

    with st.chat_message("assistant", avatar="🧠"):
        try:
            result, interrupted = _stream_graph(Command(resume=resume_value), config)
            elapsed = time.perf_counter() - start

            if interrupted:
                st.warning("Unexpected second interrupt — please try again.")
                return

            _render_answer(result, elapsed)

        except Exception as e:
            error_msg = f"❌ **Error resuming graph:** {str(e)}"
            st.error(error_msg)
            logger.exception("Graph resume error: %s", e)
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

# ── HITL: show review UI if graph is paused ──────────────────────────────

if st.session_state._hitl_pending:
    if st.session_state._hitl_resume is not None:
        # User approved — resume graph at full width
        _resume_graph(st.session_state._hitl_resume)
    else:
        # Still waiting — show review UI
        _render_hitl_review()


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
