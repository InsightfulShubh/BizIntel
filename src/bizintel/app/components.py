"""
UI components — sidebar, header, chat message renderers, source cards.

Keeps the main app file clean by extracting all widget-building logic here.
"""

from __future__ import annotations

import streamlit as st

from bizintel.config.settings import ANALYSIS_TYPES, TOP_K


# ── Analysis type metadata (icons + descriptions for sidebar) ────────────

ANALYSIS_META: dict[str, dict] = {
    "auto":       {"icon": "🤖", "label": "Auto Detect",      "desc": "Let AI choose the best analysis format"},
    "similar":    {"icon": "🔍", "label": "Similar Startups",  "desc": "Find startups similar to your description"},
    "swot":       {"icon": "📊", "label": "SWOT Analysis",     "desc": "Strengths, Weaknesses, Opportunities, Threats"},
    "competitor": {"icon": "⚔️", "label": "Competitor Analysis","desc": "Map the competitive landscape"},
    "comparison": {"icon": "⚖️", "label": "Side-by-Side",     "desc": "Compare startups head-to-head"},
    "ecosystem":  {"icon": "🌐", "label": "Ecosystem Map",     "desc": "Explore an industry ecosystem"},
}


# ── Custom CSS ───────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
    /* Main container */
    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        width: 320px !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* Source card styling */
    .source-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 4px solid #4A90D9;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
        font-size: 0.88rem;
        line-height: 1.5;
        color: #2c3e50;
    }

    .source-card strong {
        color: #1a252f;
    }

    .source-card .meta-tag {
        display: inline-block;
        background: #4A90D9;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-right: 6px;
        margin-top: 4px;
    }

    /* Stat cards in sidebar */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        color: white;
        margin-bottom: 8px;
    }

    .stat-card .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.2;
    }

    .stat-card .stat-label {
        font-size: 0.78rem;
        opacity: 0.9;
        margin-top: 2px;
    }

    /* Chat input area */
    .stChatInput {
        border-top: 1px solid #e0e0e0;
    }

    /* Welcome message */
    .welcome-box {
        text-align: center;
        padding: 3rem 2rem;
        color: #6c757d;
    }

    .welcome-box h2 {
        color: #343a40;
        margin-bottom: 0.5rem;
    }

    /* Quick action buttons */
    .quick-action {
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s;
    }

    .quick-action:hover {
        border-color: #4A90D9;
        background: #f0f6ff;
    }
</style>
"""


# ── Header ───────────────────────────────────────────────────────────────


def render_header() -> None:
    """Render the app header with branding."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────


def render_sidebar(doc_count: int) -> None:
    """
    Render the sidebar with analysis controls, filters, and stats.

    Writes selected values directly into st.session_state.
    """
    with st.sidebar:
        # ── Branding
        st.markdown("## 🧠 BizIntel")
        st.caption("AI-Powered Startup Intelligence Engine")
        st.divider()

        # ── Analysis Type selector
        st.markdown("#### 🎯 Analysis Type")

        # Build display labels with icons
        options = [
            f"{ANALYSIS_META[t]['icon']}  {ANALYSIS_META[t]['label']}"
            for t in ANALYSIS_TYPES
        ]

        selected_idx = st.selectbox(
            "Choose analysis type",
            range(len(options)),
            format_func=lambda i: options[i],
            index=ANALYSIS_TYPES.index(st.session_state.analysis_type),
            label_visibility="collapsed",
            help="Auto Detect lets the AI choose the best format for your query.",
        )
        st.session_state.analysis_type = ANALYSIS_TYPES[selected_idx]

        # Show description of selected type
        sel = ANALYSIS_META[st.session_state.analysis_type]
        st.caption(f"ℹ️ {sel['desc']}")

        st.divider()

        # ── Data Source filter
        st.markdown("#### 📂 Data Source")
        source_options = ["All", "YC (Y Combinator)", "Crunchbase"]
        source_choice = st.radio(
            "Filter by source",
            source_options,
            index=source_options.index(st.session_state.source_filter)
            if st.session_state.source_filter in source_options
            else 0,
            label_visibility="collapsed",
        )
        st.session_state.source_filter = source_choice

        st.divider()

        # ── Results count slider
        st.markdown("#### 🔢 Results to Retrieve")
        top_k = st.slider(
            "Number of startups to analyse",
            min_value=3,
            max_value=20,
            value=st.session_state.top_k,
            step=1,
            label_visibility="collapsed",
        )
        st.session_state.top_k = top_k

        st.divider()

        # ── Database stats
        st.markdown("#### 📈 Database Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-value">{doc_count:,}</div>'
                f'<div class="stat-label">Startups Indexed</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-value">{len(st.session_state.messages) // 2}</div>'
                f'<div class="stat-label">Queries Made</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Clear conversation
        if st.button("🗑️  Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ── Welcome screen (shown when chat is empty) ───────────────────────────


def render_welcome() -> None:
    """Show a welcome screen with example queries when chat is empty."""
    st.markdown(
        '<div class="welcome-box">'
        "<h2>Welcome to BizIntel 🧠</h2>"
        "<p>Your AI-powered startup intelligence analyst.<br>"
        "Ask anything about 134,000+ startups from YC & Crunchbase.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Example queries as quick-action buttons
    st.markdown("#### 💡 Try asking:")
    examples = [
        ("🔍 Find similar", "Find startups similar to Stripe in the fintech space"),
        ("📊 SWOT", "SWOT analysis of AI healthcare startups founded after 2020"),
        ("⚔️ Competitors", "Who are the main competitors in the food delivery space?"),
        ("⚖️ Compare", "Compare YC-backed edtech startups vs Crunchbase edtech companies"),
        ("🌐 Ecosystem", "Map the autonomous vehicle startup ecosystem"),
    ]

    cols = st.columns(len(examples))
    for col, (icon_label, query) in zip(cols, examples):
        with col:
            if st.button(icon_label, key=f"ex_{icon_label}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": query})
                st.session_state._pending_query = query
                st.rerun()


# ── Chat message rendering ───────────────────────────────────────────────


def render_chat_history() -> None:
    """Replay all messages in the chat history."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🧠"):
            st.markdown(msg["content"])

            # Render source cards if present (assistant messages)
            if msg.get("sources"):
                render_sources(msg["sources"])


def render_sources(sources: list[dict]) -> None:
    """Render retrieved source documents as styled cards inside an expander."""
    with st.expander(f"📚 View {len(sources)} source documents", expanded=False):
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {})
            source_tag = meta.get("source", "Unknown")
            country = meta.get("country", "")
            founded = meta.get("founded_year", "")

            # Build meta tags
            tags_html = f'<span class="meta-tag">{source_tag}</span>'
            if country:
                tags_html += f'<span class="meta-tag">📍 {country}</span>'
            if founded and founded != "":
                tags_html += f'<span class="meta-tag">📅 {founded}</span>'

            # Truncate long text
            text = src.get("text", "")
            display_text = text[:300] + "…" if len(text) > 300 else text

            st.markdown(
                f'<div class="source-card">'
                f"<strong>Source #{i}</strong> &nbsp; {tags_html}<br><br>"
                f"{display_text}"
                f"</div>",
                unsafe_allow_html=True,
            )


# ── Source filter → metadata dict ────────────────────────────────────────


def build_where_filter() -> dict | None:
    """Convert the sidebar source filter into a vector-store `where` dict."""
    source = st.session_state.source_filter
    if source == "YC (Y Combinator)":
        return {"source": "YC"}
    if source == "Crunchbase":
        return {"source": "Crunchbase"}
    return None
