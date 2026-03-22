"""Graph builder — assembles and compiles the BizIntel LangGraph pipeline.

Usage::

    from bizintel.graph.builder import build_graph

    graph = build_graph(retriever=retriever, llm_client=llm_client)
    result = graph.invoke({"user_query": "Find startups similar to Stripe"})
"""

from __future__ import annotations

import logging

from langgraph.graph import StateGraph, END

from bizintel.graph.state import InputState, OutputState, PrivateState
from bizintel.graph.nodes.classify import make_classify_node
from bizintel.graph.nodes.expand_query import make_expand_query_node
from bizintel.graph.nodes.retrieve import make_retrieve_node
from bizintel.graph.nodes.confidence import confidence_gate_node
from bizintel.graph.nodes.generate_similar import make_generate_similar_node
from bizintel.graph.nodes.generate_swot import make_generate_swot_node
from bizintel.graph.nodes.generate_competitor import make_generate_competitor_node
from bizintel.graph.nodes.generate_comparison import make_generate_comparison_node
from bizintel.graph.nodes.generate_ecosystem import make_generate_ecosystem_node
from bizintel.graph.nodes.validate import make_validate_node
from bizintel.graph.nodes.rewrite import make_rewrite_node
from bizintel.graph.edges import route_after_confidence, route_after_validate
from bizintel.rag.retriever import StartupRetriever

logger = logging.getLogger(__name__)

_REFUSAL_MSG = (
    "🚫 **I don't have enough information to answer this reliably.**\n\n"
    "The retrieved context doesn't appear relevant to your query. "
    "This can happen when:\n"
    "- The topic is outside the startup database's coverage\n"
    "- The query is too vague or ambiguous\n"
    "- No matching startups exist in our 134K dataset\n\n"
    "*Try rephrasing your query or broadening your search terms.*"
)


def _make_refuse_node():
    """Returns a node that sets the refusal answer."""

    def refuse_node(state) -> dict:
        return {"answer": _REFUSAL_MSG}

    return refuse_node


def build_graph(
    retriever: StartupRetriever,
    llm_client,
    checkpointer=None,
):
    """Build and compile the BizIntel LangGraph pipeline.

    Dependencies are injected here and captured by node closures.
    State remains pure serialisable data.

    Parameters
    ----------
    retriever : StartupRetriever
        The existing hybrid retriever (semantic + BM25 + reranker).
    llm_client
        OpenAI-compatible LLM client (Groq or OpenAI).
    checkpointer : optional
        LangGraph checkpointer for persistence/HITL (None for v1).

    Returns
    -------
    CompiledGraph
        Ready to call via ``graph.invoke({"user_query": "..."})``.
    """
    # ── Create node functions (closures capture dependencies) ─────────
    classify = make_classify_node(llm_client)
    expand = make_expand_query_node(llm_client)
    retrieve = make_retrieve_node(retriever)
    generate_similar = make_generate_similar_node(llm_client)
    generate_swot = make_generate_swot_node(llm_client)
    generate_competitor = make_generate_competitor_node(llm_client)
    generate_comparison = make_generate_comparison_node(llm_client)
    generate_ecosystem = make_generate_ecosystem_node(llm_client)
    validate = make_validate_node(llm_client)
    rewrite = make_rewrite_node(llm_client)
    refuse = _make_refuse_node()

    # ── Build the graph ──────────────────────────────────────────────
    graph = StateGraph(
        PrivateState,
        input=InputState,
        output=OutputState,
    )

    # Add nodes
    graph.add_node("classify", classify)
    graph.add_node("expand_query", expand)
    graph.add_node("retrieve", retrieve)
    graph.add_node("confidence_gate", confidence_gate_node)
    graph.add_node("generate_similar", generate_similar)
    graph.add_node("generate_swot", generate_swot)
    graph.add_node("generate_competitor", generate_competitor)
    graph.add_node("generate_comparison", generate_comparison)
    graph.add_node("generate_ecosystem", generate_ecosystem)
    graph.add_node("validate", validate)
    graph.add_node("rewrite", rewrite)
    graph.add_node("refuse", refuse)

    # ── Edges: entry → classify → expand_query → retrieve ─────────────
    graph.set_entry_point("classify")
    graph.add_edge("classify", "expand_query")
    graph.add_edge("expand_query", "retrieve")
    graph.add_edge("retrieve", "confidence_gate")

    # ── Conditional: after confidence gate → type-specific generate ───
    graph.add_conditional_edges(
        "confidence_gate",
        route_after_confidence,
        {
            "generate_similar":    "generate_similar",
            "generate_swot":       "generate_swot",
            "generate_competitor":  "generate_competitor",
            "generate_comparison":  "generate_comparison",
            "generate_ecosystem":   "generate_ecosystem",
            "rewrite":             "rewrite",
            END:                   "refuse",
        },
    )

    graph.add_edge("refuse", END)

    # All type-specific generate nodes converge to validate
    graph.add_edge("generate_similar", "validate")
    graph.add_edge("generate_swot", "validate")
    graph.add_edge("generate_competitor", "validate")
    graph.add_edge("generate_comparison", "validate")
    graph.add_edge("generate_ecosystem", "validate")

    # ── Conditional: after validate ──────────────────────────────────
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            END: END,
            "rewrite": "rewrite",
        },
    )

    # ── Rewrite loops back to expand_query ───────────────────────────
    graph.add_edge("rewrite", "expand_query")

    # ── Compile ──────────────────────────────────────────────────────
    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    compiled = graph.compile(**compile_kwargs)
    logger.info("BizIntel LangGraph pipeline compiled (12 nodes, 2 conditional edges)")

    return compiled
