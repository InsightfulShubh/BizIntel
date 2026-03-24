"""Classify node — LLM determines the analysis type from the user query."""

from __future__ import annotations

import logging

from bizintel.config.settings import LLM_MODEL, ANALYSIS_TYPES
from bizintel.graph.utils.history import format_history_context

logger = logging.getLogger(__name__)

_CLASSIFY_PROMPT = (
    "You are a query classifier for a startup intelligence system. "
    "Given a user's question, classify it into exactly ONE of these analysis types:\n\n"
    "- similar: User wants to find startups similar to a named company or description\n"
    "- swot: User wants a SWOT analysis (strengths, weaknesses, opportunities, threats)\n"
    "- competitor: User wants competitive landscape or competitor mapping\n"
    "- comparison: User wants a side-by-side comparison of specific startups or segments\n"
    "- ecosystem: User wants to explore an industry ecosystem, sub-segments, or trends\n\n"
    "If the query is a follow-up (e.g. 'their competitors', 'now do a SWOT'), use the "
    "conversation history to understand what the user is referring to.\n\n"
    "{history}"
    "User query: {query}\n"
    "Type:"
)

# Types the classifier is allowed to return (excludes "auto")
_VALID_TYPES = {t for t in ANALYSIS_TYPES if t != "auto"}


def make_classify_node(llm_client):
    """Factory: returns a classify node that closes over the LLM client."""

    def classify_node(state) -> dict:
        """Classify the user query into an analysis type."""
        query = state.user_query
        history = format_history_context(getattr(state, "conversation_history", []))

        try:
            response = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "user", "content": _CLASSIFY_PROMPT.format(query=query, history=history)},
                ],
                temperature=0.0,
                max_tokens=10,
            )
            raw = response.choices[0].message.content.strip().lower()
            analysis_type = raw if raw in _VALID_TYPES else "similar"
        except Exception as exc:
            logger.warning("Classification failed (%s), defaulting to 'similar'", exc)
            analysis_type = "similar"

        logger.info("Classified '%s' → %s", query[:60], analysis_type)
        return {"analysis_type": analysis_type}

    return classify_node


# Convenience alias when LLM client isn't needed at import time
classify_node = make_classify_node
