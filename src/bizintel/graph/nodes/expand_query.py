"""Expand Query node — LLM rewrites the user query into a rich search query."""

from __future__ import annotations

import logging

from bizintel.config.settings import LLM_MODEL
from bizintel.graph.utils.history import format_history_context

logger = logging.getLogger(__name__)

_EXPAND_PROMPT = (
    "You are a search query optimizer for a startup database. "
    "The user will give a query that may mention a company name or be vague. "
    "Rewrite it into a rich, descriptive search query that captures the "
    "BUSINESS DOMAIN, INDUSTRY, KEY PRODUCTS, and TECHNOLOGY of what the "
    "user is looking for. Do NOT include instructions like 'find' or 'search'. "
    "If the query is a follow-up referencing previous conversation, resolve "
    "ALL pronouns and references first, then expand.\n"
    "Output ONLY the rewritten query, nothing else.\n\n"
    "{history}"
    "Examples:\n"
    "  User: Find startups similar to Stripe\n"
    "  Rewritten: online payment processing fintech developer API billing "
    "infrastructure internet commerce\n\n"
    "  User: competitors of Airbnb\n"
    "  Rewritten: short-term rental marketplace vacation home booking "
    "hospitality travel accommodation platform\n\n"
    "  User: AI healthcare startups in India\n"
    "  Rewritten: artificial intelligence healthcare medical technology "
    "health-tech diagnosis India\n\n"
    "User: {query}\nRewritten:"
)


def make_expand_query_node(llm_client):
    """Factory: returns an expand-query node that closes over the LLM client."""

    def expand_query_node(state) -> dict:
        """Expand the user query into a richer semantic search query."""
        query = state.user_query
        history = format_history_context(getattr(state, "conversation_history", []))

        try:
            response = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "user", "content": _EXPAND_PROMPT.format(query=query, history=history)},
                ],
                temperature=0.0,
                max_tokens=120,
            )
            expanded = response.choices[0].message.content.strip()
            logger.info("Query expanded: '%s' → '%s'", query[:60], expanded[:80])
        except Exception as exc:
            logger.warning("Query expansion failed (%s), using original query", exc)
            expanded = query

        return {"expanded_query": expanded}

    return expand_query_node


expand_query_node = make_expand_query_node
