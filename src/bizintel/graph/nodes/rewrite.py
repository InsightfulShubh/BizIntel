"""Rewrite Query node — modifies the expanded query for retry and increments retry_count."""

from __future__ import annotations

import logging

from bizintel.config.settings import LLM_MODEL

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = (
    "You are a search query rewriter for a startup database. "
    "The previous search query did not return good results. "
    "Rewrite the query using DIFFERENT keywords, synonyms, and angles "
    "to find relevant startups. Focus on the core intent.\n\n"
    "Original user question: {user_query}\n"
    "Previous search query: {previous_query}\n\n"
    "Provide ONLY the rewritten search query, nothing else.\n"
    "Rewritten:"
)


def make_rewrite_node(llm_client):
    """Factory: returns a rewrite node that closes over the LLM client."""

    def rewrite_node(state) -> dict:
        """Rewrite the expanded query for a retry attempt."""
        user_query = state.user_query
        previous_query = getattr(state, "expanded_query", user_query)
        retry_count = getattr(state, "retry_count", 0)

        try:
            response = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": _REWRITE_PROMPT.format(
                            user_query=user_query,
                            previous_query=previous_query,
                        ),
                    },
                ],
                temperature=0.3,
                max_tokens=120,
            )
            rewritten = response.choices[0].message.content.strip()
            logger.info(
                "Query rewritten (retry %d): '%s' → '%s'",
                retry_count + 1, previous_query[:60], rewritten[:80],
            )
        except Exception as exc:
            logger.warning("Rewrite failed (%s), reusing previous query", exc)
            rewritten = previous_query

        return {
            "expanded_query": rewritten,
            "retry_count": retry_count + 1,
        }

    return rewrite_node
