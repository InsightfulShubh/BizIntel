"""Web Search node — fallback when vector DB confidence is too low.

Uses the Tavily API to search the open web for startup information,
then converts results into the same ``source_docs`` format the
generate nodes expect.  Sets ``web_searched = True`` so downstream
nodes know the context came from the web, not the local DB.
"""

from __future__ import annotations

import logging

from tavily import TavilyClient

from bizintel.config.settings import WEB_SEARCH_MAX_RESULTS

logger = logging.getLogger(__name__)


def make_web_search_node(tavily_client: TavilyClient):
    """Factory: returns a web-search node that closes over the Tavily client."""

    def web_search_node(state) -> dict:
        """Search the web for the user's query and return results as source_docs."""
        query = state.expanded_query or state.user_query

        logger.info("Web search: query=%s…", query[:60])

        response = tavily_client.search(
            query=query,
            max_results=WEB_SEARCH_MAX_RESULTS,
            search_depth="basic",
            topic="general",
        )

        source_docs = [
            {
                "doc_id": f"web_{i}",
                "text": result.get("content", ""),
                "metadata": {
                    "source": "web",
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                },
                "distance": result.get("score", 0.0),
            }
            for i, result in enumerate(response.get("results", []))
        ]

        logger.info("Web search returned %d results", len(source_docs))

        return {
            "source_docs": source_docs,
            "web_searched": True,
            "confidence": "low",  # web results → lower trust than vector DB
        }

    return web_search_node
