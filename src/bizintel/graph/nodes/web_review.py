"""Web Review node — Human-in-the-Loop checkpoint for web search results.

Pauses the graph via ``interrupt()`` so the user can review, filter, or
approve the web search results before they are passed to a generate node.

When the user resumes the graph (via ``Command(resume=...)``), the
``interrupt()`` call returns whatever value the user sent back — a list
of indices of approved results.  The node then filters ``source_docs``
to keep only the approved items.

Requires a **checkpointer** on the compiled graph (already have SqliteSaver).
"""

from __future__ import annotations

import logging

from langgraph.types import interrupt

logger = logging.getLogger(__name__)


def web_review_node(state) -> dict:
    """Present web results to user for approval, then filter to approved subset."""
    source_docs = getattr(state, "source_docs", [])

    if not source_docs:
        logger.info("Web review: no source docs to review — skipping interrupt")
        return {}

    # ── Pause graph: send web results to the UI for human review ─────
    #
    # The value passed to interrupt() is sent to the client.
    # The value returned by interrupt() is whatever the client sends
    # back via Command(resume=...).
    #
    # We expect the user to send back a list of indices (e.g. [0, 2, 4])
    # indicating which results to keep.

    approved_indices = interrupt(
        {
            "type": "web_review",
            "message": "I found these web results. Select which ones to use:",
            "results": [
                {
                    "index": i,
                    "title": doc.get("metadata", {}).get("title", f"Result {i+1}"),
                    "url": doc.get("metadata", {}).get("url", ""),
                    "snippet": doc.get("text", "")[:200],
                }
                for i, doc in enumerate(source_docs)
            ],
        }
    )

    # ── Graph resumed: filter source_docs to approved subset ─────────
    logger.info("Web review: user approved indices=%s", approved_indices)

    if approved_indices is None or approved_indices == "all":
        # User approved everything (or sent "all")
        logger.info("Web review: keeping all %d results", len(source_docs))
        return {}

    # Filter to only approved indices
    approved_docs = [
        doc for i, doc in enumerate(source_docs) if i in approved_indices
    ]

    logger.info(
        "Web review: kept %d/%d results",
        len(approved_docs), len(source_docs),
    )

    return {"source_docs": approved_docs}
