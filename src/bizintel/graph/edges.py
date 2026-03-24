"""Conditional edge functions for the BizIntel LangGraph pipeline.

Each function inspects the current state and returns the name of the
next node to route to.  These are used with ``graph.add_conditional_edges``.
"""

from __future__ import annotations

import logging

from bizintel.config.settings import MAX_RETRIES, ANALYSIS_TYPES, WEB_SEARCH_ENABLED

logger = logging.getLogger(__name__)

# Sentinel node names used by the builder
END = "__end__"

# Valid types the classifier can produce (excludes "auto")
_VALID_TYPES = {t for t in ANALYSIS_TYPES if t != "auto"}


def route_after_confidence(state) -> str:
    """Route after the Confidence Gate node.

    - high / low → type-specific generate node (by analysis_type)
    - none + retries left → rewrite (try different query)
    - none + retries exhausted + web search enabled → web_search (fallback)
    - none + retries exhausted + web search disabled → end (refuse)
    """
    confidence = getattr(state, "confidence", "high")
    retry_count = getattr(state, "retry_count", 0)

    if confidence in ("high", "low"):
        analysis_type = getattr(state, "analysis_type", "similar")
        target = f"generate_{analysis_type}"
        logger.info("Confidence=%s → %s", confidence, target)
        return target

    # confidence == "none"
    if retry_count < MAX_RETRIES:
        logger.info("Confidence=none, retry %d/%d → rewrite", retry_count, MAX_RETRIES)
        return "rewrite"

    # Retries exhausted — try web search if enabled, otherwise refuse
    if WEB_SEARCH_ENABLED:
        logger.info("Confidence=none, retries exhausted → web_search (fallback)")
        return "web_search"

    logger.info("Confidence=none, retries exhausted → end (refuse)")
    return END


def route_after_validate(state) -> str:
    """Route after the Validate node.

    - pass → end (return answer)
    - fail + confidence was "low" → end (best-effort, no retry)
    - fail + retries left → rewrite (try again)
    - fail + retries exhausted → end (best-effort)
    """
    validation_check = getattr(state, "validation_check", False)
    confidence = getattr(state, "confidence", "high")
    retry_count = getattr(state, "retry_count", 0)

    if validation_check:
        logger.info("Validation passed → end")
        return END

    # Validation failed
    if confidence == "low":
        logger.info("Validation failed + low confidence → end (best-effort)")
        return END

    if retry_count < MAX_RETRIES:
        logger.info("Validation failed, retry %d/%d → rewrite", retry_count, MAX_RETRIES)
        return "rewrite"

    logger.info("Validation failed, retries exhausted → end (best-effort)")
    return END


def route_after_web_search(state) -> str:
    """Route after the Web Search node → type-specific generate node.

    Same logic as the high-confidence path: read analysis_type and go
    to the corresponding generate node.
    """
    analysis_type = getattr(state, "analysis_type", "similar")
    target = f"generate_{analysis_type}"
    logger.info("Web search done → %s", target)
    return target
