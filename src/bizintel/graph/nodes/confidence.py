"""Confidence Gate node — classifies retrieval confidence from reranker scores."""

from __future__ import annotations

import logging

from bizintel.config.settings import (
    GUARDRAILS_ENABLED,
    CONFIDENCE_THRESHOLD_SOFT,
    CONFIDENCE_THRESHOLD_HARD,
)

logger = logging.getLogger(__name__)


def confidence_gate_node(state) -> dict:
    """Classify confidence based on best_score thresholds.

    Pure logic node — no LLM call, no external dependencies.

    Returns
    -------
    dict
        {"confidence": "high" | "low" | "none"}
    """
    best_score = getattr(state, "best_score", 0.0)
    source_docs = getattr(state, "source_docs", [])

    if not source_docs:
        confidence = "none"
    elif not GUARDRAILS_ENABLED:
        confidence = "high"
    elif best_score < CONFIDENCE_THRESHOLD_HARD:
        confidence = "none"
    elif best_score < CONFIDENCE_THRESHOLD_SOFT:
        confidence = "low"
    else:
        confidence = "high"

    logger.info(
        "Confidence gate: best_score=%.3f → %s (soft=%.2f, hard=%.2f)",
        best_score, confidence,
        CONFIDENCE_THRESHOLD_SOFT, CONFIDENCE_THRESHOLD_HARD,
    )

    return {"confidence": confidence}
