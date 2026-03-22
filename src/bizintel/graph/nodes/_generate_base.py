"""Shared generation logic used by all type-specific generate nodes.

Each type node calls ``run_generation`` with its analysis_type.
This avoids duplicating the LLM call, disclaimer, and logging boilerplate.
"""

from __future__ import annotations

import logging

from bizintel.config.settings import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from bizintel.rag.prompt_templates import get_prompt

logger = logging.getLogger(__name__)

_LOW_CONFIDENCE_DISCLAIMER = (
    "⚠️ **Low confidence** — the retrieved context may not fully match "
    "your query. Results below should be reviewed critically.\n\n---\n\n"
)


def run_generation(llm_client, state, *, analysis_type: str) -> dict:
    """Build prompt, call LLM, and return a state-update dict.

    Parameters
    ----------
    llm_client
        OpenAI-compatible LLM client.
    state
        Current graph state (Pydantic model).
    analysis_type : str
        Which prompt template to use (similar, swot, competitor, ...).
    """
    user_query = state.user_query
    source_docs = getattr(state, "source_docs", [])
    confidence = getattr(state, "confidence", "high")

    documents_text = "\n\n---\n\n".join(
        doc["text"] for doc in source_docs
    )

    prompt = get_prompt(
        analysis_type=analysis_type,
        query=user_query,
        documents=documents_text,
    )

    logger.info(
        "Generating answer: type=%s, docs=%d, model=%s",
        analysis_type, len(source_docs), LLM_MODEL,
    )

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )

    answer = response.choices[0].message.content

    if confidence == "low":
        answer = _LOW_CONFIDENCE_DISCLAIMER + answer

    logger.info("Generated %d chars", len(answer))

    return {"answer": answer}
