"""
Prompt templates for different analysis types.

Each template receives:
  - {documents}  → the retrieved startup documents (Style C text)
  - {query}      → the user's original question

The "auto" template lets the LLM decide the best analysis format.
Specialized templates force a specific output structure.
"""

from __future__ import annotations


# ── System prompts ───────────────────────────────────────────────────────

_BASE_ROLE = (
    "You are BizIntel, an AI-powered startup intelligence analyst. "
    "You have access to a database of 134,000+ startups from Y Combinator "
    "and Crunchbase. Use ONLY the startup data provided below to answer. "
    "If the data is insufficient, say so clearly.\n\n"
)

TEMPLATES: dict[str, str] = {
    # ── Auto: LLM decides the best format ────────────────────────────
    "auto": (
        _BASE_ROLE
        + "Based on the user's question, determine the most appropriate "
        "analysis format (list of similar startups, SWOT analysis, "
        "competitor analysis, side-by-side comparison, or ecosystem map) "
        "and respond accordingly. Use clear headings and structured formatting.\n\n"
        "--- Retrieved Startups ---\n{documents}\n\n"
        "--- User Query ---\n{query}"
    ),

    # ── Similar startups ─────────────────────────────────────────────
    "similar": (
        _BASE_ROLE
        + "The user wants to find similar startups. For each startup below, "
        "explain WHY it is similar to what the user described. "
        "Rank by relevance. Use this format:\n"
        "1. **Name** (Country, Founded) — Similarity explanation\n\n"
        "--- Retrieved Startups ---\n{documents}\n\n"
        "--- User Query ---\n{query}"
    ),

    # ── SWOT analysis ────────────────────────────────────────────────
    "swot": (
        _BASE_ROLE
        + "Generate a detailed SWOT analysis based on the startup data below. "
        "Use this exact structure:\n\n"
        "## Strengths\n- ...\n\n"
        "## Weaknesses\n- ...\n\n"
        "## Opportunities\n- ...\n\n"
        "## Threats\n- ...\n\n"
        "Support each point with evidence from the startup data.\n\n"
        "--- Retrieved Startups ---\n{documents}\n\n"
        "--- User Query ---\n{query}"
    ),

    # ── Competitor analysis ──────────────────────────────────────────
    "competitor": (
        _BASE_ROLE
        + "Analyze the competitive landscape using the startups below. "
        "Structure your response as:\n\n"
        "## Direct Competitors\n(Same market, same customer)\n\n"
        "## Indirect Competitors\n(Adjacent market or different approach)\n\n"
        "## Key Differentiators\n(What sets each apart)\n\n"
        "## Market Positioning\n(How they position relative to each other)\n\n"
        "--- Retrieved Startups ---\n{documents}\n\n"
        "--- User Query ---\n{query}"
    ),

    # ── Side-by-side comparison ──────────────────────────────────────
    "comparison": (
        _BASE_ROLE
        + "Compare the startups below side by side. "
        "Create a structured comparison covering:\n\n"
        "| Aspect | Startup A | Startup B | ... |\n"
        "|--------|-----------|-----------|-----|\n"
        "| Industry | ... | ... | ... |\n"
        "| Country | ... | ... | ... |\n"
        "| Founded | ... | ... | ... |\n"
        "| Key Focus | ... | ... | ... |\n\n"
        "Then provide a narrative summary of key differences and similarities.\n\n"
        "--- Retrieved Startups ---\n{documents}\n\n"
        "--- User Query ---\n{query}"
    ),

    # ── Ecosystem explorer ───────────────────────────────────────────
    "ecosystem": (
        _BASE_ROLE
        + "Map the startup ecosystem based on the data below. "
        "Structure your response as:\n\n"
        "## Ecosystem Overview\n(What industry/domain this covers)\n\n"
        "## Key Players\n(Major startups and their roles)\n\n"
        "## Sub-segments\n(Break the ecosystem into categories)\n\n"
        "## Trends\n(Patterns you observe — geography, founding years, focus areas)\n\n"
        "## Gaps & Opportunities\n(What's missing in this ecosystem)\n\n"
        "--- Retrieved Startups ---\n{documents}\n\n"
        "--- User Query ---\n{query}"
    ),
}


def get_prompt(
    analysis_type: str,
    query: str,
    documents: str,
) -> str:
    """
    Build the final prompt by inserting query + documents into the template.

    Parameters
    ----------
    analysis_type : str
        One of: auto, similar, swot, competitor, comparison, ecosystem.
    query : str
        The user's question.
    documents : str
        Concatenated retrieved documents (newline-separated).

    Returns
    -------
    str
        The complete prompt ready to send to the LLM.
    """
    template = TEMPLATES.get(analysis_type)
    if template is None:
        raise ValueError(
            f"Unknown analysis type: '{analysis_type}'. "
            f"Supported: {list(TEMPLATES.keys())}"
        )

    return template.format(documents=documents, query=query)
