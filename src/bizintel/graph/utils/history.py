"""Shared helper to format conversation history for injection into LLM prompts."""

from __future__ import annotations

from bizintel.config.settings import MEMORY_WINDOW


def format_history_context(conversation_history: list[dict]) -> str:
    """Build a concise history string from the last N turns.

    Returns an empty string when there is no history, so callers can
    simply prepend the result without extra branching.
    """
    recent = conversation_history[-MEMORY_WINDOW:]
    if not recent:
        return ""

    lines = ["Recent conversation:"]
    for turn in recent:
        role = turn.get("role", "?")
        # Truncate long answers to save tokens
        content = turn.get("content", "")
        if role == "assistant":
            content = content[:300] + "…" if len(content) > 300 else content
        lines.append(f"  {role}: {content}")

    return "\n".join(lines) + "\n\n"
