"""Record Turn node — saves the current exchange to conversation history.

Appended atomically at the END of a graph run so that both user query
and assistant answer are recorded together (or neither on crash).

Because ``conversation_history`` uses ``Annotated[list, operator.add]``,
returning a list here **appends** to the accumulated history rather than
overwriting it.  Combined with a checkpointer, this persists across
invocations on the same ``thread_id``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def record_turn_node(state) -> dict:
    """Append the current user query + assistant answer to conversation history."""
    user_query = state.user_query
    answer = getattr(state, "answer", "")

    logger.info("Recording turn: query=%s… answer=%d chars", user_query[:40], len(answer))

    return {
        "conversation_history": [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": answer},
        ]
    }
