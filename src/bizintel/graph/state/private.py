"""PrivateState — full internal working state, visible only to graph nodes."""

from __future__ import annotations

import operator
from typing import Annotated

from pydantic import Field

from bizintel.graph.state.input import InputState
from bizintel.graph.state.output import OutputState


class PrivateState(InputState, OutputState):
    """Internal state carried between nodes.

    Inherits all InputState and OutputState fields, plus internal fields
    that the caller never sees.  LangGraph uses the inheritance to know
    which fields are input-only, output-only, or internal.
    """

    # ── Internal fields (not exposed to caller) ──────────────────────

    expanded_query: str = Field(
        default="",
        description="Working query — set by Expand, overwritten by Rewrite on retry",
    )
    reranker_scores: list[float] = Field(
        default_factory=list,
        description="Cross-encoder scores from Retrieve (parallel to source_docs)",
    )
    mean_score: float = Field(
        default=0.0,
        description="Mean reranker score across kept docs",
    )
    validation_check: bool = Field(
        default=False,
        description="Post-generation validation pass/fail",
    )
    retry_count: int = Field(
        default=0,
        description="Number of rewrite-retrieve-generate retries so far",
    )
    web_searched: bool = Field(
        default=False,
        description="True if web search was used instead of vector DB for this query",
    )
    conversation_history: Annotated[list[dict], operator.add] = Field(
        default_factory=list,
        description="Accumulated chat turns [{role, content}, ...] — grows via reducer across invocations",
    )
