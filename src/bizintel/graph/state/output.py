"""OutputState — what the graph returns to the caller."""

from __future__ import annotations

from pydantic import BaseModel, Field


class OutputState(BaseModel):
    """Public output schema — returned by graph.invoke().

    Only the fields the caller (Streamlit UI) needs to consume.
    """

    answer: str = Field(default="", description="LLM-generated analysis or refusal message")
    analysis_type: str = Field(default="", description="Detected or assigned analysis type")
    source_docs: list[dict] = Field(
        default_factory=list,
        description="Retrieved documents serialised as dicts for the UI",
    )
    confidence: str = Field(default="high", description="high | low | none")
    best_score: float = Field(default=0.0, description="Top reranker score")
