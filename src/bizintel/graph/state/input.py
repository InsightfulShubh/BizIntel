"""InputState — what the caller provides when invoking the graph."""

from __future__ import annotations

from pydantic import BaseModel, Field


class InputState(BaseModel):
    """Public input schema — accepted by graph.invoke().

    Only the fields the caller (Streamlit UI) needs to provide.
    """

    user_query: str = Field(description="Original user question — never mutated")
