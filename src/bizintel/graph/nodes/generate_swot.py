"""Generate node for 'swot' analysis type.

Produces a structured SWOT (Strengths, Weaknesses, Opportunities, Threats).
Extend this node later for type-specific generation logic
(e.g. 2-pass LLM: first extract facts, then synthesise SWOT).
"""

from __future__ import annotations

from bizintel.graph.nodes._generate_base import run_generation


def make_generate_swot_node(llm_client):
    """Factory: returns a generate node for 'swot' queries."""

    def generate_swot_node(state) -> dict:
        return run_generation(llm_client, state, analysis_type="swot")

    return generate_swot_node
