"""Generate node for 'similar' analysis type.

Produces a ranked list of similar startups with explanations.
Extend this node later for type-specific generation logic
(e.g. structured JSON output, multi-pass LLM calls).
"""

from __future__ import annotations

from bizintel.graph.nodes._generate_base import run_generation


def make_generate_similar_node(llm_client):
    """Factory: returns a generate node for 'similar' queries."""

    def generate_similar_node(state) -> dict:
        return run_generation(llm_client, state, analysis_type="similar")

    return generate_similar_node
