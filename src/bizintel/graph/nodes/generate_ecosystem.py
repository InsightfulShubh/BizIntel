"""Generate node for 'ecosystem' analysis type.

Produces an industry ecosystem map, sub-segment analysis, or trend synthesis.
Extend this node later for type-specific generation logic
(e.g. multi-layer analysis, sub-segment clustering).
"""

from __future__ import annotations

from bizintel.graph.nodes._generate_base import run_generation


def make_generate_ecosystem_node(llm_client):
    """Factory: returns a generate node for 'ecosystem' queries."""

    def generate_ecosystem_node(state) -> dict:
        return run_generation(llm_client, state, analysis_type="ecosystem")

    return generate_ecosystem_node
