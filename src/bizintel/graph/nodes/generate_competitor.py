"""Generate node for 'competitor' analysis type.

Produces a competitive landscape or competitor mapping.
Extend this node later for type-specific generation logic
(e.g. structured competitor matrix, market positioning chart).
"""

from __future__ import annotations

from bizintel.graph.nodes._generate_base import run_generation


def make_generate_competitor_node(llm_client):
    """Factory: returns a generate node for 'competitor' queries."""

    def generate_competitor_node(state) -> dict:
        return run_generation(llm_client, state, analysis_type="competitor")

    return generate_competitor_node
