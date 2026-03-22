"""Generate node for 'comparison' analysis type.

Produces a side-by-side comparison of specific startups or segments.
Extend this node later for type-specific generation logic
(e.g. structured table output, per-entity sub-answers merged).
"""

from __future__ import annotations

from bizintel.graph.nodes._generate_base import run_generation


def make_generate_comparison_node(llm_client):
    """Factory: returns a generate node for 'comparison' queries."""

    def generate_comparison_node(state) -> dict:
        return run_generation(llm_client, state, analysis_type="comparison")

    return generate_comparison_node
