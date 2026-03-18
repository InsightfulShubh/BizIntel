"""
Evaluation dataset — 30 test queries across all 6 analysis types.

Each query includes:
  - query:              The user's natural language question
  - analysis_type:      Which analysis template to use
  - expected_domains:   Keywords that SHOULD appear in retrieved docs (for context relevancy)
  - bad_results:        Company names that should NOT appear (known retrieval failures)
  - expected_sections:  For structured types (SWOT, competitor), required output headings
  - description:        Human-readable note about what this test checks
"""

from __future__ import annotations

EVAL_DATASET: list[dict] = [
    # ──────────────────────────────────────────────────────────────
    # SIMILAR (6 queries)
    # ──────────────────────────────────────────────────────────────
    {
        "id": "sim_01",
        "query": "Find startups similar to Stripe",
        "analysis_type": "similar",
        "expected_domains": ["fintech", "payment", "billing", "financial"],
        "bad_results": ["StartupBus", "Startup Genome", "Startupbootcamp"],
        "expected_sections": [],
        "description": "Classic proper-noun test — must resolve Stripe → fintech via query expansion",
    },
    {
        "id": "sim_02",
        "query": "Find startups similar to Airbnb",
        "analysis_type": "similar",
        "expected_domains": ["rental", "hospitality", "travel", "accommodation", "booking"],
        "bad_results": [],
        "expected_sections": [],
        "description": "Proper-noun test — Airbnb → short-term rental marketplace",
    },
    {
        "id": "sim_03",
        "query": "AI healthcare startups in India",
        "analysis_type": "similar",
        "expected_domains": ["healthcare", "medical", "health", "AI", "artificial intelligence"],
        "bad_results": [],
        "expected_sections": [],
        "description": "Descriptive query — should match without heavy expansion",
    },
    {
        "id": "sim_04",
        "query": "Find edtech companies that teach coding to kids",
        "analysis_type": "similar",
        "expected_domains": ["education", "edtech", "coding", "learning", "kids", "children"],
        "bad_results": [],
        "expected_sections": [],
        "description": "Multi-concept query — edtech + coding + children",
    },
    {
        "id": "sim_05",
        "query": "Startups building developer tools and APIs",
        "analysis_type": "similar",
        "expected_domains": ["developer", "API", "tools", "software", "platform"],
        "bad_results": [],
        "expected_sections": [],
        "description": "Technical domain query — developer tooling",
    },
    {
        "id": "sim_06",
        "query": "Climate tech and clean energy startups",
        "analysis_type": "similar",
        "expected_domains": ["climate", "clean", "energy", "sustainability", "green", "solar", "carbon"],
        "bad_results": [],
        "expected_sections": [],
        "description": "Emerging sector — climate / cleantech",
    },

    # ──────────────────────────────────────────────────────────────
    # SWOT (5 queries)
    # ──────────────────────────────────────────────────────────────
    {
        "id": "swot_01",
        "query": "SWOT analysis of AI healthcare startups founded after 2020",
        "analysis_type": "swot",
        "expected_domains": ["healthcare", "AI", "medical"],
        "bad_results": [],
        "expected_sections": ["Strengths", "Weaknesses", "Opportunities", "Threats"],
        "description": "Must produce all 4 SWOT quadrants with healthcare context",
    },
    {
        "id": "swot_02",
        "query": "SWOT analysis of fintech startups in the US",
        "analysis_type": "swot",
        "expected_domains": ["fintech", "financial", "payment", "banking"],
        "bad_results": [],
        "expected_sections": ["Strengths", "Weaknesses", "Opportunities", "Threats"],
        "description": "SWOT for fintech — should reference US market specifics",
    },
    {
        "id": "swot_03",
        "query": "SWOT of food delivery startups",
        "analysis_type": "swot",
        "expected_domains": ["food", "delivery", "restaurant", "logistics"],
        "bad_results": [],
        "expected_sections": ["Strengths", "Weaknesses", "Opportunities", "Threats"],
        "description": "SWOT for food delivery sector",
    },
    {
        "id": "swot_04",
        "query": "SWOT analysis for cybersecurity companies",
        "analysis_type": "swot",
        "expected_domains": ["security", "cyber", "protection", "threat", "network"],
        "bad_results": [],
        "expected_sections": ["Strengths", "Weaknesses", "Opportunities", "Threats"],
        "description": "SWOT for cybersecurity domain",
    },
    {
        "id": "swot_05",
        "query": "SWOT of e-commerce startups in India",
        "analysis_type": "swot",
        "expected_domains": ["ecommerce", "e-commerce", "online", "shopping", "retail"],
        "bad_results": [],
        "expected_sections": ["Strengths", "Weaknesses", "Opportunities", "Threats"],
        "description": "SWOT for Indian e-commerce",
    },

    # ──────────────────────────────────────────────────────────────
    # COMPETITOR (5 queries)
    # ──────────────────────────────────────────────────────────────
    {
        "id": "comp_01",
        "query": "Who are the main competitors in the food delivery space?",
        "analysis_type": "competitor",
        "expected_domains": ["food", "delivery", "restaurant", "meal"],
        "bad_results": [],
        "expected_sections": ["Direct Competitors", "Indirect Competitors"],
        "description": "Competitor landscape for food delivery",
    },
    {
        "id": "comp_02",
        "query": "Competitive landscape for cloud infrastructure startups",
        "analysis_type": "competitor",
        "expected_domains": ["cloud", "infrastructure", "hosting", "server", "computing"],
        "bad_results": [],
        "expected_sections": ["Direct Competitors", "Indirect Competitors"],
        "description": "Competitor analysis for cloud infra",
    },
    {
        "id": "comp_03",
        "query": "Competitors of ride-sharing platforms",
        "analysis_type": "competitor",
        "expected_domains": ["ride", "transportation", "mobility", "taxi", "sharing"],
        "bad_results": [],
        "expected_sections": ["Direct Competitors", "Indirect Competitors"],
        "description": "Ride-sharing competitor map",
    },
    {
        "id": "comp_04",
        "query": "Who competes in the online education market?",
        "analysis_type": "competitor",
        "expected_domains": ["education", "online", "learning", "course", "edtech"],
        "bad_results": [],
        "expected_sections": ["Direct Competitors", "Indirect Competitors"],
        "description": "Online education competitors",
    },
    {
        "id": "comp_05",
        "query": "Competitive analysis of project management tools",
        "analysis_type": "competitor",
        "expected_domains": ["project", "management", "productivity", "collaboration", "task"],
        "bad_results": [],
        "expected_sections": ["Direct Competitors", "Indirect Competitors"],
        "description": "Project management tool competitors",
    },

    # ──────────────────────────────────────────────────────────────
    # COMPARISON (5 queries)
    # ──────────────────────────────────────────────────────────────
    {
        "id": "cmp_01",
        "query": "Compare YC-backed edtech startups vs Crunchbase edtech companies",
        "analysis_type": "comparison",
        "expected_domains": ["education", "edtech", "learning"],
        "bad_results": [],
        "expected_sections": [],
        "description": "Cross-source comparison for edtech",
    },
    {
        "id": "cmp_02",
        "query": "Compare fintech startups in India vs the US",
        "analysis_type": "comparison",
        "expected_domains": ["fintech", "financial", "payment"],
        "bad_results": [],
        "expected_sections": [],
        "description": "Geography comparison — India vs US fintech",
    },
    {
        "id": "cmp_03",
        "query": "Compare AI startups founded before 2015 vs after 2020",
        "analysis_type": "comparison",
        "expected_domains": ["AI", "artificial intelligence", "machine learning"],
        "bad_results": [],
        "expected_sections": [],
        "description": "Time-period comparison for AI",
    },
    {
        "id": "cmp_04",
        "query": "Compare healthcare vs biotech startups",
        "analysis_type": "comparison",
        "expected_domains": ["healthcare", "health", "biotech", "medical", "bio"],
        "bad_results": [],
        "expected_sections": [],
        "description": "Sector comparison — healthcare vs biotech",
    },
    {
        "id": "cmp_05",
        "query": "Side-by-side comparison of SaaS companies in B2B vs B2C",
        "analysis_type": "comparison",
        "expected_domains": ["SaaS", "software", "B2B", "B2C", "subscription"],
        "bad_results": [],
        "expected_sections": [],
        "description": "Business model comparison — B2B vs B2C SaaS",
    },

    # ──────────────────────────────────────────────────────────────
    # ECOSYSTEM (5 queries)
    # ──────────────────────────────────────────────────────────────
    {
        "id": "eco_01",
        "query": "Map the autonomous vehicle startup ecosystem",
        "analysis_type": "ecosystem",
        "expected_domains": ["autonomous", "vehicle", "self-driving", "automotive", "mobility"],
        "bad_results": [],
        "expected_sections": ["Ecosystem Overview", "Key Players"],
        "description": "AV ecosystem mapping",
    },
    {
        "id": "eco_02",
        "query": "Explore the blockchain and Web3 startup ecosystem",
        "analysis_type": "ecosystem",
        "expected_domains": ["blockchain", "web3", "crypto", "decentralized", "DeFi"],
        "bad_results": [],
        "expected_sections": ["Ecosystem Overview", "Key Players"],
        "description": "Blockchain/Web3 ecosystem",
    },
    {
        "id": "eco_03",
        "query": "Map the proptech real estate technology ecosystem",
        "analysis_type": "ecosystem",
        "expected_domains": ["real estate", "property", "proptech", "housing"],
        "bad_results": [],
        "expected_sections": ["Ecosystem Overview", "Key Players"],
        "description": "PropTech ecosystem mapping",
    },
    {
        "id": "eco_04",
        "query": "Startup ecosystem for mental health and wellness",
        "analysis_type": "ecosystem",
        "expected_domains": ["mental health", "wellness", "therapy", "meditation", "health"],
        "bad_results": [],
        "expected_sections": ["Ecosystem Overview", "Key Players"],
        "description": "Mental health/wellness ecosystem",
    },
    {
        "id": "eco_05",
        "query": "Map the agriculture technology (agtech) startup ecosystem",
        "analysis_type": "ecosystem",
        "expected_domains": ["agriculture", "farming", "agtech", "crop", "food"],
        "bad_results": [],
        "expected_sections": ["Ecosystem Overview", "Key Players"],
        "description": "AgTech ecosystem mapping",
    },

    # ──────────────────────────────────────────────────────────────
    # AUTO (4 queries)
    # ──────────────────────────────────────────────────────────────
    {
        "id": "auto_01",
        "query": "Tell me about robotics startups",
        "analysis_type": "auto",
        "expected_domains": ["robotics", "robot", "automation", "hardware"],
        "bad_results": [],
        "expected_sections": [],
        "description": "Vague query — auto should pick a reasonable format",
    },
    {
        "id": "auto_02",
        "query": "What's happening in the space tech industry?",
        "analysis_type": "auto",
        "expected_domains": ["space", "satellite", "aerospace", "rocket", "orbit"],
        "bad_results": [],
        "expected_sections": [],
        "description": "Open-ended question — should auto-detect ecosystem or overview",
    },
    {
        "id": "auto_03",
        "query": "Analyse the gaming startup landscape in the US",
        "analysis_type": "auto",
        "expected_domains": ["gaming", "game", "esports", "entertainment"],
        "bad_results": [],
        "expected_sections": [],
        "description": "Analysis request — should auto-detect competitor or ecosystem",
    },
    {
        "id": "auto_04",
        "query": "Find me something interesting in quantum computing",
        "analysis_type": "auto",
        "expected_domains": ["quantum", "computing", "qubit"],
        "bad_results": [],
        "expected_sections": [],
        "description": "Very vague — tests query expansion + auto format selection",
    },
]
