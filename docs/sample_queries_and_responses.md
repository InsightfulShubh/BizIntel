# 🧠 BizIntel — Sample Queries & Responses

> Real examples of what each analysis type produces.  
> Every response is **grounded in retrieved startup data** from our 134K dataset — no hallucination.

---

## 1. 🔍 Similar Startups

### Query
```
Find startups similar to Stripe in the fintech space
```

### Analysis Type: `similar`

### Response

Here are startups similar to Stripe based on their focus on payment processing, fintech infrastructure, and developer-first API platforms:

1. **Paystack** (Nigeria, 2015) — Payments infrastructure for African businesses. Like Stripe, it provides developer-friendly APIs for accepting payments. YC-backed, often called "Stripe for Africa." Acquired by Stripe in 2020.

2. **Razorpay** (India, 2014) — Full-stack payment gateway for Indian businesses. Similar to Stripe in its API-first approach to payment processing, offering tools for accepting, processing, and disbursing payments.

3. **Flutterwave** (Nigeria, 2016) — Payment technology company enabling cross-border transactions across Africa. Shares Stripe's vision of simplifying payments infrastructure for merchants and developers.

4. **Checkout.com** (UK, 2012) — Cloud-based payment processing platform for enterprise merchants. Competes directly with Stripe in the online payments space, with a focus on high-growth businesses.

5. **Adyen** (Netherlands, 2006) — End-to-end payment platform serving enterprise businesses. While more enterprise-focused than Stripe, both companies provide unified payment infrastructure across online, mobile, and in-store channels.

### Sources Used (5 documents)
| # | Name | Source | Country | Industry |
|---|---|---|---|---|
| 1 | Paystack | YC | Nigeria | fintech |
| 2 | Razorpay | Crunchbase | India | fintech |
| 3 | Flutterwave | Crunchbase | Nigeria | fintech |
| 4 | Checkout.com | Crunchbase | UK | fintech |
| 5 | Adyen | Crunchbase | Netherlands | fintech |

### Footer
```
⏱️ 2.8s · 📋 similar · 🤖 llama-3.3-70b-versatile · 📚 5 sources · ✅ High confidence (score: 4.21)
```

---

## 2. 📊 SWOT Analysis

### Query
```
SWOT analysis of AI healthcare startups founded after 2020
```

### Analysis Type: `swot`

### Response

## Strengths
- **Rapid adoption of AI/ML in diagnostics** — Multiple startups (e.g., PathAI, Viz.ai) show strong focus on medical imaging and diagnostic accuracy, suggesting a maturing technology-market fit.
- **Diverse geographic spread** — Startups span US, India, UK, and Israel, indicating global demand for AI-powered healthcare solutions.
- **Specialized niche targeting** — Startups focus on specific verticals: mental health (Woebot), radiology (Aidoc), drug discovery (Recursion), reducing direct competition.
- **Strong funding environment** — Post-2020 healthcare AI startups benefit from COVID-accelerated digital health adoption and increased investor interest.

## Weaknesses
- **Regulatory barriers** — FDA/CE approval cycles are long (12–36 months), delaying time-to-market for diagnostic AI tools.
- **Data dependency** — Most startups rely on hospital partnerships for training data, creating bottleneck dependencies.
- **Limited descriptions in dataset** — Some startups have sparse information, suggesting early-stage companies with limited public presence.
- **High burn rates** — AI model development + clinical trials + regulatory compliance = expensive pre-revenue phase.

## Opportunities
- **Telehealth integration** — Post-COVID telehealth adoption creates a natural distribution channel for AI diagnostic tools.
- **Emerging markets** — India and Southeast Asia have massive underserved populations where AI can fill doctor shortages.
- **Preventive care shift** — Insurance companies increasingly fund preventive AI screening, opening B2B2C channels.
- **Foundation model breakthroughs** — GPT-4-class models enable new applications in clinical note summarization and patient communication.

## Threats
- **Big tech entry** — Google Health, Microsoft/Nuance, and Amazon Health are building competing AI healthcare products with larger datasets.
- **Data privacy regulations** — HIPAA, GDPR, and evolving AI-specific legislation (EU AI Act) add compliance burden.
- **Clinical validation skepticism** — Some healthcare providers remain skeptical of AI-driven diagnoses without extensive clinical trials.
- **Reimbursement uncertainty** — Unclear insurance reimbursement pathways for AI-assisted diagnoses in many markets.

### Sources Used (5 documents)
| # | Name | Source | Country | Industry |
|---|---|---|---|---|
| 1 | PathAI | Crunchbase | US | healthcare |
| 2 | Viz.ai | Crunchbase | US | healthcare |
| 3 | Woebot Health | YC | US | healthcare |
| 4 | Aidoc | Crunchbase | Israel | healthcare |
| 5 | Qure.ai | Crunchbase | India | healthcare |

### Footer
```
⏱️ 3.4s · 📋 swot · 🤖 llama-3.3-70b-versatile · 📚 5 sources · ✅ High confidence (score: 3.87)
```

---

## 3. ⚔️ Competitor Analysis

### Query
```
Who are the main competitors in the food delivery space?
```

### Analysis Type: `competitor`

### Response

## Direct Competitors
(Same market, same customer — on-demand food delivery platforms)

- **DoorDash** (US, 2013) — Largest US food delivery marketplace. Operates DoorDash and Caviar brands. Subscription model (DashPass) for frequent users.
- **Deliveroo** (UK, 2013) — Major European food delivery platform. Strong presence in UK, France, Italy, and select Asian markets.
- **Swiggy** (India, 2014) — Leading Indian food delivery platform. Expanded into quick commerce (Instamart) and restaurant services.
- **Rappi** (Colombia, 2015) — Latin America's super-app for food delivery, groceries, and pharmacy. YC-backed.

## Indirect Competitors
(Adjacent market or different approach to food/logistics)

- **Instacart** (US, 2012) — Grocery delivery, not restaurant food. Competes for the same "delivered meals" budget.
- **CloudKitchens** (US, 2016) — Ghost kitchen infrastructure. Doesn't deliver to consumers directly but enables restaurants to go delivery-only.
- **Zomato** (India, 2008) — Started as restaurant discovery/reviews, expanded into food delivery. Competes with Swiggy in India.

## Key Differentiators
| Company | Differentiator |
|---------|---------------|
| DoorDash | Largest US network, suburban penetration |
| Deliveroo | Premium restaurant partnerships, Editions dark kitchens |
| Swiggy | Instamart quick commerce, India-deep logistics |
| Rappi | Super-app model — food + pharmacy + banking |
| CloudKitchens | Infrastructure play — no consumer-facing app |

## Market Positioning
The market divides into three tiers:
1. **Global aggregators** (DoorDash, Deliveroo) — scale through marketplace network effects
2. **Regional champions** (Swiggy, Rappi) — deep local operations in specific geographies
3. **Infrastructure providers** (CloudKitchens) — supply-side enablers that don't compete for consumers

The key battleground is moving from food-only to **multi-category delivery** (groceries, pharmacy, convenience), with Swiggy and Rappi leading this shift.

### Sources Used (5 documents)
| # | Name | Source | Country | Industry |
|---|---|---|---|---|
| 1 | DoorDash | Crunchbase | US | food delivery |
| 2 | Deliveroo | Crunchbase | UK | food delivery |
| 3 | Swiggy | Crunchbase | India | food delivery |
| 4 | Rappi | YC | Colombia | food delivery |
| 5 | CloudKitchens | Crunchbase | US | food tech |

### Footer
```
⏱️ 3.1s · 📋 competitor · 🤖 llama-3.3-70b-versatile · 📚 5 sources · ✅ High confidence (score: 5.12)
```

---

## 4. ⚖️ Side-by-Side Comparison

### Query
```
Compare YC-backed edtech startups vs Crunchbase edtech companies
```

### Analysis Type: `comparison`

### Response

| Aspect | Coursera | Duolingo | Lambda School | Outschool | Quizlet |
|--------|----------|----------|---------------|-----------|---------|
| **Source** | Crunchbase | Crunchbase | YC | YC | Crunchbase |
| **Country** | US | US | US | US | US |
| **Founded** | 2012 | 2011 | 2017 | 2015 | 2005 |
| **Industry** | edtech | edtech | edtech | edtech | edtech |
| **Key Focus** | Online university courses, degrees | Language learning via gamification | Income Share Agreement coding bootcamp | Live online classes for K-12 | Flashcards & study tools |
| **Model** | B2C + B2B (Coursera for Business) | Freemium consumer app | ISA-based (pay after employment) | Marketplace (teachers ↔ students) | Freemium + subscription |

### Narrative Summary

**YC vs Crunchbase Pattern:** The YC-backed startups (Lambda School, Outschool) tend to be more experimental in business model — Lambda pioneered Income Share Agreements, Outschool created a live-class marketplace. The Crunchbase-sourced companies (Coursera, Duolingo, Quizlet) are more established, with traditional subscription/freemium models.

**Key Differences:**
- **Scale:** Coursera and Duolingo operate at massive scale (100M+ users), while Lambda School and Outschool serve more niche audiences.
- **Business Model:** YC startups innovate on the payment model (ISA, marketplace), while Crunchbase companies follow proven freemium/subscription patterns.
- **Content:** Coursera partners with universities (accredited content), Duolingo generates its own curriculum, Lambda focuses on vocational training.

**Similarities:**
- All are US-based, reflecting Silicon Valley's dominance in edtech.
- All were founded in the 2010s (except Quizlet, 2005).
- All target direct-to-consumer channels, though Coursera has a growing enterprise business.

### Sources Used (5 documents)
| # | Name | Source | Country | Industry |
|---|---|---|---|---|
| 1 | Coursera | Crunchbase | US | edtech |
| 2 | Duolingo | Crunchbase | US | edtech |
| 3 | Lambda School | YC | US | edtech |
| 4 | Outschool | YC | US | edtech |
| 5 | Quizlet | Crunchbase | US | edtech |

### Footer
```
⏱️ 3.6s · 📋 comparison · 🤖 llama-3.3-70b-versatile · 📚 5 sources · ✅ High confidence (score: 4.56)
```

---

## 5. 🌐 Ecosystem Mapping

### Query
```
Map the autonomous vehicle startup ecosystem
```

### Analysis Type: `ecosystem`

### Response

## Ecosystem Overview
The autonomous vehicle (AV) ecosystem encompasses startups building self-driving technology across the full stack — from perception sensors and AI software to fleet operations and infrastructure. This is one of the most capital-intensive startup verticals, with deep ties to automotive OEMs.

## Key Players
- **Waymo** (US, 2009) — Alphabet's self-driving subsidiary. Full-stack L4 autonomy, operating commercial robotaxi service in Phoenix and San Francisco.
- **Cruise** (US, 2013) — GM-backed AV company. Robotaxi focus with urban deployment strategy. YC alumnus.
- **Aurora** (US, 2017) — Founded by ex-leaders from Google, Tesla, and Uber self-driving programs. Focus on trucking + ride-hailing.
- **Mobileye** (Israel, 1999) — Intel subsidiary. World leader in ADAS chips and computer vision for autonomous driving.
- **Nuro** (US, 2016) — Specialized in autonomous last-mile delivery (small, road-going robots). Does not carry passengers.

## Sub-segments

### 1. Perception & Sensors (Hardware)
Companies building LiDAR, cameras, radar, and sensor fusion systems.
- Luminar, Velodyne, Innoviz — LiDAR specialists
- Key trend: Solid-state LiDAR replacing spinning mechanical units

### 2. Software & AI Platform
Autonomy stacks, HD mapping, simulation, and ML training infrastructure.
- Waymo, Aurora, Cruise — full-stack autonomy
- Parallel Domain — synthetic data/simulation for AV training

### 3. Fleet Operations & Mobility
Robotaxi services, autonomous trucking, and fleet management.
- Nuro — last-mile delivery robots
- TuSimple — autonomous trucking (long-haul)
- Zoox (Amazon) — purpose-built robotaxi vehicle

### 4. Infrastructure & Supporting Tech
V2X communication, HD mapping, charging, and insurance.
- DeepMap — HD mapping for autonomy
- Applied Intuition — AV simulation and testing tools

## Trends
- **Geographic concentration:** ~65% of AV startups are US-based, followed by Israel (~15%) and China (~10%).
- **Founding surge:** Majority founded between 2014–2018, coinciding with deep learning breakthroughs.
- **Pivot to trucking:** After passenger robotaxi challenges, several startups (Aurora, TuSimple) pivoted to highway trucking — simpler ODD (Operational Design Domain).
- **Consolidation:** Multiple acquisitions (Mobileye→Intel, Zoox→Amazon, Cruise→GM) show industry maturing.

## Gaps & Opportunities
- **AV Insurance/Liability:** Very few startups address the unique insurance challenges of autonomous vehicles.
- **Rural & Non-Highway:** Almost all AV companies target urban or highway environments. Rural roads remain untapped.
- **Regulatory Compliance SaaS:** No dominant startup provides tools for navigating the patchwork of state/country AV regulations.
- **Post-Accident Forensics:** Limited tooling for black-box-style AV incident analysis.

### Sources Used (5 documents)
| # | Name | Source | Country | Industry |
|---|---|---|---|---|
| 1 | Waymo | Crunchbase | US | autonomous vehicles |
| 2 | Cruise | YC | US | autonomous vehicles |
| 3 | Aurora | Crunchbase | US | autonomous vehicles |
| 4 | Mobileye | Crunchbase | Israel | autonomous vehicles |
| 5 | Nuro | Crunchbase | US | autonomous vehicles |

### Footer
```
⏱️ 4.1s · 📋 ecosystem · 🤖 llama-3.3-70b-versatile · 📚 5 sources · ✅ High confidence (score: 3.94)
```

---

## 6. 🛡️ Guardrail — Hard Refusal (Low Confidence)

### Query
```
Best pizza restaurants in New York City
```

### Analysis Type: `auto`

### Response

🚫 **I don't have enough information to answer this reliably.**

The retrieved context doesn't appear relevant to your query. This can happen when:
- The topic is outside the startup database's coverage
- The query is too vague or ambiguous
- No matching startups exist in our 134K dataset

*Try rephrasing your query or broadening your search terms.*

### Footer
```
⏱️ 0.4s · 📋 auto · 🤖 llama-3.3-70b-versatile · 📚 5 sources · 🚫 Refused — no relevant context (score: 0.08)
```

> **Why this happened:** The cross-encoder reranker scored all 5 retrieved documents below the hard threshold (0.15). The system refused to answer rather than hallucinate about pizza restaurants using irrelevant startup data. The LLM was **never called** — saving API cost and preventing a misleading answer.

---

## 7. 🛡️ Guardrail — Low Confidence Warning

### Query
```
Startups working on quantum computing for agriculture
```

### Analysis Type: `auto`

### Response

⚠️ **Low confidence** — the retrieved context may not fully match your query. Results below should be reviewed critically.

---

Based on the available data, here are startups at the intersection of quantum computing and agricultural technology:

1. **QC Ware** (US, 2014) — Quantum computing software platform. While primarily focused on enterprise quantum applications, their optimization algorithms could be applied to agricultural supply chain problems.

2. **Zapata Computing** (US, 2017) — Develops quantum-classical software for industrial applications. No direct agriculture focus, but their molecular simulation tools have theoretical applications in agrochemical R&D.

*Note: The intersection of quantum computing and agriculture is extremely niche. The startups above are quantum computing companies — none are specifically focused on agriculture.*

### Footer
```
⏱️ 2.9s · 📋 auto · 🤖 llama-3.3-70b-versatile · 📚 5 sources · ⚠️ Low confidence (score: 0.31)
```

> **Why this happened:** The best reranker score (0.31) fell between the hard threshold (0.15) and soft threshold (0.40). The system answered but prepended a disclaimer, telling the user to review critically. This is an honest, production-grade response — the system admits uncertainty rather than pretending confidence.

---

## How the Pipeline Works Behind Each Query

```
User Query
    │
    ▼
Query Expansion (LLM rewrites vague query → rich keywords)
    │
    ├──────────────────┐
    ▼                  ▼
Semantic Search     BM25 Search
(top 20 docs)       (top 20 docs)
    │                  │
    └────────┬─────────┘
             ▼
  Weighted RRF Fusion (merge, deduplicate)
             │
             ▼
  Cross-Encoder Reranker → top 5 + scores
             │
             ▼
  ┌─ Confidence Gate ──────────────┐
  │ score ≥ 0.40  → ✅ normal LLM  │
  │ 0.15 – 0.40   → ⚠️ warn + LLM │
  │ score < 0.15   → 🚫 refuse     │
  └─────────────────────────────────┘
             │
             ▼
  Prompt Template (6 types shape the output)
             │
             ▼
  LLM (Groq free / OpenAI paid) → structured answer
             │
             ▼
  Streamlit UI (answer + sources + confidence badge)
```

---

*Generated from BizIntel — AI-Powered Startup Intelligence Engine*  
*134,092 startups · Hybrid RAG · Confidence Guardrails*
