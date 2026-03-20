<div align="center">

# 🧠 BizIntel

### AI-Powered Startup Intelligence Engine

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-F55036?logo=groq&logoColor=white)](https://console.groq.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai&logoColor=white)](https://openai.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-green)](https://www.trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Ask natural-language questions about **134,000+ startups** from Y Combinator & Crunchbase.  
Powered by **Hybrid RAG** — semantic search + BM25 keyword search + cross-encoder reranking + LLM reasoning.  
**Free to run** — uses Groq's free API by default (switchable to OpenAI).

[Features](#-features) · [Demo](#-demo) · [Quick Start](#-quick-start) · [Architecture](#-architecture) · [Project Structure](#-project-structure) · [Usage](#-usage) · [Tech Stack](#-tech-stack)

</div>

---

## 📸 Demo

<div align="center">

![BizIntel UI](docs/screenshots/bizintel_ui.png)

*Streamlit chat interface with sidebar controls, analysis type selector, and 134K indexed startups.*

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Similar Startups** | Find companies similar to any startup (e.g., "Find startups similar to Stripe") |
| 📊 **SWOT Analysis** | AI-generated Strengths, Weaknesses, Opportunities, and Threats |
| ⚔️ **Competitor Analysis** | Map the competitive landscape — direct, indirect, differentiators |
| ⚖️ **Side-by-Side Comparison** | Compare startups head-to-head in a structured table |
| 🌐 **Ecosystem Mapping** | Explore an entire industry ecosystem — key players, sub-segments, trends |
| 🤖 **Auto Detect** | Let the AI choose the best analysis format for your query |
| 📚 **Source Transparency** | Every answer shows the exact source documents used — no hallucination |
| 🔄 **Query Expansion** | LLM-powered query rewriting for better semantic matching |
| 🗂️ **Dual Vector Store** | Strategy Pattern — swap between ChromaDB and FAISS via config |
| 🆕 **Hybrid Search** | Combines semantic (embedding) + keyword (BM25) search for better recall |
| 🆕 **Weighted RRF Fusion** | Merges ranked lists with tunable weights (semantic=1.0, BM25=0.4) |
| 🆕 **Cross-Encoder Reranker** | Rescores candidates with `ms-marco-MiniLM-L-6-v2` for +0.13 relevancy gain |
| 🆕 **Multi-Provider LLM** | Switch between Groq (free) and OpenAI (paid) via one config flag |
| 🆕 **RAG Evaluation Pipeline** | 30 test queries, 6 metrics (3 LLM-as-Judge + 3 deterministic) |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** (fast Python package manager)
- **Groq API Key (free)** ([Get one here](https://console.groq.com/keys)) — *or* OpenAI API Key ([paid](https://platform.openai.com/api-keys))

### 1. Clone & Install

```bash
git clone https://github.com/InsightfulShubh/BizIntel.git
cd BizIntel
uv sync
```

### 2. Set Up Environment

```bash
cp .env.example .env
```

**Option A — Groq (free, default):**
```env
GROQ_API_KEY=gsk_your-groq-key-here
# LLM_PROVIDER=groq  ← already the default in settings.py
```

**Option B — OpenAI (paid):**
```env
OPENAI_API_KEY=sk-your-key-here
LLM_PROVIDER=openai   # override the default
```

### 3. Run the Data Pipeline

```bash
# Step 1: Clean & unify the raw CSV data (YC + Crunchbase → 134K unified rows)
uv run python -m bizintel.preprocessing.main

# Step 2: Embed all startups & store in vector DB (~20 min on CPU)
uv run python -m bizintel.pipeline.batch_embed --reset
```

### 4. Launch the App

```bash
uv run streamlit run src/bizintel/app/streamlit_app.py --server.port 8501
```

Open **http://localhost:8501** in your browser. 🎉

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        OFFLINE PIPELINE (batch)                      │
│                                                                      │
│  CSV Files ──→ Data Cleaning ──→ Document Builder ──→ Embedder       │
│  (YC + CB)     (pandas)          (Pydantic)          (MiniLM-L6)    │
│                     │                                     │          │
│                     ▼                                     ▼          │
│              Unified CSV                          Vector Store       │
│              (134K rows)                     (ChromaDB / FAISS)      │
│                                                     │                │
│                                                     ▼                │
│                                              BM25 Index (in-memory)  │
│                                              (rank-bm25 / BM25Okapi) │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                       ONLINE PIPELINE (per query)                    │
│                                                                      │
│  User Query ──→ Query Expansion ──→ Encode (MiniLM)                  │
│                   (LLM rewrite)          │                           │
│                                          ├──────────────────┐        │
│                                          ▼                  ▼        │
│                                   Semantic Search     BM25 Search    │
│                                    (top 20 docs)      (top 20 docs)  │
│                                          │                  │        │
│                                          └────────┬─────────┘        │
│                                                   ▼                  │
│                                       Weighted RRF Fusion            │
│                                     (sem=1.0, bm25=0.4, k=60)       │
│                                                   │                  │
│                                                   ▼                  │
│                                       Cross-Encoder Reranker         │
│                                      (ms-marco-MiniLM, top 5)       │
│                                                   │                  │
│                                                   ▼                  │
│                                          Prompt Template             │
│                                          (6 analysis types)          │
│                                                   │                  │
│                                                   ▼                  │
│                                        LLM (Groq / OpenAI)          │
│                                                   │                  │
│                                                   ▼                  │
│                                          Streamlit Chat UI           │
│                                          (answer + sources)          │
└──────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| **Embedding Model** | `all-MiniLM-L6-v2` (384-dim) | Free, local, fast — no API costs for 134K docs |
| **LLM** | Groq `llama-3.3-70b` (default) / OpenAI GPT-4o-mini | One-flag switch; Groq is free, OpenAI is more grounded |
| **Vector Store** | ChromaDB (default) + FAISS | Strategy Pattern — swap via config, no code changes |
| **Hybrid Search** | Semantic + BM25 keyword search | Catches exact-match terms that embeddings miss |
| **Fusion** | Weighted RRF (sem=1.0, bm25=0.4) | Equal weights regressed relevancy; tuned weights fix it |
| **Reranker** | `ms-marco-MiniLM-L-6-v2` (22 MB) | +0.13 context relevancy gain; only 22 MB, 150ms/query |
| **Document Format** | Style C (labeled key-value) | Labels act as semantic anchors for the embedding model |
| **Query Expansion** | LLM-based rewriting | Solves the "Stripe → fintech" semantic gap problem |
| **No Chunking** | 1 startup = 1 document | Documents are short (~200 tokens), fit within model limit |
| **No LangChain** | Custom chain (70 lines) | Full control, minimal dependencies, debuggable |

---

## 📁 Project Structure

```
BizIntel/
├── src/bizintel/                  # Main package (src-layout)
│   ├── config/
│   │   ├── settings.py            # Centralized config — paths, thresholds, model names
│   │   └── llm_client.py          # LLM client factory — Groq / OpenAI via one flag
│   ├── preprocessing/
│   │   ├── data_preprocess.py     # Load, clean, unify YC + Crunchbase CSVs
│   │   ├── validation.py          # Flag suspicious records (is_suspicious)
│   │   └── main.py                # Pipeline entry point
│   ├── embeddings/
│   │   ├── document_builder.py    # DataFrame → StartupDocument (vectorized)
│   │   └── embedder.py            # SentenceTransformer wrapper with batching
│   ├── vectorstore/
│   │   ├── base.py                # ABC base class + SearchResult + factory
│   │   ├── chroma_store.py        # ChromaDB backend (cosine, HNSW)
│   │   └── faiss_store.py         # FAISS backend (IndexFlatIP + JSON sidecar)
│   ├── search/                    # Keyword search & fusion (NEW)
│   │   ├── __init__.py
│   │   ├── bm25_search.py         # BM25Okapi index over 134K docs
│   │   └── fusion.py              # Weighted Reciprocal Rank Fusion
│   ├── rag/
│   │   ├── retriever.py           # 4-stage pipeline: semantic → BM25 → RRF → reranker
│   │   ├── reranker.py            # Cross-encoder reranker (ms-marco-MiniLM)
│   │   ├── prompt_templates.py    # 6 analysis templates + shared base role
│   │   └── chain.py               # RAG orchestrator + query expansion
│   ├── pipeline/                  # Batch data operations
│   │   └── batch_embed.py         # One-time batch embedding script (CLI)
│   ├── evaluation/                # RAG evaluation pipeline
│   │   ├── eval_dataset.py        # 30 test queries with expected domains
│   │   ├── evaluator.py           # LLM-as-Judge + deterministic scorers
│   │   └── run_eval.py            # CLI evaluation runner → JSON + CSV
│   └── app/
│       ├── state.py               # @st.cache_resource loaders + session state
│       ├── components.py          # Sidebar, chat, source cards, CSS
│       └── streamlit_app.py       # Streamlit entry point
├── notebooks/
│   ├── data_analysis.ipynb        # EDA — 9 visualizations + JSON/Excel export
│   └── eval_visualization.ipynb   # Evaluation results visualization
├── data-source/                   # Raw CSVs (not in git)
├── data/                          # Cleaned CSVs + vector DB (not in git)
├── docs/
│   ├── architecture_flowchart.html     # v1 interactive architecture diagram
│   ├── architecture_flowchart_v2.html  # v2 with hybrid search, reranker, Groq
│   ├── interview_prep.html             # 50+ Q&A for interview preparation
│   └── design_decisions_v2.html        # 65+ Q&A — reranker, hybrid, RRF, Groq
├── tests/
│   └── spot_check.py              # Verify real companies in results
├── eval_results/                  # Timestamped eval outputs (JSON + CSV)
├── .env.example                   # Template for API keys (GROQ + OpenAI)
├── .gitignore
├── pyproject.toml                 # Hatch build backend, dependencies
└── .python-version                # Python 3.13
```

---

## 💡 Usage

### Example Queries

| Query | Best Analysis Type |
|---|---|
| *"Find startups similar to Stripe in fintech"* | 🔍 Similar |
| *"SWOT analysis of AI healthcare startups"* | 📊 SWOT |
| *"Who are the main competitors in food delivery?"* | ⚔️ Competitor |
| *"Compare YC edtech vs Crunchbase edtech companies"* | ⚖️ Comparison |
| *"Map the autonomous vehicle startup ecosystem"* | 🌐 Ecosystem |

### CLI Options for Batch Embedding

```bash
# Full index (default: ChromaDB)
uv run python -m bizintel.pipeline.batch_embed --reset

# Quick test with 500 docs
uv run python -m bizintel.pipeline.batch_embed --limit 500 --reset

# Use FAISS backend instead
uv run python -m bizintel.pipeline.batch_embed --backend faiss --reset

# Custom batch size
uv run python -m bizintel.pipeline.batch_embed --batch-size 1000 --reset
```

### Sidebar Controls

| Control | Options | Effect |
|---|---|---|
| **Analysis Type** | Auto, Similar, SWOT, Competitor, Comparison, Ecosystem | Changes the LLM prompt template |
| **Data Source** | All, YC, Crunchbase | Metadata filter on vector store |
| **Results to Retrieve** | 3–20 (slider) | Number of top-K documents |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.13 | Runtime |
| **Package Manager** | [uv](https://docs.astral.sh/uv/) + Hatch | Fast installs, PEP 621 build |
| **Data Processing** | pandas, numpy | CSV cleaning, vectorized operations |
| **Validation** | Pydantic v2 | Immutable document models with runtime type checking |
| **Embeddings** | sentence-transformers (`all-MiniLM-L6-v2`) | 384-dim local embeddings |
| **Vector Store** | ChromaDB / FAISS | Cosine similarity search (Strategy Pattern) |
| **Keyword Search** | rank-bm25 (`BM25Okapi`) | TF-IDF keyword matching for hybrid retrieval |
| **Reranker** | cross-encoder (`ms-marco-MiniLM-L-6-v2`) | Pair-wise relevancy rescoring |
| **LLM** | Groq (`llama-3.3-70b-versatile`) / OpenAI (`gpt-4o-mini`) | Grounded analysis generation — free or paid |
| **UI** | Streamlit | Chat interface with sidebar controls |
| **Secrets** | python-dotenv | `.env` file for API keys |

---

## 📊 Data Pipeline

```
YC CSVs (2 snapshots)           Crunchbase CSV
       │                              │
       ▼                              ▼
  load_yc_companies()         load_crunchbase_companies()
       │                              │
       ├─ Rename columns              ├─ Filter entity_type = "Company"
       ├─ Parse tags (ast.literal_eval)├─ Description fallback chain
       ├─ Extract first tag → industry ├─ Parse founded_at → year
       └─ _finalize_dataframe()        └─ _finalize_dataframe()
              │                              │
              ▼                              ▼
         YC: 4,399 rows            CB: 129,693 rows
              │                              │
              └──────────┬───────────────────┘
                         ▼
                  pd.concat() → 134,092 unified rows
                         │
                         ▼
                  add_suspicious_flags()
                         │
                         ▼
                  startups_unified.csv
```

### Unified Schema

| Column | Type | Example |
|---|---|---|
| `startup_id` | str | `"yc_12345"` |
| `name` | str | `"Stripe"` |
| `description` | str | `"Stripe builds economic infrastructure..."` |
| `industry` | str | `"fintech"` |
| `tags` | str | `"B2B, SaaS, Payments, API"` |
| `country` | str | `"US"` |
| `founded_year` | Int64 | `2010` |
| `source` | str | `"YC"` or `"Crunchbase"` |

---

## 🔑 Field Selection — Text vs Metadata

> The most critical design decision in any RAG system.

| Field | Embedded Text | Metadata | Rationale |
|---|---|---|---|
| `name` | ✅ | | Semantically meaningful — company names map to domains |
| `industry` | ✅ | | Core semantic signal |
| `tags` | ✅ | | Rich keywords for matching |
| `description` | ✅ | | Richest semantic content |
| `country` | ✅ | ✅ (dual) | Semantic ("Indian startups") + exact filter |
| `founded_year` | ✅ | ✅ (dual) | Semantic ("recent") + range filter |
| `startup_id` | | ✅ | Identifier only |
| `source` | | ✅ | For YC/Crunchbase filtering |
| `is_suspicious` | | ✅ | Quality flag |

---

## 🧪 Design Patterns Used

| Pattern | Where | Why |
|---|---|---|
| **Strategy** | `VectorStoreBase` ABC + ChromaStore/FAISSStore | Swap backends via config |
| **Factory** | `create_vector_store(backend)`, `get_llm_client()` | Centralized object creation for stores & LLMs |
| **Dependency Injection** | Retriever, Chain | Testable, decoupled components |
| **Immutable Value Object** | `StartupDocument`, `SearchResult` (frozen Pydantic) | Prevent accidental mutation |
| **Template Method** | Prompt templates (shared `_BASE_ROLE`) | Common + variable behavior |
| **Pipeline** | Offline & online data flow | Clear, testable stages |
| **LLM Client Factory** | `config/llm_client.py` → `get_llm_client(provider)` | One-flag swap between Groq (free) and OpenAI (paid) |

---

## 📈 Performance

| Metric | Value |
|---|---|
| Dataset size | 134,092 startups |
| Embedding time (CPU) | ~21 min |
| Embedding throughput | ~107 docs/sec |
| Vector search latency | ~5ms (ChromaDB HNSW) |
| BM25 search latency | ~30ms |
| RRF fusion latency | ~1ms |
| Reranker latency | ~150ms (cross-encoder, 20 → 5 docs) |
| End-to-end query time | ~2–4s (Groq) / ~3–5s (OpenAI) |
| Embedding dimensions | 384 |
| Index size (ChromaDB) | ~500 MB on disk |
| BM25 index (in-memory) | ~200 MB, builds in ~3s |



---

## 🧪 RAG Evaluation

BizIntel includes a **full evaluation pipeline** using the LLM-as-Judge pattern with 30 test queries across all 6 analysis types.

### 6 Metrics Scored

| # | Metric | Type | What It Measures |
|---|---|---|---|
| 1 | **Context Relevancy** | LLM-as-Judge | Are retrieved docs relevant to the query? |
| 2 | **Groundedness** | LLM-as-Judge | Is every claim in the answer backed by sources? |
| 3 | **Answer Relevancy** | LLM-as-Judge | Does the answer address the user's question? |
| 4 | **Precision@K** | Deterministic | % of retrieved docs matching expected domain keywords |
| 5 | **Structure Score** | Deterministic | Does the answer contain expected sections (e.g., SWOT headings)? |
| 6 | **Bad Result Check** | Deterministic | Did any known-bad results appear? (e.g., "StartupBus" for Stripe query) |

### Run Evaluation

```bash
# Full eval (30 queries — takes ~5-10 min due to LLM calls)
uv run python -m bizintel.evaluation.run_eval

# Quick test (first 5 queries)
uv run python -m bizintel.evaluation.run_eval --limit 5

# Custom output directory
uv run python -m bizintel.evaluation.run_eval --output eval_results
```

### Output

Results are saved as both **JSON** (detailed) and **CSV** (for visualization):

```
eval_results/
├── eval_20260316_143022.json    # Full results + summary + per-type breakdown
└── eval_20260316_143022.csv     # One row per query — ready for pandas/notebook
```

### Example Output

```
══════════════════════════════════════════════════════════════════════
  BizIntel RAG Evaluation — Summary
══════════════════════════════════════════════════════════════════════
  Queries evaluated : 30/30
  Total time        : 312.4s
  Avg latency       : 4.21s per query

  ┌────────────────────────┬─────────┐
  │ Metric                 │  Score  │
  ├────────────────────────┼─────────┤
  │ Context Relevancy      │  0.850  │
  │ Groundedness           │  0.920  │
  │ Answer Relevancy       │  0.890  │
  │ Precision@K            │  0.760  │
  │ Structure Score        │  0.950  │
  │ Bad Result Check       │  1.000  │
  └────────────────────────┴─────────┘
══════════════════════════════════════════════════════════════════════
```

---

## 📝 License

This project is for educational and portfolio purposes.

---

<div align="center">

**Built with ❤️ by [Shubhank Dubey](https://github.com/InsightfulShubh)**

</div>
