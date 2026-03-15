<div align="center">

# 🧠 BizIntel

### AI-Powered Startup Intelligence Engine

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai&logoColor=white)](https://openai.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-green)](https://www.trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Ask natural-language questions about **134,000+ startups** from Y Combinator & Crunchbase.  
Powered by **RAG (Retrieval-Augmented Generation)** — semantic search + LLM reasoning.

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

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** (fast Python package manager)
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))

### 1. Clone & Install

```bash
git clone https://github.com/InsightfulShubh/BizIntel.git
cd BizIntel
uv sync
```

### 2. Set Up Environment

```bash
# Copy the example env file and add your OpenAI API key
cp .env.example .env
# Edit .env and set: OPENAI_API_KEY=sk-your-key-here
```

### 3. Run the Data Pipeline

```bash
# Step 1: Clean & unify the raw CSV data (YC + Crunchbase → 134K unified rows)
uv run python -m bizintel.preprocessing.main

# Step 2: Embed all startups & store in vector DB (~20 min on CPU)
uv run python scripts/batch_embed.py --reset
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
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                       ONLINE PIPELINE (per query)                    │
│                                                                      │
│  User Query ──→ Query Expansion ──→ Encode ──→ Vector Search         │
│                   (LLM rewrite)     (MiniLM)    (top-K docs)        │
│                                                       │              │
│                                                       ▼              │
│                                              Prompt Template         │
│                                              (6 analysis types)      │
│                                                       │              │
│                                                       ▼              │
│                                              OpenAI GPT-4o-mini      │
│                                                       │              │
│                                                       ▼              │
│                                              Streamlit Chat UI       │
│                                              (answer + sources)      │
└──────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| **Embedding Model** | `all-MiniLM-L6-v2` (384-dim) | Free, local, fast — no API costs for 134K docs |
| **LLM** | GPT-4o-mini | Best cost/quality ratio for structured analysis |
| **Vector Store** | ChromaDB (default) + FAISS | Strategy Pattern — swap via config, no code changes |
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
│   │   └── settings.py            # Centralized config — paths, thresholds, model names
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
│   ├── rag/
│   │   ├── retriever.py           # Encode query + search vector store (DI)
│   │   ├── prompt_templates.py    # 6 analysis templates + shared base role
│   │   └── chain.py               # RAG orchestrator + query expansion
│   └── app/
│       ├── state.py               # @st.cache_resource loaders + session state
│       ├── components.py          # Sidebar, chat, source cards, CSS
│       └── streamlit_app.py       # Streamlit entry point
├── scripts/
│   └── batch_embed.py             # One-time batch embedding script (CLI)
├── notebooks/
│   └── data_analysis.ipynb        # EDA — 9 visualizations + JSON/Excel export
├── data-source/                   # Raw CSVs (not in git)
├── data/                          # Cleaned CSVs + vector DB (not in git)
├── docs/
│   ├── architecture_flowchart.html # Interactive architecture diagram
│   └── interview_prep.html        # 50+ Q&A for interview preparation
├── tests/
│   └── spot_check.py              # Verify real companies in results
├── .env.example                   # Template for API keys
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
uv run python scripts/batch_embed.py --reset

# Quick test with 500 docs
uv run python scripts/batch_embed.py --limit 500 --reset

# Use FAISS backend instead
uv run python scripts/batch_embed.py --backend faiss --reset

# Custom batch size
uv run python scripts/batch_embed.py --batch-size 1000 --reset
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
| **LLM** | OpenAI GPT-4o-mini | Grounded analysis generation |
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
| **Factory** | `create_vector_store(backend)` | Centralized object creation |
| **Dependency Injection** | Retriever, Chain | Testable, decoupled components |
| **Immutable Value Object** | `StartupDocument`, `SearchResult` (frozen Pydantic) | Prevent accidental mutation |
| **Template Method** | Prompt templates (shared `_BASE_ROLE`) | Common + variable behavior |
| **Pipeline** | Offline & online data flow | Clear, testable stages |

---

## 📈 Performance

| Metric | Value |
|---|---|
| Dataset size | 134,092 startups |
| Embedding time (CPU) | ~21 min |
| Embedding throughput | ~107 docs/sec |
| Vector search latency | ~5ms (ChromaDB HNSW) |
| End-to-end query time | ~3–5s (including LLM) |
| Embedding dimensions | 384 |
| Index size (ChromaDB) | ~500 MB on disk |

---

## 🗺️ Docs

| Document | Description |
|---|---|
| [Architecture Flowchart](docs/architecture_flowchart.html) | Interactive HTML diagram of the entire code flow with methods |
| [Interview Prep Guide](docs/interview_prep.html) | 50+ Q&A covering every design decision for interviews |

---

## 📝 License

This project is for educational and portfolio purposes.

---

<div align="center">

**Built with ❤️ by [Shubhank Dubey](https://github.com/InsightfulShubh)**

</div>
