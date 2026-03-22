"""
Centralised configuration for the BizIntel pipeline.

All hard-coded values (paths, thresholds, column maps, filters) live here
so they can be tweaked in one place without touching business logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ── Project paths ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # BizIntel/

DATA_SOURCE_DIR = PROJECT_ROOT / "data-source"
OUTPUT_DIR = PROJECT_ROOT / "data"

YC_CSV_PATHS: list[Path] = [
    DATA_SOURCE_DIR / "2023-02-27-yc-companies.csv",
    DATA_SOURCE_DIR / "2023-07-13-yc-companies.csv",
]
CRUNCHBASE_CSV_PATH = DATA_SOURCE_DIR / "crunchbase-organizations.csv"


# ── Column rename maps ────────────────────────────────────────────────────

YC_COLUMN_MAP: dict[str, str] = {
    "company_id": "startup_id",
    "company_name": "name",
    "country": "country",
    "year_founded": "founded_year",
}

CRUNCHBASE_COLUMN_MAP: dict[str, str] = {
    "id": "startup_id",
    "country_code": "country",
    "founded_at": "founded_at",
}


# ── Standard schema ──────────────────────────────────────────────────────

STANDARD_COLUMNS: list[str] = [
    "startup_id",
    "name",
    "description",
    "industry",
    "tags",
    "country",
    "founded_year",
    "source",
]


# ── Cleaning thresholds ──────────────────────────────────────────────────

MIN_DESCRIPTION_LENGTH = 10          # descriptions with len <= this are dropped
DEDUP_SUBSET = ["name_lower", "founded_year"]

CRUNCHBASE_ENTITY_FILTER = "Company"  # entity_type value to keep


# ── Suspicious-record thresholds ─────────────────────────────────────────

SUSPICIOUS_NAME_MIN_LENGTH = 2
SUSPICIOUS_DESC_MIN_LENGTH = 20
SUSPICIOUS_YEAR_MIN = 1900
SUSPICIOUS_PLACEHOLDER_PATTERN = r"^(test|example|demo|sample|company\d*)$"


# ── Output filenames ─────────────────────────────────────────────────────

YC_OUTPUT_FILENAME = "yc_cleaned.csv"
CRUNCHBASE_OUTPUT_FILENAME = "crunchbase_cleaned.csv"
UNIFIED_OUTPUT_FILENAME = "startups_unified.csv"


# ── Document builder config ──────────────────────────────────────────────

# Fields to embed in document text (order matters — this is the rendering order)
# Each tuple: (dataframe_column, label_in_document)
DOCUMENT_FIELDS: list[tuple[str, str]] = [
    ("name", "Name"),
    ("industry", "Industry"),
    ("tags", "Tags"),
    ("country", "Country"),
    ("founded_year", "Founded"),
    ("description", "Description"),
]

# Fields stored as metadata (not embedded, used for filtering)
METADATA_FIELDS: list[str] = [
    "startup_id",
    "source",
    "is_suspicious",
]

# Fields stored as BOTH embedded text AND metadata (for filtering + LLM context)
DUAL_FIELDS: list[str] = [
    "country",
    "founded_year",
]


# ── Embedding config ─────────────────────────────────────────────────────

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # sentence-transformers model
EMBEDDING_BATCH_SIZE = 500                    # docs per batch during encoding


# ── Vector store config ──────────────────────────────────────────────────

VECTOR_STORE_BACKEND = "chroma"              # "chroma" or "faiss"

# ChromaDB settings
CHROMA_PERSIST_DIR = OUTPUT_DIR / "chroma_db"
CHROMA_COLLECTION_NAME = "startups"

# FAISS settings
FAISS_INDEX_DIR = OUTPUT_DIR / "faiss_db"
FAISS_INDEX_FILENAME = "index.faiss"
FAISS_DOCSTORE_FILENAME = "docstore.json"


# ── Retrieval config ─────────────────────────────────────────────────────

TOP_K = 5                        # final number of docs sent to the LLM


# ── Confidence / guardrail config ────────────────────────────────────────
#
# GUARDRAILS_ENABLED = True activates the confidence gate.
# When False, the system behaves like v2 — always answers, no refusals.
#
# After reranking, the best cross-encoder score is checked against these
# thresholds to decide whether to answer, warn, or refuse.
#
#   score >= SOFT    → normal answer (high confidence)
#   HARD <= score < SOFT → answer with "⚠️ low confidence" disclaimer
#   score < HARD     → refuse: "I don't have enough information…"

GUARDRAILS_ENABLED: bool = True          # master toggle for confidence gate
CONFIDENCE_THRESHOLD_SOFT: float = 0.10  # below this → add disclaimer
CONFIDENCE_THRESHOLD_HARD: float = 0.02  # below this → refuse to answer



# ── Reranker config ──────────────────────────────────────────────────────

RERANK_ENABLED = True                                          # toggle reranking on/off
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 22 MB cross-encoder
RERANK_TOP_K_INITIAL = 20                                      # over-fetch before reranking


# ── Hybrid search config ─────────────────────────────────────────────────

HYBRID_SEARCH_ENABLED = True            # toggle hybrid (semantic + BM25)
BM25_TOP_K = 20                          # keyword results to fetch
RRF_K = 60                               # RRF constant (standard default)
RRF_WEIGHT_SEMANTIC = 1.0                # semantic search weight in RRF
RRF_WEIGHT_BM25 = 0.4                    # BM25 keyword search weight in RRF


# ── LLM provider config ──────────────────────────────────────────────────
#
# LLM_PROVIDER controls which API backend to use for all LLM calls
# (RAG chain, query expansion, eval LLM-as-judge).
#
#   "openai"  → requires OPENAI_API_KEY   (paid)
#   "groq"    → requires GROQ_API_KEY     (free tier: 30 RPM)
#
# Swap the provider here; everything else adapts automatically.

LLM_PROVIDER: str = "groq"      # "openai" | "groq"

# ── OpenAI settings ──────────────────────────────────────────────────────
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_BASE_URL: str | None = None                  # default OpenAI endpoint

# ── Groq settings (free) ─────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"               # free, 30 RPM
GROQ_BASE_URL = "https://api.groq.com/openai/v1"     # OpenAI-compatible endpoint

# ── Shared LLM settings ─────────────────────────────────────────────────
# These resolve automatically based on LLM_PROVIDER:
LLM_MODEL = GROQ_MODEL if LLM_PROVIDER == "groq" else OPENAI_MODEL
LLM_TEMPERATURE = 0.3          # low = more focused; high = more creative
LLM_MAX_TOKENS = 2048


# ── Analysis types ───────────────────────────────────────────────────────

ANALYSIS_TYPES: list[str] = [
    "auto",
    "similar",
    "swot",
    "competitor",
    "comparison",
    "ecosystem",
]

DEFAULT_ANALYSIS_TYPE = "auto"


# ── Evaluation config ────────────────────────────────────────────────────

EVAL_DIR = PROJECT_ROOT / "eval"
EVAL_RESULTS_DIR = EVAL_DIR / "results"


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# ── LangGraph retry config ───────────────────────────────────────────────

MAX_RETRIES: int = 1                     # max rewrite-retrieve-generate retries
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
