"""
Batch Embedding Generator — embed all cleaned startups and store them
in the vector DB.

This is a one-time (or periodic) batch job.  Run it before launching the
Streamlit UI so the vector store has data to search.

Usage:
    uv run python scripts/batch_embed.py                  # defaults (chroma)
    uv run python scripts/batch_embed.py --backend faiss
    uv run python scripts/batch_embed.py --limit 1000     # quick test run

Takes ~15-20 min on CPU for the full 134 K dataset with all-MiniLM-L6-v2.
"""

from __future__ import annotations

import argparse
import logging
import time
import sys

import pandas as pd

from bizintel.config.settings import (
    OUTPUT_DIR,
    UNIFIED_OUTPUT_FILENAME,
    VECTOR_STORE_BACKEND,
    EMBEDDING_BATCH_SIZE,
)
from bizintel.embeddings.document_builder import build_documents
from bizintel.embeddings.embedder import StartupEmbedder
from bizintel.vectorstore.base import create_vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s",
)
logger = logging.getLogger("batch_embed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BizIntel — Batch Embedding Generator",
    )
    parser.add_argument(
        "--backend",
        choices=["chroma", "faiss"],
        default=VECTOR_STORE_BACKEND,
        help=f"Vector store backend (default: {VECTOR_STORE_BACKEND})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit rows to embed (for testing). Default: all.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the vector store before embedding (delete existing data).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=EMBEDDING_BATCH_SIZE,
        help=f"Embedding batch size (default: {EMBEDDING_BATCH_SIZE})",
    )
    args = parser.parse_args()

    # ── Step 1: Load data ────────────────────────────────────────────
    csv_path = OUTPUT_DIR / UNIFIED_OUTPUT_FILENAME
    if not csv_path.exists():
        logger.error(
            "Unified CSV not found at %s. Run the preprocessing pipeline first:\n"
            "  uv run python -m bizintel.processing.main",
            csv_path,
        )
        sys.exit(1)

    logger.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path, nrows=args.limit)
    logger.info("Loaded %d rows%s", len(df), f" (limited to {args.limit})" if args.limit else "")

    # ── Step 2: Build documents ──────────────────────────────────────
    logger.info("Building documents…")
    t0 = time.perf_counter()
    docs = build_documents(df)
    logger.info("Built %d documents in %.2fs", len(docs), time.perf_counter() - t0)

    texts = [d.text for d in docs]
    metadatas = [d.metadata for d in docs]

    # ── Step 3: Embed ────────────────────────────────────────────────
    logger.info("Embedding %d documents in batches of %d…", len(texts), args.batch_size)
    embedder = StartupEmbedder()
    t0 = time.perf_counter()
    vectors = embedder.encode(texts, batch_size=args.batch_size)
    embed_time = time.perf_counter() - t0
    logger.info(
        "Embedding complete — %d vectors (%d-dim) in %.1fs (%.0f docs/sec)",
        vectors.shape[0], vectors.shape[1], embed_time,
        vectors.shape[0] / embed_time if embed_time else 0,
    )

    # ── Step 4: Store ────────────────────────────────────────────────
    store = create_vector_store(args.backend)

    if args.reset:
        logger.info("Resetting vector store…")
        store.reset()

    existing = store.count
    if existing > 0:
        logger.warning(
            "Vector store already contains %d documents. "
            "Use --reset to clear before re-embedding.",
            existing,
        )

    logger.info("Adding %d documents to %s store…", len(texts), args.backend)
    t0 = time.perf_counter()
    store.add(texts, vectors, metadatas, batch_size=args.batch_size)
    store_time = time.perf_counter() - t0
    logger.info("Storage complete in %.1fs", store_time)

    # ── Summary ──────────────────────────────────────────────────────
    total = store.count
    logger.info(
        "\n"
        "═══════════════════════════════════════════\n"
        "  ✅  Batch embedding complete!\n"
        "  📊  Documents embedded : %d\n"
        "  🗄️  Backend            : %s\n"
        "  ⏱️  Embed time         : %.1fs\n"
        "  ⏱️  Store time         : %.1fs\n"
        "  📦  Total in store     : %d\n"
        "═══════════════════════════════════════════",
        len(docs), args.backend, embed_time, store_time, total,
    )


if __name__ == "__main__":
    main()
