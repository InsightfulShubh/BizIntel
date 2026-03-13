"""
Document builder — converts a cleaned DataFrame into structured documents
ready for embedding.

Uses **vectorized pandas operations** to build Style C text for all rows
at once, avoiding slow row-by-row iteration.

Each row becomes a StartupDocument with:
  - text:     Style C (newline-separated labeled key-value) for embedding
  - metadata: dict for filtering in the vector store
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from bizintel.config.settings import DOCUMENT_FIELDS, METADATA_FIELDS, DUAL_FIELDS


class StartupDocument(BaseModel):
    """Immutable document ready for embedding + vector storage."""

    model_config = ConfigDict(frozen=True)

    text: str
    metadata: dict


# ── Vectorized text builder ──────────────────────────────────────────────


def _build_text_column(df: pd.DataFrame) -> pd.Series:
    """
    Build the Style C document text for every row using vectorized ops.

    For each (column, label) in DOCUMENT_FIELDS:
      - clean the column to string, strip whitespace
      - where non-empty → produce "Label: value"
      - where empty     → produce ""
    Then join all non-empty parts per row with "\\n".
    """
    # Build one Series per field: "Label: value" or ""
    parts: list[pd.Series] = []

    # Clean all document columns at once (single pass)
    doc_cols = [col for col, _ in DOCUMENT_FIELDS]
    cleaned_df = df[doc_cols].fillna("").astype(str).map(str.strip)

    for col, label in DOCUMENT_FIELDS:
        cleaned = cleaned_df[col]
        # Non-empty → "Label: value"; empty → ""
        labelled = np.where(cleaned != "", label + ": " + cleaned, "")
        parts.append(pd.Series(labelled, index=df.index))

    # Stack all part-columns side-by-side, then join per row (skip empties)
    combined = pd.concat(parts, axis=1)
    texts = combined.apply(lambda row: "\n".join(v for v in row if v), axis=1)

    return texts


# ── Vectorized metadata builder ──────────────────────────────────────────


def _build_metadata_records(df: pd.DataFrame) -> list[dict]:
    """
    Build metadata dicts for every row using a single bulk conversion.
    """
    meta_cols = list({*METADATA_FIELDS, *DUAL_FIELDS})

    meta_df = df[meta_cols].fillna("").astype(str).map(str.strip)

    return meta_df.to_dict("records")


# ── Public API ───────────────────────────────────────────────────────────


def build_documents(df: pd.DataFrame) -> list[StartupDocument]:
    """
    Convert an entire DataFrame into a list of StartupDocuments.

    Uses vectorized pandas/numpy operations for the heavy lifting:
      - text column built via np.where + column concat
      - metadata built via bulk to_dict

    Parameters
    ----------
    df : pd.DataFrame
        A cleaned unified DataFrame (must contain all columns referenced
        in DOCUMENT_FIELDS, METADATA_FIELDS, and DUAL_FIELDS).

    Returns
    -------
    list[StartupDocument]
    """
    texts = _build_text_column(df)
    metadatas = _build_metadata_records(df)

    # Zip texts + metadatas and filter out empty documents
    documents = [
        StartupDocument(text=text, metadata=meta)
        for text, meta in zip(texts, metadatas)
        if text.strip()
    ]

    return documents
