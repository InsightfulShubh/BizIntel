"""
Embedder — wraps a SentenceTransformer model for batch encoding.

Loads the model once, then encodes texts in user-controlled batches
with progress logging for observability on long-running jobs.
"""

from __future__ import annotations

import logging
import time
import math

import numpy as np
from sentence_transformers import SentenceTransformer

from bizintel.config.settings import EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)


class StartupEmbedder:
    """Thin wrapper around SentenceTransformer with batched encoding."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info("Model loaded — embedding dimension: %d", self._dim)

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""
        return self._dim

    def encode(
        self,
        texts: list[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of texts into embedding vectors.

        Processes texts in batches for observability and crash resilience.
        Each batch is logged with timing info.

        Parameters
        ----------
        texts : list[str]
            Document texts to encode.
        batch_size : int
            Number of texts per batch (default from config).
        show_progress : bool
            Whether to log per-batch progress.

        Returns
        -------
        np.ndarray
            Shape (len(texts), embedding_dim).
        """
        total = len(texts)
        n_batches = math.ceil(total / batch_size)

        logger.info(
            "Encoding %d texts in %d batches (batch_size=%d)",
            total, n_batches, batch_size,
        )

        all_embeddings: list[np.ndarray] = []
        overall_start = time.perf_counter()

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, total)
            batch_texts = texts[start_idx:end_idx]

            batch_start = time.perf_counter()
            embeddings = self._model.encode(batch_texts, show_progress_bar=False)
            batch_elapsed = time.perf_counter() - batch_start

            all_embeddings.append(embeddings)

            if show_progress:
                logger.info(
                    "Batch %d/%d done — %d docs in %.2fs",
                    i + 1, n_batches, len(batch_texts), batch_elapsed,
                )

        overall_elapsed = time.perf_counter() - overall_start
        logger.info(
            "Encoding complete — %d texts in %.2fs (%.0f docs/sec)",
            total, overall_elapsed, total / overall_elapsed if overall_elapsed else 0,
        )

        return np.vstack(all_embeddings)

    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text (e.g., a user query).

        Returns
        -------
        np.ndarray
            Shape (embedding_dim,).
        """
        return self._model.encode(text, show_progress_bar=False)
