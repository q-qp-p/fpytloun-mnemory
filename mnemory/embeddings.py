"""Embedding client wrapper for mnemory.

Provides a thin, cached OpenAI-compatible embedding client with
batch support for efficient multi-text embedding.
"""

from __future__ import annotations

import logging
import time

from openai import OpenAI

from mnemory.config import EmbedConfig

logger = logging.getLogger(__name__)


def _observe_embedding(method: str, duration: float) -> None:
    """Record embedding timing in Prometheus (if enabled)."""
    from mnemory.metrics import get_collector

    collector = get_collector()
    if collector:
        collector.observe_embedding_duration("dense", method, duration)


class EmbeddingClient:
    """Cached OpenAI-compatible embedding client.

    Reuses a single HTTP connection pool across calls. Supports both
    single-text and batch embedding.
    """

    def __init__(self, config: EmbedConfig):
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self._model = config.model
        self._dims = config.dims

    @property
    def dims(self) -> int:
        """Embedding dimensions."""
        return self._dims

    def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Returns the embedding vector as a list of floats.
        """
        t0 = time.monotonic()
        text = text.replace("\n", " ")
        response = self._client.embeddings.create(
            input=[text],
            model=self._model,
            dimensions=self._dims,
        )
        duration = time.monotonic() - t0
        logger.debug(
            "Embedding: method=embed model=%s duration_ms=%d",
            self._model,
            int(duration * 1000),
        )
        _observe_embedding("embed", duration)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call.

        More efficient than calling embed() in a loop — uses one HTTP
        request instead of N.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text, in the same order.
        """
        if not texts:
            return []

        t0 = time.monotonic()
        # OpenAI supports up to ~2048 inputs per batch call, but we
        # typically have <10 facts per add_memory call.
        cleaned = [t.replace("\n", " ") for t in texts]
        response = self._client.embeddings.create(
            input=cleaned,
            model=self._model,
            dimensions=self._dims,
        )
        duration = time.monotonic() - t0
        logger.debug(
            "Embedding: method=embed_batch model=%s count=%d duration_ms=%d",
            self._model,
            len(texts),
            int(duration * 1000),
        )
        _observe_embedding("embed_batch", duration)
        # Response data is sorted by index, but sort explicitly to be safe
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [d.embedding for d in sorted_data]
