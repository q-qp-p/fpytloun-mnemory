"""BM25 sparse embedding client for hybrid search.

Provides a thin wrapper around FastEmbed's SparseTextEmbedding for
generating BM25 sparse vectors. Used alongside dense embeddings for
Qdrant hybrid search with RRF fusion.

fastembed is a required dependency — hybrid search is always active.
"""

from __future__ import annotations

import logging

from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector

logger = logging.getLogger(__name__)


class SparseEmbeddingClient:
    """BM25 sparse embedding client using FastEmbed.

    Eagerly loads the model at init time so that download failures
    surface immediately at startup rather than on the first search.

    The ``Qdrant/bm25`` model is ~50 MB and is cached by FastEmbed
    in ``~/.cache/fastembed/`` (or the Docker image layer when
    pre-downloaded at build time).
    """

    def __init__(self, model_name: str = "Qdrant/bm25"):
        self._model_name = model_name
        self._model = SparseTextEmbedding(model_name=model_name)
        logger.info("Sparse embedding model loaded: %s", model_name)

    @property
    def available(self) -> bool:
        """Whether the sparse embedding model is loaded and ready.

        Always True since fastembed is a required dependency.
        Kept for backward compatibility with callers that check this.
        """
        return True

    def embed(self, text: str) -> SparseVector | None:
        """Generate a BM25 sparse vector for a single text.

        Returns a ``qdrant_client.models.SparseVector``.
        """
        results = list(self._model.embed([text]))
        if not results:
            return None
        return SparseVector(
            indices=results[0].indices.tolist(),
            values=results[0].values.tolist(),
        )

    def embed_batch(self, texts: list[str]) -> list[SparseVector] | None:
        """Generate BM25 sparse vectors for multiple texts.

        Returns a list of ``qdrant_client.models.SparseVector`` objects
        (one per input text, in order), or ``None`` for empty input.
        """
        if not texts:
            return None

        results = list(self._model.embed(texts))
        return [
            SparseVector(
                indices=r.indices.tolist(),
                values=r.values.tolist(),
            )
            for r in results
        ]
