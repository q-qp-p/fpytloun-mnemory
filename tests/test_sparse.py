"""Tests for the BM25 sparse embedding client."""

from __future__ import annotations

from unittest.mock import MagicMock


class TestSparseEmbeddingClient:
    """Tests for SparseEmbeddingClient."""

    def test_embed_batch_returns_none_for_empty_list(self):
        """embed_batch() returns None for empty input."""
        from mnemory.sparse import SparseEmbeddingClient

        client = SparseEmbeddingClient.__new__(SparseEmbeddingClient)
        client._model = MagicMock()

        assert client.embed_batch([]) is None

    def test_embed_with_mock_model(self):
        """embed() returns SparseVector when model is available."""
        import numpy as np

        from mnemory.sparse import SparseEmbeddingClient

        # Create a mock sparse embedding result
        mock_result = MagicMock()
        mock_result.indices = np.array([1, 5, 10])
        mock_result.values = np.array([0.5, 0.3, 0.8])

        mock_model = MagicMock()
        mock_model.embed.return_value = [mock_result]

        client = SparseEmbeddingClient.__new__(SparseEmbeddingClient)
        client._model = mock_model

        result = client.embed("test query")
        assert result is not None
        assert result.indices == [1, 5, 10]
        assert result.values == [0.5, 0.3, 0.8]
        mock_model.embed.assert_called_once_with(["test query"])

    def test_embed_batch_with_mock_model(self):
        """embed_batch() returns list of SparseVectors."""
        import numpy as np

        from mnemory.sparse import SparseEmbeddingClient

        mock_result1 = MagicMock()
        mock_result1.indices = np.array([1, 2])
        mock_result1.values = np.array([0.5, 0.3])

        mock_result2 = MagicMock()
        mock_result2.indices = np.array([3, 4])
        mock_result2.values = np.array([0.7, 0.1])

        mock_model = MagicMock()
        mock_model.embed.return_value = [mock_result1, mock_result2]

        client = SparseEmbeddingClient.__new__(SparseEmbeddingClient)
        client._model = mock_model

        results = client.embed_batch(["text1", "text2"])
        assert results is not None
        assert len(results) == 2
        assert results[0].indices == [1, 2]
        assert results[1].indices == [3, 4]
        mock_model.embed.assert_called_once_with(["text1", "text2"])

    def test_available_always_true(self):
        """available property is always True (fastembed is required)."""
        from mnemory.sparse import SparseEmbeddingClient

        client = SparseEmbeddingClient.__new__(SparseEmbeddingClient)
        assert client.available is True

    def test_embed_returns_none_for_empty_result(self):
        """embed() returns None when model returns empty results."""
        from mnemory.sparse import SparseEmbeddingClient

        mock_model = MagicMock()
        mock_model.embed.return_value = []

        client = SparseEmbeddingClient.__new__(SparseEmbeddingClient)
        client._model = mock_model

        assert client.embed("test") is None
