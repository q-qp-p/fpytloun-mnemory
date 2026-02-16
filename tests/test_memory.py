"""Tests for mnemory.memory module — pure logic that doesn't require backends."""

from unittest.mock import MagicMock, patch

import pytest

from mnemory.memory import MemoryService, _validate_id

# ── _validate_id ──────────────────────────────────────────────────────


class TestValidateId:
    def test_valid_id(self):
        assert _validate_id("filip", "user_id") == "filip"

    def test_strips_whitespace(self):
        assert _validate_id("  filip  ", "user_id") == "filip"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="user_id must not be empty"):
            _validate_id("", "user_id")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="user_id must not be empty"):
            _validate_id("   ", "user_id")

    def test_too_long_raises(self):
        with pytest.raises(ValueError, match="user_id too long"):
            _validate_id("a" * 257, "user_id")

    def test_max_length_ok(self):
        result = _validate_id("a" * 256, "user_id")
        assert len(result) == 256


# ── _rerank_by_importance ─────────────────────────────────────────────


class TestRerankByImportance:
    """Test the reranking logic without needing a full MemoryService."""

    @staticmethod
    def _rerank(memories):
        """Call the static-like rerank method via the class."""
        # Access the unbound method directly
        return MemoryService._rerank_by_importance(None, memories)

    def test_critical_boosted_over_normal(self):
        memories = [
            {"score": 0.8, "metadata": {"importance": "normal"}},
            {"score": 0.7, "metadata": {"importance": "critical"}},
        ]
        result = self._rerank(memories)
        # critical: 0.7*0.7 + 1.0*0.3 = 0.79
        # normal:   0.8*0.7 + 0.4*0.3 = 0.68
        assert result[0]["metadata"]["importance"] == "critical"

    def test_high_similarity_wins_over_low_importance(self):
        memories = [
            {"score": 0.3, "metadata": {"importance": "critical"}},
            {"score": 0.95, "metadata": {"importance": "normal"}},
        ]
        result = self._rerank(memories)
        # critical: 0.3*0.7 + 1.0*0.3 = 0.51
        # normal:   0.95*0.7 + 0.4*0.3 = 0.785
        assert result[0]["metadata"]["importance"] == "normal"

    def test_combined_score_not_in_output(self):
        memories = [{"score": 0.5, "metadata": {"importance": "normal"}}]
        result = self._rerank(memories)
        assert "_combined_score" not in result[0]

    def test_missing_importance_defaults_to_normal(self):
        memories = [
            {"score": 0.5, "metadata": {}},
            {"score": 0.5, "metadata": {"importance": "high"}},
        ]
        result = self._rerank(memories)
        # high: 0.5*0.7 + 0.7*0.3 = 0.56
        # default (normal): 0.5*0.7 + 0.4*0.3 = 0.47
        assert result[0]["metadata"]["importance"] == "high"

    def test_empty_list(self):
        assert self._rerank([]) == []

    def test_all_same_score_sorted_by_importance(self):
        memories = [
            {"score": 0.5, "metadata": {"importance": "low"}},
            {"score": 0.5, "metadata": {"importance": "critical"}},
            {"score": 0.5, "metadata": {"importance": "normal"}},
            {"score": 0.5, "metadata": {"importance": "high"}},
        ]
        result = self._rerank(memories)
        importances = [m["metadata"]["importance"] for m in result]
        assert importances == ["critical", "high", "normal", "low"]


# ── Core memory cache ─────────────────────────────────────────────────


class TestCoreMemoryCache:
    """Test that get_core_memories uses caching and invalidation."""

    @staticmethod
    def _make_service():
        """Create a MemoryService with mocked backends."""
        mock_config = MagicMock()
        mock_config.memory.max_memory_length = 1000
        mock_config.memory.max_artifact_size = 102400
        mock_config.memory.max_core_context_length = 4000
        mock_config.memory.default_recent_hours = 24
        mock_config.memory.classify_cache_ttl = 300
        mock_config.memory.core_memories_cache_ttl = 300
        mock_config.memory.auto_classify = False

        with patch("mnemory.memory.VectorStore"), patch("mnemory.memory.ArtifactStore"):
            service = MemoryService(mock_config)

        # Mock vector store methods used by get_core_memories
        service.vector = MagicMock()
        service.vector.get_pinned_memories.return_value = []
        service.vector.get_recent_memories.return_value = []

        return service

    def test_cache_hit_skips_queries(self):
        """Second call should return cached result without querying."""
        service = self._make_service()

        # First call — queries the vector store
        result1 = service.get_core_memories(user_id="filip")
        assert result1 == "No core memories found."
        assert service.vector.get_pinned_memories.call_count == 1

        # Second call — should use cache
        result2 = service.get_core_memories(user_id="filip")
        assert result2 == "No core memories found."
        # Still only 1 call — cache was used
        assert service.vector.get_pinned_memories.call_count == 1

    def test_different_users_separate_cache(self):
        """Different user_ids should have separate cache entries."""
        service = self._make_service()

        service.get_core_memories(user_id="filip")
        service.get_core_memories(user_id="alice")
        # Two separate users = two separate queries
        assert service.vector.get_pinned_memories.call_count == 2

    def test_different_agent_ids_separate_cache(self):
        """Different agent_ids should have separate cache entries."""
        service = self._make_service()

        service.get_core_memories(user_id="filip", agent_id="open-webui")
        service.get_core_memories(user_id="filip", agent_id="claude")
        # Two different agent_ids = separate cache entries
        # Each call does 2 get_pinned_memories calls (agent + user)
        assert service.vector.get_pinned_memories.call_count == 4

    def test_add_memory_invalidates_cache(self):
        """add_memory should invalidate the core cache for that user."""
        service = self._make_service()
        service.vector.add.return_value = {"results": []}

        # Populate cache
        service.get_core_memories(user_id="filip")
        assert service.vector.get_pinned_memories.call_count == 1

        # Add a memory — should invalidate
        service.add_memory(content="test", user_id="filip")

        # Next get_core_memories should query again
        service.get_core_memories(user_id="filip")
        assert service.vector.get_pinned_memories.call_count == 2

    def test_delete_memory_invalidates_cache(self):
        """delete_memory should invalidate the core cache for that user."""
        service = self._make_service()

        # Populate cache
        service.get_core_memories(user_id="filip")
        assert service.vector.get_pinned_memories.call_count == 1

        # Delete a memory
        service.delete_memory("mem-123", user_id="filip")

        # Next get_core_memories should query again
        service.get_core_memories(user_id="filip")
        assert service.vector.get_pinned_memories.call_count == 2

    def test_update_memory_invalidates_cache_for_user(self):
        """update_memory with user_id should only invalidate that user's cache."""
        service = self._make_service()

        # Populate cache for two users
        service.get_core_memories(user_id="filip")
        service.get_core_memories(user_id="alice")
        assert service.vector.get_pinned_memories.call_count == 2

        # Update a memory for filip — only filip's cache invalidated
        service.update_memory("mem-123", user_id="filip", content="updated")

        # filip should re-query, alice should use cache
        service.get_core_memories(user_id="filip")
        service.get_core_memories(user_id="alice")
        assert service.vector.get_pinned_memories.call_count == 3

    def test_update_memory_clears_all_cache_without_user_id(self):
        """update_memory without user_id should clear entire core cache."""
        service = self._make_service()

        # Populate cache for two users
        service.get_core_memories(user_id="filip")
        service.get_core_memories(user_id="alice")
        assert service.vector.get_pinned_memories.call_count == 2

        # Update without user_id — clears all core cache
        service.update_memory("mem-123", content="updated")

        # Both users should re-query
        service.get_core_memories(user_id="filip")
        service.get_core_memories(user_id="alice")
        assert service.vector.get_pinned_memories.call_count == 4

    def test_delete_all_invalidates_cache(self):
        """delete_all_memories should invalidate the core cache."""
        service = self._make_service()
        service.vector.get_all.return_value = {"results": []}

        # Populate cache
        service.get_core_memories(user_id="filip")
        assert service.vector.get_pinned_memories.call_count == 1

        # Delete all
        service.delete_all_memories(user_id="filip")

        # Next get_core_memories should query again
        service.get_core_memories(user_id="filip")
        assert service.vector.get_pinned_memories.call_count == 2

    def test_cache_disabled_when_ttl_zero(self):
        """TTL=0 should effectively disable caching."""
        mock_config = MagicMock()
        mock_config.memory.max_memory_length = 1000
        mock_config.memory.max_artifact_size = 102400
        mock_config.memory.max_core_context_length = 4000
        mock_config.memory.default_recent_hours = 24
        mock_config.memory.classify_cache_ttl = 300
        mock_config.memory.core_memories_cache_ttl = 0  # Disabled
        mock_config.memory.auto_classify = False

        with patch("mnemory.memory.VectorStore"), patch("mnemory.memory.ArtifactStore"):
            service = MemoryService(mock_config)

        service.vector = MagicMock()
        service.vector.get_pinned_memories.return_value = []
        service.vector.get_recent_memories.return_value = []

        import time

        service.get_core_memories(user_id="filip")
        time.sleep(0.01)  # Ensure TTL=0 expires
        service.get_core_memories(user_id="filip")
        # Both calls should query (cache expired immediately)
        assert service.vector.get_pinned_memories.call_count == 2


# ── Infer parameter ───────────────────────────────────────────────────


class TestInferParameter:
    """Test that the infer parameter is passed through to the vector store."""

    @staticmethod
    def _make_service():
        """Create a MemoryService with mocked backends."""
        mock_config = MagicMock()
        mock_config.memory.max_memory_length = 1000
        mock_config.memory.max_artifact_size = 102400
        mock_config.memory.max_core_context_length = 4000
        mock_config.memory.default_recent_hours = 24
        mock_config.memory.classify_cache_ttl = 300
        mock_config.memory.core_memories_cache_ttl = 300
        mock_config.memory.auto_classify = False

        with patch("mnemory.memory.VectorStore"), patch("mnemory.memory.ArtifactStore"):
            service = MemoryService(mock_config)

        service.vector = MagicMock()
        service.vector.add.return_value = {"results": []}
        return service

    def test_infer_true_by_default(self):
        """Default add_memory should pass infer=True to vector store."""
        service = self._make_service()
        service.add_memory(content="test", user_id="filip")
        service.vector.add.assert_called_once()
        _, kwargs = service.vector.add.call_args
        assert kwargs["infer"] is True

    def test_infer_false_passed_through(self):
        """add_memory(infer=False) should pass infer=False to vector store."""
        service = self._make_service()
        service.add_memory(content="test", user_id="filip", infer=False)
        service.vector.add.assert_called_once()
        _, kwargs = service.vector.add.call_args
        assert kwargs["infer"] is False

    def test_infer_true_explicit(self):
        """add_memory(infer=True) should pass infer=True to vector store."""
        service = self._make_service()
        service.add_memory(content="test", user_id="filip", infer=True)
        service.vector.add.assert_called_once()
        _, kwargs = service.vector.add.call_args
        assert kwargs["infer"] is True


# ── Batch add (add_memories tool) ─────────────────────────────────────


class TestBatchAdd:
    """Test the add_memories batch tool logic via MemoryService."""

    @staticmethod
    def _make_service():
        """Create a MemoryService with mocked backends."""
        mock_config = MagicMock()
        mock_config.memory.max_memory_length = 1000
        mock_config.memory.max_artifact_size = 102400
        mock_config.memory.max_core_context_length = 4000
        mock_config.memory.default_recent_hours = 24
        mock_config.memory.classify_cache_ttl = 300
        mock_config.memory.core_memories_cache_ttl = 300
        mock_config.memory.auto_classify = False

        with patch("mnemory.memory.VectorStore"), patch("mnemory.memory.ArtifactStore"):
            service = MemoryService(mock_config)

        service.vector = MagicMock()
        service.vector.add.return_value = {"results": [{"id": "mem-1", "event": "ADD"}]}
        return service

    def test_batch_add_multiple_memories(self):
        """Batch add should process each memory independently."""
        service = self._make_service()

        # Add 3 memories
        for content in ["fact one", "fact two", "fact three"]:
            service.add_memory(content=content, user_id="filip", infer=False)

        assert service.vector.add.call_count == 3

    def test_batch_add_passes_infer_false(self):
        """Batch add with infer=False should pass it through for each item."""
        service = self._make_service()

        service.add_memory(content="test", user_id="filip", infer=False)
        _, kwargs = service.vector.add.call_args
        assert kwargs["infer"] is False

    def test_batch_add_with_metadata(self):
        """Batch add should pass per-item metadata through."""
        service = self._make_service()

        service.add_memory(
            content="test",
            user_id="filip",
            memory_type="fact",
            categories=["work"],
            importance="high",
            pinned=True,
            infer=False,
        )

        service.vector.add.assert_called_once()
        _, kwargs = service.vector.add.call_args
        assert kwargs["metadata"]["memory_type"] == "fact"
        assert kwargs["metadata"]["categories"] == ["work"]
        assert kwargs["metadata"]["importance"] == "high"
        assert kwargs["metadata"]["pinned"] is True

    def test_content_too_long_returns_error(self):
        """Content exceeding max length should return an error dict."""
        service = self._make_service()

        result = service.add_memory(content="x" * 1001, user_id="filip", infer=False)
        assert result.get("error") is True
        assert "too long" in result["message"].lower()
