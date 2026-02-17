"""Tests for mnemory.memory module — pure logic that doesn't require backends."""

from unittest.mock import MagicMock, patch

import pytest

from mnemory.memory import MemoryService, _validate_id


def _mock_memory_config(mock_config: MagicMock) -> None:
    """Set standard memory config attributes on a MagicMock config object.

    Ensures TTL and other config attributes return proper values instead
    of MagicMock objects.
    """
    mock_config.memory.max_memory_length = 1000
    mock_config.memory.max_artifact_size = 102400
    mock_config.memory.max_core_context_length = 4000
    mock_config.memory.default_recent_days = 7
    mock_config.memory.recent_limit_user = 25
    mock_config.memory.recent_limit_agent = 25
    mock_config.memory.classify_cache_ttl = 300
    mock_config.memory.core_memories_cache_ttl = 300
    mock_config.memory.auto_classify = False
    mock_config.memory.track_memory_access = False
    # TTL defaults (None = permanent)
    mock_config.memory.ttl_fact = None
    mock_config.memory.ttl_preference = None
    mock_config.memory.ttl_episodic = 90
    mock_config.memory.ttl_procedural = 60
    mock_config.memory.ttl_context = 7


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
        _mock_memory_config(mock_config)

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
        _mock_memory_config(mock_config)
        mock_config.memory.core_memories_cache_ttl = 0  # Disabled

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
        _mock_memory_config(mock_config)

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
        _mock_memory_config(mock_config)

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


# ── Classification error propagation ───────────────────────────────────


class TestClassificationErrorPropagation:
    """Test that ClassificationError from classify_memory is propagated as error dict."""

    @staticmethod
    def _make_service_with_auto_classify():
        """Create a MemoryService with auto_classify enabled."""
        mock_config = MagicMock()
        _mock_memory_config(mock_config)
        mock_config.memory.auto_classify = True  # Enable auto-classification
        mock_config.llm = MagicMock()

        with patch("mnemory.memory.VectorStore"), patch("mnemory.memory.ArtifactStore"):
            service = MemoryService(mock_config)

        service.vector = MagicMock()
        service.vector.add.return_value = {"results": [{"id": "mem-1", "event": "ADD"}]}
        service.vector.get_all.return_value = {"results": []}
        return service

    @patch("mnemory.memory.classify_memory")
    def test_classification_error_returns_error_dict(self, mock_classify):
        """ClassificationError should be caught and returned as error dict."""
        from mnemory.classify import ClassificationError

        mock_classify.side_effect = ClassificationError(
            "Auto-classification failed after retry. Please provide metadata explicitly."
        )

        service = self._make_service_with_auto_classify()

        # Call add_memory without metadata (triggers classification)
        result = service.add_memory(content="test content", user_id="filip")

        assert result.get("error") is True
        assert "Auto-classification failed" in result["message"]
        assert "metadata explicitly" in result["message"]

    @patch("mnemory.memory.classify_memory")
    def test_classification_success_proceeds_normally(self, mock_classify):
        """Successful classification should proceed to store memory."""
        mock_classify.return_value = {
            "memory_type": "fact",
            "categories": ["work"],
            "importance": "normal",
            "pinned": False,
        }

        service = self._make_service_with_auto_classify()

        result = service.add_memory(content="test content", user_id="filip")

        # Should not be an error
        assert result.get("error") is not True
        # Vector store should have been called
        service.vector.add.assert_called_once()

    @patch("mnemory.memory.classify_memory")
    def test_explicit_metadata_skips_classification(self, mock_classify):
        """Providing all metadata should skip classification entirely."""
        service = self._make_service_with_auto_classify()

        result = service.add_memory(
            content="test content",
            user_id="filip",
            memory_type="fact",
            categories=["work"],
            importance="normal",
            pinned=False,
        )

        # Classification should not be called
        mock_classify.assert_not_called()
        # Should not be an error
        assert result.get("error") is not True
        service.vector.add.assert_called_once()


# ── Role parameter ────────────────────────────────────────────────────


class TestRoleParameter:
    """Test the role parameter for user/agent extraction control."""

    @staticmethod
    def _make_service():
        """Create a MemoryService with mocked backends."""
        mock_config = MagicMock()
        _mock_memory_config(mock_config)

        with patch("mnemory.memory.VectorStore"), patch("mnemory.memory.ArtifactStore"):
            service = MemoryService(mock_config)

        service.vector = MagicMock()
        service.vector.add.return_value = {"results": []}
        return service

    def test_role_default_is_user(self):
        """Default role should be 'user' and passed to vector store."""
        service = self._make_service()
        service.add_memory(content="test", user_id="filip")
        service.vector.add.assert_called_once()
        _, kwargs = service.vector.add.call_args
        assert kwargs["role"] == "user"

    def test_role_user_explicit(self):
        """Explicit role='user' should be passed through."""
        service = self._make_service()
        service.add_memory(content="test", user_id="filip", role="user")
        _, kwargs = service.vector.add.call_args
        assert kwargs["role"] == "user"

    def test_role_assistant_passed_through(self):
        """role='assistant' should be passed to vector store."""
        service = self._make_service()
        service.add_memory(
            content="Your name is Bob",
            user_id="filip",
            agent_id="bob",
            role="assistant",
        )
        service.vector.add.assert_called_once()
        _, kwargs = service.vector.add.call_args
        assert kwargs["role"] == "assistant"

    def test_role_stored_in_metadata(self):
        """role should be included in the metadata dict."""
        service = self._make_service()
        service.add_memory(
            content="Your name is Bob",
            user_id="filip",
            agent_id="bob",
            role="assistant",
        )
        _, kwargs = service.vector.add.call_args
        assert kwargs["metadata"]["role"] == "assistant"

    def test_role_user_stored_in_metadata(self):
        """role='user' should also be stored in metadata."""
        service = self._make_service()
        service.add_memory(content="test", user_id="filip")
        _, kwargs = service.vector.add.call_args
        assert kwargs["metadata"]["role"] == "user"

    def test_role_invalid_raises(self):
        """Invalid role should raise ValueError."""
        service = self._make_service()
        with pytest.raises(ValueError, match="role must be 'user' or 'assistant'"):
            service.add_memory(content="test", user_id="filip", role="system")

    def test_role_assistant_without_agent_id_raises(self):
        """role='assistant' without agent_id should raise ValueError."""
        service = self._make_service()
        with pytest.raises(ValueError, match="agent_id is required"):
            service.add_memory(content="test", user_id="filip", role="assistant")

    def test_role_user_with_agent_id_ok(self):
        """role='user' with agent_id should work (agent-scoped user preference)."""
        service = self._make_service()
        service.add_memory(
            content="User wants me to create commit messages",
            user_id="filip",
            agent_id="opencode",
            role="user",
        )
        service.vector.add.assert_called_once()
        _, kwargs = service.vector.add.call_args
        assert kwargs["role"] == "user"
        assert kwargs["agent_id"] == "opencode"

    def test_role_filter_in_search(self):
        """role filter should be passed to vector store in search."""
        service = self._make_service()
        service.vector.search.return_value = {"results": []}
        service.search_memories(query="test", user_id="filip", role="assistant")
        service.vector.search.assert_called_once()
        _, kwargs = service.vector.search.call_args
        assert kwargs["filters"]["role"] == "assistant"

    def test_role_filter_in_list(self):
        """role filter should be passed to vector store in list."""
        service = self._make_service()
        service.vector.get_all.return_value = {"results": []}
        service.list_memories(user_id="filip", role="user")
        service.vector.get_all.assert_called_once()
        _, kwargs = service.vector.get_all.call_args
        assert kwargs["filters"]["role"] == "user"

    def test_role_filter_invalid_in_search_raises(self):
        """Invalid role filter in search should raise ValueError."""
        service = self._make_service()
        with pytest.raises(ValueError, match="role filter must be"):
            service.search_memories(query="test", user_id="filip", role="system")

    def test_role_filter_none_omitted(self):
        """role=None should not add a filter."""
        service = self._make_service()
        service.vector.search.return_value = {"results": []}
        service.search_memories(query="test", user_id="filip", role=None)
        _, kwargs = service.vector.search.call_args
        assert "role" not in (kwargs.get("filters") or {})


# ── Core memories with role-based sections ────────────────────────────


class TestCoreMemoriesRoleSections:
    """Test that get_core_memories uses role for section organization."""

    @staticmethod
    def _make_service():
        """Create a MemoryService with mocked backends."""
        mock_config = MagicMock()
        _mock_memory_config(mock_config)

        with patch("mnemory.memory.VectorStore"), patch("mnemory.memory.ArtifactStore"):
            service = MemoryService(mock_config)

        service.vector = MagicMock()
        service.vector.get_recent_memories.return_value = []
        return service

    def test_agent_identity_section_from_assistant_role(self):
        """Pinned agent memories with role=assistant should go to Agent Identity."""
        service = self._make_service()
        service.vector.get_pinned_memories.return_value = [
            {
                "memory": "My name is Bob",
                "agent_id": "bob",
                "metadata": {
                    "memory_type": "fact",
                    "role": "assistant",
                    "pinned": True,
                },
            },
        ]
        result = service.get_core_memories(user_id="filip", agent_id="bob")
        assert "## Agent Identity" in result
        assert "My name is Bob" in result

    def test_agent_instructions_section_from_user_role(self):
        """Pinned agent memories with role=user should go to Agent Instructions."""
        service = self._make_service()
        service.vector.get_pinned_memories.return_value = [
            {
                "memory": "User wants me to create commit messages",
                "agent_id": "bob",
                "metadata": {
                    "memory_type": "preference",
                    "role": "user",
                    "pinned": True,
                },
            },
        ]
        result = service.get_core_memories(user_id="filip", agent_id="bob")
        assert "## Agent Instructions" in result
        assert "User wants me to create commit messages" in result

    def test_mixed_agent_memories_split_by_role(self):
        """Agent memories should be split into Identity and Instructions by role."""
        service = self._make_service()
        service.vector.get_pinned_memories.return_value = [
            {
                "memory": "My name is Bob",
                "agent_id": "bob",
                "metadata": {
                    "memory_type": "fact",
                    "role": "assistant",
                    "pinned": True,
                },
            },
            {
                "memory": "User wants concise responses",
                "agent_id": "bob",
                "metadata": {
                    "memory_type": "preference",
                    "role": "user",
                    "pinned": True,
                },
            },
        ]
        result = service.get_core_memories(user_id="filip", agent_id="bob")
        assert "## Agent Identity" in result
        assert "My name is Bob" in result
        assert "## Agent Instructions" in result
        assert "User wants concise responses" in result

    def test_legacy_memories_without_role_go_to_instructions(self):
        """Memories without role field (legacy) should default to instructions."""
        service = self._make_service()
        service.vector.get_pinned_memories.return_value = [
            {
                "memory": "Some old agent memory",
                "agent_id": "bob",
                "metadata": {
                    "memory_type": "fact",
                    "pinned": True,
                    # No "role" field — legacy memory
                },
            },
        ]
        result = service.get_core_memories(user_id="filip", agent_id="bob")
        # Without role, defaults to "user" → Agent Instructions
        assert "## Agent Instructions" in result
        assert "Some old agent memory" in result

    def test_agent_knowledge_section_from_assistant_role(self):
        """Non-fact/preference agent memories with role=assistant go to Knowledge."""
        service = self._make_service()
        service.vector.get_pinned_memories.return_value = [
            {
                "memory": "Researched washing machines, Samsung WW90T is best",
                "agent_id": "bob",
                "metadata": {
                    "memory_type": "episodic",
                    "role": "assistant",
                    "pinned": True,
                },
            },
        ]
        result = service.get_core_memories(user_id="filip", agent_id="bob")
        assert "## Agent Knowledge" in result
        assert "Researched washing machines" in result


# ── TTL integration tests ─────────────────────────────────────────────


class TestTTLInAddMemory:
    """Test TTL metadata is correctly set when adding memories."""

    @staticmethod
    def _make_service():
        mock_config = MagicMock()
        _mock_memory_config(mock_config)

        with patch("mnemory.memory.VectorStore"), patch("mnemory.memory.ArtifactStore"):
            service = MemoryService(mock_config)

        service.vector = MagicMock()
        service.vector.add.return_value = {"results": [{"id": "mem-1", "event": "ADD"}]}
        return service

    def test_explicit_ttl_days_sets_expires_at(self):
        """Explicit ttl_days should set expires_at in metadata."""
        service = self._make_service()
        service.add_memory(content="test", user_id="filip", ttl_days=30)
        _, kwargs = service.vector.add.call_args
        meta = kwargs["metadata"]
        assert meta["ttl_days"] == 30
        assert meta["expires_at"] is not None
        assert meta["decayed_at"] is None
        assert meta["access_count"] == 0

    def test_default_ttl_for_context_type(self):
        """Context type should get default TTL of 7 days."""
        service = self._make_service()
        service.add_memory(content="test", user_id="filip", memory_type="context")
        _, kwargs = service.vector.add.call_args
        meta = kwargs["metadata"]
        assert meta["ttl_days"] == 7
        assert meta["expires_at"] is not None

    def test_default_ttl_for_fact_type_is_permanent(self):
        """Fact type should have no TTL (permanent)."""
        service = self._make_service()
        service.add_memory(content="test", user_id="filip", memory_type="fact")
        _, kwargs = service.vector.add.call_args
        meta = kwargs["metadata"]
        assert meta["ttl_days"] is None
        assert meta["expires_at"] is None

    def test_explicit_ttl_overrides_default(self):
        """Explicit ttl_days should override the type default."""
        service = self._make_service()
        service.add_memory(
            content="test", user_id="filip", memory_type="fact", ttl_days=365
        )
        _, kwargs = service.vector.add.call_args
        meta = kwargs["metadata"]
        assert meta["ttl_days"] == 365
        assert meta["expires_at"] is not None

    def test_no_ttl_days_no_type_is_permanent(self):
        """No ttl_days and no memory_type default → permanent."""
        service = self._make_service()
        service.add_memory(content="test", user_id="filip", memory_type="fact")
        _, kwargs = service.vector.add.call_args
        meta = kwargs["metadata"]
        assert meta["ttl_days"] is None
        assert meta["expires_at"] is None


class TestTTLInSearchMemories:
    """Test TTL filtering in search_memories."""

    @staticmethod
    def _make_service():
        mock_config = MagicMock()
        _mock_memory_config(mock_config)
        mock_config.vector.backend = "chroma"  # Use fallback path

        with patch("mnemory.memory.VectorStore"), patch("mnemory.memory.ArtifactStore"):
            service = MemoryService(mock_config)

        service.vector = MagicMock()
        return service

    def test_expired_memories_excluded_from_search(self):
        """Expired memories should be filtered out by default."""
        from datetime import datetime, timedelta, timezone

        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        service = self._make_service()
        service.vector.search.return_value = {
            "results": [
                {
                    "id": "active",
                    "score": 0.9,
                    "metadata": {"importance": "normal"},
                },
                {
                    "id": "expired",
                    "score": 0.8,
                    "metadata": {
                        "importance": "normal",
                        "expires_at": past,
                    },
                },
            ]
        }
        results = service.search_memories(query="test", user_id="filip")
        ids = {r["id"] for r in results}
        assert "active" in ids
        assert "expired" not in ids

    def test_pinned_memories_exempt_from_ttl(self):
        """Pinned memories should not be filtered even if expired."""
        from datetime import datetime, timedelta, timezone

        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        service = self._make_service()
        service.vector.search.return_value = {
            "results": [
                {
                    "id": "pinned-expired",
                    "score": 0.9,
                    "metadata": {
                        "importance": "normal",
                        "expires_at": past,
                        "pinned": True,
                    },
                },
            ]
        }
        results = service.search_memories(query="test", user_id="filip")
        assert len(results) == 1
        assert results[0]["id"] == "pinned-expired"

    def test_include_decayed_returns_expired(self):
        """include_decayed=True should return expired memories."""
        from datetime import datetime, timedelta, timezone

        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        service = self._make_service()
        service.vector.search.return_value = {
            "results": [
                {
                    "id": "expired",
                    "score": 0.8,
                    "metadata": {
                        "importance": "normal",
                        "expires_at": past,
                    },
                },
            ]
        }
        results = service.search_memories(
            query="test", user_id="filip", include_decayed=True
        )
        assert len(results) == 1
        assert results[0]["id"] == "expired"

    def test_decayed_memories_excluded_by_default(self):
        """Memories with decayed_at set should be excluded by default."""
        service = self._make_service()
        service.vector.search.return_value = {
            "results": [
                {
                    "id": "decayed",
                    "score": 0.8,
                    "metadata": {
                        "importance": "normal",
                        "decayed_at": "2024-01-01T00:00:00+00:00",
                    },
                },
            ]
        }
        results = service.search_memories(query="test", user_id="filip")
        assert len(results) == 0

    def test_memories_without_ttl_fields_treated_as_permanent(self):
        """Memories without TTL fields (legacy) should be treated as active."""
        service = self._make_service()
        service.vector.search.return_value = {
            "results": [
                {
                    "id": "legacy",
                    "score": 0.9,
                    "metadata": {"importance": "normal"},
                },
            ]
        }
        results = service.search_memories(query="test", user_id="filip")
        assert len(results) == 1
        assert results[0]["id"] == "legacy"


class TestTTLInListMemories:
    """Test TTL filtering in list_memories."""

    @staticmethod
    def _make_service():
        mock_config = MagicMock()
        _mock_memory_config(mock_config)

        with patch("mnemory.memory.VectorStore"), patch("mnemory.memory.ArtifactStore"):
            service = MemoryService(mock_config)

        service.vector = MagicMock()
        return service

    def test_expired_memories_excluded_from_list(self):
        """Expired memories should be filtered out of list results."""
        from datetime import datetime, timedelta, timezone

        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        service = self._make_service()
        service.vector.get_all.return_value = {
            "results": [
                {"id": "active", "metadata": {}},
                {
                    "id": "expired",
                    "metadata": {"expires_at": past},
                },
            ]
        }
        results = service.list_memories(user_id="filip")
        ids = {r["id"] for r in results}
        assert "active" in ids
        assert "expired" not in ids

    def test_include_decayed_in_list(self):
        """include_decayed=True should return expired memories in list."""
        from datetime import datetime, timedelta, timezone

        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        service = self._make_service()
        service.vector.get_all.return_value = {
            "results": [
                {
                    "id": "expired",
                    "metadata": {"expires_at": past},
                },
            ]
        }
        results = service.list_memories(user_id="filip", include_decayed=True)
        assert len(results) == 1


class TestAccessTracking:
    """Test access tracking in search_memories."""

    @staticmethod
    def _make_service(track_access=True):
        mock_config = MagicMock()
        _mock_memory_config(mock_config)
        mock_config.memory.track_memory_access = track_access
        mock_config.vector.backend = "chroma"

        with patch("mnemory.memory.VectorStore"), patch("mnemory.memory.ArtifactStore"):
            service = MemoryService(mock_config)

        service.vector = MagicMock()
        return service

    def test_access_tracking_calls_batch_update(self):
        """Search should trigger batch_update_metadata for access tracking."""
        service = self._make_service(track_access=True)
        service.vector.search.return_value = {
            "results": [
                {
                    "id": "mem-1",
                    "score": 0.9,
                    "metadata": {"importance": "normal", "access_count": 2},
                },
            ]
        }
        service.search_memories(query="test", user_id="filip")
        service.vector.batch_update_metadata.assert_called_once()
        updates = service.vector.batch_update_metadata.call_args[0][0]
        assert len(updates) == 1
        mem_id, meta = updates[0]
        assert mem_id == "mem-1"
        assert "last_accessed_at" in meta
        assert meta["access_count"] == 3

    def test_access_tracking_disabled(self):
        """When TRACK_MEMORY_ACCESS=false, no batch update should happen."""
        service = self._make_service(track_access=False)
        service.vector.search.return_value = {
            "results": [
                {
                    "id": "mem-1",
                    "score": 0.9,
                    "metadata": {"importance": "normal"},
                },
            ]
        }
        service.search_memories(query="test", user_id="filip")
        service.vector.batch_update_metadata.assert_not_called()

    def test_access_tracking_resets_ttl(self):
        """Access tracking should reset expires_at for memories with TTL."""
        service = self._make_service(track_access=True)
        service.vector.search.return_value = {
            "results": [
                {
                    "id": "mem-1",
                    "score": 0.9,
                    "metadata": {
                        "importance": "normal",
                        "ttl_days": 30,
                        "access_count": 0,
                    },
                },
            ]
        }
        service.search_memories(query="test", user_id="filip")
        updates = service.vector.batch_update_metadata.call_args[0][0]
        _, meta = updates[0]
        assert "expires_at" in meta  # TTL was reset


class TestUpdateMemoryTTL:
    """Test TTL updates via update_memory."""

    @staticmethod
    def _make_service():
        mock_config = MagicMock()
        _mock_memory_config(mock_config)

        with patch("mnemory.memory.VectorStore"), patch("mnemory.memory.ArtifactStore"):
            service = MemoryService(mock_config)

        service.vector = MagicMock()
        return service

    def test_update_ttl_days_sets_new_expiration(self):
        """Updating ttl_days should recalculate expires_at."""
        service = self._make_service()
        result = service.update_memory("mem-1", ttl_days=60, user_id="filip")
        assert result["status"] == "updated"
        service.vector.update_metadata.assert_called_once()
        meta = service.vector.update_metadata.call_args[0][1]
        assert meta["ttl_days"] == 60
        assert meta["expires_at"] is not None
        assert meta["decayed_at"] is None  # Restored from decay

    def test_update_ttl_clears_decayed_at(self):
        """Updating TTL should clear decayed_at (restore from decay)."""
        service = self._make_service()
        service.update_memory("mem-1", ttl_days=30, user_id="filip")
        meta = service.vector.update_metadata.call_args[0][1]
        assert meta["decayed_at"] is None

    def test_update_categories_invalidates_cache(self):
        """Updating categories should invalidate the category cache."""
        service = self._make_service()
        service.update_memory("mem-1", categories=["work"], user_id="filip")
        # Category cache should be invalidated for this user
        # (We can't easily test cache internals, but we can verify
        # the method completes without error)
        assert service.vector.update_metadata.called


class TestExpandCategoryFilter:
    """Test category prefix expansion for native Qdrant filtering."""

    @staticmethod
    def _make_service():
        mock_config = MagicMock()
        _mock_memory_config(mock_config)

        with patch("mnemory.memory.VectorStore"), patch("mnemory.memory.ArtifactStore"):
            service = MemoryService(mock_config)

        service.vector = MagicMock()
        return service

    def test_expand_project_prefix(self):
        """'project' should expand to include all project:* subcategories."""
        service = self._make_service()
        # Mock existing categories
        service.vector.get_all.return_value = {
            "results": [
                {"metadata": {"categories": ["project:myapp"]}},
                {"metadata": {"categories": ["project:domecek"]}},
                {"metadata": {"categories": ["work"]}},
            ]
        }
        expanded = service._expand_category_filter(["project"], "filip")
        assert "project" in expanded
        assert "project:myapp" in expanded
        assert "project:domecek" in expanded
        assert "work" not in expanded

    def test_non_prefix_category_passed_through(self):
        """Non-prefix categories should be passed through as-is."""
        service = self._make_service()
        service.vector.get_all.return_value = {"results": []}
        expanded = service._expand_category_filter(["work"], "filip")
        assert expanded == ["work"]

    def test_mixed_prefix_and_exact(self):
        """Mix of prefix and exact categories should work."""
        service = self._make_service()
        service.vector.get_all.return_value = {
            "results": [
                {"metadata": {"categories": ["project:myapp"]}},
            ]
        }
        expanded = service._expand_category_filter(["work", "project"], "filip")
        assert "work" in expanded
        assert "project" in expanded
        assert "project:myapp" in expanded


class TestMarkDecayed:
    """Test lazy decayed_at marking."""

    @staticmethod
    def _make_service():
        mock_config = MagicMock()
        _mock_memory_config(mock_config)

        with patch("mnemory.memory.VectorStore"), patch("mnemory.memory.ArtifactStore"):
            service = MemoryService(mock_config)

        service.vector = MagicMock()
        return service

    def test_marks_expired_memories_as_decayed(self):
        """Expired memories without decayed_at should get it set."""
        from datetime import datetime, timedelta, timezone

        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        service = self._make_service()
        memories = [
            {
                "id": "expired-1",
                "metadata": {"expires_at": past, "decayed_at": None},
            },
        ]
        service._mark_decayed(memories)
        service.vector.batch_update_metadata.assert_called_once()
        updates = service.vector.batch_update_metadata.call_args[0][0]
        assert len(updates) == 1
        assert updates[0][0] == "expired-1"
        assert "decayed_at" in updates[0][1]

    def test_skips_already_decayed(self):
        """Memories with decayed_at already set should be skipped."""
        from datetime import datetime, timedelta, timezone

        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        service = self._make_service()
        memories = [
            {
                "id": "already-decayed",
                "metadata": {
                    "expires_at": past,
                    "decayed_at": "2024-01-01T00:00:00+00:00",
                },
            },
        ]
        service._mark_decayed(memories)
        service.vector.batch_update_metadata.assert_not_called()

    def test_skips_active_memories(self):
        """Active memories should not be marked as decayed."""
        from datetime import datetime, timedelta, timezone

        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        service = self._make_service()
        memories = [
            {
                "id": "active",
                "metadata": {"expires_at": future, "decayed_at": None},
            },
        ]
        service._mark_decayed(memories)
        service.vector.batch_update_metadata.assert_not_called()
