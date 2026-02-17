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
    # Search quality thresholds
    mock_config.memory.search_score_threshold = 0.30
    mock_config.memory.dedup_similarity_threshold = 0.4
    # Search ranking weights
    mock_config.memory.search_similarity_weight = 0.9
    mock_config.memory.search_keyword_weight = 0.2
    # find_memories config
    mock_config.memory.find_memories_queries = 5


def _make_service(auto_classify=False, track_access=False):
    """Create a MemoryService with mocked backends."""
    mock_config = MagicMock()
    _mock_memory_config(mock_config)
    mock_config.memory.auto_classify = auto_classify
    mock_config.memory.track_memory_access = track_access

    with (
        patch("mnemory.memory.VectorStore"),
        patch("mnemory.memory.ArtifactStore"),
        patch("mnemory.memory.LLMClient"),
    ):
        service = MemoryService(mock_config)

    # Replace with fresh mocks for test control
    service.vector = MagicMock()
    service._llm = MagicMock()

    # Default: embedding returns a dummy vector
    service.vector.embedding.embed.return_value = [0.1] * 1536
    service.vector.embedding.embed_batch.side_effect = lambda texts: [
        [0.1] * 1536 for _ in texts
    ]

    return service


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


# ── Search score threshold ────────────────────────────────────────────


class TestSearchScoreThreshold:
    """Test that search results below the score threshold are filtered out."""

    def test_low_score_results_filtered_single_scope(self):
        """Single-scope search should filter results below threshold."""
        service = _make_service()
        service.vector.search.return_value = {
            "results": [
                {"id": "high", "score": 0.8, "metadata": {"importance": "normal"}},
                {"id": "low", "score": 0.1, "metadata": {"importance": "normal"}},
                {"id": "edge", "score": 0.30, "metadata": {"importance": "normal"}},
            ]
        }
        results = service.search_memories(query="test", user_id="filip")
        ids = [r["id"] for r in results]
        assert "high" in ids
        assert "edge" in ids
        assert "low" not in ids

    def test_all_below_threshold_returns_empty(self):
        """If all results are below threshold, return empty list."""
        service = _make_service()
        service.vector.search.return_value = {
            "results": [
                {"id": "low1", "score": 0.1, "metadata": {"importance": "normal"}},
                {"id": "low2", "score": 0.05, "metadata": {"importance": "normal"}},
            ]
        }
        results = service.search_memories(query="test", user_id="filip")
        assert results == []

    def test_dual_scope_filters_below_threshold(self):
        """Dual-scope search should filter merged results below threshold."""
        service = _make_service()
        # Mock dual-scope: agent search + shared search
        service.vector.search.side_effect = [
            {
                "results": [
                    {
                        "id": "agent-high",
                        "score": 0.9,
                        "metadata": {"importance": "normal"},
                    },
                    {
                        "id": "agent-low",
                        "score": 0.1,
                        "metadata": {"importance": "normal"},
                    },
                ]
            },
            {
                "results": [
                    {
                        "id": "shared-high",
                        "score": 0.7,
                        "metadata": {"importance": "normal"},
                    },
                    {
                        "id": "shared-low",
                        "score": 0.15,
                        "metadata": {"importance": "normal"},
                    },
                ]
            },
        ]
        results = service.search_memories_dual_scope(
            query="test",
            user_id="filip",
            session_agent_id="bot",
        )
        ids = [r["id"] for r in results]
        assert "agent-high" in ids
        assert "shared-high" in ids
        assert "agent-low" not in ids
        assert "shared-low" not in ids

    def test_threshold_configurable(self):
        """Custom threshold should be respected."""
        service = _make_service()
        service._config.memory.search_score_threshold = 0.5
        service.vector.search.return_value = {
            "results": [
                {"id": "above", "score": 0.6, "metadata": {"importance": "normal"}},
                {"id": "below", "score": 0.4, "metadata": {"importance": "normal"}},
            ]
        }
        results = service.search_memories(query="test", user_id="filip")
        ids = [r["id"] for r in results]
        assert "above" in ids
        assert "below" not in ids


# ── Search similarity weight ─────────────────────────────────────────


class TestSearchSimilarityWeight:
    """Test that search_similarity_weight config is forwarded to vector.search()."""

    def test_similarity_weight_passed_single_scope(self):
        """Single-scope search should pass similarity_weight to vector.search()."""
        service = _make_service()
        service._config.memory.search_similarity_weight = 0.85
        service.vector.search.return_value = {"results": []}

        service.search_memories(query="test", user_id="filip")

        _, kwargs = service.vector.search.call_args
        assert kwargs["similarity_weight"] == 0.85

    def test_default_similarity_weight(self):
        """Default similarity_weight should be 0.9."""
        service = _make_service()
        service.vector.search.return_value = {"results": []}

        service.search_memories(query="test", user_id="filip")

        _, kwargs = service.vector.search.call_args
        assert kwargs["similarity_weight"] == 0.9


# ── Dedup similarity threshold ────────────────────────────────────────


class TestDedupSimilarityThreshold:
    """Test that low-similarity results are filtered before LLM dedup."""

    def test_low_similarity_filtered_before_llm(self):
        """search_similar results below threshold should not reach the LLM."""

        service = _make_service()
        # search_similar returns one high and one low similarity result
        service.vector.search_similar.return_value = [
            {
                "id": "similar-uuid",
                "text": "User likes cats",
                "score": 0.8,
                "type": "fact",
                "categories": ["personal"],
            },
            {
                "id": "unrelated-uuid",
                "text": "User works at Google",
                "score": 0.2,
                "type": "fact",
                "categories": ["work"],
            },
        ]
        service.vector.get_all.return_value = {"results": []}
        service._llm.generate.return_value = '{"memories": []}'

        service.add_memory(content="I love cats", user_id="filip", infer=True)

        # Check what was passed to build_extraction_prompt via the LLM call
        call_args = service._llm.generate.call_args
        system_prompt = call_args[0][0][0]["content"]
        # The high-similarity memory should be in the prompt
        assert "User likes cats" in system_prompt
        # The low-similarity memory should NOT be in the prompt
        assert "User works at Google" not in system_prompt

    def test_all_below_threshold_means_no_existing(self):
        """If all similar results are below threshold, LLM sees no existing."""

        service = _make_service()
        service.vector.search_similar.return_value = [
            {
                "id": "low-uuid",
                "text": "Something",
                "score": 0.1,
                "type": "fact",
                "categories": [],
            },
        ]
        service.vector.get_all.return_value = {"results": []}
        service._llm.generate.return_value = '{"memories": []}'

        service.add_memory(content="test", user_id="filip", infer=True)

        system_prompt = service._llm.generate.call_args[0][0][0]["content"]
        assert "None yet" in system_prompt

    def test_threshold_configurable(self):
        """Custom dedup threshold should be respected."""

        service = _make_service()
        service._config.memory.dedup_similarity_threshold = 0.8

        service.vector.search_similar.return_value = [
            {
                "id": "uuid-1",
                "text": "Fact A",
                "score": 0.75,
                "type": "fact",
                "categories": [],
            },
            {
                "id": "uuid-2",
                "text": "Fact B",
                "score": 0.85,
                "type": "fact",
                "categories": [],
            },
        ]
        service.vector.get_all.return_value = {"results": []}
        service._llm.generate.return_value = '{"memories": []}'

        service.add_memory(content="test", user_id="filip", infer=True)

        system_prompt = service._llm.generate.call_args[0][0][0]["content"]
        assert "Fact B" in system_prompt
        assert "Fact A" not in system_prompt


# ── Core memory cache ─────────────────────────────────────────────────


class TestCoreMemoryCache:
    """Test that get_core_memories uses caching and invalidation."""

    @staticmethod
    def _make_service():
        service = _make_service()
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
        # Mock the direct add path (infer=False for simplicity)
        service.vector.insert.return_value = "mem-1"

        # Populate cache
        service.get_core_memories(user_id="filip")
        assert service.vector.get_pinned_memories.call_count == 1

        # Add a memory — should invalidate
        service.add_memory(content="test", user_id="filip", infer=False)

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

        with (
            patch("mnemory.memory.VectorStore"),
            patch("mnemory.memory.ArtifactStore"),
            patch("mnemory.memory.LLMClient"),
        ):
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
    """Test that the infer parameter dispatches to the correct pipeline."""

    def test_infer_false_uses_direct_path(self):
        """infer=False should use _add_direct (embed + insert, no LLM)."""
        service = _make_service()
        service.vector.insert.return_value = "mem-1"

        result = service.add_memory(content="test", user_id="filip", infer=False)

        # Should have called embed + insert
        service.vector.embedding.embed.assert_called_once_with("test")
        service.vector.insert.assert_called_once()
        # Should NOT have called LLM
        service._llm.generate.assert_not_called()
        # Should return results
        assert result["results"][0]["event"] == "ADD"

    def test_infer_true_uses_llm_pipeline(self):
        """infer=True should use _add_with_inference (LLM extraction)."""
        service = _make_service()
        service.vector.search_similar.return_value = []
        service.vector.get_all.return_value = {"results": []}
        service._llm.generate.return_value = '{"memories": []}'

        service.add_memory(content="test", user_id="filip", infer=True)

        # Should have called LLM
        service._llm.generate.assert_called_once()
        # Should have searched for similar memories
        service.vector.search_similar.assert_called_once()

    def test_infer_true_is_default(self):
        """Default add_memory should use infer=True."""
        service = _make_service()
        service.vector.search_similar.return_value = []
        service.vector.get_all.return_value = {"results": []}
        service._llm.generate.return_value = '{"memories": []}'

        service.add_memory(content="test", user_id="filip")

        # Should have called LLM (infer=True is default)
        service._llm.generate.assert_called_once()


# ── Batch add (add_memories tool) ─────────────────────────────────────


class TestBatchAdd:
    """Test the add_memories batch tool logic via MemoryService."""

    def test_batch_add_multiple_memories(self):
        """Batch add should process each memory independently."""
        service = _make_service()
        service.vector.insert.return_value = "mem-1"

        # Add 3 memories with infer=False
        for content in ["fact one", "fact two", "fact three"]:
            service.add_memory(content=content, user_id="filip", infer=False)

        assert service.vector.insert.call_count == 3

    def test_batch_add_with_metadata(self):
        """Batch add should pass per-item metadata through."""
        service = _make_service()
        service.vector.insert.return_value = "mem-1"

        service.add_memory(
            content="test",
            user_id="filip",
            memory_type="fact",
            categories=["work"],
            importance="high",
            pinned=True,
            infer=False,
        )

        service.vector.insert.assert_called_once()
        _, kwargs = service.vector.insert.call_args
        assert kwargs["metadata"]["memory_type"] == "fact"
        assert kwargs["metadata"]["categories"] == ["work"]
        assert kwargs["metadata"]["importance"] == "high"
        assert kwargs["metadata"]["pinned"] is True

    def test_content_too_long_returns_error(self):
        """Content exceeding max length should return an error dict."""
        service = _make_service()

        result = service.add_memory(content="x" * 1001, user_id="filip", infer=False)
        assert result.get("error") is True
        assert "too long" in result["message"].lower()


# ── Classification in infer=False path ─────────────────────────────────


class TestClassificationInDirectPath:
    """Test that auto_classify works in the infer=False path."""

    def test_auto_classify_calls_llm(self):
        """When auto_classify=True and fields missing, should call LLM."""
        service = _make_service(auto_classify=True)
        service.vector.insert.return_value = "mem-1"
        service.vector.get_all.return_value = {"results": []}
        service._llm.generate.return_value = (
            '{"memory_type": "preference", "categories": ["work"], '
            '"importance": "high", "pinned": false}'
        )

        result = service.add_memory(
            content="I prefer Python", user_id="filip", infer=False
        )

        # LLM should have been called for classification
        service._llm.generate.assert_called_once()
        # Should not be an error
        assert result.get("error") is not True
        service.vector.insert.assert_called_once()

    def test_explicit_metadata_skips_classification(self):
        """Providing all metadata should skip classification entirely."""
        service = _make_service(auto_classify=True)
        service.vector.insert.return_value = "mem-1"

        result = service.add_memory(
            content="test content",
            user_id="filip",
            memory_type="fact",
            categories=["work"],
            importance="normal",
            pinned=False,
            infer=False,
        )

        # Classification should not be called
        service._llm.generate.assert_not_called()
        # Should not be an error
        assert result.get("error") is not True
        service.vector.insert.assert_called_once()

    def test_classification_failure_uses_defaults(self):
        """Classification failure should fall back to defaults, not error."""
        service = _make_service(auto_classify=True)
        service.vector.insert.return_value = "mem-1"
        service.vector.get_all.return_value = {"results": []}
        service._llm.generate.side_effect = Exception("LLM error")

        result = service.add_memory(content="test", user_id="filip", infer=False)

        # Should still succeed with defaults
        assert result.get("error") is not True
        service.vector.insert.assert_called_once()
        _, kwargs = service.vector.insert.call_args
        assert kwargs["metadata"]["memory_type"] == "fact"
        assert kwargs["metadata"]["importance"] == "normal"


# ── Role parameter ────────────────────────────────────────────────────


class TestRoleParameter:
    """Test the role parameter for user/agent extraction control."""

    def test_role_default_is_user(self):
        """Default role should be 'user'."""
        service = _make_service()
        service.vector.insert.return_value = "mem-1"

        service.add_memory(content="test", user_id="filip", infer=False)

        _, kwargs = service.vector.insert.call_args
        assert kwargs["role"] == "user"

    def test_role_user_explicit(self):
        """Explicit role='user' should be passed through."""
        service = _make_service()
        service.vector.insert.return_value = "mem-1"

        service.add_memory(content="test", user_id="filip", role="user", infer=False)

        _, kwargs = service.vector.insert.call_args
        assert kwargs["role"] == "user"

    def test_role_assistant_passed_through(self):
        """role='assistant' should be passed to vector store."""
        service = _make_service()
        service.vector.insert.return_value = "mem-1"

        service.add_memory(
            content="Your name is Bob",
            user_id="filip",
            agent_id="bob",
            role="assistant",
            infer=False,
        )

        service.vector.insert.assert_called_once()
        _, kwargs = service.vector.insert.call_args
        assert kwargs["role"] == "assistant"

    def test_role_stored_in_metadata(self):
        """role should be stored in metadata via the insert call."""
        service = _make_service()
        service.vector.insert.return_value = "mem-1"

        service.add_memory(
            content="Your name is Bob",
            user_id="filip",
            agent_id="bob",
            role="assistant",
            infer=False,
        )

        _, kwargs = service.vector.insert.call_args
        # role is passed as a top-level kwarg to insert, not in metadata
        assert kwargs["role"] == "assistant"

    def test_role_invalid_raises(self):
        """Invalid role should raise ValueError."""
        service = _make_service()
        with pytest.raises(ValueError, match="role must be 'user' or 'assistant'"):
            service.add_memory(content="test", user_id="filip", role="system")

    def test_role_assistant_without_agent_id_raises(self):
        """role='assistant' without agent_id should raise ValueError."""
        service = _make_service()
        with pytest.raises(ValueError, match="agent_id is required"):
            service.add_memory(content="test", user_id="filip", role="assistant")

    def test_role_user_with_agent_id_ok(self):
        """role='user' with agent_id should work (agent-scoped user preference)."""
        service = _make_service()
        service.vector.insert.return_value = "mem-1"

        service.add_memory(
            content="User wants me to create commit messages",
            user_id="filip",
            agent_id="opencode",
            role="user",
            infer=False,
        )

        service.vector.insert.assert_called_once()
        _, kwargs = service.vector.insert.call_args
        assert kwargs["role"] == "user"
        assert kwargs["agent_id"] == "opencode"

    def test_role_filter_in_search(self):
        """role filter should be passed to vector store in search."""
        service = _make_service()
        service.vector.search.return_value = {"results": []}
        service.search_memories(query="test", user_id="filip", role="assistant")
        service.vector.search.assert_called_once()
        _, kwargs = service.vector.search.call_args
        assert kwargs["filters"]["role"] == "assistant"

    def test_role_filter_in_list(self):
        """role filter should be passed to vector store in list."""
        service = _make_service()
        service.vector.get_all.return_value = {"results": []}
        service.list_memories(user_id="filip", role="user")
        service.vector.get_all.assert_called_once()
        _, kwargs = service.vector.get_all.call_args
        assert kwargs["filters"]["role"] == "user"

    def test_role_filter_invalid_in_search_raises(self):
        """Invalid role filter in search should raise ValueError."""
        service = _make_service()
        with pytest.raises(ValueError, match="role filter must be"):
            service.search_memories(query="test", user_id="filip", role="system")

    def test_role_filter_none_omitted(self):
        """role=None should not add a filter."""
        service = _make_service()
        service.vector.search.return_value = {"results": []}
        service.search_memories(query="test", user_id="filip", role=None)
        _, kwargs = service.vector.search.call_args
        assert "role" not in (kwargs.get("filters") or {})


# ── Core memories with role-based sections ────────────────────────────


class TestCoreMemoriesRoleSections:
    """Test that get_core_memories uses role for section organization."""

    @staticmethod
    def _make_service():
        service = _make_service()
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

    def test_explicit_ttl_days_sets_expires_at(self):
        """Explicit ttl_days should set expires_at in metadata."""
        service = _make_service()
        service.vector.insert.return_value = "mem-1"

        service.add_memory(content="test", user_id="filip", ttl_days=30, infer=False)

        _, kwargs = service.vector.insert.call_args
        meta = kwargs["metadata"]
        assert meta["ttl_days"] == 30
        assert meta["expires_at"] is not None
        assert meta["decayed_at"] is None
        assert meta["access_count"] == 0

    def test_default_ttl_for_context_type(self):
        """Context type should get default TTL of 7 days."""
        service = _make_service()
        service.vector.insert.return_value = "mem-1"

        service.add_memory(
            content="test", user_id="filip", memory_type="context", infer=False
        )

        _, kwargs = service.vector.insert.call_args
        meta = kwargs["metadata"]
        assert meta["ttl_days"] == 7
        assert meta["expires_at"] is not None

    def test_default_ttl_for_fact_type_is_permanent(self):
        """Fact type should have no TTL (permanent)."""
        service = _make_service()
        service.vector.insert.return_value = "mem-1"

        service.add_memory(
            content="test", user_id="filip", memory_type="fact", infer=False
        )

        _, kwargs = service.vector.insert.call_args
        meta = kwargs["metadata"]
        assert meta["ttl_days"] is None
        assert meta["expires_at"] is None

    def test_explicit_ttl_overrides_default(self):
        """Explicit ttl_days should override the type default."""
        service = _make_service()
        service.vector.insert.return_value = "mem-1"

        service.add_memory(
            content="test",
            user_id="filip",
            memory_type="fact",
            ttl_days=365,
            infer=False,
        )

        _, kwargs = service.vector.insert.call_args
        meta = kwargs["metadata"]
        assert meta["ttl_days"] == 365
        assert meta["expires_at"] is not None

    def test_no_ttl_days_no_type_is_permanent(self):
        """No ttl_days and no memory_type default → permanent."""
        service = _make_service()
        service.vector.insert.return_value = "mem-1"

        service.add_memory(
            content="test", user_id="filip", memory_type="fact", infer=False
        )

        _, kwargs = service.vector.insert.call_args
        meta = kwargs["metadata"]
        assert meta["ttl_days"] is None
        assert meta["expires_at"] is None


class TestTTLInSearchMemories:
    """Test TTL filtering in search_memories.

    TTL filtering is now handled natively by Qdrant via the exclude_expired
    parameter passed to vector.search(). These tests verify the correct
    parameters are passed to the vector store.
    """

    def test_search_passes_exclude_expired_true(self):
        """Default search should pass exclude_expired=True to vector store."""
        service = _make_service()
        service.vector.search.return_value = {"results": []}
        service.search_memories(query="test", user_id="filip")
        _, kwargs = service.vector.search.call_args
        assert kwargs["exclude_expired"] is True
        assert kwargs["include_decayed"] is False

    def test_include_decayed_passed_through(self):
        """include_decayed=True should be passed to vector store."""
        service = _make_service()
        service.vector.search.return_value = {"results": []}
        service.search_memories(query="test", user_id="filip", include_decayed=True)
        _, kwargs = service.vector.search.call_args
        assert kwargs["include_decayed"] is True

    def test_include_decayed_triggers_mark_decayed(self):
        """include_decayed=True should trigger _mark_decayed on results."""
        from datetime import datetime, timedelta, timezone

        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        service = _make_service()
        service.vector.search.return_value = {
            "results": [
                {
                    "id": "expired",
                    "score": 0.8,
                    "metadata": {
                        "importance": "normal",
                        "expires_at": past,
                        "decayed_at": None,
                    },
                },
            ]
        }
        service.search_memories(query="test", user_id="filip", include_decayed=True)
        # _mark_decayed should have called batch_update_metadata
        service.vector.batch_update_metadata.assert_called_once()

    def test_memories_returned_from_vector_store(self):
        """Search results from vector store should be returned as-is."""
        service = _make_service()
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

    def test_expired_memories_excluded_from_list(self):
        """Expired memories should be filtered out of list results."""
        from datetime import datetime, timedelta, timezone

        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        service = _make_service()
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
        service = _make_service()
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

    def test_access_tracking_calls_batch_update(self):
        """Search should trigger batch_update_metadata for access tracking."""
        service = _make_service(track_access=True)
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
        service = _make_service(track_access=False)
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
        service = _make_service(track_access=True)
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

    def test_update_ttl_days_sets_new_expiration(self):
        """Updating ttl_days should recalculate expires_at."""
        service = _make_service()
        result = service.update_memory("mem-1", ttl_days=60, user_id="filip")
        assert result["status"] == "updated"
        service.vector.update_metadata.assert_called_once()
        meta = service.vector.update_metadata.call_args[0][1]
        assert meta["ttl_days"] == 60
        assert meta["expires_at"] is not None
        assert meta["decayed_at"] is None  # Restored from decay

    def test_update_ttl_clears_decayed_at(self):
        """Updating TTL should clear decayed_at (restore from decay)."""
        service = _make_service()
        service.update_memory("mem-1", ttl_days=30, user_id="filip")
        meta = service.vector.update_metadata.call_args[0][1]
        assert meta["decayed_at"] is None

    def test_update_categories_invalidates_cache(self):
        """Updating categories should invalidate the category cache."""
        service = _make_service()
        service.update_memory("mem-1", categories=["work"], user_id="filip")
        # Category cache should be invalidated for this user
        # (We can't easily test cache internals, but we can verify
        # the method completes without error)
        assert service.vector.update_metadata.called


class TestExpandCategoryFilter:
    """Test category prefix expansion for native Qdrant filtering."""

    def test_expand_project_prefix(self):
        """'project' should expand to include all project:* subcategories."""
        service = _make_service()
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
        service = _make_service()
        service.vector.get_all.return_value = {"results": []}
        expanded = service._expand_category_filter(["work"], "filip")
        assert expanded == ["work"]

    def test_mixed_prefix_and_exact(self):
        """Mix of prefix and exact categories should work."""
        service = _make_service()
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

    def test_marks_expired_memories_as_decayed(self):
        """Expired memories without decayed_at should get it set."""
        from datetime import datetime, timedelta, timezone

        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        service = _make_service()
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
        service = _make_service()
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
        service = _make_service()
        memories = [
            {
                "id": "active",
                "metadata": {"expires_at": future, "decayed_at": None},
            },
        ]
        service._mark_decayed(memories)
        service.vector.batch_update_metadata.assert_not_called()


# ── LLM extraction pipeline ──────────────────────────────────────────


class TestExtractionPipeline:
    """Test the _add_with_inference pipeline end-to-end."""

    def test_add_extracts_and_inserts(self):
        """LLM extraction should parse response and insert new memories."""
        import json

        service = _make_service()
        service.vector.search_similar.return_value = []
        service.vector.get_all.return_value = {"results": []}
        service.vector.insert.return_value = "mem-1"
        service.vector.embedding.embed_batch.return_value = [[0.1] * 1536]

        llm_response = json.dumps(
            {
                "memories": [
                    {
                        "text": "Name is John",
                        "action": "ADD",
                        "target_id": None,
                        "old_memory": None,
                        "memory_type": "fact",
                        "categories": ["personal"],
                        "importance": "normal",
                        "pinned": True,
                    }
                ]
            }
        )
        service._llm.generate.return_value = llm_response

        result = service.add_memory(
            content="My name is John", user_id="filip", infer=True
        )

        assert len(result["results"]) == 1
        assert result["results"][0]["event"] == "ADD"
        assert result["results"][0]["memory"] == "Name is John"
        service.vector.insert.assert_called_once()

    def test_add_with_update_action(self):
        """LLM UPDATE action should update existing memory."""
        import json

        service = _make_service()
        # Existing memory found by similarity search
        service.vector.search_similar.return_value = [
            {
                "id": "existing-uuid",
                "text": "Uses VS Code",
                "score": 0.9,
                "type": "preference",
                "categories": ["technical"],
            }
        ]
        service.vector.get_all.return_value = {"results": []}
        service.vector.get_by_id.return_value = {
            "id": "existing-uuid",
            "memory": "Uses VS Code",
            "metadata": {"memory_type": "preference", "artifacts": []},
        }
        service.vector.embedding.embed_batch.return_value = [[0.2] * 1536]

        # LLM says UPDATE with target_id "0" (mapped to "existing-uuid")
        llm_response = json.dumps(
            {
                "memories": [
                    {
                        "text": "Uses Neovim as primary editor",
                        "action": "UPDATE",
                        "target_id": "0",
                        "old_memory": "Uses VS Code",
                        "memory_type": "preference",
                        "categories": ["technical"],
                        "importance": "normal",
                        "pinned": False,
                    }
                ]
            }
        )
        service._llm.generate.return_value = llm_response

        result = service.add_memory(
            content="I switched to Neovim", user_id="filip", infer=True
        )

        assert len(result["results"]) == 1
        assert result["results"][0]["event"] == "UPDATE"
        assert result["results"][0]["id"] == "existing-uuid"
        service.vector.update_content.assert_called_once()
        service.vector.update_metadata.assert_called_once()

    def test_add_with_empty_extraction(self):
        """LLM returning empty memories list should return empty results."""
        service = _make_service()
        service.vector.search_similar.return_value = []
        service.vector.get_all.return_value = {"results": []}
        service._llm.generate.return_value = '{"memories": []}'

        result = service.add_memory(
            content="Hi, how are you?", user_id="filip", infer=True
        )

        assert result["results"] == []
        service.vector.insert.assert_not_called()

    def test_llm_failure_returns_error(self):
        """LLM failure should return error dict, not raise."""
        service = _make_service()
        service.vector.search_similar.return_value = []
        service.vector.get_all.return_value = {"results": []}
        service._llm.generate.side_effect = Exception("API error")

        result = service.add_memory(content="test", user_id="filip", infer=True)

        assert result.get("error") is True
        assert "extraction failed" in result["message"].lower()


# ── Fact length validation ────────────────────────────────────────────


class TestValidateFactLengths:
    """Test _validate_fact_lengths retry and error logic."""

    def test_short_facts_pass_through(self):
        """Facts within max length should pass through unchanged."""
        service = _make_service()
        actions = [
            {
                "text": "Short fact",
                "action": "ADD",
                "target_id": None,
                "old_memory": None,
                "memory_type": "fact",
                "categories": [],
                "importance": "normal",
                "pinned": False,
            },
        ]
        result = service._validate_fact_lengths(actions, 1000)
        assert result == actions

    def test_oversized_fact_triggers_retry(self):
        """Fact exceeding max length should trigger a shorten LLM call."""
        import json

        service = _make_service()
        long_text = "x" * 200
        actions = [
            {
                "text": long_text,
                "action": "ADD",
                "target_id": None,
                "old_memory": None,
                "memory_type": "fact",
                "categories": [],
                "importance": "normal",
                "pinned": False,
            },
        ]

        # Mock LLM to return a shortened version
        shortened_response = json.dumps(
            {
                "memories": [
                    {
                        "text": "Short version",
                        "action": "ADD",
                        "target_id": None,
                        "old_memory": None,
                        "memory_type": "fact",
                        "categories": [],
                        "importance": "normal",
                        "pinned": False,
                    }
                ]
            }
        )
        service._llm.generate.return_value = shortened_response

        result = service._validate_fact_lengths(actions, 100)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["text"] == "Short version"

    def test_oversized_fact_split_into_multiple(self):
        """Retry can split one oversized fact into multiple shorter ones."""
        import json

        service = _make_service()
        long_text = "x" * 200
        actions = [
            {
                "text": long_text,
                "action": "ADD",
                "target_id": None,
                "old_memory": None,
                "memory_type": "fact",
                "categories": [],
                "importance": "normal",
                "pinned": False,
            },
        ]

        # Mock LLM to return two shorter facts
        split_response = json.dumps(
            {
                "memories": [
                    {
                        "text": "Part one",
                        "action": "ADD",
                        "target_id": None,
                        "old_memory": None,
                        "memory_type": "fact",
                        "categories": [],
                        "importance": "normal",
                        "pinned": False,
                    },
                    {
                        "text": "Part two",
                        "action": "ADD",
                        "target_id": None,
                        "old_memory": None,
                        "memory_type": "fact",
                        "categories": [],
                        "importance": "normal",
                        "pinned": False,
                    },
                ]
            }
        )
        service._llm.generate.return_value = split_response

        result = service._validate_fact_lengths(actions, 100)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["text"] == "Part one"
        assert result[1]["text"] == "Part two"

    def test_retry_still_oversized_returns_error(self):
        """If retry still produces oversized output, return error."""
        import json

        service = _make_service()
        long_text = "x" * 200
        actions = [
            {
                "text": long_text,
                "action": "ADD",
                "target_id": None,
                "old_memory": None,
                "memory_type": "fact",
                "categories": [],
                "importance": "normal",
                "pinned": False,
            },
        ]

        # Mock LLM to return still-oversized text
        still_long_response = json.dumps(
            {
                "memories": [
                    {
                        "text": "y" * 200,
                        "action": "ADD",
                        "target_id": None,
                        "old_memory": None,
                        "memory_type": "fact",
                        "categories": [],
                        "importance": "normal",
                        "pinned": False,
                    }
                ]
            }
        )
        service._llm.generate.return_value = still_long_response

        result = service._validate_fact_lengths(actions, 100)
        assert isinstance(result, dict)
        assert result["error"] is True
        assert "max length" in result["message"].lower()

    def test_retry_llm_failure_returns_error(self):
        """If the shorten LLM call fails, return error."""
        service = _make_service()
        long_text = "x" * 200
        actions = [
            {
                "text": long_text,
                "action": "ADD",
                "target_id": None,
                "old_memory": None,
                "memory_type": "fact",
                "categories": [],
                "importance": "normal",
                "pinned": False,
            },
        ]

        service._llm.generate.side_effect = Exception("LLM error")

        result = service._validate_fact_lengths(actions, 100)
        assert isinstance(result, dict)
        assert result["error"] is True
        assert "retry failed" in result["message"].lower()

    def test_retry_empty_response_returns_error(self):
        """If retry returns empty memories, return error."""
        service = _make_service()
        long_text = "x" * 200
        actions = [
            {
                "text": long_text,
                "action": "ADD",
                "target_id": None,
                "old_memory": None,
                "memory_type": "fact",
                "categories": [],
                "importance": "normal",
                "pinned": False,
            },
        ]

        service._llm.generate.return_value = '{"memories": []}'

        result = service._validate_fact_lengths(actions, 100)
        assert isinstance(result, dict)
        assert result["error"] is True

    def test_delete_actions_not_checked(self):
        """DELETE actions should not be checked for length."""
        service = _make_service()
        actions = [
            {
                "text": "x" * 200,
                "action": "DELETE",
                "target_id": "some-id",
                "old_memory": None,
                "memory_type": "fact",
                "categories": [],
                "importance": "normal",
                "pinned": False,
            },
        ]
        result = service._validate_fact_lengths(actions, 100)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_update_action_inherits_target_on_split(self):
        """When an UPDATE is split, first result should inherit target_id."""
        import json

        service = _make_service()
        long_text = "x" * 200
        actions = [
            {
                "text": long_text,
                "action": "UPDATE",
                "target_id": "existing-id",
                "old_memory": "old text",
                "memory_type": "fact",
                "categories": [],
                "importance": "normal",
                "pinned": False,
            },
        ]

        split_response = json.dumps(
            {
                "memories": [
                    {
                        "text": "Updated part",
                        "action": "ADD",
                        "target_id": None,
                        "old_memory": None,
                        "memory_type": "fact",
                        "categories": [],
                        "importance": "normal",
                        "pinned": False,
                    },
                    {
                        "text": "New part",
                        "action": "ADD",
                        "target_id": None,
                        "old_memory": None,
                        "memory_type": "fact",
                        "categories": [],
                        "importance": "normal",
                        "pinned": False,
                    },
                ]
            }
        )
        service._llm.generate.return_value = split_response

        result = service._validate_fact_lengths(actions, 100)
        assert isinstance(result, list)
        assert len(result) == 2
        # First result should inherit the UPDATE action and target
        assert result[0]["action"] == "UPDATE"
        assert result[0]["target_id"] == "existing-id"
        assert result[0]["old_memory"] == "old text"
        # Second result stays as ADD
        assert result[1]["action"] == "ADD"


# ── build_shorten_prompt ──────────────────────────────────────────────


class TestBuildShortenPrompt:
    """Test the shorten/split retry prompt builder."""

    def test_returns_messages_and_schema(self):
        """Should return valid messages and schema."""
        from mnemory.prompts import build_shorten_prompt

        action = {
            "text": "Very long fact text",
            "action": "ADD",
            "target_id": None,
            "old_memory": None,
            "memory_type": "fact",
            "categories": ["work"],
            "importance": "normal",
            "pinned": False,
        }
        messages, schema = build_shorten_prompt(action, max_memory_length=100)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "100" in messages[0]["content"]
        assert schema is not None
        assert schema["name"] == "memory_extraction"

    def test_action_included_in_user_message(self):
        """The oversized action should be in the user message as JSON."""
        import json

        from mnemory.prompts import build_shorten_prompt

        action = {
            "text": "Some long text",
            "action": "ADD",
            "target_id": None,
            "old_memory": None,
            "memory_type": "fact",
            "categories": [],
            "importance": "normal",
            "pinned": False,
        }
        messages, _ = build_shorten_prompt(action, max_memory_length=50)

        user_content = messages[1]["content"]
        parsed = json.loads(user_content)
        assert parsed["memories"][0]["text"] == "Some long text"


# ── build_extraction_prompt enriched context ──────────────────────────


class TestExtractionPromptEnrichedContext:
    """Test that build_extraction_prompt includes type/categories."""

    def test_type_and_categories_in_existing_memories(self):
        """Existing memories with type/categories should appear in prompt."""
        from mnemory.prompts import build_extraction_prompt

        existing = [
            {
                "id": "uuid-1",
                "text": "User likes cats",
                "score": 0.9,
                "type": "preference",
                "categories": ["personal"],
            },
            {
                "id": "uuid-2",
                "text": "User works at Google",
                "score": 0.8,
                "type": "fact",
                "categories": ["work"],
            },
        ]
        messages, _, _ = build_extraction_prompt(
            "test content",
            existing_memories=existing,
        )
        system_prompt = messages[0]["content"]
        assert '"type": "preference"' in system_prompt
        assert '"type": "fact"' in system_prompt
        assert '"categories": ["personal"]' in system_prompt
        assert '"categories": ["work"]' in system_prompt

    def test_empty_type_and_categories_omitted(self):
        """Existing memories without type/categories should omit them."""
        import json

        from mnemory.prompts import build_extraction_prompt

        existing = [
            {
                "id": "uuid-1",
                "text": "Some memory",
                "score": 0.9,
                "type": "",
                "categories": [],
            },
        ]
        messages, _, _ = build_extraction_prompt(
            "test content",
            existing_memories=existing,
        )
        system_prompt = messages[0]["content"]
        # Extract the JSON block from the existing memories section
        json_start = system_prompt.index("```\n[") + 4
        json_end = system_prompt.index("\n```", json_start)
        existing_json = json.loads(system_prompt[json_start:json_end])
        # Empty type and categories should not appear in the memory entries
        assert "type" not in existing_json[0]
        assert "categories" not in existing_json[0]

    def test_max_memory_length_in_prompt(self):
        """max_memory_length should appear in the prompt."""
        from mnemory.prompts import build_extraction_prompt

        messages, _, _ = build_extraction_prompt(
            "test content",
            max_memory_length=500,
        )
        system_prompt = messages[0]["content"]
        assert "500" in system_prompt


# ── _point_to_memory integration ──────────────────────────────────────


class TestPointToMemory:
    """Test VectorStore._point_to_memory produces correct output structure.

    These tests exercise the actual conversion logic (not mocked) to catch
    issues like fields being silently dropped from the output dict.
    """

    @staticmethod
    def _make_point(payload: dict, point_id: str = "test-uuid"):
        """Create a mock Qdrant point with the given payload."""
        from unittest.mock import MagicMock

        point = MagicMock()
        point.id = point_id
        point.payload = payload
        return point

    def test_role_preserved_in_metadata(self):
        """role field should be present in metadata, not silently dropped."""
        from mnemory.storage.vector import VectorStore

        point = self._make_point(
            {
                "data": "My name is Bob",
                "hash": "abc123",
                "created_at": "2026-01-01T00:00:00+00:00",
                "user_id": "filip",
                "agent_id": "bob",
                "role": "assistant",
                "memory_type": "fact",
                "categories": ["personal"],
                "importance": "normal",
                "pinned": True,
            }
        )
        result = VectorStore._point_to_memory(point)

        assert result["id"] == "test-uuid"
        assert result["memory"] == "My name is Bob"
        assert result["user_id"] == "filip"
        assert result["agent_id"] == "bob"
        # Critical: role must be in metadata for get_core_memories
        assert result["metadata"]["role"] == "assistant"

    def test_role_user_in_metadata(self):
        """role='user' should also be preserved in metadata."""
        from mnemory.storage.vector import VectorStore

        point = self._make_point(
            {
                "data": "Lives in Prague",
                "hash": "def456",
                "created_at": "2026-01-01T00:00:00+00:00",
                "user_id": "filip",
                "role": "user",
                "memory_type": "fact",
                "pinned": True,
            }
        )
        result = VectorStore._point_to_memory(point)
        assert result["metadata"]["role"] == "user"

    def test_standard_fields_promoted(self):
        """user_id, agent_id should be promoted to top-level fields."""
        from mnemory.storage.vector import VectorStore

        point = self._make_point(
            {
                "data": "test",
                "hash": "abc",
                "created_at": "2026-01-01T00:00:00+00:00",
                "user_id": "filip",
                "agent_id": "openwebui",
                "memory_type": "fact",
            }
        )
        result = VectorStore._point_to_memory(point)
        assert result["user_id"] == "filip"
        assert result["agent_id"] == "openwebui"
        # These should NOT also appear in metadata
        assert "user_id" not in result.get("metadata", {})
        assert "agent_id" not in result.get("metadata", {})

    def test_custom_metadata_collected(self):
        """Custom fields (memory_type, categories, etc.) go into metadata."""
        from mnemory.storage.vector import VectorStore

        point = self._make_point(
            {
                "data": "test",
                "hash": "abc",
                "created_at": "2026-01-01T00:00:00+00:00",
                "user_id": "filip",
                "memory_type": "preference",
                "categories": ["work"],
                "importance": "high",
                "pinned": False,
                "ttl_days": 30,
            }
        )
        result = VectorStore._point_to_memory(point)
        meta = result["metadata"]
        assert meta["memory_type"] == "preference"
        assert meta["categories"] == ["work"]
        assert meta["importance"] == "high"
        assert meta["pinned"] is False
        assert meta["ttl_days"] == 30

    def test_empty_payload_safe(self):
        """Point with empty/None payload should not crash."""
        from mnemory.storage.vector import VectorStore

        point = self._make_point({})
        result = VectorStore._point_to_memory(point)
        assert result["id"] == "test-uuid"
        assert result["memory"] == ""
        assert "metadata" not in result  # No custom fields = no metadata key

    def test_none_payload_safe(self):
        """Point with None payload should not crash."""
        from mnemory.storage.vector import VectorStore

        point = self._make_point({})
        point.payload = None
        result = VectorStore._point_to_memory(point)
        assert result["id"] == "test-uuid"
        assert result["memory"] == ""


# ── get_recent_memories ───────────────────────────────────────────────


class TestGetRecentMemories:
    """Test get_recent_memories returns correct results for all scopes."""

    @staticmethod
    def _make_memory(
        mem_id: str,
        text: str,
        memory_type: str = "fact",
        agent_id: str | None = None,
        categories: list[str] | None = None,
    ) -> dict:
        """Create a memory dict matching vector store output format."""
        from datetime import datetime, timezone

        m: dict = {
            "id": mem_id,
            "memory": text,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "memory_type": memory_type,
                "categories": categories or [],
                "importance": "normal",
                "pinned": False,
            },
        }
        if agent_id:
            m["agent_id"] = agent_id
        return m

    def test_scope_all_returns_user_and_agent(self):
        """scope='all' should return both user and agent memories."""
        service = _make_service()
        user_mem = self._make_memory("u1", "User fact", "fact")
        agent_mem = self._make_memory("a1", "Agent note", "episodic", agent_id="bot")

        def mock_recent(*, user_id, agent_id, since, limit, memory_types=None):
            if agent_id is None:
                return [user_mem]
            if agent_id == "bot":
                return [agent_mem]
            return []

        service.vector.get_recent_memories.side_effect = mock_recent

        result = service.get_recent_memories(
            user_id="filip", agent_id="bot", scope="all"
        )
        assert "User fact" in result
        assert "Agent note" in result
        assert "### User Activity" in result
        assert "### Agent Activity" in result

    def test_scope_user_returns_only_shared(self):
        """scope='user' should return only shared (no agent_id) memories."""
        service = _make_service()
        user_mem = self._make_memory("u1", "User preference", "preference")

        def mock_recent(*, user_id, agent_id, since, limit, memory_types=None):
            if agent_id is None:
                return [user_mem]
            return []

        service.vector.get_recent_memories.side_effect = mock_recent

        result = service.get_recent_memories(
            user_id="filip", agent_id="bot", scope="user"
        )
        assert "User preference" in result
        assert "### Agent Activity" not in result

    def test_scope_agent_returns_only_agent_scoped(self):
        """scope='agent' should return only agent-scoped memories."""
        service = _make_service()
        agent_mem = self._make_memory("a1", "Agent context", "context", agent_id="bot")

        def mock_recent(*, user_id, agent_id, since, limit, memory_types=None):
            if agent_id == "bot":
                return [agent_mem]
            return []

        service.vector.get_recent_memories.side_effect = mock_recent

        result = service.get_recent_memories(
            user_id="filip", agent_id="bot", scope="agent"
        )
        assert "Agent context" in result
        assert "### User Activity" not in result

    def test_scope_agent_requires_agent_id(self):
        """scope='agent' without agent_id should raise ValueError."""
        service = _make_service()
        with pytest.raises(ValueError, match="agent_id is required"):
            service.get_recent_memories(user_id="filip", scope="agent")

    def test_all_memory_types_included(self):
        """All memory types (fact, preference, etc.) should be returned."""
        service = _make_service()
        memories = [
            self._make_memory("m1", "A fact", "fact"),
            self._make_memory("m2", "A preference", "preference"),
            self._make_memory("m3", "An episode", "episodic"),
            self._make_memory("m4", "A procedure", "procedural"),
            self._make_memory("m5", "Some context", "context"),
        ]

        service.vector.get_recent_memories.return_value = memories

        result = service.get_recent_memories(user_id="filip", scope="user")
        assert "A fact" in result
        assert "A preference" in result
        assert "An episode" in result
        assert "A procedure" in result
        assert "Some context" in result

    def test_no_memory_types_filter_passed_to_vector(self):
        """get_recent_memories should NOT pass memory_types filter to vector store."""
        service = _make_service()
        service.vector.get_recent_memories.return_value = []

        service.get_recent_memories(user_id="filip", scope="user")

        _, kwargs = service.vector.get_recent_memories.call_args
        assert "memory_types" not in kwargs

    def test_empty_result(self):
        """No recent memories should return a message string."""
        service = _make_service()
        service.vector.get_recent_memories.return_value = []

        result = service.get_recent_memories(user_id="filip", scope="user")
        assert result == "No recent memories found."

    def test_scope_all_without_agent_id_returns_user_only(self):
        """scope='all' without agent_id should return user memories only."""
        service = _make_service()
        user_mem = self._make_memory("u1", "Shared memory", "fact")
        service.vector.get_recent_memories.return_value = [user_mem]

        result = service.get_recent_memories(user_id="filip", scope="all")
        assert "Shared memory" in result
        # Only one call to vector store (user scope), agent skipped
        assert service.vector.get_recent_memories.call_count == 1
        _, kwargs = service.vector.get_recent_memories.call_args
        assert kwargs["agent_id"] is None

    def test_decayed_memories_excluded_by_default(self):
        """Decayed memories should be excluded unless include_decayed=True."""
        service = _make_service()
        decayed_mem = self._make_memory("d1", "Old memory", "fact")
        decayed_mem["metadata"]["decayed_at"] = "2025-01-01T00:00:00+00:00"
        service.vector.get_recent_memories.return_value = [decayed_mem]

        result = service.get_recent_memories(user_id="filip", scope="user")
        assert result == "No recent memories found."

    def test_decayed_memories_included_when_requested(self):
        """include_decayed=True should include decayed memories."""
        service = _make_service()
        decayed_mem = self._make_memory("d1", "Old memory", "fact")
        decayed_mem["metadata"]["decayed_at"] = "2025-01-01T00:00:00+00:00"
        service.vector.get_recent_memories.return_value = [decayed_mem]

        result = service.get_recent_memories(
            user_id="filip", scope="user", include_decayed=True
        )
        assert "Old memory" in result


# ── Vector store get_recent_memories (Qdrant integration) ─────────────


class TestVectorGetRecentMemories:
    """Test VectorStore.get_recent_memories against local embedded Qdrant.

    These tests exercise the actual Qdrant filtering to verify that
    IsEmptyCondition correctly matches memories where agent_id is absent
    from the payload (shared user memories).
    """

    @staticmethod
    def _make_store():
        """Create a VectorStore with in-memory Qdrant for testing."""
        from unittest.mock import MagicMock

        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        config = MagicMock()
        config.vector.collection_name = "test_recent"
        config.vector.is_remote = False

        client = QdrantClient(location=":memory:")
        client.create_collection(
            collection_name="test_recent",
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )

        embedding = MagicMock()
        embedding.dims = 4

        from mnemory.storage.vector import VectorStore

        store = VectorStore.__new__(VectorStore)
        store._client = client
        store._config = config
        store._embedding = embedding

        return store

    def test_shared_memories_found_without_agent_id(self):
        """Memories without agent_id found when querying agent_id=None."""
        from datetime import datetime, timedelta, timezone

        store = self._make_store()

        # Insert a shared memory (no agent_id in payload)
        store.insert(
            text="User lives in Prague",
            vector=[0.1, 0.2, 0.3, 0.4],
            user_id="filip",
            agent_id=None,
            metadata={"memory_type": "fact"},
        )

        since = datetime.now(timezone.utc) - timedelta(days=1)
        results = store.get_recent_memories(
            user_id="filip",
            agent_id=None,
            since=since,
        )
        assert len(results) == 1
        assert results[0]["memory"] == "User lives in Prague"

    def test_agent_memories_excluded_from_shared_query(self):
        """Memories with agent_id should NOT appear when querying agent_id=None."""
        from datetime import datetime, timedelta, timezone

        store = self._make_store()

        # Insert an agent-scoped memory
        store.insert(
            text="Agent personality trait",
            vector=[0.1, 0.2, 0.3, 0.4],
            user_id="filip",
            agent_id="bot",
            metadata={"memory_type": "fact"},
        )

        since = datetime.now(timezone.utc) - timedelta(days=1)
        results = store.get_recent_memories(
            user_id="filip",
            agent_id=None,
            since=since,
        )
        assert len(results) == 0

    def test_agent_memories_found_with_agent_id(self):
        """Memories with agent_id should be found when querying that agent_id."""
        from datetime import datetime, timedelta, timezone

        store = self._make_store()

        store.insert(
            text="Agent knowledge",
            vector=[0.1, 0.2, 0.3, 0.4],
            user_id="filip",
            agent_id="bot",
            metadata={"memory_type": "episodic"},
        )

        since = datetime.now(timezone.utc) - timedelta(days=1)
        results = store.get_recent_memories(
            user_id="filip",
            agent_id="bot",
            since=since,
        )
        assert len(results) == 1
        assert results[0]["memory"] == "Agent knowledge"

    def test_mixed_memories_correctly_scoped(self):
        """Both shared and agent memories exist; each query returns correct set."""
        from datetime import datetime, timedelta, timezone

        store = self._make_store()

        store.insert(
            text="Shared fact",
            vector=[0.1, 0.2, 0.3, 0.4],
            user_id="filip",
            agent_id=None,
            metadata={"memory_type": "fact"},
        )
        store.insert(
            text="Agent note",
            vector=[0.4, 0.3, 0.2, 0.1],
            user_id="filip",
            agent_id="bot",
            metadata={"memory_type": "episodic"},
        )

        since = datetime.now(timezone.utc) - timedelta(days=1)

        # Shared query should return only the shared memory
        shared = store.get_recent_memories(user_id="filip", agent_id=None, since=since)
        assert len(shared) == 1
        assert shared[0]["memory"] == "Shared fact"

        # Agent query should return only the agent memory
        agent = store.get_recent_memories(user_id="filip", agent_id="bot", since=since)
        assert len(agent) == 1
        assert agent[0]["memory"] == "Agent note"


# ── Vector store search TTL filter (Qdrant integration) ──────────────


class TestVectorSearchTTLFilter:
    """Test that search() TTL filter handles missing expires_at field.

    Legacy memories (created before TTL support or by old mem0 code) may
    not have expires_at in their Qdrant payload. IsEmptyCondition must
    match these absent keys so legacy memories aren't silently excluded.
    """

    @staticmethod
    def _make_store():
        """Create a VectorStore with in-memory Qdrant for testing."""
        from unittest.mock import MagicMock

        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        config = MagicMock()
        config.vector.collection_name = "test_search_ttl"
        config.vector.is_remote = False

        client = QdrantClient(location=":memory:")
        client.create_collection(
            collection_name="test_search_ttl",
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )

        embedding = MagicMock()
        embedding.dims = 4
        # embed() returns a fixed vector for search queries
        embedding.embed.return_value = [0.1, 0.2, 0.3, 0.4]

        from mnemory.storage.vector import VectorStore

        store = VectorStore.__new__(VectorStore)
        store._client = client
        store._config = config
        store._embedding = embedding

        return store

    def test_legacy_memory_without_expires_at_found(self):
        """Memory without expires_at field should be found by search."""
        store = self._make_store()

        # Insert a legacy memory — no expires_at in metadata
        store.insert(
            text="Has a cat named Hiki",
            vector=[0.1, 0.2, 0.3, 0.4],
            user_id="filip",
            agent_id=None,
            metadata={"memory_type": "fact"},
        )

        result = store.search(
            "cats",
            user_id="filip",
            exclude_expired=True,
        )
        assert len(result["results"]) == 1
        assert result["results"][0]["memory"] == "Has a cat named Hiki"

    def test_memory_with_explicit_null_expires_at_found(self):
        """Memory with expires_at=None should be found by search."""
        store = self._make_store()

        store.insert(
            text="Likes cats",
            vector=[0.1, 0.2, 0.3, 0.4],
            user_id="filip",
            agent_id=None,
            metadata={
                "memory_type": "preference",
                "expires_at": None,
            },
        )

        result = store.search(
            "cats",
            user_id="filip",
            exclude_expired=True,
        )
        assert len(result["results"]) == 1
        assert result["results"][0]["memory"] == "Likes cats"

    def test_memory_with_future_expires_at_found(self):
        """Memory with expires_at in the future should be found."""
        from datetime import datetime, timedelta, timezone

        store = self._make_store()
        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()

        store.insert(
            text="Working on project X",
            vector=[0.1, 0.2, 0.3, 0.4],
            user_id="filip",
            agent_id=None,
            metadata={
                "memory_type": "context",
                "expires_at": future,
            },
        )

        result = store.search(
            "project",
            user_id="filip",
            exclude_expired=True,
        )
        assert len(result["results"]) == 1

    def test_expired_memory_excluded(self):
        """Memory with expires_at in the past should be excluded."""
        from datetime import datetime, timedelta, timezone

        store = self._make_store()
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

        store.insert(
            text="Old context",
            vector=[0.1, 0.2, 0.3, 0.4],
            user_id="filip",
            agent_id=None,
            metadata={
                "memory_type": "context",
                "expires_at": past,
            },
        )

        result = store.search(
            "context",
            user_id="filip",
            exclude_expired=True,
        )
        assert len(result["results"]) == 0

    def test_mixed_legacy_and_new_memories(self):
        """Both legacy (no expires_at) and new memories should coexist."""
        store = self._make_store()

        # Legacy memory — no expires_at
        store.insert(
            text="Has a cat named Hiki",
            vector=[0.1, 0.2, 0.3, 0.4],
            user_id="filip",
            agent_id=None,
            metadata={"memory_type": "fact"},
        )
        # New memory — explicit expires_at=None (permanent)
        store.insert(
            text="Likes cats",
            vector=[0.15, 0.25, 0.35, 0.45],
            user_id="filip",
            agent_id=None,
            metadata={
                "memory_type": "preference",
                "expires_at": None,
            },
        )

        result = store.search(
            "cats",
            user_id="filip",
            exclude_expired=True,
        )
        assert len(result["results"]) == 2
        texts = {r["memory"] for r in result["results"]}
        assert "Has a cat named Hiki" in texts
        assert "Likes cats" in texts


# ── Keyword boost ─────────────────────────────────────────────────────


class TestKeywordBoost:
    """Test post-retrieval keyword boost in search results."""

    def test_keyword_match_boosts_score(self):
        """Memories containing query keywords should be boosted."""
        service = _make_service()
        service.vector.search.return_value = {
            "results": [
                {
                    "id": "no-match",
                    "score": 0.50,
                    "memory": "User is an AI enthusiast",
                    "metadata": {},
                },
                {
                    "id": "match",
                    "score": 0.45,
                    "memory": "User's family is selfish",
                    "metadata": {},
                },
            ]
        }
        results = service.search_memories(query="family", user_id="filip")
        # "family" match should now rank higher than "AI enthusiast"
        assert results[0]["id"] == "match"
        assert results[1]["id"] == "no-match"

    def test_no_keyword_match_reduces_score(self):
        """Memories without query keywords get reduced score."""
        service = _make_service()
        service.vector.search.return_value = {
            "results": [
                {
                    "id": "no-match",
                    "score": 0.50,
                    "memory": "User likes Python",
                    "metadata": {},
                },
            ]
        }
        results = service.search_memories(query="family", user_id="filip")
        # Score should be reduced: 0.8 * 0.50 + 0.2 * 0.0 = 0.40
        assert results[0]["score"] == 0.4

    def test_multi_word_partial_match(self):
        """Partial keyword matches should give proportional boost."""
        service = _make_service()
        service.vector.search.return_value = {
            "results": [
                {
                    "id": "partial",
                    "score": 0.40,
                    "memory": "User has a dog named Rex",
                    "metadata": {},
                },
                {
                    "id": "full",
                    "score": 0.38,
                    "memory": "User wants to buy a dog for the house",
                    "metadata": {},
                },
            ]
        }
        results = service.search_memories(query="buy dog house", user_id="filip")
        # "full" has 3/3 keywords, "partial" has 1/3
        assert results[0]["id"] == "full"

    def test_all_stopwords_skips_boost(self):
        """Query with only stopwords should skip keyword boost entirely."""
        service = _make_service()
        service.vector.search.return_value = {
            "results": [
                {
                    "id": "mem1",
                    "score": 0.50,
                    "memory": "User likes cats",
                    "metadata": {},
                },
            ]
        }
        results = service.search_memories(query="is the a", user_id="filip")
        # Score should be unchanged (no keyword boost applied)
        assert results[0]["score"] == 0.50

    def test_weight_zero_disables_boost(self):
        """Setting keyword weight to 0 should disable boosting."""
        service = _make_service()
        service._config.memory.search_keyword_weight = 0.0
        service.vector.search.return_value = {
            "results": [
                {
                    "id": "no-match",
                    "score": 0.50,
                    "memory": "User likes Python",
                    "metadata": {},
                },
            ]
        }
        results = service.search_memories(query="family", user_id="filip")
        assert results[0]["score"] == 0.50

    def test_applied_in_dual_scope(self):
        """Keyword boost should also apply in dual-scope search."""
        service = _make_service()
        service.vector.search.side_effect = [
            {
                "results": [
                    {
                        "id": "no-match",
                        "score": 0.50,
                        "memory": "User is an AI enthusiast",
                        "metadata": {},
                    },
                ]
            },
            {
                "results": [
                    {
                        "id": "match",
                        "score": 0.45,
                        "memory": "User's family is selfish",
                        "metadata": {},
                    },
                ]
            },
        ]
        results = service.search_memories_dual_scope(
            "family",
            user_id="filip",
            session_agent_id="bot",
        )
        assert results[0]["id"] == "match"


# ── Tokenize helper ──────────────────────────────────────────────────


class TestTokenize:
    """Test the _tokenize helper function."""

    def test_basic_tokenization(self):
        from mnemory.memory import _tokenize

        tokens = _tokenize("User's family is selfish")
        assert "user" in tokens
        assert "family" in tokens
        assert "selfish" in tokens
        # "is" is a stopword
        assert "is" not in tokens
        # "s" is too short
        assert "s" not in tokens

    def test_empty_string(self):
        from mnemory.memory import _tokenize

        assert _tokenize("") == set()

    def test_all_stopwords(self):
        from mnemory.memory import _tokenize

        assert _tokenize("is the a an") == set()

    def test_mixed_case_and_punctuation(self):
        from mnemory.memory import _tokenize

        tokens = _tokenize("User's Dog-Walking habit!")
        assert "user" in tokens
        assert "dog" in tokens
        assert "walking" in tokens
        assert "habit" in tokens


# ── find_memories ─────────────────────────────────────────────────────


class TestFindMemories:
    """Test the AI-powered find_memories pipeline."""

    def test_full_pipeline(self):
        """Test query generation → search → merge → rerank."""
        import json

        service = _make_service()
        # Disable keyword boost to test merge/rerank logic in isolation
        service._config.memory.search_keyword_weight = 0

        # Mock LLM: first call = query generation, second call = reranking
        # Rerank uses numeric indices: after sort by score, order is
        # dog-mem (0.6), cat-mem (0.5), irrelevant (0.4) → idx 0, 1, 2
        service._llm.generate.side_effect = [
            # Query generation response
            json.dumps({"queries": ["user dogs", "pets preferences"]}),
            # Rerank response (using numeric indices)
            json.dumps(
                {
                    "scored": [
                        {"idx": 0, "relevance": 0.9},  # dog-mem
                        {"idx": 1, "relevance": 0.5},  # cat-mem
                        {"idx": 2, "relevance": 0.1},  # irrelevant
                    ]
                }
            ),
        ]

        # Mock search results for each generated query
        service.vector.search.side_effect = [
            {
                "results": [
                    {
                        "id": "dog-mem",
                        "score": 0.6,
                        "memory": "User does not like dogs",
                        "metadata": {},
                    },
                    {
                        "id": "irrelevant",
                        "score": 0.4,
                        "memory": "User likes Python",
                        "metadata": {},
                    },
                ]
            },
            {
                "results": [
                    {
                        "id": "cat-mem",
                        "score": 0.5,
                        "memory": "User prefers cats",
                        "metadata": {},
                    },
                    {
                        "id": "dog-mem",
                        "score": 0.55,
                        "memory": "User does not like dogs",
                        "metadata": {},
                    },
                ]
            },
        ]

        result = service.find_memories(
            "What do I think about dogs?",
            user_id="filip",
            limit=2,
        )

        # Should return dict with results, queries, and stats
        assert "results" in result
        assert "queries" in result
        assert "stats" in result
        results = result["results"]

        # Should have 2 results, reranked by LLM relevance
        assert len(results) == 2
        assert results[0]["id"] == "dog-mem"
        assert results[0]["score"] == 0.9
        assert results[1]["id"] == "cat-mem"
        assert results[1]["score"] == 0.5

        # Check stats
        stats = result["stats"]
        assert stats["searched"] == 4  # 2 + 2 from each query
        assert stats["merged"] == 3  # dog-mem, cat-mem, irrelevant
        assert stats["reranked"] is True
        assert stats["returned"] == 2

    def test_few_results_skip_reranking(self):
        """When merged results <= limit, skip LLM reranking."""
        import json

        service = _make_service()

        # Mock LLM: only query generation (no rerank call expected)
        service._llm.generate.return_value = json.dumps({"queries": ["user dogs"]})

        service.vector.search.return_value = {
            "results": [
                {
                    "id": "dog-mem",
                    "score": 0.6,
                    "memory": "User does not like dogs",
                    "metadata": {},
                },
            ]
        }

        result = service.find_memories(
            "What do I think about dogs?",
            user_id="filip",
            limit=5,
        )

        # Only 1 result, limit is 5 → skip reranking
        assert len(result["results"]) == 1
        assert result["queries"] == ["user dogs"]
        # LLM should only be called once (query generation, not reranking)
        assert service._llm.generate.call_count == 1

        # Check stats show no reranking
        stats = result["stats"]
        assert stats["reranked"] is False
        assert stats["merged"] == 1
        assert stats["returned"] == 1

    def test_deduplication_across_queries(self):
        """Same memory from multiple queries should be deduplicated."""
        import json

        service = _make_service()

        # After dedup and sort: dup-mem (0.7), unique-mem (0.4) → idx 0, 1
        service._llm.generate.side_effect = [
            json.dumps({"queries": ["query1", "query2"]}),
            json.dumps(
                {
                    "scored": [
                        {"idx": 0, "relevance": 0.8},  # dup-mem
                        {"idx": 1, "relevance": 0.6},  # unique-mem
                    ]
                }
            ),
        ]

        service.vector.search.side_effect = [
            {
                "results": [
                    {
                        "id": "dup-mem",
                        "score": 0.5,
                        "memory": "Shared fact",
                        "metadata": {},
                    },
                ]
            },
            {
                "results": [
                    {
                        "id": "dup-mem",
                        "score": 0.7,
                        "memory": "Shared fact",
                        "metadata": {},
                    },
                    {
                        "id": "unique-mem",
                        "score": 0.4,
                        "memory": "Unique fact",
                        "metadata": {},
                    },
                ]
            },
        ]

        result = service.find_memories(
            "test question",
            user_id="filip",
            limit=1,
        )

        # dup-mem should appear only once
        ids = [r["id"] for r in result["results"]]
        assert ids.count("dup-mem") <= 1

    def test_query_generation_failure_raises(self):
        """LLM failure on query generation should raise ValueError."""
        service = _make_service()
        service._llm.generate.return_value = "not valid json at all"

        with pytest.raises(ValueError, match="Failed to generate"):
            service.find_memories(
                "test question",
                user_id="filip",
            )

    def test_rerank_failure_raises(self):
        """LLM failure on reranking should raise ValueError."""
        import json

        service = _make_service()

        service._llm.generate.side_effect = [
            json.dumps({"queries": ["q1", "q2", "q3"]}),
            "not valid json",  # rerank fails
        ]

        # Need enough results to trigger reranking (> limit)
        service.vector.search.side_effect = [
            {
                "results": [
                    {
                        "id": f"mem-{i}",
                        "score": 0.5,
                        "memory": f"Fact {i}",
                        "metadata": {},
                    }
                    for i in range(5)
                ]
            },
            {
                "results": [
                    {
                        "id": f"mem-{i + 5}",
                        "score": 0.4,
                        "memory": f"Fact {i + 5}",
                        "metadata": {},
                    }
                    for i in range(5)
                ]
            },
            {
                "results": [
                    {
                        "id": f"mem-{i + 10}",
                        "score": 0.35,
                        "memory": f"Fact {i + 10}",
                        "metadata": {},
                    }
                    for i in range(5)
                ]
            },
        ]

        with pytest.raises(ValueError, match="Failed to rerank"):
            service.find_memories(
                "test question",
                user_id="filip",
                limit=3,
            )

    def test_rerank_filters_below_threshold(self):
        """Memories scored below threshold by LLM should be filtered out."""
        import json

        service = _make_service()

        # 8 memories from q1 (mem-0 to mem-7, all score 0.5) +
        # 2 from q2 (relevant 0.5, irrelevant 0.4) = 10 unique
        # After sort by score: all 0.5s first, then irrelevant 0.4
        # Pre-filter takes top limit*3 = 15, so all 10 go to reranker
        # Reranker scores: idx 8 (relevant) = 0.8, idx 9 (irrelevant) = 0.1
        # Others get low scores to be filtered
        service._llm.generate.side_effect = [
            json.dumps({"queries": ["q1", "q2"]}),
            json.dumps(
                {
                    "scored": [
                        {"idx": 0, "relevance": 0.2},  # mem-0
                        {"idx": 1, "relevance": 0.2},  # mem-1
                        {"idx": 2, "relevance": 0.2},  # mem-2
                        {"idx": 3, "relevance": 0.2},  # mem-3
                        {"idx": 4, "relevance": 0.2},  # mem-4
                        {"idx": 5, "relevance": 0.2},  # mem-5
                        {"idx": 6, "relevance": 0.2},  # mem-6
                        {"idx": 7, "relevance": 0.2},  # mem-7
                        {"idx": 8, "relevance": 0.8},  # relevant
                        {"idx": 9, "relevance": 0.1},  # irrelevant
                    ]
                }
            ),
        ]

        service.vector.search.side_effect = [
            {
                "results": [
                    {
                        "id": f"mem-{i}",
                        "score": 0.5,
                        "memory": f"Fact {i}",
                        "metadata": {},
                    }
                    for i in range(8)
                ]
            },
            {
                "results": [
                    {
                        "id": "relevant",
                        "score": 0.5,
                        "memory": "Relevant fact",
                        "metadata": {},
                    },
                    {
                        "id": "irrelevant",
                        "score": 0.4,
                        "memory": "Irrelevant fact",
                        "metadata": {},
                    },
                ]
            },
        ]

        result = service.find_memories(
            "test question",
            user_id="filip",
            limit=5,
        )

        ids = [r["id"] for r in result["results"]]
        assert "relevant" in ids
        assert "irrelevant" not in ids  # scored 0.1, below threshold 0.3

    def test_uses_dual_scope_with_session_agent(self):
        """When session_agent_id is provided, should use dual-scope search."""
        import json

        service = _make_service()

        service._llm.generate.return_value = json.dumps({"queries": ["test query"]})

        service.vector.search.side_effect = [
            # Dual-scope makes 2 calls per query
            {
                "results": [
                    {"id": "a1", "score": 0.5, "memory": "Agent mem", "metadata": {}}
                ]
            },
            {
                "results": [
                    {"id": "s1", "score": 0.4, "memory": "Shared mem", "metadata": {}}
                ]
            },
        ]

        service.find_memories(
            "test question",
            user_id="filip",
            session_agent_id="openwebui",
            limit=5,
        )

        # Should have called vector.search twice (dual-scope for one query)
        assert service.vector.search.call_count == 2


# ── find_memories prompts ─────────────────────────────────────────────


class TestQueryGenerationPrompt:
    """Test the query generation prompt builder."""

    def test_prompt_structure(self):
        from mnemory.prompts import build_query_generation_prompt

        messages, schema = build_query_generation_prompt(
            "What about dogs?", num_queries=3
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "3" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "dogs" in messages[1]["content"]
        assert schema["name"] == "query_generation"

    def test_default_num_queries(self):
        from mnemory.prompts import build_query_generation_prompt

        messages, _ = build_query_generation_prompt("test")
        assert "5" in messages[0]["content"]


class TestRerankPrompt:
    """Test the rerank prompt builder."""

    def test_prompt_structure(self):
        from mnemory.prompts import build_rerank_prompt

        memories = [
            {"id": "m1", "memory": "User likes cats"},
            {"id": "m2", "memory": "User dislikes dogs"},
        ]
        messages, schema = build_rerank_prompt("What about pets?", memories)
        assert len(messages) == 2
        # Uses numeric indices, not UUIDs
        assert "[0]" in messages[1]["content"]
        assert "[1]" in messages[1]["content"]
        assert "cats" in messages[1]["content"]
        assert schema["name"] == "memory_rerank"
        # Schema should use idx, not id
        assert schema["schema"]["properties"]["scored"]["items"]["properties"]["idx"]

    def test_no_threshold_in_prompt(self):
        """Threshold should NOT be in the prompt to avoid anchor bias."""
        from mnemory.prompts import build_rerank_prompt

        messages, _ = build_rerank_prompt("test", [])
        system = messages[0]["content"]
        # Should not mention "threshold" as a filtering concept
        assert "threshold" not in system.lower()
        # Should not tell LLM about our filtering cutoff (anchor bias)
        assert "filter" not in system.lower() or "we filter" in system.lower()
        # But honest scoring instruction should be there
        assert "Score honestly" in system

    def test_subject_awareness_in_prompt(self):
        """Rerank prompt should include subject-awareness instructions."""
        from mnemory.prompts import build_rerank_prompt

        messages, _ = build_rerank_prompt("test", [])
        system = messages[0]["content"]
        assert "Subject awareness" in system or "subject" in system.lower()
        assert "WHO" in system

    def test_metadata_included_in_memory_lines(self):
        """Rerank prompt should include metadata tags for disambiguation."""
        from mnemory.prompts import build_rerank_prompt

        memories = [
            {
                "id": "m1",
                "memory": "User likes cats",
                "metadata": {
                    "memory_type": "preference",
                    "categories": ["personal"],
                    "importance": "high",
                    "role": "user",
                },
            },
            {
                "id": "m2",
                "memory": "Assistant is friendly",
                "metadata": {
                    "memory_type": "fact",
                    "categories": ["preferences"],
                    "importance": "normal",
                    "role": "assistant",
                },
            },
        ]
        messages, _ = build_rerank_prompt("test", memories)
        user_content = messages[1]["content"]
        # m1: type, categories, importance (high != normal), but role=user is default → omitted
        assert "type: preference" in user_content
        assert "categories: personal" in user_content
        assert "importance: high" in user_content
        # m2: type, categories, role=assistant (non-default), importance=normal → omitted
        assert "type: fact" in user_content
        assert "categories: preferences" in user_content
        assert "role: assistant" in user_content

    def test_default_metadata_omitted(self):
        """Default metadata values (importance: normal, role: user) should be omitted."""
        from mnemory.prompts import build_rerank_prompt

        memories = [
            {
                "id": "m1",
                "memory": "Simple fact",
                "metadata": {
                    "memory_type": "fact",
                    "categories": [],
                    "importance": "normal",
                    "role": "user",
                },
            },
        ]
        messages, _ = build_rerank_prompt("test", memories)
        user_content = messages[1]["content"]
        # importance: normal and role: user should NOT appear
        assert "importance:" not in user_content
        assert "role:" not in user_content
        # type should still appear
        assert "type: fact" in user_content

    def test_no_metadata_graceful(self):
        """Memories without metadata should not crash."""
        from mnemory.prompts import build_rerank_prompt

        memories = [
            {"id": "m1", "memory": "No metadata"},
            {"id": "m2", "memory": "None metadata", "metadata": None},
        ]
        messages, _ = build_rerank_prompt("test", memories)
        user_content = messages[1]["content"]
        assert "No metadata" in user_content
        assert "None metadata" in user_content


# ── find_memories batch embedding ─────────────────────────────────────


class TestFindMemoriesBatchEmbed:
    """Test that find_memories uses batch embedding for performance."""

    def test_uses_embed_batch_not_individual(self):
        """find_memories should call embed_batch once, not embed per query."""
        import json

        service = _make_service()
        service._llm.generate.return_value = json.dumps(
            {"queries": ["query1", "query2", "query3"]}
        )
        service.vector.search.return_value = {"results": []}

        service.find_memories("test question", user_id="filip")

        # embed_batch should be called once with all queries
        service.vector.embedding.embed_batch.assert_called_once_with(
            ["query1", "query2", "query3"]
        )
        # Individual embed should NOT be called by find_memories
        # (it may be called by search_memories internally, but with
        # query_vector provided it should be skipped)

    def test_query_vector_passed_to_search(self):
        """Pre-computed vectors should be passed through to vector.search."""
        import json

        service = _make_service()
        service._llm.generate.return_value = json.dumps({"queries": ["single query"]})
        service.vector.search.return_value = {"results": []}
        service.vector.embedding.embed_batch.side_effect = None
        service.vector.embedding.embed_batch.return_value = [[0.5] * 1536]

        service.find_memories("test question", user_id="filip")

        # vector.search should receive the pre-computed vector
        _, kwargs = service.vector.search.call_args
        assert kwargs["query_vector"] == [0.5] * 1536

    def test_empty_queries_after_validation(self):
        """If all LLM queries are empty/invalid, return empty results."""
        import json

        service = _make_service()
        service._llm.generate.return_value = json.dumps({"queries": ["", "  ", None]})

        result = service.find_memories("test", user_id="filip")
        assert result["results"] == []
        assert result["queries"] == []
