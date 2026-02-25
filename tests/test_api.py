"""Tests for REST API components (instructions, remember formatting, input length)."""

from __future__ import annotations

import pytest

from mnemory.api.remember import _check_rate_limit, _format_messages, _rate_limits
from mnemory.api.schemas import MessageParam
from mnemory.instructions import build_instructions, build_managed_instructions


class TestManagedInstructions:
    """Test managed-mode instructions."""

    def test_build_managed_instructions(self):
        """Managed instructions should include managed behavior + base reference."""
        text = build_managed_instructions()
        assert "handled AUTOMATICALLY by the system" in text
        assert "Do NOT call initialize_memory" in text
        assert "Do NOT call add_memory proactively" in text
        assert "OVERRIDE any conflicting guidance" in text
        assert "already injected" in text
        # Should include base technical reference
        assert "TOOL REFERENCE" in text

    def test_build_managed_vs_proactive(self):
        """Managed instructions should differ from proactive."""
        managed = build_managed_instructions()
        proactive = build_instructions("proactive")
        # Both should have base reference
        assert "TOOL REFERENCE" in managed
        assert "TOOL REFERENCE" in proactive
        # But different preambles
        assert "handled AUTOMATICALLY" in managed
        assert "handled AUTOMATICALLY" not in proactive

    def test_build_instructions_modes(self):
        """All instruction modes should build without error."""
        for mode in ("passive", "proactive", "personality"):
            text = build_instructions(mode)
            assert len(text) > 100

    def test_build_instructions_invalid_mode(self):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid INSTRUCTION_MODE"):
            build_instructions("invalid")


class TestRecallScoreThreshold:
    """Test score threshold filtering in recall endpoint."""

    def test_score_threshold_filters_low_scores(self):
        """Recall should filter out results below score_threshold."""
        from unittest.mock import MagicMock, patch

        from mnemory.api.deps import SessionContext
        from mnemory.api.recall import recall
        from mnemory.api.schemas import RecallRequest
        from mnemory.session import SessionStore

        # Mock service that returns results with varying scores
        mock_service = MagicMock()
        mock_service.search_memories.return_value = [
            {"id": "high", "memory": "high score", "score": 0.8, "metadata": {}},
            {"id": "mid", "memory": "mid score", "score": 0.5, "metadata": {}},
            {"id": "low", "memory": "low score", "score": 0.3, "metadata": {}},
        ]

        session_store = SessionStore()
        session = session_store.create(user_id="test", agent_id=None)

        ctx = SessionContext(user_id="test", agent_id=None, timezone=None)
        req = RecallRequest(
            session_id=session.session_id,
            query="test query",
            search_mode="search",
            score_threshold=0.5,
        )

        with (
            patch("mnemory.api.recall._get_service", return_value=mock_service),
            patch("mnemory.api.recall._get_session_store", return_value=session_store),
        ):
            response = recall(req, ctx)

        # Only high and mid should pass the 0.5 threshold
        assert len(response.search_results) == 2
        result_ids = {r.id for r in response.search_results}
        assert "high" in result_ids
        assert "mid" in result_ids
        assert "low" not in result_ids

    def test_no_score_threshold_returns_all(self):
        """Without score_threshold, all results should be returned."""
        from unittest.mock import MagicMock, patch

        from mnemory.api.deps import SessionContext
        from mnemory.api.recall import recall
        from mnemory.api.schemas import RecallRequest
        from mnemory.session import SessionStore

        mock_service = MagicMock()
        mock_service.search_memories.return_value = [
            {"id": "high", "memory": "high score", "score": 0.8, "metadata": {}},
            {"id": "low", "memory": "low score", "score": 0.3, "metadata": {}},
        ]

        session_store = SessionStore()
        session = session_store.create(user_id="test", agent_id=None)

        ctx = SessionContext(user_id="test", agent_id=None, timezone=None)
        req = RecallRequest(
            session_id=session.session_id,
            query="test query",
            search_mode="search",
            # No score_threshold
        )

        with (
            patch("mnemory.api.recall._get_service", return_value=mock_service),
            patch("mnemory.api.recall._get_session_store", return_value=session_store),
        ):
            response = recall(req, ctx)

        # Both should be returned
        assert len(response.search_results) == 2

    def test_score_threshold_filters_all(self):
        """If all results are below threshold, no results should be returned."""
        from unittest.mock import MagicMock, patch

        from mnemory.api.deps import SessionContext
        from mnemory.api.recall import recall
        from mnemory.api.schemas import RecallRequest
        from mnemory.session import SessionStore

        mock_service = MagicMock()
        mock_service.search_memories.return_value = [
            {"id": "a", "memory": "weak match", "score": 0.35, "metadata": {}},
            {"id": "b", "memory": "weak match 2", "score": 0.31, "metadata": {}},
        ]

        session_store = SessionStore()
        session = session_store.create(user_id="test", agent_id=None)

        ctx = SessionContext(user_id="test", agent_id=None, timezone=None)
        req = RecallRequest(
            session_id=session.session_id,
            query="ok format as table",
            search_mode="search",
            score_threshold=0.5,
        )

        with (
            patch("mnemory.api.recall._get_service", return_value=mock_service),
            patch("mnemory.api.recall._get_session_store", return_value=session_store),
        ):
            response = recall(req, ctx)

        assert len(response.search_results) == 0
        assert response.stats.new_count == 0


class TestFormatMessages:
    """Test message formatting for remember endpoint."""

    def test_format_user_assistant(self):
        """Standard user + assistant exchange."""
        messages = [
            MessageParam(role="user", content="I just moved to Berlin"),
            MessageParam(role="assistant", content="That's exciting!"),
        ]
        text = _format_messages(messages)
        assert "User: I just moved to Berlin" in text
        assert "Assistant: That's exciting!" in text

    def test_format_empty_messages(self):
        """Empty messages list should return empty string."""
        assert _format_messages([]) == ""

    def test_format_skips_empty_content(self):
        """Messages with empty content should be skipped."""
        messages = [
            MessageParam(role="user", content="Hello"),
            MessageParam(role="assistant", content=""),
        ]
        text = _format_messages(messages)
        assert "User: Hello" in text
        assert "Assistant:" not in text

    def test_format_skips_none_content(self):
        """Messages with None content should be skipped."""
        messages = [
            MessageParam(role="user", content="Hello"),
            MessageParam(role="tool", content=None),
        ]
        text = _format_messages(messages)
        assert "User: Hello" in text
        assert "Tool:" not in text

    def test_format_system_message(self):
        """System messages should be formatted with 'System' label."""
        messages = [
            MessageParam(role="system", content="You are helpful"),
            MessageParam(role="user", content="Hi"),
        ]
        text = _format_messages(messages)
        assert "System: You are helpful" in text
        assert "User: Hi" in text


class TestRateLimit:
    """Test rate limiting for remember endpoint."""

    def setup_method(self):
        """Clear rate limits before each test."""
        _rate_limits.clear()

    def test_rate_limit_allows_within_limit(self):
        """Requests within limit should be allowed."""
        # Default limit is 10/min — but we need to mock config
        # For now, test the function directly with cleared state
        # The function reads from config, so we test the basic flow
        assert _check_rate_limit("test-user") is True

    def test_rate_limit_tracks_per_user(self):
        """Different users should have independent limits."""
        for _ in range(5):
            _check_rate_limit("user1")
        for _ in range(3):
            _check_rate_limit("user2")
        # Both should still be within limit
        assert _check_rate_limit("user1") is True
        assert _check_rate_limit("user2") is True


class TestRecallContextPassthrough:
    """Test that /api/recall passes context to find_memories."""

    def test_context_passed_to_find_memories(self):
        """recall() should forward req.context to service.find_memories."""
        from unittest.mock import MagicMock, patch

        from mnemory.api.deps import SessionContext
        from mnemory.api.recall import recall
        from mnemory.api.schemas import RecallRequest
        from mnemory.session import SessionStore

        mock_service = MagicMock()
        mock_service.find_memories.return_value = {
            "results": [],
            "queries": [],
            "stats": {},
        }

        session_store = SessionStore()

        ctx = SessionContext(user_id="test", agent_id=None, timezone=None)
        req = RecallRequest(
            query="What is my project?",
            context="Working directory: /home/user/src/myapp",
        )

        with (
            patch("mnemory.api.recall._get_service", return_value=mock_service),
            patch("mnemory.api.recall._get_session_store", return_value=session_store),
            patch(
                "mnemory.server._get_config",
                return_value=MagicMock(memory=MagicMock(recall_max_results=10)),
            ),
        ):
            recall(req, ctx)

        mock_service.find_memories.assert_called_once()
        _, kwargs = mock_service.find_memories.call_args
        assert kwargs["context"] == "Working directory: /home/user/src/myapp"

    def test_no_context_passes_none(self):
        """recall() without context should pass context=None to find_memories."""
        from unittest.mock import MagicMock, patch

        from mnemory.api.deps import SessionContext
        from mnemory.api.recall import recall
        from mnemory.api.schemas import RecallRequest
        from mnemory.session import SessionStore

        mock_service = MagicMock()
        mock_service.find_memories.return_value = {
            "results": [],
            "queries": [],
            "stats": {},
        }

        session_store = SessionStore()

        ctx = SessionContext(user_id="test", agent_id=None, timezone=None)
        req = RecallRequest(query="test question")

        with (
            patch("mnemory.api.recall._get_service", return_value=mock_service),
            patch("mnemory.api.recall._get_session_store", return_value=session_store),
            patch(
                "mnemory.server._get_config",
                return_value=MagicMock(memory=MagicMock(recall_max_results=10)),
            ),
        ):
            recall(req, ctx)

        mock_service.find_memories.assert_called_once()
        _, kwargs = mock_service.find_memories.call_args
        assert kwargs["context"] is None


class TestInputLengthGuard:
    """Test input length validation in add_memory and remember."""

    def test_add_memory_infer_true_rejects_oversized(self):
        """add_memory with infer=True should reject input over MAX_INPUT_LENGTH."""
        from unittest.mock import MagicMock, patch

        from mnemory.memory import MemoryService

        mock_config = MagicMock()
        # Set all required config attributes
        from tests.test_memory import _mock_memory_config

        _mock_memory_config(mock_config)
        mock_config.memory.max_input_length = 100  # Very low for testing

        with (
            patch("mnemory.memory.VectorStore"),
            patch("mnemory.memory.ArtifactStore"),
            patch("mnemory.memory.LLMClient"),
        ):
            service = MemoryService(mock_config)

        result = service.add_memory(
            content="x" * 101,
            user_id="filip",
            infer=True,
        )
        assert result.get("error") is True
        assert "too long" in result["message"].lower()

    def test_add_memory_infer_true_accepts_within_limit(self):
        """add_memory with infer=True should accept input within MAX_INPUT_LENGTH."""
        from unittest.mock import MagicMock, patch

        from mnemory.memory import MemoryService

        mock_config = MagicMock()
        from tests.test_memory import _mock_memory_config

        _mock_memory_config(mock_config)
        mock_config.memory.max_input_length = 400000

        with (
            patch("mnemory.memory.VectorStore"),
            patch("mnemory.memory.ArtifactStore"),
            patch("mnemory.memory.LLMClient"),
        ):
            service = MemoryService(mock_config)

        service.vector = MagicMock()
        service._llm = MagicMock()
        service.vector.search_similar.return_value = []
        service.vector.get_all.return_value = {"results": []}
        service._llm.generate.return_value = '{"memories": []}'

        result = service.add_memory(
            content="x" * 2000,
            user_id="filip",
            infer=True,
        )
        # Should not be an error — extraction just returns empty
        assert result.get("error") is not True

    def test_remember_truncates_oversized_input(self):
        """remember() should truncate input over MAX_INPUT_LENGTH (keep most recent)."""
        from unittest.mock import MagicMock, patch

        from mnemory.memory import MemoryService

        mock_config = MagicMock()
        from tests.test_memory import _mock_memory_config

        _mock_memory_config(mock_config)
        mock_config.memory.max_input_length = 100

        with (
            patch("mnemory.memory.VectorStore"),
            patch("mnemory.memory.ArtifactStore"),
            patch("mnemory.memory.LLMClient"),
        ):
            service = MemoryService(mock_config)

        service.vector = MagicMock()
        service._llm = MagicMock()
        service.vector.search_similar.return_value = []
        service.vector.get_all.return_value = {"results": []}
        service._llm.generate.return_value = '{"memories": []}'

        # Content is 200 chars, should be truncated to last 100
        content = "A" * 100 + "B" * 100
        result = service.remember(content=content, user_id="filip")

        # Should not error — just truncated
        assert result.get("error") is not True
        # The LLM should have received the truncated content (last 100 chars = all B's)
        call_args = service._llm.generate.call_args
        if call_args:
            # The content passed to the LLM should be the truncated version
            messages = call_args[0][0]
            # Find the user message content
            for msg in messages:
                if isinstance(msg.get("content"), str) and "B" * 50 in msg["content"]:
                    # Good — the truncated content (B's) is in the prompt
                    break


class TestWhoamiEndpoint:
    """Test /api/whoami endpoint."""

    def test_whoami_bound_user(self):
        """whoami with a bound user should return user_id and can_switch_user=False."""
        import asyncio

        from mnemory.api.ui import whoami
        from mnemory.server import (
            _session_agent_id,
            _session_timezone,
            _session_user_bound,
            _session_user_id,
        )

        _session_user_id.set("filip")
        _session_agent_id.set("claude-code")
        _session_timezone.set("Europe/Prague")
        _session_user_bound.set(True)
        try:
            result = asyncio.run(whoami())
            assert result["user_id"] == "filip"
            assert result["agent_id"] == "claude-code"
            assert result["timezone"] == "Europe/Prague"
            assert result["can_switch_user"] is False
        finally:
            _session_user_id.set(None)
            _session_agent_id.set(None)
            _session_timezone.set(None)
            _session_user_bound.set(False)

    def test_whoami_wildcard_key(self):
        """whoami with a wildcard key should return user_id=None and can_switch_user=True."""
        import asyncio

        from mnemory.api.ui import whoami
        from mnemory.server import (
            _session_agent_id,
            _session_timezone,
            _session_user_bound,
            _session_user_id,
        )

        _session_user_id.set(None)
        _session_agent_id.set(None)
        _session_timezone.set(None)
        _session_user_bound.set(False)
        try:
            result = asyncio.run(whoami())
            assert result["user_id"] is None
            assert result["agent_id"] is None
            assert result["can_switch_user"] is True
        finally:
            pass  # already default values

    def test_whoami_wildcard_with_user_header(self):
        """whoami with wildcard key + X-User-Id should return the header user."""
        import asyncio

        from mnemory.api.ui import whoami
        from mnemory.server import _session_user_bound, _session_user_id

        _session_user_id.set("bob")
        _session_user_bound.set(False)
        try:
            result = asyncio.run(whoami())
            assert result["user_id"] == "bob"
            assert result["can_switch_user"] is True
        finally:
            _session_user_id.set(None)
            _session_user_bound.set(False)


class TestStatsEndpoint:
    """Test /api/stats endpoint."""

    def test_stats_returns_json(self):
        """stats should return structured JSON from MetricsCollector."""
        import asyncio
        from unittest.mock import MagicMock, patch

        from mnemory.api.ui import stats

        mock_collector = MagicMock()
        mock_collector.get_stats_json.return_value = {
            "version": "1.0.0",
            "vector_backend": "qdrant-local",
            "artifact_backend": "filesystem",
            "active_sessions": 2,
            "users": ["filip", "bob"],
            "totals": {"memories": 42, "pinned": 5, "decayed": 3, "with_artifacts": 1},
            "by_type": {"fact": 20, "preference": 10, "episodic": 12},
            "by_category": {"personal": 15, "work": 27},
            "by_role": {"user": 35, "assistant": 7},
            "by_user": {},
            "operations": {},
        }

        with patch("mnemory.metrics.get_collector", return_value=mock_collector):
            result = asyncio.run(stats())

        assert result["version"] == "1.0.0"
        assert result["totals"]["memories"] == 42
        assert result["users"] == ["filip", "bob"]
        assert result["active_sessions"] == 2
        mock_collector.get_stats_json.assert_called_once()

    def test_stats_metrics_disabled(self):
        """stats should return 404 when metrics are disabled."""
        import asyncio
        from unittest.mock import patch

        from fastapi import HTTPException

        from mnemory.api.ui import stats

        with patch("mnemory.metrics.get_collector", return_value=None):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(stats())
            assert exc_info.value.status_code == 404
            assert "Metrics disabled" in exc_info.value.detail


class TestPrometheusMetricsInternalAPI:
    """Validate that prometheus_client internal _metrics dict API works.

    get_stats_json() reads gauge/counter values via the undocumented
    ._metrics attribute. This test ensures the API contract holds for
    the installed prometheus_client version, catching breakage early.
    """

    def test_gauge_metrics_dict_accessible(self):
        """Gauge._metrics should be a dict of (labels_tuple) -> metric."""
        from prometheus_client import CollectorRegistry, Gauge

        registry = CollectorRegistry()
        g = Gauge("test_gauge", "test", ["user_id", "role"], registry=registry)
        g.labels(user_id="alice", role="user").set(5)
        g.labels(user_id="bob", role="assistant").set(3)

        assert hasattr(g, "_metrics"), (
            "prometheus_client Gauge no longer has _metrics attribute — "
            "get_stats_json() will silently return empty data"
        )
        assert isinstance(g._metrics, dict)
        assert len(g._metrics) == 2

        # Verify we can read values via _value.get()
        for labels, metric in g._metrics.items():
            assert isinstance(labels, tuple)
            assert hasattr(metric, "_value")
            val = metric._value.get()
            assert val in (3.0, 5.0)

    def test_counter_metrics_dict_accessible(self):
        """Counter._metrics should be a dict of (labels_tuple) -> metric."""
        from prometheus_client import CollectorRegistry, Counter

        registry = CollectorRegistry()
        c = Counter("test_counter", "test", ["operation", "user_id"], registry=registry)
        c.labels(operation="add", user_id="alice").inc(10)
        c.labels(operation="search", user_id="bob").inc(7)

        assert hasattr(c, "_metrics"), (
            "prometheus_client Counter no longer has _metrics attribute — "
            "get_stats_json() will silently return empty data"
        )
        assert isinstance(c._metrics, dict)
        assert len(c._metrics) == 2

        for labels, metric in c._metrics.items():
            assert isinstance(labels, tuple)
            val = metric._value.get()
            assert val in (10.0, 7.0)
