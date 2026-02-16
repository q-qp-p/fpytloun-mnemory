"""Tests for agent_id scoping: protection, dual-scope, and ownership verification."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mnemory.memory import MemoryService

# ── _resolve_agent_id ──────────────────────────────────────────────────


class TestResolveAgentId:
    """Test the _resolve_agent_id function from server.py."""

    def _resolve(self, param_agent_id, session_agent_id=None):
        """Call _resolve_agent_id with a mocked session context."""
        from mnemory.server import _resolve_agent_id, _session_agent_id

        token = _session_agent_id.set(session_agent_id)
        try:
            return _resolve_agent_id(param_agent_id)
        finally:
            _session_agent_id.reset(token)

    def test_no_session_no_param(self):
        assert self._resolve(None, session_agent_id=None) is None

    def test_no_session_with_param(self):
        assert self._resolve("bob", session_agent_id=None) == "bob"

    def test_session_no_param_returns_none(self):
        """LLM omits agent_id → shared memory (None)."""
        assert self._resolve(None, session_agent_id="openwebui") is None

    def test_session_same_param(self):
        """LLM passes same agent_id as session → allowed."""
        assert self._resolve("openwebui", session_agent_id="openwebui") == "openwebui"

    def test_session_different_param_raises(self):
        """LLM passes different agent_id → blocked."""
        with pytest.raises(ValueError, match="Cannot use agent_id"):
            self._resolve("other-agent", session_agent_id="openwebui")


# ── _resolve_agent_id_for_core ─────────────────────────────────────────


class TestResolveAgentIdForCore:
    """Test the _resolve_agent_id_for_core function from server.py."""

    def _resolve(self, param_agent_id, session_agent_id=None):
        from mnemory.server import _resolve_agent_id_for_core, _session_agent_id

        token = _session_agent_id.set(session_agent_id)
        try:
            return _resolve_agent_id_for_core(param_agent_id)
        finally:
            _session_agent_id.reset(token)

    def test_no_session_no_param(self):
        assert self._resolve(None, session_agent_id=None) is None

    def test_no_session_with_param(self):
        assert self._resolve("bob", session_agent_id=None) == "bob"

    def test_session_no_param_auto_injects(self):
        """Unlike _resolve_agent_id, this auto-injects session agent."""
        assert self._resolve(None, session_agent_id="openwebui") == "openwebui"

    def test_session_same_param(self):
        assert self._resolve("openwebui", session_agent_id="openwebui") == "openwebui"

    def test_session_different_param_raises(self):
        with pytest.raises(ValueError, match="Cannot use agent_id"):
            self._resolve("other-agent", session_agent_id="openwebui")


# ── _get_session_agent_id ──────────────────────────────────────────────


class TestGetSessionAgentId:
    def test_returns_none_by_default(self):
        from mnemory.server import _get_session_agent_id, _session_agent_id

        token = _session_agent_id.set(None)
        try:
            assert _get_session_agent_id() is None
        finally:
            _session_agent_id.reset(token)

    def test_returns_session_value(self):
        from mnemory.server import _get_session_agent_id, _session_agent_id

        token = _session_agent_id.set("claude-code")
        try:
            assert _get_session_agent_id() == "claude-code"
        finally:
            _session_agent_id.reset(token)


# ── verify_memory_access ──────────────────────────────────────────────


class TestVerifyMemoryAccess:
    """Test ownership verification in MemoryService."""

    @staticmethod
    def _make_service(get_by_id_return=None):
        """Create a MemoryService with a mocked vector store."""
        service = object.__new__(MemoryService)
        service.vector = MagicMock()
        service.vector.get_by_id.return_value = get_by_id_return
        service.artifact = MagicMock()
        return service

    def test_no_session_agent_allows_all(self):
        """No session agent → no protection."""
        service = self._make_service()
        # Should not raise, should not even call get_by_id
        service.verify_memory_access("mem-1", session_agent_id=None)
        service.vector.get_by_id.assert_not_called()

    def test_memory_not_found_passes(self):
        """Memory not found → let downstream handle 404."""
        service = self._make_service(get_by_id_return=None)
        service.verify_memory_access("mem-1", session_agent_id="openwebui")
        # No error raised

    def test_shared_memory_allowed(self):
        """Shared memory (no agent_id) → accessible by any agent."""
        mem = {"id": "mem-1", "memory": "test"}
        service = self._make_service(get_by_id_return=mem)
        service.verify_memory_access("mem-1", session_agent_id="openwebui")

    def test_own_agent_memory_allowed(self):
        """Own agent's memory → accessible."""
        mem = {"id": "mem-1", "memory": "test", "agent_id": "openwebui"}
        service = self._make_service(get_by_id_return=mem)
        service.verify_memory_access("mem-1", session_agent_id="openwebui")

    def test_other_agent_memory_blocked(self):
        """Other agent's memory → blocked."""
        mem = {"id": "mem-1", "memory": "test", "agent_id": "claude-code"}
        service = self._make_service(get_by_id_return=mem)
        with pytest.raises(ValueError, match="Cannot access memory"):
            service.verify_memory_access("mem-1", session_agent_id="openwebui")

    def test_agent_id_none_in_memory_allowed(self):
        """Memory with agent_id=None (explicit) → shared, accessible."""
        mem = {"id": "mem-1", "memory": "test", "agent_id": None}
        service = self._make_service(get_by_id_return=mem)
        service.verify_memory_access("mem-1", session_agent_id="openwebui")


# ── Dual-scope search ─────────────────────────────────────────────────


class TestSearchMemoriesDualScope:
    """Test dual-scope search merging and deduplication."""

    @staticmethod
    def _make_service():
        service = object.__new__(MemoryService)
        service.vector = MagicMock()
        service.artifact = MagicMock()
        service._config = MagicMock()
        return service

    def test_merges_agent_and_shared(self):
        service = self._make_service()
        service.vector.search.side_effect = [
            # Agent-scoped results
            {
                "results": [
                    {"id": "a1", "score": 0.9, "metadata": {"importance": "normal"}},
                ]
            },
            # Shared results
            {
                "results": [
                    {"id": "s1", "score": 0.8, "metadata": {"importance": "normal"}},
                ]
            },
        ]

        results = service.search_memories_dual_scope(
            "test query",
            user_id="filip",
            session_agent_id="openwebui",
        )
        assert len(results) == 2
        ids = {r["id"] for r in results}
        assert ids == {"a1", "s1"}

    def test_deduplicates_by_id(self):
        service = self._make_service()
        service.vector.search.side_effect = [
            {
                "results": [
                    {"id": "dup1", "score": 0.9, "metadata": {"importance": "normal"}},
                ]
            },
            {
                "results": [
                    {"id": "dup1", "score": 0.85, "metadata": {"importance": "normal"}},
                    {"id": "s1", "score": 0.7, "metadata": {"importance": "normal"}},
                ]
            },
        ]

        results = service.search_memories_dual_scope(
            "test query",
            user_id="filip",
            session_agent_id="openwebui",
        )
        assert len(results) == 2
        ids = [r["id"] for r in results]
        assert "dup1" in ids
        assert "s1" in ids

    def test_respects_limit(self):
        service = self._make_service()
        service.vector.search.side_effect = [
            {
                "results": [
                    {
                        "id": f"a{i}",
                        "score": 0.9 - i * 0.01,
                        "metadata": {"importance": "normal"},
                    }
                    for i in range(5)
                ]
            },
            {
                "results": [
                    {
                        "id": f"s{i}",
                        "score": 0.85 - i * 0.01,
                        "metadata": {"importance": "normal"},
                    }
                    for i in range(5)
                ]
            },
        ]

        results = service.search_memories_dual_scope(
            "test query",
            user_id="filip",
            session_agent_id="openwebui",
            limit=3,
        )
        assert len(results) == 3

    def test_category_filter_applied(self):
        service = self._make_service()
        service.vector.search.side_effect = [
            {
                "results": [
                    {
                        "id": "a1",
                        "score": 0.9,
                        "metadata": {"importance": "normal", "categories": ["work"]},
                    },
                ]
            },
            {
                "results": [
                    {
                        "id": "s1",
                        "score": 0.8,
                        "metadata": {
                            "importance": "normal",
                            "categories": ["personal"],
                        },
                    },
                    {
                        "id": "s2",
                        "score": 0.7,
                        "metadata": {"importance": "normal", "categories": ["work"]},
                    },
                ]
            },
        ]

        results = service.search_memories_dual_scope(
            "test query",
            user_id="filip",
            session_agent_id="openwebui",
            categories=["work"],
        )
        ids = {r["id"] for r in results}
        assert ids == {"a1", "s2"}


# ── Dual-scope list ───────────────────────────────────────────────────


class TestListMemoriesDualScope:
    """Test dual-scope list merging and deduplication."""

    @staticmethod
    def _make_service():
        service = object.__new__(MemoryService)
        service.vector = MagicMock()
        service.artifact = MagicMock()
        service._config = MagicMock()
        return service

    def test_merges_agent_and_shared(self):
        service = self._make_service()
        service.vector.get_all.side_effect = [
            {"results": [{"id": "a1", "metadata": {}}]},
            {"results": [{"id": "s1", "metadata": {}}]},
        ]

        results = service.list_memories_dual_scope(
            user_id="filip",
            session_agent_id="openwebui",
        )
        assert len(results) == 2

    def test_deduplicates_by_id(self):
        service = self._make_service()
        service.vector.get_all.side_effect = [
            {"results": [{"id": "dup1", "metadata": {}}]},
            {"results": [{"id": "dup1", "metadata": {}}, {"id": "s1", "metadata": {}}]},
        ]

        results = service.list_memories_dual_scope(
            user_id="filip",
            session_agent_id="openwebui",
        )
        assert len(results) == 2
        ids = {r["id"] for r in results}
        assert ids == {"dup1", "s1"}

    def test_respects_limit(self):
        service = self._make_service()
        service.vector.get_all.side_effect = [
            {"results": [{"id": f"a{i}", "metadata": {}} for i in range(5)]},
            {"results": [{"id": f"s{i}", "metadata": {}} for i in range(5)]},
        ]

        results = service.list_memories_dual_scope(
            user_id="filip",
            session_agent_id="openwebui",
            limit=3,
        )
        assert len(results) == 3


# ── User identity header priority ─────────────────────────────────────


class TestUserIdentityHeaders:
    """Test the header priority chain for user_id resolution in the middleware."""

    @staticmethod
    def _make_request(headers: dict[str, str]):
        """Create a mock Starlette Request with the given headers."""
        mock_request = MagicMock()
        # Starlette normalizes header names to lowercase; use a real dict
        # so .get() works naturally (dict.get matches the Starlette API).
        mock_request.headers = {k.lower(): v for k, v in headers.items()}
        return mock_request

    def test_x_user_id_takes_priority(self):
        """X-User-Id should be used when both headers are present."""
        from mnemory.server import APIKeyMiddleware, _session_user_id

        middleware = object.__new__(APIKeyMiddleware)
        request = self._make_request(
            {
                "X-User-Id": "filip",
                "X-OpenWebUI-User-Email": "filip@example.com",
            }
        )

        token = _session_user_id.set(None)
        try:
            middleware._set_identity_from_headers(request)
            assert _session_user_id.get() == "filip"
        finally:
            _session_user_id.reset(token)

    def test_openwebui_email_fallback(self):
        """X-OpenWebUI-User-Email should be used when X-User-Id is absent."""
        from mnemory.server import APIKeyMiddleware, _session_user_id

        middleware = object.__new__(APIKeyMiddleware)
        request = self._make_request({"X-OpenWebUI-User-Email": "filip@example.com"})

        token = _session_user_id.set(None)
        try:
            middleware._set_identity_from_headers(request)
            assert _session_user_id.get() == "filip@example.com"
        finally:
            _session_user_id.reset(token)

    def test_empty_x_user_id_falls_through(self):
        """Empty X-User-Id should fall through to X-OpenWebUI-User-Email."""
        from mnemory.server import APIKeyMiddleware, _session_user_id

        middleware = object.__new__(APIKeyMiddleware)
        request = self._make_request(
            {
                "X-User-Id": "  ",
                "X-OpenWebUI-User-Email": "filip@example.com",
            }
        )

        token = _session_user_id.set(None)
        try:
            middleware._set_identity_from_headers(request)
            assert _session_user_id.get() == "filip@example.com"
        finally:
            _session_user_id.reset(token)

    def test_no_identity_headers(self):
        """No identity headers → session user_id stays None."""
        from mnemory.server import APIKeyMiddleware, _session_user_id

        middleware = object.__new__(APIKeyMiddleware)
        request = self._make_request({})

        token = _session_user_id.set(None)
        try:
            middleware._set_identity_from_headers(request)
            assert _session_user_id.get() is None
        finally:
            _session_user_id.reset(token)

    def test_agent_id_still_read(self):
        """X-Agent-Id should still be read alongside email fallback."""
        from mnemory.server import (
            APIKeyMiddleware,
            _session_agent_id,
            _session_user_id,
        )

        middleware = object.__new__(APIKeyMiddleware)
        request = self._make_request(
            {
                "X-OpenWebUI-User-Email": "filip@example.com",
                "X-Agent-Id": "open-webui",
            }
        )

        uid_token = _session_user_id.set(None)
        aid_token = _session_agent_id.set(None)
        try:
            middleware._set_identity_from_headers(request)
            assert _session_user_id.get() == "filip@example.com"
            assert _session_agent_id.get() == "open-webui"
        finally:
            _session_user_id.reset(uid_token)
            _session_agent_id.reset(aid_token)
