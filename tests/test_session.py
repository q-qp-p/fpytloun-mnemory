"""Tests for MemorySession and SessionStore."""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from mnemory.session import MemorySession, SessionStore
from mnemory.storage.session import (
    MemoryBackend,
    SQLiteBackend,
    _mask_url,
    create_session_backend,
    deserialize_session,
    serialize_session,
)


class TestMemorySession:
    """Test MemorySession dataclass behavior."""

    def test_create_session(self):
        """Session should be created with defaults."""
        session = MemorySession(
            session_id="test-1",
            user_id="filip",
            agent_id=None,
        )
        assert session.session_id == "test-1"
        assert session.user_id == "filip"
        assert session.agent_id is None
        assert session.known_memory_ids == set()
        assert session.ttl_seconds == 3600
        assert not session.is_expired

    def test_session_with_agent(self):
        """Session should store agent_id."""
        session = MemorySession(
            session_id="test-2",
            user_id="filip",
            agent_id="claude-code",
        )
        assert session.agent_id == "claude-code"

    def test_session_expiry(self):
        """Session should be expired after TTL."""
        session = MemorySession(
            session_id="test-3",
            user_id="filip",
            agent_id=None,
            ttl_seconds=1,
            last_accessed=datetime.now(timezone.utc) - timedelta(seconds=2),
        )
        assert session.is_expired

    def test_session_not_expired(self):
        """Session should not be expired within TTL."""
        session = MemorySession(
            session_id="test-4",
            user_id="filip",
            agent_id=None,
            ttl_seconds=3600,
        )
        assert not session.is_expired

    def test_known_memory_ids(self):
        """Session should track known memory IDs."""
        session = MemorySession(
            session_id="test-5",
            user_id="filip",
            agent_id=None,
        )
        session.known_memory_ids.add("mem-1")
        session.known_memory_ids.add("mem-2")
        assert "mem-1" in session.known_memory_ids
        assert "mem-2" in session.known_memory_ids
        assert len(session.known_memory_ids) == 2


class TestSessionStore:
    """Test SessionStore operations."""

    def test_create_session(self):
        """Store should create a session with unique ID."""
        store = SessionStore(default_ttl=3600)
        session = store.create(user_id="filip")
        assert session.session_id
        assert session.user_id == "filip"
        assert session.ttl_seconds == 3600

    def test_create_with_custom_ttl(self):
        """Store should respect custom TTL."""
        store = SessionStore(default_ttl=3600)
        session = store.create(user_id="filip", ttl_seconds=60)
        assert session.ttl_seconds == 60

    def test_create_with_agent(self):
        """Store should store agent_id."""
        store = SessionStore()
        session = store.create(user_id="filip", agent_id="claude-code")
        assert session.agent_id == "claude-code"

    def test_get_session(self):
        """Store should retrieve session by ID."""
        store = SessionStore()
        created = store.create(user_id="filip")
        retrieved = store.get(created.session_id)
        assert retrieved is not None
        assert retrieved.session_id == created.session_id

    def test_get_nonexistent_session(self):
        """Store should return None for unknown session ID."""
        store = SessionStore()
        assert store.get("nonexistent") is None

    def test_get_expired_session(self):
        """Store should return None and remove expired sessions."""
        store = SessionStore(default_ttl=1)
        session = store.create(user_id="filip")
        # Manually expire it
        session.last_accessed = datetime.now(timezone.utc) - timedelta(seconds=2)
        assert store.get(session.session_id) is None

    def test_touch_updates_last_accessed(self):
        """Touch should update last_accessed timestamp."""
        store = SessionStore()
        session = store.create(user_id="filip")
        old_accessed = session.last_accessed
        time.sleep(0.01)
        store.touch(session.session_id)
        retrieved = store.get(session.session_id)
        assert retrieved is not None
        assert retrieved.last_accessed >= old_accessed

    def test_add_known_ids(self):
        """Store should add memory IDs to session."""
        store = SessionStore()
        session = store.create(user_id="filip")
        store.add_known_ids(session.session_id, {"mem-1", "mem-2"})
        known = store.get_known_ids(session.session_id)
        assert known == {"mem-1", "mem-2"}

    def test_add_known_ids_accumulates(self):
        """Multiple add_known_ids calls should accumulate."""
        store = SessionStore()
        session = store.create(user_id="filip")
        store.add_known_ids(session.session_id, {"mem-1"})
        store.add_known_ids(session.session_id, {"mem-2", "mem-3"})
        known = store.get_known_ids(session.session_id)
        assert known == {"mem-1", "mem-2", "mem-3"}

    def test_add_known_ids_empty_set(self):
        """Adding empty set should be a no-op."""
        store = SessionStore()
        session = store.create(user_id="filip")
        store.add_known_ids(session.session_id, set())
        assert store.get_known_ids(session.session_id) == set()

    def test_add_known_ids_nonexistent_session(self):
        """Adding to nonexistent session should be a no-op."""
        store = SessionStore()
        store.add_known_ids("nonexistent", {"mem-1"})
        # Should not raise

    def test_get_known_ids_nonexistent(self):
        """Getting known IDs for nonexistent session returns empty set."""
        store = SessionStore()
        assert store.get_known_ids("nonexistent") == set()

    def test_active_count(self):
        """Active count should reflect non-expired sessions."""
        store = SessionStore()
        store.create(user_id="user1")
        store.create(user_id="user2")
        assert store.active_count == 2

    def test_sweep_removes_expired(self):
        """Sweep should remove expired sessions."""
        store = SessionStore(default_ttl=1)
        s1 = store.create(user_id="user1")
        s2 = store.create(user_id="user2")
        # Expire s1
        s1.last_accessed = datetime.now(timezone.utc) - timedelta(seconds=2)
        removed = store._sweep()
        assert removed == 1
        assert store.get(s1.session_id) is None
        assert store.get(s2.session_id) is not None

    def test_sweep_no_expired(self):
        """Sweep with no expired sessions should return 0."""
        store = SessionStore()
        store.create(user_id="user1")
        removed = store._sweep()
        assert removed == 0

    def test_multiple_users_independent(self):
        """Sessions for different users should be independent."""
        store = SessionStore()
        s1 = store.create(user_id="user1")
        s2 = store.create(user_id="user2")
        store.add_known_ids(s1.session_id, {"mem-1"})
        store.add_known_ids(s2.session_id, {"mem-2"})
        assert store.get_known_ids(s1.session_id) == {"mem-1"}
        assert store.get_known_ids(s2.session_id) == {"mem-2"}


class TestSessionStoreRememberContext:
    """Test remember pipeline context methods on SessionStore."""

    def test_add_extracted_memories(self):
        """Should append extracted memory texts to session."""
        store = SessionStore()
        session = store.create(user_id="filip")
        store.add_extracted_memories(session.session_id, ["fact A", "fact B"])
        ctx = store.get_remember_context(session.session_id)
        assert ctx["extracted_memories"] == ["fact A", "fact B"]

    def test_add_extracted_memories_accumulates(self):
        """Multiple calls should accumulate texts."""
        store = SessionStore()
        session = store.create(user_id="filip")
        store.add_extracted_memories(session.session_id, ["fact A"])
        store.add_extracted_memories(session.session_id, ["fact B", "fact C"])
        ctx = store.get_remember_context(session.session_id)
        assert ctx["extracted_memories"] == ["fact A", "fact B", "fact C"]

    def test_add_extracted_memories_fifo_eviction(self):
        """Should evict oldest entries when exceeding max_entries."""
        store = SessionStore()
        session = store.create(user_id="filip")
        store.add_extracted_memories(
            session.session_id, [f"fact-{i}" for i in range(10)], max_entries=5
        )
        ctx = store.get_remember_context(session.session_id)
        assert len(ctx["extracted_memories"]) == 5
        # Should keep the most recent 5
        assert ctx["extracted_memories"] == [f"fact-{i}" for i in range(5, 10)]

    def test_add_extracted_memories_empty_list_noop(self):
        """Empty list should be a no-op."""
        store = SessionStore()
        session = store.create(user_id="filip")
        store.add_extracted_memories(session.session_id, [])
        ctx = store.get_remember_context(session.session_id)
        assert ctx["extracted_memories"] == []

    def test_add_extracted_memories_nonexistent_session(self):
        """Adding to nonexistent session should be a no-op."""
        store = SessionStore()
        store.add_extracted_memories("nonexistent", ["fact A"])
        # Should not raise

    def test_append_summary(self):
        """Should append turn summary to session."""
        store = SessionStore()
        session = store.create(user_id="filip")
        store.append_summary(session.session_id, "Turn 1 summary")
        ctx = store.get_remember_context(session.session_id)
        assert ctx["conversation_summary"] == "Turn 1 summary"

    def test_append_summary_accumulates(self):
        """Multiple appends should join with newlines."""
        store = SessionStore()
        session = store.create(user_id="filip")
        store.append_summary(session.session_id, "Turn 1 summary")
        store.append_summary(session.session_id, "Turn 2 summary")
        ctx = store.get_remember_context(session.session_id)
        assert ctx["conversation_summary"] == "Turn 1 summary\nTurn 2 summary"

    def test_append_summary_strips_whitespace(self):
        """Should strip whitespace from turn summaries."""
        store = SessionStore()
        session = store.create(user_id="filip")
        store.append_summary(session.session_id, "  Turn 1  ")
        ctx = store.get_remember_context(session.session_id)
        assert ctx["conversation_summary"] == "Turn 1"

    def test_append_summary_empty_noop(self):
        """Empty or whitespace-only summary should be a no-op."""
        store = SessionStore()
        session = store.create(user_id="filip")
        store.append_summary(session.session_id, "")
        store.append_summary(session.session_id, "   ")
        ctx = store.get_remember_context(session.session_id)
        assert ctx["conversation_summary"] == ""

    def test_append_summary_nonexistent_session(self):
        """Appending to nonexistent session should be a no-op."""
        store = SessionStore()
        store.append_summary("nonexistent", "Turn 1 summary")
        # Should not raise

    def test_set_summary(self):
        """Should replace the conversation summary."""
        store = SessionStore()
        session = store.create(user_id="filip")
        store.append_summary(session.session_id, "Turn 1")
        store.append_summary(session.session_id, "Turn 2")
        store.set_summary(session.session_id, "Compacted summary")
        ctx = store.get_remember_context(session.session_id)
        assert ctx["conversation_summary"] == "Compacted summary"

    def test_set_summary_nonexistent_session(self):
        """Setting summary on nonexistent session should be a no-op."""
        store = SessionStore()
        store.set_summary("nonexistent", "summary")
        # Should not raise

    def test_get_remember_context_empty(self):
        """Fresh session should return empty context."""
        store = SessionStore()
        session = store.create(user_id="filip")
        ctx = store.get_remember_context(session.session_id)
        assert ctx == {"extracted_memories": [], "conversation_summary": ""}

    def test_get_remember_context_nonexistent(self):
        """Nonexistent session should return empty context."""
        store = SessionStore()
        ctx = store.get_remember_context("nonexistent")
        assert ctx == {"extracted_memories": [], "conversation_summary": ""}

    def test_get_remember_context_returns_copy(self):
        """Returned extracted_memories should be a copy, not a reference."""
        store = SessionStore()
        session = store.create(user_id="filip")
        store.add_extracted_memories(session.session_id, ["fact A"])
        ctx = store.get_remember_context(session.session_id)
        ctx["extracted_memories"].append("injected")
        # Original should be unaffected
        ctx2 = store.get_remember_context(session.session_id)
        assert ctx2["extracted_memories"] == ["fact A"]

    def test_expired_session_returns_empty_context(self):
        """Expired session should return empty context."""
        store = SessionStore(default_ttl=1)
        session = store.create(user_id="filip")
        store.add_extracted_memories(session.session_id, ["fact A"])
        store.append_summary(session.session_id, "Turn 1")
        # Manually expire
        session.last_accessed = datetime.now(timezone.utc) - timedelta(seconds=2)
        ctx = store.get_remember_context(session.session_id)
        assert ctx == {"extracted_memories": [], "conversation_summary": ""}


# ── Serialization Tests ──────────────────────────────────────────────


class TestSessionSerialization:
    """Test serialize_session / deserialize_session roundtrip."""

    def test_roundtrip_basic(self):
        """Serialize then deserialize should produce equivalent session."""
        session = MemorySession(
            session_id="test-rt-1",
            user_id="filip",
            agent_id="claude-code",
            ttl_seconds=86400,
        )
        data = serialize_session(session)
        restored = deserialize_session(data)
        assert restored.session_id == session.session_id
        assert restored.user_id == session.user_id
        assert restored.agent_id == session.agent_id
        assert restored.ttl_seconds == session.ttl_seconds
        assert restored.known_memory_ids == set()
        assert restored.extracted_memories == []
        assert restored.conversation_summary == ""

    def test_roundtrip_with_data(self):
        """Roundtrip should preserve known_memory_ids and remember context."""
        session = MemorySession(
            session_id="test-rt-2",
            user_id="filip",
            agent_id=None,
            known_memory_ids={"mem-1", "mem-2", "mem-3"},
            extracted_memories=["fact A", "fact B"],
            conversation_summary="User discussed project plans.",
        )
        data = serialize_session(session)
        restored = deserialize_session(data)
        assert restored.known_memory_ids == {"mem-1", "mem-2", "mem-3"}
        assert restored.extracted_memories == ["fact A", "fact B"]
        assert restored.conversation_summary == "User discussed project plans."

    def test_roundtrip_none_agent_id(self):
        """None agent_id should survive roundtrip."""
        session = MemorySession(
            session_id="test-rt-3",
            user_id="filip",
            agent_id=None,
        )
        data = serialize_session(session)
        restored = deserialize_session(data)
        assert restored.agent_id is None

    def test_roundtrip_timestamps(self):
        """Timestamps should survive roundtrip."""
        now = datetime.now(timezone.utc)
        session = MemorySession(
            session_id="test-rt-4",
            user_id="filip",
            agent_id=None,
            created_at=now,
            last_accessed=now,
        )
        data = serialize_session(session)
        restored = deserialize_session(data)
        # Allow small delta due to ISO format precision
        assert abs((restored.created_at - now).total_seconds()) < 0.001
        assert abs((restored.last_accessed - now).total_seconds()) < 0.001

    def test_deserialize_json_string_fields(self):
        """Deserialize should handle JSON-encoded string fields (from SQLite/Redis)."""
        data = {
            "session_id": "test-json",
            "user_id": "filip",
            "agent_id": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_accessed": datetime.now(timezone.utc).isoformat(),
            "ttl_seconds": 3600,
            "known_memory_ids": '["mem-1", "mem-2"]',
            "extracted_memories": '["fact A"]',
            "conversation_summary": "summary text",
        }
        restored = deserialize_session(data)
        assert restored.known_memory_ids == {"mem-1", "mem-2"}
        assert restored.extracted_memories == ["fact A"]

    def test_deserialize_empty_fields(self):
        """Deserialize should handle missing/empty optional fields."""
        data = {
            "session_id": "test-empty",
            "user_id": "filip",
        }
        restored = deserialize_session(data)
        assert restored.agent_id is None
        assert restored.known_memory_ids == set()
        assert restored.extracted_memories == []
        assert restored.conversation_summary == ""
        assert restored.ttl_seconds == 86400  # default in deserialize


# ── SQLite Backend Tests ─────────────────────────────────────────────


class TestSQLiteBackend:
    """Test SQLiteBackend persistence."""

    def _make_backend(self, tmp_path: str) -> SQLiteBackend:
        db_path = os.path.join(tmp_path, "test_sessions.db")
        return SQLiteBackend(db_path)

    def _make_session_data(self, **overrides) -> dict:
        defaults = {
            "session_id": "test-sqlite-1",
            "user_id": "filip",
            "agent_id": "claude-code",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_accessed": datetime.now(timezone.utc).isoformat(),
            "ttl_seconds": 86400,
            "known_memory_ids": ["mem-1", "mem-2"],
            "extracted_memories": ["fact A"],
            "conversation_summary": "test summary",
        }
        defaults.update(overrides)
        return defaults

    def test_save_and_load(self, tmp_path):
        """Save then load should return the same data."""
        backend = self._make_backend(str(tmp_path))
        data = self._make_session_data()
        backend.save(data)
        loaded = backend.load("test-sqlite-1")
        assert loaded is not None
        assert loaded["session_id"] == "test-sqlite-1"
        assert loaded["user_id"] == "filip"
        assert loaded["agent_id"] == "claude-code"
        backend.close()

    def test_load_nonexistent(self, tmp_path):
        """Load should return None for unknown session."""
        backend = self._make_backend(str(tmp_path))
        assert backend.load("nonexistent") is None
        backend.close()

    def test_delete(self, tmp_path):
        """Delete should remove the session."""
        backend = self._make_backend(str(tmp_path))
        data = self._make_session_data()
        backend.save(data)
        backend.delete("test-sqlite-1")
        assert backend.load("test-sqlite-1") is None
        backend.close()

    def test_upsert(self, tmp_path):
        """Saving same session_id should update, not duplicate."""
        backend = self._make_backend(str(tmp_path))
        data = self._make_session_data()
        backend.save(data)
        data["conversation_summary"] = "updated summary"
        backend.save(data)
        loaded = backend.load("test-sqlite-1")
        assert loaded["conversation_summary"] == "updated summary"
        assert backend.count_active() == 1
        backend.close()

    def test_sweep_expired(self, tmp_path):
        """Sweep should remove expired sessions."""
        backend = self._make_backend(str(tmp_path))
        # Active session
        backend.save(self._make_session_data(session_id="active", ttl_seconds=86400))
        # Expired session (last_accessed 2 days ago, TTL 1 second)
        expired_time = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        backend.save(
            self._make_session_data(
                session_id="expired",
                last_accessed=expired_time,
                ttl_seconds=1,
            )
        )
        removed = backend.sweep_expired()
        assert removed == 1
        assert backend.load("active") is not None
        assert backend.load("expired") is None
        backend.close()

    def test_count_active(self, tmp_path):
        """Count should only include non-expired sessions."""
        backend = self._make_backend(str(tmp_path))
        backend.save(self._make_session_data(session_id="s1"))
        backend.save(self._make_session_data(session_id="s2"))
        assert backend.count_active() == 2
        backend.close()

    def test_persistence_across_connections(self, tmp_path):
        """Data should survive closing and reopening the backend."""
        db_path = os.path.join(str(tmp_path), "persist_test.db")
        backend1 = SQLiteBackend(db_path)
        backend1.save(self._make_session_data(session_id="persist-1"))
        backend1.close()

        # Open a new connection to the same DB
        backend2 = SQLiteBackend(db_path)
        loaded = backend2.load("persist-1")
        assert loaded is not None
        assert loaded["session_id"] == "persist-1"
        backend2.close()

    def test_set_type_known_memory_ids(self, tmp_path):
        """Save should handle set-type known_memory_ids."""
        backend = self._make_backend(str(tmp_path))
        data = self._make_session_data(
            known_memory_ids={"mem-a", "mem-b"}  # set, not list
        )
        backend.save(data)
        loaded = backend.load("test-sqlite-1")
        assert loaded is not None
        # SQLite stores as JSON string
        import json

        ids = json.loads(loaded["known_memory_ids"])
        assert set(ids) == {"mem-a", "mem-b"}
        backend.close()


# ── MemoryBackend Tests ──────────────────────────────────────────────


class TestMemoryBackend:
    """Test MemoryBackend (no-op backend)."""

    def test_save_load_returns_none(self):
        """MemoryBackend.load always returns None (no persistence)."""
        backend = MemoryBackend()
        backend.save({"session_id": "test", "user_id": "filip"})
        assert backend.load("test") is None

    def test_delete_noop(self):
        """Delete should not raise."""
        backend = MemoryBackend()
        backend.delete("nonexistent")

    def test_sweep_returns_zero(self):
        """Sweep should return 0."""
        assert MemoryBackend().sweep_expired() == 0

    def test_count_returns_zero(self):
        """Count should return 0."""
        assert MemoryBackend().count_active() == 0

    def test_close_noop(self):
        """Close should not raise."""
        MemoryBackend().close()


# ── Write-Through Cache Tests ────────────────────────────────────────


class TestWriteThroughCache:
    """Test SessionStore write-through cache behavior with SQLite backend."""

    def _make_store(self, tmp_path: str) -> tuple[SessionStore, SQLiteBackend]:
        db_path = os.path.join(tmp_path, "cache_test.db")
        backend = SQLiteBackend(db_path)
        store = SessionStore(default_ttl=3600, backend=backend)
        return store, backend

    def test_create_persists_to_backend(self, tmp_path):
        """Creating a session should write to both cache and backend."""
        store, backend = self._make_store(str(tmp_path))
        session = store.create(user_id="filip")
        # Verify it's in the backend
        loaded = backend.load(session.session_id)
        assert loaded is not None
        assert loaded["user_id"] == "filip"
        store.close()

    def test_cache_hit(self, tmp_path):
        """Get should return from cache without hitting backend."""
        store, backend = self._make_store(str(tmp_path))
        session = store.create(user_id="filip")
        # Get should hit cache
        retrieved = store.get(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id
        store.close()

    def test_cache_miss_loads_from_backend(self, tmp_path):
        """Cache miss should load from backend and populate cache."""
        db_path = os.path.join(str(tmp_path), "miss_test.db")
        backend = SQLiteBackend(db_path)

        # Create session with first store
        store1 = SessionStore(default_ttl=3600, backend=backend)
        session = store1.create(user_id="filip", agent_id="test")
        sid = session.session_id

        # Create second store with same backend (simulates restart)
        store2 = SessionStore(default_ttl=3600, backend=backend)
        # Cache is empty, should load from backend
        retrieved = store2.get(sid)
        assert retrieved is not None
        assert retrieved.user_id == "filip"
        assert retrieved.agent_id == "test"

        # Now it should be in cache
        assert sid in store2._cache
        store1.close()
        store2.close()

    def test_touch_persists(self, tmp_path):
        """Touch should persist updated last_accessed to backend."""
        store, backend = self._make_store(str(tmp_path))
        session = store.create(user_id="filip")
        time.sleep(0.01)
        store.touch(session.session_id)
        loaded = backend.load(session.session_id)
        assert loaded is not None
        restored = deserialize_session(loaded)
        assert restored.last_accessed > session.created_at
        store.close()

    def test_add_known_ids_persists(self, tmp_path):
        """add_known_ids should persist to backend."""
        store, backend = self._make_store(str(tmp_path))
        session = store.create(user_id="filip")
        store.add_known_ids(session.session_id, {"mem-1", "mem-2"})
        loaded = backend.load(session.session_id)
        assert loaded is not None
        restored = deserialize_session(loaded)
        assert restored.known_memory_ids == {"mem-1", "mem-2"}
        store.close()

    def test_add_extracted_memories_persists(self, tmp_path):
        """add_extracted_memories should persist to backend."""
        store, backend = self._make_store(str(tmp_path))
        session = store.create(user_id="filip")
        store.add_extracted_memories(session.session_id, ["fact A", "fact B"])
        loaded = backend.load(session.session_id)
        assert loaded is not None
        restored = deserialize_session(loaded)
        assert restored.extracted_memories == ["fact A", "fact B"]
        store.close()

    def test_append_summary_persists(self, tmp_path):
        """append_summary should persist to backend."""
        store, backend = self._make_store(str(tmp_path))
        session = store.create(user_id="filip")
        store.append_summary(session.session_id, "Turn 1")
        loaded = backend.load(session.session_id)
        assert loaded is not None
        restored = deserialize_session(loaded)
        assert restored.conversation_summary == "Turn 1"
        store.close()

    def test_set_summary_persists(self, tmp_path):
        """set_summary should persist to backend."""
        store, backend = self._make_store(str(tmp_path))
        session = store.create(user_id="filip")
        store.set_summary(session.session_id, "Compacted summary")
        loaded = backend.load(session.session_id)
        assert loaded is not None
        restored = deserialize_session(loaded)
        assert restored.conversation_summary == "Compacted summary"
        store.close()

    def test_expired_session_deleted_from_backend_on_access(self, tmp_path):
        """Accessing an expired session should delete it from backend too."""
        store, backend = self._make_store(str(tmp_path))
        session = store.create(user_id="filip", ttl_seconds=1)
        sid = session.session_id
        # Manually expire
        session.last_accessed = datetime.now(timezone.utc) - timedelta(seconds=2)
        # Access should return None and clean up
        assert store.get(sid) is None
        # Backend should also be cleaned
        assert backend.load(sid) is None
        store.close()

    def test_expired_backend_session_deleted_on_load(self, tmp_path):
        """Loading an expired session from backend should delete it."""
        db_path = os.path.join(str(tmp_path), "expire_backend.db")
        backend = SQLiteBackend(db_path)

        # Create session and expire it
        store1 = SessionStore(default_ttl=1, backend=backend)
        session = store1.create(user_id="filip")
        sid = session.session_id
        session.last_accessed = datetime.now(timezone.utc) - timedelta(seconds=2)
        # Persist the expired state
        store1._persist(session)

        # New store (empty cache) tries to load expired session
        store2 = SessionStore(default_ttl=1, backend=backend)
        assert store2.get(sid) is None
        # Backend should be cleaned up
        assert backend.load(sid) is None

    def test_sweep_cleans_backend(self, tmp_path):
        """Sweep should clean both cache and backend."""
        store, backend = self._make_store(str(tmp_path))
        s1 = store.create(user_id="user1", ttl_seconds=1)
        store.create(user_id="user2")
        # Expire s1
        s1.last_accessed = datetime.now(timezone.utc) - timedelta(seconds=2)
        store._persist(s1)  # Persist expired state to backend
        removed = store._sweep()
        assert removed >= 1
        assert backend.load(s1.session_id) is None
        store.close()

    def test_session_survives_restart(self, tmp_path):
        """Full session data should survive a simulated restart."""
        db_path = os.path.join(str(tmp_path), "restart_test.db")
        backend = SQLiteBackend(db_path)

        # Create session with data
        store1 = SessionStore(default_ttl=86400, backend=backend)
        session = store1.create(user_id="filip", agent_id="claude-code")
        sid = session.session_id
        store1.add_known_ids(sid, {"mem-1", "mem-2"})
        store1.add_extracted_memories(sid, ["fact A", "fact B"])
        store1.append_summary(sid, "Turn 1 summary")

        # Simulate restart: new store, same backend, empty cache
        store2 = SessionStore(default_ttl=86400, backend=backend)
        assert sid not in store2._cache

        # Get should load from backend
        retrieved = store2.get(sid)
        assert retrieved is not None
        assert retrieved.user_id == "filip"
        assert retrieved.agent_id == "claude-code"
        assert retrieved.known_memory_ids == {"mem-1", "mem-2"}

        # Remember context should also survive
        ctx = store2.get_remember_context(sid)
        assert ctx["extracted_memories"] == ["fact A", "fact B"]
        assert ctx["conversation_summary"] == "Turn 1 summary"

        backend.close()

    def test_backend_failure_does_not_break_cache(self, tmp_path):
        """Backend failures should be logged but not break cache operations."""
        mock_backend = MagicMock()
        mock_backend.save.side_effect = Exception("DB error")
        mock_backend.load.return_value = None

        store = SessionStore(default_ttl=3600, backend=mock_backend)
        # Create should succeed (cache works even if backend fails)
        session = store.create(user_id="filip")
        assert session is not None

        # Get should still work from cache
        retrieved = store.get(session.session_id)
        assert retrieved is not None
        assert retrieved.user_id == "filip"

    def test_default_backend_is_memory(self):
        """SessionStore without backend arg should use MemoryBackend."""
        store = SessionStore()
        assert isinstance(store._backend, MemoryBackend)


# ── URL Masking Tests ────────────────────────────────────────────────


class TestMaskUrl:
    """Test _mask_url helper for safe logging."""

    def test_mask_password(self):
        """Should mask username and password."""
        assert _mask_url("redis://user:secret@host:6379/0") == (
            "redis://***:***@host:6379/0"
        )

    def test_mask_password_only(self):
        """Should mask when only password is present."""
        assert _mask_url("redis://:secret@host:6379") == "redis://***:***@host:6379"

    def test_no_credentials(self):
        """Should return URL unchanged when no credentials."""
        url = "redis://host:6379/0"
        assert _mask_url(url) == url

    def test_empty_string(self):
        """Should handle empty string."""
        assert _mask_url("") == ""

    def test_plain_host(self):
        """Should handle plain host without scheme."""
        assert _mask_url("localhost:6379") == "localhost:6379"


# ── Factory Tests ────────────────────────────────────────────────────


class TestCreateSessionBackend:
    """Test create_session_backend factory function."""

    def test_create_memory_backend(self):
        """Should create MemoryBackend."""
        backend = create_session_backend("memory")
        assert isinstance(backend, MemoryBackend)

    def test_create_sqlite_backend(self, tmp_path):
        """Should create SQLiteBackend with given path."""
        db_path = os.path.join(str(tmp_path), "factory_test.db")
        backend = create_session_backend("sqlite", session_path=db_path)
        assert isinstance(backend, SQLiteBackend)
        backend.close()

    def test_create_redis_without_url_raises(self):
        """Should raise ValueError when redis_url is missing."""
        with pytest.raises(ValueError, match="REDIS_URL is required"):
            create_session_backend("redis")

    def test_create_invalid_backend_raises(self):
        """Should raise ValueError for unknown backend type."""
        with pytest.raises(ValueError, match="Unsupported SESSION_BACKEND"):
            create_session_backend("invalid")
