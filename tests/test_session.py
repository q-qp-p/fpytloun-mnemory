"""Tests for MemorySession and SessionStore."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from mnemory.session import MemorySession, SessionStore


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
