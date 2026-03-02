"""Session storage backends for persistent session state.

Supports three backends:
- Memory: In-memory only (no persistence, for tests/dev)
- SQLite: Local file-based persistence (default for single-node)
- Redis: Distributed persistence (for clustered deployments)

All backends work with serialized session dicts. The SessionStore
handles MemorySession <-> dict conversion and maintains a write-through
in-memory cache on top of the backend.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Protocol
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger("mnemory")


def _mask_url(url: str) -> str:
    """Mask credentials in a URL for safe logging.

    Replaces the password (and optionally username) with '***'.
    Returns the original string if it's not a valid URL.
    """
    try:
        parsed = urlparse(url)
        if parsed.password or parsed.username:
            # Rebuild netloc with masked credentials
            host = parsed.hostname or ""
            port = f":{parsed.port}" if parsed.port else ""
            netloc = f"***:***@{host}{port}"
            return urlunparse(parsed._replace(netloc=netloc))
        return url
    except Exception:
        return "<masked>"


# ── Serialization helpers ────────────────────────────────────────────


def serialize_session(session: Any) -> dict[str, Any]:
    """Serialize a MemorySession to a plain dict for storage.

    Args:
        session: A MemorySession dataclass instance.

    Returns:
        Dict with all fields as JSON-safe types.
    """
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "agent_id": session.agent_id,
        "created_at": session.created_at.isoformat(),
        "last_accessed": session.last_accessed.isoformat(),
        "ttl_seconds": session.ttl_seconds,
        "known_memory_ids": list(session.known_memory_ids),
        "extracted_memories": list(session.extracted_memories),
        "conversation_summary": session.conversation_summary,
    }


def deserialize_session(data: dict[str, Any]) -> Any:
    """Deserialize a plain dict back to a MemorySession.

    Imports MemorySession lazily to avoid circular imports.

    Args:
        data: Dict from serialize_session or backend storage.

    Returns:
        A MemorySession dataclass instance.
    """
    from mnemory.session import MemorySession

    # Parse known_memory_ids — could be JSON string or list
    known_ids = data.get("known_memory_ids", [])
    if isinstance(known_ids, str):
        try:
            known_ids = json.loads(known_ids)
        except (json.JSONDecodeError, TypeError):
            known_ids = []

    # Parse extracted_memories — could be JSON string or list
    extracted = data.get("extracted_memories", [])
    if isinstance(extracted, str):
        try:
            extracted = json.loads(extracted)
        except (json.JSONDecodeError, TypeError):
            extracted = []
    if not isinstance(extracted, list):
        extracted = []

    return MemorySession(
        session_id=data["session_id"],
        user_id=data["user_id"],
        agent_id=data.get("agent_id") or None,
        known_memory_ids=set(known_ids) if known_ids else set(),
        created_at=_parse_datetime(data.get("created_at", "")),
        last_accessed=_parse_datetime(data.get("last_accessed", "")),
        ttl_seconds=int(data.get("ttl_seconds", 86400)),
        extracted_memories=list(extracted),
        conversation_summary=str(data.get("conversation_summary", "")),
    )


def _parse_datetime(value: Any) -> datetime:
    """Parse an ISO 8601 datetime string to a timezone-aware datetime."""
    if not value:
        return datetime.now(timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


# ── Backend Protocol ─────────────────────────────────────────────────


class SessionBackend(Protocol):
    """Protocol for session storage backends.

    Backends handle durable persistence of serialized session dicts.
    The SessionStore maintains an in-memory write-through cache on top.

    All methods must be thread-safe.
    """

    def save(self, session_data: dict[str, Any]) -> None:
        """Save or update a session.

        Args:
            session_data: Serialized session dict from serialize_session().
        """
        ...

    def load(self, session_id: str) -> dict[str, Any] | None:
        """Load a session by ID.

        Returns:
            Serialized session dict, or None if not found.
            Does NOT check expiry — the caller handles that.
        """
        ...

    def delete(self, session_id: str) -> None:
        """Delete a session by ID."""
        ...

    def sweep_expired(self) -> int:
        """Remove expired sessions from the backend.

        Returns:
            Number of sessions removed.
        """
        ...

    def count_active(self) -> int:
        """Count non-expired sessions in the backend."""
        ...

    def close(self) -> None:
        """Clean up resources (connections, file handles)."""
        ...


# ── Memory Backend (no persistence) ─────────────────────────────────


class MemoryBackend:
    """In-memory session backend with no persistence.

    All data lives only in the SessionStore's write-through cache.
    This backend is a no-op — save/load/delete do nothing.

    Use for tests or when persistence is not needed.
    """

    def save(self, session_data: dict[str, Any]) -> None:
        """No-op — data lives only in the cache."""

    def load(self, session_id: str) -> dict[str, Any] | None:
        """Always returns None — no persistent storage."""
        return None

    def delete(self, session_id: str) -> None:
        """No-op."""

    def sweep_expired(self) -> int:
        """No-op — cache handles expiry."""
        return 0

    def count_active(self) -> int:
        """Returns 0 — cache handles counting."""
        return 0

    def close(self) -> None:
        """No-op."""


# ── SQLite Backend ───────────────────────────────────────────────────


class SQLiteBackend:
    """SQLite-based session persistence.

    Stores sessions in a local SQLite database file. Uses WAL mode
    for concurrent read performance. Thread-safe via an explicit
    write lock and check_same_thread=False.

    Default path: ~/.mnemory/sessions.db
    """

    def __init__(self, db_path: str):
        self._db_path = db_path

        # Ensure parent directory exists
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        self._write_lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create the sessions table if it doesn't exist."""
        with self._write_lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    agent_id TEXT,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    ttl_seconds INTEGER NOT NULL,
                    known_memory_ids TEXT NOT NULL DEFAULT '[]',
                    extracted_memories TEXT NOT NULL DEFAULT '[]',
                    conversation_summary TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_last_accessed
                    ON sessions(last_accessed);
                """
            )

    def save(self, session_data: dict[str, Any]) -> None:
        """Insert or replace a session in the database."""
        known_ids = session_data.get("known_memory_ids", [])
        if isinstance(known_ids, (set, frozenset)):
            known_ids = list(known_ids)
        extracted = session_data.get("extracted_memories", [])

        with self._write_lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO sessions
                    (session_id, user_id, agent_id, created_at, last_accessed,
                     ttl_seconds, known_memory_ids, extracted_memories,
                     conversation_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_data["session_id"],
                    session_data["user_id"],
                    session_data.get("agent_id"),
                    session_data.get("created_at", ""),
                    session_data.get("last_accessed", ""),
                    int(session_data.get("ttl_seconds", 86400)),
                    json.dumps(known_ids),
                    json.dumps(extracted),
                    session_data.get("conversation_summary", ""),
                ),
            )
            self._conn.commit()

    def load(self, session_id: str) -> dict[str, Any] | None:
        """Load a session from the database."""
        with self._write_lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

        if row is None:
            return None

        return dict(row)

    def delete(self, session_id: str) -> None:
        """Delete a session from the database."""
        with self._write_lock:
            self._conn.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            self._conn.commit()

    def sweep_expired(self) -> int:
        """Remove expired sessions based on last_accessed + ttl_seconds."""
        now = datetime.now(timezone.utc).isoformat()
        with self._write_lock:
            cursor = self._conn.execute(
                """
                DELETE FROM sessions
                WHERE julianday(?) - julianday(last_accessed) >
                      (ttl_seconds / 86400.0)
                """,
                (now,),
            )
            self._conn.commit()
            return cursor.rowcount

    def count_active(self) -> int:
        """Count non-expired sessions."""
        now = datetime.now(timezone.utc).isoformat()
        with self._write_lock:
            row = self._conn.execute(
                """
                SELECT COUNT(*) FROM sessions
                WHERE julianday(?) - julianday(last_accessed) <=
                      (ttl_seconds / 86400.0)
                """,
                (now,),
            ).fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except Exception:
            pass


# ── Redis Backend ────────────────────────────────────────────────────


class RedisBackend:
    """Redis-based session persistence for clustered deployments.

    Each session is stored as a Redis hash at key
    ``mnemory:session:{session_id}``. Uses Redis TTL for automatic
    expiry — no sweep needed.

    Requires the ``redis`` package (optional dependency).
    """

    _PREFIX = "mnemory:session:"

    def __init__(self, redis_url: str):
        try:
            import redis as redis_lib
        except ImportError:
            raise ImportError(
                "The 'redis' package is required for SESSION_BACKEND=redis. "
                "Install it with: pip install mnemory[redis]"
            )

        self._redis: Any = redis_lib.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )
        # Verify connection
        try:
            self._redis.ping()
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Redis at {_mask_url(redis_url)}: {e}"
            ) from e

        logger.info("Redis session backend connected: %s", _mask_url(redis_url))

    def _key(self, session_id: str) -> str:
        """Build the Redis key for a session."""
        return f"{self._PREFIX}{session_id}"

    def save(self, session_data: dict[str, Any]) -> None:
        """Save a session as a Redis hash with TTL."""
        session_id = session_data["session_id"]
        ttl_seconds = int(session_data.get("ttl_seconds", 86400))
        key = self._key(session_id)

        # Serialize complex fields to JSON strings for Redis hash
        known_ids = session_data.get("known_memory_ids", [])
        if isinstance(known_ids, (set, frozenset)):
            known_ids = list(known_ids)
        extracted = session_data.get("extracted_memories", [])

        hash_data = {
            "session_id": session_id,
            "user_id": session_data["user_id"],
            "agent_id": session_data.get("agent_id") or "",
            "created_at": str(session_data.get("created_at", "")),
            "last_accessed": str(session_data.get("last_accessed", "")),
            "ttl_seconds": str(ttl_seconds),
            "known_memory_ids": json.dumps(known_ids),
            "extracted_memories": json.dumps(extracted),
            "conversation_summary": session_data.get("conversation_summary", ""),
        }

        pipe = self._redis.pipeline()
        pipe.hset(key, mapping=hash_data)
        # Redis TTL = session TTL + 10% buffer + 60s for clock skew
        redis_ttl = int(ttl_seconds * 1.1) + 60
        pipe.expire(key, redis_ttl)
        pipe.execute()

    def load(self, session_id: str) -> dict[str, Any] | None:
        """Load a session from Redis."""
        data = self._redis.hgetall(self._key(session_id))
        if not data:
            return None

        # Convert agent_id empty string back to None
        if data.get("agent_id") == "":
            data["agent_id"] = None

        # ttl_seconds is stored as string in Redis
        if "ttl_seconds" in data:
            data["ttl_seconds"] = int(data["ttl_seconds"])

        return data

    def delete(self, session_id: str) -> None:
        """Delete a session from Redis."""
        self._redis.delete(self._key(session_id))

    def sweep_expired(self) -> int:
        """No-op — Redis TTL handles expiry automatically."""
        return 0

    def count_active(self) -> int:
        """Count active session keys in Redis via SCAN."""
        count = 0
        cursor = 0
        pattern = f"{self._PREFIX}*"
        while True:
            cursor, keys = self._redis.scan(cursor=cursor, match=pattern, count=100)
            count += len(keys)
            if cursor == 0:
                break
        return count

    def close(self) -> None:
        """Close the Redis connection."""
        try:
            self._redis.close()
        except Exception:
            pass


# ── Factory ──────────────────────────────────────────────────────────


def create_session_backend(
    backend_type: str,
    *,
    session_path: str = "",
    redis_url: str = "",
) -> SessionBackend:
    """Create the appropriate session backend from configuration.

    Args:
        backend_type: One of "memory", "sqlite", "redis".
        session_path: Path for SQLite database file.
        redis_url: Redis connection URL.

    Returns:
        A SessionBackend instance.
    """
    if backend_type == "memory":
        logger.info("Session backend: in-memory (no persistence)")
        return MemoryBackend()

    if backend_type == "sqlite":
        if not session_path:
            from mnemory.config import _data_dir

            session_path = os.path.join(_data_dir(), "sessions.db")
        logger.info("Session backend: SQLite (%s)", session_path)
        return SQLiteBackend(session_path)

    if backend_type == "redis":
        if not redis_url:
            raise ValueError("REDIS_URL is required when SESSION_BACKEND=redis")
        logger.info("Session backend: Redis (%s)", _mask_url(redis_url))
        return RedisBackend(redis_url)

    raise ValueError(
        f"Unsupported SESSION_BACKEND: {backend_type}. "
        "Must be one of: memory, sqlite, redis"
    )
