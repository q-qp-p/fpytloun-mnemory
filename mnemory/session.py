"""Server-side memory session tracking.

Tracks which memories the client already has in its context, so subsequent
recall calls only return NEW relevant memories. Prevents context bloat.

Losing a session is harmless — worst case, the client gets memories it
already has (minor duplication, not a failure).

Supports pluggable backends for persistent storage with a write-through
in-memory cache. Cache starts empty; sessions are loaded lazily from
the backend on first access.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("mnemory")

# Max known memory IDs per session. Prevents unbounded growth for
# long-lived sessions with many recall calls. When exceeded, oldest
# entries are discarded (harmless — may cause minor duplication).
_MAX_KNOWN_IDS = 10000


@dataclass
class MemorySession:
    """Tracks client-side memory state on the server.

    Keeps track of which memories the client already has in its context,
    so subsequent recall calls only return NEW relevant memories.

    Also maintains remember pipeline context: a running conversation
    summary and list of already-extracted memory texts, so subsequent
    remember calls can avoid re-extracting known facts and maintain
    conversation continuity.
    """

    session_id: str
    user_id: str
    agent_id: str | None
    known_memory_ids: set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_seconds: int = 3600

    # Remember pipeline context — accumulated across remember calls.
    # extracted_memories: texts of facts already extracted in this session,
    #   used to tell the extraction LLM to skip known facts.
    # conversation_summary: running summary of all conversation turns,
    #   used to give the extraction LLM conversation continuity.
    extracted_memories: list[str] = field(default_factory=list)
    conversation_summary: str = ""

    @property
    def is_expired(self) -> bool:
        """Check if the session has expired (idle timeout)."""
        elapsed = (datetime.now(timezone.utc) - self.last_accessed).total_seconds()
        return elapsed > self.ttl_seconds


class SessionStore:
    """Session store with write-through cache and pluggable backend.

    Maintains an in-memory cache for fast access, with all mutations
    written through to the backend for persistence. On cache miss,
    sessions are loaded lazily from the backend.

    Thread-safe via a lock. Sessions expire after an idle timeout
    (no access within ttl_seconds). A background task periodically
    sweeps expired sessions from both cache and backend.
    """

    def __init__(
        self,
        default_ttl: int = 3600,
        sweep_interval: int = 300,
        backend: Any | None = None,
    ):
        self._cache: dict[str, MemorySession] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl
        self._sweep_interval = sweep_interval
        self._sweep_task: asyncio.Task | None = None

        # Backend for persistent storage. None = MemoryBackend (no persistence).
        # Import lazily to avoid circular imports at module level.
        from mnemory.storage.session import (
            MemoryBackend,
            deserialize_session,
            serialize_session,
        )

        if backend is None:
            self._backend: Any = MemoryBackend()
        else:
            self._backend = backend

        # Cache serialization functions to avoid repeated import lookups
        self._serialize = serialize_session
        self._deserialize = deserialize_session

    def _persist(self, session: MemorySession) -> None:
        """Serialize and save a session to the backend.

        Called after every mutation to maintain write-through consistency.
        Errors are logged but not raised — cache is the source of truth
        for the current process; backend persistence is best-effort.
        """
        try:
            data = self._serialize(session)
            self._backend.save(data)
        except Exception:
            logger.warning(
                "Failed to persist session %s to backend",
                session.session_id,
                exc_info=True,
            )

    def _load_from_backend(self, session_id: str) -> MemorySession | None:
        """Try to load a session from the backend into the cache.

        Returns the session if found and not expired, None otherwise.
        Expired sessions loaded from backend are deleted from backend.
        """
        try:
            data = self._backend.load(session_id)
            if data is None:
                return None

            session = self._deserialize(data)

            if session.is_expired:
                # Clean up expired session from backend
                try:
                    self._backend.delete(session_id)
                except Exception:
                    pass
                logger.debug("Session expired on backend load: %s", session_id)
                return None

            # Populate cache
            self._cache[session_id] = session
            return session
        except Exception:
            logger.warning(
                "Failed to load session %s from backend",
                session_id,
                exc_info=True,
            )
            return None

    def create(
        self,
        user_id: str,
        agent_id: str | None = None,
        ttl_seconds: int | None = None,
    ) -> MemorySession:
        """Create a new session.

        Args:
            user_id: Owner of the session.
            agent_id: Optional agent scope.
            ttl_seconds: Idle TTL in seconds. Uses default if not specified.

        Returns:
            The newly created MemorySession.
        """
        session = MemorySession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            agent_id=agent_id,
            ttl_seconds=ttl_seconds if ttl_seconds is not None else self._default_ttl,
        )
        with self._lock:
            self._cache[session.session_id] = session
            self._persist(session)
        logger.debug(
            "Session created: %s (user=%s, agent=%s, ttl=%ds)",
            session.session_id,
            user_id,
            agent_id,
            session.ttl_seconds,
        )
        return session

    def get(self, session_id: str) -> MemorySession | None:
        """Get session by ID. Returns None if not found or expired.

        Checks cache first, then falls back to backend on cache miss.
        Expired sessions are lazily removed on access.
        """
        with self._lock:
            # Check cache first
            session = self._cache.get(session_id)
            if session is not None:
                if session.is_expired:
                    del self._cache[session_id]
                    try:
                        self._backend.delete(session_id)
                    except Exception:
                        pass
                    logger.debug("Session expired on access: %s", session_id)
                    return None
                return session

            # Cache miss — try backend
            return self._load_from_backend(session_id)

    def touch(self, session_id: str) -> None:
        """Update last_accessed timestamp to keep session alive."""
        with self._lock:
            session = self._cache.get(session_id)
            if session is None:
                # Try loading from backend
                session = self._load_from_backend(session_id)
            if session and not session.is_expired:
                session.last_accessed = datetime.now(timezone.utc)
                self._persist(session)

    def add_known_ids(self, session_id: str, memory_ids: set[str]) -> None:
        """Add memory IDs to the session's known set.

        These IDs will be filtered out on subsequent recall calls
        to avoid returning memories the client already has.

        If the known set exceeds the max cap, it is trimmed by
        discarding arbitrary entries. This is harmless — worst case,
        a previously-seen memory is returned again.
        """
        if not memory_ids:
            return
        with self._lock:
            session = self._cache.get(session_id)
            if session is None:
                session = self._load_from_backend(session_id)
            if session and not session.is_expired:
                session.known_memory_ids.update(memory_ids)
                # Cap the set size to prevent unbounded growth
                if len(session.known_memory_ids) > _MAX_KNOWN_IDS:
                    excess = len(session.known_memory_ids) - _MAX_KNOWN_IDS
                    # Discard arbitrary entries (set has no ordering)
                    it = iter(session.known_memory_ids)
                    to_remove = [next(it) for _ in range(excess)]
                    session.known_memory_ids.difference_update(to_remove)
                self._persist(session)

    def get_known_ids(self, session_id: str) -> set[str]:
        """Get the set of known memory IDs for a session."""
        with self._lock:
            session = self._cache.get(session_id)
            if session is None:
                session = self._load_from_backend(session_id)
            if session and not session.is_expired:
                return set(session.known_memory_ids)
            return set()

    # ── Remember pipeline context ────────────────────────────────────

    def add_extracted_memories(
        self,
        session_id: str,
        texts: list[str],
        *,
        max_entries: int = 50,
    ) -> None:
        """Append extracted memory texts to the session.

        Used by the remember pipeline to track which facts have already
        been extracted, so subsequent calls skip them.

        Capped at max_entries with FIFO eviction — older entries are
        still findable via Stage 2's per-fact vector search.
        """
        if not texts:
            return
        with self._lock:
            session = self._cache.get(session_id)
            if session is None:
                session = self._load_from_backend(session_id)
            if session and not session.is_expired:
                session.extracted_memories.extend(texts)
                if len(session.extracted_memories) > max_entries:
                    session.extracted_memories = session.extracted_memories[
                        -max_entries:
                    ]
                self._persist(session)

    def append_summary(self, session_id: str, turn_summary: str) -> None:
        """Append a turn summary to the session's conversation summary.

        Each remember call produces a 1-2 sentence summary of the
        exchange. These are accumulated to give the extraction LLM
        full conversation context.

        Summaries are appended as-is. Compaction (if needed) is handled
        by the caller when the summary exceeds a threshold.
        """
        if not turn_summary or not turn_summary.strip():
            return
        with self._lock:
            session = self._cache.get(session_id)
            if session is None:
                session = self._load_from_backend(session_id)
            if session and not session.is_expired:
                if session.conversation_summary:
                    session.conversation_summary += "\n" + turn_summary.strip()
                else:
                    session.conversation_summary = turn_summary.strip()
                self._persist(session)

    def set_summary(self, session_id: str, summary: str) -> None:
        """Replace the conversation summary (used after compaction)."""
        with self._lock:
            session = self._cache.get(session_id)
            if session is None:
                session = self._load_from_backend(session_id)
            if session and not session.is_expired:
                session.conversation_summary = summary
                self._persist(session)

    def get_remember_context(self, session_id: str) -> dict[str, Any]:
        """Get remember pipeline context for prompt building.

        Returns:
            Dict with 'extracted_memories' (list[str]) and
            'conversation_summary' (str). Empty values if session
            is missing or expired.
        """
        with self._lock:
            session = self._cache.get(session_id)
            if session is None:
                session = self._load_from_backend(session_id)
            if session and not session.is_expired:
                return {
                    "extracted_memories": list(session.extracted_memories),
                    "conversation_summary": session.conversation_summary,
                }
            return {"extracted_memories": [], "conversation_summary": ""}

    @property
    def active_count(self) -> int:
        """Number of active (non-expired) sessions.

        Returns the count from the in-memory cache. For the memory
        backend this is the complete count. For persistent backends,
        this may undercount sessions not yet loaded into cache, but
        is accurate for monitoring purposes (active sessions are
        always in cache after first access).
        """
        with self._lock:
            return sum(1 for s in self._cache.values() if not s.is_expired)

    def start_cleanup_task(self) -> None:
        """Start periodic background sweep for expired sessions.

        Safe to call multiple times — only starts one task.
        Must be called from within an async context (event loop running).
        """
        if self._sweep_task is not None:
            return
        self._sweep_task = asyncio.create_task(self._sweep_loop())
        logger.info("Session cleanup task started (interval=%ds)", self._sweep_interval)

    async def stop_cleanup_task(self) -> None:
        """Stop the background sweep task."""
        if self._sweep_task is not None:
            self._sweep_task.cancel()
            try:
                await self._sweep_task
            except asyncio.CancelledError:
                pass
            self._sweep_task = None
            logger.info("Session cleanup task stopped")

    async def _sweep_loop(self) -> None:
        """Periodically remove expired sessions."""
        while True:
            await asyncio.sleep(self._sweep_interval)
            removed = self._sweep()
            if removed > 0:
                logger.info(
                    "Session sweep: removed %d expired sessions (%d active)",
                    removed,
                    self.active_count,
                )

    def _sweep(self) -> int:
        """Remove expired sessions from cache and backend. Returns count removed."""
        with self._lock:
            # Sweep cache
            expired = [sid for sid, s in self._cache.items() if s.is_expired]
            for sid in expired:
                del self._cache[sid]

        # Sweep backend (outside lock — backend has its own thread safety).
        # Backend removes the same expired sessions plus any that were
        # never loaded into cache, so use max() to avoid double-counting.
        try:
            backend_removed = self._backend.sweep_expired()
        except Exception:
            logger.warning("Backend sweep failed", exc_info=True)
            backend_removed = 0

        return max(len(expired), backend_removed)

    def close(self) -> None:
        """Clean up backend resources."""
        try:
            self._backend.close()
        except Exception:
            pass
