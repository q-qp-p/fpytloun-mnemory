"""Server-side memory session tracking.

Tracks which memories the client already has in its context, so subsequent
recall calls only return NEW relevant memories. Prevents context bloat.

Losing a session is harmless — worst case, the client gets memories it
already has (minor duplication, not a failure).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger("mnemory")


@dataclass
class MemorySession:
    """Tracks client-side memory state on the server.

    Keeps track of which memories the client already has in its context,
    so subsequent recall calls only return NEW relevant memories.
    """

    session_id: str
    user_id: str
    agent_id: str | None
    known_memory_ids: set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_seconds: int = 3600

    @property
    def is_expired(self) -> bool:
        """Check if the session has expired (idle timeout)."""
        elapsed = (datetime.now(timezone.utc) - self.last_accessed).total_seconds()
        return elapsed > self.ttl_seconds


class SessionStore:
    """In-memory session store with periodic sweep cleanup.

    Thread-safe via a lock. Sessions expire after an idle timeout
    (no access within ttl_seconds). A background task periodically
    sweeps expired sessions to prevent memory leaks.
    """

    def __init__(self, default_ttl: int = 3600, sweep_interval: int = 300):
        self._sessions: dict[str, MemorySession] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl
        self._sweep_interval = sweep_interval
        self._sweep_task: asyncio.Task | None = None

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
            self._sessions[session.session_id] = session
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

        Expired sessions are lazily removed on access.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.is_expired:
                del self._sessions[session_id]
                logger.debug("Session expired on access: %s", session_id)
                return None
            return session

    def touch(self, session_id: str) -> None:
        """Update last_accessed timestamp to keep session alive."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session and not session.is_expired:
                session.last_accessed = datetime.now(timezone.utc)

    def add_known_ids(self, session_id: str, memory_ids: set[str]) -> None:
        """Add memory IDs to the session's known set.

        These IDs will be filtered out on subsequent recall calls
        to avoid returning memories the client already has.
        """
        if not memory_ids:
            return
        with self._lock:
            session = self._sessions.get(session_id)
            if session and not session.is_expired:
                session.known_memory_ids.update(memory_ids)

    def get_known_ids(self, session_id: str) -> set[str]:
        """Get the set of known memory IDs for a session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session and not session.is_expired:
                return set(session.known_memory_ids)
            return set()

    @property
    def active_count(self) -> int:
        """Number of active (non-expired) sessions."""
        with self._lock:
            return sum(1 for s in self._sessions.values() if not s.is_expired)

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
        """Remove expired sessions. Returns count removed."""
        with self._lock:
            expired = [sid for sid, s in self._sessions.items() if s.is_expired]
            for sid in expired:
                del self._sessions[sid]
            return len(expired)
