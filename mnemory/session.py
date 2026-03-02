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

        If the known set exceeds the max cap, it is trimmed by
        discarding arbitrary entries. This is harmless — worst case,
        a previously-seen memory is returned again.
        """
        if not memory_ids:
            return
        with self._lock:
            session = self._sessions.get(session_id)
            if session and not session.is_expired:
                session.known_memory_ids.update(memory_ids)
                # Cap the set size to prevent unbounded growth
                if len(session.known_memory_ids) > _MAX_KNOWN_IDS:
                    excess = len(session.known_memory_ids) - _MAX_KNOWN_IDS
                    # Discard arbitrary entries (set has no ordering)
                    it = iter(session.known_memory_ids)
                    to_remove = [next(it) for _ in range(excess)]
                    session.known_memory_ids.difference_update(to_remove)

    def get_known_ids(self, session_id: str) -> set[str]:
        """Get the set of known memory IDs for a session."""
        with self._lock:
            session = self._sessions.get(session_id)
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
            session = self._sessions.get(session_id)
            if session and not session.is_expired:
                session.extracted_memories.extend(texts)
                if len(session.extracted_memories) > max_entries:
                    session.extracted_memories = session.extracted_memories[
                        -max_entries:
                    ]

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
            session = self._sessions.get(session_id)
            if session and not session.is_expired:
                if session.conversation_summary:
                    session.conversation_summary += "\n" + turn_summary.strip()
                else:
                    session.conversation_summary = turn_summary.strip()

    def set_summary(self, session_id: str, summary: str) -> None:
        """Replace the conversation summary (used after compaction)."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session and not session.is_expired:
                session.conversation_summary = summary

    def get_remember_context(self, session_id: str) -> dict[str, Any]:
        """Get remember pipeline context for prompt building.

        Returns:
            Dict with 'extracted_memories' (list[str]) and
            'conversation_summary' (str). Empty values if session
            is missing or expired.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session and not session.is_expired:
                return {
                    "extracted_memories": list(session.extracted_memories),
                    "conversation_summary": session.conversation_summary,
                }
            return {"extracted_memories": [], "conversation_summary": ""}

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
