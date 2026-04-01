"""
Shared helpers: prompt-injection escaping, system-text builder,
session store, and conversation extraction.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Prompt injection safety
# ---------------------------------------------------------------------------


def escape_for_prompt(text: str) -> str:
    """HTML-entity escape for memory content injected into prompts."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


# ---------------------------------------------------------------------------
# System text builder
# ---------------------------------------------------------------------------


def build_system_text(recall_result: dict[str, Any]) -> str:
    """Build the context string to inject from a recall API response.

    The *recall_result* dict is the JSON body returned by ``POST /api/recall``.
    """
    parts: list[str] = []

    instructions = recall_result.get("instructions")
    if instructions:
        parts.append(instructions)

    core = recall_result.get("core_memories")
    if core:
        parts.append(core)

    search_results = recall_result.get("search_results") or []
    memories = [
        f"- {escape_for_prompt(r['memory'])}"
        for r in search_results
        if r.get("memory", "").strip()
    ]
    if memories:
        parts.append(f"## Recalled Memories\n{chr(10).join(memories)}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

MAX_SESSIONS = 100


@dataclass
class SessionState:
    """Per-session state tracked by the plugin."""

    mnemory_session_id: str | None = None
    """Mnemory-side session ID (returned by ``/api/recall``)."""

    last_history_len: int = 0
    """Length of ``conversation_history`` on the previous turn (for compaction detection)."""

    turn_count: int = 0
    """Conversation turn counter (0 = first turn).  Used to select search mode."""

    init_result: dict[str, Any] | None = None
    """Cached init recall result (instructions + core_memories)."""

    init_event: threading.Event = field(default_factory=threading.Event)
    """Signalled when the background init-recall completes."""

    last_search_results: list[dict[str, Any]] | None = None
    """Per-turn search results.  Replaced (not accumulated) each turn."""


class SessionStore:
    """In-memory session store with FIFO eviction."""

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._max = max_sessions
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> SessionState:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is not None:
                return state

            # Evict oldest entries if at capacity
            while len(self._sessions) >= self._max:
                oldest = next(iter(self._sessions))
                del self._sessions[oldest]

            state = SessionState()
            self._sessions[session_id] = state
            return state

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


# ---------------------------------------------------------------------------
# Conversation extraction
# ---------------------------------------------------------------------------


def extract_last_exchange(
    conversation_history: list[dict[str, Any]],
    after_index: int,
    include_assistant: bool,
) -> dict[str, Any] | None:
    """Extract the last user + assistant exchange from *conversation_history*.

    Returns ``{"user": str, "assistant": str | None, "new_count": int}``
    or ``None`` if no new user message is found.
    """
    sliced = conversation_history[after_index:]
    if not sliced:
        return None

    last_user: str | None = None
    last_assistant: str | None = None

    for msg in sliced:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = _extract_text(msg.get("content"))
        if not content:
            continue

        if role == "user":
            last_user = content
        elif role == "assistant" and include_assistant:
            last_assistant = content

    if not last_user:
        return None

    return {
        "user": last_user,
        "assistant": last_assistant,
        "new_count": len(conversation_history),
    }


def _extract_text(content: Any) -> str | None:
    """Extract plain text from message content (string or list-of-blocks)."""
    if isinstance(content, str):
        return content.strip() or None

    if isinstance(content, list):
        # Take the last text block (captures conclusions, not narration)
        last_text: str | None = None
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = (block.get("text") or "").strip()
                if text:
                    last_text = text
        return last_text

    return None
