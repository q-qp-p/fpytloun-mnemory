"""POST /api/recall — Combined initialize + search endpoint.

Designed for plugins to call on each user message. Handles session
management, core memory loading, and intelligent search with
deduplication against already-known memories.

First call (no session_id):
  - Creates MemorySession
  - Fetches core memories
  - Runs find_memories (LLM query expansion, multi-search, rerank)
  - Returns session_id + instructions + core memories + search results

Subsequent call (with session_id):
  - Looks up session (if expired/missing → treats as first call)
  - Runs search_memories (fast, no LLM)
  - Filters out already-known memory IDs
  - Returns only NEW results

Never fails for transient errors — graceful degradation chain.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends

from mnemory.api.deps import SessionContext, get_session_context
from mnemory.api.schemas import (
    RecallRequest,
    RecallResponse,
    RecallStats,
    format_memory_item,
)

logger = logging.getLogger("mnemory")

router = APIRouter()


def _get_service():
    from mnemory.server import _get_service

    return _get_service()


def _get_session_store():
    from mnemory.api import _session_store

    return _session_store


def _search_with_scope(
    service: object,
    query: str,
    ctx: SessionContext,
) -> list[dict]:
    """Search memories using dual-scope when agent_id is set.

    Mirrors the MCP search_memories behavior: when a session has an
    agent_id, use dual-scope to return both agent-specific and shared
    user memories.
    """
    if ctx.agent_id:
        return service.search_memories_dual_scope(
            query=query,
            user_id=ctx.user_id,
            session_agent_id=ctx.agent_id,
        )
    return service.search_memories(
        query=query,
        user_id=ctx.user_id,
        agent_id=None,
    )


def _extract_query(req: RecallRequest) -> str:
    """Extract search query from request — explicit query or last user message."""
    if req.query:
        return req.query
    if req.messages:
        for msg in reversed(req.messages):
            if msg.role == "user":
                return msg.content or ""
    return ""


@router.post("/recall", response_model=RecallResponse)
def recall(
    req: RecallRequest,
    ctx: SessionContext = Depends(get_session_context),
):
    """Recall memories for a conversation turn.

    First call creates a session and returns core memories + search results.
    Subsequent calls return only NEW relevant memories not yet seen.
    """
    start_time = time.monotonic()
    service = _get_service()
    session_store = _get_session_store()

    query = _extract_query(req)

    # Determine if this is a first call or subsequent
    session = None
    is_first_call = True

    if req.session_id:
        session = session_store.get(req.session_id)
        if session:
            is_first_call = False
            session_store.touch(session.session_id)

    # Create session if needed
    if session is None:
        session = session_store.create(
            user_id=ctx.user_id,
            agent_id=ctx.agent_id,
            ttl_seconds=req.ttl,
        )

    stats = RecallStats()
    response = RecallResponse(session_id=session.session_id)

    # Build instructions if requested
    if req.include_instructions:
        try:
            if req.managed:
                from mnemory.instructions import build_managed_instructions

                response.instructions = build_managed_instructions()
            elif req.instruction_mode:
                from mnemory.instructions import build_instructions

                response.instructions = build_instructions(req.instruction_mode)
            else:
                from mnemory.instructions import build_instructions
                from mnemory.server import _get_config

                mode = _get_config().server.instruction_mode
                response.instructions = build_instructions(mode)
        except Exception:
            logger.warning("Failed to build instructions", exc_info=True)

    # Load core memories on first call
    if is_first_call:
        try:
            core_text = service.get_core_memories(
                user_id=ctx.user_id,
                agent_id=ctx.agent_id,
                recent_days=req.recent_days,
            )
            if core_text:
                response.core_memories = core_text
                # Count core memories (rough: count lines starting with "- ")
                stats.core_count = core_text.count("\n- ") + (
                    1 if core_text.startswith("- ") else 0
                )
        except Exception:
            logger.warning("Failed to load core memories", exc_info=True)

    # Search for relevant memories
    if query:
        search_results: list[dict] = []
        known_ids = session_store.get_known_ids(session.session_id)

        if is_first_call and (req.search_mode or "find") == "find":
            # First call with find mode: AI-powered multi-query search
            try:
                from mnemory.server import _get_config

                max_results = _get_config().memory.recall_max_results
                result = service.find_memories(
                    question=query,
                    user_id=ctx.user_id,
                    session_agent_id=ctx.agent_id,
                    agent_id=ctx.agent_id,
                    limit=max_results,
                    session_timezone=ctx.timezone,
                )
                search_results = result.get("results", [])
            except Exception:
                logger.warning(
                    "find_memories failed, falling back to search_memories",
                    exc_info=True,
                )
                # Fallback to simple search
                try:
                    search_results = _search_with_scope(service, query, ctx)
                except Exception:
                    logger.warning("search_memories also failed", exc_info=True)
        else:
            # Subsequent call or search mode: fast vector search, no LLM
            try:
                search_results = _search_with_scope(service, query, ctx)
            except Exception:
                logger.warning("search_memories failed", exc_info=True)

        # Filter out already-known memories
        stats.search_count = len(search_results)
        new_results = []
        for r in search_results:
            rid = r.get("id", "")
            if rid and rid not in known_ids:
                new_results.append(r)
            else:
                stats.known_skipped += 1

        stats.new_count = len(new_results)

        # Add new IDs to session
        new_ids = {r["id"] for r in new_results if r.get("id")}
        if new_ids:
            session_store.add_known_ids(session.session_id, new_ids)

        response.search_results = [format_memory_item(r) for r in new_results]

    elapsed_ms = int((time.monotonic() - start_time) * 1000)
    stats.latency_ms = elapsed_ms
    response.stats = stats

    return response
