"""POST /api/remember — Store memories from conversation turns.

Fire-and-forget endpoint for plugins. Accepts conversation messages,
formats them as text, and runs through the extraction pipeline in
a background task. Returns immediately with {"accepted": true}.

Rate-limited per user to prevent abuse.
"""

from __future__ import annotations

import logging
import threading
import time

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from mnemory.api.deps import SessionContext, get_session_context
from mnemory.api.schemas import MessageParam, RememberRequest, RememberResponse

logger = logging.getLogger("mnemory")

router = APIRouter()

# Simple in-memory rate limiter: user_id -> list of timestamps
_rate_limits: dict[str, list[float]] = {}
_rate_lock = threading.Lock()


def _get_service():
    from mnemory.server import _get_service

    return _get_service()


def _get_session_store():
    from mnemory.api import _session_store

    return _session_store


def _check_rate_limit(user_id: str) -> bool:
    """Check if user is within rate limit. Returns True if allowed.

    Uses a sliding window of 60 seconds. Cleans up old entries
    and removes empty user keys to prevent unbounded growth.
    """
    from mnemory.server import _get_config

    limit = _get_config().memory.remember_rate_limit
    if limit <= 0:
        return True  # No limit

    now = time.monotonic()
    window = 60.0  # 1 minute

    with _rate_lock:
        timestamps = _rate_limits.get(user_id, [])
        # Remove entries outside the window
        timestamps = [t for t in timestamps if now - t < window]
        if len(timestamps) >= limit:
            _rate_limits[user_id] = timestamps
            return False
        timestamps.append(now)
        _rate_limits[user_id] = timestamps

        # Periodic cleanup: remove empty user entries to prevent memory leak.
        # Only run when the dict grows large enough to matter.
        if len(_rate_limits) > 100:
            empty_keys = [k for k, v in _rate_limits.items() if not v]
            for k in empty_keys:
                del _rate_limits[k]

        return True


def _format_messages(messages: list[MessageParam]) -> str:
    """Format OpenAI-style messages into text for extraction.

    Example output:
        User: I just moved to Berlin
        Assistant: That's exciting! How do you like it?
    """
    parts = []
    for msg in messages:
        content = msg.content
        if not content:
            continue
        label = msg.role.capitalize()
        parts.append(f"{label}: {content}")
    return "\n".join(parts)


def _process_remember(
    content: str,
    user_id: str,
    agent_id: str | None,
    session_id: str | None,
    timezone: str | None,
) -> None:
    """Background task: extract and store memories from conversation content.

    All exceptions are caught and logged — never propagated.
    """
    try:
        service = _get_service()
        result = service.remember(
            content=content,
            user_id=user_id,
            agent_id=agent_id,
            session_timezone=timezone,
        )

        # Update session with stored memory IDs (prevents echo on next recall)
        if session_id and result.get("results"):
            stored_ids = {r["id"] for r in result["results"] if r.get("id")}
            if stored_ids:
                try:
                    session_store = _get_session_store()
                    session_store.add_known_ids(session_id, stored_ids)
                except Exception:
                    logger.warning(
                        "Failed to update session %s with stored IDs",
                        session_id,
                    )

        stored_count = len(result.get("results", []))
        if stored_count > 0:
            logger.info(
                "Remember: stored %d memories for user=%s",
                stored_count,
                user_id,
            )

    except Exception:
        logger.warning(
            "Remember background task failed for user=%s",
            user_id,
            exc_info=True,
        )


@router.post("/remember", response_model=RememberResponse)
def remember(
    req: RememberRequest,
    background_tasks: BackgroundTasks,
    ctx: SessionContext = Depends(get_session_context),
):
    """Store memories from a conversation exchange (fire-and-forget).

    Accepts OpenAI-format messages (typically last 2: user + assistant),
    formats them as text, and processes asynchronously. Returns immediately.
    """
    from mnemory.metrics import get_collector

    collector = get_collector()
    if collector:
        collector.record_operation("remember", ctx.user_id, ctx.agent_id)

    # Rate limit check
    if not _check_rate_limit(ctx.user_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later.",
        )

    # Format messages to text
    content = _format_messages(req.messages)
    if not content.strip():
        return RememberResponse(accepted=True)  # Nothing to process

    # Fire-and-forget background processing
    background_tasks.add_task(
        _process_remember,
        content=content,
        user_id=ctx.user_id,
        agent_id=ctx.agent_id,
        session_id=req.session_id,
        timezone=ctx.timezone,
    )

    return RememberResponse(accepted=True)
