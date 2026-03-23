"""REST API endpoints for session summaries.

Provides endpoints to list and view persistent session summaries
stored in the _mnemory_sessions Qdrant collection.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from mnemory.api.deps import SessionContext, get_session_context

logger = logging.getLogger("mnemory")

router = APIRouter(tags=["sessions"])


def _get_service():
    from mnemory.server import _get_service

    return _get_service()


@router.get("/sessions")
async def list_sessions(
    limit: int = Query(50, ge=1, le=200),
    consolidation_state: str | None = Query(None),
    ctx: SessionContext = Depends(get_session_context),
) -> dict:
    """List persistent session summaries for the current user.

    Returns session summaries ordered by most recently updated first.
    These are conversation summaries persisted by the remember endpoint,
    used by the consolidation service to synthesize durable knowledge.
    """
    service = _get_service()

    sessions = service._session_summary_store.list_for_user(
        ctx.user_id,
        limit=limit,
        consolidation_state=consolidation_state,
    )

    return {
        "sessions": sessions,
        "total": len(sessions),
    }


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    ctx: SessionContext = Depends(get_session_context),
) -> dict:
    """Get a single session summary by ID.

    Returns the full session summary including linked memory IDs
    and consolidation state.
    """
    service = _get_service()

    session = service._session_summary_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Verify user owns this session
    if session.get("user_id") != ctx.user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    return session
