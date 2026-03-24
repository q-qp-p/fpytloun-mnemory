"""REST API endpoints for session summaries.

Provides endpoints to list and view persistent session summaries
stored in the _mnemory_sessions Qdrant collection.
"""

from __future__ import annotations

import asyncio
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


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    delete_memories: bool = Query(False),
    ctx: SessionContext = Depends(get_session_context),
) -> dict:
    """Delete a session summary and optionally its linked raw memories.

    Args:
        session_id: Session to delete.
        delete_memories: If true, also delete linked raw memories
            (not consolidated). Artifacts on deleted memories are
            cleaned up automatically.
    """
    service = _get_service()

    # Verify session exists and user owns it
    session = service._session_summary_store.get(session_id)
    if session is None or session.get("user_id") != ctx.user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    # Prevent deletion during active consolidation
    if session.get("consolidation_state") == "consolidating":
        raise HTTPException(
            status_code=409,
            detail="Cannot delete session while consolidation is in progress",
        )

    deleted_memories = 0

    # Optionally delete linked raw memories
    if delete_memories:
        memory_ids = session.get("memory_ids") or []
        for mid in memory_ids:
            try:
                # Check if memory exists and is raw before deleting
                mem = service.vector.get(mid)
                if mem is None:
                    continue
                meta = mem.get("metadata") or {}
                layer = meta.get("memory_layer", "raw")
                if layer != "raw":
                    continue
                # delete_memory handles artifact cleanup
                service.delete_memory(mid, user_id=ctx.user_id)
                deleted_memories += 1
            except Exception:
                logger.warning("Failed to delete linked memory %s", mid, exc_info=True)

    # Delete the session record
    service._session_summary_store.delete(session_id)

    return {
        "deleted_session": True,
        "deleted_memories": deleted_memories,
    }


@router.post("/sessions/{session_id}/consolidate")
async def consolidate_session_endpoint(
    session_id: str,
    ctx: SessionContext = Depends(get_session_context),
) -> dict:
    """Trigger consolidation for a specific session.

    Ignores the idle threshold — consolidates immediately.
    Returns the consolidation result.
    """
    service = _get_service()

    # Verify session exists and user owns it
    session = service._session_summary_store.get(session_id)
    if session is None or session.get("user_id") != ctx.user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check if already consolidating (race condition guard)
    if session.get("consolidation_state") == "consolidating":
        raise HTTPException(status_code=409, detail="Consolidation already in progress")

    # Reset failed sessions to idle so consolidate_session() can proceed
    if session.get("consolidation_state") == "failed":
        service._session_summary_store.update_consolidation_state(session_id, "idle")

    # Reuse the ConsolidationService from MaintenanceService
    from mnemory.server import _maintenance_service

    if _maintenance_service is None or _maintenance_service._consolidation is None:
        raise HTTPException(
            status_code=503, detail="Consolidation service not available"
        )

    # Run in thread pool to avoid blocking the event loop
    result = await asyncio.to_thread(
        _maintenance_service._consolidation.consolidate_session,
        session_id,
    )

    return {
        "session_id": result.session_id,
        "state": result.state,
        "memories_produced": result.memories_produced,
        "memories_superseded": result.memories_superseded,
        "consolidated_memory_ids": result.consolidated_memory_ids or [],
        "duration_seconds": round(result.duration_seconds, 2),
        "error": result.error,
    }
