"""Memory CRUD endpoints for the REST API.

Thin wrappers around MemoryService methods. FastAPI handles sync→async
automatically for sync route functions.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from mnemory.api.deps import SessionContext, get_session_context
from mnemory.api.schemas import (
    AddMemoriesBatchRequest,
    AddMemoryRequest,
    AddMemoryResponse,
    CoreMemoriesResponse,
    FindMemoriesRequest,
    ListMemoriesResponse,
    RecentMemoriesResponse,
    SaveArtifactRequest,
    SearchMemoriesRequest,
    SearchMemoriesResponse,
    UpdateMemoryRequest,
    format_memory_item,
)

logger = logging.getLogger("mnemory")

router = APIRouter()


def _get_service():
    """Get the MemoryService singleton (lazy import to avoid circular deps)."""
    from mnemory.server import _get_service

    return _get_service()


# ── Memory CRUD ───────────────────────────────────────────────────────


@router.post("", response_model=AddMemoryResponse)
def add_memory(
    req: AddMemoryRequest,
    ctx: SessionContext = Depends(get_session_context),
):
    """Add a single memory with optional LLM-based fact extraction."""
    service = _get_service()
    try:
        result = service.add_memory(
            content=req.content,
            user_id=ctx.user_id,
            agent_id=ctx.agent_id,
            memory_type=req.memory_type,
            categories=req.categories,
            importance=req.importance,
            pinned=req.pinned,
            infer=req.infer,
            role=req.role,
            ttl_days=req.ttl_days,
            event_date=req.event_date,
            session_timezone=ctx.timezone,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if result.get("error"):
        raise HTTPException(
            status_code=400, detail=result.get("message", "Unknown error")
        )

    return result


@router.post("/batch", response_model=list[AddMemoryResponse])
def add_memories_batch(
    req: AddMemoriesBatchRequest,
    ctx: SessionContext = Depends(get_session_context),
):
    """Batch-add multiple memories."""
    service = _get_service()
    results = []
    for item in req.memories:
        try:
            result = service.add_memory(
                content=item.content,
                user_id=ctx.user_id,
                agent_id=ctx.agent_id,
                memory_type=item.memory_type,
                categories=item.categories,
                importance=item.importance,
                pinned=item.pinned,
                infer=item.infer,
                role=item.role,
                ttl_days=item.ttl_days,
                event_date=item.event_date,
                session_timezone=ctx.timezone,
            )
            results.append(result)
        except Exception:
            logger.exception("Failed to add memory in batch")
            results.append(
                {"results": [], "error": True, "message": "Failed to add memory"}
            )
    return results


@router.post("/search", response_model=SearchMemoriesResponse)
def search_memories(
    req: SearchMemoriesRequest,
    ctx: SessionContext = Depends(get_session_context),
):
    """Semantic search across memories."""
    service = _get_service()
    try:
        # When session has agent_id, use dual-scope to return both
        # agent-specific and shared user memories (mirrors MCP behavior).
        if ctx.agent_id:
            results = service.search_memories_dual_scope(
                query=req.query,
                user_id=ctx.user_id,
                session_agent_id=ctx.agent_id,
                memory_type=req.memory_type,
                categories=req.categories,
                role=req.role,
                limit=req.limit,
                include_decayed=req.include_decayed,
            )
        else:
            results = service.search_memories(
                query=req.query,
                user_id=ctx.user_id,
                agent_id=None,
                memory_type=req.memory_type,
                categories=req.categories,
                role=req.role,
                limit=req.limit,
                include_decayed=req.include_decayed,
            )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    items = [format_memory_item(r) for r in results]
    return SearchMemoriesResponse(results=items)


@router.post("/find", response_model=SearchMemoriesResponse)
def find_memories(
    req: FindMemoriesRequest,
    ctx: SessionContext = Depends(get_session_context),
):
    """AI-powered multi-query search with LLM reranking."""
    service = _get_service()
    try:
        result = service.find_memories(
            question=req.question,
            user_id=ctx.user_id,
            session_agent_id=ctx.agent_id,
            agent_id=ctx.agent_id,
            memory_type=req.memory_type,
            categories=req.categories,
            role=req.role,
            limit=req.limit,
            include_decayed=req.include_decayed,
            session_timezone=ctx.timezone,
            context=req.context,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    items = [format_memory_item(r) for r in result.get("results", [])]
    return SearchMemoriesResponse(results=items)


@router.get("/core", response_model=CoreMemoriesResponse)
def get_core_memories(
    recent_days: int = Query(7, description="Days of recent context"),
    ctx: SessionContext = Depends(get_session_context),
):
    """Load pinned + recent context memories."""
    service = _get_service()
    text = service.get_core_memories(
        user_id=ctx.user_id,
        agent_id=ctx.agent_id,
        recent_days=recent_days,
    )
    return CoreMemoriesResponse(text=text)


@router.get("/recent", response_model=RecentMemoriesResponse)
def get_recent_memories(
    days: int = Query(7, description="Days back to look"),
    scope: str = Query("all", description="Scope: all, user, agent"),
    limit: int = Query(25, ge=1, le=100, description="Max results per scope"),
    include_decayed: bool = Query(False, description="Include expired memories"),
    ctx: SessionContext = Depends(get_session_context),
):
    """Get recent memories from the last N days."""
    service = _get_service()
    try:
        result = service.get_recent_memories(
            user_id=ctx.user_id,
            agent_id=ctx.agent_id,
            days=days,
            scope=scope,
            limit=limit,
            include_decayed=include_decayed,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return RecentMemoriesResponse(text=result)


@router.get("", response_model=ListMemoriesResponse)
def list_memories(
    memory_type: str | None = Query(None, description="Filter by type"),
    categories: str | None = Query(None, description="Comma-separated categories"),
    role: str | None = Query(None, description="Filter by role"),
    limit: int = Query(50, ge=1, le=500, description="Max results"),
    include_decayed: bool = Query(False, description="Include expired memories"),
    ctx: SessionContext = Depends(get_session_context),
):
    """List all or filtered memories."""
    service = _get_service()
    cat_list = [c.strip() for c in categories.split(",")] if categories else None
    try:
        # When session has agent_id, use dual-scope to return both
        # agent-specific and shared user memories (mirrors MCP behavior).
        if ctx.agent_id:
            results = service.list_memories_dual_scope(
                user_id=ctx.user_id,
                session_agent_id=ctx.agent_id,
                memory_type=memory_type,
                categories=cat_list,
                role=role,
                limit=limit,
                include_decayed=include_decayed,
            )
        else:
            results = service.list_memories(
                user_id=ctx.user_id,
                agent_id=None,
                memory_type=memory_type,
                categories=cat_list,
                role=role,
                limit=limit,
                include_decayed=include_decayed,
            )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    items = [format_memory_item(r) for r in results]
    return ListMemoriesResponse(results=items)


@router.put("/{memory_id}")
def update_memory(
    memory_id: str,
    req: UpdateMemoryRequest,
    ctx: SessionContext = Depends(get_session_context),
):
    """Update an existing memory's content or metadata."""
    service = _get_service()
    try:
        # Verify ownership: must be own agent's memory or shared
        service.verify_memory_access(memory_id, session_agent_id=ctx.agent_id)
        result = service.update_memory(
            memory_id=memory_id,
            user_id=ctx.user_id,
            content=req.content,
            memory_type=req.memory_type,
            categories=req.categories,
            importance=req.importance,
            pinned=req.pinned,
            ttl_days=req.ttl_days,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if result.get("error"):
        raise HTTPException(
            status_code=400, detail=result.get("message", "Unknown error")
        )
    return result


@router.delete("/{memory_id}")
def delete_memory(
    memory_id: str,
    ctx: SessionContext = Depends(get_session_context),
):
    """Delete a memory and its artifacts."""
    service = _get_service()
    try:
        # Verify ownership: must be own agent's memory or shared
        service.verify_memory_access(memory_id, session_agent_id=ctx.agent_id)
        result = service.delete_memory(
            memory_id=memory_id,
            user_id=ctx.user_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if result.get("error"):
        raise HTTPException(
            status_code=400, detail=result.get("message", "Unknown error")
        )
    return result


# ── Artifacts ─────────────────────────────────────────────────────────


@router.post("/{memory_id}/artifacts")
def save_artifact(
    memory_id: str,
    req: SaveArtifactRequest,
    ctx: SessionContext = Depends(get_session_context),
):
    """Attach an artifact to a memory."""
    service = _get_service()
    try:
        result = service.save_artifact(
            memory_id=memory_id,
            user_id=ctx.user_id,
            content=req.content,
            filename=req.filename,
            content_type=req.content_type,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if result.get("error"):
        raise HTTPException(
            status_code=400, detail=result.get("message", "Unknown error")
        )
    return result


@router.get("/{memory_id}/artifacts")
def list_artifacts(
    memory_id: str,
    ctx: SessionContext = Depends(get_session_context),
):
    """List artifacts attached to a memory."""
    service = _get_service()
    try:
        result = service.list_artifacts(
            memory_id=memory_id,
            user_id=ctx.user_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return result


@router.get("/{memory_id}/artifacts/{artifact_id}")
def get_artifact(
    memory_id: str,
    artifact_id: str,
    offset: int = Query(0, description="Character/byte offset"),
    limit: int = Query(5000, description="Max characters/bytes to return"),
    ctx: SessionContext = Depends(get_session_context),
):
    """Retrieve artifact content with pagination."""
    service = _get_service()
    try:
        result = service.get_artifact(
            memory_id=memory_id,
            artifact_id=artifact_id,
            user_id=ctx.user_id,
            offset=offset,
            limit=limit,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if result.get("error"):
        raise HTTPException(
            status_code=400, detail=result.get("message", "Unknown error")
        )
    return result


@router.delete("/{memory_id}/artifacts/{artifact_id}")
def delete_artifact(
    memory_id: str,
    artifact_id: str,
    ctx: SessionContext = Depends(get_session_context),
):
    """Delete an artifact from a memory."""
    service = _get_service()
    try:
        result = service.delete_artifact(
            memory_id=memory_id,
            artifact_id=artifact_id,
            user_id=ctx.user_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if result.get("error"):
        raise HTTPException(
            status_code=400, detail=result.get("message", "Unknown error")
        )
    return result


# ── Categories ────────────────────────────────────────────────────────

categories_router = APIRouter()


@categories_router.get("", tags=["categories"])
def list_categories(
    ctx: SessionContext = Depends(get_session_context),
):
    """List all available memory categories with descriptions and counts."""
    service = _get_service()
    result = service.list_categories(user_id=ctx.user_id)
    return result
