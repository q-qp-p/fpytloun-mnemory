"""MCP server for mnemory.

Exposes 12 tools over Streamable HTTP for memory management:
- add_memory, search_memories, get_core_memories, list_memories
- update_memory, delete_memory, delete_all_memories
- list_categories
- save_artifact, get_artifact, list_artifacts, delete_artifact

Includes a /health endpoint and optional API key authentication.
"""

from __future__ import annotations

import contextlib
import hmac
import json
import logging
import os
import sys

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from mnemory import __version__
from mnemory.config import load_config
from mnemory.instructions import SERVER_INSTRUCTIONS
from mnemory.memory import MemoryService

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mnemory")

# ── Configuration ─────────────────────────────────────────────────────
# Lazy initialization: config and service are created on first access
# to avoid import-time side effects (env var requirements, backend connections).

_config = None
_service = None


def _get_config():
    global _config
    if _config is None:
        _config = load_config()
    return _config


def _get_service():
    global _service
    if _service is None:
        _service = MemoryService(_get_config())
    return _service


# ── MCP Server ────────────────────────────────────────────────────────

mcp = FastMCP(
    "mnemory",
    instructions=SERVER_INSTRUCTIONS,
    stateless_http=True,
    json_response=True,
    # Disable DNS rebinding protection — mnemory runs behind a reverse
    # proxy / ingress that handles host validation and TLS termination.
    # Without this, the MCP SDK rejects requests with external Host
    # headers (e.g., mem.fpy.cz) with HTTP 421.
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
)


# ── Tool 1: add_memory ───────────────────────────────────────────────


@mcp.tool()
def add_memory(
    content: str,
    user_id: str,
    memory_type: str = "fact",
    categories: list[str] | None = None,
    importance: str = "normal",
    pinned: bool = False,
    agent_id: str | None = None,
) -> str:
    """Store a memory about the user or agent.

    Call this whenever the user shares personal information, preferences,
    facts, decisions, project context, or anything worth remembering.

    Content must be concise (max 1000 chars). For detailed content, store
    a summary here and attach the full content with save_artifact.

    Args:
        content: The memory to store. Keep concise — conclusions, not raw data.
        user_id: User identifier (required).
        memory_type: One of: preference, fact, episodic, procedural, context.
        categories: Tags from the PREDEFINED set only. Do NOT invent categories.
                    Valid: personal, preferences, health, work, technical,
                    finance, home, vehicles, travel, entertainment, goals,
                    decisions, project. Use project:<name> for project-specific.
                    Call list_categories to see the full list.
        importance: low, normal, high, or critical. Affects search ranking.
        pinned: If true, this memory loads at every conversation start via
                get_core_memories. Use for essential facts and identity.
        agent_id: ONLY for memories specific to this agent (your identity,
                  personality, agent-specific knowledge). Do NOT set for user
                  facts, preferences, or context — those must be shared across
                  all agents. Memories with agent_id are INVISIBLE to other
                  agents. When in doubt, leave empty.
    """
    try:
        result = _get_service().add_memory(
            content,
            user_id=user_id,
            agent_id=agent_id,
            memory_type=memory_type,
            categories=categories,
            importance=importance,
            pinned=pinned,
        )
        return json.dumps(result, default=str)
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in add_memory")
        return json.dumps(
            {"error": True, "message": "Internal error processing add_memory"}
        )


# ── Tool 2: search_memories ──────────────────────────────────────────


@mcp.tool()
def search_memories(
    query: str,
    user_id: str,
    memory_type: str | None = None,
    categories: list[str] | None = None,
    limit: int = 10,
    agent_id: str | None = None,
) -> str:
    """Search memories by semantic similarity with filtering and importance reranking.

    Call this when you need to recall information about the user — their
    preferences, past interactions, facts, or project context. Results
    are ranked by relevance and importance. Memories with artifacts show
    has_artifacts: true — use get_artifact to fetch details.

    Args:
        query: What to search for (natural language).
        user_id: User identifier (required).
        memory_type: Filter by type (preference/fact/episodic/procedural/context).
        categories: Filter by categories. "project" matches all project:* entries.
        limit: Max results to return (default 10).
        agent_id: Only set to find YOUR agent-specific memories. Omit to
                  search shared user memories (the common case). Setting this
                  restricts results to memories stored with this agent_id only.
    """
    try:
        results = _get_service().search_memories(
            query,
            user_id=user_id,
            agent_id=agent_id,
            memory_type=memory_type,
            categories=categories,
            limit=limit,
        )
        return _format_memories(results)
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in search_memories")
        return json.dumps(
            {"error": True, "message": "Internal error processing search_memories"}
        )


# ── Tool 3: get_core_memories ────────────────────────────────────────


@mcp.tool()
def get_core_memories(
    user_id: str,
    agent_id: str | None = None,
    recent_hours: int = 24,
) -> str:
    """Load essential context at the start of every conversation.

    Returns pinned memories organized by scope:
    - Agent Identity: who you are, how you behave (if agent_id set)
    - Agent Knowledge: things you've learned and researched (if agent_id set)
    - User Facts: critical information about the user
    - User Preferences: how the user likes to interact
    - Recent Context: activity from the last N hours (chronological)

    IMPORTANT: Call this ONCE at the beginning of each new conversation.

    Args:
        user_id: User identifier (required).
        agent_id: Your agent identifier (if you have one). Loads agent-specific
                  identity and knowledge memories.
        recent_hours: How many hours back to include recent context (default 24).
    """
    try:
        return _get_service().get_core_memories(
            user_id=user_id,
            agent_id=agent_id,
            recent_hours=recent_hours,
        )
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in get_core_memories")
        return json.dumps(
            {"error": True, "message": "Internal error processing get_core_memories"}
        )


# ── Tool 4: list_memories ────────────────────────────────────────────


@mcp.tool()
def list_memories(
    user_id: str,
    memory_type: str | None = None,
    categories: list[str] | None = None,
    limit: int = 50,
    agent_id: str | None = None,
) -> str:
    """List all stored memories for a user, optionally filtered.

    Use this to review what's known about a user, find memories to
    update or delete, or browse by category/type.

    Args:
        user_id: User identifier (required).
        memory_type: Filter by type (preference/fact/episodic/procedural/context).
        categories: Filter by categories.
        limit: Max results (default 50).
        agent_id: Filter to agent-specific memories only.
    """
    try:
        results = _get_service().list_memories(
            user_id=user_id,
            agent_id=agent_id,
            memory_type=memory_type,
            categories=categories,
            limit=limit,
        )
        return _format_memories(results)
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in list_memories")
        return json.dumps(
            {"error": True, "message": "Internal error processing list_memories"}
        )


# ── Tool 5: update_memory ────────────────────────────────────────────


@mcp.tool()
def update_memory(
    memory_id: str,
    content: str | None = None,
    memory_type: str | None = None,
    categories: list[str] | None = None,
    importance: str | None = None,
    pinned: bool | None = None,
) -> str:
    """Update an existing memory's content or metadata.

    Use when information has changed (user moved, changed job, updated
    a preference) or to correct/recategorize a memory.

    Args:
        memory_id: ID of the memory to update.
        content: New content text (if changing). Max 1000 chars.
        memory_type: New type (preference/fact/episodic/procedural/context).
        categories: New categories (replaces existing).
        importance: New importance level (low/normal/high/critical).
        pinned: New pinned state (true/false).
    """
    try:
        result = _get_service().update_memory(
            memory_id,
            content=content,
            memory_type=memory_type,
            categories=categories,
            importance=importance,
            pinned=pinned,
        )
        return json.dumps(result, default=str)
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in update_memory")
        return json.dumps(
            {"error": True, "message": "Internal error processing update_memory"}
        )


# ── Tool 6: delete_memory ────────────────────────────────────────────


@mcp.tool()
def delete_memory(memory_id: str, user_id: str) -> str:
    """Delete a specific memory and all its artifacts.

    Use when information is no longer relevant or the user explicitly
    asks to forget something.

    Args:
        memory_id: ID of the memory to delete.
        user_id: User identifier (needed to locate artifacts).
    """
    try:
        result = _get_service().delete_memory(memory_id, user_id=user_id)
        return json.dumps(result)
    except Exception:
        logger.exception("Error in delete_memory")
        return json.dumps(
            {"error": True, "message": "Internal error processing delete_memory"}
        )


# ── Tool 7: delete_all_memories ──────────────────────────────────────


@mcp.tool()
def delete_all_memories(user_id: str, agent_id: str | None = None) -> str:
    """Delete ALL memories for a user. Only use when explicitly requested.

    This is destructive and cannot be undone.

    Args:
        user_id: User identifier (required).
        agent_id: If set, only delete memories for this agent scope.
    """
    try:
        result = _get_service().delete_all_memories(user_id=user_id, agent_id=agent_id)
        return json.dumps(result)
    except Exception:
        logger.exception("Error in delete_all_memories")
        return json.dumps(
            {"error": True, "message": "Internal error processing delete_all_memories"}
        )


# ── Tool 8: list_categories ──────────────────────────────────────────


@mcp.tool()
def list_categories(user_id: str) -> str:
    """List all available memory categories with descriptions and counts.

    Shows predefined categories and any dynamic project:<name> categories.
    ALWAYS call this before tagging memories with categories. Categories
    are PREDEFINED — do not use categories not in this list. Use
    project:<name> for project-specific scoping.

    Args:
        user_id: User identifier (required).
    """
    try:
        result = _get_service().list_categories(user_id=user_id)
        lines = []
        for cat in result["categories"]:
            count = cat["count"]
            marker = f" ({count})" if count > 0 else ""
            lines.append(f"  {cat['name']}{marker} — {cat['description']}")
        header = f"Categories ({result['total_memories']} total memories):"
        return header + "\n" + "\n".join(lines)
    except Exception:
        logger.exception("Error in list_categories")
        return json.dumps(
            {"error": True, "message": "Internal error processing list_categories"}
        )


# ── Tool 9: save_artifact ────────────────────────────────────────────


@mcp.tool()
def save_artifact(
    memory_id: str,
    user_id: str,
    content: str,
    filename: str = "note.md",
    content_type: str = "text/markdown",
) -> str:
    """Attach an artifact to a memory (slow memory tier).

    Use for detailed content too long for fast memory — research reports,
    analysis, logs, notes, code, data. The fast memory holds the searchable
    summary; the artifact holds the full details.

    For text content, pass the content directly. For binary content (images,
    PDFs), pass base64-encoded content and set the appropriate content_type.

    Max size: 100KB per artifact.

    Args:
        memory_id: ID of the parent fast memory.
        user_id: User identifier (required).
        content: Text content or base64-encoded binary content.
        filename: Name for the artifact (default: note.md).
        content_type: MIME type (default: text/markdown). Use appropriate
                      type for binary content (image/png, application/pdf, etc.).
    """
    try:
        result = _get_service().save_artifact(
            memory_id,
            user_id=user_id,
            content=content,
            filename=filename,
            content_type=content_type,
        )
        return json.dumps(result, default=str)
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in save_artifact")
        return json.dumps(
            {"error": True, "message": "Internal error processing save_artifact"}
        )


# ── Tool 10: get_artifact ────────────────────────────────────────────


@mcp.tool()
def get_artifact(
    memory_id: str,
    artifact_id: str,
    user_id: str,
    offset: int = 0,
    limit: int = 5000,
) -> str:
    """Retrieve artifact content with pagination.

    For large artifacts, use offset to read in chunks. Response includes
    total_size and has_more to indicate if there's more content.

    Args:
        memory_id: ID of the parent fast memory.
        artifact_id: ID of the artifact to retrieve.
        user_id: User identifier (required).
        offset: Character offset for text, byte offset for binary (default 0).
        limit: Max characters/bytes to return (default 5000).
    """
    try:
        result = _get_service().get_artifact(
            memory_id,
            artifact_id,
            user_id=user_id,
            offset=offset,
            limit=limit,
        )
        return json.dumps(result, default=str)
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in get_artifact")
        return json.dumps(
            {"error": True, "message": "Internal error processing get_artifact"}
        )


# ── Tool 11: list_artifacts ──────────────────────────────────────────


@mcp.tool()
def list_artifacts(memory_id: str, user_id: str) -> str:
    """List all artifacts attached to a memory.

    Returns id, filename, content_type, size, and created_at for each.

    Args:
        memory_id: ID of the parent fast memory.
        user_id: User identifier (required).
    """
    try:
        artifacts = _get_service().list_artifacts(memory_id, user_id=user_id)
        if not artifacts:
            return json.dumps({"artifacts": [], "message": "No artifacts found."})
        return json.dumps({"artifacts": artifacts})
    except Exception:
        logger.exception("Error in list_artifacts")
        return json.dumps(
            {"error": True, "message": "Internal error processing list_artifacts"}
        )


# ── Tool 12: delete_artifact ─────────────────────────────────────────


@mcp.tool()
def delete_artifact(memory_id: str, artifact_id: str, user_id: str) -> str:
    """Delete an artifact from a memory.

    Args:
        memory_id: ID of the parent fast memory.
        artifact_id: ID of the artifact to delete.
        user_id: User identifier (required).
    """
    try:
        result = _get_service().delete_artifact(memory_id, artifact_id, user_id=user_id)
        return json.dumps(result)
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in delete_artifact")
        return json.dumps(
            {"error": True, "message": "Internal error processing delete_artifact"}
        )


# ── Helpers ───────────────────────────────────────────────────────────


def _format_memories(memories: list[dict]) -> str:
    """Format memory results for LLM consumption."""
    if not memories:
        return json.dumps({"results": [], "count": 0})

    formatted = []
    for mem in memories:
        entry: dict = {
            "id": mem.get("id"),
            "memory": mem.get("memory", ""),
            "created_at": mem.get("created_at"),
        }
        # Include score if present (from search)
        if "score" in mem:
            entry["score"] = round(mem["score"], 4)

        metadata = mem.get("metadata", {})
        if metadata.get("memory_type"):
            entry["type"] = metadata["memory_type"]
        if metadata.get("categories"):
            entry["categories"] = metadata["categories"]
        if metadata.get("importance") and metadata["importance"] != "normal":
            entry["importance"] = metadata["importance"]
        if metadata.get("pinned"):
            entry["pinned"] = True
        if metadata.get("artifacts"):
            entry["has_artifacts"] = True
            entry["artifact_count"] = len(metadata["artifacts"])

        if mem.get("agent_id"):
            entry["agent_id"] = mem["agent_id"]

        formatted.append(entry)

    return json.dumps({"results": formatted, "count": len(formatted)}, default=str)


# ── HTTP Application ──────────────────────────────────────────────────


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Optional API key authentication middleware."""

    async def dispatch(self, request: Request, call_next):
        # Skip auth for health checks
        if request.url.path == "/health":
            return await call_next(request)

        api_key = _get_config().server.api_key
        if not api_key:
            # No API key configured — auth disabled
            return await call_next(request)

        # Check Authorization header (Bearer token) or X-API-Key header
        auth_header = request.headers.get("authorization", "")
        x_api_key = request.headers.get("x-api-key", "")

        token = ""
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:]
        elif x_api_key:
            token = x_api_key

        if not hmac.compare_digest(token, api_key):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

        return await call_next(request)


async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint for Kubernetes probes."""
    cfg = _get_config()
    return JSONResponse(
        {
            "status": "healthy",
            "service": "mnemory",
            "version": __version__,
            "vector_backend": cfg.vector.backend,
            "artifact_backend": cfg.artifact.backend,
        }
    )


@contextlib.asynccontextmanager
async def lifespan(app):
    """Application lifespan manager."""
    cfg = _get_config()
    logger.info(
        "mnemory starting (vector=%s, artifact=%s, port=%d)",
        cfg.vector.backend,
        cfg.artifact.backend,
        cfg.server.port,
    )
    # Eagerly initialize the service so startup failures are caught early
    _get_service()
    async with mcp.session_manager.run():
        yield
    logger.info("mnemory shutting down")


def create_app() -> Starlette:
    """Create the ASGI application."""
    middleware = [Middleware(APIKeyMiddleware)]

    return Starlette(
        routes=[
            Route("/health", health_check),
            Mount("/", app=mcp.streamable_http_app()),
        ],
        middleware=middleware,
        lifespan=lifespan,
    )


app = create_app()


def main():
    """Run the server with uvicorn."""
    import uvicorn

    cfg = _get_config()
    uvicorn.run(
        "mnemory.server:app",
        host=cfg.server.host,
        port=cfg.server.port,
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
