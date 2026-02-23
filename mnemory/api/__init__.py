"""REST API sub-application for mnemory.

Provides OpenAPI-documented endpoints for memory operations, mounted
at /api/ alongside the MCP server at /mcp. Auto-generates OpenAPI spec
at /api/openapi.json and Swagger UI at /api/docs.
"""

from __future__ import annotations

from fastapi import FastAPI

from mnemory import __version__
from mnemory.session import SessionStore

# Module-level session store — shared across all API endpoints.
# Initialized with defaults; reconfigured in create_api_app() from config.
_session_store = SessionStore()


def create_api_app() -> FastAPI:
    """Create the FastAPI sub-application for REST API.

    Routers are imported here (not at module level) to avoid circular
    imports — route handlers reference server.py globals.
    """
    global _session_store

    # Reconfigure session store from config
    from mnemory.server import _get_config

    cfg = _get_config()
    _session_store = SessionStore(
        default_ttl=cfg.memory.memory_session_ttl,
        sweep_interval=cfg.memory.memory_session_sweep_interval,
    )

    app = FastAPI(
        title="mnemory",
        description="Persistent memory for AI agents — REST API",
        version=__version__,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    from mnemory.api.memories import categories_router
    from mnemory.api.memories import router as memories_router
    from mnemory.api.recall import router as recall_router
    from mnemory.api.remember import router as remember_router

    app.include_router(memories_router, prefix="/memories", tags=["memories"])
    app.include_router(categories_router, prefix="/categories", tags=["categories"])
    app.include_router(recall_router, tags=["intelligence"])
    app.include_router(remember_router, tags=["intelligence"])

    return app
