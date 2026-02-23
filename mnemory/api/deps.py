"""FastAPI dependencies for REST API authentication and identity.

Reuses the same identity resolution as the MCP APIKeyMiddleware —
reads from contextvars set by the middleware. The middleware already
handles auth and sets session identity before the request reaches
FastAPI route handlers.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException


@dataclass
class SessionContext:
    """Session identity resolved from request headers and API key mapping."""

    user_id: str
    agent_id: str | None = None
    timezone: str | None = None


def get_session_context() -> SessionContext:
    """Extract and validate session identity from request context.

    The APIKeyMiddleware has already authenticated the request and set
    contextvars for user_id, agent_id, and timezone. This dependency
    reads those values and validates that user_id is present.

    Resolution priority for user_id:
    1. API key mapping (non-wildcard) — most secure
    2. X-User-Id header
    3. X-OpenWebUI-User-Email header

    Agent ID from X-Agent-Id header.
    Timezone from X-Timezone header.

    Raises:
        HTTPException(401): If user_id cannot be resolved.
    """
    from mnemory.server import _session_agent_id, _session_timezone, _session_user_id

    user_id = _session_user_id.get()
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="user_id could not be resolved. Set it via API key mapping, "
            "X-User-Id header, or X-OpenWebUI-User-Email header.",
        )

    return SessionContext(
        user_id=user_id,
        agent_id=_session_agent_id.get(),
        timezone=_session_timezone.get(),
    )
