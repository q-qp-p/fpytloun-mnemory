"""REST API endpoints for the management UI.

Provides /whoami for session identity and /stats for metrics data
in a JSON format suitable for the dashboard.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["ui"])


@router.get("/whoami")
async def whoami() -> dict:
    """Return the resolved session identity.

    Used by the management UI to verify authentication and determine
    whether user switching is allowed (wildcard API keys allow switching,
    non-wildcard keys are hard-bound to a specific user).

    Unlike other endpoints, this does NOT require user_id — wildcard
    API keys may not have a user bound yet. The UI uses this to decide
    whether to show the user switcher.
    """
    from mnemory.server import (
        _session_agent_id,
        _session_timezone,
        _session_user_bound,
        _session_user_id,
    )

    return {
        "user_id": _session_user_id.get(),
        "agent_id": _session_agent_id.get(),
        "timezone": _session_timezone.get(),
        "can_switch_user": not _session_user_bound.get(),
    }


@router.get("/stats")
async def stats() -> dict:
    """Return metrics data as structured JSON for the management UI dashboard.

    Reuses the MetricsCollector's cached Qdrant aggregation — no extra
    scroll. Returns totals, breakdowns by type/category/role/user, and
    operation counts.

    Does not require user_id — returns global stats for all users.
    Auth is still enforced by the middleware (valid API key required).
    """
    from mnemory.metrics import get_collector

    collector = get_collector()
    if collector is None:
        raise HTTPException(status_code=404, detail="Metrics disabled")

    return collector.get_stats_json()
