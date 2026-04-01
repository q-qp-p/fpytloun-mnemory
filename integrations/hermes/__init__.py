"""
Hermes Agent plugin for mnemory -- persistent two-tier memory.

Provides auto-recall (inject relevant memories before each agent turn),
auto-capture (extract and store memories after conversations), and
16 explicit memory tools (search, add, update, delete, list, artifacts).

Uses mnemory's REST API (/api/recall, /api/remember, /api/memories/*).
"""

from __future__ import annotations

import logging
from typing import Any

from .client import MnemoryClient
from .config import load_config
from .helpers import SessionStore
from .hooks import create_hooks
from .tools import register_tools

logger = logging.getLogger("hermes_mnemory")


def register(ctx: Any) -> None:
    """Hermes plugin entry point.  Called once at startup."""

    # Load configuration from environment variables
    try:
        config = load_config()
    except ValueError as exc:
        logger.error("mnemory: configuration error: %s", exc)
        raise

    # Create the HTTP client
    client = MnemoryClient(
        url=config.url,
        api_key=config.api_key,
        user_id=config.user_id,
        timeout=config.timeout,
    )

    agent_id = config.agent_prefix or None
    sessions = SessionStore()

    # Register lifecycle hooks (conditionally based on config)
    hooks = create_hooks(client, config, sessions)
    for hook_name, callback in hooks.items():
        ctx.register_hook(hook_name, callback)

    # Register all 16 memory tools
    register_tools(ctx.register_tool, client, agent_id)

    logger.info(
        "mnemory: plugin loaded (url=%s, agent_id=%s, auto_recall=%s, auto_capture=%s, %d hooks, 16 tools)",
        config.url,
        agent_id or "(none)",
        config.auto_recall,
        config.auto_capture,
        len(hooks),
    )
