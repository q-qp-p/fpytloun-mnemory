"""MCP server for mnemory.

Exposes 16 tools over Streamable HTTP for memory management:
- initialize_memory (session init with instructions + core memories)
- add_memory, add_memories, search_memories, find_memories
- get_core_memories, get_recent_memories
- list_memories, update_memory, delete_memory, delete_all_memories
- list_categories
- save_artifact, get_artifact, list_artifacts, delete_artifact

Includes /health and /metrics endpoints, optional API key authentication,
session-level identity resolution (user_id from API key mapping,
agent_id from X-Agent-Id header), and an optional management port for
unauthenticated health/metrics access.
"""

from __future__ import annotations

import contextlib
import contextvars
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
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Mount, Route

from mnemory import __version__
from mnemory.config import load_config
from mnemory.instructions import VALID_MODES, build_instructions
from mnemory.memory import MemoryService
from mnemory.metrics import get_collector
from mnemory.sanitize import escape_memory_headers

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mnemory")

# ── Session Identity Context ──────────────────────────────────────────
# Set by the auth middleware per-request, read by tool handlers via
# _resolve_user_id / _resolve_agent_id helpers.

_session_user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "session_user_id", default=None
)
_session_agent_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "session_agent_id", default=None
)
_session_timezone: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "session_timezone", default=None
)
_session_user_bound: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "session_user_bound", default=False
)

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


_fsck_service = None


def _get_fsck_service():
    """Get or create the FsckService singleton."""
    global _fsck_service
    if _fsck_service is None:
        from mnemory.fsck import FsckService, FsckStore

        cfg = _get_config()
        service = _get_service()
        _fsck_service = FsckService(
            config=cfg,
            vector=service.vector,
            llm=service._llm,
            store=FsckStore(default_ttl=cfg.memory.fsck_cache_ttl),
        )
    return _fsck_service


# ── Identity Resolution ───────────────────────────────────────────────


def _resolve_user_id(param_user_id: str | None = None) -> str:
    """Resolve user_id from session context or tool parameter.

    Priority:
    1. API key mapping (set by middleware from MCP_API_KEYS, non-wildcard)
    2. X-User-Id header (set by middleware)
    3. Tool parameter (backward compat)
    4. Error if none available
    """
    session_uid = _session_user_id.get()
    if session_uid:
        return session_uid
    if param_user_id:
        return param_user_id
    raise ValueError(
        "user_id is required — configure via API key mapping (MCP_API_KEYS), "
        "X-User-Id header, or pass as tool parameter"
    )


def _is_sub_agent(agent_id: str, session_agent_id: str) -> bool:
    """Check if agent_id is a sub-agent of the session agent.

    Sub-agents use a colon-separated prefix: if the session agent is
    "openwebui", then "openwebui:bob" is a valid sub-agent. This allows
    creating multiple independent agent identities under a single session.
    """
    return agent_id.startswith(session_agent_id + ":")


def _resolve_agent_id(param_agent_id: str | None = None) -> str | None:
    """Resolve agent_id with session-level protection.

    Supports "self" as a sentinel value — resolves to the session's
    agent_id from X-Agent-Id header. Raises ValueError if "self" is
    used without a session agent.

    When X-Agent-Id header is set (session has an agent):
    - param="self"  → returns session agent_id
    - param=None  → returns None (shared memory, no agent scope)
    - param=session value → returns session agent_id
    - param="session:sub" → returns param (sub-agent allowed)
    - param=different value → raises ValueError (cross-agent blocked)

    When X-Agent-Id is NOT set:
    - param="self" → raises ValueError (no session agent to resolve)
    - returns param as-is (no protection, backward compatible)
    """
    session_aid = _session_agent_id.get()
    if param_agent_id == "self":
        if session_aid is None:
            raise ValueError("agent_id='self' requires X-Agent-Id header to be set")
        return session_aid
    if session_aid is None:
        # No session agent — pass through whatever the LLM sent
        return param_agent_id
    # Session agent is set — enforce protection
    if param_agent_id is None:
        return None  # LLM wants shared/no agent scope
    if param_agent_id == session_aid:
        return session_aid
    if _is_sub_agent(param_agent_id, session_aid):
        return param_agent_id
    raise ValueError(
        f"Cannot use agent_id '{param_agent_id}' — "
        f"session is bound to agent '{session_aid}'"
    )


def _resolve_agent_id_for_core(
    param_agent_id: str | None = None,
) -> str | None:
    """Resolve agent_id for get_core_memories (auto-injects session agent).

    Unlike _resolve_agent_id, this auto-injects the session agent_id
    when the LLM doesn't pass one, because get_core_memories always
    needs the agent_id to load agent identity and knowledge.

    Supports "self" sentinel and sub-agents — same as _resolve_agent_id.
    """
    session_aid = _session_agent_id.get()
    if param_agent_id == "self":
        if session_aid is None:
            raise ValueError("agent_id='self' requires X-Agent-Id header to be set")
        return session_aid
    if session_aid is None:
        return param_agent_id
    if param_agent_id is None:
        return session_aid  # Auto-inject
    if param_agent_id == session_aid:
        return session_aid
    if _is_sub_agent(param_agent_id, session_aid):
        return param_agent_id
    raise ValueError(
        f"Cannot use agent_id '{param_agent_id}' — "
        f"session is bound to agent '{session_aid}'"
    )


def _get_session_agent_id() -> str | None:
    """Get the session-level agent_id (from X-Agent-Id header), or None."""
    return _session_agent_id.get()


def _get_session_timezone() -> str | None:
    """Get the session-level timezone (from X-Timezone header), or None."""
    return _session_timezone.get()


# ── MCP Server ────────────────────────────────────────────────────────

mcp = FastMCP(
    "mnemory",
    instructions=build_instructions(os.environ.get("INSTRUCTION_MODE", "proactive")),
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


# ── Tool 0: initialize_memory ────────────────────────────────────────


@mcp.tool()
def initialize_memory(
    user_id: str | None = None,
    agent_id: str | None = None,
    recent_days: int = 7,
    mode: str | None = None,
) -> str:
    """ALWAYS call this at the start of every conversation to initialize memory.

    Returns behavioral instructions for using memory tools effectively,
    followed by your core memories (pinned facts, identity, recent context).

    This is the recommended entry point for clients that do not inject MCP
    server instructions into the system prompt (e.g., Open WebUI). If your
    client DOES inject MCP server instructions (e.g., Claude Code, Cursor),
    you can call get_core_memories directly instead to avoid redundant
    instructions.

    Args:
        user_id: User identifier. Optional if pre-configured via API key mapping.
        agent_id: Your agent identifier. Optional if pre-configured via
                  X-Agent-Id header.
        recent_days: How many days back to include recent context (default 7).
        mode: Override instruction mode. One of: passive, proactive, personality.
              If omitted, uses the server's INSTRUCTION_MODE setting.
              - passive: Use memory when asked or clearly relevant
              - proactive: Always search, proactively store, memory-first
              - personality: Proactive + identity development for agents with
                evolving personality
    """
    # Validate mode if provided
    if mode is not None and mode not in VALID_MODES:
        return json.dumps(
            {
                "error": True,
                "message": f"Invalid mode: '{mode}'. Must be one of: {', '.join(VALID_MODES)}",
            }
        )

    # Determine effective mode
    effective_mode = mode if mode is not None else _get_config().server.instruction_mode

    # Build instructions
    instructions = build_instructions(effective_mode)

    # Get core memories (fail gracefully)
    core_memories = None
    core_memories_error = None
    try:
        uid = _resolve_user_id(user_id)
        aid = _resolve_agent_id_for_core(agent_id)
        collector = get_collector()
        if collector:
            collector.record_operation("initialize_memory", uid, aid)
        core_memories = _get_service().get_core_memories(
            user_id=uid,
            agent_id=aid,
            recent_days=recent_days,
        )
    except ValueError as e:
        # Sanitize error message to prevent user-influenced content
        # from being injected into the LLM context via error text
        err_msg = str(e)[:200].replace("\n", " ")
        core_memories_error = err_msg
    except Exception:
        logger.exception("Error getting core memories in initialize_memory")
        core_memories_error = "Internal error loading core memories"

    # Build response
    parts = ["## MEMORY INSTRUCTIONS", "", instructions]

    if core_memories is not None:
        parts.extend(["", "---", "", "## CORE MEMORIES", "", core_memories])
    elif core_memories_error:
        parts.extend(
            [
                "",
                "---",
                "",
                "## CORE MEMORIES",
                "",
                f"(Error loading core memories: {core_memories_error})",
            ]
        )

    return "\n".join(parts)


# ── Tool 1: add_memory ───────────────────────────────────────────────


@mcp.tool()
def add_memory(
    content: str,
    user_id: str | None = None,
    memory_type: str | None = None,
    categories: list[str] | None = None,
    importance: str | None = None,
    pinned: bool | None = None,
    agent_id: str | None = None,
    infer: bool = True,
    role: str = "user",
    ttl_days: int | None = None,
    event_date: str | None = None,
) -> str:
    """Store a memory about the user or agent.

    Call this whenever the user shares personal information, preferences,
    facts, decisions, project context, or anything worth remembering.

    Content must be concise (max 1000 chars). For detailed content, store
    a summary here and attach the full content with save_artifact.

    When infer=True and content exceeds the max length, the server
    automatically extracts concise facts AND preserves the original as
    an artifact if the content has reference value (e.g., documents,
    code, specs). When infer=False and content exceeds the max length,
    the content is auto-saved as an artifact with a truncated memory.

    All metadata fields are OPTIONAL — if omitted, the server auto-classifies
    them using an LLM. You can provide any combination of fields; only the
    missing ones will be auto-classified.

    Args:
        content: The memory to store. Keep concise — conclusions, not raw data.
        user_id: User identifier. Optional if pre-configured via API key mapping.
        memory_type: One of: preference, fact, episodic, procedural, context.
                     Optional — auto-classified if omitted.
        categories: Tags from the PREDEFINED set only. Do NOT invent categories.
                    Valid: personal, preferences, health, work, technical,
                    finance, home, vehicles, travel, entertainment, goals,
                    decisions, project. Use project:<name> for project-specific.
                    Call list_categories to see the full list.
                    Optional — auto-classified if omitted.
        importance: low, normal, high, or critical. Affects search ranking.
                    Optional — auto-classified if omitted.
        pinned: If true, this memory loads at every conversation start via
                get_core_memories. Use for essential facts and identity.
                Optional — auto-classified if omitted.
        agent_id: For memories scoped to a specific agent. Set this for:
                  (1) agent identity/personality (with role="assistant"), or
                  (2) user preferences specific to this agent (with role="user").
                  Memories with agent_id are INVISIBLE to other agents.
                  Use "self" to resolve to your agent_id from the session.
                  Do NOT set for general user facts shared across all agents.
                  Optional if pre-configured via X-Agent-Id header.
        infer: If true (default), the server uses LLM to extract key facts
               from content and check for duplicates/contradictions with
               existing memories. If false, content is stored as-is with only
               an embedding — much faster but skips dedup. Use infer=false
               when your content is already a clean, concise fact.
               Note: infer=False is NOT allowed for role="assistant" —
               agent identity memories must go through fact extraction.
        role: Who this memory is about. "user" (default) for facts about the
              user — their preferences, personal info, context. "assistant"
              for facts about the agent itself — identity, personality,
              capabilities. When role="assistant", agent_id is required.
              Use role="assistant" with agent_id="self" for agent identity.
              Keep role="user" for everything else, including agent-scoped
              user preferences (e.g., "User wants me to create commit
              messages" with agent_id="self").
        ttl_days: Time-to-live in days. After this many days, the memory
                  decays (soft-expires). Omit to use the default TTL for the
                  memory type (fact/preference=permanent, episodic=90d,
                  procedural=60d, context=7d). Pinned memories are exempt
                  from TTL. Accessed memories have their TTL reset
                  (reinforcement).
        event_date: ISO 8601 datetime for when the event occurred (e.g.,
                    "2023-05-08T13:56:00+02:00" or "2023-05-08"). Used to
                    anchor relative time references during extraction (e.g.,
                    "yesterday" resolves to the day before event_date, not
                    today). Also stored as metadata for temporal queries.
                    Timezone priority: explicit offset in the string >
                    X-Timezone header > DEFAULT_TIMEZONE env > server local.
    """
    try:
        uid = _resolve_user_id(user_id)
        aid = _resolve_agent_id(agent_id)
        collector = get_collector()
        if collector:
            collector.record_operation("add_memory", uid, aid)
        result = _get_service().add_memory(
            content,
            user_id=uid,
            agent_id=aid,
            memory_type=memory_type,
            categories=categories,
            importance=importance,
            pinned=pinned,
            infer=infer,
            role=role,
            ttl_days=ttl_days,
            event_date=event_date,
            session_timezone=_get_session_timezone(),
        )
        return json.dumps(result, default=str)
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in add_memory")
        return json.dumps(
            {"error": True, "message": "Internal error processing add_memory"}
        )


# ── Tool 1b: add_memories (batch) ─────────────────────────────────────

MAX_BATCH_SIZE = 20


@mcp.tool()
def add_memories(
    memories: list[dict],
    user_id: str | None = None,
    agent_id: str | None = None,
    infer: bool = True,
    role: str = "user",
) -> str:
    """Store multiple memories in a single call (batch operation).

    Use this instead of calling add_memory repeatedly when you have several
    memories to store at once. Saves round-trip latency. Each memory is
    processed independently — failures on individual items do not block
    the rest.

    Args:
        memories: List of memory objects. Each must have "content" (str).
                  Optional per-item fields: memory_type, categories,
                  importance, pinned, role, ttl_days, event_date. Example:
                  [{"content": "User lives in Prague", "categories": ["personal"]},
                   {"content": "Started new job", "event_date": "2023-05-08"}]
        user_id: User identifier (shared for all items). Optional if
                 pre-configured via API key mapping.
        agent_id: Agent scope (shared for all items). Same rules as
                  add_memory — only set for agent-specific memories.
                  Use "self" to resolve to your agent_id from the session.
                  Optional if pre-configured via X-Agent-Id header.
        infer: If true (default), each memory uses LLM fact extraction
               and dedup. If false, content is stored as-is (faster).
               Applies to all items in the batch.
        role: Default role for all items ("user" or "assistant"). Can be
              overridden per item via the "role" field in each memory object.
              See add_memory for details on role semantics.
    """
    try:
        uid = _resolve_user_id(user_id)
        aid = _resolve_agent_id(agent_id)
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})

    collector = get_collector()
    if collector:
        collector.record_operation("add_memories", uid, aid)

    if not memories:
        return json.dumps({"error": True, "message": "memories list is empty"})

    if len(memories) > MAX_BATCH_SIZE:
        return json.dumps(
            {
                "error": True,
                "message": f"Too many memories: {len(memories)} (max {MAX_BATCH_SIZE})",
            }
        )

    service = _get_service()
    results = []
    errors = []

    for i, mem in enumerate(memories):
        if not isinstance(mem, dict) or "content" not in mem:
            errors.append(
                {
                    "index": i,
                    "error": True,
                    "message": "Each memory must be an object with a 'content' field",
                }
            )
            continue

        try:
            item_role = mem.get("role", role)
            result = service.add_memory(
                mem["content"],
                user_id=uid,
                agent_id=aid,
                memory_type=mem.get("memory_type"),
                categories=mem.get("categories"),
                importance=mem.get("importance"),
                pinned=mem.get("pinned"),
                infer=infer,
                role=item_role,
                ttl_days=mem.get("ttl_days"),
                event_date=mem.get("event_date"),
                session_timezone=_get_session_timezone(),
            )
            # Check for error returned by add_memory (e.g., content too long)
            if result.get("error"):
                errors.append({"index": i, **result})
            else:
                results.append({"index": i, **result})
        except ValueError as e:
            errors.append({"index": i, "error": True, "message": str(e)})
        except Exception:
            logger.exception("Error in add_memories item %d", i)
            errors.append(
                {
                    "index": i,
                    "error": True,
                    "message": f"Internal error processing item {i}",
                }
            )

    return json.dumps(
        {
            "results": results,
            "errors": errors,
            "total": len(memories),
            "succeeded": len(results),
            "failed": len(errors),
        },
        default=str,
    )


# ── Tool 2: search_memories ──────────────────────────────────────────


@mcp.tool()
def search_memories(
    query: str,
    user_id: str | None = None,
    memory_type: str | None = None,
    categories: list[str] | None = None,
    role: str | None = None,
    limit: int = 10,
    agent_id: str | None = None,
    include_decayed: bool = False,
    date_start: str | None = None,
    date_end: str | None = None,
) -> str:
    """Search memories by semantic similarity with filtering and importance reranking.

    Call this when you need to recall information about the user — their
    preferences, past interactions, facts, or project context. Results
    are ranked by relevance and importance. Memories with artifacts show
    has_artifacts: true — use get_artifact to fetch details.

    When your session has an agent_id (via X-Agent-Id header), search
    automatically returns BOTH your agent-specific memories AND shared
    user memories, merged and deduplicated. You don't need to pass
    agent_id for this.

    Args:
        query: What to search for (natural language).
        user_id: User identifier. Optional if pre-configured via API key mapping.
        memory_type: Filter by type (preference/fact/episodic/procedural/context).
        categories: Filter by categories. "project" matches all project:* entries.
        role: Filter by role — "user" for memories about the user, "assistant"
              for memories about the agent. Omit to search all roles.
        limit: Max results to return (default 10).
        agent_id: Ignored when session agent is set (automatic dual-scope).
                  Only used as fallback for direct API callers without
                  X-Agent-Id header.
        include_decayed: If true, include expired/decayed memories in results.
                        Useful for browsing historical memories. Default false.
        date_start: Filter by date range start (YYYY-MM-DD). Matches memories
                    with event_date >= date_start, or created_at >= date_start
                    when event_date is not set.
        date_end: Filter by date range end (YYYY-MM-DD). Matches memories
                  with event_date <= date_end, or created_at <= date_end
                  when event_date is not set.
    """
    try:
        uid = _resolve_user_id(user_id)
        session_aid = _get_session_agent_id()
        collector = get_collector()
        if collector:
            collector.record_operation("search_memories", uid, session_aid)

        # When session has agent_id, always use dual-scope to return
        # both agent-specific and shared user memories.
        if session_aid:
            results = _get_service().search_memories_dual_scope(
                query,
                user_id=uid,
                session_agent_id=session_aid,
                memory_type=memory_type,
                categories=categories,
                role=role,
                limit=limit,
                include_decayed=include_decayed,
                date_start=date_start,
                date_end=date_end,
            )
        else:
            # No session agent — use param for backward compat
            aid = _resolve_agent_id(agent_id)
            results = _get_service().search_memories(
                query,
                user_id=uid,
                agent_id=aid,
                memory_type=memory_type,
                categories=categories,
                role=role,
                limit=limit,
                include_decayed=include_decayed,
                date_start=date_start,
                date_end=date_end,
            )
        return _format_memories(results)
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in search_memories")
        return json.dumps(
            {"error": True, "message": "Internal error processing search_memories"}
        )


# ── Tool: find_memories ───────────────────────────────────────────────


@mcp.tool()
def find_memories(
    question: str,
    user_id: str | None = None,
    memory_type: str | None = None,
    categories: list[str] | None = None,
    role: str | None = None,
    limit: int = 10,
    agent_id: str | None = None,
    include_decayed: bool = False,
    context: str | None = None,
) -> str:
    """Find memories relevant to a complex question using AI-powered search.

    Unlike search_memories which takes a simple query and does a single
    vector search, this tool takes a full natural language question,
    generates multiple targeted searches covering different angles and
    associations, and uses AI to rank results by relevance to your
    question. Scores are on the same 0.0-1.0 scale as search_memories.

    Temporal-aware: resolves time references like "last week", "in 2023",
    "recently" into concrete date ranges for filtering. Uses event_date
    metadata for temporal scoring during reranking.

    Use this for multi-faceted questions like "What do I think about dogs?
    Should I buy one?" where a single search query wouldn't capture all
    relevant memories. The AI follows associations — e.g., for dogs it
    might also search for pets, partner, house, garden, lifestyle.

    For simple lookups, prefer search_memories (faster, no extra LLM calls).

    This tool makes 2 LLM calls (query generation + reranking) plus
    multiple vector searches, so it is slower and more expensive than
    search_memories. Use it when quality matters more than speed.

    The response includes the generated search queries under "queries"
    for transparency.

    Args:
        question: The user's question in natural language.
        user_id: User identifier. Optional if pre-configured via API key.
        memory_type: Filter by type (preference/fact/episodic/procedural/context).
        categories: Filter by categories. "project" matches all project:* entries.
        role: Filter by role — "user" or "assistant". Omit for all.
        limit: Max results to return (default 10).
        agent_id: Ignored when session agent is set (automatic dual-scope).
                  Only used as fallback for direct API callers without
                  X-Agent-Id header.
        include_decayed: If true, include expired/decayed memories in results.
                        Useful for browsing historical memories. Default false.
        context: Optional context hint for query generation (e.g., current
                 working directory, active project). Generates additional
                 relevant queries without filtering exclusively to this context.
    """
    try:
        uid = _resolve_user_id(user_id)
        session_aid = _get_session_agent_id()
        collector = get_collector()
        if collector:
            collector.record_operation("find_memories", uid, session_aid)

        session_tz = _get_session_timezone()

        if session_aid:
            result = _get_service().find_memories(
                question,
                user_id=uid,
                session_agent_id=session_aid,
                memory_type=memory_type,
                categories=categories,
                role=role,
                limit=limit,
                include_decayed=include_decayed,
                session_timezone=session_tz,
                context=context,
            )
        else:
            aid = _resolve_agent_id(agent_id)
            result = _get_service().find_memories(
                question,
                user_id=uid,
                agent_id=aid,
                memory_type=memory_type,
                categories=categories,
                role=role,
                limit=limit,
                include_decayed=include_decayed,
                session_timezone=session_tz,
                context=context,
            )
        memories = result.get("results", [])
        queries = result.get("queries", [])
        stats = result.get("stats", {})
        formatted = _format_memories(memories)
        # Inject queries and stats into the response JSON
        response = json.loads(formatted)
        response["queries"] = queries
        response["stats"] = stats
        return json.dumps(response, default=str)
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in find_memories")
        return json.dumps(
            {"error": True, "message": "Internal error processing find_memories"}
        )


# ── Tool 3: get_core_memories ────────────────────────────────────────


@mcp.tool()
def get_core_memories(
    user_id: str | None = None,
    agent_id: str | None = None,
    recent_days: int = 7,
) -> str:
    """Load essential context at the start of every conversation.

    If your client does NOT inject MCP server instructions into the system
    prompt (e.g., Open WebUI), use initialize_memory instead — it returns
    both behavioral instructions and core memories in one call.

    If your client DOES inject MCP server instructions (e.g., Claude Code,
    Cursor), call this tool directly to load memories without redundant
    instructions.

    Returns pinned memories organized by scope:
    - Agent Identity: who you are, how you behave (role=assistant, agent-scoped)
    - Agent Knowledge: things you've learned and researched (role=assistant)
    - Agent Instructions: user preferences specific to you (role=user, agent-scoped)
    - User Facts: critical information about the user
    - User Preferences: how the user likes to interact
    - Recent Context: recent activity with User/Agent subsections

    Args:
        user_id: User identifier. Optional if pre-configured via API key mapping.
        agent_id: Your agent identifier (if you have one). Loads agent-specific
                  identity and knowledge memories.
                  Optional if pre-configured via X-Agent-Id header.
        recent_days: How many days back to include recent context (default 7).
    """
    try:
        uid = _resolve_user_id(user_id)
        aid = _resolve_agent_id_for_core(agent_id)
        collector = get_collector()
        if collector:
            collector.record_operation("get_core_memories", uid, aid)
        return _get_service().get_core_memories(
            user_id=uid,
            agent_id=aid,
            recent_days=recent_days,
        )
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in get_core_memories")
        return json.dumps(
            {"error": True, "message": "Internal error processing get_core_memories"}
        )


# ── Tool 3b: get_recent_memories ─────────────────────────────────────


@mcp.tool()
def get_recent_memories(
    user_id: str | None = None,
    agent_id: str | None = None,
    days: int = 7,
    scope: str = "all",
    limit: int = 25,
    include_decayed: bool = False,
) -> str:
    """Get recent memories from the last N days.

    Use this to see recent activity without loading full core memories.
    Returns memories of all types ordered by most recent first.

    Args:
        user_id: User identifier. Optional if pre-configured via API key mapping.
        agent_id: Agent identifier. Optional if pre-configured via X-Agent-Id header.
        days: How many days back to look (default 7).
        scope: Which memories to include:
               - "all": Both user and agent memories (default)
               - "user": Only shared user memories (no agent_id)
               - "agent": Only agent-scoped memories (requires agent_id)
        limit: Max results per scope (default 25).
        include_decayed: Include expired/decayed memories (default false).
    """
    try:
        uid = _resolve_user_id(user_id)
        aid = _resolve_agent_id(agent_id)
        # Auto-resolve agent_id from session for scopes that include
        # agent memories, so the LLM doesn't have to pass it explicitly.
        if aid is None and scope in ("all", "agent"):
            aid = _session_agent_id.get()
        collector = get_collector()
        if collector:
            collector.record_operation("get_recent_memories", uid, aid)
        return _get_service().get_recent_memories(
            user_id=uid,
            agent_id=aid,
            days=days,
            scope=scope,
            limit=limit,
            include_decayed=include_decayed,
        )
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in get_recent_memories")
        return json.dumps(
            {"error": True, "message": "Internal error processing get_recent_memories"}
        )


# ── Tool 4: list_memories ────────────────────────────────────────────


@mcp.tool()
def list_memories(
    user_id: str | None = None,
    memory_type: str | None = None,
    categories: list[str] | None = None,
    role: str | None = None,
    limit: int = 50,
    agent_id: str | None = None,
    include_decayed: bool = False,
) -> str:
    """List all stored memories for a user, optionally filtered.

    Use this to review what's known about a user, find memories to
    update or delete, or browse by category/type.

    When your session has an agent_id (via X-Agent-Id header), this
    automatically returns BOTH your agent-specific memories AND shared
    user memories, merged and deduplicated. You don't need to pass
    agent_id for this.

    Args:
        user_id: User identifier. Optional if pre-configured via API key mapping.
        memory_type: Filter by type (preference/fact/episodic/procedural/context).
        categories: Filter by categories.
        role: Filter by role — "user" for memories about the user, "assistant"
              for memories about the agent. Omit to list all roles.
        limit: Max results (default 50).
        agent_id: Ignored when session agent is set (automatic dual-scope).
                  Only used as fallback for direct API callers without
                  X-Agent-Id header.
        include_decayed: If true, include expired/decayed memories in results.
                        Useful for browsing historical memories. Default false.
    """
    try:
        uid = _resolve_user_id(user_id)
        session_aid = _get_session_agent_id()
        collector = get_collector()
        if collector:
            collector.record_operation("list_memories", uid, session_aid)

        # When session has agent_id, always use dual-scope to return
        # both agent-specific and shared user memories.
        if session_aid:
            results = _get_service().list_memories_dual_scope(
                user_id=uid,
                session_agent_id=session_aid,
                memory_type=memory_type,
                categories=categories,
                role=role,
                limit=limit,
                include_decayed=include_decayed,
            )
        else:
            # No session agent — use param for backward compat
            aid = _resolve_agent_id(agent_id)
            results = _get_service().list_memories(
                user_id=uid,
                agent_id=aid,
                memory_type=memory_type,
                categories=categories,
                role=role,
                limit=limit,
                include_decayed=include_decayed,
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
    ttl_days: int | None = None,
    event_date: str | None = None,
) -> str:
    """Update an existing memory's content or metadata.

    Use when information has changed (user moved, changed job, updated
    a preference) or to correct/recategorize a memory.

    You can only update your own agent-scoped memories and shared memories.
    Updating another agent's memory will be blocked.

    Args:
        memory_id: ID of the memory to update.
        content: New content text (if changing). Max 1000 chars.
        memory_type: New type (preference/fact/episodic/procedural/context).
        categories: New categories (replaces existing).
        importance: New importance level (low/normal/high/critical).
        pinned: New pinned state (true/false).
        ttl_days: New TTL in days. Recalculates expires_at from now and
                  restores decayed memories. Pass null to make permanent.
        event_date: New event date (ISO 8601, e.g., "2023-05-08" or
                    "2023-05-08T13:56:00+02:00"). Pass null to clear.
    """
    try:
        # Verify ownership: must be own agent's memory or shared
        session_aid = _get_session_agent_id()
        collector = get_collector()
        if collector:
            collector.record_operation(
                "update_memory", _session_user_id.get() or "", session_aid
            )
        _get_service().verify_memory_access(memory_id, session_agent_id=session_aid)

        # Build kwargs, using sentinel for ttl_days to distinguish
        # "not provided" from "explicitly set to None"
        kwargs: dict = {
            "user_id": _session_user_id.get(),
            "content": content,
            "memory_type": memory_type,
            "categories": categories,
            "importance": importance,
            "pinned": pinned,
        }
        if ttl_days is not None:
            kwargs["ttl_days"] = ttl_days
        if event_date is not None:
            kwargs["event_date"] = event_date

        result = _get_service().update_memory(memory_id, **kwargs)
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
def delete_memory(memory_id: str, user_id: str | None = None) -> str:
    """Delete a specific memory and all its artifacts.

    Use when information is no longer relevant or the user explicitly
    asks to forget something.

    You can only delete your own agent-scoped memories and shared memories.
    Deleting another agent's memory will be blocked.

    Args:
        memory_id: ID of the memory to delete.
        user_id: User identifier. Optional if pre-configured via API key mapping.
    """
    try:
        uid = _resolve_user_id(user_id)
        # Verify ownership: must be own agent's memory or shared
        session_aid = _get_session_agent_id()
        collector = get_collector()
        if collector:
            collector.record_operation("delete_memory", uid, session_aid)
        _get_service().verify_memory_access(memory_id, session_agent_id=session_aid)
        result = _get_service().delete_memory(memory_id, user_id=uid)
        return json.dumps(result)
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in delete_memory")
        return json.dumps(
            {"error": True, "message": "Internal error processing delete_memory"}
        )


# ── Tool 7: delete_all_memories ──────────────────────────────────────


@mcp.tool()
def delete_all_memories(user_id: str | None = None, agent_id: str | None = None) -> str:
    """Delete ALL memories for a user. Only use when explicitly requested.

    This is destructive and cannot be undone.
    This tool is disabled by default. Set ENABLE_DELETE_ALL=true to enable.

    Args:
        user_id: User identifier. Optional if pre-configured via API key mapping.
        agent_id: If set, only delete memories for this agent scope.
                  Optional if pre-configured via X-Agent-Id header.
    """
    try:
        uid = _resolve_user_id(user_id)
        aid = _resolve_agent_id(agent_id)
        collector = get_collector()
        if collector:
            collector.record_operation("delete_all", uid, aid)
        result = _get_service().delete_all_memories(user_id=uid, agent_id=aid)
        return json.dumps(result)
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in delete_all_memories")
        return json.dumps(
            {"error": True, "message": "Internal error processing delete_all_memories"}
        )


# ── Tool 8: list_categories ──────────────────────────────────────────


@mcp.tool()
def list_categories(user_id: str | None = None) -> str:
    """List all available memory categories with descriptions and counts.

    Shows predefined categories and any dynamic project:<name> categories.
    ALWAYS call this before tagging memories with categories. Categories
    are PREDEFINED — do not use categories not in this list. Use
    project:<name> for project-specific scoping.

    Args:
        user_id: User identifier. Optional if pre-configured via API key mapping.
    """
    try:
        uid = _resolve_user_id(user_id)
        collector = get_collector()
        if collector:
            collector.record_operation("list_categories", uid)
        result = _get_service().list_categories(user_id=uid)
        lines = []
        for cat in result["categories"]:
            count = cat["count"]
            marker = f" ({count})" if count > 0 else ""
            lines.append(f"  {cat['name']}{marker} — {cat['description']}")
        header = f"Categories ({result['total_memories']} total memories):"
        return header + "\n" + "\n".join(lines)
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in list_categories")
        return json.dumps(
            {"error": True, "message": "Internal error processing list_categories"}
        )


# ── Tool 9: save_artifact ────────────────────────────────────────────


@mcp.tool()
def save_artifact(
    memory_id: str,
    content: str,
    user_id: str | None = None,
    filename: str = "note.md",
    content_type: str = "text/markdown",
) -> str:
    """Attach an artifact to a memory (slow memory tier).

    Use for detailed content too long for fast memory — research reports,
    analysis, logs, notes, code, data. The fast memory holds the searchable
    summary; the artifact holds the full details.

    Note: When using add_memory with long content, artifacts may be created
    automatically. Use this tool for explicit artifact attachment to an
    existing memory. Multiple memories can reference the same artifact.

    For text content, pass the content directly. For binary content (images,
    PDFs), pass base64-encoded content and set the appropriate content_type.

    Max size: 100KB per artifact.

    Args:
        memory_id: ID of the parent fast memory.
        content: Text content or base64-encoded binary content.
        user_id: User identifier. Optional if pre-configured via API key mapping.
        filename: Name for the artifact (default: note.md).
        content_type: MIME type (default: text/markdown). Use appropriate
                      type for binary content (image/png, application/pdf, etc.).
    """
    try:
        uid = _resolve_user_id(user_id)
        collector = get_collector()
        if collector:
            collector.record_operation("save_artifact", uid)
        result = _get_service().save_artifact(
            memory_id,
            user_id=uid,
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
    user_id: str | None = None,
    offset: int = 0,
    limit: int = 5000,
) -> str:
    """Retrieve artifact content with pagination.

    For large artifacts, use offset to read in chunks. Response includes
    total_size and has_more to indicate if there's more content.

    Args:
        memory_id: ID of any memory that references this artifact.
                   Multiple memories may reference the same artifact —
                   any of them can be used here.
        artifact_id: ID of the artifact to retrieve.
        user_id: User identifier. Optional if pre-configured via API key mapping.
        offset: Character offset for text, byte offset for binary (default 0).
        limit: Max characters/bytes to return (default 5000).
    """
    try:
        uid = _resolve_user_id(user_id)
        collector = get_collector()
        if collector:
            collector.record_operation("get_artifact", uid)
        result = _get_service().get_artifact(
            memory_id,
            artifact_id,
            user_id=uid,
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
def list_artifacts(memory_id: str, user_id: str | None = None) -> str:
    """List all artifacts attached to a memory.

    Returns id, filename, content_type, size, and created_at for each.

    Args:
        memory_id: ID of the parent fast memory.
        user_id: User identifier. Optional if pre-configured via API key mapping.
    """
    try:
        uid = _resolve_user_id(user_id)
        collector = get_collector()
        if collector:
            collector.record_operation("list_artifacts", uid)
        artifacts = _get_service().list_artifacts(memory_id, user_id=uid)
        if not artifacts:
            return json.dumps({"artifacts": [], "message": "No artifacts found."})
        return json.dumps({"artifacts": artifacts})
    except ValueError as e:
        return json.dumps({"error": True, "message": str(e)})
    except Exception:
        logger.exception("Error in list_artifacts")
        return json.dumps(
            {"error": True, "message": "Internal error processing list_artifacts"}
        )


# ── Tool 12: delete_artifact ─────────────────────────────────────────


@mcp.tool()
def delete_artifact(
    memory_id: str, artifact_id: str, user_id: str | None = None
) -> str:
    """Delete an artifact from a memory.

    Args:
        memory_id: ID of the parent fast memory.
        artifact_id: ID of the artifact to delete.
        user_id: User identifier. Optional if pre-configured via API key mapping.
    """
    try:
        uid = _resolve_user_id(user_id)
        collector = get_collector()
        if collector:
            collector.record_operation("delete_artifact", uid)
        result = _get_service().delete_artifact(memory_id, artifact_id, user_id=uid)
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
    """Format memory results for LLM consumption.

    Memory text is escaped to prevent markdown header forgery
    when results are included in LLM context.
    """
    if not memories:
        return json.dumps({"results": [], "count": 0})

    formatted = []
    for mem in memories:
        entry: dict = {
            "id": mem.get("id"),
            "memory": escape_memory_headers(mem.get("memory", "")),
            "created_at": mem.get("created_at"),
        }
        # Include score if present (from search)
        if "score" in mem:
            entry["score"] = round(mem["score"], 4)

        metadata = mem.get("metadata") or {}
        if metadata.get("memory_type"):
            entry["type"] = metadata["memory_type"]
        if metadata.get("categories"):
            entry["categories"] = metadata["categories"]
        if metadata.get("importance") and metadata["importance"] != "normal":
            entry["importance"] = metadata["importance"]
        if metadata.get("pinned"):
            entry["pinned"] = True
        if metadata.get("role") and metadata["role"] != "user":
            entry["role"] = metadata["role"]
        if metadata.get("artifacts"):
            entry["has_artifacts"] = True
            entry["artifact_count"] = len(metadata["artifacts"])

        # Event date (when the event occurred)
        if metadata.get("event_date"):
            entry["event_date"] = metadata["event_date"]

        # TTL state
        if metadata.get("expires_at"):
            entry["expires_at"] = metadata["expires_at"]
        if metadata.get("decayed_at"):
            entry["is_decayed"] = True
            entry["decayed_at"] = metadata["decayed_at"]

        if mem.get("agent_id"):
            entry["agent_id"] = mem["agent_id"]

        formatted.append(entry)

    return json.dumps({"results": formatted, "count": len(formatted)}, default=str)


# ── HTTP Application ──────────────────────────────────────────────────


class APIKeyMiddleware(BaseHTTPMiddleware):
    """API key authentication and session identity middleware.

    Authentication:
    1. Check token against MCP_API_KEYS (JSON dict: key -> user_id)
    2. Check token against MCP_API_KEY (single key, backward compat)
    3. If neither is configured, auth is disabled
    4. If configured but no match, return 401

    Identity resolution (set as contextvars for tool handlers):
    - user_id: from API key mapping (non-wildcard) or X-User-Id header
    - agent_id: from X-Agent-Id header
    """

    async def dispatch(self, request: Request, call_next):
        cfg = _get_config().server
        has_auth = bool(cfg.api_keys or cfg.api_key)

        # Skip auth for UI static files — no sensitive data.
        # Auth is enforced on all /api/ calls from the browser.
        if request.url.path.startswith("/ui"):
            self._set_identity_from_headers(request)
            try:
                return await call_next(request)
            finally:
                _session_user_id.set(None)
                _session_agent_id.set(None)
                _session_timezone.set(None)
                _session_user_bound.set(False)

        if not has_auth:
            # No auth configured — still check identity headers
            self._set_identity_from_headers(request)
        else:
            # Extract token from Authorization: Bearer or X-API-Key header
            token = self._extract_token(request)
            if not token:
                return JSONResponse({"error": "Unauthorized"}, status_code=401)

            # Try MCP_API_KEYS first (key -> user_id mapping)
            mapped_user_id = None
            if cfg.api_keys:
                for key, uid in cfg.api_keys.items():
                    if hmac.compare_digest(token, key):
                        # "*" means wildcard — auth OK but no user binding
                        mapped_user_id = uid if uid != "*" else None
                        break
                else:
                    # Token not in api_keys — try legacy single key
                    if cfg.api_key and hmac.compare_digest(token, cfg.api_key):
                        pass  # Auth OK, no user mapping
                    else:
                        return JSONResponse({"error": "Unauthorized"}, status_code=401)
            elif cfg.api_key:
                # Only legacy single key configured
                if not hmac.compare_digest(token, cfg.api_key):
                    return JSONResponse({"error": "Unauthorized"}, status_code=401)

            # Set session identity
            if mapped_user_id:
                _session_user_id.set(mapped_user_id)
                _session_user_bound.set(True)
            else:
                # Fall back to identity headers (X-User-Id, then
                # X-OpenWebUI-User-Email for Open WebUI integration)
                header_uid = (
                    request.headers.get("x-user-id", "").strip()
                    or request.headers.get("x-openwebui-user-email", "").strip()
                )
                if header_uid:
                    _session_user_id.set(header_uid)

            # Agent ID from header
            header_aid = request.headers.get("x-agent-id", "").strip()
            if header_aid:
                _session_agent_id.set(header_aid)

            # Timezone from header (overrides DEFAULT_TIMEZONE for this session)
            header_tz = request.headers.get("x-timezone", "").strip()
            if header_tz:
                _session_timezone.set(header_tz)

        try:
            return await call_next(request)
        finally:
            # Reset contextvars after request — ensures no identity leakage
            # between requests, regardless of auth path taken.
            _session_user_id.set(None)
            _session_agent_id.set(None)
            _session_timezone.set(None)
            _session_user_bound.set(False)

    def _extract_token(self, request: Request) -> str:
        """Extract API token from request headers."""
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            return auth_header[7:]
        return request.headers.get("x-api-key", "")

    def _set_identity_from_headers(self, request: Request) -> None:
        """Set session identity from HTTP headers (no-auth mode).

        User is never hard-bound when resolved from headers — switching
        is always allowed.
        """
        header_uid = (
            request.headers.get("x-user-id", "").strip()
            or request.headers.get("x-openwebui-user-email", "").strip()
        )
        if header_uid:
            _session_user_id.set(header_uid)
        _session_user_bound.set(False)
        header_aid = request.headers.get("x-agent-id", "").strip()
        if header_aid:
            _session_agent_id.set(header_aid)
        header_tz = request.headers.get("x-timezone", "").strip()
        if header_tz:
            _session_timezone.set(header_tz)


async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint for Kubernetes probes."""
    cfg = _get_config()
    return JSONResponse(
        {
            "status": "healthy",
            "service": "mnemory",
            "version": __version__,
            "vector_backend": "qdrant" if cfg.vector.is_remote else "qdrant-local",
            "artifact_backend": cfg.artifact.backend,
        }
    )


async def metrics_endpoint(request: Request) -> Response:
    """Prometheus metrics endpoint.

    Collects gauge values from Qdrant (cached) and returns all metrics
    in Prometheus text exposition format.
    """
    from mnemory.metrics import get_collector

    collector = get_collector()
    if collector is None:
        return JSONResponse({"error": "Metrics disabled"}, status_code=404)

    collector.collect_gauges()
    return Response(
        content=collector.generate_metrics(),
        media_type=collector.content_type,
    )


@contextlib.asynccontextmanager
async def lifespan(app):
    """Application lifespan manager."""
    cfg = _get_config()
    auth_mode = (
        "api_keys"
        if cfg.server.api_keys
        else ("api_key" if cfg.server.api_key else "disabled")
    )

    # Remove delete_all_memories tool if disabled (default)
    if not cfg.server.enable_delete_all:
        try:
            mcp.remove_tool("delete_all_memories")
            logger.info(
                "delete_all_memories tool disabled "
                "(set ENABLE_DELETE_ALL=true to enable)"
            )
        except Exception:
            pass  # Tool may not exist if already removed

    # Eagerly initialize the service so startup failures are caught early
    service = _get_service()

    # Initialize metrics collector
    from mnemory.api import _session_store
    from mnemory.metrics import init_collector

    init_collector(
        vector_store=service.vector,
        session_store=_session_store,
        config=cfg,
    )

    mgmt_info = (
        f"mgmt_port={cfg.server.mgmt_port}"
        if cfg.server.has_mgmt_port
        else "mgmt_port=disabled"
    )
    logger.info(
        "mnemory starting (vector=%s, artifact=%s, port=%d, auth=%s, "
        "auto_classify=%s, metrics=%s, %s)",
        "qdrant-remote" if cfg.vector.is_remote else "qdrant-local",
        cfg.artifact.backend,
        cfg.server.port,
        auth_mode,
        cfg.memory.auto_classify,
        cfg.server.enable_metrics,
        mgmt_info,
    )

    # Start session cleanup task for REST API
    _session_store.start_cleanup_task()

    async with mcp.session_manager.run():
        yield

    await _session_store.stop_cleanup_task()
    logger.info("mnemory shutting down")


def _build_mgmt_routes() -> list[Route]:
    """Build the list of management routes (/health, optionally /metrics)."""
    routes: list[Route] = [Route("/health", health_check)]
    cfg = _get_config()
    if cfg.server.enable_metrics:
        routes.append(Route("/metrics", metrics_endpoint))
    return routes


def create_app() -> Starlette:
    """Create the main ASGI application.

    When MGMT_PORT is not set (default), /health and /metrics are served
    on the main port and go through standard API key authentication.
    When MGMT_PORT is set, management routes are excluded from the main
    app and served on the separate management port without auth.

    The management UI is mounted at /ui when the static directory exists.
    UI static files are exempt from API key auth (handled in middleware).
    """
    from pathlib import Path

    from starlette.staticfiles import StaticFiles

    from mnemory.api import create_api_app

    cfg = _get_config()
    middleware = [Middleware(APIKeyMiddleware)]

    routes: list[Route | Mount] = []

    # Include management routes on main port only when no separate mgmt port
    if not cfg.server.has_mgmt_port:
        routes.extend(_build_mgmt_routes())

    # Mount management UI if static directory exists (graceful degradation)
    ui_static_dir = Path(__file__).parent / "ui" / "static"
    if ui_static_dir.is_dir():
        # Redirect /ui (no trailing slash) → /ui/ so index.html is served
        async def _ui_redirect(request: Request) -> RedirectResponse:
            return RedirectResponse("/ui/", status_code=301)

        routes.append(Route("/ui", _ui_redirect))
        routes.append(Mount("/ui", app=StaticFiles(directory=ui_static_dir, html=True)))

    routes.extend(
        [
            Mount("/api", app=create_api_app()),
            Mount("/", app=mcp.streamable_http_app()),
        ]
    )

    return Starlette(
        routes=routes,
        middleware=middleware,
        lifespan=lifespan,
    )


def create_mgmt_app() -> Starlette:
    """Create the management ASGI application.

    Lightweight app serving /health and /metrics without authentication.
    Only used when MGMT_PORT is configured. Shares the same service and
    session store singletons as the main app.
    """
    return Starlette(routes=_build_mgmt_routes())


app = create_app()


def main():
    """Run the server with uvicorn.

    When MGMT_PORT is configured, runs two servers in parallel:
    the main MCP/API server and a lightweight management server.
    """
    import asyncio

    import uvicorn

    cfg = _get_config()
    log_level = os.environ.get("LOG_LEVEL", "info").lower()

    if cfg.server.has_mgmt_port:
        # Run main app + management app on separate ports
        async def _serve_dual():
            main_config = uvicorn.Config(
                app,
                host=cfg.server.host,
                port=cfg.server.port,
                log_level=log_level,
            )
            mgmt_config = uvicorn.Config(
                create_mgmt_app(),
                host=cfg.server.mgmt_host,
                port=cfg.server.mgmt_port,
                log_level=log_level,
            )
            main_server = uvicorn.Server(main_config)
            mgmt_server = uvicorn.Server(mgmt_config)
            await asyncio.gather(
                main_server.serve(),
                mgmt_server.serve(),
            )

        asyncio.run(_serve_dual())
    else:
        # Single server — management routes are on the main port (with auth)
        uvicorn.run(
            "mnemory.server:app",
            host=cfg.server.host,
            port=cfg.server.port,
            log_level=log_level,
        )


if __name__ == "__main__":
    main()
