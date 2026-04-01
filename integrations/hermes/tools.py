"""
Hermes tool definitions for the mnemory plugin.

Each tool is defined as a ``(schema, handler)`` pair.  The schema follows
the OpenAI function-calling format.  Handlers accept a single ``args``
dict and return a JSON string.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .client import MnemoryClient

logger = logging.getLogger("hermes_mnemory.tools")

# Maximum items allowed in a single batch add call.
MAX_BATCH_SIZE = 5

TOOLSET = "mnemory"


def _ok(data: Any) -> str:
    """Wrap a successful result as a JSON string."""
    return json.dumps(data)


def _err(message: str) -> str:
    """Return a JSON error string."""
    return json.dumps({"error": message})


def register_tools(
    register_tool: Any,
    client: MnemoryClient,
    agent_id: str | None,
) -> None:
    """Register all 16 mnemory tools with the Hermes plugin context."""

    # ==================================================================
    # Search tools
    # ==================================================================

    register_tool(
        name="memory_search",
        toolset=TOOLSET,
        schema={
            "name": "memory_search",
            "description": (
                "Search memories by semantic similarity. Returns relevant memories "
                "ranked by relevance and importance. Use for simple lookups."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "memory_type": {
                        "type": "string",
                        "description": "Filter by type: preference, fact, episodic, procedural, context",
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by categories",
                    },
                    "role": {
                        "type": "string",
                        "description": "Filter by role: user or assistant",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 10)",
                        "default": 10,
                    },
                    "include_decayed": {
                        "type": "boolean",
                        "description": "Include expired/decayed memories",
                        "default": False,
                    },
                    "date_start": {
                        "type": "string",
                        "description": "Filter memories after this date (ISO 8601)",
                    },
                    "date_end": {
                        "type": "string",
                        "description": "Filter memories before this date (ISO 8601)",
                    },
                    "labels": {
                        "type": "object",
                        "description": "Filter by key-value labels",
                    },
                },
                "required": ["query"],
            },
        },
        handler=_make_search_handler(client, agent_id),
    )

    register_tool(
        name="memory_find",
        toolset=TOOLSET,
        schema={
            "name": "memory_find",
            "description": (
                "AI-powered multi-query search. Generates multiple targeted searches "
                "covering different angles and associations, then reranks results by "
                "relevance. Slower (2 extra LLM calls) but higher quality for complex "
                "questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Natural language question",
                    },
                    "memory_type": {
                        "type": "string",
                        "description": "Filter by type: preference, fact, episodic, procedural, context",
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by categories",
                    },
                    "role": {
                        "type": "string",
                        "description": "Filter by role: user or assistant",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 10)",
                        "default": 10,
                    },
                    "include_decayed": {
                        "type": "boolean",
                        "description": "Include expired/decayed memories",
                        "default": False,
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context to guide the search",
                    },
                    "labels": {
                        "type": "object",
                        "description": "Filter by key-value labels",
                    },
                },
                "required": ["question"],
            },
        },
        handler=_make_find_handler(client, agent_id),
    )

    register_tool(
        name="memory_ask",
        toolset=TOOLSET,
        schema={
            "name": "memory_ask",
            "description": (
                "Ask a question and get a human-readable answer based on stored memories. "
                "Most expensive operation (3 LLM calls). Use when you need a synthesised "
                "answer rather than raw memory results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Natural language question",
                    },
                    "memory_type": {
                        "type": "string",
                        "description": "Filter by type: preference, fact, episodic, procedural, context",
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by categories",
                    },
                    "role": {
                        "type": "string",
                        "description": "Filter by role: user or assistant",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 10)",
                        "default": 10,
                    },
                    "include_decayed": {
                        "type": "boolean",
                        "description": "Include expired/decayed memories",
                        "default": False,
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context to guide the search",
                    },
                    "include_memories": {
                        "type": "boolean",
                        "description": "Also return the supporting memories used to generate the answer",
                        "default": False,
                    },
                    "labels": {
                        "type": "object",
                        "description": "Filter by key-value labels",
                    },
                },
                "required": ["question"],
            },
        },
        handler=_make_ask_handler(client, agent_id),
    )

    # ==================================================================
    # CRUD tools
    # ==================================================================

    register_tool(
        name="memory_add",
        toolset=TOOLSET,
        schema={
            "name": "memory_add",
            "description": (
                "Store a new memory. Content is automatically analysed for facts, "
                "classified, and deduplicated against existing memories."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Memory content (max 1000 chars)",
                    },
                    "memory_type": {
                        "type": "string",
                        "description": "Type: preference, fact, episodic, procedural, context",
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Category tags from the predefined set",
                    },
                    "importance": {
                        "type": "string",
                        "description": "Importance: low, normal, high, critical",
                    },
                    "pinned": {
                        "type": "boolean",
                        "description": "Pin as a core memory (never expires)",
                    },
                    "infer": {
                        "type": "boolean",
                        "description": "Use LLM to extract facts and deduplicate (default true)",
                        "default": True,
                    },
                    "role": {
                        "type": "string",
                        "description": "Who the memory is about: user (default) or assistant",
                    },
                    "ttl_days": {
                        "type": "integer",
                        "description": "Time-to-live in days (overrides type default)",
                    },
                    "event_date": {
                        "type": "string",
                        "description": "When the event occurred (ISO 8601)",
                    },
                    "labels": {
                        "type": "object",
                        "description": "Key-value metadata labels",
                    },
                },
                "required": ["content"],
            },
        },
        handler=_make_add_handler(client, agent_id),
    )

    register_tool(
        name="memory_add_batch",
        toolset=TOOLSET,
        schema={
            "name": "memory_add_batch",
            "description": (
                f"Store multiple memories in a single call (max {MAX_BATCH_SIZE}). "
                "Each memory is processed independently."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "memory_type": {"type": "string"},
                                "categories": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "importance": {"type": "string"},
                                "pinned": {"type": "boolean"},
                                "infer": {"type": "boolean"},
                                "role": {"type": "string"},
                                "ttl_days": {"type": "integer"},
                                "event_date": {"type": "string"},
                                "labels": {"type": "object"},
                            },
                            "required": ["content"],
                        },
                        "description": f"Array of memories to store (max {MAX_BATCH_SIZE})",
                    },
                },
                "required": ["memories"],
            },
        },
        handler=_make_add_batch_handler(client, agent_id),
    )

    register_tool(
        name="memory_update",
        toolset=TOOLSET,
        schema={
            "name": "memory_update",
            "description": "Update an existing memory's content or metadata by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "ID of the memory to update",
                    },
                    "content": {"type": "string", "description": "New content"},
                    "memory_type": {"type": "string", "description": "New type"},
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "New categories",
                    },
                    "importance": {"type": "string", "description": "New importance"},
                    "pinned": {"type": "boolean", "description": "Pin/unpin"},
                    "ttl_days": {"type": "integer", "description": "New TTL in days"},
                    "event_date": {"type": "string", "description": "New event date"},
                    "labels": {
                        "type": "object",
                        "description": "New labels (empty dict to clear)",
                    },
                },
                "required": ["memory_id"],
            },
        },
        handler=_make_update_handler(client, agent_id),
    )

    register_tool(
        name="memory_delete",
        toolset=TOOLSET,
        schema={
            "name": "memory_delete",
            "description": "Delete a memory by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "ID of the memory to delete",
                    },
                },
                "required": ["memory_id"],
            },
        },
        handler=_make_delete_handler(client, agent_id),
    )

    register_tool(
        name="memory_delete_batch",
        toolset=TOOLSET,
        schema={
            "name": "memory_delete_batch",
            "description": "Delete multiple memories in a single call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of memories to delete",
                    },
                },
                "required": ["memory_ids"],
            },
        },
        handler=_make_delete_batch_handler(client, agent_id),
    )

    # ==================================================================
    # Browse tools
    # ==================================================================

    register_tool(
        name="memory_list",
        toolset=TOOLSET,
        schema={
            "name": "memory_list",
            "description": "List stored memories with optional filters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_type": {"type": "string", "description": "Filter by type"},
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by categories",
                    },
                    "role": {
                        "type": "string",
                        "description": "Filter by role: user or assistant",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results",
                        "default": 20,
                    },
                    "include_decayed": {
                        "type": "boolean",
                        "description": "Include expired/decayed memories",
                        "default": False,
                    },
                    "labels": {"type": "object", "description": "Filter by labels"},
                },
            },
        },
        handler=_make_list_handler(client, agent_id),
    )

    register_tool(
        name="memory_categories",
        toolset=TOOLSET,
        schema={
            "name": "memory_categories",
            "description": "List all available memory categories with descriptions and counts.",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=_make_categories_handler(client, agent_id),
    )

    register_tool(
        name="memory_recent",
        toolset=TOOLSET,
        schema={
            "name": "memory_recent",
            "description": "Get recent memories from the last N days, ordered by most recent first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Number of days to look back",
                        "default": 7,
                    },
                    "scope": {"type": "string", "description": "Scope filter"},
                    "limit": {
                        "type": "integer",
                        "description": "Max results",
                        "default": 20,
                    },
                    "include_decayed": {
                        "type": "boolean",
                        "description": "Include expired/decayed memories",
                        "default": False,
                    },
                },
            },
        },
        handler=_make_recent_handler(client, agent_id),
    )

    # ==================================================================
    # Artifact tools
    # ==================================================================

    register_tool(
        name="memory_save_artifact",
        toolset=TOOLSET,
        schema={
            "name": "memory_save_artifact",
            "description": (
                "Attach an artifact to a memory (slow memory tier). Use for detailed "
                "content too long for fast memory -- research reports, analysis, logs, "
                "notes, code, data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "ID of the parent memory",
                    },
                    "content": {"type": "string", "description": "Artifact content"},
                    "filename": {
                        "type": "string",
                        "description": "Filename (default: note.md)",
                        "default": "note.md",
                    },
                    "content_type": {
                        "type": "string",
                        "description": "MIME type (default: text/markdown)",
                        "default": "text/markdown",
                    },
                },
                "required": ["memory_id", "content"],
            },
        },
        handler=_make_save_artifact_handler(client, agent_id),
    )

    register_tool(
        name="memory_get_artifact",
        toolset=TOOLSET,
        schema={
            "name": "memory_get_artifact",
            "description": "Retrieve artifact content attached to a memory. Supports pagination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "ID of the parent memory",
                    },
                    "artifact_id": {
                        "type": "string",
                        "description": "ID of the artifact",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Byte offset for pagination",
                    },
                    "limit": {"type": "integer", "description": "Max bytes to return"},
                },
                "required": ["memory_id", "artifact_id"],
            },
        },
        handler=_make_get_artifact_handler(client, agent_id),
    )

    register_tool(
        name="memory_get_artifact_url",
        toolset=TOOLSET,
        schema={
            "name": "memory_get_artifact_url",
            "description": (
                "Generate a short-lived signed URL for direct artifact download. "
                "Use for binary artifacts (images, PDFs) or large artifacts (>1 MB)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "ID of the parent memory",
                    },
                    "artifact_id": {
                        "type": "string",
                        "description": "ID of the artifact",
                    },
                    "ttl": {
                        "type": "integer",
                        "description": "URL lifetime in seconds (min 60)",
                        "default": 3600,
                    },
                },
                "required": ["memory_id", "artifact_id"],
            },
        },
        handler=_make_get_artifact_url_handler(client, agent_id),
    )

    register_tool(
        name="memory_list_artifacts",
        toolset=TOOLSET,
        schema={
            "name": "memory_list_artifacts",
            "description": "List all artifacts attached to a memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "ID of the parent memory",
                    },
                },
                "required": ["memory_id"],
            },
        },
        handler=_make_list_artifacts_handler(client, agent_id),
    )

    register_tool(
        name="memory_delete_artifact",
        toolset=TOOLSET,
        schema={
            "name": "memory_delete_artifact",
            "description": "Delete an artifact from a memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "ID of the parent memory",
                    },
                    "artifact_id": {
                        "type": "string",
                        "description": "ID of the artifact",
                    },
                },
                "required": ["memory_id", "artifact_id"],
            },
        },
        handler=_make_delete_artifact_handler(client, agent_id),
    )


# ======================================================================
# Handler factories
# ======================================================================


def _make_search_handler(client: MnemoryClient, agent_id: str | None):  # noqa: ANN202
    def handler(args: dict[str, Any], **kwargs: Any) -> str:
        result = client.search_memories(
            query=args["query"],
            memory_type=args.get("memory_type"),
            categories=args.get("categories"),
            role=args.get("role"),
            limit=args.get("limit", 10),
            include_decayed=args.get("include_decayed", False),
            date_start=args.get("date_start"),
            date_end=args.get("date_end"),
            labels=args.get("labels"),
            agent_id=agent_id,
        )
        if result is None:
            return _err("Failed to search memories (mnemory server unreachable)")
        return _ok(result)

    return handler


def _make_find_handler(client: MnemoryClient, agent_id: str | None):  # noqa: ANN202
    def handler(args: dict[str, Any], **kwargs: Any) -> str:
        result = client.find_memories(
            question=args["question"],
            memory_type=args.get("memory_type"),
            categories=args.get("categories"),
            role=args.get("role"),
            limit=args.get("limit", 10),
            include_decayed=args.get("include_decayed", False),
            context=args.get("context"),
            labels=args.get("labels"),
            agent_id=agent_id,
        )
        if result is None:
            return _err("Failed to find memories (mnemory server unreachable)")
        return _ok(result)

    return handler


def _make_ask_handler(client: MnemoryClient, agent_id: str | None):  # noqa: ANN202
    def handler(args: dict[str, Any], **kwargs: Any) -> str:
        result = client.ask_memories(
            question=args["question"],
            memory_type=args.get("memory_type"),
            categories=args.get("categories"),
            role=args.get("role"),
            limit=args.get("limit", 10),
            include_decayed=args.get("include_decayed", False),
            context=args.get("context"),
            include_memories=args.get("include_memories", False),
            labels=args.get("labels"),
            agent_id=agent_id,
        )
        if result is None:
            return _err("Failed to ask memories (mnemory server unreachable)")
        return _ok(result)

    return handler


def _make_add_handler(client: MnemoryClient, agent_id: str | None):  # noqa: ANN202
    def handler(args: dict[str, Any], **kwargs: Any) -> str:
        # Auto-add source label
        labels = dict(args.get("labels") or {})
        labels.setdefault("source", "hermes")

        result = client.add_memory(
            content=args["content"],
            memory_type=args.get("memory_type"),
            categories=args.get("categories"),
            importance=args.get("importance"),
            pinned=args.get("pinned"),
            infer=args.get("infer"),
            role=args.get("role"),
            ttl_days=args.get("ttl_days"),
            event_date=args.get("event_date"),
            labels=labels,
            agent_id=agent_id,
        )
        if result is None:
            return _err("Failed to add memory (mnemory server unreachable)")
        return _ok(result)

    return handler


def _make_add_batch_handler(client: MnemoryClient, agent_id: str | None):  # noqa: ANN202
    def handler(args: dict[str, Any], **kwargs: Any) -> str:
        memories = args.get("memories", [])
        if len(memories) > MAX_BATCH_SIZE:
            return _err(
                f"Too many memories ({len(memories)}). Maximum is {MAX_BATCH_SIZE}."
            )

        # Auto-add source label to each item
        for m in memories:
            labels = dict(m.get("labels") or {})
            labels.setdefault("source", "hermes")
            m["labels"] = labels

        result = client.add_memories_batch(memories=memories, agent_id=agent_id)
        if result is None:
            return _err("Failed to add memories (mnemory server unreachable)")
        return _ok(result)

    return handler


def _make_update_handler(client: MnemoryClient, agent_id: str | None):  # noqa: ANN202
    def handler(args: dict[str, Any], **kwargs: Any) -> str:
        ok = client.update_memory(
            args["memory_id"],
            content=args.get("content"),
            memory_type=args.get("memory_type"),
            categories=args.get("categories"),
            importance=args.get("importance"),
            pinned=args.get("pinned"),
            ttl_days=args.get("ttl_days"),
            event_date=args.get("event_date"),
            labels=args.get("labels"),
            agent_id=agent_id,
        )
        if not ok:
            return _err("Failed to update memory")
        return _ok({"success": True})

    return handler


def _make_delete_handler(client: MnemoryClient, agent_id: str | None):  # noqa: ANN202
    def handler(args: dict[str, Any], **kwargs: Any) -> str:
        ok = client.delete_memory(args["memory_id"], agent_id=agent_id)
        if not ok:
            return _err("Failed to delete memory")
        return _ok({"success": True})

    return handler


def _make_delete_batch_handler(client: MnemoryClient, agent_id: str | None):  # noqa: ANN202
    def handler(args: dict[str, Any], **kwargs: Any) -> str:
        result = client.delete_memories_batch(args["memory_ids"], agent_id=agent_id)
        if result is None:
            return _err("Failed to delete memories (mnemory server unreachable)")
        return _ok(result)

    return handler


def _make_list_handler(client: MnemoryClient, agent_id: str | None):  # noqa: ANN202
    def handler(args: dict[str, Any], **kwargs: Any) -> str:
        result = client.list_memories(
            memory_type=args.get("memory_type"),
            categories=args.get("categories"),
            role=args.get("role"),
            limit=args.get("limit", 20),
            include_decayed=args.get("include_decayed", False),
            labels=args.get("labels"),
            agent_id=agent_id,
        )
        if result is None:
            return _err("Failed to list memories (mnemory server unreachable)")
        return _ok(result)

    return handler


def _make_categories_handler(client: MnemoryClient, agent_id: str | None):  # noqa: ANN202
    def handler(args: dict[str, Any], **kwargs: Any) -> str:
        result = client.list_categories(agent_id=agent_id)
        if result is None:
            return _err("Failed to list categories (mnemory server unreachable)")
        return _ok(result)

    return handler


def _make_recent_handler(client: MnemoryClient, agent_id: str | None):  # noqa: ANN202
    def handler(args: dict[str, Any], **kwargs: Any) -> str:
        result = client.get_recent_memories(
            days=args.get("days"),
            scope=args.get("scope"),
            limit=args.get("limit", 20),
            include_decayed=args.get("include_decayed", False),
            agent_id=agent_id,
        )
        if result is None:
            return _err("Failed to get recent memories (mnemory server unreachable)")
        return _ok(result)

    return handler


def _make_save_artifact_handler(client: MnemoryClient, agent_id: str | None):  # noqa: ANN202
    def handler(args: dict[str, Any], **kwargs: Any) -> str:
        result = client.save_artifact(
            args["memory_id"],
            content=args["content"],
            filename=args.get("filename", "note.md"),
            content_type=args.get("content_type", "text/markdown"),
            agent_id=agent_id,
        )
        if result is None:
            return _err("Failed to save artifact (mnemory server unreachable)")
        return _ok(result)

    return handler


def _make_get_artifact_handler(client: MnemoryClient, agent_id: str | None):  # noqa: ANN202
    def handler(args: dict[str, Any], **kwargs: Any) -> str:
        result = client.get_artifact(
            args["memory_id"],
            args["artifact_id"],
            offset=args.get("offset"),
            limit=args.get("limit"),
            agent_id=agent_id,
        )
        if result is None:
            return _err("Failed to get artifact (mnemory server unreachable)")
        return _ok(result)

    return handler


def _make_get_artifact_url_handler(client: MnemoryClient, agent_id: str | None):  # noqa: ANN202
    def handler(args: dict[str, Any], **kwargs: Any) -> str:
        result = client.get_artifact_url(
            args["memory_id"],
            args["artifact_id"],
            ttl=args.get("ttl"),
            agent_id=agent_id,
        )
        if result is None:
            return _err("Failed to get artifact URL (mnemory server unreachable)")
        return _ok(result)

    return handler


def _make_list_artifacts_handler(client: MnemoryClient, agent_id: str | None):  # noqa: ANN202
    def handler(args: dict[str, Any], **kwargs: Any) -> str:
        result = client.list_artifacts(args["memory_id"], agent_id=agent_id)
        if result is None:
            return _err("Failed to list artifacts (mnemory server unreachable)")
        return _ok(result)

    return handler


def _make_delete_artifact_handler(client: MnemoryClient, agent_id: str | None):  # noqa: ANN202
    def handler(args: dict[str, Any], **kwargs: Any) -> str:
        ok = client.delete_artifact(
            args["memory_id"],
            args["artifact_id"],
            agent_id=agent_id,
        )
        if not ok:
            return _err("Failed to delete artifact")
        return _ok({"success": True})

    return handler
