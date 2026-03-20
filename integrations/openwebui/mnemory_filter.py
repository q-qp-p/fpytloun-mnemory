"""
title: Mnemory - Persistent Memory
author: mnemory
description: Automatic memory recall and storage for conversations
version: 0.1.0
"""

import asyncio
from typing import Callable, Optional

import aiohttp
from pydantic import BaseModel, Field


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Filter priority (lower = runs first)",
        )
        mnemory_url: str = Field(
            default="http://localhost:8050",
            description="Mnemory server base URL",
        )
        api_key: str = Field(
            default="",
            description="API key for mnemory authentication",
        )
        agent_id: str = Field(
            default="open-webui",
            description="Agent ID sent to mnemory",
        )
        recall_mode: str = Field(
            default="always",
            description=(
                "When to recall memories: "
                "'always' = every message (recommended), "
                "'first_only' = first message only (no subsequent recalls)"
            ),
        )
        recall_search_mode: str = Field(
            default="search",
            description=(
                "Search mode for recall: "
                "'find' = AI-powered multi-query search (thorough, slower), "
                "'search' = single vector search (fast, no LLM)"
            ),
        )
        recall_find_first: bool = Field(
            default=True,
            description=(
                "When recall_search_mode is 'search', use 'find' for the "
                "first message in a session (thorough initial context). "
                "Ignored when recall_search_mode is 'find'."
            ),
        )
        recall_score_threshold: float = Field(
            default=0.5,
            description=(
                "Minimum relevance score (0.0-1.0) for recalled memories. "
                "Higher = fewer but more relevant memories injected. "
                "Prevents context bloat from weak matches on follow-up messages."
            ),
        )
        show_status: bool = Field(
            default=True,
            description="Show memory status messages in chat (can be overridden per-user)",
        )
        request_timeout: int = Field(
            default=30,
            description="HTTP request timeout in seconds for mnemory API calls",
        )
        strip_redundant_mcp_tools: bool = Field(
            default=True,
            description=(
                "Remove mnemory MCP tools that the filter handles automatically "
                "(initialize_memory, get_core_memories, get_recent_memories) "
                "from the request to reduce prompt token usage."
            ),
        )

    class UserValves(BaseModel):
        enabled: bool = Field(
            default=True,
            description="Enable memory for this user",
        )
        show_status: bool = Field(
            default=True,
            description="Show memory status messages in chat",
        )

    # Max tracked sessions before evicting oldest entries.
    # Prevents unbounded memory growth in long-running instances.
    _MAX_SESSIONS = 1000

    # Mnemory MCP tools that the filter handles automatically.
    # Stripped from tool_ids to save prompt tokens (~800 tokens/request).
    _MANAGED_TOOL_SUFFIXES = {
        "initialize_memory",
        "get_core_memories",
        "get_recent_memories",
    }

    def __init__(self):
        self.valves = self.Valves()
        # Track which chats have been initialized (chat_id -> session_id)
        self._sessions: dict[str, str] = {}
        # Pending session from first message when chat_id is not yet available.
        # Open WebUI may not provide chat_id on the first message of a new
        # chat (chat not yet saved to DB). This slot holds the server-side
        # session_id until the real chat_id arrives on the second message.
        self._pending_session_id: str | None = None

    async def _post(self, path: str, payload: dict, user: dict) -> dict | None:
        """Make a POST request to mnemory REST API."""
        headers = {
            "Content-Type": "application/json",
            "X-Agent-Id": self.valves.agent_id,
            "X-User-Id": user.get("email", user.get("id", "")),
        }
        if self.valves.api_key:
            headers["Authorization"] = f"Bearer {self.valves.api_key}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.valves.mnemory_url}{path}",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.valves.request_timeout),
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    return None
        except Exception:
            return None  # Graceful degradation

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable] = None,
    ) -> dict:
        """Before LLM: recall memories and inject into context."""
        if not __user__:
            return body

        # Check user valves
        user_valves = __user__.get("valves")
        if user_valves and hasattr(user_valves, "enabled") and not user_valves.enabled:
            return body

        # Strip redundant mnemory MCP tools to save prompt tokens.
        # The filter handles recall automatically — these tools would
        # only waste tokens in the tools[] array on every LLM request.
        if self.valves.strip_redundant_mcp_tools:
            tool_ids = body.get("tool_ids")
            if tool_ids:
                body["tool_ids"] = [
                    t
                    for t in tool_ids
                    if not any(t.endswith(s) for s in self._MANAGED_TOOL_SUFFIXES)
                ]

        chat_id = body.get("chat_id") or ""

        # Look up or adopt server-side session for dedup tracking.
        # Open WebUI may not provide chat_id on the first message of a
        # new chat (chat not yet saved to DB).  When the real chat_id
        # arrives on the second message, adopt the pending session so
        # known_ids for dedup are preserved across the transition.
        if chat_id:
            session_id = self._sessions.get(chat_id)
            if session_id is None and self._pending_session_id:
                session_id = self._pending_session_id
                self._sessions[chat_id] = session_id
                self._pending_session_id = None
        else:
            session_id = self._pending_session_id

        # Determine first turn from conversation history, not session
        # tracking.  Session-based detection is unreliable because
        # chat_id may be absent on the first message, causing the
        # empty-key entry in _sessions to collide across different chats.
        messages = body.get("messages", [])
        user_msg_count = sum(1 for m in messages if m.get("role") == "user")
        is_first = user_msg_count <= 1

        # In first_only mode, skip recall on subsequent messages entirely
        if not is_first and self.valves.recall_mode == "first_only":
            return body

        # Extract query from last user message
        query = ""
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                query = content if isinstance(content, str) else ""
                break

        if not query and not is_first:
            return body  # No query on subsequent turn — skip

        # Show status (admin valve AND user valve must both be true)
        show_status = self.valves.show_status
        if show_status and user_valves and hasattr(user_valves, "show_status"):
            show_status = user_valves.show_status
        if __event_emitter__ and show_status:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Recalling memories...",
                        "done": False,
                    },
                }
            )

        # Determine search mode for this call
        if is_first and self.valves.recall_find_first:
            search_mode = "find"
        else:
            search_mode = self.valves.recall_search_mode

        # Call recall endpoint
        payload: dict = {
            "session_id": session_id,
            "query": query,
            "search_mode": search_mode,
            "score_threshold": self.valves.recall_score_threshold,
        }
        if is_first:
            payload["include_instructions"] = True
            payload["managed"] = True

        result = await self._post("/api/recall", payload, __user__)

        if result and result.get("session_id"):
            if chat_id:
                self._sessions[chat_id] = result["session_id"]
                self._pending_session_id = None
                # Evict oldest entries if over limit to prevent unbounded growth
                if len(self._sessions) > self._MAX_SESSIONS:
                    excess = len(self._sessions) - self._MAX_SESSIONS
                    for key in list(self._sessions)[:excess]:
                        del self._sessions[key]
            else:
                self._pending_session_id = result["session_id"]

        # Build injection text
        parts = []
        if result:
            if result.get("instructions"):
                parts.append(result["instructions"])
            if result.get("core_memories"):
                parts.append(result["core_memories"])
            if result.get("search_results"):
                memories_text = "\n".join(
                    f"- {m['memory']}"
                    for m in result["search_results"]
                    if m.get("memory")
                )
                if memories_text:
                    parts.append(f"## Recalled Memories\n{memories_text}")

        if parts:
            # Append after the last user message to preserve the
            # conversation prefix for LLM prompt caching.  The entire
            # conversation history (system prompt, prior user/assistant
            # exchanges, and the current user message) forms a stable,
            # append-only prefix that OpenAI can cache across turns.
            # Inserting *before* the last user message would shift the
            # cached prefix boundary on every turn, breaking the cache.
            body["messages"].append(
                {
                    "role": "system",
                    "content": "\n\n".join(parts),
                },
            )

        # Show detailed status with stats
        if __event_emitter__ and show_status:
            desc = self._build_status(result, is_first)
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": desc, "done": True},
                }
            )

        return body

    @staticmethod
    def _build_status(result: dict | None, is_first: bool) -> str:
        """Build a detailed status message from recall stats."""
        if not result:
            return "Memory unavailable"

        stats = result.get("stats", {})
        ms = stats.get("latency_ms", 0)
        core = stats.get("core_count", 0)
        new = stats.get("new_count", 0)

        if is_first:
            if core and new:
                return f"Recalled {core} core + {new} relevant memories ({ms}ms)"
            if core:
                return f"Recalled {core} core memories ({ms}ms)"
            if new:
                return f"Found {new} relevant memories ({ms}ms)"
            return f"Memory ready ({ms}ms)"

        # Subsequent call
        if new:
            return f"Found {new} new memories ({ms}ms)"
        return f"No new memories ({ms}ms)"

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable] = None,
    ) -> dict:
        """After LLM: store memories from the exchange (fire-and-forget)."""
        if not __user__:
            return body

        user_valves = __user__.get("valves")
        if user_valves and hasattr(user_valves, "enabled") and not user_valves.enabled:
            return body

        chat_id = body.get("chat_id", "")
        session_id = self._sessions.get(chat_id)
        messages = body.get("messages", [])

        # Only user/assistant messages — exclude system prompts and tool results
        conversation = [m for m in messages if m.get("role") in ("user", "assistant")]

        if len(conversation) < 2:
            return body

        # Last 2 user/assistant messages (current exchange)
        last_two = conversation[-2:]

        # Build context from the first user message to give the extraction
        # LLM topic awareness. Without this, memories extracted from the
        # last exchange can be vague (e.g., "User wants to search the web")
        # because the LLM doesn't know what the conversation is about.
        context = None
        first_user_msg = next(
            (
                m.get("content", "")
                for m in conversation
                if m.get("role") == "user" and m.get("content")
            ),
            None,
        )
        if first_user_msg and isinstance(first_user_msg, str):
            # Cap context to avoid sending huge first messages
            context = f"Conversation topic: {first_user_msg[:500]}"

        # Fire-and-forget
        payload: dict = {"session_id": session_id, "messages": last_two}
        if context:
            payload["context"] = context
        # Attach labels for provenance tracking (chat_id links memories
        # to a specific conversation, source identifies the client)
        labels: dict[str, str] = {"source": "open-webui"}
        if chat_id:
            labels["chat_id"] = chat_id
        payload["labels"] = labels
        asyncio.create_task(self._post("/api/remember", payload, __user__))

        return body
