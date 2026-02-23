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

    def __init__(self):
        self.valves = self.Valves()
        # Track which chats have been initialized (chat_id -> session_id)
        self._sessions: dict[str, str] = {}

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
                    timeout=aiohttp.ClientTimeout(total=10),
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

        chat_id = body.get("chat_id", "")
        session_id = self._sessions.get(chat_id)
        is_first = session_id is None

        # Extract query from last user message
        query = ""
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                query = content if isinstance(content, str) else ""
                break

        if not query and not is_first:
            return body  # No query on subsequent turn — skip

        # Show status
        show_status = True
        if user_valves and hasattr(user_valves, "show_status"):
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

        # Call recall endpoint
        payload: dict = {
            "session_id": session_id,
            "query": query,
        }
        if is_first:
            payload["include_instructions"] = True
            payload["managed"] = True

        result = await self._post("/api/recall", payload, __user__)

        if result and result.get("session_id"):
            self._sessions[chat_id] = result["session_id"]
            # Evict oldest entries if over limit to prevent unbounded growth
            if len(self._sessions) > self._MAX_SESSIONS:
                excess = len(self._sessions) - self._MAX_SESSIONS
                for key in list(self._sessions)[:excess]:
                    del self._sessions[key]

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
                    parts.append(f"## Relevant Context\n{memories_text}")

        if parts:
            body["messages"].insert(
                0,
                {
                    "role": "system",
                    "content": "\n\n".join(parts),
                },
            )

        if __event_emitter__ and show_status:
            count = len(result.get("search_results", [])) if result else 0
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": (
                            f"Loaded {count} memories" if count else "Memory ready"
                        ),
                        "done": True,
                    },
                }
            )

        return body

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

        if len(messages) < 2:
            return body

        # Last 2 messages only (current exchange: user + assistant)
        last_two = messages[-2:]

        # Fire-and-forget
        asyncio.create_task(
            self._post(
                "/api/remember",
                {"session_id": session_id, "messages": last_two},
                __user__,
            )
        )

        return body
