"""
Mnemory REST API client.

Thin HTTP wrapper around mnemory's REST endpoints using ``requests``.
All methods use graceful error handling -- API failures are logged
but never raised, so the agent keeps working if mnemory is offline.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import quote

import requests

logger = logging.getLogger("hermes_mnemory.client")

DEFAULT_TIMEOUT = 60  # seconds


class MnemoryClient:
    """Synchronous HTTP client for the mnemory REST API."""

    def __init__(
        self,
        *,
        url: str,
        api_key: str = "",
        user_id: str = "",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._base_url = url.rstrip("/")
        self._api_key = api_key
        self._user_id = user_id
        self._timeout = timeout
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self, agent_id: str | None = None) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        if self._user_id:
            h["X-User-Id"] = self._user_id
        if agent_id:
            h["X-Agent-Id"] = agent_id
        return h

    def _post(
        self,
        path: str,
        body: dict[str, Any],
        agent_id: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        try:
            resp = self._session.post(
                f"{self._base_url}{path}",
                headers=self._headers(agent_id),
                data=json.dumps(body),
                timeout=timeout or self._timeout,
            )
            if not resp.ok:
                logger.warning(
                    "mnemory: POST %s returned %s: %s",
                    path,
                    resp.status_code,
                    resp.text,
                )
                return None
            return resp.json()  # type: ignore[no-any-return]
        except Exception:
            logger.warning("mnemory: POST %s failed", path, exc_info=True)
            return None

    def _get(
        self,
        path: str,
        agent_id: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any] | list[Any] | None:
        try:
            resp = self._session.get(
                f"{self._base_url}{path}",
                headers=self._headers(agent_id),
                timeout=timeout or self._timeout,
            )
            if not resp.ok:
                logger.warning(
                    "mnemory: GET %s returned %s: %s", path, resp.status_code, resp.text
                )
                return None
            return resp.json()  # type: ignore[no-any-return]
        except Exception:
            logger.warning("mnemory: GET %s failed", path, exc_info=True)
            return None

    def _put(
        self,
        path: str,
        body: dict[str, Any],
        agent_id: str | None = None,
    ) -> bool:
        try:
            resp = self._session.put(
                f"{self._base_url}{path}",
                headers=self._headers(agent_id),
                data=json.dumps(body),
                timeout=self._timeout,
            )
            if not resp.ok:
                logger.warning(
                    "mnemory: PUT %s returned %s: %s", path, resp.status_code, resp.text
                )
                return False
            return True
        except Exception:
            logger.warning("mnemory: PUT %s failed", path, exc_info=True)
            return False

    def _delete(self, path: str, agent_id: str | None = None) -> bool:
        try:
            resp = self._session.delete(
                f"{self._base_url}{path}",
                headers=self._headers(agent_id),
                timeout=self._timeout,
            )
            if not resp.ok:
                logger.warning(
                    "mnemory: DELETE %s returned %s: %s",
                    path,
                    resp.status_code,
                    resp.text,
                )
                return False
            return True
        except Exception:
            logger.warning("mnemory: DELETE %s failed", path, exc_info=True)
            return False

    @staticmethod
    def _enc(value: str) -> str:
        return quote(value, safe="")

    # ------------------------------------------------------------------
    # Recall / Remember (lifecycle hooks)
    # ------------------------------------------------------------------

    def recall(
        self,
        *,
        session_id: str | None = None,
        query: str | None = None,
        include_instructions: bool = False,
        managed: bool = False,
        instruction_mode: str | None = None,
        search_mode: str | None = None,
        score_threshold: float = 0.5,
        context: str | None = None,
        labels: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any] | None:
        """``POST /api/recall`` -- combined initialise + search."""
        body: dict[str, Any] = {
            "score_threshold": score_threshold,
        }
        if session_id:
            body["session_id"] = session_id
        if query:
            body["query"] = query
        if include_instructions:
            body["include_instructions"] = True
        if managed:
            body["managed"] = True
        if instruction_mode:
            body["instruction_mode"] = instruction_mode
        if search_mode:
            body["search_mode"] = search_mode
        if context:
            body["context"] = context
        if labels:
            body["labels"] = labels
        return self._post("/api/recall", body, agent_id=agent_id)

    def remember(
        self,
        *,
        session_id: str | None = None,
        messages: list[dict[str, str]],
        context: str | None = None,
        labels: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ) -> None:
        """``POST /api/remember`` -- fire-and-forget memory extraction."""
        body: dict[str, Any] = {"messages": messages}
        if session_id:
            body["session_id"] = session_id
        if context:
            body["context"] = context
        if labels:
            body["labels"] = labels
        # Fire-and-forget: we don't need the response
        self._post("/api/remember", body, agent_id=agent_id)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_memories(
        self,
        *,
        query: str,
        memory_type: str | None = None,
        categories: list[str] | None = None,
        role: str | None = None,
        limit: int = 10,
        include_decayed: bool = False,
        date_start: str | None = None,
        date_end: str | None = None,
        labels: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any] | None:
        """``POST /api/memories/search`` -- semantic search."""
        body: dict[str, Any] = {"query": query, "limit": limit}
        if memory_type:
            body["memory_type"] = memory_type
        if categories:
            body["categories"] = categories
        if role:
            body["role"] = role
        if include_decayed:
            body["include_decayed"] = True
        if date_start:
            body["date_start"] = date_start
        if date_end:
            body["date_end"] = date_end
        if labels:
            body["labels"] = labels
        return self._post("/api/memories/search", body, agent_id=agent_id)

    def find_memories(
        self,
        *,
        question: str,
        memory_type: str | None = None,
        categories: list[str] | None = None,
        role: str | None = None,
        limit: int = 10,
        include_decayed: bool = False,
        context: str | None = None,
        labels: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any] | None:
        """``POST /api/memories/find`` -- AI-powered multi-query search."""
        body: dict[str, Any] = {"question": question, "limit": limit}
        if memory_type:
            body["memory_type"] = memory_type
        if categories:
            body["categories"] = categories
        if role:
            body["role"] = role
        if include_decayed:
            body["include_decayed"] = True
        if context:
            body["context"] = context
        if labels:
            body["labels"] = labels
        return self._post("/api/memories/find", body, agent_id=agent_id)

    def ask_memories(
        self,
        *,
        question: str,
        memory_type: str | None = None,
        categories: list[str] | None = None,
        role: str | None = None,
        limit: int = 10,
        include_decayed: bool = False,
        context: str | None = None,
        include_memories: bool = False,
        labels: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any] | None:
        """``POST /api/memories/ask`` -- question answering from memories."""
        body: dict[str, Any] = {"question": question, "limit": limit}
        if memory_type:
            body["memory_type"] = memory_type
        if categories:
            body["categories"] = categories
        if role:
            body["role"] = role
        if include_decayed:
            body["include_decayed"] = True
        if context:
            body["context"] = context
        if include_memories:
            body["include_memories"] = True
        if labels:
            body["labels"] = labels
        return self._post("/api/memories/ask", body, agent_id=agent_id)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_memory(
        self,
        *,
        content: str,
        memory_type: str | None = None,
        categories: list[str] | None = None,
        importance: str | None = None,
        pinned: bool | None = None,
        infer: bool | None = None,
        role: str | None = None,
        ttl_days: int | None = None,
        event_date: str | None = None,
        labels: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any] | None:
        """``POST /api/memories`` -- add a single memory."""
        body: dict[str, Any] = {"content": content}
        if memory_type is not None:
            body["memory_type"] = memory_type
        if categories is not None:
            body["categories"] = categories
        if importance is not None:
            body["importance"] = importance
        if pinned is not None:
            body["pinned"] = pinned
        if infer is not None:
            body["infer"] = infer
        if role is not None:
            body["role"] = role
        if ttl_days is not None:
            body["ttl_days"] = ttl_days
        if event_date is not None:
            body["event_date"] = event_date
        if labels is not None:
            body["labels"] = labels
        return self._post("/api/memories", body, agent_id=agent_id)

    def add_memories_batch(
        self,
        *,
        memories: list[dict[str, Any]],
        agent_id: str | None = None,
    ) -> dict[str, Any] | None:
        """``POST /api/memories/batch`` -- add multiple memories."""
        items = []
        for m in memories:
            item: dict[str, Any] = {"content": m["content"]}
            for key in (
                "memory_type",
                "categories",
                "importance",
                "pinned",
                "infer",
                "role",
                "ttl_days",
                "event_date",
                "labels",
            ):
                if m.get(key) is not None:
                    item[key] = m[key]
            items.append(item)
        return self._post(
            "/api/memories/batch",
            {"memories": items},
            agent_id=agent_id,
            timeout=self._timeout * 5,
        )

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        memory_type: str | None = None,
        categories: list[str] | None = None,
        importance: str | None = None,
        pinned: bool | None = None,
        ttl_days: int | None = None,
        event_date: str | None = None,
        labels: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ) -> bool:
        """``PUT /api/memories/:id`` -- update a memory."""
        body: dict[str, Any] = {}
        if content is not None:
            body["content"] = content
        if memory_type is not None:
            body["memory_type"] = memory_type
        if categories is not None:
            body["categories"] = categories
        if importance is not None:
            body["importance"] = importance
        if pinned is not None:
            body["pinned"] = pinned
        if ttl_days is not None:
            body["ttl_days"] = ttl_days
        if event_date is not None:
            body["event_date"] = event_date
        if labels is not None:
            body["labels"] = labels
        return self._put(
            f"/api/memories/{self._enc(memory_id)}", body, agent_id=agent_id
        )

    def delete_memory(self, memory_id: str, *, agent_id: str | None = None) -> bool:
        """``DELETE /api/memories/:id`` -- delete a memory."""
        return self._delete(f"/api/memories/{self._enc(memory_id)}", agent_id=agent_id)

    def delete_memories_batch(
        self,
        memory_ids: list[str],
        *,
        agent_id: str | None = None,
    ) -> dict[str, Any] | None:
        """``POST /api/memories/batch/delete`` -- delete multiple memories."""
        return self._post(
            "/api/memories/batch/delete", {"memory_ids": memory_ids}, agent_id=agent_id
        )

    # ------------------------------------------------------------------
    # List / Recent / Categories
    # ------------------------------------------------------------------

    def list_memories(
        self,
        *,
        memory_type: str | None = None,
        categories: list[str] | None = None,
        role: str | None = None,
        limit: int | None = None,
        include_decayed: bool = False,
        labels: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any] | None:
        """``GET /api/memories`` -- list memories with optional filters."""
        params: list[str] = []
        if memory_type:
            params.append(f"memory_type={quote(memory_type, safe='')}")
        if role:
            params.append(f"role={quote(role, safe='')}")
        if limit is not None:
            params.append(f"limit={limit}")
        if include_decayed:
            params.append("include_decayed=true")
        if categories:
            for cat in categories:
                params.append(f"categories={quote(cat, safe='')}")
        if labels:
            params.append(f"labels={quote(json.dumps(labels), safe='')}")
        qs = f"?{'&'.join(params)}" if params else ""
        return self._get(f"/api/memories{qs}", agent_id=agent_id)  # type: ignore[return-value]

    def get_recent_memories(
        self,
        *,
        days: int | None = None,
        scope: str | None = None,
        limit: int | None = None,
        include_decayed: bool = False,
        agent_id: str | None = None,
    ) -> dict[str, Any] | None:
        """``GET /api/memories/recent`` -- recent memories from last N days."""
        params: list[str] = []
        if days is not None:
            params.append(f"days={days}")
        if scope:
            params.append(f"scope={quote(scope, safe='')}")
        if limit is not None:
            params.append(f"limit={limit}")
        if include_decayed:
            params.append("include_decayed=true")
        qs = f"?{'&'.join(params)}" if params else ""
        return self._get(f"/api/memories/recent{qs}", agent_id=agent_id)  # type: ignore[return-value]

    def list_categories(self, *, agent_id: str | None = None) -> dict[str, Any] | None:
        """``GET /api/categories`` -- list all memory categories."""
        return self._get("/api/categories", agent_id=agent_id)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def save_artifact(
        self,
        memory_id: str,
        *,
        content: str,
        filename: str = "note.md",
        content_type: str = "text/markdown",
        agent_id: str | None = None,
    ) -> dict[str, Any] | None:
        """``POST /api/memories/:id/artifacts`` -- attach an artifact."""
        return self._post(
            f"/api/memories/{self._enc(memory_id)}/artifacts",
            {"content": content, "filename": filename, "content_type": content_type},
            agent_id=agent_id,
        )

    def get_artifact(
        self,
        memory_id: str,
        artifact_id: str,
        *,
        offset: int | None = None,
        limit: int | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any] | None:
        """``GET /api/memories/:id/artifacts/:aid`` -- retrieve artifact content."""
        params: list[str] = []
        if offset is not None:
            params.append(f"offset={offset}")
        if limit is not None:
            params.append(f"limit={limit}")
        qs = f"?{'&'.join(params)}" if params else ""
        return self._get(
            f"/api/memories/{self._enc(memory_id)}/artifacts/{self._enc(artifact_id)}{qs}",
            agent_id=agent_id,
        )  # type: ignore[return-value]

    def get_artifact_url(
        self,
        memory_id: str,
        artifact_id: str,
        *,
        ttl: int | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any] | None:
        """``GET /api/memories/:id/artifacts/:aid/url`` -- signed download URL."""
        params: list[str] = []
        if ttl is not None:
            params.append(f"ttl={ttl}")
        qs = f"?{'&'.join(params)}" if params else ""
        return self._get(
            f"/api/memories/{self._enc(memory_id)}/artifacts/{self._enc(artifact_id)}/url{qs}",
            agent_id=agent_id,
        )  # type: ignore[return-value]

    def list_artifacts(
        self,
        memory_id: str,
        *,
        agent_id: str | None = None,
    ) -> list[dict[str, Any]] | None:
        """``GET /api/memories/:id/artifacts`` -- list artifacts on a memory."""
        return self._get(
            f"/api/memories/{self._enc(memory_id)}/artifacts",
            agent_id=agent_id,
        )  # type: ignore[return-value]

    def delete_artifact(
        self,
        memory_id: str,
        artifact_id: str,
        *,
        agent_id: str | None = None,
    ) -> bool:
        """``DELETE /api/memories/:id/artifacts/:aid`` -- delete an artifact."""
        return self._delete(
            f"/api/memories/{self._enc(memory_id)}/artifacts/{self._enc(artifact_id)}",
            agent_id=agent_id,
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self, timeout: float = 5) -> bool:
        """``GET /health`` -- check if the mnemory server is reachable."""
        try:
            resp = self._session.get(
                f"{self._base_url}/health",
                headers=self._headers(),
                timeout=timeout,
            )
            return resp.ok
        except Exception:
            return False
