"""Vector store abstraction wrapping mem0 with direct backend access.

mem0 handles the core operations (add, search, get_all, update, delete)
including LLM-driven fact extraction and deduplication. For operations
mem0 doesn't support (date filtering, metadata-only updates, null field
queries), we access the vector backend directly.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from mem0 import Memory

from mnemory.config import Config

logger = logging.getLogger(__name__)


class VectorStore:
    """Wrapper around mem0 Memory with direct backend access for advanced queries."""

    def __init__(self, config: Config):
        self._config = config
        self._mem0_config = config.build_mem0_config()
        self.memory = Memory.from_config(self._mem0_config)
        self._qdrant_client = None

        # Patch mem0's Qdrant vector store to handle vector=None correctly.
        # mem0 calls update(vector=None) to update only metadata/session IDs
        # (e.g., on NONE events with agent_id set), but the Qdrant
        # implementation passes None to PointStruct which fails with
        # validation errors. Use set_payload() for metadata-only updates.
        if hasattr(self.memory, "vector_store") and hasattr(
            self.memory.vector_store, "client"
        ):
            _original_vs_update = self.memory.vector_store.update
            _vs = self.memory.vector_store

            def _patched_vs_update(vector_id, vector=None, payload=None):
                if vector is None and payload is not None:
                    _vs.client.set_payload(
                        collection_name=_vs.collection_name,
                        payload=payload,
                        points=[vector_id],
                    )
                else:
                    _original_vs_update(vector_id, vector=vector, payload=payload)

            self.memory.vector_store.update = _patched_vs_update

    @property
    def qdrant_client(self):
        """Lazy-initialized direct Qdrant client.

        Used for operations mem0 doesn't support:
        - Date/time filtering (created_at range queries)
        - Metadata-only updates (set_payload)
        - Null field queries (memories without agent_id)
        """
        if self._config.vector.backend != "qdrant":
            raise RuntimeError(
                "Direct Qdrant client is only available with VECTOR_BACKEND=qdrant"
            )
        if self._qdrant_client is None:
            from qdrant_client import QdrantClient

            kwargs: dict[str, Any] = {
                "host": self._config.vector.qdrant_host,
                "port": self._config.vector.qdrant_port,
            }
            if self._config.vector.qdrant_api_key:
                kwargs["api_key"] = self._config.vector.qdrant_api_key
            self._qdrant_client = QdrantClient(**kwargs)
        return self._qdrant_client

    @property
    def collection_name(self) -> str:
        return self._config.vector.collection_name

    # ── mem0 pass-through operations ──────────────────────────────────

    def add(
        self,
        content: str,
        *,
        user_id: str,
        agent_id: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Add a memory via mem0 (with fact extraction and deduplication)."""
        kwargs: dict[str, Any] = {"user_id": user_id}
        if agent_id:
            kwargs["agent_id"] = agent_id
        if metadata:
            kwargs["metadata"] = metadata
        return self.memory.add(content, **kwargs)

    def search(
        self,
        query: str,
        *,
        user_id: str,
        agent_id: str | None = None,
        filters: dict | None = None,
        limit: int = 100,
    ) -> dict:
        """Search memories via mem0."""
        kwargs: dict[str, Any] = {"user_id": user_id, "limit": limit}
        if agent_id:
            kwargs["agent_id"] = agent_id
        if filters:
            kwargs["filters"] = filters
        return self.memory.search(query, **kwargs)

    def get_all(
        self,
        *,
        user_id: str,
        agent_id: str | None = None,
        filters: dict | None = None,
        limit: int = 100,
    ) -> dict:
        """Get all memories via mem0."""
        kwargs: dict[str, Any] = {"user_id": user_id, "limit": limit}
        if agent_id:
            kwargs["agent_id"] = agent_id
        if filters:
            kwargs["filters"] = filters
        return self.memory.get_all(**kwargs)

    def get_by_id(self, memory_id: str) -> dict | None:
        """Get a single memory by ID via mem0.

        Returns the memory dict, or None if not found.
        """
        try:
            return self.memory.get(memory_id)
        except Exception:
            logger.debug("Memory %s not found", memory_id)
            return None

    def update(self, memory_id: str, content: str) -> dict:
        """Update memory content via mem0."""
        return self.memory.update(memory_id, data=content)

    def delete(self, memory_id: str) -> None:
        """Delete a memory via mem0."""
        self.memory.delete(memory_id)

    def delete_all(self, *, user_id: str, agent_id: str | None = None) -> None:
        """Delete all memories via mem0."""
        kwargs: dict[str, Any] = {"user_id": user_id}
        if agent_id:
            kwargs["agent_id"] = agent_id
        self.memory.delete_all(**kwargs)

    # ── Direct backend operations (Qdrant-specific) ──────────────────

    def update_metadata(self, memory_id: str, metadata: dict) -> None:
        """Update metadata fields on a memory without changing content.

        Uses Qdrant's set_payload for partial updates. For Chroma backend,
        falls back to a full read-modify-write cycle.
        """
        if self._config.vector.backend == "qdrant":
            self.qdrant_client.set_payload(
                collection_name=self.collection_name,
                payload=metadata,
                points=[memory_id],
            )
        else:
            # Chroma fallback: read, merge, write back via mem0 update.
            # Less efficient but works for the local backend.
            logger.warning(
                "Metadata-only update on non-Qdrant backend uses full rewrite"
            )
            mem = self.get_by_id(memory_id)
            if mem:
                existing_meta = mem.get("metadata", {})
                existing_meta.update(metadata)
                # mem0's update() only changes content, so we re-add with
                # merged metadata. This triggers dedup but preserves metadata.
                content = mem.get("memory", "")
                self.memory.update(memory_id, data=content)
                # After content update, attempt to set metadata via mem0's
                # internal vector store if accessible
                try:
                    self.memory.update(memory_id, data=content, metadata=existing_meta)
                except TypeError:
                    # mem0 version may not support metadata kwarg on update;
                    # metadata update is best-effort on Chroma backend
                    logger.warning(
                        "Chroma backend: metadata update not fully supported "
                        "by this mem0 version. Metadata changes may be lost."
                    )

    def get_recent_memories(
        self,
        *,
        user_id: str,
        agent_id: str | None = None,
        since: datetime,
        limit: int = 50,
    ) -> list[dict]:
        """Get memories created after a given timestamp.

        Uses direct Qdrant DatetimeRange filtering since mem0 doesn't
        support date queries. For Chroma backend, falls back to get_all
        with post-filtering.
        """
        if self._config.vector.backend == "qdrant":
            return self._qdrant_recent_memories(
                user_id=user_id, agent_id=agent_id, since=since, limit=limit
            )
        else:
            return self._fallback_recent_memories(
                user_id=user_id, agent_id=agent_id, since=since, limit=limit
            )

    def get_pinned_memories(
        self,
        *,
        user_id: str,
        agent_id: str | None = None,
        exclude_agent: bool = False,
    ) -> list[dict]:
        """Get pinned memories, optionally filtering by agent scope.

        Args:
            user_id: Required user scope.
            agent_id: If set, also include agent-specific pinned memories.
            exclude_agent: If True, only return memories WITHOUT agent_id
                          (user-scoped memories shared across agents).
        """
        # mem0 can filter by pinned=True
        all_pinned = self.get_all(user_id=user_id, filters={"pinned": True}, limit=500)
        results = all_pinned.get("results", [])

        filtered = []
        for mem in results:
            mem_agent_id = mem.get("agent_id")
            if exclude_agent and mem_agent_id:
                continue
            if (
                not exclude_agent
                and agent_id
                and mem_agent_id
                and mem_agent_id != agent_id
            ):
                continue
            filtered.append(mem)

        return filtered

    def _qdrant_recent_memories(
        self,
        *,
        user_id: str,
        agent_id: str | None,
        since: datetime,
        limit: int,
    ) -> list[dict]:
        """Fetch recent memories using Qdrant's DatetimeRange filter."""
        from qdrant_client.models import (
            DatetimeRange,
            FieldCondition,
            Filter,
            MatchValue,
        )

        must_conditions = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            FieldCondition(
                key="created_at",
                range=DatetimeRange(gte=since.isoformat()),
            ),
        ]
        if agent_id:
            must_conditions.append(
                FieldCondition(key="agent_id", match=MatchValue(value=agent_id))
            )

        points, _ = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=must_conditions),
            limit=limit,
            with_payload=True,
        )

        return [self._qdrant_point_to_memory(p) for p in points]

    def _fallback_recent_memories(
        self,
        *,
        user_id: str,
        agent_id: str | None,
        since: datetime,
        limit: int,
    ) -> list[dict]:
        """Fallback for non-Qdrant backends: get_all + post-filter by date."""
        kwargs: dict[str, Any] = {"user_id": user_id, "limit": 1000}
        if agent_id:
            kwargs["agent_id"] = agent_id
        all_memories = self.memory.get_all(**kwargs)

        since_str = since.isoformat()
        results = []
        for mem in all_memories.get("results", []):
            created = mem.get("created_at", "")
            if created and created >= since_str:
                results.append(mem)

        # Sort by created_at ascending (chronological)
        results.sort(key=lambda m: m.get("created_at", ""))
        return results[:limit]

    @staticmethod
    def _qdrant_point_to_memory(point) -> dict:
        """Convert a Qdrant point to a mem0-style memory dict."""
        payload = point.payload or {}
        memory = {
            "id": str(point.id),
            "memory": payload.get("data", ""),
            "hash": payload.get("hash", ""),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
        }
        # Promote standard fields
        for field in ("user_id", "agent_id", "run_id"):
            if field in payload:
                memory[field] = payload[field]
        # Collect remaining fields as metadata
        skip_keys = {
            "data",
            "hash",
            "created_at",
            "updated_at",
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        }
        metadata = {k: v for k, v in payload.items() if k not in skip_keys}
        if metadata:
            memory["metadata"] = metadata
        return memory
