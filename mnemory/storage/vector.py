"""Vector store using Qdrant directly.

Handles all vector operations: insert, search, update, delete, scroll.
Supports both remote Qdrant (production) and local/embedded Qdrant
(development) via qdrant-client's path mode.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    OrderBy,
    PayloadField,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from mnemory.config import Config
from mnemory.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)

# Use SHA-256 for content hashing (dedup, not security-critical,
# but avoids MD5 deprecation warnings from security scanners).
_hash = hashlib.sha256


class VectorStore:
    """Direct Qdrant vector store for memory operations.

    Supports two modes:
    - Remote: connects to a Qdrant server (QDRANT_HOST set)
    - Local: embedded Qdrant via qdrant-client path mode (no server needed)
    """

    def __init__(self, config: Config):
        self._config = config
        self._embedding = EmbeddingClient(config.embed)
        self._client = self._create_client(config)
        self._ensure_collection()

        # Local embedded Qdrant uses SQLite which isn't thread-safe for
        # concurrent writes. A lock serializes write operations in local
        # mode. Remote Qdrant handles concurrency server-side, so no
        # lock is needed.
        self._write_lock: threading.Lock | None = (
            threading.Lock() if not config.vector.is_remote else None
        )

    @contextmanager
    def _write_guard(self):
        """Acquire write lock for local embedded mode (no-op for remote)."""
        if self._write_lock is not None:
            with self._write_lock:
                yield
        else:
            yield

    @staticmethod
    def _create_client(config: Config) -> QdrantClient:
        """Create a Qdrant client based on configuration."""
        vc = config.vector
        if vc.is_remote:
            kwargs: dict[str, Any] = {
                "host": vc.qdrant_host,
                "port": vc.qdrant_port,
            }
            if vc.qdrant_api_key:
                kwargs["api_key"] = vc.qdrant_api_key
            logger.info(
                "Connecting to remote Qdrant at %s:%d", vc.qdrant_host, vc.qdrant_port
            )
            return QdrantClient(**kwargs)
        else:
            logger.info("Using local Qdrant at %s", vc.qdrant_path)
            return QdrantClient(path=vc.qdrant_path)

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't exist, then ensure indexes."""
        name = self.collection_name
        existing = [c.name for c in self._client.get_collections().collections]
        if name not in existing:
            logger.info(
                "Creating collection '%s' (dims=%d, distance=COSINE)",
                name,
                self._embedding.dims,
            )
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self._embedding.dims,
                    distance=Distance.COSINE,
                ),
            )
        else:
            logger.debug("Collection '%s' already exists", name)

        # Ensure payload indexes exist (idempotent — safe to run every startup)
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """Create payload indexes for efficient filtering and ordering.

        Runs on every startup (not just collection creation) to ensure
        indexes exist for existing collections that were created before
        new indexes were added. Qdrant's create_payload_index is
        idempotent — it no-ops if the index already exists.

        Only applies to remote Qdrant; local embedded mode doesn't
        need explicit indexes.
        """
        if not self._config.vector.is_remote:
            return

        name = self.collection_name
        # Keyword/bool indexes for filtering
        for field in ("user_id", "agent_id", "pinned", "memory_type"):
            try:
                self._client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema="keyword" if field != "pinned" else "bool",
                )
            except Exception:
                pass  # Index may already exist
        # Datetime indexes for DatetimeRange filters and order_by
        for field in ("created_at", "expires_at"):
            try:
                self._client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema="datetime",
                )
            except Exception:
                pass  # Index may already exist
        # Keyword index for event_date (stored as YYYY-MM-DD string,
        # supports lexicographic Range filtering for date queries)
        try:
            self._client.create_payload_index(
                collection_name=name,
                field_name="event_date",
                field_schema="keyword",
            )
        except Exception:
            pass  # Index may already exist

    @property
    def collection_name(self) -> str:
        return self._config.vector.collection_name

    @property
    def embedding(self) -> EmbeddingClient:
        """Access the embedding client for external use."""
        return self._embedding

    # ── Core operations ──────────────────────────────────────────────

    def insert(
        self,
        *,
        text: str,
        vector: list[float],
        user_id: str,
        agent_id: str | None = None,
        metadata: dict[str, Any],
        role: str = "user",
    ) -> str:
        """Insert a new memory point.

        Args:
            text: The memory content text.
            vector: Pre-computed embedding vector.
            user_id: User scope.
            agent_id: Optional agent scope.
            metadata: Custom metadata fields (memory_type, categories, etc.).
            role: "user" or "assistant".

        Returns:
            The generated memory ID (UUID string).
        """
        memory_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        payload: dict[str, Any] = {
            "data": text,
            "hash": _hash(text.encode()).hexdigest(),
            "created_at": now.isoformat(),
            "user_id": user_id,
            "role": role,
        }
        if agent_id:
            payload["agent_id"] = agent_id

        # Merge custom metadata (memory_type, categories, importance, etc.)
        payload.update(metadata)

        with self._write_guard():
            self._client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(id=memory_id, vector=vector, payload=payload)],
            )
        return memory_id

    def insert_batch(
        self,
        points: list[dict[str, Any]],
    ) -> list[str]:
        """Insert multiple memory points in a single call.

        Each point dict should have: text, vector, user_id, agent_id (optional),
        metadata, role.

        Returns list of generated memory IDs.
        """
        if not points:
            return []

        now = datetime.now(timezone.utc)
        structs = []
        ids = []

        for p in points:
            memory_id = str(uuid.uuid4())
            ids.append(memory_id)

            payload: dict[str, Any] = {
                "data": p["text"],
                "hash": _hash(p["text"].encode()).hexdigest(),
                "created_at": now.isoformat(),
                "user_id": p["user_id"],
                "role": p.get("role", "user"),
            }
            if p.get("agent_id"):
                payload["agent_id"] = p["agent_id"]
            payload.update(p.get("metadata", {}))

            structs.append(
                PointStruct(id=memory_id, vector=p["vector"], payload=payload)
            )

        with self._write_guard():
            self._client.upsert(
                collection_name=self.collection_name,
                points=structs,
            )
        return ids

    def search(
        self,
        query: str,
        *,
        user_id: str,
        agent_id: str | None = None,
        shared_only: bool = False,
        filters: dict | None = None,
        categories: list[str] | None = None,
        limit: int = 100,
        exclude_expired: bool = False,
        include_decayed: bool = False,
        similarity_weight: float = 0.9,
        query_vector: list[float] | None = None,
        date_start: str | None = None,
        date_end: str | None = None,
    ) -> dict:
        """Semantic search with TTL, category, and importance filtering.

        Uses Qdrant's query_points() with native filtering and
        formula-based importance reranking.

        Args:
            query: Search query text.
            user_id: Required user scope.
            agent_id: Optional agent scope.
            shared_only: If True and agent_id is None, restrict to memories
                without any agent_id (shared user memories only). Used by
                dual-scope search to avoid leaking sub-agent memories.
            filters: Simple key-value metadata filters (memory_type, role).
            categories: Category filter (OR logic via MatchAny).
            limit: Maximum results.
            exclude_expired: If True, filter out expired/decayed memories.
            include_decayed: If True, include decayed memories.
            similarity_weight: Weight for cosine similarity in the combined
                score formula (0.0-1.0). Importance gets 1 - similarity_weight.
                Default 0.9 (90% similarity, 10% importance).
            query_vector: Pre-computed embedding vector. If provided, skips
                the embedding API call. Used by find_memories for batch
                embedding optimization.
            date_start: Optional start date (YYYY-MM-DD) for temporal
                filtering. Matches memories where event_date >= start OR
                (event_date is missing AND created_at >= start).
            date_end: Optional end date (YYYY-MM-DD) for temporal filtering.
                Matches memories where event_date <= end OR (event_date is
                missing AND created_at <= end).

        Returns:
            Dict with "results" key containing list of memory dicts.
        """
        from qdrant_client.models import (
            DatetimeRange,
            IsEmptyCondition,
            MatchAny,
            Prefetch,
        )

        from mnemory.categories import IMPORTANCE_WEIGHTS

        # 1. Embed the query (skip if pre-computed vector provided)
        embeddings = (
            query_vector if query_vector is not None else self._embedding.embed(query)
        )

        # 2. Build filter conditions
        must_conditions: list = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
        ]
        if agent_id:
            must_conditions.append(
                FieldCondition(key="agent_id", match=MatchValue(value=agent_id))
            )
        elif shared_only:
            # Restrict to memories without any agent_id (shared user memories).
            # Use IsEmptyCondition (not IsNullCondition) because agent_id is
            # omitted from the payload for shared memories — IsNullCondition
            # only matches keys explicitly set to null, not absent keys.
            must_conditions.append(
                IsEmptyCondition(is_empty=PayloadField(key="agent_id"))
            )
        # Metadata filters (memory_type, role)
        if filters:
            for key, value in filters.items():
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        # Category filter (expanded list, OR logic via MatchAny)
        if categories:
            must_conditions.append(
                FieldCondition(key="categories", match=MatchAny(any=categories))
            )
        # TTL filter: active memories only.
        # Use IsEmptyCondition (not IsNullCondition) because legacy memories
        # may not have expires_at in their payload at all — IsNullCondition
        # only matches keys explicitly set to null, not absent keys.
        if exclude_expired and not include_decayed:
            now = datetime.now(timezone.utc)
            ttl_filter = Filter(
                should=[
                    IsEmptyCondition(is_empty=PayloadField(key="expires_at")),
                    FieldCondition(key="expires_at", range=DatetimeRange(gt=now)),
                    FieldCondition(key="pinned", match=MatchValue(value=True)),
                ]
            )
            must_conditions.append(ttl_filter)

        # Date range filter: match memories by event_date OR created_at.
        # Uses OR logic: event_date in range, OR (no event_date AND
        # created_at in range). This ensures memories without event_date
        # are still found based on when they were stored.
        if date_start or date_end:
            date_conditions = self._build_date_range_filter(date_start, date_end)
            if date_conditions:
                must_conditions.append(date_conditions)

        query_filter = Filter(must=must_conditions)

        # 3. Build formula for importance reranking
        importance_weight = 1.0 - similarity_weight
        try:
            from qdrant_client.models import (
                FormulaQuery,
                SumExpression,
            )

            formula_terms: list = [
                {"mult": [similarity_weight, "$score"]},
            ]
            for imp_level, imp_weight in IMPORTANCE_WEIGHTS.items():
                boost = importance_weight * imp_weight
                if boost > 0:
                    formula_terms.append(
                        {
                            "mult": [
                                boost,
                                {
                                    "key": "importance",
                                    "match": {"value": imp_level},
                                },
                            ]
                        }
                    )

            result = self._client.query_points(
                collection_name=self.collection_name,
                prefetch=Prefetch(
                    query=embeddings,
                    filter=query_filter,
                    limit=limit,
                ),
                query=FormulaQuery(formula=SumExpression(sum=formula_terms)),
                limit=limit,
                with_payload=True,
            )
        except Exception:
            # Fallback: if formula query fails (e.g., older Qdrant version)
            logger.warning(
                "Formula-based reranking failed, falling back to simple search"
            )
            result = self._client.query_points(
                collection_name=self.collection_name,
                query=embeddings,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )

        # 4. Convert to memory dicts
        memories = []
        for point in result.points:
            mem = self._point_to_memory(point)
            mem["score"] = point.score
            memories.append(mem)

        return {"results": memories}

    def search_similar(
        self,
        vector: list[float],
        *,
        user_id: str,
        agent_id: str | None = None,
        shared_only: bool = False,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search for similar existing memories using a pre-computed vector.

        Used during the add pipeline to find candidates for deduplication.
        Excludes expired/decayed memories so the LLM doesn't deduplicate
        against memories the user can no longer see.

        Args:
            vector: Pre-computed embedding vector.
            user_id: Required user scope.
            agent_id: Optional agent scope.
            shared_only: If True and agent_id is None, restrict to memories
                without any agent_id (shared user memories only). Used by
                dual-scope dedup to search shared memories separately.
            limit: Maximum results.

        Returns simple dicts with "id" and "text" keys.
        """
        from qdrant_client.models import (
            DatetimeRange,
            IsEmptyCondition,
        )

        must_conditions: list = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
        ]
        if agent_id:
            must_conditions.append(
                FieldCondition(key="agent_id", match=MatchValue(value=agent_id))
            )
        elif shared_only:
            must_conditions.append(
                IsEmptyCondition(is_empty=PayloadField(key="agent_id"))
            )

        # Exclude expired/decayed memories from dedup candidates.
        # Use IsEmptyCondition for the same reason as in search().
        now = datetime.now(timezone.utc)
        ttl_filter = Filter(
            should=[
                IsEmptyCondition(is_empty=PayloadField(key="expires_at")),
                FieldCondition(key="expires_at", range=DatetimeRange(gt=now)),
                FieldCondition(key="pinned", match=MatchValue(value=True)),
            ]
        )
        must_conditions.append(ttl_filter)

        result = self._client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=Filter(must=must_conditions),
            limit=limit,
            with_payload=True,
        )

        return [
            {
                "id": str(p.id),
                "text": p.payload.get("data", ""),
                "score": p.score,
                "type": p.payload.get("memory_type", ""),
                "categories": p.payload.get("categories", []),
            }
            for p in result.points
        ]

    def get_by_id(self, memory_id: str) -> dict | None:
        """Get a single memory by ID.

        Returns the memory dict (in standard format), or None if not found.
        """
        try:
            result = self._client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id],
                with_payload=True,
            )
            if result:
                return self._point_to_memory(result[0])
            return None
        except Exception:
            logger.debug("Memory %s not found", memory_id)
            return None

    def get_all(
        self,
        *,
        user_id: str,
        agent_id: str | None = None,
        shared_only: bool = False,
        filters: dict | None = None,
        limit: int = 100,
        order_by: OrderBy | None = None,
    ) -> dict:
        """List memories with optional filters and ordering.

        Args:
            user_id: Required user scope.
            agent_id: Optional agent scope.
            shared_only: If True and agent_id is None, restrict to memories
                without any agent_id (shared user memories only). Used by
                dual-scope list to avoid leaking sub-agent memories.
            filters: Metadata filters. Scalar values use exact match,
                list values use any-of match (MatchAny).
            limit: Maximum results.
            order_by: Optional Qdrant OrderBy for server-side sorting.
                When set, results are returned in the specified order.
                Note: points missing the ordered field are excluded.

        Returns dict with "results" key containing list of memory dicts.
        """
        from qdrant_client.models import IsEmptyCondition, MatchAny

        must_conditions: list = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
        ]
        if agent_id:
            must_conditions.append(
                FieldCondition(key="agent_id", match=MatchValue(value=agent_id))
            )
        elif shared_only:
            must_conditions.append(
                IsEmptyCondition(is_empty=PayloadField(key="agent_id"))
            )
        if filters:
            for key, value in filters.items():
                if isinstance(value, list):
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=value))
                    )
                else:
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )

        scroll_kwargs: dict[str, Any] = {
            "collection_name": self.collection_name,
            "scroll_filter": Filter(must=must_conditions),
            "limit": limit,
            "with_payload": True,
            "with_vectors": False,
        }
        if order_by is not None:
            scroll_kwargs["order_by"] = order_by

        try:
            points, _ = self._client.scroll(**scroll_kwargs)
        except Exception:
            if order_by is not None:
                # Fallback: order_by failed (e.g., missing payload index).
                # Retry without order_by and sort in Python instead.
                logger.warning(
                    "order_by scroll failed (missing index?), "
                    "falling back to client-side sort"
                )
                scroll_kwargs.pop("order_by", None)
                points, _ = self._client.scroll(**scroll_kwargs)
                results = [self._point_to_memory(p) for p in points]
                # Replicate the requested sort order in Python
                reverse = (
                    order_by.direction is None
                    or str(order_by.direction).lower() != "asc"
                )
                results.sort(
                    key=lambda m: m.get(order_by.key, ""),  # type: ignore[union-attr]
                    reverse=reverse,
                )
                return {"results": results}
            raise

        return {"results": [self._point_to_memory(p) for p in points]}

    def update_content(
        self,
        memory_id: str,
        text: str,
        vector: list[float] | None = None,
    ) -> None:
        """Update a memory's content text and re-embed.

        Preserves all existing metadata. Only changes: data, hash,
        updated_at, and the embedding vector.

        Args:
            memory_id: The memory to update.
            text: New content text.
            vector: Pre-computed embedding vector. If None, the text
                    is re-embedded automatically.
        """
        # Fetch existing payload to preserve metadata
        existing = self._client.retrieve(
            collection_name=self.collection_name,
            ids=[memory_id],
            with_payload=True,
            with_vectors=False,
        )
        if not existing:
            raise ValueError(f"Memory {memory_id} not found")

        payload = dict(existing[0].payload or {})
        payload["data"] = text
        payload["hash"] = _hash(text.encode()).hexdigest()
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Use pre-computed vector or re-embed
        if vector is None:
            vector = self._embedding.embed(text)

        with self._write_guard():
            self._client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(id=memory_id, vector=vector, payload=payload)],
            )

    def update_metadata(self, memory_id: str, metadata: dict) -> None:
        """Update metadata fields on a memory without changing content.

        Uses Qdrant's set_payload for efficient partial updates.
        """
        with self._write_guard():
            self._client.set_payload(
                collection_name=self.collection_name,
                payload=metadata,
                points=[memory_id],
            )

    def batch_update_metadata(self, updates: list[tuple[str, dict]]) -> None:
        """Batch update metadata on multiple memories.

        Args:
            updates: List of (memory_id, metadata_dict) tuples.
        """
        with self._write_guard():
            for memory_id, metadata in updates:
                try:
                    self._client.set_payload(
                        collection_name=self.collection_name,
                        payload=metadata,
                        points=[memory_id],
                    )
                except Exception:
                    logger.warning("Failed to update metadata for memory %s", memory_id)

    def delete(self, memory_id: str) -> None:
        """Delete a single memory by ID."""
        with self._write_guard():
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=[memory_id]),
            )

    def delete_all(self, *, user_id: str, agent_id: str | None = None) -> None:
        """Delete all memories for a user/agent scope.

        Uses filter-based deletion — does NOT destroy the collection.
        """
        must_conditions: list = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
        ]
        if agent_id:
            must_conditions.append(
                FieldCondition(key="agent_id", match=MatchValue(value=agent_id))
            )

        with self._write_guard():
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(must=must_conditions),
            )

    def artifact_has_references(
        self,
        *,
        artifact_id: str,
        exclude_memory_id: str,
    ) -> bool:
        """Check if any memory (other than exclude_memory_id) references an artifact.

        Uses Qdrant nested field filtering on artifacts[].id to find
        memories that reference the given artifact_id, excluding the
        specified memory.

        Returns True if at least one other memory references the artifact.
        """
        from qdrant_client.models import HasIdCondition

        scroll_filter = Filter(
            must=[
                FieldCondition(
                    key="artifacts[].id",
                    match=MatchValue(value=artifact_id),
                ),
            ],
            must_not=[
                HasIdCondition(has_id=[exclude_memory_id]),
            ],
        )

        results, _ = self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            limit=1,
        )

        return len(results) > 0

    # ── Fsck helpers ───────────────────────────────────────────────────

    def scroll_with_vectors(
        self,
        *,
        user_id: str,
        agent_id: str | None = None,
        shared_only: bool = False,
        filters: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Scroll ALL memories for a user, including stored embedding vectors.

        Used by the fsck (memory check) feature to retrieve stored vectors
        for similarity comparison without re-embedding.

        Paginates through all Qdrant points using the scroll cursor so that
        users with large memory sets are fully covered — no hard limit.

        Args:
            user_id: Required user scope.
            agent_id: Optional agent scope.
            shared_only: If True and agent_id is None, restrict to memories
                without any agent_id (shared user memories only). Used by
                dual-scope fsck to scroll shared memories separately.
            filters: Additional metadata filters.

        Returns list of memory dicts with an extra "vector" key containing
        the stored embedding.
        """
        from qdrant_client.models import IsEmptyCondition, MatchAny

        must_conditions: list = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
        ]
        if agent_id:
            must_conditions.append(
                FieldCondition(key="agent_id", match=MatchValue(value=agent_id))
            )
        elif shared_only:
            must_conditions.append(
                IsEmptyCondition(is_empty=PayloadField(key="agent_id"))
            )
        if filters:
            for key, value in filters.items():
                if isinstance(value, list):
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=value))
                    )
                else:
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )

        scroll_filter = Filter(must=must_conditions)
        # Use a page size that balances memory and round-trips.
        # with_vectors=True makes payloads larger, so keep pages moderate.
        _PAGE_SIZE = 256

        results = []
        offset = None  # None = start from beginning

        while True:
            points, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=_PAGE_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for p in points:
                mem = self._point_to_memory(p)
                mem["vector"] = p.vector
                results.append(mem)

            if next_offset is None or len(points) < _PAGE_SIZE:
                break
            offset = next_offset

        return results

    def search_by_vector(
        self,
        vector: list[float],
        *,
        user_id: str,
        agent_id: str | None = None,
        shared_only: bool = False,
        limit: int = 5,
        exclude_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar memories using a pre-computed vector.

        Returns full memory dicts (not just id/text like search_similar).
        Used by fsck for duplicate detection with stored vectors.

        Args:
            vector: Pre-computed embedding vector.
            user_id: Required user scope.
            agent_id: Optional agent scope.
            shared_only: If True and agent_id is None, restrict to memories
                without any agent_id (shared user memories only). Used by
                dual-scope fsck to search shared memories separately.
            limit: Maximum results.
            exclude_ids: Point IDs to exclude from results.
        """
        from qdrant_client.models import HasIdCondition, IsEmptyCondition

        must_conditions: list = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
        ]
        if agent_id:
            must_conditions.append(
                FieldCondition(key="agent_id", match=MatchValue(value=agent_id))
            )
        elif shared_only:
            must_conditions.append(
                IsEmptyCondition(is_empty=PayloadField(key="agent_id"))
            )

        must_not: list = []
        if exclude_ids:
            must_not.append(HasIdCondition(has_id=exclude_ids))

        result = self._client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=Filter(must=must_conditions, must_not=must_not or None),
            limit=limit,
            with_payload=True,
        )

        memories = []
        for point in result.points:
            mem = self._point_to_memory(point)
            mem["score"] = point.score
            memories.append(mem)
        return memories

    # ── Specialized queries ──────────────────────────────────────────

    def get_recent_memories(
        self,
        *,
        user_id: str,
        agent_id: str | None = None,
        since: datetime,
        limit: int = 50,
        memory_types: list[str] | None = None,
    ) -> list[dict]:
        """Get memories created after a given timestamp.

        Uses Qdrant's DatetimeRange filtering for efficient date queries.

        Args:
            user_id: Required user scope.
            agent_id: If set, filter to only memories with this agent_id.
                      If None, returns memories without agent_id (shared).
            since: Only return memories created after this timestamp.
            limit: Maximum number of results.
            memory_types: If set, filter to only these memory types.

        Returns memories ordered by created_at descending (most recent first).
        """
        from qdrant_client.models import (
            DatetimeRange,
            IsEmptyCondition,
            MatchAny,
        )

        must_conditions: list = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            FieldCondition(
                key="created_at",
                range=DatetimeRange(gte=since),
            ),
        ]

        # Filter by agent_id: if provided, match exactly; if None, match only
        # memories without agent_id (shared user memories).
        # Use IsEmptyCondition (not IsNullCondition) because agent_id is
        # omitted from the payload for shared memories — IsNullCondition
        # only matches keys explicitly set to null, not absent keys.
        if agent_id:
            must_conditions.append(
                FieldCondition(key="agent_id", match=MatchValue(value=agent_id))
            )
        else:
            must_conditions.append(
                IsEmptyCondition(is_empty=PayloadField(key="agent_id"))
            )

        # Filter by memory types if specified
        if memory_types:
            must_conditions.append(
                FieldCondition(key="memory_type", match=MatchAny(any=memory_types))
            )

        points, _ = self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=must_conditions),
            limit=limit,
            with_payload=True,
        )

        results = [self._point_to_memory(p) for p in points]

        # Sort by created_at descending (most recent first)
        results.sort(key=lambda m: m.get("created_at", ""), reverse=True)

        return results

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
        # Fetch all pinned memories for the user
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

    # ── Date range filtering ────────────────────────────────────────

    @staticmethod
    def _build_date_range_filter(
        date_start: str | None,
        date_end: str | None,
    ) -> Filter | None:
        """Build a Qdrant filter for date range queries.

        Matches memories where:
        - event_date is in the range, OR
        - event_date is missing AND created_at is in the range

        This ensures memories without event_date metadata (legacy or
        timeless facts) are still found based on their creation date.

        Args:
            date_start: Start date (YYYY-MM-DD), inclusive.
            date_end: End date (YYYY-MM-DD), inclusive. The end date
                is extended to end-of-day (23:59:59) for created_at
                comparison.

        Returns:
            A Qdrant Filter with OR logic, or None if no dates provided.
        """
        from qdrant_client.models import (
            DatetimeRange,
            IsEmptyCondition,
        )

        if not date_start and not date_end:
            return None

        # Build range for event_date (YYYY-MM-DD string comparison)
        event_date_range: dict[str, Any] = {}
        created_at_range: dict[str, Any] = {}

        if date_start:
            # event_date is stored as YYYY-MM-DD string
            event_date_range["gte"] = date_start
            # created_at is stored as ISO 8601 datetime
            created_at_range["gte"] = datetime.fromisoformat(
                f"{date_start}T00:00:00+00:00"
            )
        if date_end:
            event_date_range["lte"] = date_end
            # End of day for created_at
            created_at_range["lte"] = datetime.fromisoformat(
                f"{date_end}T23:59:59+00:00"
            )

        # Branch 1: event_date exists and is in range
        event_date_conditions: list = []
        from qdrant_client.models import Range

        event_date_conditions.append(
            FieldCondition(key="event_date", range=Range(**event_date_range))
        )

        # Branch 2: event_date is missing AND created_at is in range
        no_event_date_conditions: list = [
            IsEmptyCondition(is_empty=PayloadField(key="event_date")),
            FieldCondition(key="created_at", range=DatetimeRange(**created_at_range)),
        ]

        return Filter(
            should=[
                Filter(must=event_date_conditions),
                Filter(must=no_event_date_conditions),
            ]
        )

    def list_user_ids(self) -> list[str]:
        """Return a sorted list of all distinct user_ids in the collection.

        Scrolls all points (payload only, no vectors) and collects unique
        user_id values. Handles pagination automatically.

        Returns:
            Sorted list of user_id strings.
        """
        user_ids: set[str] = set()
        offset = None
        batch_size = 256

        while True:
            points, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,
                limit=batch_size,
                offset=offset,
                with_payload=["user_id"],
                with_vectors=False,
            )
            for point in points:
                uid = (point.payload or {}).get("user_id")
                if uid:
                    user_ids.add(uid)
            if next_offset is None:
                break
            offset = next_offset

        return sorted(user_ids)

    # ── Result formatting ────────────────────────────────────────────

    @staticmethod
    def _point_to_memory(point) -> dict:
        """Convert a Qdrant point to a standard memory dict.

        Payload structure in Qdrant:
            data, hash, created_at, updated_at, user_id, agent_id,
            role, + custom metadata fields

        Output structure:
            id, memory, hash, created_at, updated_at, user_id, agent_id,
            metadata: {memory_type, categories, importance, pinned, ...}
        """
        payload = point.payload or {}
        memory: dict[str, Any] = {
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
        }
        metadata = {k: v for k, v in payload.items() if k not in skip_keys}
        if metadata:
            memory["metadata"] = metadata
        return memory
