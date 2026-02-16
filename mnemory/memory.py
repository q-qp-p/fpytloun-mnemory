"""Business logic layer for mnemory.

Orchestrates the vector store (fast memory) and artifact store (slow memory)
to provide a unified memory interface. Handles validation, reranking,
core memory assembly, and artifact lifecycle.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from mnemory.cache import TTLCache
from mnemory.categories import (
    IMPORTANCE_WEIGHTS,
    PREDEFINED_CATEGORIES,
    count_categories,
    matches_category_filter,
    validate_categories,
    validate_importance,
    validate_memory_type,
)
from mnemory.classify import classify_memory
from mnemory.config import Config
from mnemory.storage.artifact import ArtifactStore
from mnemory.storage.vector import VectorStore

logger = logging.getLogger(__name__)

# Max length for user_id and agent_id to prevent abuse
_MAX_ID_LENGTH = 256


def _validate_id(value: str, name: str) -> str:
    """Validate a user_id or agent_id string."""
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty")
    value = value.strip()
    if len(value) > _MAX_ID_LENGTH:
        raise ValueError(f"{name} too long (max {_MAX_ID_LENGTH} chars)")
    return value


class MemoryService:
    """High-level memory service combining fast and slow memory tiers.

    Fast memory: concise facts/summaries in a vector store (searchable).
    Slow memory: detailed artifacts in S3/filesystem (retrieved on demand).
    """

    def __init__(self, config: Config):
        self._config = config
        self.vector = VectorStore(config)
        self.artifact = ArtifactStore(
            config.artifact,
            max_artifact_size=config.memory.max_artifact_size,
        )
        self._category_cache: TTLCache[str, list[str]] = TTLCache(
            ttl_seconds=config.memory.classify_cache_ttl,
        )
        self._core_cache: TTLCache[tuple, str] = TTLCache(
            ttl_seconds=config.memory.core_memories_cache_ttl,
        )

    # ── Add Memory ────────────────────────────────────────────────────

    def add_memory(
        self,
        content: str,
        *,
        user_id: str,
        agent_id: str | None = None,
        memory_type: str | None = None,
        categories: list[str] | None = None,
        importance: str | None = None,
        pinned: bool | None = None,
    ) -> dict:
        """Store a fast memory with metadata.

        Content is processed by mem0's LLM for fact extraction and
        deduplication. Max length is enforced.

        When memory_type, categories, importance, or pinned are not provided
        (None), they are auto-classified by an LLM call if AUTO_CLASSIFY is
        enabled, or fall back to sensible defaults.

        Returns the mem0 add result with memory IDs.
        """
        # Validate inputs
        user_id = _validate_id(user_id, "user_id")
        if agent_id:
            agent_id = _validate_id(agent_id, "agent_id")

        max_len = self._config.memory.max_memory_length
        if len(content) > max_len:
            return {
                "error": True,
                "message": (
                    f"Content too long: {len(content)} chars (max {max_len}). "
                    "Store only key conclusions and findings. Use save_artifact "
                    "for detailed content."
                ),
            }

        # Determine which fields need auto-classification
        missing: set[str] = set()
        if memory_type is None:
            missing.add("memory_type")
        if categories is None:
            missing.add("categories")
        if importance is None:
            missing.add("importance")
        if pinned is None:
            missing.add("pinned")

        if missing and self._config.memory.auto_classify:
            available_cats = self._get_available_categories(user_id)
            classified = classify_memory(
                content,
                missing_fields=missing,
                llm_config=self._config.llm,
                available_categories=available_cats,
            )
            if memory_type is None:
                memory_type = classified.get("memory_type", "fact")
            if categories is None:
                categories = classified.get("categories", [])
            if importance is None:
                importance = classified.get("importance", "normal")
            if pinned is None:
                pinned = classified.get("pinned", False)
        else:
            # Defaults when auto_classify is off or all fields provided
            if memory_type is None:
                memory_type = "fact"
            if categories is None:
                categories = []
            if importance is None:
                importance = "normal"
            if pinned is None:
                pinned = False

        # memory_type, importance, pinned are guaranteed non-None at this point
        # (set by caller, classifier, or defaults above).
        memory_type = validate_memory_type(memory_type)  # type: ignore[arg-type]
        importance = validate_importance(importance)  # type: ignore[arg-type]
        if categories:
            categories = validate_categories(categories)

        metadata = {
            "memory_type": memory_type,
            "categories": categories or [],
            "importance": importance,
            "pinned": pinned,
            "artifacts": [],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        result = self.vector.add(
            content, user_id=user_id, agent_id=agent_id, metadata=metadata
        )

        # Invalidate caches — new memory may affect core memories or categories
        self._core_cache.invalidate_prefix(user_id)
        if categories:
            self._category_cache.invalidate(user_id)

        return result

    # ── Search Memories ───────────────────────────────────────────────

    def search_memories(
        self,
        query: str,
        *,
        user_id: str,
        agent_id: str | None = None,
        memory_type: str | None = None,
        categories: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search memories with semantic similarity, filtered and reranked.

        Fetches 3x limit from mem0, applies category/type post-filtering,
        reranks by combined similarity + importance score, returns top N.
        """
        user_id = _validate_id(user_id, "user_id")
        if agent_id:
            agent_id = _validate_id(agent_id, "agent_id")

        # Build mem0-compatible filters
        filters = {}
        if memory_type:
            memory_type = validate_memory_type(memory_type)
            filters["memory_type"] = memory_type

        # Fetch extra for post-filtering and reranking
        fetch_limit = limit * 3
        result = self.vector.search(
            query,
            user_id=user_id,
            agent_id=agent_id,
            filters=filters if filters else None,
            limit=fetch_limit,
        )

        memories = result.get("results", [])

        # Post-filter by categories (mem0 can't do prefix matching)
        if categories:
            memories = [
                m
                for m in memories
                if matches_category_filter(
                    m.get("metadata", {}).get("categories", []), categories
                )
            ]

        # Rerank by combined score: similarity * 0.7 + importance * 0.3
        memories = self._rerank_by_importance(memories)

        return memories[:limit]

    def search_memories_dual_scope(
        self,
        query: str,
        *,
        user_id: str,
        session_agent_id: str,
        memory_type: str | None = None,
        categories: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search both agent-scoped and shared memories, merge and deduplicate.

        Used when a session has an agent_id but the LLM doesn't pass one
        (meaning: search everything I can see). Performs two searches:
        1. Agent-scoped memories (agent_id=session_agent_id)
        2. Shared memories (agent_id=None)
        Then merges, deduplicates by memory ID, and reranks.
        """
        user_id = _validate_id(user_id, "user_id")
        session_agent_id = _validate_id(session_agent_id, "agent_id")

        filters = {}
        if memory_type:
            memory_type = validate_memory_type(memory_type)
            filters["memory_type"] = memory_type

        fetch_limit = limit * 3

        # Search 1: agent-scoped memories
        agent_result = self.vector.search(
            query,
            user_id=user_id,
            agent_id=session_agent_id,
            filters=filters if filters else None,
            limit=fetch_limit,
        )
        # Search 2: shared memories (no agent_id)
        shared_result = self.vector.search(
            query,
            user_id=user_id,
            agent_id=None,
            filters=filters if filters else None,
            limit=fetch_limit,
        )

        # Merge and deduplicate by memory ID
        seen_ids: set[str] = set()
        memories: list[dict] = []
        for mem in agent_result.get("results", []) + shared_result.get("results", []):
            mid = mem.get("id")
            if mid and mid not in seen_ids:
                seen_ids.add(mid)
                memories.append(mem)

        # Post-filter by categories
        if categories:
            memories = [
                m
                for m in memories
                if matches_category_filter(
                    m.get("metadata", {}).get("categories", []), categories
                )
            ]

        memories = self._rerank_by_importance(memories)
        return memories[:limit]

    # ── Get Core Memories ─────────────────────────────────────────────

    def get_core_memories(
        self,
        *,
        user_id: str,
        agent_id: str | None = None,
        recent_hours: int | None = None,
    ) -> str:
        """Assemble core context for conversation start.

        Returns a structured text with:
        1. Pinned agent memories (identity, knowledge) — if agent_id provided
        2. Pinned user memories (facts, preferences) — shared across agents
        3. Recent context memories from the last N hours

        Total output is capped at MAX_CORE_CONTEXT_LENGTH.
        """
        user_id = _validate_id(user_id, "user_id")
        if agent_id:
            agent_id = _validate_id(agent_id, "agent_id")

        if recent_hours is None:
            recent_hours = self._config.memory.default_recent_hours
        recent_hours = int(recent_hours)

        # Check cache
        cache_key = (user_id, agent_id or "", recent_hours)
        cached = self._core_cache.get(cache_key)
        if cached is not None:
            return cached

        max_len = self._config.memory.max_core_context_length

        sections: list[str] = []

        # 1. Pinned agent memories (identity + knowledge)
        if agent_id:
            agent_pinned = self.vector.get_pinned_memories(
                user_id=user_id, agent_id=agent_id
            )
            # Only include memories that have this agent_id
            agent_only = [m for m in agent_pinned if m.get("agent_id") == agent_id]
            if agent_only:
                identity = []
                knowledge = []
                for m in agent_only:
                    mt = m.get("metadata", {}).get("memory_type", "fact")
                    text = m.get("memory", "")
                    if mt in ("preference", "fact"):
                        identity.append(f"- {text}")
                    else:
                        knowledge.append(f"- {text}")
                if identity:
                    sections.append("## Agent Identity\n" + "\n".join(identity))
                if knowledge:
                    sections.append("## Agent Knowledge\n" + "\n".join(knowledge))

        # 2. Pinned user memories (no agent_id — shared across agents)
        user_pinned = self.vector.get_pinned_memories(
            user_id=user_id, exclude_agent=True
        )
        if user_pinned:
            facts = []
            prefs = []
            other = []
            for m in user_pinned:
                mt = m.get("metadata", {}).get("memory_type", "fact")
                text = m.get("memory", "")
                if mt == "fact":
                    facts.append(f"- {text}")
                elif mt == "preference":
                    prefs.append(f"- {text}")
                else:
                    other.append(f"- {text}")
            if facts:
                sections.append("## User Facts\n" + "\n".join(facts))
            if prefs:
                sections.append("## User Preferences\n" + "\n".join(prefs))
            if other:
                sections.append("## User Context\n" + "\n".join(other))

        # 3. Recent context memories
        since = datetime.now(timezone.utc) - timedelta(hours=recent_hours)
        recent = self.vector.get_recent_memories(
            user_id=user_id, agent_id=agent_id, since=since, limit=50
        )
        if recent:
            # Sort chronologically
            recent.sort(key=lambda m: m.get("created_at", ""))
            lines = []
            for m in recent:
                ts = m.get("created_at", "")
                # Extract just the date+time portion for readability
                if ts and "T" in ts:
                    ts = ts.split("T")[0] + " " + ts.split("T")[1][:5]
                text = m.get("memory", "")
                lines.append(f"- [{ts}] {text}")
            sections.append(
                f"## Recent Context (last {recent_hours}h)\n" + "\n".join(lines)
            )

        if not sections:
            output = "No core memories found."
            self._core_cache.set(cache_key, output)
            return output

        output = "\n\n".join(sections)

        # Truncate if too long (trim recent context first)
        if len(output) > max_len:
            # Try removing recent context
            if len(sections) > 1 and sections[-1].startswith("## Recent"):
                sections = sections[:-1]
                output = "\n\n".join(sections)
            # If still too long, hard truncate
            if len(output) > max_len:
                output = output[: max_len - 20] + "\n\n[...truncated]"

        self._core_cache.set(cache_key, output)
        return output

    # ── List Memories ─────────────────────────────────────────────────

    def list_memories(
        self,
        *,
        user_id: str,
        agent_id: str | None = None,
        memory_type: str | None = None,
        categories: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """List memories with optional filtering."""
        user_id = _validate_id(user_id, "user_id")
        if agent_id:
            agent_id = _validate_id(agent_id, "agent_id")

        filters = {}
        if memory_type:
            memory_type = validate_memory_type(memory_type)
            filters["memory_type"] = memory_type

        result = self.vector.get_all(
            user_id=user_id,
            agent_id=agent_id,
            filters=filters if filters else None,
            limit=limit * 2,  # Fetch extra for post-filtering
        )

        memories = result.get("results", [])

        if categories:
            memories = [
                m
                for m in memories
                if matches_category_filter(
                    m.get("metadata", {}).get("categories", []), categories
                )
            ]

        return memories[:limit]

    def list_memories_dual_scope(
        self,
        *,
        user_id: str,
        session_agent_id: str,
        memory_type: str | None = None,
        categories: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """List both agent-scoped and shared memories, merge and deduplicate.

        Used when a session has an agent_id but the LLM doesn't pass one
        (meaning: list everything I can see).
        """
        user_id = _validate_id(user_id, "user_id")
        session_agent_id = _validate_id(session_agent_id, "agent_id")

        filters = {}
        if memory_type:
            memory_type = validate_memory_type(memory_type)
            filters["memory_type"] = memory_type

        fetch_limit = limit * 2

        # Fetch 1: agent-scoped memories
        agent_result = self.vector.get_all(
            user_id=user_id,
            agent_id=session_agent_id,
            filters=filters if filters else None,
            limit=fetch_limit,
        )
        # Fetch 2: shared memories (no agent_id)
        shared_result = self.vector.get_all(
            user_id=user_id,
            agent_id=None,
            filters=filters if filters else None,
            limit=fetch_limit,
        )

        # Merge and deduplicate by memory ID
        seen_ids: set[str] = set()
        memories: list[dict] = []
        for mem in agent_result.get("results", []) + shared_result.get("results", []):
            mid = mem.get("id")
            if mid and mid not in seen_ids:
                seen_ids.add(mid)
                memories.append(mem)

        # Post-filter by categories
        if categories:
            memories = [
                m
                for m in memories
                if matches_category_filter(
                    m.get("metadata", {}).get("categories", []), categories
                )
            ]

        return memories[:limit]

    # ── Ownership Verification ─────────────────────────────────────────

    def verify_memory_access(
        self,
        memory_id: str,
        *,
        session_agent_id: str | None,
    ) -> None:
        """Verify the session agent can access a memory.

        When session_agent_id is set, the memory must either:
        - Have no agent_id (shared memory) — accessible by all agents
        - Have agent_id == session_agent_id — own agent memory

        Memories belonging to a different agent are blocked.

        Raises ValueError if access is denied.
        Does nothing if session_agent_id is None (no protection).
        """
        if session_agent_id is None:
            return  # No session agent — no protection

        mem = self.vector.get_by_id(memory_id)
        if mem is None:
            return  # Memory not found — let downstream handle 404

        mem_agent_id = mem.get("agent_id")
        if mem_agent_id and mem_agent_id != session_agent_id:
            raise ValueError(
                f"Cannot access memory '{memory_id}' — "
                f"it belongs to agent '{mem_agent_id}'"
            )

    # ── Update Memory ─────────────────────────────────────────────────

    def update_memory(
        self,
        memory_id: str,
        *,
        user_id: str | None = None,
        content: str | None = None,
        memory_type: str | None = None,
        categories: list[str] | None = None,
        importance: str | None = None,
        pinned: bool | None = None,
    ) -> dict:
        """Update a memory's content and/or metadata.

        Content updates go through mem0 (re-embedding). Metadata updates
        use direct vector store payload updates to avoid losing custom fields.
        """
        metadata_updates: dict[str, Any] = {}

        if memory_type is not None:
            metadata_updates["memory_type"] = validate_memory_type(memory_type)
        if categories is not None:
            metadata_updates["categories"] = validate_categories(categories)
        if importance is not None:
            metadata_updates["importance"] = validate_importance(importance)
        if pinned is not None:
            metadata_updates["pinned"] = pinned

        # Update content via mem0 (re-embeds)
        if content is not None:
            max_len = self._config.memory.max_memory_length
            if len(content) > max_len:
                return {
                    "error": True,
                    "message": (
                        f"Content too long: {len(content)} chars (max {max_len})."
                    ),
                }
            self.vector.update(memory_id, content)

        # Update metadata directly on the vector store
        if metadata_updates:
            metadata_updates["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            self.vector.update_metadata(memory_id, metadata_updates)

        # Invalidate core cache — updated memory may be pinned or recent.
        if user_id:
            self._core_cache.invalidate_prefix(user_id)
        else:
            self._core_cache.clear()

        return {"status": "updated", "memory_id": memory_id}

    # ── Delete Memory ─────────────────────────────────────────────────

    def delete_memory(self, memory_id: str, *, user_id: str) -> dict:
        """Delete a memory and all its artifacts."""
        user_id = _validate_id(user_id, "user_id")

        # Delete artifacts first
        try:
            self.artifact.delete_all_for_memory(user_id=user_id, memory_id=memory_id)
        except Exception as e:
            logger.warning("Failed to delete artifacts for %s: %s", memory_id, e)

        self.vector.delete(memory_id)

        # Invalidate core cache for this user
        self._core_cache.invalidate_prefix(user_id)

        return {"status": "deleted", "memory_id": memory_id}

    def delete_all_memories(self, *, user_id: str, agent_id: str | None = None) -> dict:
        """Delete all memories for a user (and optionally agent scope)."""
        user_id = _validate_id(user_id, "user_id")
        if agent_id:
            agent_id = _validate_id(agent_id, "agent_id")

        # Delete artifacts for each memory before removing vector entries
        try:
            result = self.vector.get_all(user_id=user_id, agent_id=agent_id, limit=1000)
            for mem in result.get("results", []):
                mem_id = mem.get("id")
                if mem_id and mem.get("metadata", {}).get("artifacts"):
                    try:
                        self.artifact.delete_all_for_memory(
                            user_id=user_id, memory_id=mem_id
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to delete artifacts for %s: %s", mem_id, e
                        )
        except Exception as e:
            logger.warning("Failed to enumerate memories for artifact cleanup: %s", e)

        self.vector.delete_all(user_id=user_id, agent_id=agent_id)

        # Invalidate all caches for this user
        self._core_cache.invalidate_prefix(user_id)
        self._category_cache.invalidate(user_id)

        return {"status": "deleted_all", "user_id": user_id, "agent_id": agent_id}

    # ── List Categories ───────────────────────────────────────────────

    def list_categories(self, *, user_id: str) -> dict:
        """List all categories with memory counts and descriptions.

        Returns predefined categories (even if empty) plus any dynamic
        project: categories found in the user's memories.
        """
        user_id = _validate_id(user_id, "user_id")
        result = self.vector.get_all(user_id=user_id, limit=1000)
        memories = result.get("results", [])
        counts = count_categories(memories)

        categories = []
        # Add predefined categories
        for name, description in PREDEFINED_CATEGORIES.items():
            # Count includes both exact matches and subcategories
            count = counts.get(name, 0)
            # Also count subcategories (e.g., project:foo for "project")
            for cat_name, cat_count in counts.items():
                if cat_name.startswith(name + ":"):
                    count += cat_count
            categories.append(
                {"name": name, "description": description, "count": count}
            )

        # Add dynamic subcategories not in predefined set
        for cat_name, cat_count in sorted(counts.items()):
            if ":" in cat_name:
                prefix = cat_name.split(":", 1)[0]
                if prefix in PREDEFINED_CATEGORIES:
                    categories.append(
                        {
                            "name": cat_name,
                            "description": (
                                f"{PREDEFINED_CATEGORIES[prefix]} (subcategory)"
                            ),
                            "count": cat_count,
                        }
                    )

        return {"categories": categories, "total_memories": len(memories)}

    # ── Artifact Operations ───────────────────────────────────────────

    def save_artifact(
        self,
        memory_id: str,
        *,
        user_id: str,
        content: str,
        filename: str = "note.md",
        content_type: str = "text/markdown",
    ) -> dict:
        """Save an artifact attached to a memory.

        Stores the content in the artifact backend and updates the
        memory's metadata with the artifact reference.
        """
        user_id = _validate_id(user_id, "user_id")

        meta = self.artifact.save(
            user_id=user_id,
            memory_id=memory_id,
            content=content,
            filename=filename,
            content_type=content_type,
        )

        # Update the memory's artifacts list in vector store metadata
        try:
            mem = self.vector.get_by_id(memory_id)
            current_artifacts = (
                mem.get("metadata", {}).get("artifacts", []) if mem else []
            )

            current_artifacts.append(meta.to_dict())
            self.vector.update_metadata(memory_id, {"artifacts": current_artifacts})
        except Exception as e:
            logger.error("Failed to update artifact metadata on memory: %s", e)

        return {
            "status": "saved",
            "artifact": meta.to_dict(),
            "memory_id": memory_id,
        }

    def get_artifact(
        self,
        memory_id: str,
        artifact_id: str,
        *,
        user_id: str,
        offset: int = 0,
        limit: int = 5000,
    ) -> dict:
        """Retrieve artifact content with pagination."""
        user_id = _validate_id(user_id, "user_id")
        # Get artifact metadata from the memory
        artifacts_meta = self._get_artifacts_meta(user_id, memory_id)
        return self.artifact.load(
            user_id=user_id,
            memory_id=memory_id,
            artifact_id=artifact_id,
            artifacts_meta=artifacts_meta,
            offset=offset,
            limit=limit,
        )

    def list_artifacts(self, memory_id: str, *, user_id: str) -> list[dict]:
        """List all artifacts attached to a memory."""
        return self._get_artifacts_meta(user_id, memory_id)

    def delete_artifact(
        self,
        memory_id: str,
        artifact_id: str,
        *,
        user_id: str,
    ) -> dict:
        """Delete an artifact and update the memory's metadata."""
        user_id = _validate_id(user_id, "user_id")
        artifacts_meta = self._get_artifacts_meta(user_id, memory_id)

        self.artifact.delete(
            user_id=user_id,
            memory_id=memory_id,
            artifact_id=artifact_id,
            artifacts_meta=artifacts_meta,
        )

        # Update memory metadata to remove the artifact reference
        updated = [a for a in artifacts_meta if a.get("id") != artifact_id]
        self.vector.update_metadata(memory_id, {"artifacts": updated})

        return {"status": "deleted", "artifact_id": artifact_id, "memory_id": memory_id}

    # ── Private Helpers ───────────────────────────────────────────────

    def _get_available_categories(self, user_id: str) -> list[str]:
        """Get the list of available categories for auto-classification.

        Returns predefined categories plus any dynamic project:* subcategories
        found in the user's existing memories. Results are cached with TTL
        to avoid querying the vector store on every add_memory call.
        """
        cached = self._category_cache.get(user_id)
        if cached is not None:
            return cached

        # Start with predefined categories
        categories = list(PREDEFINED_CATEGORIES.keys())

        # Add dynamic subcategories from existing memories
        try:
            result = self.vector.get_all(user_id=user_id, limit=1000)
            counts = count_categories(result.get("results", []))
            for cat_name in sorted(counts.keys()):
                if ":" in cat_name and cat_name not in categories:
                    categories.append(cat_name)
        except Exception:
            logger.warning("Failed to fetch existing categories for classification")

        self._category_cache.set(user_id, categories)
        return categories

    def _get_artifacts_meta(self, user_id: str, memory_id: str) -> list[dict]:
        """Get artifact metadata list from a memory's vector store entry."""
        mem = self.vector.get_by_id(memory_id)
        if mem:
            return mem.get("metadata", {}).get("artifacts", [])
        return []

    def _rerank_by_importance(self, memories: list[dict]) -> list[dict]:
        """Rerank search results by combined similarity + importance score.

        Combined score = similarity * 0.7 + importance_weight * 0.3
        """
        for mem in memories:
            sim_score = mem.get("score", 0.0)
            importance = mem.get("metadata", {}).get("importance", "normal")
            imp_weight = IMPORTANCE_WEIGHTS.get(importance, 0.4)
            mem["_combined_score"] = sim_score * 0.7 + imp_weight * 0.3

        memories.sort(key=lambda m: m.get("_combined_score", 0), reverse=True)

        # Clean up internal field
        for mem in memories:
            mem.pop("_combined_score", None)

        return memories
