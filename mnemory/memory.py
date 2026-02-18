"""Business logic layer for mnemory.

Orchestrates the vector store (fast memory) and artifact store (slow memory)
to provide a unified memory interface. Handles validation, reranking,
core memory assembly, and artifact lifecycle.

The add pipeline uses a single LLM call for fact extraction, per-fact
classification, and deduplication against existing memories.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from mnemory.cache import TTLCache
from mnemory.categories import (
    PREDEFINED_CATEGORIES,
    count_categories,
    matches_category_filter,
    validate_categories,
    validate_importance,
    validate_memory_type,
)
from mnemory.config import Config
from mnemory.llm import LLMClient, parse_json_response
from mnemory.prompts import (
    build_classification_prompt,
    build_extraction_prompt,
    build_query_generation_prompt,
    build_rerank_prompt,
    build_shorten_prompt,
    parse_extraction_response,
)
from mnemory.storage.artifact import ArtifactStore
from mnemory.storage.vector import VectorStore
from mnemory.ttl import (
    build_expiry_metadata,
    build_reinforcement_metadata,
    is_expired,
    should_exclude,
)

logger = logging.getLogger(__name__)

# Max length for user_id and agent_id to prevent abuse
_MAX_ID_LENGTH = 256

# Minimal English stopwords for keyword boost tokenization.
# Kept small to avoid filtering meaningful short words.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "and",
        "or",
        "but",
        "not",
        "with",
        "by",
        "from",
        "as",
        "if",
        "it",
        "its",
        "this",
        "that",
        "my",
        "your",
        "his",
        "her",
        "our",
        "their",
        "me",
        "him",
        "us",
        "them",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "will",
        "would",
        "can",
        "could",
        "should",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "when",
        "where",
        "why",
        "about",
        "so",
        "no",
        "yes",
        "all",
        "any",
        "some",
    }
)

# Regex for splitting text into tokens (non-word characters)
_TOKEN_SPLIT = re.compile(r"\W+")


def _tokenize(text: str) -> set[str]:
    """Tokenize text into a set of lowercase words for keyword matching.

    Splits on non-word characters, lowercases, removes stopwords and
    tokens shorter than 2 characters.
    """
    tokens = _TOKEN_SPLIT.split(text.lower())
    return {t for t in tokens if t and len(t) >= 2 and t not in _STOPWORDS}


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
        self._llm = LLMClient(config.llm)
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
        infer: bool = True,
        role: str = "user",
        ttl_days: int | None = None,
    ) -> dict:
        """Store a fast memory with metadata.

        When infer=True (default): Uses a single LLM call for fact extraction,
        per-fact classification, and deduplication against existing memories.
        When infer=False: Content is stored as-is with only an embedding call
        (much faster, no LLM, no dedup).

        When memory_type, categories, importance, or pinned are not provided
        (None), they are auto-classified:
        - infer=True: classification is part of the unified extraction prompt
        - infer=False: a separate classification LLM call if AUTO_CLASSIFY is on

        Args:
            role: Message role controlling fact extraction. "user" (default)
                  extracts facts about the user. "assistant" extracts facts
                  about the agent (identity, personality, capabilities).
                  Requires agent_id when set to "assistant".

        Returns dict with "results" key containing list of memory actions.
        """
        # Validate inputs
        if role not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        user_id = _validate_id(user_id, "user_id")
        if agent_id:
            agent_id = _validate_id(agent_id, "agent_id")
        if role == "assistant" and not agent_id:
            raise ValueError("agent_id is required when role='assistant'")

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

        if infer:
            result = self._add_with_inference(
                content,
                user_id=user_id,
                agent_id=agent_id,
                memory_type=memory_type,
                categories=categories,
                importance=importance,
                pinned=pinned,
                role=role,
                ttl_days=ttl_days,
            )
        else:
            result = self._add_direct(
                content,
                user_id=user_id,
                agent_id=agent_id,
                memory_type=memory_type,
                categories=categories,
                importance=importance,
                pinned=pinned,
                role=role,
                ttl_days=ttl_days,
            )

        # Invalidate caches — new memory may affect core memories or categories
        self._core_cache.invalidate_prefix(user_id)
        self._category_cache.invalidate(user_id)

        return result

    def _add_with_inference(
        self,
        content: str,
        *,
        user_id: str,
        agent_id: str | None,
        memory_type: str | None,
        categories: list[str] | None,
        importance: str | None,
        pinned: bool | None,
        role: str,
        ttl_days: int | None,
    ) -> dict:
        """Add memory with LLM-driven extraction, classification, and dedup.

        Single LLM call pipeline:
        1. Embed the raw content
        2. Search for similar existing memories
        3. Build unified prompt (extraction + classification + dedup)
        4. Single LLM call
        5. Execute actions (ADD/UPDATE/DELETE)
        """
        # 1. Embed the raw content for similarity search
        content_vector = self.vector.embedding.embed(content)

        # 2. Search for similar existing memories (excludes expired/decayed)
        existing_raw = self.vector.search_similar(
            content_vector,
            user_id=user_id,
            agent_id=agent_id,
            limit=10,
        )

        # Filter out low-similarity results to avoid false dedup matches
        dedup_threshold = self._config.memory.dedup_similarity_threshold
        existing = [m for m in existing_raw if m.get("score", 0) >= dedup_threshold]

        # 3. Build the unified prompt
        available_cats = self._get_available_categories(user_id)

        # Build explicit fields dict for fields the caller provided
        explicit_fields: dict[str, Any] = {}
        if memory_type is not None:
            explicit_fields["memory_type"] = memory_type
        if categories is not None:
            explicit_fields["categories"] = categories
        if importance is not None:
            explicit_fields["importance"] = importance
        if pinned is not None:
            explicit_fields["pinned"] = pinned

        max_len = self._config.memory.max_memory_length
        messages, json_schema, id_mapping = build_extraction_prompt(
            content,
            role=role,
            existing_memories=existing,
            available_categories=available_cats,
            explicit_fields=explicit_fields if explicit_fields else None,
            max_memory_length=max_len,
        )

        # 4. Single LLM call
        try:
            response_text = self._llm.generate(
                messages,
                json_schema=json_schema,
                max_tokens=2000,
            )
        except Exception:
            logger.exception("LLM extraction call failed")
            return {"error": True, "message": "Memory extraction failed"}

        # 5. Parse response and execute actions
        actions = parse_extraction_response(response_text, id_mapping)

        if not actions:
            return {"results": []}

        # 5b. Validate extracted fact lengths; retry oversized ones
        actions = self._validate_fact_lengths(actions, max_len)
        if isinstance(actions, dict) and actions.get("error"):
            return actions

        # Batch embed all new/updated texts
        texts_to_embed = [
            a["text"] for a in actions if a["action"] in ("ADD", "UPDATE")
        ]
        if texts_to_embed:
            vectors = self.vector.embedding.embed_batch(texts_to_embed)
            vector_map = dict(zip(texts_to_embed, vectors))
        else:
            vector_map = {}

        results = []
        for action in actions:
            try:
                result_entry = self._execute_action(
                    action,
                    user_id=user_id,
                    agent_id=agent_id,
                    role=role,
                    ttl_days=ttl_days,
                    explicit_fields=explicit_fields,
                    vector_map=vector_map,
                )
                if result_entry:
                    results.append(result_entry)
            except Exception:
                logger.exception(
                    "Failed to execute %s action for memory", action["action"]
                )

        return {"results": results}

    def _validate_fact_lengths(
        self,
        actions: list[dict[str, Any]],
        max_len: int,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Validate extracted fact lengths, retrying oversized ones via LLM.

        For each action whose text exceeds max_len, makes a focused LLM call
        to shorten or split the fact. If the retry still produces oversized
        output, returns an error dict.

        Returns:
            The (possibly modified) actions list, or an error dict on failure.
        """
        oversized = [
            (i, a)
            for i, a in enumerate(actions)
            if a["action"] in ("ADD", "UPDATE") and len(a["text"]) > max_len
        ]

        if not oversized:
            return actions

        for idx, action in oversized:
            logger.info(
                "Extracted fact too long (%d chars, max %d), retrying: %.80s...",
                len(action["text"]),
                max_len,
                action["text"],
            )
            try:
                messages, schema = build_shorten_prompt(
                    action, max_memory_length=max_len
                )
                response_text = self._llm.generate(
                    messages, json_schema=schema, max_tokens=2000
                )
                shortened = parse_extraction_response(response_text, {})
            except Exception:
                logger.exception("LLM shorten/split call failed")
                return {
                    "error": True,
                    "message": (
                        f"Extracted fact exceeds max length "
                        f"({len(action['text'])} chars, max {max_len}) "
                        f"and retry failed. Please reformulate the memory "
                        f"content to be more concise."
                    ),
                }

            if not shortened:
                return {
                    "error": True,
                    "message": (
                        f"Extracted fact exceeds max length "
                        f"({len(action['text'])} chars, max {max_len}) "
                        f"and retry produced no results. Please reformulate "
                        f"the memory content to be more concise."
                    ),
                }

            # Check retry results are within limits
            still_oversized = [
                s
                for s in shortened
                if s["action"] in ("ADD", "UPDATE") and len(s["text"]) > max_len
            ]
            if still_oversized:
                return {
                    "error": True,
                    "message": (
                        f"Extracted fact exceeds max length "
                        f"({len(still_oversized[0]['text'])} chars, max "
                        f"{max_len}) even after retry. Please reformulate "
                        f"the memory content to be more concise."
                    ),
                }

            # Replace the oversized action with the shortened/split results.
            # For retry results: preserve the original action type and
            # target_id for UPDATE/DELETE, force ADD for split results.
            for s in shortened:
                if s["action"] == "ADD" and action["action"] == "UPDATE":
                    # First split result inherits the UPDATE target
                    s["action"] = action["action"]
                    s["target_id"] = action["target_id"]
                    s["old_memory"] = action.get("old_memory")
                    # Only the first one gets UPDATE, rest stay ADD
                    break

            # Replace original action with shortened results
            actions[idx : idx + 1] = shortened

        return actions

    def _validate_metadata(
        self,
        text: str,
        mem_type: str,
        cats: list[str],
        imp: str,
        user_id: str,
    ) -> tuple[str, list[str], str]:
        """Validate metadata fields with LLM retry on failure.

        Tries validation first. On ValueError (e.g., LLM hallucinated an
        invalid category like 'professional'), re-classifies the failing
        fields via a focused LLM call that includes the error message.
        Falls back to safe defaults if retry also fails.

        Returns:
            Tuple of (mem_type, cats, imp) — always valid values.
        """
        try:
            mem_type = validate_memory_type(mem_type)
            if cats:
                cats = validate_categories(cats)
            imp = validate_importance(imp)
            return mem_type, cats, imp
        except ValueError as first_error:
            validation_error = first_error
            logger.warning(
                "Metadata validation failed: %s — retrying via LLM", first_error
            )

        # Retry: re-classify all metadata fields with the error context
        try:
            available_cats = self._get_available_categories(user_id)
            msgs, schema = build_classification_prompt(
                f"{text}\n\nPrevious classification error: {validation_error}",
                missing_fields={"memory_type", "categories", "importance"},
                available_categories=available_cats,
            )
            if msgs:
                response_text = self._llm.generate(msgs, json_schema=schema)
                classified = parse_json_response(response_text)
                mem_type = validate_memory_type(classified.get("memory_type", "fact"))
                cats = validate_categories(classified.get("categories", []))
                imp = validate_importance(classified.get("importance", "normal"))
                logger.info(
                    "Metadata re-classification succeeded: type=%s, "
                    "categories=%s, importance=%s",
                    mem_type,
                    cats,
                    imp,
                )
                return mem_type, cats, imp
        except Exception:
            logger.warning(
                "Metadata re-classification also failed, using safe defaults"
            )

        # Final fallback — safe defaults, never lose the memory
        return "fact", [], "normal"

    def _execute_action(
        self,
        action: dict[str, Any],
        *,
        user_id: str,
        agent_id: str | None,
        role: str,
        ttl_days: int | None,
        explicit_fields: dict[str, Any],
        vector_map: dict[str, list[float]],
    ) -> dict[str, Any] | None:
        """Execute a single memory action (ADD, UPDATE, or DELETE)."""
        act = action["action"]
        text = action["text"]

        # Apply explicit overrides — caller-provided fields take precedence
        mem_type = explicit_fields.get("memory_type", action["memory_type"])
        cats = explicit_fields.get("categories", action["categories"])
        imp = explicit_fields.get("importance", action["importance"])
        pin = explicit_fields.get("pinned", action["pinned"])

        # Validate (with LLM retry on failure)
        mem_type, cats, imp = self._validate_metadata(
            text, mem_type, cats, imp, user_id
        )

        if act == "ADD":
            vector = vector_map.get(text)
            if vector is None:
                vector = self.vector.embedding.embed(text)

            metadata = {
                "memory_type": mem_type,
                "categories": cats or [],
                "importance": imp,
                "pinned": pin,
                "artifacts": [],
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            ttl_meta = build_expiry_metadata(ttl_days, mem_type, self._config.memory)
            metadata.update(ttl_meta)

            memory_id = self.vector.insert(
                text=text,
                vector=vector,
                user_id=user_id,
                agent_id=agent_id,
                metadata=metadata,
                role=role,
            )
            return {"id": memory_id, "memory": text, "event": "ADD"}

        elif act == "UPDATE":
            target_id = action["target_id"]
            vector = vector_map.get(text)
            if vector is None:
                vector = self.vector.embedding.embed(text)

            # Fetch existing memory to preserve metadata we don't want to lose
            existing = self.vector.get_by_id(target_id)
            if existing is None:
                logger.warning(
                    "UPDATE target %s not found, converting to ADD", target_id
                )
                # Fall through to ADD instead
                action_copy = dict(action)
                action_copy["action"] = "ADD"
                action_copy["target_id"] = None
                return self._execute_action(
                    action_copy,
                    user_id=user_id,
                    agent_id=agent_id,
                    role=role,
                    ttl_days=ttl_days,
                    explicit_fields=explicit_fields,
                    vector_map=vector_map,
                )

            # Build updated metadata — preserve existing, override with new
            existing_meta = existing.get("metadata") or {}
            metadata_update = {
                "memory_type": mem_type,
                "categories": cats or [],
                "importance": imp,
                "pinned": pin,
            }
            # Recalculate TTL based on new memory_type
            ttl_meta = build_expiry_metadata(ttl_days, mem_type, self._config.memory)
            metadata_update.update(ttl_meta)

            # Preserve artifacts from existing memory
            metadata_update["artifacts"] = existing_meta.get("artifacts", [])
            metadata_update["updated_at_utc"] = datetime.now(timezone.utc).isoformat()

            # Update content + re-embed via full point replacement
            # (preserves all payload fields we set)
            self.vector.update_content(target_id, text, vector=vector)
            # Then update metadata
            self.vector.update_metadata(target_id, metadata_update)

            return {
                "id": target_id,
                "memory": text,
                "event": "UPDATE",
                "previous_memory": action.get("old_memory"),
            }

        elif act == "DELETE":
            target_id = action["target_id"]
            # Delete artifacts first
            try:
                self.artifact.delete_all_for_memory(
                    user_id=user_id, memory_id=target_id
                )
            except Exception:
                logger.warning("Failed to delete artifacts for %s", target_id)

            self.vector.delete(target_id)
            return {"id": target_id, "memory": text, "event": "DELETE"}

        return None

    def _add_direct(
        self,
        content: str,
        *,
        user_id: str,
        agent_id: str | None,
        memory_type: str | None,
        categories: list[str] | None,
        importance: str | None,
        pinned: bool | None,
        role: str,
        ttl_days: int | None,
    ) -> dict:
        """Add memory directly without LLM inference (infer=False path).

        Stores content as-is with only an embedding call. Uses a separate
        classification LLM call if AUTO_CLASSIFY is enabled and metadata
        fields are missing.
        """
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
            try:
                msgs, schema = build_classification_prompt(
                    content,
                    missing_fields=missing,
                    available_categories=available_cats,
                )
                if msgs:
                    response_text = self._llm.generate(msgs, json_schema=schema)
                    classified = parse_json_response(response_text)
                else:
                    classified = {}
            except Exception:
                logger.warning("Classification failed, using defaults")
                classified = {}

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

        # Validate (with LLM retry on failure)
        memory_type, categories, importance = self._validate_metadata(
            content,
            memory_type,  # type: ignore[arg-type]
            categories or [],
            importance,  # type: ignore[arg-type]
            user_id,
        )

        metadata = {
            "memory_type": memory_type,
            "categories": categories or [],
            "importance": importance,
            "pinned": pinned,
            "artifacts": [],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        # Add TTL metadata
        ttl_meta = build_expiry_metadata(ttl_days, memory_type, self._config.memory)
        metadata.update(ttl_meta)

        # Embed and store
        vector = self.vector.embedding.embed(content)
        memory_id = self.vector.insert(
            text=content,
            vector=vector,
            user_id=user_id,
            agent_id=agent_id,
            metadata=metadata,
            role=role,
        )

        return {"results": [{"id": memory_id, "memory": content, "event": "ADD"}]}

    # ── Search Helpers ────────────────────────────────────────────────

    def _keyword_boost(self, query: str, memories: list[dict]) -> list[dict]:
        """Apply post-retrieval keyword boost to search results.

        Blends a keyword overlap score into the Qdrant score:
        final = (1 - weight) * qdrant_score + weight * keyword_score

        keyword_score = fraction of query tokens found in the memory text.
        If the query has no meaningful tokens (all stopwords), results are
        returned unchanged.

        Args:
            query: The original search query.
            memories: Search results with "score" and "memory" fields.

        Returns:
            Memories with updated scores, re-sorted by new score.
        """
        weight = self._config.memory.search_keyword_weight
        if weight <= 0 or not memories:
            return memories

        query_tokens = _tokenize(query)
        if not query_tokens:
            # All stopwords or empty query — skip keyword boost
            return memories

        for mem in memories:
            mem_text = mem.get("memory", "")
            mem_tokens = _tokenize(mem_text)
            overlap = len(query_tokens & mem_tokens) / len(query_tokens)
            original_score = mem.get("score", 0)
            mem["score"] = round((1 - weight) * original_score + weight * overlap, 4)

        memories.sort(key=lambda m: m.get("score", 0), reverse=True)
        return memories

    # ── Search Memories ───────────────────────────────────────────────

    def search_memories(
        self,
        query: str,
        *,
        user_id: str,
        agent_id: str | None = None,
        memory_type: str | None = None,
        categories: list[str] | None = None,
        role: str | None = None,
        limit: int = 10,
        include_decayed: bool = False,
        query_vector: list[float] | None = None,
    ) -> list[dict]:
        """Search memories with semantic similarity, filtered and reranked.

        Uses Qdrant's query_points() with native TTL, category, and
        importance filtering via FormulaQuery reranking.

        Args:
            query_vector: Pre-computed embedding vector. If provided, skips
                the embedding API call. Used by find_memories for batch
                embedding optimization.
        """
        user_id = _validate_id(user_id, "user_id")
        if agent_id:
            agent_id = _validate_id(agent_id, "agent_id")

        # Build metadata filters (memory_type, role)
        filters = {}
        if memory_type:
            memory_type = validate_memory_type(memory_type)
            filters["memory_type"] = memory_type
        if role:
            if role not in ("user", "assistant"):
                raise ValueError("role filter must be 'user' or 'assistant'")
            filters["role"] = role

        # Expand category prefixes for native Qdrant filtering
        expanded_categories = None
        if categories:
            expanded_categories = self._expand_category_filter(categories, user_id)

        result = self.vector.search(
            query,
            user_id=user_id,
            agent_id=agent_id,
            filters=filters if filters else None,
            categories=expanded_categories,
            limit=limit,
            exclude_expired=True,
            include_decayed=include_decayed,
            similarity_weight=self._config.memory.search_similarity_weight,
            query_vector=query_vector,
        )

        memories = result.get("results", [])

        # Filter out low-relevance results
        threshold = self._config.memory.search_score_threshold
        memories = [m for m in memories if m.get("score", 0) >= threshold]

        # Post-retrieval keyword boost: blend keyword overlap into scores
        memories = self._keyword_boost(query, memories)

        # Post-filtering for decayed memories (when include_decayed=True,
        # the search doesn't filter them, so we need to mark them)
        if include_decayed:
            self._mark_decayed(memories)

        # Access tracking: update last_accessed_at, access_count, reset TTL
        self._track_access(memories)

        return memories

    def search_memories_dual_scope(
        self,
        query: str,
        *,
        user_id: str,
        session_agent_id: str,
        memory_type: str | None = None,
        categories: list[str] | None = None,
        role: str | None = None,
        limit: int = 10,
        include_decayed: bool = False,
        query_vector: list[float] | None = None,
    ) -> list[dict]:
        """Search both agent-scoped and shared memories, merge and deduplicate.

        Used when a session has an agent_id but the LLM doesn't pass one
        (meaning: search everything I can see). Performs two searches:
        1. Agent-scoped memories (agent_id=session_agent_id)
        2. Shared memories (agent_id=None)
        Then merges, deduplicates by memory ID, and reranks.

        When query_vector is provided, both searches reuse the same
        pre-computed vector, avoiding redundant embedding API calls.
        """
        user_id = _validate_id(user_id, "user_id")
        session_agent_id = _validate_id(session_agent_id, "agent_id")

        filters = {}
        if memory_type:
            memory_type = validate_memory_type(memory_type)
            filters["memory_type"] = memory_type
        if role:
            if role not in ("user", "assistant"):
                raise ValueError("role filter must be 'user' or 'assistant'")
            filters["role"] = role

        # Expand category prefixes for native Qdrant filtering
        expanded_categories = None
        if categories:
            expanded_categories = self._expand_category_filter(categories, user_id)

        search_kwargs: dict[str, Any] = {
            "filters": filters if filters else None,
            "categories": expanded_categories,
            "limit": limit,
            "exclude_expired": True,
            "include_decayed": include_decayed,
            "similarity_weight": self._config.memory.search_similarity_weight,
            "query_vector": query_vector,
        }

        # Search 1: agent-scoped memories
        agent_result = self.vector.search(
            query,
            user_id=user_id,
            agent_id=session_agent_id,
            **search_kwargs,
        )
        # Search 2: shared memories only (no agent_id).
        # shared_only=True ensures we only get memories without any agent_id,
        # preventing sub-agent memories from leaking to the parent agent.
        shared_result = self.vector.search(
            query,
            user_id=user_id,
            agent_id=None,
            shared_only=True,
            **search_kwargs,
        )

        # Merge and deduplicate by memory ID
        seen_ids: set[str] = set()
        memories: list[dict] = []
        for mem in agent_result.get("results", []) + shared_result.get("results", []):
            mid = mem.get("id")
            if mid and mid not in seen_ids:
                seen_ids.add(mid)
                memories.append(mem)

        # Sort merged results by score (Qdrant FormulaQuery already applied
        # importance reranking, so the score already includes importance weight)
        memories.sort(key=lambda m: m.get("score", 0), reverse=True)

        # Filter out low-relevance results
        threshold = self._config.memory.search_score_threshold
        memories = [m for m in memories if m.get("score", 0) >= threshold]

        # Post-retrieval keyword boost: blend keyword overlap into scores
        memories = self._keyword_boost(query, memories)

        memories = memories[:limit]

        # Lazily mark decayed_at on expired memories found via include_decayed
        if include_decayed:
            self._mark_decayed(memories)

        # Access tracking
        self._track_access(memories)

        return memories

    # ── Find Memories (AI-powered multi-query search) ─────────────────

    def find_memories(
        self,
        question: str,
        *,
        user_id: str,
        session_agent_id: str | None = None,
        agent_id: str | None = None,
        memory_type: str | None = None,
        categories: list[str] | None = None,
        limit: int = 10,
        role: str | None = None,
        include_decayed: bool = False,
    ) -> dict:
        """Find memories relevant to a complex question using AI-powered search.

        Generates multiple targeted search queries from the question,
        runs each through the vector store, merges and deduplicates results,
        then uses an LLM to rerank by relevance to the original question.

        Args:
            question: The user's natural language question.
            user_id: Required user scope.
            session_agent_id: Session agent for dual-scope search.
            agent_id: Fallback agent_id when no session agent.
            memory_type: Optional memory type filter.
            categories: Optional category filter.
            limit: Maximum results to return.
            role: Optional role filter ("user" or "assistant").
            include_decayed: If True, include expired/decayed memories.

        Returns:
            Dict with "results" (list of memory dicts sorted by relevance),
            "queries" (list of generated search queries), and "stats"
            (search statistics including searched, merged, dropped counts).

        Raises:
            ValueError: If LLM calls fail (query generation or reranking).
        """
        user_id = _validate_id(user_id, "user_id")
        num_queries = self._config.memory.find_memories_queries
        threshold = self._config.memory.search_score_threshold

        # Statistics tracking
        total_searched = 0

        # Step 1: Generate diverse search queries from the question
        messages, schema = build_query_generation_prompt(
            question,
            num_queries=num_queries,
        )
        raw_response = self._llm.generate(messages, json_schema=schema)
        try:
            data = parse_json_response(raw_response)
            queries = data.get("queries", [])
            if not isinstance(queries, list) or not queries:
                raise ValueError("LLM returned no search queries")
        except (ValueError, KeyError) as e:
            raise ValueError(f"Failed to generate search queries: {e}") from e

        logger.info(
            "find_memories: generated %d queries for question: %s",
            len(queries),
            question[:100],
        )

        # Step 2: Batch embed all queries upfront (single API call)
        valid_queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        if not valid_queries:
            return {
                "results": [],
                "queries": [],
                "stats": {
                    "searched": 0,
                    "merged": 0,
                    "reranked": False,
                    "dropped": 0,
                    "returned": 0,
                },
            }

        query_vectors = self.vector.embedding.embed_batch(valid_queries)

        # Step 3: Run each query through search, collect results
        per_query_limit = limit * 2
        merged_by_id: dict[str, dict] = {}

        for q, qvec in zip(valid_queries, query_vectors):
            try:
                if session_agent_id:
                    results = self.search_memories_dual_scope(
                        q,
                        user_id=user_id,
                        session_agent_id=session_agent_id,
                        memory_type=memory_type,
                        categories=categories,
                        role=role,
                        limit=per_query_limit,
                        include_decayed=include_decayed,
                        query_vector=qvec,
                    )
                else:
                    results = self.search_memories(
                        q,
                        user_id=user_id,
                        agent_id=agent_id,
                        memory_type=memory_type,
                        categories=categories,
                        role=role,
                        limit=per_query_limit,
                        include_decayed=include_decayed,
                        query_vector=qvec,
                    )
            except Exception:
                logger.warning("find_memories: search failed for query '%s'", q)
                continue

            total_searched += len(results)

            for mem in results:
                mid = mem.get("id")
                if not mid:
                    continue
                existing = merged_by_id.get(mid)
                if existing is None:
                    merged_by_id[mid] = mem
                elif mem.get("score", 0) > existing.get("score", 0):
                    existing["score"] = mem["score"]

        merged = list(merged_by_id.values())

        logger.info(
            "find_memories: %d unique memories from %d queries (searched %d)",
            len(merged),
            len(valid_queries),
            total_searched,
        )

        if not merged:
            return {
                "results": [],
                "queries": valid_queries,
                "stats": {
                    "searched": total_searched,
                    "merged": 0,
                    "reranked": False,
                    "dropped": 0,
                    "returned": 0,
                },
            }

        # Step 4: If few enough results, skip reranking
        if len(merged) <= limit:
            merged.sort(key=lambda m: m.get("score", 0), reverse=True)
            self._track_access(merged)
            return {
                "results": merged,
                "queries": valid_queries,
                "stats": {
                    "searched": total_searched,
                    "merged": len(merged),
                    "reranked": False,
                    "dropped": 0,
                    "returned": len(merged),
                },
            }

        # Step 5: Pre-filter to top candidates before reranking (performance)
        # Sort by vector score and take top limit*3 to reduce LLM payload
        merged.sort(key=lambda m: m.get("score", 0), reverse=True)
        rerank_candidates = merged[: limit * 3]
        pre_filtered = len(merged) - len(rerank_candidates)

        # Step 6: LLM reranking with the original question as context
        # Uses numeric indices instead of UUIDs to reduce output tokens
        messages, schema = build_rerank_prompt(question, rerank_candidates)
        raw_response = self._llm.generate(messages, json_schema=schema)
        try:
            data = parse_json_response(raw_response)
            scored = data.get("scored", [])
            if not isinstance(scored, list):
                raise ValueError("LLM returned invalid rerank response")
        except (ValueError, KeyError) as e:
            raise ValueError(f"Failed to rerank memories: {e}") from e

        # Map LLM relevance scores back to memories using numeric indices
        score_map: dict[int, float] = {}
        for item in scored:
            if isinstance(item, dict):
                idx = item.get("idx")
                rel = item.get("relevance")
                if isinstance(idx, int) and isinstance(rel, (int, float)):
                    score_map[idx] = float(rel)

        # Apply LLM scores, filter by threshold, sort
        reranked = []
        for idx, mem in enumerate(rerank_candidates):
            if idx in score_map:
                mem["score"] = round(score_map[idx], 4)
                if mem["score"] >= threshold:
                    reranked.append(mem)

        reranked.sort(key=lambda m: m.get("score", 0), reverse=True)
        reranked = reranked[:limit]

        dropped = len(rerank_candidates) - len(reranked)

        self._track_access(reranked)
        return {
            "results": reranked,
            "queries": valid_queries,
            "stats": {
                "searched": total_searched,
                "merged": len(merged),
                "reranked": True,
                "dropped": dropped + pre_filtered,
                "returned": len(reranked),
            },
        }

    # ── Get Core Memories ─────────────────────────────────────────────

    # Memory types to include in recent context
    RECENT_MEMORY_TYPES = ["episodic", "context", "procedural"]

    def _format_recent_memory_line(self, memory: dict) -> str:
        """Format a memory for recent context display.

        Format: - [YYYY-MM-DD HH:MM] [type | cat1, cat2] Memory text
        """
        ts = memory.get("created_at", "")
        if ts and "T" in ts:
            ts = ts.split("T")[0] + " " + ts.split("T")[1][:5]

        text = memory.get("memory", "")
        meta = memory.get("metadata") or {}
        mtype = meta.get("memory_type", "")
        cats = meta.get("categories", [])

        # Format: [type | cat1, cat2] or [type] or [cat1, cat2]
        parts = []
        if mtype:
            parts.append(mtype)
        if cats:
            parts.append(", ".join(cats))

        if parts:
            meta_str = f"[{' | '.join(parts)}] "
        else:
            meta_str = ""

        return f"- [{ts}] {meta_str}{text}"

    def get_core_memories(
        self,
        *,
        user_id: str,
        agent_id: str | None = None,
        recent_days: int | None = None,
    ) -> str:
        """Assemble core context for conversation start.

        Returns a structured text with:
        1. Pinned agent memories (identity, knowledge) — if agent_id provided
        2. Pinned user memories (facts, preferences) — shared across agents
        3. Recent context memories from the last N days (dual-scope: user + agent)

        Total output is capped at MAX_CORE_CONTEXT_LENGTH.
        """
        user_id = _validate_id(user_id, "user_id")
        if agent_id:
            agent_id = _validate_id(agent_id, "agent_id")

        if recent_days is None:
            recent_days = self._config.memory.default_recent_days
        recent_days = int(recent_days)

        # Check cache
        cache_key = (user_id, agent_id or "", recent_days)
        cached = self._core_cache.get(cache_key)
        if cached is not None:
            return cached

        max_len = self._config.memory.max_core_context_length

        sections: list[str] = []

        # Collect IDs of pinned memories for deduplication with recent
        pinned_ids: set[str] = set()

        # 1. Pinned agent memories (identity, knowledge, and agent-scoped instructions)
        if agent_id:
            agent_pinned = self.vector.get_pinned_memories(
                user_id=user_id, agent_id=agent_id
            )
            # Only include memories that have this agent_id
            agent_only = [m for m in agent_pinned if m.get("agent_id") == agent_id]
            for m in agent_only:
                if m.get("id"):
                    pinned_ids.add(m["id"])
            if agent_only:
                identity = []
                knowledge = []
                instructions = []
                for m in agent_only:
                    meta = m.get("metadata") or {}
                    mt = meta.get("memory_type", "fact")
                    mr = meta.get("role", "user")
                    text = m.get("memory", "")
                    if mr == "assistant":
                        # Memory about the agent itself
                        if mt in ("preference", "fact"):
                            identity.append(f"- {text}")
                        else:
                            knowledge.append(f"- {text}")
                    else:
                        # Memory about the user, scoped to this agent
                        instructions.append(f"- {text}")
                if identity:
                    sections.append("## Agent Identity\n" + "\n".join(identity))
                if knowledge:
                    sections.append("## Agent Knowledge\n" + "\n".join(knowledge))
                if instructions:
                    sections.append("## Agent Instructions\n" + "\n".join(instructions))

        # 2. Pinned user memories (no agent_id — shared across agents)
        user_pinned = self.vector.get_pinned_memories(
            user_id=user_id, exclude_agent=True
        )
        for m in user_pinned:
            if m.get("id"):
                pinned_ids.add(m["id"])
        if user_pinned:
            facts = []
            prefs = []
            other = []
            for m in user_pinned:
                mt = (m.get("metadata") or {}).get("memory_type", "fact")
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

        # 3. Recent context memories (dual-scope: user + agent)
        since = datetime.now(timezone.utc) - timedelta(days=recent_days)
        limit_user = self._config.memory.recent_limit_user
        limit_agent = self._config.memory.recent_limit_agent

        # Fetch user recent memories (no agent_id — shared)
        user_recent = self.vector.get_recent_memories(
            user_id=user_id,
            agent_id=None,
            since=since,
            limit=limit_user,
            memory_types=self.RECENT_MEMORY_TYPES,
        )
        # Filter out expired/decayed and already-pinned
        user_recent = [
            m
            for m in user_recent
            if not should_exclude(m) and m.get("id") not in pinned_ids
        ]

        # Fetch agent recent memories (with agent_id)
        agent_recent = []
        if agent_id:
            agent_recent = self.vector.get_recent_memories(
                user_id=user_id,
                agent_id=agent_id,
                since=since,
                limit=limit_agent,
                memory_types=self.RECENT_MEMORY_TYPES,
            )
            # Filter out expired/decayed and already-pinned
            agent_recent = [
                m
                for m in agent_recent
                if not should_exclude(m) and m.get("id") not in pinned_ids
            ]

        # Build recent context section with subsections
        if user_recent or agent_recent:
            recent_parts = ["## Recent Context"]
            if user_recent:
                recent_parts.append("\n### User Activity")
                for m in user_recent:
                    recent_parts.append(self._format_recent_memory_line(m))
            if agent_recent:
                recent_parts.append("\n### Agent Activity")
                for m in agent_recent:
                    recent_parts.append(self._format_recent_memory_line(m))
            sections.append("\n".join(recent_parts))

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

    # ── Get Recent Memories ───────────────────────────────────────────

    def get_recent_memories(
        self,
        *,
        user_id: str,
        agent_id: str | None = None,
        days: int | None = None,
        scope: str = "all",
        limit: int | None = None,
        include_decayed: bool = False,
    ) -> str:
        """Get recent memories from the last N days.

        Returns episodic, context, and procedural memories ordered by most
        recent first, formatted for display.

        Args:
            user_id: Required user scope.
            agent_id: Agent identifier for agent-scoped memories.
            days: How many days back to look (default from config).
            scope: Which memories to include:
                   - "all": Both user and agent memories (default)
                   - "user": Only shared user memories (no agent_id)
                   - "agent": Only agent-scoped memories (requires agent_id)
            limit: Max results per scope (default from config).
            include_decayed: Include expired/decayed memories (default false).

        Returns formatted text with recent memories.
        """
        user_id = _validate_id(user_id, "user_id")
        if agent_id:
            agent_id = _validate_id(agent_id, "agent_id")

        if scope not in ("all", "user", "agent"):
            raise ValueError(
                f"Invalid scope: {scope}. Must be 'all', 'user', or 'agent'"
            )

        if scope == "agent" and not agent_id:
            raise ValueError("agent_id is required when scope='agent'")

        if days is None:
            days = self._config.memory.default_recent_days
        days = int(days)

        if limit is None:
            # Use config defaults based on scope
            if scope == "all":
                limit_user = self._config.memory.recent_limit_user
                limit_agent = self._config.memory.recent_limit_agent
            elif scope == "user":
                limit_user = (
                    self._config.memory.recent_limit_user
                    + self._config.memory.recent_limit_agent
                )
                limit_agent = 0
            else:  # agent
                limit_user = 0
                limit_agent = (
                    self._config.memory.recent_limit_user
                    + self._config.memory.recent_limit_agent
                )
        else:
            # Use provided limit, split evenly for "all" scope
            if scope == "all":
                limit_user = limit
                limit_agent = limit
            elif scope == "user":
                limit_user = limit
                limit_agent = 0
            else:  # agent
                limit_user = 0
                limit_agent = limit

        since = datetime.now(timezone.utc) - timedelta(days=days)

        user_recent = []
        agent_recent = []

        # Fetch user recent memories
        if limit_user > 0 and scope in ("all", "user"):
            user_recent = self.vector.get_recent_memories(
                user_id=user_id,
                agent_id=None,
                since=since,
                limit=limit_user,
            )
            user_recent = [
                m for m in user_recent if not should_exclude(m, include_decayed)
            ]

        # Fetch agent recent memories
        if limit_agent > 0 and scope in ("all", "agent") and agent_id:
            agent_recent = self.vector.get_recent_memories(
                user_id=user_id,
                agent_id=agent_id,
                since=since,
                limit=limit_agent,
            )
            agent_recent = [
                m for m in agent_recent if not should_exclude(m, include_decayed)
            ]

        if not user_recent and not agent_recent:
            return "No recent memories found."

        # Build output
        parts = ["## Recent Memories"]
        if user_recent:
            parts.append("\n### User Activity")
            for m in user_recent:
                parts.append(self._format_recent_memory_line(m))
        if agent_recent:
            parts.append("\n### Agent Activity")
            for m in agent_recent:
                parts.append(self._format_recent_memory_line(m))

        return "\n".join(parts)

    # ── List Memories ─────────────────────────────────────────────────

    def list_memories(
        self,
        *,
        user_id: str,
        agent_id: str | None = None,
        memory_type: str | None = None,
        categories: list[str] | None = None,
        role: str | None = None,
        limit: int = 50,
        include_decayed: bool = False,
    ) -> list[dict]:
        """List memories with optional filtering.

        By default, expired and decayed memories are excluded. Set
        include_decayed=True to include them (useful for browsing
        historical memories).
        """
        user_id = _validate_id(user_id, "user_id")
        if agent_id:
            agent_id = _validate_id(agent_id, "agent_id")

        filters = {}
        if memory_type:
            memory_type = validate_memory_type(memory_type)
            filters["memory_type"] = memory_type
        if role:
            if role not in ("user", "assistant"):
                raise ValueError("role filter must be 'user' or 'assistant'")
            filters["role"] = role

        result = self.vector.get_all(
            user_id=user_id,
            agent_id=agent_id,
            filters=filters if filters else None,
            limit=limit * 2,  # Fetch extra for post-filtering
        )

        memories = result.get("results", [])

        # Post-filter TTL (get_all has no native TTL filtering)
        memories = [m for m in memories if not should_exclude(m, include_decayed)]

        if categories:
            memories = [
                m
                for m in memories
                if matches_category_filter(
                    (m.get("metadata") or {}).get("categories", []), categories
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
        role: str | None = None,
        limit: int = 50,
        include_decayed: bool = False,
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
        if role:
            if role not in ("user", "assistant"):
                raise ValueError("role filter must be 'user' or 'assistant'")
            filters["role"] = role

        fetch_limit = limit * 2

        # Fetch 1: agent-scoped memories
        agent_result = self.vector.get_all(
            user_id=user_id,
            agent_id=session_agent_id,
            filters=filters if filters else None,
            limit=fetch_limit,
        )
        # Fetch 2: shared memories only (no agent_id).
        # shared_only=True ensures we only get memories without any agent_id,
        # preventing sub-agent memories from leaking to the parent agent.
        shared_result = self.vector.get_all(
            user_id=user_id,
            agent_id=None,
            shared_only=True,
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

        # Post-filter TTL
        memories = [m for m in memories if not should_exclude(m, include_decayed)]

        # Post-filter by categories
        if categories:
            memories = [
                m
                for m in memories
                if matches_category_filter(
                    (m.get("metadata") or {}).get("categories", []), categories
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
        - Have agent_id that is a sub-agent of session_agent_id
          (e.g., memory agent_id="openwebui:bob", session="openwebui")

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
            # Allow access to sub-agent memories (e.g., "openwebui:bob"
            # is accessible from session "openwebui")
            if not mem_agent_id.startswith(session_agent_id + ":"):
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
        ttl_days: int | None = ...,  # type: ignore[assignment]
    ) -> dict:
        """Update a memory's content and/or metadata.

        Content updates re-embed and replace the vector store point.
        Metadata updates use direct payload updates to avoid losing
        custom fields.

        ttl_days: Set a new TTL (recalculates expires_at from now). Pass 0
                  or None to make permanent. Uses sentinel default (...) to
                  distinguish "not provided" from "explicitly set to None".
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

        # TTL update: recalculate expires_at, clear decayed_at (restore)
        if ttl_days is not ...:
            from mnemory.ttl import calculate_expiration

            effective_ttl = ttl_days if ttl_days else None  # 0 → None (permanent)
            metadata_updates["ttl_days"] = effective_ttl
            metadata_updates["expires_at"] = calculate_expiration(effective_ttl)
            metadata_updates["decayed_at"] = None  # Restore from decay

        # Update content (re-embeds)
        if content is not None:
            max_len = self._config.memory.max_memory_length
            if len(content) > max_len:
                return {
                    "error": True,
                    "message": (
                        f"Content too long: {len(content)} chars (max {max_len})."
                    ),
                }
            self.vector.update_content(memory_id, content)

        # Update metadata directly on the vector store
        if metadata_updates:
            metadata_updates["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            self.vector.update_metadata(memory_id, metadata_updates)

        # Invalidate caches — updated memory may be pinned or recent.
        if user_id:
            self._core_cache.invalidate_prefix(user_id)
        else:
            self._core_cache.clear()

        # Invalidate category cache when categories are updated
        if categories is not None:
            if user_id:
                self._category_cache.invalidate(user_id)
            else:
                self._category_cache.clear()

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
                if mem_id and (mem.get("metadata") or {}).get("artifacts"):
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
                (mem.get("metadata") or {}).get("artifacts", []) if mem else []
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
            return (mem.get("metadata") or {}).get("artifacts", [])
        return []

    def _expand_category_filter(self, categories: list[str], user_id: str) -> list[str]:
        """Expand category prefix patterns into concrete category values.

        For example, "project" expands to ["project", "project:myapp",
        "project:domecek"] based on existing memories. Non-prefix categories
        are passed through as-is.

        This allows Qdrant's MatchAny to handle category filtering natively
        instead of post-filtering in Python.
        """
        available = self._get_available_categories(user_id)
        expanded: list[str] = []
        for cat in categories:
            expanded.append(cat)
            # Expand prefix: "project" → also include "project:foo", etc.
            prefix = cat + ":"
            for avail in available:
                if avail.startswith(prefix) and avail not in expanded:
                    expanded.append(avail)
        return expanded

    def _mark_decayed(self, memories: list[dict]) -> None:
        """Lazily set decayed_at on expired memories that haven't been marked yet.

        Called when include_decayed=True returns expired memories. Sets
        decayed_at via batch metadata update so subsequent queries can
        distinguish "newly expired" from "already decayed".
        """
        updates: list[tuple[str, dict]] = []
        now_iso = datetime.now(timezone.utc).isoformat()
        for mem in memories:
            metadata = mem.get("metadata") or {}
            if metadata.get("decayed_at") is None:
                if is_expired(mem):
                    updates.append((mem["id"], {"decayed_at": now_iso}))
        if updates:
            try:
                self.vector.batch_update_metadata(updates)
            except Exception:
                logger.warning("Failed to mark decayed memories")

    def _track_access(self, memories: list[dict]) -> None:
        """Update access tracking metadata on searched memories.

        Sets last_accessed_at, increments access_count, and resets TTL
        (reinforcement) for memories that have a ttl_days value.

        Gated by TRACK_MEMORY_ACCESS config. Runs synchronously via
        batch set_payload after search returns.
        """
        if not self._config.memory.track_memory_access:
            return
        if not memories:
            return

        updates: list[tuple[str, dict]] = []
        for mem in memories:
            mem_id = mem.get("id")
            if not mem_id:
                continue
            reinforcement = build_reinforcement_metadata(mem)
            updates.append((mem_id, reinforcement))

        if updates:
            try:
                self.vector.batch_update_metadata(updates)
            except Exception:
                logger.warning("Failed to track memory access")
