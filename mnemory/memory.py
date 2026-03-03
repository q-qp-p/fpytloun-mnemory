"""Business logic layer for mnemory.

Orchestrates the vector store (fast memory) and artifact store (slow memory)
to provide a unified memory interface. Handles validation, reranking,
core memory assembly, and artifact lifecycle.

The add pipeline uses a single LLM call for fact extraction, per-fact
classification, and deduplication against existing memories.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime, timedelta, timezone
from typing import Any

from mnemory.cache import TTLCache
from mnemory.categories import (
    IMPORTANCE_WEIGHTS,
    PREDEFINED_CATEGORIES,
    count_categories,
    importance_levels_at_or_above,
    matches_category_filter,
    validate_categories,
    validate_importance,
    validate_memory_type,
)
from mnemory.config import Config
from mnemory.llm import LLMClient, parse_json_response
from mnemory.prompts import (
    build_classification_prompt,
    build_dedup_prompt,
    build_extraction_prompt,
    build_query_generation_prompt,
    build_remember_extraction_prompt,
    build_rerank_prompt,
    build_shorten_prompt,
    build_summary_compaction_prompt,
    parse_dedup_response,
    parse_extraction_response,
    parse_remember_extraction_response,
)
from mnemory.sanitize import (
    CORE_MEMORIES_PREAMBLE,
    detect_injection_patterns,
    log_injection_warning,
    wrap_memory_item,
)
from mnemory.sparse import SparseEmbeddingClient
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


@dataclass
class CoreMemoriesResult:
    """Result from get_core_memories() with text and included memory IDs.

    Attributes:
        text: Formatted core memories text for injection into LLM context.
        memory_ids: Set of memory IDs included in the text, used to
            deduplicate against search results.
    """

    text: str
    memory_ids: set[str] = dataclass_field(default_factory=set)


def _validate_id(value: str, name: str) -> str:
    """Validate a user_id or agent_id string."""
    if not value or not value.strip():
        raise ValueError(f"{name} must not be empty")
    value = value.strip()
    if len(value) > _MAX_ID_LENGTH:
        raise ValueError(f"{name} too long (max {_MAX_ID_LENGTH} chars)")
    return value


def _sort_to_order_by(sort: str | None):
    """Convert a sort string to a Qdrant OrderBy object.

    Args:
        sort: "newest" for descending by created_at, "oldest" for ascending.
              None or empty string returns None (no ordering).

    Returns:
        OrderBy instance or None.
    """
    if not sort:
        return None
    from qdrant_client.models import Direction, OrderBy

    if sort == "newest":
        return OrderBy(key="created_at", direction=Direction.DESC)
    elif sort == "oldest":
        return OrderBy(key="created_at", direction=Direction.ASC)
    return None


def _parse_event_date(value: str, default_tz: str = "") -> str:
    """Parse and normalize an event_date string to UTC ISO 8601.

    Accepts ISO 8601 strings (with or without timezone).
    Returns UTC ISO 8601 string with timezone offset.

    Args:
        value: ISO 8601 datetime string to parse.
        default_tz: IANA timezone name (e.g., "UTC", "Europe/Prague") to
            assume for naive datetimes (no timezone info). Empty string
            means use the server's local timezone.

    Raises:
        ValueError: If the string cannot be parsed as a datetime.
    """
    from zoneinfo import ZoneInfo

    try:
        dt = datetime.fromisoformat(value)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid event_date format: '{value}'. "
            "Use ISO 8601 format (e.g., '2023-05-08T13:56:00+02:00' "
            "or '2023-05-08')."
        ) from e

    # If no timezone, apply the configured default
    if dt.tzinfo is None:
        if default_tz:
            try:
                tz = ZoneInfo(default_tz)
            except (KeyError, Exception) as e:
                raise ValueError(
                    f"Invalid DEFAULT_TIMEZONE: '{default_tz}'. "
                    "Use an IANA timezone name (e.g., 'UTC', 'Europe/Prague')."
                ) from e
        else:
            # Server local timezone
            tz = datetime.now().astimezone().tzinfo
        dt = dt.replace(tzinfo=tz)

    # Normalize to UTC
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.isoformat()


class MemoryService:
    """High-level memory service combining fast and slow memory tiers.

    Fast memory: concise facts/summaries in a vector store (searchable).
    Slow memory: detailed artifacts in S3/filesystem (retrieved on demand).
    """

    def __init__(
        self,
        config: Config,
        session_store: Any | None = None,
    ):
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
        self._core_cache: TTLCache[tuple, CoreMemoriesResult] = TTLCache(
            ttl_seconds=config.memory.core_memories_cache_ttl,
        )

        # Session store for remember pipeline context (optional).
        # When set, remember() uses session context for better extraction
        # and dedup. When None, remember() falls back to stateless mode.
        self._session_store = session_store

        # Per-user lock for serializing concurrent remember calls.
        # Prevents race conditions where two calls search before either
        # writes, causing both to ADD the same fact.
        self._user_locks: dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()
        self._max_user_locks = 1000

        # Sparse embedding client for hybrid search (BM25).
        # Initialized eagerly — fail fast at startup if model fails to load.
        sparse_model = config.memory.search_sparse_model
        self._sparse = SparseEmbeddingClient(model_name=sparse_model)

        # Warn if deprecated keyword weight is set
        if config.memory.search_keyword_weight > 0:
            logger.warning(
                "SEARCH_KEYWORD_WEIGHT is deprecated and ignored. "
                "BM25 sparse vectors replace the Python keyword boost."
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
        event_date: str | None = None,
        session_timezone: str | None = None,
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
            event_date: ISO 8601 datetime for when the event occurred. Used to
                  anchor relative time references during extraction and stored
                  as metadata.
            session_timezone: IANA timezone from X-Timezone header. Overrides
                  DEFAULT_TIMEZONE for naive event_date parsing. Priority:
                  explicit tz in event_date > session_timezone > DEFAULT_TIMEZONE
                  > server local.

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

        # Agent identity memories must go through LLM extraction to prevent
        # direct injection of arbitrary instructions as agent personality.
        if role == "assistant" and not infer:
            raise ValueError(
                "infer=False is not allowed for role='assistant'. "
                "Agent identity memories must go through fact extraction "
                "to prevent prompt injection. Use infer=True (default)."
            )

        # Log when infer=False is used — it bypasses extraction, dedup,
        # and most security processing. Legitimate but worth monitoring.
        if not infer:
            logger.info(
                "infer=False used for add_memory: user_id=%s agent_id=%s "
                "content_length=%d",
                user_id,
                agent_id or "",
                len(content),
            )

        # Detect potential prompt injection patterns (log only, never block)
        patterns = detect_injection_patterns(content)
        if patterns:
            log_injection_warning(
                content,
                patterns,
                user_id=user_id,
                agent_id=agent_id or "",
                operation="add_memory",
            )

        # Parse and normalize event_date to UTC ISO 8601
        # Timezone priority: explicit tz in string > session header > config > server local
        normalized_event_date: str | None = None
        if event_date is not None:
            default_tz = session_timezone or self._config.memory.default_timezone
            normalized_event_date = _parse_event_date(event_date, default_tz)

        if infer:
            # For infer=True: cap input at model context budget, not memory length.
            # The LLM extracts concise facts; max_memory_length applies to output.
            max_input = self._config.memory.max_input_length
            if len(content) > max_input:
                return {
                    "error": True,
                    "message": (
                        f"Input too long: {len(content)} chars "
                        f"(max {max_input}). Shorten the input or split "
                        "into multiple calls."
                    ),
                }
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
                event_date=normalized_event_date,
                session_timezone=session_timezone,
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
                event_date=normalized_event_date,
            )

        # Invalidate caches — new memory may affect core memories or categories
        self._core_cache.invalidate_prefix(user_id)
        self._category_cache.invalidate(user_id)

        return result

    def remember(
        self,
        content: str,
        *,
        user_id: str,
        agent_id: str | None = None,
        role: str = "user",
        session_id: str | None = None,
        session_timezone: str | None = None,
        context: str | None = None,
    ) -> dict:
        """Process conversation content for memory extraction.

        Uses a two-stage pipeline:
        1. Extract facts from conversation text (with session context)
        2. Dedup each extracted fact against stored memories (per-fact
           embedding for accurate similarity search)

        Unlike add_memory, this method:
        - Accepts longer input (conversation turns) without MAX_MEMORY_LENGTH
          check on input
        - Auto-classifies everything (no explicit metadata)
        - Uses REMEMBER_ARTIFACT_THRESHOLD for auto-artifact decisions
        - Maintains session context (conversation summary + extracted facts)
        - Designed for fire-and-forget background processing from plugins

        Args:
            session_id: Session ID for context tracking. When provided,
                the pipeline uses accumulated conversation context from
                previous remember calls in the same session.
            context: Optional context hint (e.g., working directory, active
                project). Passed to the extraction prompt to help identify
                the project and produce self-contained memories.

        The extraction pipeline produces concise facts that respect
        MAX_MEMORY_LENGTH. Returns the same format as add_memory.
        """
        # Validate inputs
        if role not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        user_id = _validate_id(user_id, "user_id")
        if agent_id:
            agent_id = _validate_id(agent_id, "agent_id")
        if role == "assistant" and not agent_id:
            raise ValueError("agent_id is required when role='assistant'")

        # Cap at model context budget (keep most recent content)
        max_input = self._config.memory.max_input_length
        if len(content) > max_input:
            content = content[-max_input:]

        # Acquire per-user lock to serialize concurrent remember calls.
        # This prevents race conditions where two calls search before
        # either writes, causing both to ADD the same fact.
        lock = self._get_user_lock(user_id)
        with lock:
            result = self._remember_pipeline(
                content,
                user_id=user_id,
                agent_id=agent_id,
                role=role,
                session_id=session_id,
                session_timezone=session_timezone,
                context=context,
            )

        # Invalidate caches
        self._core_cache.invalidate_prefix(user_id)
        self._category_cache.invalidate(user_id)

        return result

    def _get_user_lock(self, user_id: str) -> threading.Lock:
        """Get or create a per-user lock for serializing remember calls."""
        with self._locks_lock:
            lock = self._user_locks.get(user_id)
            if lock is None:
                # Evict oldest entries if over limit
                if len(self._user_locks) >= self._max_user_locks:
                    # Remove first entry (oldest insertion in CPython 3.7+)
                    oldest_key = next(iter(self._user_locks))
                    del self._user_locks[oldest_key]
                lock = threading.Lock()
                self._user_locks[user_id] = lock
            return lock

    def _remember_pipeline(
        self,
        content: str,
        *,
        user_id: str,
        agent_id: str | None,
        role: str,
        session_id: str | None,
        session_timezone: str | None,
        context: str | None,
    ) -> dict:
        """Two-stage remember pipeline: extract then dedup.

        Stage 1: Extract facts from conversation text with session context.
        Stage 2: Dedup each fact against stored memories using per-fact
                 embeddings for accurate similarity search.
        """
        # 1. Get session context (if available)
        session_context: dict[str, Any] | None = None
        if session_id and self._session_store:
            session_context = self._session_store.get_remember_context(session_id)
            # Empty context is equivalent to no context
            if session_context and not (
                session_context.get("extracted_memories")
                or session_context.get("conversation_summary")
            ):
                session_context = None

        # 2. Stage 1: Extract facts from conversation
        available_cats = self._get_available_categories(user_id)
        max_len = self._config.memory.max_memory_length

        facts, summary, store_artifact = self._remember_extract(
            content,
            role=role,
            session_context=session_context,
            available_categories=available_cats,
            max_memory_length=max_len,
            session_timezone=session_timezone,
            context=context,
        )

        if not facts:
            # Still update session with summary even if no facts extracted
            self._update_session_context(session_id, [], summary)
            return {"results": []}

        # 2b. Truncate any oversized facts (the extraction prompt already
        # constrains length via max_length, so this is a rare safety net).
        # We skip _validate_fact_lengths here because its split logic can
        # change the list length, breaking the 1:1 correspondence with facts.
        for f in facts:
            if len(f["text"]) > max_len:
                logger.warning(
                    "Remember extraction produced oversized fact (%d chars, "
                    "max %d), truncating: %.80s...",
                    len(f["text"]),
                    max_len,
                    f["text"],
                )
                f["text"] = f["text"][:max_len]

        # 3. Stage 2: Dedup + Store
        results = self._remember_dedup_and_store(
            facts,
            user_id=user_id,
            agent_id=agent_id,
            role=role,
        )

        # 4. Update session context
        extracted_texts = [f["text"] for f in facts]
        self._update_session_context(session_id, extracted_texts, summary)

        # 5. Auto-artifact
        artifact_threshold = self._config.memory.remember_artifact_threshold
        response: dict[str, Any] = {"results": results}
        artifact_info = self._maybe_create_auto_artifact(
            store_artifact=store_artifact,
            content=content,
            user_id=user_id,
            results=results,
            threshold=artifact_threshold,
        )
        if artifact_info:
            response["artifact"] = artifact_info

        return response

    def _remember_extract(
        self,
        content: str,
        *,
        role: str = "user",
        session_context: dict[str, Any] | None,
        available_categories: list[str],
        max_memory_length: int,
        session_timezone: str | None,
        context: str | None,
    ) -> tuple[list[dict[str, Any]], str, bool]:
        """Stage 1: Extract facts from conversation text.

        Retries once if the LLM returns a non-empty response that fails
        to parse (non-deterministic JSON formatting issues).

        Args:
            role: "user" (default) or "assistant". Selects which extraction
                prompt template to use. "assistant" extracts facts about the
                agent itself (identity, personality, capabilities, research
                conclusions).

        Returns:
            Tuple of (facts, summary, store_artifact).
            facts: list of dicts with text, memory_type, categories,
                   importance, pinned, event_date.
            summary: turn summary for session context.
            store_artifact: whether to save original as artifact.
        """
        messages, json_schema = build_remember_extraction_prompt(
            content,
            role=role,
            session_context=session_context,
            available_categories=available_categories,
            max_memory_length=max_memory_length,
            session_timezone=session_timezone,
            context=context,
        )

        facts: list[dict[str, Any]] = []
        summary = ""
        store_artifact = False

        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                response_text = self._llm.generate(
                    messages,
                    json_schema=json_schema,
                    max_tokens=2000,
                )
            except Exception:
                logger.exception("Remember extraction LLM call failed")
                return [], "", False

            logger.debug("Remember extraction response: %s", response_text)

            facts, summary, store_artifact = parse_remember_extraction_response(
                response_text
            )

            # Retry on parse failure: non-empty LLM response that produced
            # no facts and no summary (likely malformed JSON). Empty LLM
            # responses are already retried at the LLM client level.
            if (
                not facts
                and not summary
                and response_text.strip()
                and attempt < max_attempts - 1
            ):
                logger.info(
                    "Remember Stage 1: parse returned empty from non-empty "
                    "response, retrying (attempt %d/%d)",
                    attempt + 1,
                    max_attempts,
                )
                continue

            break

        logger.info(
            "Remember Stage 1: extracted %d facts, summary=%d chars",
            len(facts),
            len(summary),
        )

        return facts, summary, store_artifact

    def _remember_dedup_and_store(
        self,
        facts: list[dict[str, Any]],
        *,
        user_id: str,
        agent_id: str | None,
        role: str,
    ) -> list[dict[str, Any]]:
        """Stage 2: Dedup extracted facts against stored memories and store.

        For each fact:
        1. Embed the fact text (batch)
        2. Search Qdrant for similar existing memories
        3. If any facts have candidates: LLM dedup call
        4. Execute actions (ADD/UPDATE/DELETE)

        Returns list of result dicts from _execute_action.
        """
        # 1. Batch embed all fact texts
        fact_texts = [f["text"] for f in facts]
        try:
            vectors = self.vector.embedding.embed_batch(fact_texts)
        except Exception:
            logger.exception("Remember dedup: batch embedding failed")
            return []
        fact_vectors = dict(zip(fact_texts, vectors))

        # 2. Per-fact similarity search
        dedup_threshold = self._config.memory.dedup_similarity_threshold
        facts_with_candidates: list[dict[str, Any]] = []
        has_any_candidates = False

        for i, fact in enumerate(facts):
            vector = fact_vectors.get(fact["text"])
            if vector is None:
                facts_with_candidates.append(
                    {"index": i, "text": fact["text"], "candidates": []}
                )
                continue

            # Search for similar existing memories using the fact's embedding
            existing_raw = self.vector.search_similar(
                vector,
                user_id=user_id,
                agent_id=agent_id,
                limit=5,
            )
            # Dual-scope: also check shared memories when agent_id is set
            if agent_id:
                shared_raw = self.vector.search_similar(
                    vector,
                    user_id=user_id,
                    agent_id=None,
                    shared_only=True,
                    limit=5,
                )
                seen_ids: set[str] = set()
                merged: list[dict[str, Any]] = []
                for mem in sorted(
                    existing_raw + shared_raw,
                    key=lambda m: m.get("score", 0),
                    reverse=True,
                ):
                    mid = mem.get("id")
                    if mid and mid not in seen_ids:
                        seen_ids.add(mid)
                        merged.append(mem)
                existing_raw = merged[:5]

            # Filter by dedup threshold
            candidates = [
                m for m in existing_raw if m.get("score", 0) >= dedup_threshold
            ]

            if candidates:
                has_any_candidates = True

            facts_with_candidates.append(
                {"index": i, "text": fact["text"], "candidates": candidates}
            )

        # 3. Dedup decision
        if has_any_candidates:
            # Build dedup prompt and make LLM call
            actions = self._dedup_with_llm(facts_with_candidates, facts)
        else:
            # No candidates for any fact — all are new, skip LLM call
            logger.debug(
                "Remember Stage 2: no dedup candidates, adding all %d facts",
                len(facts),
            )
            actions = [
                {
                    "text": f["text"],
                    "action": "ADD",
                    "target_id": None,
                    "old_memory": None,
                    "memory_type": f["memory_type"],
                    "categories": f["categories"],
                    "importance": f["importance"],
                    "pinned": f["pinned"],
                    "event_date": f.get("event_date"),
                }
                for f in facts
            ]

        if not actions:
            return []

        # 4. Build vector map for execution (reuse embeddings from step 1)
        vector_map = fact_vectors

        # 5. Execute actions
        results = []
        for action in actions:
            try:
                result_entry = self._execute_action(
                    action,
                    user_id=user_id,
                    agent_id=agent_id,
                    role=role,
                    ttl_days=None,
                    explicit_fields={},
                    vector_map=vector_map,
                    event_date=None,
                )
                if result_entry:
                    results.append(result_entry)
            except Exception:
                logger.exception(
                    "Failed to execute %s action in remember pipeline",
                    action["action"],
                )

        logger.info(
            "Remember Stage 2: executed %d actions (%s)",
            len(results),
            [r.get("event") for r in results],
        )

        return results

    def _dedup_with_llm(
        self,
        facts_with_candidates: list[dict[str, Any]],
        facts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Make a single LLM call for dedup decisions.

        Returns list of action dicts compatible with _execute_action.
        Falls back to ADD-all on LLM failure.
        """
        messages, json_schema, id_mapping = build_dedup_prompt(facts_with_candidates)

        try:
            response_text = self._llm.generate(
                messages,
                json_schema=json_schema,
                max_tokens=2000,
            )
        except Exception:
            logger.exception("Remember dedup LLM call failed, falling back to ADD-all")
            return [
                {
                    "text": f["text"],
                    "action": "ADD",
                    "target_id": None,
                    "old_memory": None,
                    "memory_type": f["memory_type"],
                    "categories": f["categories"],
                    "importance": f["importance"],
                    "pinned": f["pinned"],
                    "event_date": f.get("event_date"),
                }
                for f in facts
            ]

        logger.debug("Remember dedup response: %s", response_text)

        actions, decided_indices = parse_dedup_response(
            response_text, id_mapping, facts
        )

        # For facts not mentioned in the dedup response, default to ADD
        for i, f in enumerate(facts):
            if i not in decided_indices:
                logger.debug(
                    "Remember dedup: fact %d not in response, defaulting to ADD",
                    i,
                )
                actions.append(
                    {
                        "text": f["text"],
                        "action": "ADD",
                        "target_id": None,
                        "old_memory": None,
                        "memory_type": f["memory_type"],
                        "categories": f["categories"],
                        "importance": f["importance"],
                        "pinned": f["pinned"],
                        "event_date": f.get("event_date"),
                    }
                )

        return actions

    def _update_session_context(
        self,
        session_id: str | None,
        extracted_texts: list[str],
        summary: str,
    ) -> None:
        """Update session with extracted memories and conversation summary.

        Also handles summary compaction when the summary exceeds the
        configured threshold.
        """
        if not session_id or not self._session_store:
            return

        try:
            max_memories = self._config.memory.remember_max_session_memories
            if extracted_texts:
                self._session_store.add_extracted_memories(
                    session_id, extracted_texts, max_entries=max_memories
                )

            if summary:
                self._session_store.append_summary(session_id, summary)

                # Check if compaction is needed
                compaction_threshold = (
                    self._config.memory.remember_summary_compaction_threshold
                )
                if compaction_threshold > 0:
                    ctx = self._session_store.get_remember_context(session_id)
                    current_summary = ctx.get("conversation_summary", "")
                    if len(current_summary) > compaction_threshold:
                        self._compact_summary(session_id, current_summary)

        except Exception:
            logger.warning(
                "Failed to update session context for %s",
                session_id,
                exc_info=True,
            )

    def _compact_summary(self, session_id: str, summary: str) -> None:
        """Compact a conversation summary that exceeds the threshold.

        Makes an LLM call to condense the summary. On failure, keeps
        the original (never replaces with a failed result).
        """
        logger.info(
            "Compacting conversation summary (%d chars) for session %s",
            len(summary),
            session_id,
        )

        messages, json_schema = build_summary_compaction_prompt(summary)

        try:
            response_text = self._llm.generate(
                messages,
                json_schema=json_schema,
                max_tokens=2000,
            )
            data = parse_json_response(response_text)
            compacted = str(data.get("summary", "")).strip()

            if compacted and len(compacted) < len(summary):
                self._session_store.set_summary(session_id, compacted)
                logger.info(
                    "Summary compacted: %d -> %d chars",
                    len(summary),
                    len(compacted),
                )
            else:
                logger.warning(
                    "Summary compaction produced no improvement, keeping original"
                )
        except Exception:
            logger.warning(
                "Summary compaction failed, keeping original",
                exc_info=True,
            )

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
        event_date: str | None = None,
        artifact_threshold: int | None = None,
        session_timezone: str | None = None,
        context: str | None = None,
    ) -> dict:
        """Add memory with LLM-driven extraction, classification, and dedup.

        Single LLM call pipeline:
        1. Embed the raw content
        2. Search for similar existing memories
        3. Build unified prompt (extraction + classification + dedup)
        4. Single LLM call
        5. Execute actions (ADD/UPDATE/DELETE)
        6. Auto-artifact if LLM recommends it and content exceeds threshold

        Args:
            artifact_threshold: Minimum content length for auto-artifact
                creation. Defaults to MAX_MEMORY_LENGTH. Pass a different
                value for remember() which uses REMEMBER_ARTIFACT_THRESHOLD.
                Pass 0 to disable auto-artifact entirely.
            session_timezone: IANA timezone from X-Timezone header. When
                event_date is None, used to compute "Today's date" in the
                extraction prompt using the user's local date instead of UTC.
            context: Optional context hint (e.g., working directory, active
                project). Injected into the extraction prompt to help the
                LLM identify the project and produce self-contained facts.
        """
        # 1. Embed the raw content for similarity search
        content_vector = self.vector.embedding.embed(content)

        # 2. Search for similar existing memories (excludes expired/decayed).
        # When agent_id is set, perform dual-scope search: check both
        # agent-scoped AND shared (no agent_id) memories. This prevents
        # duplicates when the same fact already exists as a shared memory.
        existing_raw = self.vector.search_similar(
            content_vector,
            user_id=user_id,
            agent_id=agent_id,
            limit=10,
        )
        if agent_id:
            shared_raw = self.vector.search_similar(
                content_vector,
                user_id=user_id,
                agent_id=None,
                shared_only=True,
                limit=10,
            )
            # Merge and deduplicate by memory ID, keep highest score
            seen_ids: set[str] = set()
            merged: list[dict[str, Any]] = []
            for mem in sorted(
                existing_raw + shared_raw,
                key=lambda m: m.get("score", 0),
                reverse=True,
            ):
                mid = mem.get("id")
                if mid and mid not in seen_ids:
                    seen_ids.add(mid)
                    merged.append(mem)
            existing_raw = merged[:10]

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
            event_date=event_date,
            session_timezone=session_timezone,
            context=context,
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

        logger.debug("Extraction LLM response: %s", response_text)

        # 5. Parse response and execute actions
        actions, store_artifact = parse_extraction_response(response_text, id_mapping)

        logger.debug(
            "Parsed %d actions: %s (store_artifact=%s)",
            len(actions),
            [a["action"] for a in actions],
            store_artifact,
        )

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
                    event_date=event_date,
                )
                if result_entry:
                    results.append(result_entry)
            except Exception:
                logger.exception(
                    "Failed to execute %s action for memory", action["action"]
                )

        # 6. Auto-artifact: save original content when LLM recommends it
        effective_threshold = (
            artifact_threshold if artifact_threshold is not None else max_len
        )
        response: dict[str, Any] = {"results": results}
        artifact_info = self._maybe_create_auto_artifact(
            store_artifact=store_artifact,
            content=content,
            user_id=user_id,
            results=results,
            threshold=effective_threshold,
        )
        if artifact_info:
            response["artifact"] = artifact_info

        return response

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
                shortened, _ = parse_extraction_response(response_text, {})
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
        event_date: str | None = None,
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

        # Resolve event_date: LLM-extracted per-fact date takes precedence
        # over caller-provided date. The caller's event_date is primarily
        # used to anchor "Today's date" in the extraction prompt; the LLM's
        # per-fact event_date is more accurate (actual event date vs session
        # timestamp). Falls back to caller's date when LLM returns null.
        effective_event_date = action.get("event_date") or event_date

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
            if effective_event_date is not None:
                metadata["event_date"] = effective_event_date
            ttl_meta = build_expiry_metadata(ttl_days, mem_type, self._config.memory)
            metadata.update(ttl_meta)

            memory_id = self.vector.insert(
                text=text,
                vector=vector,
                user_id=user_id,
                agent_id=agent_id,
                metadata=metadata,
                role=role,
                sparse_vector=self._get_sparse_vector(text),
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
                    event_date=event_date,
                )

            # Build updated metadata — preserve existing, override with new
            existing_meta = existing.get("metadata") or {}
            metadata_update = {
                "memory_type": mem_type,
                "categories": cats or [],
                "importance": imp,
                "pinned": pin,
                # Always write role explicitly so that updating a memory
                # with a different role (e.g. user→assistant) takes effect.
                # update_metadata() uses Qdrant set_payload (patch semantics)
                # and would silently preserve the old role if omitted.
                "role": role,
            }
            # Preserve or update event_date
            # Priority: caller-provided > LLM-extracted > existing
            if effective_event_date is not None:
                metadata_update["event_date"] = effective_event_date
            elif existing_meta.get("event_date"):
                metadata_update["event_date"] = existing_meta["event_date"]
            # Recalculate TTL based on new memory_type
            ttl_meta = build_expiry_metadata(ttl_days, mem_type, self._config.memory)
            metadata_update.update(ttl_meta)

            # Preserve artifacts from existing memory
            metadata_update["artifacts"] = existing_meta.get("artifacts", [])
            metadata_update["updated_at_utc"] = datetime.now(timezone.utc).isoformat()

            # Update content + re-embed via full point replacement
            # (preserves all payload fields we set)
            self.vector.update_content(
                target_id,
                text,
                vector=vector,
                sparse_vector=self._get_sparse_vector(text),
            )
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
            # Clean up artifacts with reference checking
            try:
                self._cleanup_memory_artifacts(user_id, target_id)
            except Exception:
                logger.warning("Failed to delete artifacts for %s", target_id)

            self.vector.delete(target_id)
            return {"id": target_id, "memory": text, "event": "DELETE"}

        return None

    def _maybe_create_auto_artifact(
        self,
        *,
        store_artifact: bool,
        content: str,
        user_id: str,
        results: list[dict[str, Any]],
        threshold: int,
    ) -> dict[str, Any] | None:
        """Create an auto-artifact if the LLM recommends it and content is long enough.

        Saves the original content as an artifact and links it to all
        ADD/UPDATE memories from the extraction results.

        Args:
            store_artifact: Whether the LLM recommended storing an artifact.
                If the LLM didn't return the field, callers should default
                to True when content exceeds the threshold (safe fallback).
            content: The original content to save as artifact.
            user_id: Owner of the memories.
            results: List of result dicts from _execute_action, each with
                "id", "memory", and "event" keys.
            threshold: Minimum content length to create an artifact.

        Returns:
            Artifact info dict if created, None otherwise.
        """
        if not store_artifact:
            return None
        if len(content) <= threshold:
            return None
        if not results:
            return None

        # Only link to memories that were ADD'd or UPDATE'd
        linkable = [r for r in results if r.get("event") in ("ADD", "UPDATE")]
        if not linkable:
            return None

        try:
            meta = self.artifact.save(
                user_id=user_id,
                content=content,
                filename="content.md",
                content_type="text/markdown",
            )
            artifact_dict = meta.to_dict()

            # Link artifact to all ADD/UPDATE memories
            linked_count = 0
            for result in linkable:
                memory_id = result["id"]
                try:
                    mem = self.vector.get_by_id(memory_id)
                    current_artifacts = (
                        (mem.get("metadata") or {}).get("artifacts", []) if mem else []
                    )
                    current_artifacts.append(artifact_dict)
                    self.vector.update_metadata(
                        memory_id, {"artifacts": current_artifacts}
                    )
                    linked_count += 1
                except Exception:
                    logger.warning("Failed to link artifact to memory %s", memory_id)

            logger.info(
                "Auto-artifact created: %s (%d bytes, linked to %d memories)",
                meta.artifact_id,
                meta.size,
                linked_count,
            )

            return {
                "id": meta.artifact_id,
                "filename": meta.filename,
                "size": meta.size,
                "linked_memories": linked_count,
            }

        except Exception:
            logger.warning(
                "Auto-artifact creation failed, memories stored without artifact",
                exc_info=True,
            )
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
        event_date: str | None = None,
    ) -> dict:
        """Add memory directly without LLM inference (infer=False path).

        Stores content as-is with only an embedding call. Uses a separate
        classification LLM call if AUTO_CLASSIFY is enabled and metadata
        fields are missing.
        """
        # Validate input length (same cap as infer=True path)
        max_input = self._config.memory.max_input_length
        if len(content) > max_input:
            return {
                "error": True,
                "message": (
                    f"Input too long: {len(content)} chars "
                    f"(max {max_input}). Shorten the input or split "
                    "into multiple calls."
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
        if event_date is not None:
            metadata["event_date"] = event_date

        # Add TTL metadata
        ttl_meta = build_expiry_metadata(ttl_days, memory_type, self._config.memory)
        metadata.update(ttl_meta)

        # Auto-artifact for oversized content (infer=False path)
        max_len = self._config.memory.max_memory_length
        store_content = content
        artifact_info: dict[str, Any] | None = None

        if len(content) > max_len:
            # Save full content as artifact, truncate memory text
            try:
                art_meta = self.artifact.save(
                    user_id=user_id,
                    content=content,
                    filename="content.md",
                    content_type="text/markdown",
                )
                metadata["artifacts"] = [art_meta.to_dict()]
                artifact_info = {
                    "id": art_meta.artifact_id,
                    "filename": art_meta.filename,
                    "size": art_meta.size,
                    "linked_memories": 1,
                }
                # Truncate content for the searchable memory
                store_content = content[:max_len]
                logger.info(
                    "Auto-artifact (infer=False): %s (%d bytes), "
                    "memory truncated to %d chars",
                    art_meta.artifact_id,
                    art_meta.size,
                    max_len,
                )
            except Exception:
                logger.exception(
                    "Auto-artifact creation failed for infer=False content"
                )
                return {
                    "error": True,
                    "message": (
                        f"Content too long ({len(content)} chars, max {max_len}) "
                        "and artifact creation failed. Use save_artifact "
                        "for detailed content."
                    ),
                }

        # Embed and store
        vector = self.vector.embedding.embed(store_content)
        memory_id = self.vector.insert(
            text=store_content,
            vector=vector,
            user_id=user_id,
            agent_id=agent_id,
            metadata=metadata,
            role=role,
            sparse_vector=self._get_sparse_vector(store_content),
        )

        response: dict[str, Any] = {
            "results": [{"id": memory_id, "memory": store_content, "event": "ADD"}]
        }
        if artifact_info:
            response["artifact"] = artifact_info

        return response

    # ── Search Helpers ────────────────────────────────────────────────

    def _importance_boost(self, memories: list[dict]) -> list[dict]:
        """Apply importance-based score boost in Python post-processing.

        Used after RRF fusion where FormulaQuery is not available
        (hybrid search mode). Adds a small additive bonus based on
        importance level, then re-sorts by score.

        Formula: score = score + importance_weight * importance_value
        Where importance_weight = 1 - similarity_weight (default 0.1)

        Args:
            memories: Search results with "score" and "metadata" fields.

        Returns:
            Memories with updated scores, re-sorted by new score.
        """
        importance_weight = 1.0 - self._config.memory.search_similarity_weight
        if importance_weight <= 0 or not memories:
            return memories

        for mem in memories:
            imp_level = (mem.get("metadata") or {}).get("importance", "normal")
            imp_value = IMPORTANCE_WEIGHTS.get(imp_level, 0.4)
            mem["score"] = round(mem.get("score", 0) + importance_weight * imp_value, 4)

        memories.sort(key=lambda m: m.get("score", 0), reverse=True)
        return memories

    def _get_sparse_vector(self, text: str) -> Any | None:
        """Generate a BM25 sparse vector for hybrid search.

        Returns None on failure so callers fall back to dense-only mode.
        """
        try:
            return self._sparse.embed(text)
        except Exception:
            logger.warning("Sparse embedding failed, using dense-only", exc_info=True)
            return None

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
        query_sparse_vector: Any = None,
        date_start: str | None = None,
        date_end: str | None = None,
    ) -> list[dict]:
        """Search memories with semantic similarity, filtered and reranked.

        When hybrid search is available (fastembed installed), uses
        dense + BM25 sparse vectors with RRF fusion. Falls back to
        dense-only search with FormulaQuery importance reranking.

        Args:
            query_vector: Pre-computed dense embedding vector. If provided,
                skips the embedding API call. Used by find_memories for
                batch embedding optimization.
            query_sparse_vector: Pre-computed BM25 sparse vector. If
                provided, enables hybrid search. If None and hybrid is
                available, generated automatically from the query.
            date_start: Optional start date (YYYY-MM-DD) for temporal
                filtering. Passed to Qdrant for event_date/created_at
                range filtering.
            date_end: Optional end date (YYYY-MM-DD) for temporal filtering.
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

        # Generate sparse query vector if not provided and hybrid is available
        if query_sparse_vector is None:
            query_sparse_vector = self._get_sparse_vector(query)

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
            query_sparse_vector=query_sparse_vector,
            date_start=date_start,
            date_end=date_end,
        )

        memories = result.get("results", [])
        used_hybrid = result.get("used_hybrid", False)

        # In hybrid mode, apply importance boost in Python (FormulaQuery
        # is not used with RRF fusion). In dense-only mode, FormulaQuery
        # already handles importance reranking server-side.
        if used_hybrid:
            memories = self._importance_boost(memories)

        # Score threshold: use hybrid threshold for RRF scores (which are
        # much smaller than cosine similarity), dense threshold otherwise
        threshold = (
            self._config.memory.search_score_threshold_hybrid
            if used_hybrid
            else self._config.memory.search_score_threshold
        )
        memories = [m for m in memories if m.get("score", 0) >= threshold]

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
        date_start: str | None = None,
        date_end: str | None = None,
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
            "date_start": date_start,
            "date_end": date_end,
        }

        # Generate sparse query vector for hybrid search (shared across
        # both sub-searches to avoid redundant BM25 tokenization)
        query_sparse_vector = self._get_sparse_vector(query)
        if query_sparse_vector is not None:
            search_kwargs["query_sparse_vector"] = query_sparse_vector

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

        # Sort merged results by score
        memories.sort(key=lambda m: m.get("score", 0), reverse=True)

        # Use the actual mode that was used (either sub-search may have fallen back)
        used_hybrid = agent_result.get("used_hybrid", False) or shared_result.get(
            "used_hybrid", False
        )

        # In hybrid mode, apply importance boost in Python (FormulaQuery
        # is not used with RRF fusion). In dense-only mode, FormulaQuery
        # already handles importance reranking server-side.
        if used_hybrid:
            memories = self._importance_boost(memories)

        # Score threshold: use hybrid threshold for RRF scores
        threshold = (
            self._config.memory.search_score_threshold_hybrid
            if used_hybrid
            else self._config.memory.search_score_threshold
        )
        memories = [m for m in memories if m.get("score", 0) >= threshold]

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
        session_timezone: str | None = None,
        context: str | None = None,
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
            session_timezone: IANA timezone from X-Timezone header. Used to
                determine "today's date" for temporal query generation.
            context: Optional context hint for query generation (e.g., current
                working directory, active project). Injected into the query
                generation prompt as background information — the LLM uses it
                to generate additional relevant queries where appropriate, but
                does not limit queries exclusively to this context.

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

        # Fetch project categories for context-aware query generation
        project_cats = self._get_project_categories(user_id)

        # Compute today's date in the session timezone (for temporal queries)
        today: str | None = None
        if session_timezone:
            try:
                from zoneinfo import ZoneInfo

                today = datetime.now(ZoneInfo(session_timezone)).strftime("%Y-%m-%d")
            except (KeyError, Exception):
                logger.warning(
                    "Invalid session timezone '%s', falling back to server local",
                    session_timezone,
                )
                today = datetime.now().astimezone().strftime("%Y-%m-%d")
        else:
            today = datetime.now().astimezone().strftime("%Y-%m-%d")

        # Statistics tracking
        total_searched = 0

        # Step 1: Generate diverse search queries from the question
        messages, schema = build_query_generation_prompt(
            question,
            num_queries=num_queries,
            today=today,
            context=context,
            project_categories=project_cats or None,
        )
        raw_response = self._llm.generate(messages, json_schema=schema)
        try:
            data = parse_json_response(raw_response)
            queries = data.get("queries", [])
            if not isinstance(queries, list):
                raise ValueError("LLM returned invalid queries format")
        except (ValueError, KeyError) as e:
            raise ValueError(f"Failed to generate search queries: {e}") from e

        # Extract date range for temporal filtering (if LLM provided one)
        date_range = data.get("date_range")
        date_start: str | None = None
        date_end: str | None = None
        if isinstance(date_range, dict):
            date_start = date_range.get("start")
            date_end = date_range.get("end")
            if date_start or date_end:
                logger.info(
                    "find_memories: date range filter: %s to %s",
                    date_start,
                    date_end,
                )

        # LLM may return 0 queries when the input doesn't need memory search
        if not queries:
            logger.info(
                "find_memories: LLM returned 0 queries for: %s",
                question[:100],
            )
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
                        date_start=date_start,
                        date_end=date_end,
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
                        date_start=date_start,
                        date_end=date_end,
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
        messages, schema = build_rerank_prompt(question, rerank_candidates, today=today)
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

        Format: - [YYYY-MM-DD] [cat1, cat2] Memory text
        Date-only timestamp (no HH:MM). Categories shown if present,
        memory_type omitted (internal classification, not useful to LLM).
        """
        ts = memory.get("created_at", "")
        if ts and "T" in ts:
            ts = ts.split("T")[0]

        text = wrap_memory_item(memory.get("memory", ""))
        meta = memory.get("metadata") or {}
        cats = meta.get("categories", [])

        cat_str = f" [{', '.join(cats)}]" if cats else ""

        return f"- [{ts}]{cat_str} {text}"

    @staticmethod
    def _sort_memories_by_importance(memories: list[dict]) -> list[dict]:
        """Sort memories by importance (desc) then created_at (desc).

        Critical > high > normal > low, then most recent first within
        the same importance level.
        """
        return sorted(
            memories,
            key=lambda m: (
                IMPORTANCE_WEIGHTS.get(
                    (m.get("metadata") or {}).get("importance", "normal"), 0.4
                ),
                (m.get("metadata") or {}).get("created_at_utc", "")
                or m.get("created_at", ""),
            ),
            reverse=True,
        )

    def _classify_agent_memory(self, memory: dict) -> tuple[str, str]:
        """Classify an agent-scoped memory into section and formatted line.

        Returns (section_key, formatted_line) where section_key is one of:
        "identity", "knowledge", "instructions".
        """
        meta = memory.get("metadata") or {}
        mt = meta.get("memory_type", "fact")
        mr = meta.get("role", "user")
        text = wrap_memory_item(memory.get("memory", ""))
        line = f"- {text}"

        if mr == "assistant":
            if mt in ("preference", "fact"):
                return "identity", line
            return "knowledge", line
        return "instructions", line

    def _classify_user_memory(self, memory: dict) -> tuple[str, str]:
        """Classify a shared user memory into section and formatted line.

        Returns (section_key, formatted_line) where section_key is one of:
        "facts", "prefs", "other".
        """
        mt = (memory.get("metadata") or {}).get("memory_type", "fact")
        text = wrap_memory_item(memory.get("memory", ""))
        line = f"- {text}"

        if mt == "fact":
            return "facts", line
        if mt == "preference":
            return "prefs", line
        return "other", line

    def get_core_memories(
        self,
        *,
        user_id: str,
        agent_id: str | None = None,
        recent_days: int | None = None,
    ) -> CoreMemoriesResult:
        """Assemble core context for conversation start.

        Returns a CoreMemoriesResult with:
        - text: Structured markdown with pinned + top-N + recent memories
        - memory_ids: Set of all memory IDs included (for dedup vs search)

        Sections:
        1. Pinned agent memories (identity, knowledge) — if agent_id provided
        2. Pinned user memories (facts, preferences) — shared across agents
        3. Top-N non-pinned memories by importance (configurable)
        4. Recent context memories from the last N days (dual-scope)

        Pinned memories always appear first in each section, followed by
        top-N non-pinned memories sorted by importance. Total output is
        capped at MAX_CORE_CONTEXT_LENGTH with graceful truncation.
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
        top_n = self._config.memory.core_top_memories
        min_importance = self._config.memory.core_min_importance

        sections: list[str] = []

        # Collect IDs of all memories included in main sections (for dedup)
        included_ids: set[str] = set()

        # ── 1. Fetch all pinned memories in a single query ────────────
        all_pinned = self.vector.get_pinned_memories(user_id=user_id)

        # Split into agent-scoped and shared user memories
        agent_pinned = (
            [m for m in all_pinned if m.get("agent_id") == agent_id] if agent_id else []
        )
        user_pinned = [m for m in all_pinned if not m.get("agent_id")]

        # Track pinned IDs
        for m in agent_pinned + user_pinned:
            if m.get("id"):
                included_ids.add(m["id"])

        # ── 2+3. Fetch top-N non-pinned memories (parallel) ─────────
        # Agent and user top-N queries are independent — run in parallel.
        imp_levels = importance_levels_at_or_above(min_importance) if top_n > 0 else []
        agent_non_pinned_raw: dict = {"results": []}
        user_non_pinned_raw: dict = {"results": []}

        if top_n > 0:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {}
                if agent_id:
                    futures["agent"] = pool.submit(
                        self.vector.get_all,
                        user_id=user_id,
                        agent_id=agent_id,
                        filters={"pinned": False, "importance": imp_levels},
                        limit=top_n * 3,
                    )
                futures["user"] = pool.submit(
                    self.vector.get_all,
                    user_id=user_id,
                    shared_only=True,
                    filters={"pinned": False, "importance": imp_levels},
                    limit=top_n * 3,
                )
                if "agent" in futures:
                    agent_non_pinned_raw = futures["agent"].result()
                user_non_pinned_raw = futures["user"].result()

        # ── 2. Agent sections (pinned + top-N non-pinned) ────────────
        if agent_id:
            # Sort pinned agent memories by importance
            agent_pinned_sorted = self._sort_memories_by_importance(agent_pinned)

            # Classify pinned agent memories into buckets
            agent_sections: dict[str, list[str]] = {
                "identity": [],
                "knowledge": [],
                "instructions": [],
            }
            for m in agent_pinned_sorted:
                key, line = self._classify_agent_memory(m)
                agent_sections[key].append(line)

            # Process top-N non-pinned agent memories
            if top_n > 0:
                agent_non_pinned = [
                    m
                    for m in agent_non_pinned_raw.get("results", [])
                    if m.get("id") not in included_ids and not should_exclude(m)
                ]
                agent_non_pinned = self._sort_memories_by_importance(agent_non_pinned)[
                    :top_n
                ]

                for m in agent_non_pinned:
                    if m.get("id"):
                        included_ids.add(m["id"])
                    key, line = self._classify_agent_memory(m)
                    agent_sections[key].append(line)

            if agent_sections["identity"]:
                sections.append(
                    "## Agent Identity\n" + "\n".join(agent_sections["identity"])
                )
            if agent_sections["knowledge"]:
                sections.append(
                    "## Agent Knowledge\n" + "\n".join(agent_sections["knowledge"])
                )
            if agent_sections["instructions"]:
                sections.append(
                    "## Agent Instructions\n"
                    + "\n".join(agent_sections["instructions"])
                )

        # ── 3. User sections (pinned + top-N non-pinned) ─────────────
        # Sort pinned user memories by importance
        user_pinned_sorted = self._sort_memories_by_importance(user_pinned)

        user_sections: dict[str, list[str]] = {
            "facts": [],
            "prefs": [],
            "other": [],
        }
        for m in user_pinned_sorted:
            key, line = self._classify_user_memory(m)
            user_sections[key].append(line)

        # Process top-N non-pinned user memories
        if top_n > 0:
            user_non_pinned = [
                m
                for m in user_non_pinned_raw.get("results", [])
                if m.get("id") not in included_ids and not should_exclude(m)
            ]
            user_non_pinned = self._sort_memories_by_importance(user_non_pinned)[:top_n]

            for m in user_non_pinned:
                if m.get("id"):
                    included_ids.add(m["id"])
                key, line = self._classify_user_memory(m)
                user_sections[key].append(line)

        if user_sections["facts"]:
            sections.append("## User Facts\n" + "\n".join(user_sections["facts"]))
        if user_sections["prefs"]:
            sections.append("## User Preferences\n" + "\n".join(user_sections["prefs"]))
        if user_sections["other"]:
            sections.append(
                "## Other User Memories\n" + "\n".join(user_sections["other"])
            )

        # ── 4. Recent context memories (parallel, dual-scope) ─────────
        since = datetime.now(timezone.utc) - timedelta(days=recent_days)
        limit_user = self._config.memory.recent_limit_user
        limit_agent = self._config.memory.recent_limit_agent
        recent_imp_levels = importance_levels_at_or_above(
            self._config.memory.core_recent_min_importance
        )

        # Fetch user + agent recent memories in parallel
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_user_recent = pool.submit(
                self.vector.get_recent_memories,
                user_id=user_id,
                agent_id=None,
                since=since,
                limit=limit_user,
                memory_types=self.RECENT_MEMORY_TYPES,
                importance_levels=recent_imp_levels,
            )
            f_agent_recent = None
            if agent_id:
                f_agent_recent = pool.submit(
                    self.vector.get_recent_memories,
                    user_id=user_id,
                    agent_id=agent_id,
                    since=since,
                    limit=limit_agent,
                    memory_types=self.RECENT_MEMORY_TYPES,
                    importance_levels=recent_imp_levels,
                )
            user_recent_raw = f_user_recent.result()
            agent_recent_raw = f_agent_recent.result() if f_agent_recent else []

        user_recent = [
            m
            for m in user_recent_raw
            if not should_exclude(m) and m.get("id") not in included_ids
        ]
        agent_recent: list[dict] = [
            m
            for m in agent_recent_raw
            if not should_exclude(m) and m.get("id") not in included_ids
        ]

        # Sort recent by importance (desc) then date (desc) so truncation
        # removes least important entries first
        user_recent = self._sort_memories_by_importance(user_recent)
        agent_recent = self._sort_memories_by_importance(agent_recent)

        # Build recent context lines paired with memory IDs so that
        # truncation can remove IDs for entries that get dropped.
        recent_user_pairs: list[tuple[str, str | None]] = [
            (self._format_recent_memory_line(m), m.get("id")) for m in user_recent
        ]
        recent_agent_pairs: list[tuple[str, str | None]] = [
            (self._format_recent_memory_line(m), m.get("id")) for m in agent_recent
        ]

        recent_user_lines = [line for line, _ in recent_user_pairs]
        recent_agent_lines = [line for line, _ in recent_agent_pairs]

        if not sections and not recent_user_lines and not recent_agent_lines:
            result = CoreMemoriesResult(text="No core memories found.")
            self._core_cache.set(cache_key, result)
            return result

        # ── 5. Assembly with graceful truncation ──────────────────────
        # Prepend security preamble
        preamble = CORE_MEMORIES_PREAMBLE
        main_text = "\n\n".join(sections) if sections else ""

        # Build recent context section
        recent_section = self._build_recent_section(
            recent_user_lines, recent_agent_lines
        )

        # Assemble full output
        parts = [p for p in [preamble, main_text, recent_section] if p]
        output = "\n\n".join(parts)

        # Graceful truncation: trim recent entries one by one from the end.
        # Also pop from the paired lists so we know which IDs survived.
        if len(output) > max_len and (recent_user_lines or recent_agent_lines):
            # Try trimming agent lines first, then user lines
            while len(output) > max_len and recent_agent_lines:
                recent_agent_lines.pop()
                recent_agent_pairs.pop()
                recent_section = self._build_recent_section(
                    recent_user_lines, recent_agent_lines
                )
                parts = [p for p in [preamble, main_text, recent_section] if p]
                output = "\n\n".join(parts)

            while len(output) > max_len and recent_user_lines:
                recent_user_lines.pop()
                recent_user_pairs.pop()
                recent_section = self._build_recent_section(
                    recent_user_lines, recent_agent_lines
                )
                parts = [p for p in [preamble, main_text, recent_section] if p]
                output = "\n\n".join(parts)

        # Now collect IDs only for recent memories that survived truncation
        for _, mid in recent_user_pairs + recent_agent_pairs:
            if mid:
                included_ids.add(mid)

        # If still too long after removing all recent context, hard truncate
        if len(output) > max_len:
            output = output[: max_len - 20] + "\n\n[...truncated]"

        result = CoreMemoriesResult(text=output, memory_ids=included_ids)
        self._core_cache.set(cache_key, result)
        return result

    @staticmethod
    def _build_recent_section(user_lines: list[str], agent_lines: list[str]) -> str:
        """Build the recent context section from pre-formatted lines.

        Returns empty string if no lines remain.
        """
        if not user_lines and not agent_lines:
            return ""
        parts = ["## Recent Context"]
        if user_lines:
            parts.append("\n### User Activity")
            parts.extend(user_lines)
        if agent_lines:
            parts.append("\n### Agent Activity")
            parts.extend(agent_lines)
        return "\n".join(parts)

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
        sort: str | None = None,
    ) -> list[dict]:
        """List memories with optional filtering and sorting.

        By default, expired and decayed memories are excluded. Set
        include_decayed=True to include them (useful for browsing
        historical memories).

        Args:
            sort: Optional sort order. "newest" for descending by
                created_at, "oldest" for ascending. None for default
                (arbitrary Qdrant scroll order).
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

        order_by = _sort_to_order_by(sort)

        result = self.vector.get_all(
            user_id=user_id,
            agent_id=agent_id,
            filters=filters if filters else None,
            limit=limit * 2,  # Fetch extra for post-filtering
            order_by=order_by,
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
        sort: str | None = None,
    ) -> list[dict]:
        """List both agent-scoped and shared memories, merge and deduplicate.

        Used when a session has an agent_id but the LLM doesn't pass one
        (meaning: list everything I can see).

        Args:
            sort: Optional sort order. "newest" for descending by
                created_at, "oldest" for ascending. None for default.
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
        order_by = _sort_to_order_by(sort)

        # Fetch 1: agent-scoped memories
        agent_result = self.vector.get_all(
            user_id=user_id,
            agent_id=session_agent_id,
            filters=filters if filters else None,
            limit=fetch_limit,
            order_by=order_by,
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
            order_by=order_by,
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

        # When server-side ordering was used, re-sort after merge to
        # maintain correct order across the two fetches.
        if sort == "newest":
            memories.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        elif sort == "oldest":
            memories.sort(key=lambda m: m.get("created_at", ""))

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
        event_date: str | None = ...,  # type: ignore[assignment]
        agent_id: str | None = ...,  # type: ignore[assignment]
    ) -> dict:
        """Update a memory's content and/or metadata.

        Content updates re-embed and replace the vector store point.
        Metadata updates use direct payload updates to avoid losing
        custom fields.

        ttl_days: Set a new TTL (recalculates expires_at from now). Pass 0
                  or None to make permanent. Uses sentinel default (...) to
                  distinguish "not provided" from "explicitly set to None".
        event_date: Set a new event date (ISO 8601). Pass None to clear.
                    Uses sentinel default (...) to distinguish "not provided"
                    from "explicitly set to None".
        agent_id: Set or clear the agent_id. Pass None to clear. Uses
                  sentinel default (...) to distinguish "not provided"
                  from "explicitly set to None". Caller is responsible
                  for authorization checks.
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

        # agent_id update: set or clear
        if agent_id is not ...:
            if agent_id is not None:
                agent_id = _validate_id(agent_id, "agent_id")
            metadata_updates["agent_id"] = agent_id

        # event_date update: parse and normalize, or clear if None
        if event_date is not ...:
            if event_date is not None:
                default_tz = self._config.memory.default_timezone
                metadata_updates["event_date"] = _parse_event_date(
                    event_date, default_tz
                )
            else:
                metadata_updates["event_date"] = None

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
            self.vector.update_content(
                memory_id,
                content,
                sparse_vector=self._get_sparse_vector(content),
            )

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
        """Delete a memory and clean up orphaned artifacts."""
        user_id = _validate_id(user_id, "user_id")

        # Clean up artifacts with reference checking
        try:
            self._cleanup_memory_artifacts(user_id, memory_id)
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

        # When deleting all memories, we can bulk-delete all artifacts
        # for the user since no references will remain.
        # For agent-scoped deletes, we need per-artifact reference checking.
        if agent_id is None:
            # Deleting ALL user memories — safe to delete all artifacts
            try:
                self.artifact.delete_all_for_user(user_id=user_id)
            except Exception as e:
                logger.warning(
                    "Failed to bulk-delete artifacts for user %s: %s", user_id, e
                )
        else:
            # Deleting only agent-scoped memories — need reference checking
            try:
                result = self.vector.get_all(
                    user_id=user_id, agent_id=agent_id, limit=1000
                )
                for mem in result.get("results", []):
                    mem_id = mem.get("id")
                    if mem_id and (mem.get("metadata") or {}).get("artifacts"):
                        try:
                            self._cleanup_memory_artifacts(user_id, mem_id)
                        except Exception as e:
                            logger.warning(
                                "Failed to delete artifacts for %s: %s", mem_id, e
                            )
            except Exception as e:
                logger.warning(
                    "Failed to enumerate memories for artifact cleanup: %s", e
                )

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
        """Delete an artifact and update the memory's metadata.

        Checks if other memories still reference this artifact before
        deleting the actual content. If other references exist, only
        removes the reference from this memory's metadata.
        """
        user_id = _validate_id(user_id, "user_id")
        artifacts_meta = self._get_artifacts_meta(user_id, memory_id)

        # Update memory metadata to remove the artifact reference
        updated = [a for a in artifacts_meta if a.get("id") != artifact_id]
        self.vector.update_metadata(memory_id, {"artifacts": updated})

        # Check if other memories still reference this artifact
        if not self._artifact_has_other_references(
            artifact_id, exclude_memory_id=memory_id
        ):
            # No other references — safe to delete the actual content
            try:
                self.artifact.delete(
                    user_id=user_id,
                    artifact_id=artifact_id,
                    artifacts_meta=artifacts_meta,
                )
            except Exception as e:
                logger.warning(
                    "Failed to delete artifact content %s: %s", artifact_id, e
                )

        return {"status": "deleted", "artifact_id": artifact_id, "memory_id": memory_id}

    # ── Private Helpers ───────────────────────────────────────────────

    def _artifact_has_other_references(
        self,
        artifact_id: str,
        *,
        exclude_memory_id: str,
    ) -> bool:
        """Check if any other memory references this artifact.

        Uses Qdrant nested field filtering on artifacts[].id to find
        memories that reference the artifact, excluding the memory being
        deleted.

        Returns True if at least one other memory references the artifact.
        """
        try:
            return self.vector.artifact_has_references(
                artifact_id=artifact_id,
                exclude_memory_id=exclude_memory_id,
            )
        except Exception:
            logger.warning(
                "Failed to check artifact references for %s, assuming referenced",
                artifact_id,
            )
            # Safe fallback: assume referenced (don't delete)
            return True

    def _cleanup_memory_artifacts(self, user_id: str, memory_id: str) -> None:
        """Clean up artifacts when a memory is deleted.

        For each artifact referenced by the memory, checks if other memories
        still reference it. If not, deletes the artifact content from storage.
        """
        mem = self.vector.get_by_id(memory_id)
        if mem is None:
            return

        artifacts = (mem.get("metadata") or {}).get("artifacts", [])
        if not artifacts:
            return

        for art in artifacts:
            art_id = art.get("id")
            if not art_id:
                continue
            if not self._artifact_has_other_references(
                art_id, exclude_memory_id=memory_id
            ):
                # No other references — delete the artifact content
                try:
                    self.artifact.delete_by_id(user_id=user_id, artifact_id=art_id)
                except Exception as e:
                    logger.warning(
                        "Failed to delete orphaned artifact %s: %s", art_id, e
                    )

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

    def _get_project_categories(self, user_id: str) -> list[str]:
        """Get the list of project:* categories for this user.

        Reuses the category cache from _get_available_categories.
        Returns only categories with the "project:" prefix that exist
        in the user's memories (i.e., have at least one memory tagged).
        """
        all_cats = self._get_available_categories(user_id)
        return [c for c in all_cats if c.startswith("project:")]

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
