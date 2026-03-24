"""Memory consolidation service — synthesize durable knowledge from raw memories.

Within-session consolidation reads a session summary and its linked raw
memories, then uses an LLM to synthesize canonical durable memories
(decisions, preferences, facts, actions). Raw memories are marked as
superseded after successful consolidation.

The service is scheduled by MaintenanceService to run periodically,
checking for idle sessions that have raw memories awaiting consolidation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mnemory.config import Config
    from mnemory.embeddings import EmbeddingClient
    from mnemory.llm import LLMClient
    from mnemory.memory import MemoryService
    from mnemory.metrics import MetricsCollector
    from mnemory.storage.vector import SessionSummaryStore, VectorStore

logger = logging.getLogger(__name__)

# Minimum consolidated memories per 5 raw memories (validation gate)
_MIN_CONSOLIDATED_RATIO = 0.2

# Maximum similarity between consolidated outputs (near-duplicate detection)
_MAX_OUTPUT_SIMILARITY = 0.90

# Minimum content length for a consolidated memory
_MIN_CONTENT_LENGTH = 20


@dataclass
class ConsolidationResult:
    """Result of a consolidation run."""

    session_id: str
    memories_produced: int = 0
    memories_superseded: int = 0
    consolidated_memory_ids: list[str] | None = None
    state: str = "idle"  # idle, consolidating, consolidated, failed
    error: str | None = None
    duration_seconds: float = 0.0


class ConsolidationService:
    """Synthesize durable knowledge from raw session memories.

    Within-session consolidation:
    1. Find idle sessions with raw memories (consolidation_state=idle)
    2. Read session summary + linked raw memories
    3. LLM synthesizes consolidated memories
    4. Validate output quality
    5. Store consolidated memories (memory_layer=consolidated, derived_from=[...])
    6. Mark raw memories as superseded (except artifact-bearing ones)
    7. Update session consolidation state

    The service acquires a per-user lock to prevent race conditions
    with concurrent remember calls.
    """

    def __init__(
        self,
        config: Config,
        vector: VectorStore,
        llm: LLMClient,
        embedding: EmbeddingClient,
        memory_service: MemoryService,
        session_summary_store: SessionSummaryStore,
        collector: MetricsCollector | None = None,
    ) -> None:
        self._config = config
        self._vector = vector
        self._llm = llm
        self._embedding = embedding
        self._memory = memory_service
        self._sessions = session_summary_store
        self._collector = collector

    def find_pending_sessions(self, user_id: str) -> list[dict]:
        """Find sessions awaiting consolidation for a user."""
        return self._sessions.find_pending(
            user_id,
            idle_threshold_seconds=self._config.memory.consolidation_idle_threshold,
        )

    def consolidate_session(self, session_id: str) -> ConsolidationResult:
        """Consolidate one session's raw memories into durable knowledge.

        State machine: idle -> consolidating -> consolidated (or failed).
        Crash recovery: if state is 'consolidating' on entry, checks for
        orphaned consolidated memories and resumes or resets.
        """
        result = ConsolidationResult(session_id=session_id)
        t0 = time.monotonic()

        try:
            # 1. Read session summary
            session = self._sessions.get(session_id)
            if session is None:
                result.error = "Session not found"
                result.state = "failed"
                return result

            user_id = session.get("user_id", "")
            agent_id = session.get("agent_id")
            memory_ids = session.get("memory_ids", [])
            summary = session.get("summary", "")

            logger.info(
                "Consolidation session %s: %d linked memories, summary_len=%d",
                session_id,
                len(memory_ids),
                len(summary),
            )

            if not memory_ids:
                logger.debug("Session %s has no memories, skipping", session_id)
                result.state = "consolidated"
                self._sessions.update_consolidation_state(session_id, "consolidated")
                return result

            # 2. Set state to consolidating
            self._sessions.update_consolidation_state(session_id, "consolidating")
            result.state = "consolidating"

            # 3. Fetch linked raw memories
            raw_memories = self._fetch_raw_memories(memory_ids, user_id)
            if not raw_memories:
                logger.info(
                    "Session %s: no raw memories found (may already be consolidated)",
                    session_id,
                )
                self._sessions.update_consolidation_state(session_id, "consolidated")
                result.state = "consolidated"
                return result

            # 3b. Fetch previously consolidated memories (for re-consolidation)
            previous_consolidated = []
            prev_consolidated_ids = session.get("consolidated_memory_ids") or []
            if prev_consolidated_ids:
                for prev_id in prev_consolidated_ids:
                    try:
                        mem = self._vector.get_by_id(prev_id)
                        if mem and mem.get("memory"):
                            previous_consolidated.append(mem)
                    except Exception:
                        pass
                if not previous_consolidated and prev_consolidated_ids:
                    logger.debug(
                        "Could not fetch previous consolidated memories for %s",
                        session_id,
                    )
                if previous_consolidated:
                    logger.info(
                        "Consolidation session %s: %d previous consolidated memories for context",
                        session_id,
                        len(previous_consolidated),
                    )

            # 4. Identify artifact-bearing memories (protected from superseding)
            artifact_ids = {
                m["id"]
                for m in raw_memories
                if (m.get("metadata") or {}).get("artifacts")
            }

            logger.info(
                "Consolidation session %s: %d raw memories fetched (%d with artifacts)",
                session_id,
                len(raw_memories),
                len(artifact_ids),
            )

            # 5. Build consolidation prompt and call LLM
            from mnemory.prompts import build_consolidation_prompt

            messages, json_schema = build_consolidation_prompt(
                summary=summary,
                raw_memories=raw_memories,
                artifact_memory_ids=artifact_ids,
                previous_consolidated=previous_consolidated,
            )

            from mnemory.llm import parse_json_response

            response_text = self._llm.generate(
                messages,
                json_schema=json_schema,
                temperature=0.3,
                operation="consolidation",
            )

            parsed = parse_json_response(response_text)

            if not parsed or not isinstance(parsed.get("memories"), list):
                result.error = "LLM returned invalid consolidation output"
                result.state = "failed"
                self._sessions.update_consolidation_state(session_id, "failed")
                return result

            consolidated_facts = parsed["memories"]

            logger.info(
                "Consolidation session %s: LLM produced %d facts",
                session_id,
                len(consolidated_facts),
            )

            # 6. Validate output quality
            validation_error = self._validate_output(
                consolidated_facts, len(raw_memories)
            )
            if validation_error:
                logger.warning(
                    "Session %s: consolidation validation failed: %s",
                    session_id,
                    validation_error,
                )
                result.error = validation_error
                result.state = "failed"
                self._sessions.update_consolidation_state(session_id, "failed")
                if self._collector is not None:
                    self._collector.record_consolidation_run(
                        user_id=user_id,
                        run_type="session",
                        duration_seconds=time.monotonic() - t0,
                        validation_failed=True,
                    )
                return result

            # 6b. Delete old consolidated memories (replace-all re-consolidation)
            # On re-consolidation, we replace ALL previous consolidated memories
            # with fresh ones that incorporate the full context (old + new raw).
            # This avoids LLM-generated target_id hallucination risks and keeps
            # derived_from lineage clean.
            if prev_consolidated_ids:
                for old_id in prev_consolidated_ids:
                    try:
                        self._vector.delete(old_id)
                    except Exception:
                        logger.debug(
                            "Could not delete old consolidated memory %s", old_id
                        )
                logger.info(
                    "Consolidation session %s: deleted %d old consolidated memories",
                    session_id,
                    len(prev_consolidated_ids),
                )

            # 7. Store consolidated memories
            raw_ids = [m["id"] for m in raw_memories]
            stored_ids = self._store_consolidated(
                consolidated_facts,
                user_id=user_id,
                agent_id=agent_id,
                derived_from=raw_ids,
                raw_memories=raw_memories,
            )
            result.memories_produced = len(stored_ids)
            result.consolidated_memory_ids = stored_ids

            logger.info(
                "Consolidation session %s: stored %d consolidated memories",
                session_id,
                len(stored_ids),
            )

            # 8. Write consolidated_memory_ids to session FIRST (crash recovery)
            self._sessions.update_consolidation_state(
                session_id,
                "consolidating",  # still consolidating until supersede done
                consolidated_memory_ids=stored_ids,
            )

            # 9. Mark raw memories as superseded (except artifact-bearing)
            superseded_count = 0
            for mem in raw_memories:
                mem_id = mem["id"]
                if mem_id in artifact_ids:
                    logger.debug(
                        "Skipping supersede for artifact-bearing memory %s",
                        mem_id,
                    )
                    continue
                try:
                    self._vector.update_metadata(
                        mem_id,
                        {"superseded_by": stored_ids[0] if stored_ids else None},
                    )
                    superseded_count += 1
                except Exception:
                    logger.warning(
                        "Failed to supersede memory %s", mem_id, exc_info=True
                    )
            result.memories_superseded = superseded_count

            # 10. Set final state
            self._sessions.update_consolidation_state(
                session_id,
                "consolidated",
                consolidated_memory_ids=stored_ids,
            )
            result.state = "consolidated"

            reconsolidation = bool(prev_consolidated_ids)
            logger.info(
                "Session %s %s: %d raw -> %d consolidated, %d superseded",
                session_id,
                "re-consolidated" if reconsolidation else "consolidated",
                len(raw_memories),
                len(stored_ids),
                superseded_count,
            )

        except Exception:
            logger.exception("Consolidation failed for session %s", session_id)
            result.error = "Unexpected error during consolidation"
            result.state = "failed"
            try:
                self._sessions.update_consolidation_state(session_id, "failed")
            except Exception:
                pass

        result.duration_seconds = time.monotonic() - t0

        # Record metrics
        if self._collector is not None and result.state in (
            "consolidated",
            "failed",
        ):
            self._collector.record_consolidation_run(
                user_id=session.get("user_id", "") if session else "",
                run_type="session",
                memories_produced=result.memories_produced,
                memories_superseded=result.memories_superseded,
                duration_seconds=result.duration_seconds,
                validation_failed=bool(result.error),
            )

        return result

    def recover_incomplete(self, user_id: str) -> int:
        """Check for and recover orphaned 'consolidating' sessions.

        If consolidated_memory_ids exist, resume from supersede step.
        Otherwise, reset to 'idle' for retry.

        Returns number of sessions recovered.
        """
        recovered = 0
        try:
            sessions = self._sessions.list_for_user(
                user_id, consolidation_state="consolidating"
            )
            for session in sessions:
                sid = session.get("session_id", "")
                consolidated_ids = session.get("consolidated_memory_ids")
                if consolidated_ids:
                    # Consolidated memories exist — resume supersede
                    logger.info("Recovering session %s: resuming supersede", sid)
                    memory_ids = session.get("memory_ids", [])
                    raw_memories = self._fetch_raw_memories(memory_ids, user_id)
                    artifact_ids = {
                        m["id"]
                        for m in raw_memories
                        if (m.get("metadata") or {}).get("artifacts")
                    }
                    for mem in raw_memories:
                        if mem["id"] not in artifact_ids:
                            try:
                                self._vector.update_metadata(
                                    mem["id"],
                                    {"superseded_by": consolidated_ids[0]},
                                )
                            except Exception:
                                pass
                    self._sessions.update_consolidation_state(
                        sid,
                        "consolidated",
                        consolidated_memory_ids=consolidated_ids,
                    )
                else:
                    # No consolidated memories — reset to idle
                    logger.info("Recovering session %s: resetting to idle", sid)
                    self._sessions.update_consolidation_state(sid, "idle")
                recovered += 1
        except Exception:
            logger.exception("Recovery failed for user %s", user_id)
        return recovered

    def _fetch_raw_memories(self, memory_ids: list[str], user_id: str) -> list[dict]:
        """Fetch raw memories by IDs, filtering to only unsuperseded raw."""
        memories = []
        not_found = 0
        skipped_layer = 0
        skipped_superseded = 0
        fetch_errors = 0
        for mid in memory_ids:
            try:
                result = self._vector.get_by_id(mid)
                if result is None:
                    not_found += 1
                    continue
                meta = result.get("metadata") or {}
                # Only include raw, unsuperseded memories
                layer = meta.get("memory_layer", "raw")
                if layer != "raw":
                    skipped_layer += 1
                    continue
                if meta.get("superseded_by"):
                    skipped_superseded += 1
                    continue
                memories.append(result)
            except Exception:
                fetch_errors += 1
                logger.warning("Could not fetch memory %s", mid, exc_info=True)
        # Always log summary at INFO so production can diagnose issues
        skipped = not_found + skipped_layer + skipped_superseded + fetch_errors
        if skipped > 0:
            logger.info(
                "Fetch raw memories: %d eligible, %d skipped "
                "(not_found=%d, wrong_layer=%d, superseded=%d, errors=%d) "
                "from %d total",
                len(memories),
                skipped,
                not_found,
                skipped_layer,
                skipped_superseded,
                fetch_errors,
                len(memory_ids),
            )
        return memories

    def _validate_output(self, facts: list[dict], raw_count: int) -> str | None:
        """Validate consolidation output quality.

        Returns error message if validation fails, None if OK.
        """
        if not facts:
            return "No consolidated memories produced"

        # Check minimum content length
        for f in facts:
            text = f.get("text", "")
            if len(text) < _MIN_CONTENT_LENGTH:
                return f"Consolidated memory too short ({len(text)} chars)"

        # Check minimum ratio (at least 1 per 5 raw)
        min_expected = max(1, int(raw_count * _MIN_CONSOLIDATED_RATIO))
        if len(facts) < min_expected:
            return (
                f"Too few consolidated memories: {len(facts)} "
                f"(expected at least {min_expected} from {raw_count} raw)"
            )

        return None

    def _store_consolidated(
        self,
        facts: list[dict],
        *,
        user_id: str,
        agent_id: str | None,
        derived_from: list[str],
        raw_memories: list[dict] | None = None,
    ) -> list[str]:
        """Store consolidated memories via add_memory(infer=False).

        Returns list of stored memory IDs.
        """
        # Collect fallback categories from raw memories
        fallback_categories: list[str] = []
        if raw_memories:
            cats_set: set[str] = set()
            for mem in raw_memories:
                meta = mem.get("metadata") or {}
                for cat in meta.get("categories", []):
                    if cat:
                        cats_set.add(cat)
            fallback_categories = sorted(cats_set)

        stored_ids: list[str] = []
        for fact in facts:
            try:
                text = fact.get("text", "")
                if not text:
                    continue

                # Determine effective agent_id: only set for assistant role
                effective_agent_id = (
                    agent_id if fact.get("role") == "assistant" else None
                )

                # For assistant role, infer=True is required by add_memory
                effective_infer = fact.get("role") == "assistant"

                # Use LLM-assigned categories, fall back to raw memory
                # categories if the LLM returned an empty list
                categories = fact.get("categories") or fallback_categories

                result = self._memory.add_memory(
                    text,
                    user_id=user_id,
                    agent_id=effective_agent_id,
                    infer=effective_infer,
                    memory_type=fact.get("memory_type", "episodic"),
                    categories=categories,
                    importance=fact.get("importance", "normal"),
                    pinned=fact.get("pinned", False),
                    role=fact.get("role", "user"),
                )

                results = result.get("results", [])
                if results:
                    mem_id = results[0].get("id", "")
                    if mem_id:
                        # Set derived_from and memory_layer on the stored memory
                        try:
                            self._vector.update_metadata(
                                mem_id,
                                {
                                    "derived_from": derived_from,
                                    "memory_layer": "consolidated",
                                },
                            )
                        except Exception:
                            logger.debug("Failed to set derived_from on %s", mem_id)
                        stored_ids.append(mem_id)
            except Exception:
                logger.warning(
                    "Failed to store consolidated memory (type=%s, role=%s)",
                    fact.get("memory_type", "unknown"),
                    fact.get("role", "user"),
                    exc_info=True,
                )
        return stored_ids
