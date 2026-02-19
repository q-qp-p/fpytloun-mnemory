"""Stage 1: Ingest — Feed LoCoMo conversations into mnemory.

For each conversation, creates a unique user_id and ingests all turns
from all sessions using MemoryService.

Two modes:
- infer=True (default): Calls add_memory per turn with LLM extraction,
  classification, and deduplication. Tests the full mnemory pipeline.
- infer=False (--no-infer): Batch-embeds all turns and inserts directly
  into the vector store. Fast (~seconds per conversation) but no
  extraction or dedup.
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from benchmarks.locomo.config import BenchmarkConfig
from benchmarks.locomo.dataset import Conversation

logger = logging.getLogger(__name__)

# Batch size for embedding API calls (OpenAI supports up to ~2048)
EMBED_BATCH_SIZE = 100

# Regex for LoCoMo date format: "1:56 pm on 8 May, 2023"
_LOCOMO_DATE_RE = re.compile(
    r"(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+),?\s+(\d{4})",
    re.IGNORECASE,
)

_MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _parse_locomo_date(date_str: str) -> str | None:
    """Parse LoCoMo's human-readable date to ISO 8601.

    Input format: "1:56 pm on 8 May, 2023"
    Output format: "2023-05-08T13:56:00" (naive, no timezone)

    Returns None if parsing fails.
    """
    if not date_str:
        return None
    m = _LOCOMO_DATE_RE.match(date_str.strip())
    if not m:
        logger.warning("Could not parse LoCoMo date: %s", date_str)
        return None

    hour = int(m.group(1))
    minute = int(m.group(2))
    ampm = m.group(3).lower()
    day = int(m.group(4))
    month_name = m.group(5).lower()
    year = int(m.group(6))

    month = _MONTH_MAP.get(month_name)
    if month is None:
        logger.warning("Unknown month '%s' in date: %s", month_name, date_str)
        return None

    # Convert 12-hour to 24-hour
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0

    try:
        dt = datetime(year, month, day, hour, minute)
        return dt.isoformat()
    except ValueError:
        logger.warning("Invalid date components in: %s", date_str)
        return None


@dataclass
class IngestResult:
    """Result of ingesting a single conversation."""

    conversation_index: int
    user_id: str
    sessions_processed: int
    turns_processed: int
    memories_created: int
    errors: int
    elapsed_seconds: float


@dataclass
class IngestState:
    """Complete state after the ingest stage."""

    results: list[IngestResult] = field(default_factory=list)
    total_turns: int = 0
    total_memories: int = 0
    total_errors: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [
                {
                    "conversation_index": r.conversation_index,
                    "user_id": r.user_id,
                    "sessions_processed": r.sessions_processed,
                    "turns_processed": r.turns_processed,
                    "memories_created": r.memories_created,
                    "errors": r.errors,
                    "elapsed_seconds": r.elapsed_seconds,
                }
                for r in self.results
            ],
            "total_turns": self.total_turns,
            "total_memories": self.total_memories,
            "total_errors": self.total_errors,
            "elapsed_seconds": self.elapsed_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IngestState:
        state = cls(
            total_turns=data.get("total_turns", 0),
            total_memories=data.get("total_memories", 0),
            total_errors=data.get("total_errors", 0),
            elapsed_seconds=data.get("elapsed_seconds", 0.0),
        )
        for r in data.get("results", []):
            state.results.append(
                IngestResult(
                    conversation_index=r["conversation_index"],
                    user_id=r["user_id"],
                    sessions_processed=r["sessions_processed"],
                    turns_processed=r["turns_processed"],
                    memories_created=r["memories_created"],
                    errors=r["errors"],
                    elapsed_seconds=r["elapsed_seconds"],
                )
            )
        return state

    def get_user_id(self, conversation_index: int) -> str | None:
        """Look up the user_id for a conversation."""
        for r in self.results:
            if r.conversation_index == conversation_index:
                return r.user_id
        return None


def make_user_id(conversation_index: int) -> str:
    """Generate a deterministic user_id for a conversation."""
    return f"locomo_{conversation_index}"


def _ingest_batch(
    conversation: Conversation,
    memory_service: Any,
    config: BenchmarkConfig,
) -> IngestResult:
    """Batch-ingest all turns for a conversation (no-infer mode).

    1. Collects all turn texts
    2. Batch-embeds in chunks using embed_batch()
    3. Batch-inserts into Qdrant using insert_batch()

    Much faster than per-turn add_memory — reduces API calls from N to N/100.
    """
    from mnemory.ttl import build_expiry_metadata

    user_id = make_user_id(conversation.index)
    start = time.time()

    # Collect all turns with their session dates (respecting max_turns limit)
    all_turn_data: list[tuple] = []  # (turn, session_event_date)
    for session in conversation.sessions:
        session_event_date = _parse_locomo_date(session.date_time)
        for turn in session.turns:
            all_turn_data.append((turn, session_event_date))
    if config.max_turns > 0:
        all_turn_data = all_turn_data[: config.max_turns]
    texts = [turn.format_for_memory() for turn, _ in all_turn_data]
    event_dates = [ed for _, ed in all_turn_data]
    total = len(texts)

    logger.info(
        "Batch-ingesting conversation %d (%s & %s): %d turns",
        conversation.index,
        conversation.speaker_a,
        conversation.speaker_b,
        total,
    )

    # Batch embed in chunks
    all_vectors: list[list[float]] = []
    errors = 0
    for i in range(0, total, EMBED_BATCH_SIZE):
        chunk = texts[i : i + EMBED_BATCH_SIZE]
        try:
            vectors = memory_service.vector.embedding.embed_batch(chunk)
            all_vectors.extend(vectors)
        except Exception:
            logger.exception(
                "Error batch-embedding chunk %d-%d for conversation %d",
                i,
                i + len(chunk),
                conversation.index,
            )
            errors += 1
            # Fill with empty vectors so indices stay aligned
            all_vectors.extend([[] for _ in chunk])

    if len(all_vectors) != total:
        logger.error(
            "Vector count mismatch: %d vectors for %d texts",
            len(all_vectors),
            total,
        )

    # Build default metadata
    now = datetime.now(timezone.utc).isoformat()
    base_metadata: dict[str, Any] = {
        "memory_type": "episodic",
        "categories": [],
        "importance": "normal",
        "pinned": False,
        "artifacts": [],
        "created_at_utc": now,
    }
    ttl_meta = build_expiry_metadata(None, "episodic", memory_service._config.memory)
    base_metadata.update(ttl_meta)

    # Build points for batch insert (skip any with empty vectors from errors)
    points = []
    for text, vec, event_date in zip(texts, all_vectors, event_dates):
        if not vec:
            continue
        meta = dict(base_metadata)
        if event_date:
            meta["event_date"] = event_date
        points.append(
            {
                "text": text,
                "vector": vec,
                "user_id": user_id,
                "metadata": meta,
                "role": "user",
            }
        )

    # Batch insert into Qdrant
    memories_created = 0
    try:
        ids = memory_service.vector.insert_batch(points)
        memories_created = len(ids)
    except Exception:
        logger.exception(
            "Error batch-inserting %d points for conversation %d",
            len(points),
            conversation.index,
        )
        errors += 1

    elapsed = time.time() - start
    logger.info(
        "Conversation %d done: %d turns, %d memories (%.1fs)",
        conversation.index,
        total,
        memories_created,
        elapsed,
    )

    return IngestResult(
        conversation_index=conversation.index,
        user_id=user_id,
        sessions_processed=len(conversation.sessions),
        turns_processed=total,
        memories_created=memories_created,
        errors=errors,
        elapsed_seconds=elapsed,
    )


def _ingest_sequential(
    conversation: Conversation,
    memory_service: Any,
    config: BenchmarkConfig,
) -> IngestResult:
    """Ingest turns one-by-one with LLM inference (infer=True mode).

    Each turn goes through the full add_memory pipeline: embedding,
    LLM extraction, classification, and deduplication.
    """
    user_id = make_user_id(conversation.index)
    turns_processed = 0
    memories_created = 0
    errors = 0
    start = time.time()

    total_turns = conversation.total_turns
    effective_total = total_turns
    if config.max_turns > 0:
        effective_total = min(total_turns, config.max_turns)
    logger.info(
        "Ingesting conversation %d (%s & %s): %d sessions, %d/%d turns (infer=True)",
        conversation.index,
        conversation.speaker_a,
        conversation.speaker_b,
        len(conversation.sessions),
        effective_total,
        total_turns,
    )

    turn_count = 0
    for session in conversation.sessions:
        # Parse session date for temporal anchoring
        session_event_date = _parse_locomo_date(session.date_time)
        for turn in session.turns:
            if config.max_turns > 0 and turn_count >= config.max_turns:
                break
            content = turn.format_for_memory()
            try:
                result = memory_service.add_memory(
                    content,
                    user_id=user_id,
                    infer=True,
                    event_date=session_event_date,
                )
                turns_processed += 1

                # Count memories created from the result.
                # add_memory returns {"results": [{"id": ..., "event": "ADD"}, ...]}
                if isinstance(result, dict):
                    created = result.get("results", [])
                    if isinstance(created, list):
                        memories_created += len(created)

                if turns_processed % 25 == 0:
                    logger.info(
                        "  [conv %d] %d/%d turns processed, %d memories",
                        conversation.index,
                        turns_processed,
                        effective_total,
                        memories_created,
                    )

            except Exception:
                logger.exception(
                    "Error ingesting turn %s in conversation %d",
                    turn.dia_id,
                    conversation.index,
                )
                errors += 1
                turns_processed += 1
            turn_count += 1
        if config.max_turns > 0 and turn_count >= config.max_turns:
            break

    elapsed = time.time() - start
    logger.info(
        "Conversation %d done: %d turns, %d memories, %d errors (%.1fs)",
        conversation.index,
        turns_processed,
        memories_created,
        errors,
        elapsed,
    )

    return IngestResult(
        conversation_index=conversation.index,
        user_id=user_id,
        sessions_processed=len(conversation.sessions),
        turns_processed=turns_processed,
        memories_created=memories_created,
        errors=errors,
        elapsed_seconds=elapsed,
    )


def run_ingest(
    conversations: list[Conversation],
    memory_service: Any,
    config: BenchmarkConfig,
) -> IngestState:
    """Run the ingest stage for all selected conversations.

    When infer=True (default): Sequential per-turn ingestion with full
    LLM extraction, classification, and deduplication pipeline.
    When infer=False (--no-infer): Uses batch embedding + batch insert
    with parallel workers across conversations.
    """
    state = IngestState()
    start = time.time()

    if config.infer:
        # Sequential — dedup requires ordered processing within each conversation
        for conv in conversations:
            result = _ingest_sequential(conv, memory_service, config)
            state.results.append(result)
            state.total_turns += result.turns_processed
            state.total_memories += result.memories_created
            state.total_errors += result.errors
    else:
        # Parallel batch ingestion
        workers = config.effective_workers
        logger.info(
            "Batch ingestion with %d workers for %d conversations",
            workers,
            len(conversations),
        )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_ingest_batch, conv, memory_service, config): conv
                for conv in conversations
            }
            for future in as_completed(futures):
                conv = futures[future]
                try:
                    result = future.result()
                except Exception:
                    logger.exception("Failed to ingest conversation %d", conv.index)
                    result = IngestResult(
                        conversation_index=conv.index,
                        user_id=make_user_id(conv.index),
                        sessions_processed=0,
                        turns_processed=0,
                        memories_created=0,
                        errors=1,
                        elapsed_seconds=0.0,
                    )
                state.results.append(result)
                state.total_turns += result.turns_processed
                state.total_memories += result.memories_created
                state.total_errors += result.errors

    # Sort results by conversation index for consistent ordering
    state.results.sort(key=lambda r: r.conversation_index)

    state.elapsed_seconds = time.time() - start
    logger.info(
        "Ingest complete: %d conversations, %d turns, %d memories, %d errors (%.1fs)",
        len(conversations),
        state.total_turns,
        state.total_memories,
        state.total_errors,
        state.elapsed_seconds,
    )
    return state
