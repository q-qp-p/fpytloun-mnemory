"""Stage 1: Ingest — Feed LoCoMo conversations into mnemory.

For each conversation, creates a unique user_id and ingests all turns
from all sessions using MemoryService.

Two modes:
- infer=False (default): Batch-embeds all turns and inserts directly
  into the vector store. Fast (~seconds per conversation).
- infer=True: Calls add_memory per turn with LLM extraction + dedup.
  Slow but tests the full mnemory pipeline.
"""

from __future__ import annotations

import logging
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

    # Collect all turns
    texts = [
        turn.format_for_memory()
        for session in conversation.sessions
        for turn in session.turns
    ]
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
    for text, vec in zip(texts, all_vectors):
        if not vec:
            continue
        points.append(
            {
                "text": text,
                "vector": vec,
                "user_id": user_id,
                "metadata": dict(base_metadata),
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
    logger.info(
        "Ingesting conversation %d (%s & %s): %d sessions, %d turns (infer=True)",
        conversation.index,
        conversation.speaker_a,
        conversation.speaker_b,
        len(conversation.sessions),
        total_turns,
    )

    for session in conversation.sessions:
        for turn in session.turns:
            content = turn.format_for_memory()
            try:
                result = memory_service.add_memory(
                    content,
                    user_id=user_id,
                    infer=True,
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
                        total_turns,
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

    When infer=False (default): Uses batch embedding + batch insert with
    parallel workers across conversations.
    When infer=True: Sequential per-turn ingestion with full LLM pipeline.
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
                pool.submit(_ingest_batch, conv, memory_service): conv
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
