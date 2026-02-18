"""Stage 1: Ingest — Feed LoCoMo conversations into mnemory.

For each conversation, creates a unique user_id and ingests all turns
from all sessions using MemoryService.add_memory().
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from benchmarks.locomo.config import BenchmarkConfig
from benchmarks.locomo.dataset import Conversation

logger = logging.getLogger(__name__)


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


def ingest_conversation(
    conversation: Conversation,
    memory_service: Any,
    config: BenchmarkConfig,
) -> IngestResult:
    """Ingest all turns from a conversation into mnemory.

    Each turn is formatted as "{Speaker}: {text}" and stored via add_memory.
    """
    user_id = make_user_id(conversation.index)
    turns_processed = 0
    memories_created = 0
    errors = 0
    start = time.time()

    total_turns = conversation.total_turns
    logger.info(
        "Ingesting conversation %d (%s & %s): %d sessions, %d turns",
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
                    infer=config.infer,
                )
                turns_processed += 1

                # Count memories created from the result
                if isinstance(result, dict):
                    created = result.get("memories", [])
                    if isinstance(created, list):
                        memories_created += len(created)
                    elif result.get("id"):
                        memories_created += 1

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
    """Run the ingest stage for all selected conversations."""
    state = IngestState()
    start = time.time()

    for conv in conversations:
        result = ingest_conversation(conv, memory_service, config)
        state.results.append(result)
        state.total_turns += result.turns_processed
        state.total_memories += result.memories_created
        state.total_errors += result.errors

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
