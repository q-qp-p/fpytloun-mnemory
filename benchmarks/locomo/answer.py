"""Stage 3: Answer — Generate answers using retrieved memories + eval LLM.

For each question, formats the retrieved memories as context and asks
an LLM to answer the question based on that context.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from benchmarks.locomo.config import BenchmarkConfig
from benchmarks.locomo.search import SearchState

logger = logging.getLogger(__name__)

# System prompt for the answer LLM
_ANSWER_SYSTEM_PROMPT = """\
You are a helpful assistant answering questions about people based on their \
conversation history. You will be given memories extracted from past \
conversations between two people, and a question about one of them.

## How to answer

Think step by step:
1. Read ALL the provided memories carefully
2. Identify which memories are relevant to the question
3. Connect facts across multiple memories when needed
4. For temporal questions, calculate dates from timestamps and relative references
5. Formulate a precise, concise answer

Answer based on the provided memories. Use logical reasoning to connect \
facts and draw conclusions when the answer is not stated directly but can \
be inferred from the evidence.

For hypothetical or likelihood questions (e.g., "Would X do Y?"), reason \
from available evidence about the person's preferences, experiences, and \
stated intentions. Give your best assessment rather than saying you don't know.

Only say "I don't know" if the memories contain truly no relevant information.

## Temporal reasoning

Some memories include a timestamp [DD Month YYYY] showing when the event \
occurred. Use these timestamps to answer temporal questions:
- The timestamp is the actual date of the event described in the memory.
- When asked "when" something happened, use the timestamp directly.
- When asked about duration or "how long", calculate from the relevant \
timestamps.
- If a memory has no timestamp, its date is unknown — rely on the memory \
text for any temporal clues.

## Answer format

Keep your answer to 5-6 words when possible, at most one sentence. \
Be direct and factual."""

_ANSWER_USER_TEMPLATE = """\
Memories from past conversations:
{memories}

Question: {question}

Think step by step, then answer concisely (5-6 words):"""


@dataclass
class AnswerResult:
    """Answer result for a single question."""

    conversation_index: int
    question: str
    ground_truth: str  # expected answer
    category: int
    category_name: str
    generated_answer: str
    num_memories: int  # how many memories were in context
    elapsed_seconds: float = 0.0
    error: str | None = None


@dataclass
class AnswerState:
    """Complete state after the answer stage."""

    results: list[AnswerResult] = field(default_factory=list)
    total_questions: int = 0
    total_errors: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [
                {
                    "conversation_index": r.conversation_index,
                    "question": r.question,
                    "ground_truth": r.ground_truth,
                    "category": r.category,
                    "category_name": r.category_name,
                    "generated_answer": r.generated_answer,
                    "num_memories": r.num_memories,
                    "elapsed_seconds": r.elapsed_seconds,
                    "error": r.error,
                }
                for r in self.results
            ],
            "total_questions": self.total_questions,
            "total_errors": self.total_errors,
            "elapsed_seconds": self.elapsed_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnswerState:
        state = cls(
            total_questions=data.get("total_questions", 0),
            total_errors=data.get("total_errors", 0),
            elapsed_seconds=data.get("elapsed_seconds", 0.0),
        )
        for r in data.get("results", []):
            state.results.append(
                AnswerResult(
                    conversation_index=r["conversation_index"],
                    question=r["question"],
                    ground_truth=r["ground_truth"],
                    category=r["category"],
                    category_name=r["category_name"],
                    generated_answer=r.get("generated_answer", ""),
                    num_memories=r.get("num_memories", 0),
                    elapsed_seconds=r.get("elapsed_seconds", 0.0),
                    error=r.get("error"),
                )
            )
        return state


def _format_memories(memories: list[dict[str, Any]]) -> str:
    """Format retrieved memories into a context string for the LLM.

    Includes event_date as a timestamp prefix when available, following
    Mem0's approach of "{timestamp}: {memory}" formatting for temporal
    reasoning.
    """
    if not memories:
        return "(No relevant memories found)"

    lines = []
    for i, mem in enumerate(memories, 1):
        # Handle both search_memories and find_memories result formats
        text = mem.get("memory") or mem.get("text") or mem.get("content", "")

        # Extract event_date from metadata or top-level (search results
        # include it at top level via server._format_memories)
        event_date = None
        metadata = mem.get("metadata") or {}
        event_date = mem.get("event_date") or metadata.get("event_date")

        # Format timestamp prefix
        ts_prefix = ""
        if event_date:
            # Extract just the date portion for readability
            try:
                dt = datetime.fromisoformat(event_date)
                ts_prefix = f"[{dt.strftime('%d %B %Y')}] "
            except (ValueError, TypeError):
                ts_prefix = f"[{event_date}] "

        score = mem.get("score", 0.0)
        if score:
            lines.append(f"[{i}] {ts_prefix}(score: {score:.2f}) {text}")
        else:
            lines.append(f"[{i}] {ts_prefix}{text}")
    return "\n".join(lines)


def _generate_answer(
    question: str,
    memories: list[dict[str, Any]],
    llm_client: Any,
    answer_limit: int = 0,
) -> str:
    """Generate an answer using the eval LLM.

    The model is determined by the llm_client's configuration.

    Args:
        answer_limit: Max memories to include in context. 0 = all.
    """
    if answer_limit > 0:
        memories = memories[:answer_limit]
    context = _format_memories(memories)
    user_msg = _ANSWER_USER_TEMPLATE.format(
        memories=context,
        question=question,
    )

    messages = [
        {"role": "system", "content": _ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    response = llm_client.generate(messages)
    return response.strip()


def run_answer(
    search_state: SearchState,
    llm_client: Any,
    config: BenchmarkConfig,
    model: str,
) -> AnswerState:
    """Run the answer stage for all search results.

    Args:
        search_state: Output from the search stage.
        llm_client: LLM client with a generate() method.
        config: Benchmark configuration.
        model: Model name to use for answer generation.
    """
    state = AnswerState()
    start = time.time()

    total = len(search_state.results)
    logger.info("Generating answers for %d questions (model=%s)", total, model)

    for qi, search_result in enumerate(search_state.results):
        q_start = time.time()
        generated = ""
        error = None

        if search_result.error:
            # Propagate search errors
            error = f"Search failed: {search_result.error}"
        else:
            try:
                generated = _generate_answer(
                    search_result.question,
                    search_result.memories,
                    llm_client,
                    answer_limit=config.answer_limit,
                )
            except Exception as e:
                logger.exception(
                    "Error generating answer for: %s",
                    search_result.question[:80],
                )
                error = str(e)

        elapsed = time.time() - q_start
        state.results.append(
            AnswerResult(
                conversation_index=search_result.conversation_index,
                question=search_result.question,
                ground_truth=search_result.answer,
                category=search_result.category,
                category_name=search_result.category_name,
                generated_answer=generated,
                num_memories=len(search_result.memories),
                elapsed_seconds=elapsed,
                error=error,
            )
        )
        state.total_questions += 1
        if error:
            state.total_errors += 1

        if (qi + 1) % 50 == 0:
            logger.info("  %d/%d answers generated", qi + 1, total)

    state.elapsed_seconds = time.time() - start
    logger.info(
        "Answer complete: %d questions, %d errors (%.1fs)",
        state.total_questions,
        state.total_errors,
        state.elapsed_seconds,
    )
    return state
