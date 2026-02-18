"""Stage 2: Search — Query mnemory for each LoCoMo question.

Retrieves relevant memories for each question using either
search_memories (single vector search) or find_memories (AI-powered).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from benchmarks.locomo.config import BenchmarkConfig
from benchmarks.locomo.dataset import Conversation, Question
from benchmarks.locomo.ingest import IngestState

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result for a single question."""

    conversation_index: int
    question: str
    answer: str  # ground truth
    category: int
    category_name: str
    memories: list[dict[str, Any]]  # retrieved memories
    queries: list[str] | None = None  # for find_memories
    elapsed_seconds: float = 0.0
    error: str | None = None


@dataclass
class SearchState:
    """Complete state after the search stage."""

    results: list[SearchResult] = field(default_factory=list)
    total_questions: int = 0
    total_errors: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [
                {
                    "conversation_index": r.conversation_index,
                    "question": r.question,
                    "answer": r.answer,
                    "category": r.category,
                    "category_name": r.category_name,
                    "memories": r.memories,
                    "queries": r.queries,
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
    def from_dict(cls, data: dict[str, Any]) -> SearchState:
        state = cls(
            total_questions=data.get("total_questions", 0),
            total_errors=data.get("total_errors", 0),
            elapsed_seconds=data.get("elapsed_seconds", 0.0),
        )
        for r in data.get("results", []):
            state.results.append(
                SearchResult(
                    conversation_index=r["conversation_index"],
                    question=r["question"],
                    answer=r["answer"],
                    category=r["category"],
                    category_name=r["category_name"],
                    memories=r.get("memories", []),
                    queries=r.get("queries"),
                    elapsed_seconds=r.get("elapsed_seconds", 0.0),
                    error=r.get("error"),
                )
            )
        return state


def _search_single(
    question: Question,
    user_id: str,
    memory_service: Any,
    config: BenchmarkConfig,
) -> SearchResult:
    """Search for a single question using the configured method."""
    start = time.time()
    memories: list[dict[str, Any]] = []
    queries: list[str] | None = None
    error: str | None = None

    try:
        if config.search_method == "find_memories":
            result = memory_service.find_memories(
                question=question.question,
                user_id=user_id,
                limit=config.search_limit,
            )
            memories = result.get("results", [])
            queries = result.get("queries", [])
        else:
            memories = memory_service.search_memories(
                query=question.question,
                user_id=user_id,
                limit=config.search_limit,
            )
    except Exception as e:
        logger.exception("Error searching for question: %s", question.question[:80])
        error = str(e)

    elapsed = time.time() - start
    return SearchResult(
        conversation_index=-1,  # set by caller
        question=question.question,
        answer=question.answer,
        category=question.category,
        category_name=question.category_name,
        memories=memories,
        queries=queries,
        elapsed_seconds=elapsed,
        error=error,
    )


def _limit_questions_per_category(
    questions: list[Question],
    max_per_category: int,
) -> list[Question]:
    """Limit questions to max_per_category per category.

    Preserves original ordering within each category.
    """
    counts: dict[int, int] = {}
    result: list[Question] = []
    for q in questions:
        count = counts.get(q.category, 0)
        if count < max_per_category:
            result.append(q)
            counts[q.category] = count + 1
    return result


def run_search(
    conversations: list[Conversation],
    memory_service: Any,
    config: BenchmarkConfig,
    ingest_state: IngestState,
) -> SearchState:
    """Run the search stage for all questions in selected conversations."""
    state = SearchState()
    start = time.time()

    for conv in conversations:
        user_id = ingest_state.get_user_id(conv.index)
        if not user_id:
            logger.error(
                "No user_id found for conversation %d — was ingest run?",
                conv.index,
            )
            continue

        questions = conv.get_questions(config.categories)

        # Apply per-category question limit if configured
        if config.max_questions > 0:
            questions = _limit_questions_per_category(questions, config.max_questions)

        logger.info(
            "Searching conversation %d: %d questions (user_id=%s, method=%s)",
            conv.index,
            len(questions),
            user_id,
            config.search_method,
        )

        for qi, question in enumerate(questions):
            result = _search_single(question, user_id, memory_service, config)
            result.conversation_index = conv.index
            state.results.append(result)
            state.total_questions += 1
            if result.error:
                state.total_errors += 1

            if (qi + 1) % 50 == 0:
                logger.info(
                    "  [conv %d] %d/%d questions searched",
                    conv.index,
                    qi + 1,
                    len(questions),
                )

    state.elapsed_seconds = time.time() - start
    logger.info(
        "Search complete: %d questions, %d errors (%.1fs)",
        state.total_questions,
        state.total_errors,
        state.elapsed_seconds,
    )
    return state
