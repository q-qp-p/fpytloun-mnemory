"""Stage 4: Evaluate — LLM judge scoring of generated answers.

Uses an LLM judge to compare generated answers against ground truth,
following the standard LoCoMo accuracy evaluation methodology.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from benchmarks.locomo.answer import AnswerState
from benchmarks.locomo.config import BenchmarkConfig

logger = logging.getLogger(__name__)

# LLM judge prompt — adapted from the standard LoCoMo evaluation methodology.
# Generous grading: as long as the answer touches the same topic, it's CORRECT.
_JUDGE_SYSTEM_PROMPT = """\
You are evaluating conversational AI memory recall. Return JSON only."""

_JUDGE_USER_TEMPLATE = """\
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. \
You will be given:
(1) a question (posed by one user to another user),
(2) a 'gold' (ground truth) answer,
(3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know \
about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes \
the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with \
your grading - as long as it touches on the same topic as the gold answer, \
it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, \
year, etc. The generated answer might be much longer or use relative time \
references (like "last Tuesday" or "next month"), but you should be generous \
with your grading - as long as it refers to the same date or time period as \
the gold answer, it should be counted as CORRECT. Even if the format differs \
(e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then \
finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break \
the evaluation script.

Return your response in JSON format with two keys: "reasoning" for your \
explanation and "label" for CORRECT or WRONG."""


@dataclass
class EvalResult:
    """Evaluation result for a single question."""

    conversation_index: int
    question: str
    ground_truth: str
    generated_answer: str
    category: int
    category_name: str
    is_correct: bool | None  # None if evaluation failed
    reasoning: str
    elapsed_seconds: float = 0.0
    error: str | None = None


@dataclass
class EvalState:
    """Complete state after the evaluate stage."""

    results: list[EvalResult] = field(default_factory=list)
    total_evaluated: int = 0
    total_correct: int = 0
    total_errors: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [
                {
                    "conversation_index": r.conversation_index,
                    "question": r.question,
                    "ground_truth": r.ground_truth,
                    "generated_answer": r.generated_answer,
                    "category": r.category,
                    "category_name": r.category_name,
                    "is_correct": r.is_correct,
                    "reasoning": r.reasoning,
                    "elapsed_seconds": r.elapsed_seconds,
                    "error": r.error,
                }
                for r in self.results
            ],
            "total_evaluated": self.total_evaluated,
            "total_correct": self.total_correct,
            "total_errors": self.total_errors,
            "elapsed_seconds": self.elapsed_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalState:
        state = cls(
            total_evaluated=data.get("total_evaluated", 0),
            total_correct=data.get("total_correct", 0),
            total_errors=data.get("total_errors", 0),
            elapsed_seconds=data.get("elapsed_seconds", 0.0),
        )
        for r in data.get("results", []):
            state.results.append(
                EvalResult(
                    conversation_index=r["conversation_index"],
                    question=r["question"],
                    ground_truth=r["ground_truth"],
                    generated_answer=r.get("generated_answer", ""),
                    category=r["category"],
                    category_name=r["category_name"],
                    is_correct=r.get("is_correct"),
                    reasoning=r.get("reasoning", ""),
                    elapsed_seconds=r.get("elapsed_seconds", 0.0),
                    error=r.get("error"),
                )
            )
        return state


def _judge_answer(
    question: str,
    gold_answer: str,
    generated_answer: str,
    llm_client: Any,
) -> tuple[bool | None, str, str | None]:
    """Judge a single answer using the LLM.

    The model is determined by the llm_client's configuration.

    Returns:
        Tuple of (is_correct, reasoning, error).
    """
    prompt = _JUDGE_USER_TEMPLATE.format(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
    )

    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        response = llm_client.generate(messages)
        # Parse JSON response
        from mnemory.llm import parse_json_response

        data = parse_json_response(response)
        label = data.get("label", "WRONG").upper().strip()
        reasoning = data.get("reasoning", "No reasoning provided")
        is_correct = label == "CORRECT"
        return is_correct, reasoning, None
    except Exception as e:
        return None, "", str(e)


def run_evaluate(
    answer_state: AnswerState,
    llm_client: Any,
    config: BenchmarkConfig,
    model: str,
) -> EvalState:
    """Run the evaluate stage for all answer results.

    Args:
        answer_state: Output from the answer stage.
        llm_client: LLM client with a generate() method.
        config: Benchmark configuration.
        model: Model name to use for judging.
    """
    state = EvalState()
    start = time.time()

    total = len(answer_state.results)
    logger.info("Evaluating %d answers (judge model=%s)", total, model)

    for qi, answer_result in enumerate(answer_state.results):
        q_start = time.time()

        if answer_result.error:
            # Propagate upstream errors
            state.results.append(
                EvalResult(
                    conversation_index=answer_result.conversation_index,
                    question=answer_result.question,
                    ground_truth=answer_result.ground_truth,
                    generated_answer=answer_result.generated_answer,
                    category=answer_result.category,
                    category_name=answer_result.category_name,
                    is_correct=None,
                    reasoning="",
                    error=f"Upstream error: {answer_result.error}",
                )
            )
            state.total_errors += 1
            continue

        is_correct, reasoning, error = _judge_answer(
            answer_result.question,
            answer_result.ground_truth,
            answer_result.generated_answer,
            llm_client,
        )

        elapsed = time.time() - q_start
        state.results.append(
            EvalResult(
                conversation_index=answer_result.conversation_index,
                question=answer_result.question,
                ground_truth=answer_result.ground_truth,
                generated_answer=answer_result.generated_answer,
                category=answer_result.category,
                category_name=answer_result.category_name,
                is_correct=is_correct,
                reasoning=reasoning,
                elapsed_seconds=elapsed,
                error=error,
            )
        )

        if error:
            state.total_errors += 1
        else:
            state.total_evaluated += 1
            if is_correct:
                state.total_correct += 1

        if (qi + 1) % 50 == 0:
            acc = (
                state.total_correct / state.total_evaluated * 100
                if state.total_evaluated > 0
                else 0.0
            )
            logger.info(
                "  %d/%d evaluated — running accuracy: %.1f%%",
                qi + 1,
                total,
                acc,
            )

    state.elapsed_seconds = time.time() - start
    acc = (
        state.total_correct / state.total_evaluated * 100
        if state.total_evaluated > 0
        else 0.0
    )
    logger.info(
        "Evaluate complete: %d/%d correct (%.1f%%), %d errors (%.1fs)",
        state.total_correct,
        state.total_evaluated,
        acc,
        state.total_errors,
        state.elapsed_seconds,
    )
    return state
