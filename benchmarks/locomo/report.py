"""Report — Format and display benchmark results.

Generates results tables with per-category accuracy and comparison
against published scores from other memory systems.
"""

from __future__ import annotations

from typing import Any

from benchmarks.locomo.config import BenchmarkConfig
from benchmarks.locomo.evaluate import EvalState
from benchmarks.locomo.ingest import IngestState

# Published scores for comparison.
#
# NOTE: The LoCoMo paper does not explicitly define the category-to-number
# mapping, leading to inconsistencies across implementations. We use the
# Mem0/Memobase convention: cat1=single_hop(282), cat2=temporal(321),
# cat3=multi_hop(96), cat4=open_domain(841). Some systems (EverMemOS,
# MemMachine) swap single_hop and open_domain. Overall scores are comparable
# regardless of naming since they're weighted by question count.
#
# Scores below use the Memobase convention (gpt-4o-mini eval LLM).
REFERENCE_SCORES: dict[str, dict[str, float]] = {
    "Memobase": {
        "single_hop": 70.9,
        "multi_hop": 52.1,
        "temporal": 85.0,
        "open_domain": 77.2,
        "overall": 75.8,
    },
    "Mem0": {
        "single_hop": 67.1,
        "multi_hop": 51.2,
        "temporal": 55.5,
        "open_domain": 72.9,
        "overall": 66.9,
    },
    "Mem0-Graph": {
        "single_hop": 65.7,
        "multi_hop": 47.2,
        "temporal": 58.1,
        "open_domain": 75.7,
        "overall": 68.4,
    },
    "Zep": {
        "single_hop": 61.7,
        "multi_hop": 41.4,
        "temporal": 49.3,
        "open_domain": 76.6,
        "overall": 66.0,
    },
    "LangMem": {
        "single_hop": 62.2,
        "multi_hop": 47.9,
        "temporal": 23.4,
        "open_domain": 71.1,
        "overall": 58.1,
    },
}


def _compute_scores(eval_state: EvalState) -> dict[str, dict[str, Any]]:
    """Compute per-category and overall scores from evaluation results.

    Returns a dict mapping category_name -> {correct, total, accuracy}.
    Includes an "overall" key with weighted average.
    """
    by_category: dict[str, dict[str, int]] = {}

    for result in eval_state.results:
        cat = result.category_name
        if cat not in by_category:
            by_category[cat] = {"correct": 0, "total": 0}
        if result.is_correct is not None:
            by_category[cat]["total"] += 1
            if result.is_correct:
                by_category[cat]["correct"] += 1

    scores: dict[str, dict[str, Any]] = {}
    total_correct = 0
    total_evaluated = 0

    for cat_name in ["single_hop", "multi_hop", "temporal", "open_domain"]:
        data = by_category.get(cat_name, {"correct": 0, "total": 0})
        correct = data["correct"]
        total = data["total"]
        accuracy = (correct / total * 100) if total > 0 else 0.0
        scores[cat_name] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
        }
        total_correct += correct
        total_evaluated += total

    # Weighted overall (weighted by category size, matching LoCoMo methodology)
    overall_accuracy = (
        total_correct / total_evaluated * 100 if total_evaluated > 0 else 0.0
    )
    scores["overall"] = {
        "correct": total_correct,
        "total": total_evaluated,
        "accuracy": overall_accuracy,
    }

    return scores


def format_report(
    eval_state: EvalState,
    config: BenchmarkConfig,
    mnemory_config: Any | None,
    total_time: float,
    ingest_state: IngestState | None = None,
) -> str:
    """Format the benchmark results as a human-readable report string."""
    scores = _compute_scores(eval_state)
    lines: list[str] = []

    # Header
    try:
        from mnemory import __version__

        version = __version__
    except ImportError:
        version = "unknown"

    lines.append("")
    lines.append(f"LoCoMo Benchmark Results - mnemory v{version}")
    lines.append("=" * 60)

    # Configuration
    if mnemory_config:
        llm_model = mnemory_config.llm.model
        embed_model = mnemory_config.embed.model
    else:
        llm_model = config.llm_model or "(unknown)"
        embed_model = config.embed_model or "(unknown)"

    eval_model = config.resolve_eval_model(llm_model)
    judge_model = config.resolve_judge_model(llm_model)

    lines.append(f"LLM model:      {llm_model}")
    lines.append(f"Embed model:    {embed_model}")
    lines.append(f"Eval model:     {eval_model}")
    lines.append(f"Judge model:    {judge_model}")
    lines.append(
        f"Search method:  {config.search_method} (limit={config.search_limit})"
    )
    lines.append(f"Infer:          {config.infer}")
    lines.append(f"Categories:     {config.categories}")
    lines.append("")

    # Results table
    lines.append(f"{'Category':<15} {'Correct':>8} {'Total':>6} {'Accuracy':>9}")
    lines.append("-" * 42)

    for cat_name in ["single_hop", "multi_hop", "temporal", "open_domain"]:
        s = scores.get(cat_name, {"correct": 0, "total": 0, "accuracy": 0.0})
        lines.append(
            f"{cat_name:<15} {s['correct']:>8} {s['total']:>6} {s['accuracy']:>8.1f}%"
        )

    lines.append("-" * 42)
    overall = scores["overall"]
    lines.append(
        f"{'Overall':<15} {overall['correct']:>8} {overall['total']:>6} "
        f"{overall['accuracy']:>8.1f}%"
    )
    lines.append("")

    # Comparison table
    lines.append("Comparison with published results:")
    lines.append(
        f"{'System':<15} {'single':>7} {'multi':>7} {'temporal':>9} "
        f"{'open':>7} {'Overall':>8}"
    )
    lines.append("-" * 58)

    for system, ref in REFERENCE_SCORES.items():
        lines.append(
            f"{system:<15} {ref['single_hop']:>7.1f} {ref['multi_hop']:>7.1f} "
            f"{ref['temporal']:>9.1f} {ref['open_domain']:>7.1f} "
            f"{ref['overall']:>8.1f}"
        )

    # Our scores
    mnemory_scores = scores
    lines.append(
        f"{'mnemory':<15} "
        f"{mnemory_scores.get('single_hop', {}).get('accuracy', 0.0):>7.1f} "
        f"{mnemory_scores.get('multi_hop', {}).get('accuracy', 0.0):>7.1f} "
        f"{mnemory_scores.get('temporal', {}).get('accuracy', 0.0):>9.1f} "
        f"{mnemory_scores.get('open_domain', {}).get('accuracy', 0.0):>7.1f} "
        f"{mnemory_scores.get('overall', {}).get('accuracy', 0.0):>8.1f}"
    )
    lines.append("")

    # Timing
    if total_time > 0:
        lines.append(f"Total time: {total_time:.1f}s")
    if ingest_state:
        lines.append(
            f"  Ingest:   {ingest_state.elapsed_seconds:.1f}s "
            f"({ingest_state.total_turns} turns -> "
            f"{ingest_state.total_memories} memories)"
        )
    if eval_state.elapsed_seconds > 0:
        lines.append(f"  Evaluate: {eval_state.elapsed_seconds:.1f}s")

    # Error summary
    if eval_state.total_errors > 0:
        lines.append(f"\nErrors: {eval_state.total_errors}")

    return "\n".join(lines)


def print_report(
    eval_state: EvalState,
    config: BenchmarkConfig,
    mnemory_config: Any | None,
    total_time: float,
    ingest_state: IngestState | None = None,
) -> None:
    """Print the benchmark report to stdout."""
    print(format_report(eval_state, config, mnemory_config, total_time, ingest_state))
