"""Benchmark configuration for LoCoMo evaluation.

Defines BenchmarkConfig dataclass and argument parsing for the CLI.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

# Default dataset URL (snap-research/locomo on GitHub)
DATASET_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/"
    "refs/heads/main/data/locomo10.json"
)

# Default data directory for benchmark artifacts
DEFAULT_DATA_DIR = Path(__file__).parent / "data"
DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"

# LoCoMo category mapping (from the original paper)
CATEGORY_MAP = {
    1: "single_hop",
    2: "temporal",
    3: "multi_hop",
    4: "open_domain",
    5: "adversarial",
}

# Categories to evaluate by default (exclude adversarial)
DEFAULT_CATEGORIES = [1, 2, 3, 4]


@dataclass
class BenchmarkConfig:
    """Configuration for a LoCoMo benchmark run."""

    # Stages to run
    stages: list[str] = field(
        default_factory=lambda: ["ingest", "search", "answer", "evaluate"]
    )

    # Dataset
    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    dataset_url: str = DATASET_URL

    # Results
    results_dir: Path = field(default_factory=lambda: DEFAULT_RESULTS_DIR)
    resume: str | None = None  # Path to previous run to resume from

    # Conversation selection
    conversations: list[int] | None = None  # None = all 10
    categories: list[int] = field(default_factory=lambda: list(DEFAULT_CATEGORIES))

    # Ingestion
    infer: bool = False  # Default: raw storage (fast). --infer enables LLM extraction.

    # Search
    search_method: str = "search_memories"  # or "find_memories"
    search_limit: int = 10

    # LLM models
    eval_model: str = ""  # Model for answering questions (default: use mnemory's LLM)
    judge_model: str = ""  # Model for judging answers (default: use mnemory's LLM)

    # mnemory configuration overrides
    llm_model: str = ""  # Override LLM_MODEL for mnemory
    embed_model: str = ""  # Override EMBED_MODEL for mnemory

    # Parallelization
    workers: int = 0  # 0 = auto (4 for no-infer, 1 for infer)

    # Runtime
    verbose: bool = False

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.results_dir = Path(self.results_dir)

    @property
    def dataset_path(self) -> Path:
        return self.data_dir / "locomo10.json"

    @property
    def effective_workers(self) -> int:
        """Resolve worker count: 0 = auto-detect based on infer mode."""
        if self.workers > 0:
            return self.workers
        return 1 if self.infer else 4

    def resolve_eval_model(self, fallback: str) -> str:
        """Resolve eval model, falling back to mnemory's LLM model."""
        return self.eval_model or fallback

    def resolve_judge_model(self, fallback: str) -> str:
        """Resolve judge model, falling back to eval model, then mnemory's LLM."""
        return self.judge_model or self.eval_model or fallback


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.locomo",
        description="LoCoMo benchmark for evaluating mnemory's long-term memory",
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # --- download ---
    dl = sub.add_parser("download", help="Download the LoCoMo dataset")
    dl.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory to store dataset (default: benchmarks/locomo/data/)",
    )

    # --- run ---
    run = sub.add_parser("run", help="Run the benchmark")
    run.add_argument(
        "--stages",
        type=str,
        default="ingest,search,answer,evaluate",
        help="Comma-separated stages to run (default: all)",
    )
    run.add_argument(
        "--conversations",
        type=str,
        default=None,
        help="Comma-separated conversation indices to run (default: all)",
    )
    run.add_argument(
        "--categories",
        type=str,
        default="1,2,3,4",
        help="Comma-separated category numbers to evaluate (default: 1,2,3,4)",
    )
    run.add_argument(
        "--infer",
        action="store_true",
        help=(
            "Enable LLM fact extraction + classification during ingestion. "
            "Default is off (raw storage with embedding only, much faster)."
        ),
    )
    run.add_argument(
        "--search-method",
        choices=["search_memories", "find_memories"],
        default="search_memories",
        help="Search method to use (default: search_memories)",
    )
    run.add_argument(
        "--search-limit",
        type=int,
        default=10,
        help="Number of search results to retrieve (default: 10)",
    )
    run.add_argument(
        "--eval-model",
        type=str,
        default="",
        help="LLM model for answering questions (default: mnemory's LLM_MODEL)",
    )
    run.add_argument(
        "--judge-model",
        type=str,
        default="",
        help="LLM model for judging answers (default: eval-model)",
    )
    run.add_argument(
        "--llm-model",
        type=str,
        default="",
        help="Override LLM_MODEL for mnemory's extraction pipeline",
    )
    run.add_argument(
        "--embed-model",
        type=str,
        default="",
        help="Override EMBED_MODEL for mnemory's embeddings",
    )
    run.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel workers for ingestion (default: auto — 4 for raw, 1 for --infer)",
    )
    run.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the dataset",
    )
    run.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory to store results",
    )
    run.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to previous run directory to resume from",
    )
    run.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    # --- report ---
    rpt = sub.add_parser("report", help="Show results from a previous run")
    rpt.add_argument(
        "run_dir",
        type=str,
        help="Path to the run results directory",
    )

    return parser


def parse_args(argv: list[str] | None = None) -> tuple[str, BenchmarkConfig]:
    """Parse CLI arguments and return (command, config).

    Returns:
        Tuple of (command_name, BenchmarkConfig).
        command_name is one of: "download", "run", "report".
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        raise SystemExit(1)

    config = BenchmarkConfig(data_dir=getattr(args, "data_dir", DEFAULT_DATA_DIR))

    if args.command == "run":
        config.stages = [s.strip() for s in args.stages.split(",")]
        config.infer = args.infer
        config.search_method = args.search_method
        config.search_limit = args.search_limit
        config.eval_model = args.eval_model
        config.judge_model = args.judge_model
        config.llm_model = args.llm_model
        config.embed_model = args.embed_model
        config.workers = args.workers
        config.results_dir = args.results_dir
        config.resume = args.resume
        config.verbose = args.verbose
        config.categories = [int(c) for c in args.categories.split(",")]
        if args.conversations:
            config.conversations = [int(c) for c in args.conversations.split(",")]

    elif args.command == "report":
        config.resume = args.run_dir

    return args.command, config
