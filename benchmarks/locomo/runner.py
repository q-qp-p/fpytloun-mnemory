"""Runner — Orchestrates all benchmark stages and manages state.

Creates the mnemory MemoryService, runs stages in order, saves/loads
intermediate state for resumability.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.locomo.answer import AnswerState, run_answer
from benchmarks.locomo.config import BenchmarkConfig
from benchmarks.locomo.dataset import download_dataset, load_dataset
from benchmarks.locomo.evaluate import EvalState, run_evaluate
from benchmarks.locomo.ingest import IngestState, run_ingest
from benchmarks.locomo.report import print_report
from benchmarks.locomo.search import SearchState, run_search

logger = logging.getLogger(__name__)


def _create_mnemory_config(
    bench_config: BenchmarkConfig,
    data_dir: str,
) -> Any:
    """Create a mnemory Config for the benchmark.

    Uses a temporary directory for Qdrant storage (embedded mode) and
    filesystem artifacts. Picks up LLM/embed config from environment
    variables, with optional overrides from bench_config.
    """
    from mnemory.config import (
        ArtifactConfig,
        Config,
        EmbedConfig,
        LLMConfig,
        MemoryConfig,
        ServerConfig,
        VectorConfig,
    )

    # Override env vars if CLI args provided
    if bench_config.llm_model:
        os.environ["LLM_MODEL"] = bench_config.llm_model
    if bench_config.embed_model:
        os.environ["EMBED_MODEL"] = bench_config.embed_model

    config = Config(
        vector=VectorConfig(
            qdrant_host="",  # local embedded mode
            qdrant_path=os.path.join(data_dir, "qdrant"),
        ),
        llm=LLMConfig(
            reasoning_effort=bench_config.reasoning_effort or None,
        ),
        embed=EmbedConfig(),
        artifact=ArtifactConfig(
            backend="filesystem",
            filesystem_path=os.path.join(data_dir, "artifacts"),
        ),
        server=ServerConfig(),
        memory=MemoryConfig(
            # Disable TTL for benchmark — we want all memories to persist
            ttl_fact=None,
            ttl_preference=None,
            ttl_episodic=None,
            ttl_procedural=None,
            ttl_context=None,
            # Disable auto-classification when not using inference —
            # avoids a separate LLM call per memory during raw ingestion
            auto_classify=bench_config.infer,
        ),
    )
    config.validate()
    return config


def _create_eval_llm_client(
    mnemory_config: Any,
    model: str,
) -> Any:
    """Create an LLM client for eval/judge, potentially with a different model.

    If model matches mnemory's configured model, reuses the same config.
    Otherwise creates a new LLMConfig with the specified model.
    """
    from mnemory.config import LLMConfig
    from mnemory.llm import LLMClient

    if not model or model == mnemory_config.llm.model:
        return LLMClient(mnemory_config.llm)

    eval_config = LLMConfig(
        model=model,
        base_url=mnemory_config.llm.base_url,
        api_key=mnemory_config.llm.api_key,
    )
    return eval_config, LLMClient(eval_config)


class BenchmarkRunner:
    """Orchestrates the LoCoMo benchmark pipeline."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._run_dir: Path | None = None
        self._mnemory_config: Any = None
        self._memory_service: Any = None
        self._data_dir: str | None = None

    @property
    def run_dir(self) -> Path:
        if self._run_dir is None:
            raise RuntimeError("run_dir not initialized — call run() first")
        return self._run_dir

    def _init_run_dir(self) -> Path:
        """Create a timestamped run directory for results."""
        if self.config.resume:
            run_dir = Path(self.config.resume)
            if not run_dir.exists():
                raise FileNotFoundError(f"Resume directory not found: {run_dir}")
            logger.info("Resuming from %s", run_dir)
            return run_dir

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = self.config.results_dir / f"locomo_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Results directory: %s", run_dir)
        return run_dir

    def _save_state(self, name: str, state: Any) -> None:
        """Save stage state to JSON."""
        path = self.run_dir / f"{name}_state.json"
        with open(path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
        logger.info("Saved %s state to %s", name, path)

    def _load_state(self, name: str, cls: type) -> Any | None:
        """Load stage state from JSON if it exists."""
        path = self.run_dir / f"{name}_state.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        logger.info("Loaded %s state from %s", name, path)
        return cls.from_dict(data)

    def _save_run_config(self) -> None:
        """Save the benchmark configuration for this run."""
        path = self.run_dir / "config.json"
        config_dict = {
            "stages": self.config.stages,
            "infer": self.config.infer,
            "search_method": self.config.search_method,
            "search_limit": self.config.search_limit,
            "eval_model": self.config.eval_model,
            "judge_model": self.config.judge_model,
            "llm_model": self.config.llm_model,
            "embed_model": self.config.embed_model,
            "conversations": self.config.conversations,
            "categories": self.config.categories,
            "max_questions": self.config.max_questions,
            "max_turns": self.config.max_turns,
            "reasoning_effort": self.config.reasoning_effort,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        # Add resolved model names after mnemory config is created
        if self._mnemory_config:
            config_dict["resolved_llm_model"] = self._mnemory_config.llm.model
            config_dict["resolved_embed_model"] = self._mnemory_config.embed.model
            config_dict["resolved_eval_model"] = self.config.resolve_eval_model(
                self._mnemory_config.llm.model
            )
            config_dict["resolved_judge_model"] = self.config.resolve_judge_model(
                self._mnemory_config.llm.model
            )
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)

    def _init_mnemory(self) -> None:
        """Initialize mnemory MemoryService with benchmark config."""
        from mnemory.memory import MemoryService

        # Use a persistent data dir inside the run directory so we can
        # resume. If resuming, reuse the existing Qdrant data.
        self._data_dir = str(self.run_dir / "mnemory_data")
        os.makedirs(self._data_dir, exist_ok=True)

        self._mnemory_config = _create_mnemory_config(self.config, self._data_dir)
        self._memory_service = MemoryService(self._mnemory_config)
        logger.info(
            "Initialized mnemory (LLM=%s, embed=%s, qdrant=%s)",
            self._mnemory_config.llm.model,
            self._mnemory_config.embed.model,
            self._data_dir,
        )

    def run(self) -> None:
        """Run the complete benchmark pipeline."""
        start = time.time()

        # Initialize
        self._run_dir = self._init_run_dir()

        # Load dataset
        download_dataset(self.config.data_dir)
        dataset = load_dataset(self.config.dataset_path)
        conversations = dataset.get_conversations(self.config.conversations)

        print(f"\n{dataset.summary(self.config.categories)}")
        print(f"Selected conversations: {len(conversations)}")
        print(f"Stages: {', '.join(self.config.stages)}")
        print(f"Infer: {self.config.infer}")
        print(f"Workers: {self.config.effective_workers}")
        print(f"Search: {self.config.search_method} (limit={self.config.search_limit})")
        if self.config.max_questions > 0:
            print(f"Max questions per category: {self.config.max_questions}")
        if self.config.max_turns > 0:
            print(f"Max turns per conversation: {self.config.max_turns}")
        if self.config.reasoning_effort:
            print(f"Reasoning effort: {self.config.reasoning_effort}")
        print()

        # Initialize mnemory
        self._init_mnemory()
        self._save_run_config()

        # Stage 1: Ingest
        ingest_state: IngestState | None = None
        if "ingest" in self.config.stages:
            print("=" * 60)
            print("Stage 1: INGEST")
            print("=" * 60)
            ingest_state = run_ingest(conversations, self._memory_service, self.config)
            self._save_state("ingest", ingest_state)
            print(
                f"  Turns: {ingest_state.total_turns}, "
                f"Memories: {ingest_state.total_memories}, "
                f"Errors: {ingest_state.total_errors}, "
                f"Time: {ingest_state.elapsed_seconds:.1f}s"
            )
            print()
        else:
            ingest_state = self._load_state("ingest", IngestState)
            if ingest_state is None:
                raise RuntimeError(
                    "Ingest state not found. Run the 'ingest' stage first."
                )

        # Stage 2: Search
        search_state: SearchState | None = None
        if "search" in self.config.stages:
            print("=" * 60)
            print("Stage 2: SEARCH")
            print("=" * 60)
            search_state = run_search(
                conversations, self._memory_service, self.config, ingest_state
            )
            self._save_state("search", search_state)
            print(
                f"  Questions: {search_state.total_questions}, "
                f"Errors: {search_state.total_errors}, "
                f"Time: {search_state.elapsed_seconds:.1f}s"
            )
            print()
        else:
            search_state = self._load_state("search", SearchState)
            if search_state is None and (
                "answer" in self.config.stages or "evaluate" in self.config.stages
            ):
                raise RuntimeError(
                    "Search state not found. Run the 'search' stage first."
                )

        # Stage 3: Answer
        answer_state: AnswerState | None = None
        if "answer" in self.config.stages and search_state is not None:
            print("=" * 60)
            print("Stage 3: ANSWER")
            print("=" * 60)
            eval_model = self.config.resolve_eval_model(self._mnemory_config.llm.model)
            eval_llm = self._make_llm_client(eval_model)
            answer_state = run_answer(search_state, eval_llm, self.config, eval_model)
            self._save_state("answer", answer_state)
            print(
                f"  Questions: {answer_state.total_questions}, "
                f"Errors: {answer_state.total_errors}, "
                f"Time: {answer_state.elapsed_seconds:.1f}s"
            )
            print()
        else:
            answer_state = self._load_state("answer", AnswerState)
            if answer_state is None and "evaluate" in self.config.stages:
                raise RuntimeError(
                    "Answer state not found. Run the 'answer' stage first."
                )

        # Stage 4: Evaluate
        eval_state: EvalState | None = None
        if "evaluate" in self.config.stages and answer_state is not None:
            print("=" * 60)
            print("Stage 4: EVALUATE")
            print("=" * 60)
            judge_model = self.config.resolve_judge_model(
                self._mnemory_config.llm.model
            )
            judge_llm = self._make_llm_client(judge_model)
            eval_state = run_evaluate(answer_state, judge_llm, self.config, judge_model)
            self._save_state("evaluate", eval_state)
            print(
                f"  Evaluated: {eval_state.total_evaluated}, "
                f"Correct: {eval_state.total_correct}, "
                f"Errors: {eval_state.total_errors}, "
                f"Time: {eval_state.elapsed_seconds:.1f}s"
            )
            print()
        else:
            eval_state = self._load_state("evaluate", EvalState)

        # Report
        total_time = time.time() - start
        if eval_state:
            print_report(
                eval_state,
                self.config,
                self._mnemory_config,
                total_time,
                ingest_state,
            )
            # Save report
            report_path = self.run_dir / "report.json"
            with open(report_path, "w") as f:
                json.dump(eval_state.to_dict(), f, indent=2)
            print(f"\nResults saved to: {self.run_dir}")

    def _make_llm_client(self, model: str) -> Any:
        """Create an LLM client, reusing mnemory's if model matches."""
        from mnemory.config import LLMConfig
        from mnemory.llm import LLMClient

        if model == self._mnemory_config.llm.model:
            return self._memory_service._llm

        return LLMClient(
            LLMConfig(
                model=model,
                base_url=self._mnemory_config.llm.base_url,
                api_key=self._mnemory_config.llm.api_key,
            )
        )


def run_report(config: BenchmarkConfig) -> None:
    """Load and display results from a previous run."""
    if not config.resume:
        raise ValueError("No run directory specified")

    run_dir = Path(config.resume)
    eval_path = run_dir / "evaluate_state.json"
    config_path = run_dir / "config.json"
    ingest_path = run_dir / "ingest_state.json"

    if not eval_path.exists():
        raise FileNotFoundError(f"No evaluation results found in {run_dir}")

    with open(eval_path) as f:
        eval_state = EvalState.from_dict(json.load(f))

    # Load run config for display
    run_config = BenchmarkConfig()
    mnemory_config = None
    if config_path.exists():
        with open(config_path) as f:
            saved = json.load(f)
        run_config.search_method = saved.get("search_method", "search_memories")
        run_config.search_limit = saved.get("search_limit", 10)
        run_config.infer = saved.get("infer", True)
        run_config.categories = saved.get("categories", [1, 2, 3, 4])

    ingest_state = None
    if ingest_path.exists():
        with open(ingest_path) as f:
            ingest_state = IngestState.from_dict(json.load(f))

    print_report(eval_state, run_config, mnemory_config, 0.0, ingest_state)
