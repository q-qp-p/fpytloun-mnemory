"""CLI entry point for the LoCoMo benchmark.

Usage:
    python -m benchmarks.locomo download
    python -m benchmarks.locomo run [options]
    python -m benchmarks.locomo report <run_dir>
"""

from __future__ import annotations

import logging
import sys

from benchmarks.locomo.config import parse_args
from benchmarks.locomo.dataset import download_dataset
from benchmarks.locomo.runner import BenchmarkRunner, run_report


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        command, config = parse_args(argv)
    except SystemExit:
        return

    if config.verbose:
        logging.getLogger("benchmarks").setLevel(logging.DEBUG)
        logging.getLogger("mnemory").setLevel(logging.DEBUG)

    if command == "download":
        path = download_dataset(config.data_dir)
        print(f"Dataset ready: {path}")

    elif command == "run":
        runner = BenchmarkRunner(config)
        runner.run()

    elif command == "report":
        run_report(config)

    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
