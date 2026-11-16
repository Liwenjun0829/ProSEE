#!/usr/bin/env python3
"""Shared command-line support for the archived research datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from prosee.symbolic.core.formula_discovery_pipeline import (  # noqa: E402
    run_mode4_paper_experiment,
)


def run_experiments(config_names: list[str], description: str) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--n-jobs", type=int, default=24,
        help="parallel LM workers (default matches the archived runs)",
    )
    parser.add_argument(
        "--serial-lm", action="store_true",
        help="run LM serially, preserving the serial random-number call order",
    )
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--target-iterations", type=int, default=40)
    parser.add_argument("--retrain-interval", type=int, default=5)
    args = parser.parse_args()
    if args.n_jobs < 1:
        parser.error("--n-jobs must be positive")
    if args.seed_count < 1 or args.target_iterations < 1 or args.retrain_interval < 1:
        parser.error("seed count, target iterations, and retrain interval must be positive")

    for config_name in config_names:
        config_path = PROJECT_ROOT / "configs" / "experiments" / config_name
        run_mode4_paper_experiment(
            config_path,
            n_jobs=(1 if args.serial_lm else args.n_jobs),
            serial_lm=args.serial_lm,
            experiment_mode=5,
            retrain_interval=args.retrain_interval,
            seed_count=args.seed_count,
            target_iterations=args.target_iterations,
        )
