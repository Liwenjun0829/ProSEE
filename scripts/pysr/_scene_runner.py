"""Shared CLI helper for scenario-specific PySR entry points."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from run_pysr_experiment import run_pysr_experiment


def run_scene(config_name: str) -> None:
    parser = argparse.ArgumentParser(
        description=f"Run the PySR baseline for {config_name.removesuffix('.json')}"
    )
    parser.add_argument("--seed-count", type=int, default=10)
    args = parser.parse_args()
    if args.seed_count < 1:
        parser.error("--seed-count must be positive")
    config_path = PROJECT_ROOT / "configs" / "experiments" / config_name
    pysr_config_path = (
        PROJECT_ROOT / "configs" / "experiments" / "pysr" / config_name
    )
    run_pysr_experiment(config_path, pysr_config_path, seed_count=args.seed_count)
