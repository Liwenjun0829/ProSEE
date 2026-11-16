"""End-to-end experiment orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._core import engine

run = engine.run


def run_mode5_experiment(
    config_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the paper experiment orchestrator in Mode 5."""
    return engine.run_mode4_paper_experiment(
        Path(config_path), experiment_mode=5, **kwargs
    )


__all__ = ["run", "run_mode5_experiment"]
