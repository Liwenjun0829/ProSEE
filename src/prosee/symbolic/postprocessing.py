"""Formula simplification and repeated-structure analysis."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from ._core import engine

simplify_pareto_front_expressions = engine.simplify_pareto_front_expressions


def analyze_repeated_structures(experiment_directory: str | Path) -> Any:
    """Generate the repeated-structure analysis."""
    module = importlib.import_module("canonicalize_repeated_structures")
    return module.main(Path(experiment_directory))


__all__ = ["analyze_repeated_structures", "simplify_pareto_front_expressions"]
