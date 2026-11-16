"""Symbolic formula discovery for propagation-model studies.

The public package is organized by research concern. The preserved numerical
kernel lives in :mod:`prosee.symbolic.core` and is loaded without algorithmic
changes so migration equivalence can be audited byte for byte.
"""

from .configuration import load_config
from .experiments import run, run_mode5_experiment

__all__ = ["load_config", "run", "run_mode5_experiment"]
