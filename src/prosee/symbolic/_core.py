"""Load the byte-preserved computational kernel.

The historical kernel uses flat-module imports and multiprocessing targets
whose module names are part of the execution contract. This loader keeps those
names stable while exposing a structured package API.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

CORE_DIRECTORY = Path(__file__).resolve().parent / "core"


def load_engine() -> ModuleType:
    """Return the preserved ProSEE formula-discovery pipeline module."""
    core_path = str(CORE_DIRECTORY)
    if core_path not in sys.path:
        sys.path.insert(0, core_path)
    return importlib.import_module("formula_discovery_pipeline")


engine = load_engine()
