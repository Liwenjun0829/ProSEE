"""Configuration loading and validation for symbolic-regression studies."""

from ._core import engine

DEFAULT_CONFIG = engine.DEFAULT_CONFIG
DEFAULT_FORMULA_GENERATION_CONFIG = engine.DEFAULT_FORMULA_GENERATION_CONFIG
load_config = engine.load_config

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_FORMULA_GENERATION_CONFIG",
    "load_config",
]
