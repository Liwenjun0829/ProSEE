"""Numerical fitting and piecewise-model estimation."""

from ._core import engine

FormulaEvaluator = engine.FormulaEvaluator
fit_coarse_piecewise_formula = engine.fit_coarse_piecewise_formula
fit_fine_piecewise_formula = engine.fit_fine_piecewise_formula
fit_formula_per_dataset = engine.fit_formula_per_dataset
fit_piecewise_formula = engine.fit_piecewise_formula
fit_with_restarts = engine.fit_with_restarts
search_piecewise_formula_pairs = engine.search_piecewise_formula_pairs

__all__ = [
    "FormulaEvaluator",
    "fit_coarse_piecewise_formula",
    "fit_fine_piecewise_formula",
    "fit_formula_per_dataset",
    "fit_piecewise_formula",
    "fit_with_restarts",
    "search_piecewise_formula_pairs",
]
