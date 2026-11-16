"""Metrics, Pareto-front evaluation, and hypervolume reporting."""

from ._core import engine

calculate_hypervolume = engine._calculate_validation_hv
compute_joint_metrics = engine.compute_joint_metrics
compute_metrics = engine.compute_metrics
hypervolume_objectives = engine._validation_hv_objectives
save_validation_results = engine.save_validation_results

__all__ = [
    "calculate_hypervolume",
    "compute_joint_metrics",
    "compute_metrics",
    "hypervolume_objectives",
    "save_validation_results",
]
