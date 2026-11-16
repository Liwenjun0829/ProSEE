#!/usr/bin/env python3
"""Formula bases -> candidates -> binary formulas -> two-stage LM fitting."""
from __future__ import annotations

import argparse
from bisect import bisect_left
import copy
import csv
import hashlib
import itertools
import json
import logging
import math
import multiprocessing as mp
import os
import queue
import random
import re
import select
import signal
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Sequence

import numpy as np
import sympy as sp
from scipy.optimize import least_squares, minimize
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr

from generate_zero_sum_bases import generate_zero_sum_bases
from formula_structure_features import FORMULA_WRAPPERS
from structure_r2_ranker import (
    _diagnostic_features,
    save_feature_correlation_diagnostic,
    save_top_k_feature_correlation_diagnostic,
    train_and_rank,
    validate_ranking_dependencies,
)
from structure_screen.tree_search_ops import complexity as formula_complexity
from data_io import load_csv_data, resolve_path
from pareto_formula_evolution import (
    _has_log_abs_x1,
    merge_log_formulas,
    prune_insignificant_constants,
    filter_nondominated_initial_seeds,
    pareto_front,
    crowding_distances,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - progress remains optional
    def tqdm(iterable, **_kwargs):
        return iterable


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[3]
MEASURED_PATHLOSS_PROFILE = "measured_pathloss"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "formula_discovery" / "pipeline.json"
DEFAULT_FORMULA_GENERATION_CONFIG = (
    PROJECT_ROOT / "configs" / "formula_discovery" / "formula_generation.json"
)
DEFAULT_LM_MAX_WORKERS = 24
DEFAULT_MODE2_R2_WINDOW = 0.1
# Compatibility alias used by the separately imported PySR reporting helpers.
DEFAULT_PYSR_R2_WINDOW = DEFAULT_MODE2_R2_WINDOW
# Temporarily skip the potentially expensive final SymPy simplification of
# the Pareto expressions.  Formula generation, fitting, constant checks,
# selection, validation, and serialization remain unchanged.
ENABLE_FINAL_PARETO_SYMPY_SIMPLIFICATION = False
# Keep the PySR numerical baseline and its saved metrics, but do not generate
# plots that present it as a visual comparison with the tree/random searches.
ENABLE_PYSR_VISUAL_COMPARISON = False


def _is_measured_pathloss_profile(data_config_or_name: dict | str | None) -> bool:
    """Identify Maritime's measured-pathloss semantics independent of file format."""
    if isinstance(data_config_or_name, dict):
        return (
            data_config_or_name.get("profile") == MEASURED_PATHLOSS_PROFILE
            or data_config_or_name.get("mode") == "realdata_mat"
        )
    return str(data_config_or_name or "") in {
        MEASURED_PATHLOSS_PROFILE, "realdata_mat"
    }


def _configured_data_profile(data_config: dict) -> str:
    if _is_measured_pathloss_profile(data_config):
        return MEASURED_PATHLOSS_PROFILE
    return str(data_config.get("mode", "csv"))


# Keep the wrapper set explicit.  ``log_abs_sin`` and ``log_abs_cos`` are
# intentionally nested wrappers: they represent log(|sin(expr)|) and
# log(|cos(expr)|), respectively.
WRAPPERS = FORMULA_WRAPPERS
SYMBOLIC_CONSTANT_BASE_NAME = "c_symbolic_constant_0"

def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value

def _array_data_sha256(*arrays: np.ndarray) -> str:
    """Hash array values together with their dtype and shape."""
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()



def prompt_with_timeout(prompt: str, timeout: float = 20.0,
                        default: str = "y") -> str:
    """Read one terminal answer, returning the default after a timeout."""
    print(prompt, end="", flush=True)
    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            answer = sys.stdin.readline().strip().lower()
            if answer:
                return answer
    except (OSError, ValueError, KeyboardInterrupt):
        pass
    print(f"\n{timeout:g}s elapsed without input; defaulting to {'Yes' if default == 'y' else 'No'}.",
          flush=True)
    return default


def _result_r2(result: dict | None) -> float | None:
    """Return the comparable joint R² stored on a fit result."""
    if not result:
        return None
    value = result.get("pooled_r2", result.get("r2"))
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _mode2_requires_binary_formula(
    config: dict, pipeline_mode: int, feature_count: int,
) -> bool:
    """Whether the configured input has multiple features in a surrogate mode."""
    return (
        int(pipeline_mode) in (2, 3)
        and config.get("data", {}).get("mode", "csv") == "csv"
        and int(feature_count) > 1
    )


def _is_zero_only_constant_candidate(row: dict) -> bool:
    """Whether a row was produced by constant-zeroing tests."""
    assignments = row.get("constant_assignments")
    if not isinstance(assignments, dict) or not assignments:
        return False
    values = tuple(assignments.values())
    return any(value == 0 for value in values) and all(
        value in (None, "unchanged", 0) for value in values
    )


def _select_mode5_min_complexity_zero_bases(rows: Sequence[dict]) -> list[dict]:
    """Keep the simplest accepted pure-zeroing result per source formula.

    Complexity-layer boundaries use one accepted pure-zeroing structure per
    initial source: the one with the lowest structural complexity. R2 and
    formula ID provide deterministic tie breaks without changing acceptance.
    """
    selected: dict[str, dict] = {}
    for row in rows:
        source_id = str(row.get("source_formula_id") or row.get("formula_id", ""))
        value = row.get("complexity")
        if value is None:
            try:
                value = formula_complexity(sp.sympify(row["readable_expression"]))
            except (KeyError, sp.SympifyError, TypeError, ValueError, OverflowError):
                continue
        try:
            complexity_value = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        r2 = row.get("pooled_r2", row.get("mean_r2", row.get("r2")))
        try:
            r2_value = float(r2)
            if not math.isfinite(r2_value):
                r2_value = -math.inf
        except (TypeError, ValueError):
            r2_value = -math.inf
        candidate_key = (complexity_value, -r2_value, str(row.get("formula_id", "")))
        previous = selected.get(source_id)
        if previous is None or candidate_key < previous["_mode5_basis_key"]:
            item = dict(row)
            item["_mode5_basis_key"] = candidate_key
            selected[source_id] = item
    result = []
    for row in selected.values():
        row.pop("_mode5_basis_key", None)
        result.append(row)
    return sorted(result, key=lambda row: (
        int(row.get("complexity", formula_complexity(
            sp.sympify(row["readable_expression"])))),
        str(row.get("source_formula_id") or row.get("formula_id", "")),
    ))


def _mode2_initial_pareto_complexity_intervals(
    formula_library: Iterable[dict], pareto_rows: Iterable[dict],
    *, allow_empty_front: bool = False,
) -> list[dict]:
    """Count library formulas in adjacent Pareto complexity intervals.

    Intervals use ``(lower, upper]`` semantics.  Pareto rows with the same
    complexity are collapsed into one boundary, since they do not define a
    non-empty interval between them.
    """
    def row_complexity(row: dict) -> int | None:
        value = row.get("complexity")
        try:
            if value is not None:
                return int(value)
            return int(formula_complexity(sp.sympify(row["readable_expression"])))
        except (KeyError, TypeError, ValueError, OverflowError, sp.SympifyError):
            return None

    boundaries = sorted({
        complexity for row in pareto_rows
        if (complexity := row_complexity(row)) is not None
    })
    if not boundaries and allow_empty_front:
        return []
    library_complexities = [
        complexity for row in formula_library
        if (complexity := row_complexity(row)) is not None
    ]
    intervals = [
        {
            "lower_complexity": 0,
            "upper_complexity": boundaries[0],
            "formula_count": sum(0 < value <= boundaries[0] for value in library_complexities),
        }
    ] if boundaries else []
    intervals.extend([
        {
            "lower_complexity": lower,
            "upper_complexity": upper,
            "formula_count": sum(lower < value <= upper for value in library_complexities),
        }
        for lower, upper in zip(boundaries, boundaries[1:])
    ])
    # Keep the high-complexity tail in the Mode 3 sampling pool as well.  It
    # is intentionally bounded by the maximum complexity in the complete
    # formula library, rather than by an arbitrary configured maximum.
    library_max_complexity = max(library_complexities, default=None)
    if (library_max_complexity is not None
            and library_max_complexity > boundaries[-1]):
        intervals.append({
            "lower_complexity": boundaries[-1],
            "upper_complexity": library_max_complexity,
            "formula_count": sum(
                boundaries[-1] < value <= library_max_complexity
                for value in library_complexities
            ),
        })
    return intervals


def _formula_input_variable_count(expression: sp.Expr) -> int:
    """Count distinct input variables (x1, x2, ...) used by a formula."""
    return sum(
        1 for symbol in expression.free_symbols
        if str(symbol).startswith("x")
    )


def _has_required_formula_variables(
    expression: sp.Expr, required_count: int,
) -> bool:
    """Check that a formula uses at least the configured number of inputs."""
    return _formula_input_variable_count(expression) >= max(0, int(required_count))


def _result_not_worse(candidate: dict | None, baseline: dict | None,
                      tolerance: float = 1e-12) -> bool:
    """Whether candidate preserves the baseline joint R².

    R² is the contract exposed to users.  Loss is only a tie breaker because
    the pipeline can use different R² aggregations for multiple datasets.
    """
    if baseline is None:
        return True
    candidate_r2, baseline_r2 = _result_r2(candidate), _result_r2(baseline)
    if candidate_r2 is None:
        return False
    if baseline_r2 is not None and candidate_r2 < baseline_r2 - tolerance:
        return False
    return True


def _best_result_r2(results: Sequence[dict] | None) -> float | None:
    values = [_result_r2(row) for row in (results or [])]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def preprocess_constant_significance_results(
    results: Sequence[dict], fit_formula, config: dict, *,
    allow_single_variable: bool, feature_count: int, seed: int,
    n_jobs: int = 1, require_log_abs_x1: bool = False,
    apply_pareto_filter: bool = True,
) -> tuple[list[dict], list[dict], int, dict]:
    """Run constant-significance checks before final formula selection.

    ``apply_pareto_filter`` lets a search retain all constant-zeroing
    candidates for later processing when needed.
    """
    max_r2_drop = float(config.get("postprocessing", {}).get(
        "merge_log_max_r2_drop", 0.01
    ))
    initial_perturbation = float(config.get("fitting", {}).get(
        "zero_constant_initial_perturbation", 0.05
    ))
    lm_initializations = int(config.get("fitting", {}).get(
        "zero_constant_lm_initializations",
        config.get("fitting", {}).get("lm_initializations", 20),
    ))
    processed, checks, zeroed_count = prune_insignificant_constants(
        results, fit_formula, max_r2_drop, initial_perturbation,
        lm_initializations,
        allow_single_variable=allow_single_variable,
        test_reverse_order=True,
        feature_count=feature_count,
        seed=seed,
        n_jobs=n_jobs,
        require_log_abs_x1=require_log_abs_x1,
    )
    constraint_filtered_count = 0
    if require_log_abs_x1:
        constrained = []
        for row in processed:
            try:
                expression = sp.sympify(row["readable_expression"])
                compliant = _has_log_abs_x1(expression)
            except (KeyError, sp.SympifyError, TypeError, ValueError):
                compliant = False
            if compliant:
                constrained.append(row)
            else:
                constraint_filtered_count += 1
        processed = constrained
    if apply_pareto_filter:
        processed, selection_metadata = filter_nondominated_initial_seeds(
            processed,
            use_min_r2=allow_single_variable,
            use_piecewise_difference=False,
            use_pooled_r2=allow_single_variable,
        )
    else:
        selection_metadata = {
            "input_count": len(processed),
            "deduplicated_count": len(processed),
            "nondominated_count": len(processed),
            "objectives": [],
            "pareto_filter_applied": False,
        }
    metadata = {
        "max_r2_drop": max_r2_drop,
        "selection": selection_metadata,
        "orders": ["forward", "reverse"],
        "require_log_abs_x1": require_log_abs_x1,
        "constraint_filtered_count": constraint_filtered_count,
        "pareto_filter_applied": apply_pareto_filter,
    }
    return processed, checks, zeroed_count, metadata


def rank_lm_results_for_ablation(
    payload: dict, ranking_config: dict,
) -> tuple[list[dict], dict]:
    """Build a tree-ranker-compatible ranking directly from LM results."""
    rows = []
    for family, key in (("single", "single_best"),
                        ("combination", "combination_best")):
        for result in payload.get(key, []):
            r2 = _result_r2(result)
            if r2 is None:
                continue
            rows.append({
                "formula_id": result["formula_id"],
                "predicted_rank": 0,
                # The tree ablation answers a specific question: how does
                # the LM-fitted R² ranking perform without a tree model?
                "ranking_score": r2,
                "formula_family": family,
                "readable_expression": result["readable_expression"],
                "sympy_expression": result.get("sympy_expression"),
                "is_training_formula": True,
                "complexity_value": formula_complexity(
                    sp.sympify(result["readable_expression"])
                ),
                "lm_r2": r2,
            })
    rows.sort(key=lambda row: (-float(row["ranking_score"]), row["formula_id"]))
    for index, row in enumerate(rows, start=1):
        row["predicted_rank"] = index
    top_k = int(ranking_config.get("refit_top_k", 10))
    metadata = {
        "enabled": True,
        "ablation_enabled": True,
        "model_type": "lm_ablation",
        "training_samples": len(rows),
        "training_formula_ids": sorted({row["formula_id"] for row in rows}),
        "ranked_formulas": len(rows),
        "refit_top_k": min(top_k, len(rows)),
        "selection_basis": "direct_lm_r2",
        "validation": {"top_k_overlap": None},
    }
    return rows, metadata


def has_only_integer_exponents(expression: object) -> bool:
    """Return whether every explicit power has an integer exponent."""
    try:
        parsed = sp.sympify(str(expression))
    except (sp.SympifyError, TypeError, ValueError):
        return False
    for node in sp.preorder_traversal(parsed):
        if node.func is sp.Pow:
            exponent = node.exp
            if not exponent.is_Integer:
                return False
    return True


def formula_row_has_only_integer_exponents(row: dict) -> bool:
    """Check a normal formula row or a two-formula piecewise row."""
    nested_expressions = []
    for side in ("left_formula", "right_formula"):
        nested = row.get(side)
        if isinstance(nested, dict):
            nested_expressions.append(nested.get("readable_expression"))
    # A piecewise pair's top-level readable expression is a conditional text
    # string, not a standalone SymPy expression; inspect its two sides.
    expressions = (nested_expressions if nested_expressions
                   else [row.get("readable_expression")])
    return all(has_only_integer_exponents(expression) for expression in expressions)


def filter_gp_integer_exponents(results: Sequence[dict]) -> tuple[list[dict], dict]:
    """Filter final GP candidates to formulae with integer exponents."""
    kept, rejected = [], []
    for row in results:
        expression = row.get("readable_expression")
        if formula_row_has_only_integer_exponents(row):
            kept.append(row)
        else:
            rejected.append({
                "formula_id": row.get("formula_id"),
                "readable_expression": expression,
                "reason": "non_integer_exponent",
            })
    return kept, {
        "enabled": True,
        "input_count": len(results),
        "kept_count": len(kept),
        "rejected_count": len(rejected),
        "rejected": rejected,
    }


def filter_all_formula_results(payload: dict) -> dict:
    """Apply the exponent rule to every persisted formula-result collection."""
    keys = (
        "single_best", "combination_best", "structure_r2_ranking",
        "top_ranked_refit_results", "gp_pareto_front", "piecewise_formula_pairs",
    )
    report = {}
    for key in keys:
        rows = payload.get(key)
        if not isinstance(rows, list):
            continue
        filtered, detail = filter_gp_integer_exponents(rows)
        payload[key] = filtered
        report[key] = detail
    return report


def simplify_pareto_front_expressions(rows: list[dict]) -> dict:
    """Apply safe SymPy simplification to final Pareto expressions.

    Keep the fitted parameters and metrics only when simplification preserves
    the parameter-symbol set.  This avoids displaying an expression whose
    parameter vector no longer matches the stored fit.
    """
    changed = 0
    skipped = 0
    for row in rows:
        try:
            expression = sp.sympify(row["readable_expression"])
            parameter_symbols = {
                symbol for symbol in expression.free_symbols
                if str(symbol).startswith(("a", "c", "p"))
            }
            simplified = sp.simplify(expression)
            simplified_parameters = {
                symbol for symbol in simplified.free_symbols
                if str(symbol).startswith(("a", "c", "p"))
            }
            if simplified_parameters != parameter_symbols:
                skipped += 1
                continue
            if simplified != expression:
                row["readable_expression"] = str(simplified)
                row["sympy_expression"] = sp.srepr(simplified)
                row["complexity"] = int(formula_complexity(simplified))
                row["sympy_simplified"] = True
                changed += 1
        except (sp.SympifyError, TypeError, ValueError, OverflowError):
            skipped += 1
    return {"input_count": len(rows), "changed_count": changed,
            "skipped_count": skipped}


def archive_previous_results(output_dir: Path, history_path: Path) -> Path | None:
    """Archive the previous result, retaining only the newest snapshot."""
    if not history_path.is_file():
        return None
    archive_dir = output_dir / "history"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"formula_discovery_results_{stamp}.json"
    suffix = 1
    while archive_path.exists():
        archive_path = archive_dir / f"formula_discovery_results_{stamp}_{suffix}.json"
        suffix += 1
    shutil.copy2(history_path, archive_path)
    # Keep the new copy before deleting older archives, so an interruption
    # never leaves the history directory without a recoverable result.
    for previous_archive in archive_dir.glob("formula_discovery_results_*.json"):
        if previous_archive != archive_path:
            previous_archive.unlink()
    return archive_path


def _apply_pipeline_mode(config: dict, pipeline_mode: int) -> dict:
    """Apply the Mode 3 pipeline configuration."""
    try:
        pipeline_mode = int(pipeline_mode)
    except (TypeError, ValueError) as exc:
        raise ValueError("Only pipeline_mode=3 is supported") from exc
    if pipeline_mode not in (3, 4, 5, 6):
        raise ValueError("Only pipeline_mode=3, 4, 5, or 6 is supported")
    if pipeline_mode in (3, 4, 5, 6):
        ranking_config = config.setdefault("ranking", {})
        ranking_config["symbol_operator_stratified"] = False
        ranking_config["_allow_reduced_top_k"] = True
        mode2_config = config.setdefault("mode2", {})
        mode2_config.setdefault("initial_top_k", 15)
        mode2_config.setdefault("batch_size", 5)
        mode2_config.setdefault("validation_r2_window", DEFAULT_MODE2_R2_WINDOW)
        mode2_config["max_iterations"] = 40
        # Mode 2 does not impose a global exact-fit budget.  The search is
        # bounded by max_iterations and the finite unseen formula pool.
        mode2_config["max_exact_fits"] = None
        mode2_config.setdefault("r2_drop_threshold", 0.01)
        mode2_config["interval_layered_selection"] = pipeline_mode in (3, 5, 6)
    return config




def load_config(path: Path, pipeline_mode: int | None = 3) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    # These controls apply to constant checks and log-term refitting, not GP.
    # Migrate the former location so older user configs retain identical values.
    postprocessing = config.setdefault("postprocessing", {})
    legacy_gp = config.get("gp", {}) or {}
    for key, default in (
        ("merge_log_max_r2_drop", 0.01),
        ("log_merge_initial_perturbation", 0.05),
    ):
        if key not in postprocessing and key in legacy_gp:
            postprocessing[key] = legacy_gp[key]
        postprocessing.setdefault(key, default)
        legacy_gp.pop(key, None)
    if not legacy_gp:
        config.pop("gp", None)
    # PySR has an optional sidecar configuration. Main-pipeline configs do not
    # need to carry PySR settings, and omitting that sidecar disables the
    # baseline without changing the main search.
    if "pysr_baseline" in config:
        config = _apply_pysr_reproducibility_mode(config)
    # Formula and feature precomputation is mode independent. Callers can pass
    # None to validate the shared config without applying a search-mode preset.
    if pipeline_mode is not None:
        config = _apply_pipeline_mode(config, pipeline_mode)
    required = ("seed", "data", "fitting", "output")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Configuration is missing fields: {missing}")
    if config.get("stage", "all") not in ("generate", "fit", "all"):
        raise ValueError("stage must be generate, fit, or all")
    pysr_config = config.get("pysr_baseline", {}) or {}
    if pysr_config:
        for timeout_key, default_value in (
            ("pysr_iteration_timeout_seconds", 100),
            ("process_watchdog_iteration_timeout_seconds", 160),
            ("process_watchdog_startup_timeout_seconds", 1800),
        ):
            timeout_value = pysr_config.get(timeout_key, default_value)
            if timeout_value is not None and float(timeout_value) <= 0:
                raise ValueError(f"pysr_baseline.{timeout_key} must be positive")
    fitting_config = config["fitting"]
    fitting_optimizer = str(fitting_config.get("optimizer", "LM")).strip().upper()
    if fitting_optimizer not in ("LM", "BFGS"):
        raise ValueError("fitting.optimizer must be LM or BFGS")
    fitting_config["optimizer"] = fitting_optimizer
    if pysr_config:
        fitting_config["bfgs"] = {
            "optimizer_nrestarts": int(pysr_config.get("optimizer_nrestarts", 20)),
            "optimizer_iterations": int(pysr_config.get("optimizer_iterations", 8)),
            "optimizer_f_calls_limit": pysr_config.get("optimizer_f_calls_limit", 10000),
            "gtol": float(pysr_config.get("constant_refit_gtol", 1e-5)),
            "initialization_range": list(pysr_config.get(
                "constant_refit_initialization_range", [-10.0, 10.0]
            )),
        }
    else:
        fitting_config.setdefault("bfgs", {
            "optimizer_nrestarts": 20,
            "optimizer_iterations": 8,
            "optimizer_f_calls_limit": 10000,
            "gtol": 1e-5,
            "initialization_range": [-10.0, 10.0],
        })
    if fitting_config["bfgs"]["optimizer_nrestarts"] < 1:
        raise ValueError("pysr_baseline.optimizer_nrestarts must be at least 1")
    if fitting_config["bfgs"]["optimizer_iterations"] < 1:
        raise ValueError("pysr_baseline.optimizer_iterations must be at least 1")
    if (fitting_config["bfgs"]["optimizer_f_calls_limit"] is not None
            and int(fitting_config["bfgs"]["optimizer_f_calls_limit"]) < 1):
        raise ValueError("pysr_baseline.optimizer_f_calls_limit must be positive or null")
    bfgs_low, bfgs_high = map(float, fitting_config["bfgs"]["initialization_range"])
    if not bfgs_low < bfgs_high:
        raise ValueError("pysr_baseline.constant_refit_initialization_range must satisfy low < high")
    if (bool(pysr_config.get("enabled", False))
            and bool(pysr_config.get("reproducible", False))):
        if not bool(pysr_config.get("deterministic", False)):
            raise ValueError(
                "When pysr_baseline.reproducible=true, deterministic=true must be set"
            )
        if str(pysr_config.get("parallelism", "multithreading")) != "serial":
            raise ValueError(
                "When pysr_baseline.reproducible=true, parallelism='serial' must be set"
            )
        if pysr_config.get("timeout_in_seconds") is not None:
            raise ValueError(
                "When pysr_baseline.reproducible=true, timeout_in_seconds must be null; "
                "use niterations as the deterministic stopping condition"
            )
    data_mode = config["data"].get("mode", "csv")
    if data_mode not in ("csv", "realdata_mat"):
        raise ValueError("data.mode must be csv; realdata_mat is accepted for old configs")
    if _is_measured_pathloss_profile(config["data"]):
        realdata_cfg = config["data"].get("realdata", {})
        if data_mode == "realdata_mat" and not config["data"].get("csv_path"):
            raise ValueError("data.csv_path is required for realdata_mat mode (CSV input)")
        segmentation = realdata_cfg.get("segmentation", {})
        if segmentation.get("enabled", True):
            strategy = segmentation.get("strategy", "coarse_gp_fine")
            if strategy not in ("coarse_gp_fine", "legacy"):
                raise ValueError("data.realdata.segmentation.strategy must be coarse_gp_fine or legacy")
            if int(segmentation.get("min_segment_samples", 8)) < 2:
                raise ValueError("data.realdata.segmentation.min_segment_samples must be at least 2")
            if float(segmentation.get("warm_start_perturbation", 0.05)) < 0:
                raise ValueError("data.realdata.segmentation.warm_start_perturbation cannot be negative")
            if int(segmentation.get("candidate_stride", 1)) < 1:
                raise ValueError("data.realdata.segmentation.candidate_stride must be at least 1")
            if segmentation.get("independent_segment_fitting", segmentation.get("formula_pair_search", False)):
                if int(segmentation.get("segment_formula_top_k", 5)) < 1:
                    raise ValueError("data.realdata.segmentation.segment_formula_top_k must be at least 1")
                if int(segmentation.get("top_pair_results", 10)) < 1:
                    raise ValueError("data.realdata.segmentation.top_pair_results must be at least 1")
            if strategy == "coarse_gp_fine" and int(segmentation.get("coarse_split_count", 12)) < 1:
                raise ValueError("data.realdata.segmentation.coarse_split_count must be at least 1")
            if strategy == "coarse_gp_fine" and int(segmentation.get("fine_candidate_stride", 1)) < 1:
                raise ValueError("data.realdata.segmentation.fine_candidate_stride must be at least 1")
    # Validation and test data are created by splitting data.csv_files;
    # retain only output/ratio settings here for backward-compatible configs.
    validation = config.setdefault("validation", {}) or {}
    if validation.get("enabled", True) is False:
        raise ValueError("Validation is required; validation.enabled cannot be false")
    generation_config = json.loads(
        DEFAULT_FORMULA_GENERATION_CONFIG.read_text(encoding="utf-8")
    )
    candidates_config = config.get("candidates", generation_config["candidates"])
    combinations_config = config.get("combinations", generation_config["combinations"])
    ratio = float(candidates_config["two_base_ratio"])
    if not 0 <= ratio <= 1:
        raise ValueError("candidates.two_base_ratio must be in [0, 1]")
    max_combination_terms = int(combinations_config.get("max_terms", 2))
    if max_combination_terms < 2:
        raise ValueError("combinations.max_terms must be at least 2")
    if int(config["fitting"]["lm_initializations"]) < 1:
        raise ValueError("fitting.lm_initializations must be at least 1")
    patience = int(config["fitting"].get("patience", 10))
    if patience < 1:
        raise ValueError("fitting.patience must be at least 1")
    fitting_config = config["fitting"]
    if int(config["fitting"].get(
            "log_merge_lm_initializations",
            config["fitting"]["lm_initializations"],
    )) < 1:
        raise ValueError("fitting.log_merge_lm_initializations must be at least 1")
    fit_mode = config["fitting"].get("fit_mode", "refit")
    if fit_mode not in ("refit", "reuse_history"):
        raise ValueError("fitting.fit_mode must be refit or reuse_history")
    # Feature-correlation visual diagnostics are disabled globally for every
    # feature mode.  Keep the separate Top-K correlation threshold available
    # as a model-selection constraint, but never spend time generating the
    # diagnostic matrices/heatmaps.
    ranking_config = dict(config.get("ranking", {}))
    ranking_config["feature_correlation"] = {"enabled": False}
    if ranking_config.get("ablation_enabled", False) and not ranking_config.get("enabled", False):
        raise ValueError("ranking.ablation_enabled requires ranking.enabled")
    try:
        direct_fit_top_k = int(ranking_config.get("direct_fit_top_k", 5))
    except (TypeError, ValueError) as exc:
        raise ValueError("ranking.direct_fit_top_k must be a non-negative integer") from exc
    if direct_fit_top_k < 0:
        raise ValueError("ranking.direct_fit_top_k must be a non-negative integer")
    if not ranking_config.get("ablation_enabled", False):
        configured_tree_nodes_per_level = ranking_config.get(
            "tree_max_nodes_per_level", None
        )
        if (configured_tree_nodes_per_level is not None
                and int(configured_tree_nodes_per_level) < 1):
            raise ValueError("ranking.tree_max_nodes_per_level must be at least 1 or null")
    low, high = map(float, config["fitting"]["initialization_range"])
    if not low < high:
        raise ValueError("fitting.initialization_range must satisfy low < high")
    return config


def formula_record(expr: sp.Expr, **metadata) -> dict:
    return {
        **metadata,
        "sympy_expression": sp.srepr(expr),
        "readable_expression": str(expr),
    }


def formula_generation_signature(
    config: dict, feature_count: int, *, include_symbolic_constant_base: bool = False,
) -> str:
    """Hash the formula-library configuration, excluding runtime randomness."""
    candidates = dict(config.get("candidates", {}))
    # Preserve the signature of older artifacts where this behavior was the
    # implicit default, while allowing the generation config to state it.
    if candidates.get("all_single_base_forms") is False:
        candidates.pop("all_single_base_forms")
    descriptor = {
        "feature_count": int(feature_count),
        "bases": config.get("bases", {}),
        "candidates": candidates,
        "combinations": config.get("combinations", {}),
        "combination_generation": "exhaustive_canonical_v2",
        "include_symbolic_constant_base": bool(include_symbolic_constant_base),
        "symbolic_constant_base_name": (
            SYMBOLIC_CONSTANT_BASE_NAME if include_symbolic_constant_base else None
        ),
    }
    return hashlib.sha256(json.dumps(
        _json_value(descriptor), ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")).hexdigest()


def load_precomputed_formula_artifact(
    path: str | Path, config: dict, feature_count: int,
) -> dict:
    """Load the mandatory frozen formula library used by pipeline runs."""
    import joblib

    artifact_path = Path(path)
    if not artifact_path.is_absolute():
        artifact_path = ROOT / artifact_path
    if not artifact_path.is_file():
        raise FileNotFoundError(
            "Precomputed formula/feature artifact does not exist; run "
            f"precompute_formula_features.py：{artifact_path}"
        )
    try:
        artifact = joblib.load(artifact_path)
    except Exception as exc:
        raise RuntimeError(f"Could not read precomputed formula/feature artifact: {artifact_path}") from exc
    if not isinstance(artifact, dict) or int(artifact.get("version", -1)) != 2:
        raise ValueError(f"Unsupported precomputed formula/feature artifact version: {artifact_path}")
    if not artifact.get("formula_generation_signature"):
        raise ValueError("Precomputed formula/feature artifact is missing its formula-generation signature")
    # The formula library is intentionally generated from the independent
    # generation config.  Runtime pipeline settings must not invalidate it;
    # the artifact's own signature and formula/feature integrity checks remain
    # authoritative.
    for key in ("bases", "single_candidates", "binary_combinations"):
        if not isinstance(artifact.get(key), list):
            raise ValueError(f"Precomputed artifact is missing formula collection: {key}")
    artifact["artifact_path"] = str(artifact_path)
    return artifact


def select_precomputed_formula_artifact(config: dict) -> str | Path | None:
    """Select the frozen formula library from the generation config."""
    generation_config = json.loads(
        DEFAULT_FORMULA_GENERATION_CONFIG.read_text(encoding="utf-8")
    )
    outputs = generation_config.get("output", {})
    data_config = config.get("data", {})
    if _is_measured_pathloss_profile(data_config):
        legacy_path = config.get("precomputed_formula_feature_artifact_realdata_mat")
        if legacy_path:
            return legacy_path
        return outputs.get(
            "realdata_mat_artifact",
            config.get("precomputed_formula_feature_artifact_realdata_mat",
                       outputs.get("artifact", config.get("precomputed_formula_feature_artifact"))),
        )
    return config.get("precomputed_formula_feature_artifact", outputs.get("artifact"))


def _mode2_fit_task(task: tuple) -> tuple[list[dict], dict]:
    """Process-pool worker for the iterative surrogate-assisted search."""
    (record, X, y, fitting_config, dataset_groups, segmentation_config,
     piecewise, seed) = task
    rng = np.random.default_rng(int(seed))
    if piecewise:
        return fit_piecewise_formula(
            record, X, y, fitting_config, rng, segmentation_config
        )
    return fit_formula_per_dataset(
        record, X, y, fitting_config, rng, dataset_groups,
        fit_rng=rng,
    )


def _parallel_mode2_results(tasks: Sequence[tuple], max_workers: int | None = None):
    if not tasks:
        return []
    if max_workers is None:
        max_workers = min(
            len(tasks), DEFAULT_LM_MAX_WORKERS,
            max(1, (os.cpu_count() or 2) - 1),
        )
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_mode2_fit_task, tasks))


def _mode2_dataset_fit_task(task: tuple) -> tuple[list[dict], dict]:
    """Fit one Mode 2 formula on one dataset in one worker process."""
    (record, X, y, fitting_config, group, initial_values,
     initial_perturbation, lm_initializations, global_seed) = task
    dataset_name = str(group["name"])
    rng = np.random.default_rng(
        _stable_lm_seed(global_seed, dataset_name, record["id"])
    )
    start, end = int(group["start"]), int(group["end"])
    X_part, y_part = X[start:end], y[start:end]
    # fit_formula_per_dataset computes joint metrics against the arrays it is
    # given.  This worker handles exactly one dataset, so use a local group
    # index and the corresponding sliced arrays rather than the full payload.
    local_group = dict(group, start=0, end=len(y_part))
    # fit_formula_per_dataset computes joint metrics against the arrays it is
    # given.  This worker handles exactly one dataset, so use a local group
    # index and the corresponding sliced arrays rather than the full payload.
    local_group = dict(group, start=0, end=len(y_part))
    return fit_formula_per_dataset(
        record, X_part, y_part, fitting_config, rng, [local_group],
        initial_values_by_dataset={dataset_name: initial_values}
        if initial_values else None,
        initial_perturbation=initial_perturbation,
        warm_start_fraction=0.5,
        lm_initializations=lm_initializations,
        fit_rng=rng,
    )


def _combine_mode2_dataset_fits(
    record: dict, X: np.ndarray, y: np.ndarray, cfg: dict,
    dataset_groups: Sequence[dict], fitted: Sequence[tuple[list[dict], dict]],
) -> tuple[list[dict], dict]:
    """Combine independent Mode 2 dataset fits without refitting them."""
    all_attempts = []
    dataset_best = []
    predictions = []
    valid_parts = []
    expr = sp.sympify(record["readable_expression"])
    aggregation = str(cfg.get("joint_r2_aggregation", "minimum"))
    for group, (attempts, combined) in zip(dataset_groups, fitted):
        all_attempts.extend(attempts)
        rows = combined.get("dataset_fit_results", [])
        best = dict(rows[0]) if rows else {
            "dataset_name": group["name"], "parameters": None,
            "parameter_names": [], "r2": None, "mse": None, "rmse": None,
            "loss": None, "converged": False,
        }
        best.update(dataset_name=group["name"], dataset_path=group.get("path"))
        dataset_best.append(best)
        start, end = int(group["start"]), int(group["end"])
        X_part, y_part = X[start:end], y[start:end]
        if best.get("parameters") is None:
            predictions.append(np.full(len(y_part), np.nan))
            valid_parts.append(np.zeros(len(y_part), dtype=bool))
        else:
            evaluator = FormulaEvaluator(
                expr, X_part, float(cfg.get("max_abs_value", 1e150)),
                float(cfg.get("complex_tolerance", 1e-12)),
            )
            prediction, valid, _ = evaluator(best["parameters"])
            predictions.append(prediction)
            valid_parts.append(valid)
    prediction = np.concatenate(predictions) if predictions else np.array([])
    valid = np.concatenate(valid_parts) if valid_parts else np.array([], dtype=bool)
    metrics = compute_joint_metrics(y, prediction, valid, dataset_groups, aggregation)
    parameters_by_dataset = {
        row["dataset_name"]: row.get("parameters") for row in dataset_best
    }
    initialization_by_dataset = {
        row["dataset_name"]: row.get("initialization_index") for row in dataset_best
    }
    successful_by_dataset = {
        row["dataset_name"]: row.get("successful_initializations", 0)
        for row in dataset_best
    }
    failures = [f"{row['dataset_name']}: {row.get('failure_reason')}"
                for row in dataset_best if not row.get("converged")]
    validation_values = [float(row["validation_r2"]) for row in dataset_best
                         if row.get("validation_r2") is not None
                         and math.isfinite(float(row["validation_r2"]))]
    first_names = dataset_best[0].get("parameter_names", []) if dataset_best else []
    combined = {
        "formula_id": record["id"], "fit_scale": "raw",
        "parameters": parameters_by_dataset,
        "parameters_by_dataset": parameters_by_dataset,
        "parameter_names": first_names,
        "initialization_index": initialization_by_dataset,
        "initialization_index_by_dataset": initialization_by_dataset,
        "successful_initializations": successful_by_dataset,
        "successful_initializations_by_dataset": successful_by_dataset,
        "optimizer_success": all(row.get("optimizer_success", False) for row in dataset_best),
        "converged": all(row.get("converged", False) for row in dataset_best),
        "failure_reason": "; ".join(failures) if failures else None,
        "optimization_loss": sum(float(row.get("optimization_loss") or 0.0)
                                  for row in dataset_best),
        "all_datasets_valid": all(row.get("all_datasets_valid", False)
                                  for row in dataset_best),
        "dataset_fit_results": dataset_best,
        "expected_dataset_count": len(dataset_groups),
        "constants_shared_across_datasets": False,
        "validation_r2": (
            min(validation_values) if validation_values else None
        ),
        "validation_dataset_r2": {
            row["dataset_name"]: row.get("validation_r2")
            for row in dataset_best
        },
        **metrics,
    }
    return all_attempts, _json_value(combined)


def _parallel_mode2_dataset_results(
    tasks: Sequence[tuple], max_workers: int = 24,
) -> list[tuple[list[dict], dict]]:
    """Run Mode 2 formula/dataset jobs with a bounded process pool."""
    if not tasks:
        return []
    with ProcessPoolExecutor(max_workers=min(max_workers, len(tasks))) as executor:
        return list(executor.map(_mode2_dataset_fit_task, tasks))


def _mode2_result_row(record: dict, best: dict, *, score: float | None = None,
                      predicted_rank: int | None = None) -> dict:
    """Combine a structure record and an already computed LM result."""
    family = "combination" if record.get("combination_type") else "single"
    row = {
        "formula_id": record["id"],
        "formula_family": family,
        "readable_expression": record["readable_expression"],
        "sympy_expression": record.get("sympy_expression"),
        "complexity": int(formula_complexity(
            sp.sympify(record["readable_expression"])
        )),
        **best,
    }
    if score is not None:
        row["ranking_score"] = float(score)
    if predicted_rank is not None:
        row["predicted_rank"] = int(predicted_rank)
    return row


def _select_mode4_initial_complexity_layers(
    candidates: Sequence[dict], combinations: Sequence[dict],
    total_count: int, layer_count: int = 5,
) -> tuple[list[dict], list[dict]]:
    """Select an initial Mode 4 pool evenly across uniform complexity ranges."""
    records = [*candidates, *combinations]
    if not records or total_count <= 0:
        return [], []
    layer_count = max(1, int(layer_count))
    values = [(row, int(row.get("complexity", formula_complexity(
        sp.sympify(row["readable_expression"])))) ) for row in records]
    lower, upper = min(v for _, v in values), max(v for _, v in values)
    width = (upper - lower) / layer_count if upper > lower else 1.0
    quota = max(1, int(total_count) // layer_count)
    selected, metadata = [], []
    for layer in range(layer_count):
        lo = lower + layer * width
        hi = upper if layer == layer_count - 1 else lower + (layer + 1) * width
        pool = [row for row, value in values if (
            lo <= value <= hi if layer == layer_count - 1 else lo <= value < hi
        )]
        pool.sort(key=lambda row: (int(row.get("complexity", 0)), str(row.get("id", ""))))
        chosen = pool[:quota]
        selected.extend(chosen)
        metadata.append({"layer": layer + 1, "lower": lo, "upper": hi,
                         "available_count": len(pool), "selected_count": len(chosen),
                         "training_capacity": min(len(pool), quota),
                         "quota": quota})
    return selected, metadata


def _mode4_rank_crowding_labels(
    rows: Sequence[dict], is_realdata_mat: bool,
    use_piecewise_difference: bool = False,
) -> tuple[dict[str, float], dict[str, int]]:
    """Return scalar labels with Pareto rank taking precedence over crowding."""
    remaining = []
    primary = "pooled_r2" if is_realdata_mat else "mean_r2"
    for source in rows:
        row = dict(source)
        try:
            required = [float(row.get(primary)), float(row.get("r2_std")),
                        float(row.get("complexity", formula_complexity(
                            sp.sympify(row["readable_expression"]))))]
            if not all(math.isfinite(value) for value in required):
                continue
            row["complexity"] = required[-1]
        except (KeyError, TypeError, ValueError, OverflowError, sp.SympifyError):
            continue
        remaining.append(row)
    layers: list[list[dict]] = []
    while remaining:
        front, _ = filter_nondominated_initial_seeds(
            remaining, use_min_r2=False,
            use_piecewise_difference=use_piecewise_difference,
            use_pooled_r2=is_realdata_mat,
        )
        if not front:
            break
        layers.append(front)
        front_keys = {
            sp.srepr(sp.sympify(row["readable_expression"])) for row in front
        }
        remaining = [
            row for row in remaining
            if sp.srepr(sp.sympify(row["readable_expression"])) not in front_keys
        ]
    labels: dict[str, float] = {}
    ranks: dict[str, int] = {}
    layer_distances: list[dict[str, float]] = []
    secondary = "pooled_r2" if is_realdata_mat else "r2_std"
    for rank, layer in enumerate(layers, start=1):
        for row in layer:
            row["expression_key"] = sp.srepr(sp.sympify(row["readable_expression"]))
        distances = crowding_distances(
            layer, secondary, use_piecewise_difference=use_piecewise_difference
        )
        finite = [value for value in distances.values() if math.isfinite(value)]
        max_finite = max(finite, default=0.0)
        normalized = {
            key: float(max_finite + 1.0 if math.isinf(value) else value)
            for key, value in distances.items()
        }
        layer_distances.append(normalized)
    # A rank bonus dominates every within-layer crowding value.  The remaining
    # distance is normalized to [0, 1] per layer for stable tree labels.
    max_distance = max(
        (max(values.values(), default=0.0) for values in layer_distances),
        default=0.0,
    )
    rank_scale = max(1.0, max_distance + 1.0)
    for rank, (layer, distances) in enumerate(zip(layers, layer_distances), start=1):
        for row in layer:
            key = row["expression_key"]
            formula_id = str(row.get("formula_id"))
            labels[formula_id] = (len(layers) - rank) * rank_scale + distances.get(key, 0.0)
            ranks[formula_id] = rank
    return labels, ranks


def _select_mode3_complexity_interval_batch(
    rows: Sequence[dict], intervals: Sequence[dict],
) -> tuple[list[dict], list[dict]]:
    """Select one highest-ranked pending formula from each Pareto interval.

    Each row's complexity is calculated once and assigned to its interval;
    this avoids rescanning and re-parsing the complete pending pool for every
    interval.
    """
    def row_complexity(row: dict) -> int:
        value = row.get("complexity")
        return int(value) if value is not None else int(formula_complexity(
            sp.sympify(row["readable_expression"])))

    selected, metadata = [], []
    normalized_intervals = [
        (int(interval["lower_complexity"]), int(interval["upper_complexity"]))
        for interval in intervals
    ]
    buckets = [[] for _ in normalized_intervals]
    upper_bounds = [upper for _lower, upper in normalized_intervals]
    for row in rows:
        try:
            complexity = row_complexity(row)
        except (KeyError, sp.SympifyError, TypeError, ValueError, OverflowError):
            continue
        bucket_index = bisect_left(upper_bounds, complexity)
        if bucket_index >= len(normalized_intervals):
            continue
        lower, upper = normalized_intervals[bucket_index]
        if lower < complexity <= upper:
            buckets[bucket_index].append(row)

    for index, ((lower, upper), eligible) in enumerate(
        zip(normalized_intervals, buckets), start=1
    ):
        if not eligible:
            metadata.append({
                "layer": index, "lower_complexity": lower,
                "upper_complexity": upper, "available_count": 0,
                "selected_count": 0,
                "selected_formula_id": None,
            })
            continue
        representative = min(
            eligible,
            key=lambda row: (
                row.get("predicted_rank", 10**9),
                str(row.get("formula_id", "")),
            ),
        )
        representative = dict(representative)
        representative["complexity_layer"] = index
        representative["complexity_interval"] = [lower, upper]
        representative["complexity_value"] = row_complexity(representative)
        selected.append(representative)
        metadata.append({
            "layer": index, "lower_complexity": lower,
            "upper_complexity": upper, "available_count": len(eligible),
            "selected_count": 1,
            "selected_formula_id": representative.get("formula_id"),
        })
    selected.sort(key=lambda row: row.get("predicted_rank", 10**9))
    return selected, metadata


def _select_ranked_complexity_layer_batch(
    rows: Sequence[dict], intervals: Sequence[dict],
) -> tuple[list[dict], list[dict]]:
    """Select the highest-ranked pending formula from every Pareto interval."""
    layer_bounds = [(
        float(item["lower_complexity"]),
        float(item["upper_complexity"]),
    ) for item in intervals]
    if not layer_bounds:
        return [], []
    layer_count = len(layer_bounds)
    buckets = [[] for _ in range(layer_count)]
    for row in rows:
        try:
            value = row.get("complexity")
            if value is None:
                value = formula_complexity(sp.sympify(row["readable_expression"]))
            value = int(value)
        except (KeyError, sp.SympifyError, TypeError, ValueError, OverflowError):
            continue
        for layer, (lower, upper) in enumerate(layer_bounds):
            in_layer = (lower < value <= upper)
            if in_layer:
                buckets[layer].append(row)
                break

    selected, metadata = [], []
    for index, bucket in enumerate(buckets):
        lo, hi = layer_bounds[index]
        bucket.sort(key=lambda row: (
            row.get("predicted_rank", 10**9), str(row.get("formula_id", ""))))
        chosen = bucket[:1]
        selected.extend(chosen)
        metadata.append({
            "layer": index + 1,
            "lower_complexity": lo,
            "upper_complexity": hi,
            "available_count": len(bucket),
        })
    selected.sort(key=lambda row: row.get("predicted_rank", 10**9))
    return selected, metadata


def _mode3_complexity_layer_spearman(
    ranking: Sequence[dict], fitted_rows: Sequence[dict],
    intervals: Sequence[dict],
) -> list[dict]:
    """Compare initial surrogate ranks with fitted mean-R² ranks per interval."""
    ranking_by_id = {
        str(row.get("formula_id")): row for row in ranking
        if row.get("formula_id") is not None
    }
    normalized_intervals = [
        (int(item["lower_complexity"]), int(item["upper_complexity"]))
        for item in intervals
    ]
    rows_by_layer = [[] for _ in normalized_intervals]
    upper_bounds = [upper for _lower, upper in normalized_intervals]
    for row in fitted_rows:
        formula_id = str(row.get("formula_id"))
        ranking_row = ranking_by_id.get(formula_id)
        if ranking_row is None:
            continue
        try:
            complexity_value = row.get("complexity")
            if complexity_value is None:
                complexity_value = formula_complexity(
                    sp.sympify(row["readable_expression"])
                )
            complexity = int(complexity_value)
            mean_r2 = float(_mode2_mean_r2(row))
            predicted_rank = float(ranking_row["predicted_rank"])
        except (KeyError, TypeError, ValueError, OverflowError, sp.SympifyError):
            continue
        if not math.isfinite(mean_r2) or not math.isfinite(predicted_rank):
            continue
        layer_index = bisect_left(upper_bounds, complexity)
        if layer_index >= len(normalized_intervals):
            continue
        lower, upper = normalized_intervals[layer_index]
        if lower < complexity <= upper:
            rows_by_layer[layer_index].append((predicted_rank, mean_r2, formula_id))

    results = []
    for index, ((lower, upper), rows) in enumerate(
        zip(normalized_intervals, rows_by_layer), start=1
    ):
        coefficient = None
        if len(rows) >= 2:
            # Both vectors use ascending rank numbers: lower predicted rank
            # and higher fitted mean-R² are better, so rank the latter by
            # ``-mean_r2``.  rankdata gives tied fitted scores their average
            # rank, matching the standard Spearman definition.
            coefficient = float(spearmanr(
                [item[0] for item in rows],
                rankdata([-item[1] for item in rows], method="average"),
            ).statistic)
            if not math.isfinite(coefficient):
                coefficient = None
        results.append({
            "layer": index,
            "lower_complexity": lower,
            "upper_complexity": upper,
            "fitted_count": len(rows),
            "spearman_rho": coefficient,
        })
    return results


def _mode2_mean_r2(row: dict | None) -> float | None:
    """Return Mode 2's mean R² only when every dataset fit is valid."""
    if not row:
        return None
    dataset_rows = row.get("dataset_fit_results") or row.get("dataset_metrics")
    values = []
    for dataset_row in dataset_rows or []:
        value = dataset_row.get("r2") if isinstance(dataset_row, dict) else None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        values.append(value)
    if values:
        expected_count = row.get("expected_dataset_count")
        if expected_count is not None and len(values) != int(expected_count):
            return None
        return float(np.mean(values))
    value = row.get("mean_r2")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _annotate_mode2_mean_r2(row: dict) -> dict:
    value = _mode2_mean_r2(row)
    if value is not None:
        row["mean_r2"] = value
    return row


def _annotate_mode2_training_rows(rows: Sequence[dict]) -> None:
    """Use one mean-R² label per formula across all configured datasets."""
    grouped: dict[str, list[float]] = {}
    for row in rows:
        formula_id = row.get("formula_id")
        if formula_id is None:
            continue
        dataset_rows = row.get("dataset_fit_results") or row.get("dataset_metrics")
        values = []
        if dataset_rows:
            values = [float(item["r2"]) for item in dataset_rows
                      if isinstance(item, dict) and item.get("r2") is not None
                      and math.isfinite(float(item["r2"]))]
        if not values and row.get("r2") is not None:
            value = float(row["r2"])
            if math.isfinite(value):
                values = [value]
        grouped.setdefault(str(formula_id), []).extend(values)
    for row in rows:
        values = grouped.get(str(row.get("formula_id")), [])
        if values:
            row["mean_r2"] = float(np.mean(values))


def _mode2_pareto_rows(rows: Sequence[dict], config: dict) -> list[dict]:
    """Apply the same objective convention as the normal GP Pareto stage."""
    is_realdata_mat = _is_measured_pathloss_profile(config.get("data", {}))
    prepared = []
    for row in rows:
        if _mode2_mean_r2(row) is None:
            continue
        values = [float(item.get("r2")) for item in row.get(
            "dataset_fit_results", []) if item.get("r2") is not None
        ]
        if not values:
            values = [float(_mode2_mean_r2(row))]
        item = dict(row)
        item["mean_r2"] = float(np.mean(values))
        item["r2_std"] = float(np.std(values))
        item["min_r2"] = float(np.min(values))
        item["pooled_r2"] = float(row.get("pooled_r2", _mode2_mean_r2(row)))
        item["expression_key"] = sp.srepr(sp.sympify(item["readable_expression"]))
        item["complexity"] = int(formula_complexity(
            sp.sympify(item["readable_expression"])
        ))
        prepared.append(item)
    if not prepared:
        return []
    # A formula is eligible for the Pareto pool only when its active primary
    # R² objective is within the configured Mode 2 quality window of the best
    # current row.  realdata_mat uses pooled R²; other modes retain mean R².
    primary_r2_key = "pooled_r2" if is_realdata_mat else "mean_r2"
    highest_primary_r2 = max(item[primary_r2_key] for item in prepared)
    prepared = [item for item in prepared
                if item[primary_r2_key] >= highest_primary_r2 - 0.1]
    if not prepared:
        return []
    # Mode 2 keeps the legacy mean-R² convention except for realdata_mat,
    # where pooled R² is the primary fit-quality objective.  The real-data
    # breakpoint difference remains an additional objective.
    secondary = "pooled_r2" if is_realdata_mat else "r2_std"
    use_piecewise_difference = is_realdata_mat
    front = pareto_front(prepared, secondary, use_piecewise_difference)
    distances = crowding_distances(front, secondary, use_piecewise_difference)
    output = []
    for rank, item in enumerate(front, 1):
        item = dict(item)
        item["pareto_rank"] = rank
        item["crowding_distance"] = distances.get(item["expression_key"], 0.0)
        item.pop("expression_key", None)
        output.append(item)
    return output


def run_iterative_surrogate_mode2(
    payload: dict, candidates: Sequence[dict], combinations: Sequence[dict],
    X: np.ndarray, y: np.ndarray, dataset_groups: Sequence[dict], config: dict,
    ranking_config: dict, output_dir: Path, seed: int, n_jobs: int,
    segmentation_config: dict, piecewise_enabled: bool,
    is_realdata_mat: bool, history_payload: dict | None = None,
) -> dict:
    """Run the new Mode 2 without changing the legacy pipeline branches."""
    fitting = config["fitting"]
    required_formula_variable_count = X.shape[1]
    fitting_count = int(fitting.get("single_sample_count", 40))
    combination_count = int(fitting.get("combination_sample_count", 120))
    batch_size = max(1, int(config.get("gp", {}).get("surrogate_batch_size", 5)))
    max_iterations = max(1, int(config.get("gp", {}).get(
        "surrogate_max_iterations", 50
    )))
    max_budget = max(1, int(config.get("gp", {}).get(
        "surrogate_max_exact_fits", max_iterations * batch_size
    )))
    records = {row["id"]: row for row in [*candidates, *combinations]}
    expression_records = {
        sp.srepr(sp.sympify(row["readable_expression"])): row
        for row in records.values()
    }
    exact: dict[str, dict] = {}
    exact_by_expression: dict[str, dict] = {}
    all_attempts: list[dict] = []

    def fit_records(selected: Sequence[dict]) -> None:
        selected = [row for row in selected if row["id"] not in exact]
        if not selected:
            return
        tasks = [(
            row, X, y, fitting, dataset_groups, segmentation_config,
            piecewise_enabled, _stable_lm_seed(seed, "mode2", row["id"]),
        ) for row in selected]
        if n_jobs != 1 and len(tasks) > 1:
            results = _parallel_mode2_results(tasks, max_workers=None if n_jobs < 0 else n_jobs)
        else:
            results = [_mode2_fit_task(task) for task in tasks]
        for record, (attempts, best) in zip(selected, results):
            all_attempts.extend(attempts)
            row = _mode2_result_row(record, best)
            exact[record["id"]] = row
            exact_by_expression[sp.srepr(sp.sympify(record["readable_expression"]))] = row

    if history_payload is not None:
        for key in ("single_best", "combination_best"):
            for old in history_payload.get(key, []):
                record = records.get(old.get("formula_id"))
                if record is not None:
                    row = _mode2_result_row(record, old)
                    exact[record["id"]] = row
                    exact_by_expression[sp.srepr(sp.sympify(record["readable_expression"]))] = row
    else:
        rng = random.Random(seed)
        singles = list(candidates)
        combos = list(combinations)
        rng.shuffle(singles)
        rng.shuffle(combos)
        fit_records([*singles[:fitting_count], *combos[:combination_count]])

    training_rows = list(exact.values())
    payload["single_best"] = [row for row in training_rows if row["formula_family"] == "single"]
    payload["combination_best"] = [row for row in training_rows if row["formula_family"] == "combination"]
    payload["single_initializations"] = all_attempts
    payload["metadata"].update(
        mode2="iterative_surrogate_assisted_search",
        mode2_initial_training_counts={"single": fitting_count, "combination": combination_count},
        mode2_batch_size=batch_size, mode2_max_iterations=max_iterations,
        mode2_max_exact_fits=max_budget,
    )

    ranking_config = dict(ranking_config)
    ranking_config.update(enabled=True, ablation_enabled=False)
    ranking, ranking_meta = train_and_rank(payload, ranking_config, seed, output_dir)
    payload["structure_r2_ranking"] = ranking
    payload["ranking_metadata"] = ranking_meta
    payload["top_ranked_refit_results"] = []

    # Top-K is intentionally a cache read.  Run the normal significance test
    # on those rows; its LM calls are the only expected extra fits here.
    top_ids = [row["formula_id"] for row in ranking[:int(config.get("gp", {}).get("top_k", 15))]]
    top_rows = [exact[item] for item in top_ids if item in exact]
    def fit_cached_or_new(record, **kwargs):
        key = sp.srepr(sp.sympify(record["readable_expression"]))
        if key in exact_by_expression:
            return exact_by_expression[key]
        _attempts, best = _mode2_fit_task((
            record, X, y, fitting, dataset_groups, segmentation_config,
            piecewise_enabled, _stable_lm_seed(seed, "mode2-significance", record["id"]),
        ))
        row = _mode2_result_row(record, best)
        exact_by_expression[key] = row
        return row
    processed, checks, zeroed, sig_meta = preprocess_constant_significance_results(
        top_rows, fit_cached_or_new, config, allow_single_variable=is_realdata_mat,
        feature_count=X.shape[1], seed=seed, n_jobs=1,
        require_log_abs_x1=bool(config.get("gp", {}).get("require_log_abs_x1_initial", False)),
        apply_pareto_filter=False,
    )
    payload["top_ranked_refit_results"] = processed
    payload["top_ranked_constant_significance_checks"] = checks
    pareto_pool = list(exact.values())
    payload["gp_pareto_front"] = _mode2_pareto_rows(pareto_pool, config)
    initial_front_keys = {
        sp.srepr(sp.sympify(row["readable_expression"]))
        for row in payload["gp_pareto_front"]
    }
    exact_budget = 0
    iteration = 0
    global_formula_change_count = 0
    cumulative_global_change_keys = set()
    while iteration < max_iterations and exact_budget < max_budget:
        iteration += 1
        ranking, ranking_meta = train_and_rank(payload, ranking_config, seed + iteration, output_dir)
        unseen = [row for row in ranking if row["formula_id"] not in exact]
        selected = [records[row["formula_id"]] for row in unseen[:batch_size] if row["formula_id"] in records]
        if not selected:
            break
        fit_records(selected)
        exact_budget += len(selected)
        # Enumerate every parameter-zero subset for each newly evaluated source.
        for source in selected:
            source_row = exact[source["id"]]
            expr = sp.sympify(source["readable_expression"])
            params = sorted(
                [s for s in expr.free_symbols if str(s).startswith(("a", "c", "p"))],
                key=str,
            )
            source_r2 = _result_r2(source_row)
            if source_r2 is None:
                continue
            zero_candidates = []
            for count in range(1, len(params) + 1):
                for subset in itertools.combinations(params, count):
                    candidate_expr = expr.subs({symbol: 0 for symbol in subset})
                    if (required_formula_variable_count
                            and not _has_required_formula_variables(
                                candidate_expr, required_formula_variable_count
                            )):
                        # Do not fit a zeroed candidate that uses fewer input
                        # variables than the configured dataset provides.
                        continue
                    key = sp.srepr(candidate_expr)
                    cached = exact_by_expression.get(key)
                    if cached is None:
                        candidate_record = formula_record(
                            candidate_expr, id=f"{source['id']}_zero_{count}_{hashlib.blake2b(key.encode(), digest_size=5).hexdigest()}",
                            structure_type=source.get("structure_type", "mode2_zeroed"),
                        )
                        cached = fit_cached_or_new(candidate_record)
                    candidate_r2 = _result_r2(cached)
                    if candidate_r2 is not None and source_r2 - candidate_r2 <= float(
                        config.get("gp", {}).get(
                            "r2_drop_threshold",
                            config.get("postprocessing", {}).get(
                                "merge_log_max_r2_drop", 0.01
                            ),
                        )
                    ):
                        zero_candidates.append(cached)
            if zero_candidates:
                best_zero = min(zero_candidates, key=lambda row: (int(row.get("complexity", formula_complexity(sp.sympify(row["readable_expression"])))), -(_result_r2(row) or -1e100)))
                exact_by_expression[sp.srepr(sp.sympify(best_zero["readable_expression"]))] = best_zero
        payload["single_best"] = [row for row in exact.values() if row["formula_family"] == "single"]
        payload["combination_best"] = [row for row in exact.values() if row["formula_family"] == "combination"]
        new_front = _mode2_pareto_rows(list(exact.values()), config)
        new_keys = {
            sp.srepr(sp.sympify(row["readable_expression"]))
            for row in new_front
        }
        # Count every formula that has ever differed from the initial front.
        # Keep this historical set so the diagnostic cannot decrease when a
        # later Pareto front happens to move back toward the initial front.
        cumulative_global_change_keys.update(
            initial_front_keys.symmetric_difference(new_keys)
        )
        global_formula_change_count = len(cumulative_global_change_keys)
        payload["gp_pareto_front"] = new_front
        payload["structure_r2_ranking"] = ranking
        payload["ranking_metadata"] = ranking_meta
        payload["metadata"].update(mode2_iteration=iteration, mode2_exact_fit_budget_used=exact_budget,
                                    mode2_global_formula_change_count=global_formula_change_count)
        pareto_text = f"Pareto={len(new_front)}, change from initial front={global_formula_change_count}"
        print(f"Mode 2  round  {iteration} : exact fits  {len(selected)} items, cumulative budget {exact_budget}/{max_budget}, {pareto_text}", flush=True)
        save_checkpoint(output_dir, payload, f"mode2_iteration_{iteration}")
    payload["metadata"].update(mode2_iterations=iteration, mode2_stop_reason=(
        "exact_fit_budget" if exact_budget >= max_budget else "no_unseen_candidates"),
        exact_evaluated_formula_count=len(exact))
    payload["gp_metadata"] = {"enabled": False, "mode": "iterative_surrogate_assisted_search",
                               "iterations": iteration, "exact_fit_budget_used": exact_budget}
    save_outputs(output_dir, payload)
    return payload


def build_base_records(
    config: dict, feature_count: int, *, include_symbolic_constant_base: bool = False,
) -> list[dict]:
    configured_arities = config["bases"].get("arities")
    if configured_arities in (None, "auto"):
        configured_arities = [feature_count]
    max_abs = int(config["bases"].get("max_abs_exponent", 1))
    allow_binary_reciprocals = bool(
        config["bases"].get("allow_binary_reciprocals", True)
    )
    records = []
    for arity in configured_arities:
        arity = int(arity)
        if arity > feature_count:
            logging.warning("Skipping %d-ary bases: data has only %d features", arity, feature_count)
            continue
        for expr in generate_zero_sum_bases(
            arity,
            max_abs,
            allow_binary_reciprocals=allow_binary_reciprocals,
        ):
            records.append(formula_record(
                expr,
                id=f"base_{len(records):06d}",
                structure_type="projected_zero_sum_monomial",
                source_arity=4,
                arity=arity,
                projection_substitutions={f"x{i + 1}": 1 for i in range(arity, 4)},
                exponents=[int(expr.as_powers_dict().get(sp.Symbol(f"x{i + 1}"), 0))
                           for i in range(arity)],
                variables=[f"x{i + 1}" for i in range(arity)],
            ))
    if include_symbolic_constant_base:
        # This is a fitted fitted symbolic constant that acts as an additional
        # base. Its c_ namespace is disjoint from generated c1/c2/... weights;
        # combination construction scopes it per term in _renamed_candidate.
        constant = sp.Symbol(SYMBOLIC_CONSTANT_BASE_NAME)
        records.append(formula_record(
            constant,
            id=f"base_{len(records):06d}",
            structure_type="symbolic_constant_base",
            source_arity=0,
            arity=0,
            projection_substitutions={},
            exponents=[],
            variables=[],
        ))
    if not records:
        raise ValueError("No nontrivial zero-sum bases were generated for the current data arity")
    return records


def apply_wrapper(expr: sp.Expr, wrapper: str) -> sp.Expr:
    if wrapper == "identity":
        return expr
    if wrapper == "log_abs":
        return sp.log(sp.Abs(expr))
    if wrapper == "sin":
        return sp.sin(expr)
    if wrapper == "cos":
        return sp.cos(expr)
    if wrapper == "log_abs_sin":
        return sp.log(sp.Abs(sp.sin(expr)))
    if wrapper == "log_abs_cos":
        return sp.log(sp.Abs(sp.cos(expr)))
    raise ValueError(f"Unsupported outer wrapper: {wrapper}")


def _collapse_symbolic_constant_products(expr: sp.Expr) -> sp.Expr:
    """Reparameterize redundant products introduced by the constant base.

    Pure constant expressions are represented by one fitted scalar. In a
    formula that also depends on x variables, only the product of an ordinary
    fitted coefficient and the symbolic constant base is replaced by one
    fitted parameter; the surrounding formula structure is retained.
    """
    constant = sp.Symbol(SYMBOLIC_CONSTANT_BASE_NAME)
    constant_parameters = {
        symbol for symbol in expr.free_symbols
        if symbol == constant
        or str(symbol).startswith(("cconst_product_", "cconst_output_"))
    }
    if not constant_parameters:
        return expr
    has_input = any(str(symbol).startswith("x") for symbol in expr.free_symbols)
    if not has_input:
        return sp.Symbol("cconst_output_0")

    result = expr
    product_index = 0
    # Replace the smallest matching product first, so a surrounding amplitude
    # coefficient remains separate from the inner constant-base product.
    def replace_direct_products(node: sp.Expr) -> sp.Expr:
        nonlocal product_index
        if not node.args:
            return node

        # Recurse first, then inspect only the direct factors of this node.
        # ``free_symbols`` includes symbols nested inside function arguments;
        # therefore ``a2*log(x + cconst)`` must not be treated as a direct
        # coefficient-times-constant product.
        rebuilt_args = tuple(replace_direct_products(arg) for arg in node.args)
        if rebuilt_args != node.args:
            node = node.func(*rebuilt_args)
        if node.func is not sp.Mul:
            return node

        direct_constant_factors = [
            factor for factor in node.args
            if factor.is_Symbol and factor in constant_parameters
        ]
        direct_coefficient_factors = [
            factor for factor in node.args
            if factor.is_Symbol
            and factor not in constant_parameters
            and str(factor).startswith(("a", "c", "p"))
        ]
        if not direct_constant_factors or not direct_coefficient_factors:
            return node

        selected = set(direct_constant_factors + direct_coefficient_factors)
        residual = [factor for factor in node.args if factor not in selected]
        parameter = sp.Symbol(f"cconst_product_{product_index}")
        product_index += 1
        return sp.Mul(parameter, *residual)

    return replace_direct_products(result)


def _sample_unique(
    count: int, builder, rng: random.Random, max_attempts: int, seen: set[str]
) -> list[dict]:
    records = []
    for _ in range(max_attempts):
        if len(records) >= count:
            break
        record, key = builder()
        if key in seen:
            continue
        seen.add(key)
        records.append(record)
    if len(records) < count:
        logging.warning("Candidate space too small: requested %d, generated %d", count, len(records))
    return records


def build_candidate_records(base_records: Sequence[dict], config: dict, rng: random.Random) -> list[dict]:
    cfg = config["candidates"]
    wrappers = list(cfg.get("wrappers", WRAPPERS))
    unsupported = sorted(set(wrappers) - set(WRAPPERS))
    if unsupported:
        raise ValueError(f"Unsupported candidates.wrappers: {unsupported}")
    base_exprs = {row["id"]: sp.sympify(row["readable_expression"]) for row in base_records}
    symbolic_constant_base_ids = {
        row["id"] for row in base_records
        if row.get("structure_type") in {
            "symbolic_constant_base", "symbolic_dimensionless_constant"
        }
    }

    def allowed_wrappers(chosen_base_ids: Sequence[str]) -> list[str]:
        """Reject candidates whose only base is the symbolic constant.

        A symbolic constant may participate in a nonlinear expression with a
        variable/base, such as ``log(c1*x + c2)``.  It must not
        be the sole base, which would create a constant-only candidate such as
        ``c``, ``sin(c)`` or ``log_abs(c)``.  Existing fitted coefficients such
        as ``c1``/``c2`` are unaffected.
        """
        if (chosen_base_ids
                and all(base_id in symbolic_constant_base_ids
                        for base_id in chosen_base_ids)):
            return []
        if any(base_id in symbolic_constant_base_ids for base_id in chosen_base_ids):
            return [wrapper for wrapper in wrappers if wrapper != "identity"]
        return wrappers
    ids = list(base_exprs)

    # Deterministic exhaustive mode used by the neural-model pilot: enumerate
    # every single-base primitive with every configured wrapper.  This avoids
    # the usual random max_count/two_base_ratio sampling path.
    if cfg.get("all_single_base_forms", False):
        records = []
        for base_id in ids:
            for wrapper in allowed_wrappers([base_id]):
                inner = sp.Symbol("c1") * base_exprs[base_id]
                expr = _collapse_symbolic_constant_products(
                    sp.Symbol("a0") * apply_wrapper(inner, wrapper)
                )
                records.append(formula_record(
                    expr,
                    id=f"candidate_{len(records):06d}",
                    structure_type="single_candidate", base_count=1,
                    base_ids=[base_id], bases=[str(base_exprs[base_id])],
                    operator=wrapper, parameter_symbols=sorted(
                        str(s) for s in expr.free_symbols
                        if str(s).startswith(("a", "c"))
                    ),
                ))
        return records

    total = int(cfg["max_count"])
    two_count = min(round(total * float(cfg["two_base_ratio"])), total)
    one_count = total - two_count
    attempts = int(cfg.get("max_sampling_attempts", max(1000, total * 100)))
    seen: set[str] = set()

    def make(base_count: int):
        for _attempt in range(max(1, attempts)):
            chosen = rng.sample(ids, base_count)
            candidate_wrappers = allowed_wrappers(chosen)
            if candidate_wrappers:
                break
        else:
            raise ValueError(
                "The candidate space contains no valid symbolic-constant candidate with a non-symbolic base"
            )
        inner = sum((sp.Symbol(f"c{i + 1}") * base_exprs[base_id]
                     for i, base_id in enumerate(chosen)), sp.S.Zero)
        wrapper = rng.choice(candidate_wrappers)
        # Each single formula retains one optimizable outer amplitude coefficient. A combination formula will
        # remove it in _renamed_candidate, then generate a1/a2 at the combination layer.
        expr = _collapse_symbolic_constant_products(
            sp.Symbol("a0") * apply_wrapper(inner, wrapper)
        )
        key = f"{wrapper}:{','.join(sorted(chosen))}"
        return formula_record(
            expr,
            id="", structure_type="single_candidate", base_count=base_count,
            base_ids=chosen, bases=[str(base_exprs[item]) for item in chosen],
            operator=wrapper, parameter_symbols=sorted(
                str(s) for s in expr.free_symbols
                if str(s).startswith(("a", "c"))
        ),
        ), key

    records = _sample_unique(one_count, lambda: make(1), rng, attempts, seen)
    if len(ids) >= 2:
        records.extend(_sample_unique(two_count, lambda: make(2), rng, attempts, seen))
    elif two_count:
        logging.warning("Fewer than two bases; cannot generate two-base candidates")
    rng.shuffle(records)
    for index, record in enumerate(records):
        record["id"] = f"candidate_{index:06d}"
    return records


def _renamed_candidate(record: dict, side: str) -> sp.Expr:
    expr = sp.sympify(record["readable_expression"])
    # A single formula has its own outer amplitude coefficient a0; when combined, the combination layer's a1/a2
    # handle the amplitude to avoid a duplicated coefficient such as a1*a0*term.
    outer_coefficient = sp.Symbol("a0")
    if outer_coefficient in expr.free_symbols:
        expr = expr / outer_coefficient
    substitutions = {
        symbol: sp.Symbol(f"{symbol}_{side}")
        for symbol in expr.free_symbols if str(symbol).startswith("c")
    }
    return expr.xreplace(substitutions)


def build_combination_records(
    candidates: Sequence[dict], config: dict, rng: random.Random,
    *, shuffle_records: bool = True,
) -> list[dict]:
    cfg = config["combinations"]
    count = int(cfg["max_count"])
    max_terms = int(cfg.get("max_terms", 2))
    if max_terms < 2:
        raise ValueError("combinations.max_terms must be at least 2")
    types = list(cfg.get("types", ["add", "multiply"]))
    if not set(types) <= {"add", "multiply"}:
        raise ValueError("combinations.types must be add or multiply")
    if len(candidates) < 2:
        return []
    records = []
    by_id = {row["id"]: row for row in candidates}
    ids = list(by_id)
    # Enumerate the finite combination space exactly.  The previous random
    # retry loop could stop before covering all unique pairs.  Shuffle only
    # after enumeration so the persisted order remains random-looking but is
    # fully reproducible through the caller's seeded Random instance.
    for term_count in range(2, min(max_terms, len(ids)) + 1):
        for chosen_ids in itertools.combinations(ids, term_count):
            sides = (("left", "right") if term_count == 2 else
                     tuple(f"term{index}" for index in range(1, term_count + 1)))
            terms = [
                _renamed_candidate(by_id[candidate_id], side)
                for candidate_id, side in zip(chosen_ids, sides)
            ]
            for kind in types:
                if kind == "add":
                    expr = sp.Add(*[
                        sp.Symbol(f"a{index}") * term
                        for index, term in enumerate(terms, start=1)
                    ])
                else:
                    expr = sp.Mul(sp.Symbol("a1"), *terms)
                if expr.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
                    continue
                expr = _collapse_symbolic_constant_products(expr)
                records.append(formula_record(
                    expr, id="",
                    structure_type=("binary_combination" if term_count == 2
                                    else "multi_combination"),
                    combination_type=kind, combination_term_count=term_count,
                    candidate_ids=list(chosen_ids),
                    operators=[by_id[candidate_id]["operator"] for candidate_id in chosen_ids],
                    base_ids=[by_id[candidate_id]["base_ids"] for candidate_id in chosen_ids],
                    parameter_symbols=sorted(str(s) for s in expr.free_symbols
                                             if str(s).startswith(("a", "c"))),
                ))
    if shuffle_records:
        rng.shuffle(records)
    if len(records) > count:
        records = records[:count]
    if len(records) < count:
        logging.warning("Combination space too small: requested %d, generated %d", count, len(records))
    for index, record in enumerate(records):
        record["id"] = f"combination_{index:06d}"
    return records


def _parameter_symbols(expr: sp.Expr, feature_count: int) -> list[sp.Symbol]:
    variables = {sp.Symbol(f"x{i + 1}") for i in range(feature_count)}
    unknown_variables = [symbol for symbol in expr.free_symbols
                         if str(symbol).startswith("x") and symbol not in variables]
    if unknown_variables:
        raise ValueError(f"Formula uses variables not present in the data: {unknown_variables}")
    return sorted(expr.free_symbols - variables, key=str)


@dataclass
class FormulaEvaluator:
    expr: sp.Expr
    X: np.ndarray
    max_abs_value: float
    complex_tolerance: float

    def __post_init__(self):
        self.variables = [sp.Symbol(f"x{i + 1}") for i in range(self.X.shape[1])]
        self.params = _parameter_symbols(self.expr, self.X.shape[1])
        self.function = sp.lambdify([*self.variables, *self.params], self.expr, modules="numpy")

    def __call__(self, values: Sequence[float]) -> tuple[np.ndarray, np.ndarray, str | None]:
        try:
            with np.errstate(all="ignore"):
                raw = self.function(*[self.X[:, i] for i in range(self.X.shape[1])], *values)
            array = np.asarray(raw)
            if array.ndim == 0:
                array = np.full(len(self.X), array)
            array = np.broadcast_to(array, (len(self.X),)).copy()
            if np.iscomplexobj(array):
                complex_bad = np.abs(np.imag(array)) > self.complex_tolerance
                array = np.real(array)
            else:
                complex_bad = np.zeros(len(self.X), dtype=bool)
            array = np.asarray(array, dtype=float)
            valid = (~complex_bad) & np.isfinite(array) & (np.abs(array) <= self.max_abs_value)
            return array, valid, None if np.all(valid) else "non_finite_complex_or_overflow"
        except (ArithmeticError, TypeError, ValueError, OverflowError) as exc:
            return np.full(len(self.X), np.nan), np.zeros(len(self.X), dtype=bool), str(exc)


def compute_metrics(y: np.ndarray, prediction: np.ndarray, valid: np.ndarray) -> dict:
    count = int(np.sum(valid))
    if count == 0:
        return {"mse": None, "rmse": None, "r2": None, "loss": None, "valid_samples": 0}
    actual, predicted = y[valid], prediction[valid]
    residual = actual - predicted
    with np.errstate(all="ignore"):
        loss = float(np.dot(residual, residual))
        mse = loss / count
        total = float(np.dot(actual - np.mean(actual), actual - np.mean(actual)))
        r2 = 1.0 - loss / total if total > 0 else float("nan")
    return _json_value({"mse": mse, "rmse": math.sqrt(mse), "r2": r2,
                        "loss": loss, "valid_samples": count})


def compute_joint_metrics(y: np.ndarray, prediction: np.ndarray, valid: np.ndarray,
                          dataset_groups: Sequence[dict], aggregation: str) -> dict:
    """Keep pooled metrics while using a cross-dataset R² as the rank label."""
    pooled = compute_metrics(y, prediction, valid)
    per_dataset = []
    for group in dataset_groups:
        start, end = int(group["start"]), int(group["end"])
        metrics = compute_metrics(y[start:end], prediction[start:end], valid[start:end])
        per_dataset.append({"name": group["name"], "path": group.get("path"), **metrics})
    finite_r2 = [float(row["r2"]) for row in per_dataset if row["r2"] is not None]
    if not finite_r2:
        universal_r2 = None
    elif aggregation == "minimum":
        universal_r2 = min(finite_r2)
    elif aggregation == "mean":
        universal_r2 = float(np.mean(finite_r2))
    elif aggregation == "pooled":
        universal_r2 = pooled["r2"]
    else:
        raise ValueError("fitting.joint_r2_aggregation must be minimum, mean, or pooled")
    return _json_value({
        **pooled, "pooled_r2": pooled["r2"], "r2": universal_r2,
        "joint_r2": universal_r2, "joint_r2_aggregation": aggregation,
        "dataset_metrics": per_dataset,
    })


def _residual_autocorrelation_metrics(residual: np.ndarray) -> dict:
    """Return lightweight residual autocorrelation diagnostics in row order."""
    residual = np.asarray(residual, dtype=float)
    residual = residual[np.isfinite(residual)]
    if len(residual) < 2:
        return {"samples": int(len(residual)), "durbin_watson": None, "lag_1": None}
    denominator = float(np.dot(residual, residual))
    durbin_watson = (float(np.dot(np.diff(residual), np.diff(residual))) / denominator
                      if denominator > 0 else None)
    left, right = residual[:-1], residual[1:]
    left_centered, right_centered = left - np.mean(left), right - np.mean(right)
    scale = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    lag_1 = (float(np.dot(left_centered, right_centered) / scale) if scale > 0 else None)
    return _json_value({"samples": int(len(residual)), "durbin_watson": durbin_watson,
                        "lag_1": lag_1})


def add_gp_residual_autocorrelation(
    gp_results: list[dict], X: np.ndarray, y: np.ndarray,
    dataset_groups: Sequence[dict], fitting_config: dict,
) -> None:
    """Attach residual autocorrelation diagnostics to each GP-front dataset fit.

    CSV row order is treated as the observation sequence.  The diagnostics are
    deliberately post-fit only: they do not alter GP selection objectives.
    """
    groups_by_name = {group["name"]: group for group in dataset_groups}
    for row in gp_results:
        try:
            expression = sp.sympify(row["readable_expression"])
        except (sp.SympifyError, TypeError, ValueError):
            continue
        for dataset in row.get("dataset_fit_results", []):
            group = groups_by_name.get(dataset.get("dataset_name"))
            parameters = dataset.get("parameters")
            if group is None or parameters is None:
                dataset["residual_autocorrelation"] = None
                continue
            start, end = int(group["start"]), int(group["end"])
            try:
                evaluator = FormulaEvaluator(
                    expression, X[start:end], float(fitting_config.get("max_abs_value", 1e150)),
                    float(fitting_config.get("complex_tolerance", 1e-12)),
                )
                prediction, valid, _error = evaluator(parameters)
                dataset["residual_autocorrelation"] = _residual_autocorrelation_metrics(
                    y[start:end][valid] - prediction[valid]
                )
            except (ArithmeticError, TypeError, ValueError, OverflowError):
                dataset["residual_autocorrelation"] = None


def save_formula_mode_gp_plots(
    output_dir: Path, gp_results: Sequence[dict], X: np.ndarray, y: np.ndarray,
    dataset_groups: Sequence[dict], fitting_config: dict,
    baseline_predictions: dict[str, np.ndarray] | None = None,
) -> list[str]:
    """Save ``y-x1`` and ``y-x2`` overlays for formula-mode GP results.

    The generated samples are scattered in feature space, so both the actual
    observations and the GP predictions are drawn as points. Optional
    ``baseline_predictions`` are keyed by dataset name and are drawn on the
    same figures. Predictions are evaluated on the exact ``X`` passed to the
    initial fitting stage; no new grid or resampling is introduced.
    """
    if X.ndim != 2 or X.shape[1] < 2 or not gp_results:
        return []

    # Import lazily so CSV/MAT modes and non-plotting runs do not require a
    # plotting backend during normal pipeline startup.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "formula_mode_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    groups_by_name = {group["name"]: group for group in dataset_groups}
    saved_paths: list[str] = []
    max_abs_value = float(fitting_config.get("max_abs_value", 1e150))
    complex_tolerance = float(fitting_config.get("complex_tolerance", 1e-12))

    for formula_index, row in enumerate(gp_results, start=1):
        try:
            expression = sp.sympify(row["readable_expression"])
        except (sp.SympifyError, TypeError, ValueError):
            logging.warning("Skipping GP visualization formula that could not be parsed: %s", row.get("formula_id"))
            continue

        prediction = np.full(len(y), np.nan, dtype=float)
        for dataset in row.get("dataset_fit_results", []) or []:
            group = groups_by_name.get(dataset.get("dataset_name"))
            parameters = dataset.get("parameters")
            if group is None or parameters is None:
                continue
            start, end = int(group["start"]), int(group["end"])
            try:
                evaluator = FormulaEvaluator(
                    expression, X[start:end], max_abs_value, complex_tolerance
                )
                values, valid, _error = evaluator(parameters)
                prediction[start:end] = np.where(valid, values, np.nan)
            except (ArithmeticError, TypeError, ValueError, OverflowError):
                logging.warning("GP formula %s could not produce complete predictions", row.get("formula_id"))

        formula_id = str(row.get("formula_id") or f"formula_{formula_index:04d}")
        safe_id = "".join(char if char.isalnum() or char in "-_" else "_"
                           for char in formula_id)
        for feature_index, feature_name in enumerate(("x1", "x2")):
            figure, axis = plt.subplots(figsize=(8, 5.5))
            axis.scatter(X[:, feature_index], y, s=8, alpha=0.45,
                         color="tab:blue", label="Actual data")
            valid = np.isfinite(X[:, feature_index]) & np.isfinite(prediction)
            axis.scatter(X[valid, feature_index], prediction[valid], s=10,
                         alpha=0.65, color="tab:orange", label="GP formula")
            if baseline_predictions:
                for dataset_name, values in baseline_predictions.items():
                    group = groups_by_name.get(dataset_name)
                    if group is None:
                        continue
                    start, end = int(group["start"]), int(group["end"])
                    values = np.asarray(values, dtype=float)
                    if len(values) != end - start:
                        continue
                    baseline_valid = (
                        np.isfinite(X[start:end, feature_index])
                        & np.isfinite(values)
                    )
                    if np.any(baseline_valid):
                        axis.scatter(
                            X[start:end, feature_index][baseline_valid],
                            values[baseline_valid], s=12, alpha=0.55,
                            marker="x", label=f"PySR ({dataset_name})",
                        )
            axis.set_xlabel(feature_name)
            axis.set_ylabel("y")
            axis.set_title(f"{formula_id}: y-{feature_name}")
            axis.grid(True, alpha=0.25)
            axis.legend()
            figure.tight_layout()
            path = plot_dir / f"{safe_id}_{feature_name}.png"
            figure.savefig(path, dpi=160)
            plt.close(figure)
            saved_paths.append(str(path))
    return saved_paths






















































def _mode5_callable_worker(result_queue, callable_obj, args, kwargs):
    """Execute one Mode 5 LM candidate in a killable child process.

    The callable is intentionally passed through a ``fork`` context by the
    parent.  This lets the Mode 5 closure use the already prepared fitting
    state without making the large nested pipeline state pickleable.
    """
    try:
        if os.name == "posix":
            try:
                os.setsid()
            except OSError:
                pass
        result_queue.put(("ok", callable_obj(*args, **kwargs)))
    except BaseException as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_mode5_callable_with_watchdog(callable_obj, args, kwargs, timeout):
    """Run one Mode 5 candidate with a parent-enforced wall-clock limit."""
    if timeout is None or float(timeout) <= 0:
        return callable_obj(*args, **kwargs)
    methods = mp.get_all_start_methods()
    if "fork" not in methods:
        # The current pipeline is Linux-oriented; retain correctness on
        # platforms without fork rather than attempting to pickle closures.
        return callable_obj(*args, **kwargs)
    context = mp.get_context("fork")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_mode5_callable_worker,
        args=(result_queue, callable_obj, tuple(args), dict(kwargs)),
        daemon=False,
    )
    process.start()
    try:
        try:
            status, payload = result_queue.get(timeout=float(timeout))
        except queue.Empty as exc:
            if process.is_alive():
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except (OSError, ProcessLookupError):
                        process.terminate()
                else:
                    process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
            raise TimeoutError(
                f"Mode 5 constant candidates exceeded {float(timeout):g}s and were terminated"
            ) from exc
    finally:
        result_queue.close()
        result_queue.join_thread()
    process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join()
    if status != "ok":
        raise RuntimeError(f"Mode 5 constant-candidate child process failed: {payload}")
    return payload


































def _joint_residual_weights(
    y: np.ndarray, groups: Sequence[dict], cfg: dict,
) -> np.ndarray:
    weights = np.ones(len(y), dtype=float)
    weighting = cfg.get("joint_residual_weighting", "equal_dataset")
    if len(groups) <= 1 or weighting == "equal_sample":
        return weights
    if weighting != "equal_dataset":
        raise ValueError("fitting.joint_residual_weighting must be equal_dataset or equal_sample")
    for group in groups:
        start, end = int(group["start"]), int(group["end"])
        count = max(1, end - start)
        weight = math.sqrt(len(y) / (len(groups) * count))
        if cfg.get("normalize_dataset_scale", False):
            target = y[start:end]
            target_std = float(np.std(target)) if target.size else math.nan
            if math.isfinite(target_std) and target_std > 0.0:
                weight /= target_std
        weights[start:end] = weight
    return np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)


def fit_with_restarts(
    record: dict, X: np.ndarray, y: np.ndarray, cfg: dict, rng: np.random.Generator,
    dataset_groups: Sequence[dict] | None = None,
    initial_parameters: Sequence[float] | None = None,
    initial_perturbation: float = 0.0,
    warm_start_fraction: float = 1.0,
    validation_X: np.ndarray | None = None,
    validation_y: np.ndarray | None = None,
) -> tuple[list[dict], dict]:
    expr = sp.sympify(record["readable_expression"])
    optimizer_name = str(cfg.get("optimizer", "LM")).strip().upper()
    if optimizer_name not in ("LM", "BFGS"):
        raise ValueError("fitting.optimizer must be LM or BFGS")
    bfgs_cfg = cfg.get("bfgs", {}) or {}
    if validation_X is None:
        validation_X = cfg.get("_validation_X")
    if validation_y is None:
        validation_y = cfg.get("_validation_y")
    evaluator = FormulaEvaluator(expr, X, float(cfg.get("max_abs_value", 1e150)),
                                 float(cfg.get("complex_tolerance", 1e-12)))
    restarts = int(
        bfgs_cfg.get("optimizer_nrestarts", 20)
        if optimizer_name == "BFGS" else cfg["lm_initializations"]
    )
    patience = int(cfg.get("patience", 10))
    low, high = map(float, (
        bfgs_cfg.get("initialization_range", [-10.0, 10.0])
        if optimizer_name == "BFGS" else cfg["initialization_range"]
    ))
    bfgs_maxiter = int(bfgs_cfg.get("optimizer_iterations", 8))
    raw_bfgs_maxfun = bfgs_cfg.get("optimizer_f_calls_limit", 10000)
    bfgs_maxfun = None if raw_bfgs_maxfun is None else int(raw_bfgs_maxfun)
    bfgs_gtol = float(bfgs_cfg.get("gtol", 1e-5))
    penalty = float(cfg.get("invalid_residual_penalty", 1e6))
    # Require sufficient coverage on every dataset; otherwise a formula with
    # a restricted domain could score well on only a favorable subset.
    min_valid = max(2, len(evaluator.params) + 1)
    minimum_valid_fraction = float(cfg.get("minimum_valid_fraction", 0.9))
    if not 0.0 <= minimum_valid_fraction <= 1.0:
        raise ValueError("fitting.minimum_valid_fraction must be in [0, 1]")
    dataset_groups = list(dataset_groups or (
        {"name": "dataset_1", "start": 0, "end": len(y)},
    ))
    aggregation = str(cfg.get(
        "joint_r2_aggregation", "minimum" if len(dataset_groups) > 1 else "pooled"
    ))
    sample_weights = _joint_residual_weights(y, dataset_groups, cfg)
    def residual(params):
        prediction, valid, _ = evaluator(params)
        result = np.full(len(y), penalty, dtype=float)
        result[valid] = prediction[valid] - y[valid]
        result[~np.isfinite(result)] = penalty
        return np.clip(result * sample_weights, -penalty, penalty)

    def required_valid_count(sample_count: int) -> int:
        return max(min(2, sample_count), int(math.ceil(
            sample_count * minimum_valid_fraction
        )))

    def all_groups_valid(valid):
        # Enforce coverage separately for every dataset.
        for group in dataset_groups:
            start, end = int(group["start"]), int(group["end"])
            if int(np.sum(valid[start:end])) < required_valid_count(end - start):
                return False
        return True

    attempts = []
    warm_start = None
    if initial_parameters is not None:
        candidate = np.asarray(initial_parameters, dtype=float)
        if candidate.shape == (len(evaluator.params),) and np.all(np.isfinite(candidate)):
            warm_start = candidate
    warm_start_count = 0
    if warm_start is not None:
        warm_start_fraction = min(1.0, max(0.0, float(warm_start_fraction)))
        warm_start_count = int(math.ceil(restarts * warm_start_fraction))
    for initialization_index in range(restarts):
        if warm_start is None or initialization_index >= warm_start_count:
            initial = rng.uniform(low, high, len(evaluator.params))
            initialization_mode = "random"
        elif initial_perturbation <= 0:
            initial = warm_start.copy()
            initialization_mode = "warm_start"
        else:
            # Use an additive perturbation scaled to each parameter so that
            # parameters near zero can still escape the exact global optimum.
            scale_by_parameter = np.maximum(1.0, np.abs(warm_start))
            initial = warm_start + rng.normal(
                0.0, float(initial_perturbation), len(warm_start)
            ) * scale_by_parameter
            initialization_mode = "warm_start_perturbed"
        attempt = {
            "formula_id": record["id"], "initialization_index": initialization_index,
            "sympy_expression": record.get("sympy_expression", sp.srepr(expr)),
            "readable_expression": record["readable_expression"],
            "structure_type": record.get("structure_type", "single_candidate"),
            "base_ids": record.get("base_ids", []),
            "operator": record.get("operator"),
            "fit_scale": "raw", "initial_parameters": initial.tolist(),
            "initialization_mode": initialization_mode,
            "parameters": None, "mse": None, "rmse": None, "r2": None,
            "loss": None, "optimization_loss": None, "valid_samples": 0,
            "all_datasets_valid": False, "converged": False,
            "optimizer_success": False, "failure_reason": None, "lm_steps": 0,
            "optimizer": optimizer_name, "optimizer_nfev": 0,
            "optimizer_iterations": 0, "optimizer_within_budget": True,
            "validation_r2": None, "validation_metrics": None,
            "best_validation_r2": None, "best_validation_parameters": None,
            "validation_history": [],
        }
        try:
            if len(evaluator.params) == 0:
                optimized, optimizer_success, message, lm_steps = initial, True, "no_parameters", 0
            elif optimizer_name == "BFGS":
                # Match LM's validation selection contract as closely as
                # SciPy BFGS permits.  BFGS invokes its callback once per
                # optimizer iteration (rather than once per residual call),
                # so patience is measured in BFGS iterations here.
                class _BFGSValidationEarlyStop(Exception):
                    pass

                val_evaluator = None
                best_validation_r2 = -math.inf
                best_validation_parameters = None
                stale_steps = 0
                bfgs_objective_evaluations = 0
                bfgs_callback_iterations = 0
                if validation_X is not None and validation_y is not None:
                    val_evaluator = FormulaEvaluator(
                        expr, validation_X, float(cfg.get("max_abs_value", 1e150)),
                        float(cfg.get("complex_tolerance", 1e-12)),
                    )

                def check_bfgs_validation(params):
                    nonlocal best_validation_r2, best_validation_parameters, stale_steps
                    if val_evaluator is None:
                        return False
                    val_prediction, val_valid, _ = val_evaluator(params)
                    val_r2 = compute_metrics(
                        validation_y, val_prediction, val_valid
                    ).get("r2")
                    attempt["validation_history"].append({
                        "step": len(attempt["validation_history"]) + 1,
                        "r2": val_r2,
                        "parameters": np.asarray(params, dtype=float).tolist(),
                    })
                    if val_r2 is not None and val_r2 > best_validation_r2:
                        best_validation_r2 = float(val_r2)
                        best_validation_parameters = np.asarray(
                            params, dtype=float
                        ).copy()
                        stale_steps = 0
                    else:
                        stale_steps += 1
                    return stale_steps >= patience

                def objective(params):
                    nonlocal bfgs_objective_evaluations
                    bfgs_objective_evaluations += 1
                    objective_residual = residual(params)
                    if not np.all(np.isfinite(objective_residual)):
                        return float("inf")
                    value = float(np.dot(objective_residual, objective_residual))
                    return value if math.isfinite(value) else float("inf")

                # Include the initial point, as LM's residual callback does.
                # This lets BFGS retain a strong initial validation solution
                # even if every subsequent step makes validation worse.
                initial_patience_exhausted = check_bfgs_validation(initial)

                def bfgs_callback(params):
                    nonlocal bfgs_callback_iterations
                    bfgs_callback_iterations += 1
                    if check_bfgs_validation(params):
                        raise _BFGSValidationEarlyStop

                try:
                    if initial_patience_exhausted:
                        raise _BFGSValidationEarlyStop
                    result = minimize(
                        objective, initial, method="BFGS", callback=bfgs_callback,
                        options={"maxiter": bfgs_maxiter, "gtol": bfgs_gtol},
                    )
                    optimized = np.asarray(result.x, dtype=float)
                    within_budget = (
                        bfgs_maxfun is None or int(result.nfev) <= bfgs_maxfun
                    )
                    optimizer_success = bool(result.success and within_budget)
                    message = str(result.message)
                    lm_steps = int(result.nit)
                    attempt["optimizer_nfev"] = int(result.nfev)
                    attempt["optimizer_iterations"] = int(result.nit)
                except _BFGSValidationEarlyStop:
                    optimized = (best_validation_parameters
                                 if best_validation_parameters is not None else initial)
                    within_budget = (
                        bfgs_maxfun is None
                        or bfgs_objective_evaluations <= bfgs_maxfun
                    )
                    optimizer_success = bool(within_budget)
                    message = "validation patience exhausted"
                    lm_steps = bfgs_callback_iterations
                    attempt["optimizer_nfev"] = bfgs_objective_evaluations
                    attempt["optimizer_iterations"] = bfgs_callback_iterations
                attempt["optimizer_within_budget"] = bool(within_budget)
                # As with LM, expose the parameters selected by validation,
                # not necessarily BFGS's last training-objective iterate.
                if best_validation_parameters is not None:
                    optimized = best_validation_parameters
            else:
                # Keep one continuous MINPACK LM call.  Its residual entry
                # point acts as a local callback, so validation checkpoints do
                # not require repeatedly restarting LM and losing damping
                # state.
                class _ValidationEarlyStop(Exception):
                    pass

                val_evaluator = None
                best_validation_r2 = -math.inf
                best_validation_parameters = None
                stale_steps = 0
                if validation_X is not None and validation_y is not None:
                    val_evaluator = FormulaEvaluator(
                        expr, validation_X, float(cfg.get("max_abs_value", 1e150)),
                        float(cfg.get("complex_tolerance", 1e-12)),
                    )

                def residual_with_validation(params):
                    nonlocal best_validation_r2, best_validation_parameters, stale_steps
                    values = residual(params)
                    if val_evaluator is None:
                        return values
                    val_prediction, val_valid, _ = val_evaluator(params)
                    val_r2 = compute_metrics(
                        validation_y, val_prediction, val_valid
                    ).get("r2")
                    attempt["validation_history"].append({
                        "step": len(attempt["validation_history"]) + 1,
                        "r2": val_r2,
                        "parameters": np.asarray(params, dtype=float).tolist(),
                    })
                    if val_r2 is not None and val_r2 > best_validation_r2:
                        best_validation_r2 = float(val_r2)
                        best_validation_parameters = np.asarray(
                            params, dtype=float
                        ).copy()
                        stale_steps = 0
                    else:
                        stale_steps += 1
                    if stale_steps >= patience:
                        raise _ValidationEarlyStop
                    return values

                try:
                    result = least_squares(
                        residual_with_validation, initial, method="lm",
                        max_nfev=int(cfg.get("max_nfev", 5000)),
                    )
                    optimized, optimizer_success = result.x, bool(result.success)
                    message, lm_steps = str(result.message), int(result.nfev)
                    attempt["optimizer_nfev"] = int(result.nfev)
                    attempt["optimizer_iterations"] = int(result.nfev)
                except _ValidationEarlyStop:
                    optimized = (best_validation_parameters
                                 if best_validation_parameters is not None else initial)
                    optimizer_success, message = True, "validation patience exhausted"
                    lm_steps = len(attempt["validation_history"])
                # A normal LM convergence does not imply that its final trial
                # point is the validation-best point.  The public fit result
                # must consistently retain the checkpoint selected by the
                # validation criterion, whether LM stopped normally or was
                # interrupted by patience.
                if best_validation_parameters is not None:
                    optimized = best_validation_parameters
            full_parameters = optimized
            prediction, valid, evaluation_error = evaluator(optimized)
            metrics = compute_joint_metrics(
                y, prediction, valid, dataset_groups, aggregation,
            )
            attempt.update(metrics)
            attempt["all_datasets_valid"] = all_groups_valid(valid)
            optimized_residual = residual(optimized)
            attempt["optimization_loss"] = float(np.dot(optimized_residual, optimized_residual))
            attempt.update(parameters=full_parameters.tolist(), optimizer_success=optimizer_success,
                           optimizer_message=message, lm_steps=lm_steps)
            if validation_X is not None and validation_y is not None:
                val_evaluator = FormulaEvaluator(
                    expr, validation_X, float(cfg.get("max_abs_value", 1e150)),
                    float(cfg.get("complex_tolerance", 1e-12)),
                )
                val_prediction, val_valid, _ = val_evaluator(optimized)
                val_metrics = compute_metrics(validation_y, val_prediction, val_valid)
                attempt["validation_metrics"] = val_metrics
                attempt["validation_r2"] = val_metrics.get("r2")
                attempt["best_validation_r2"] = val_metrics.get("r2")
                attempt["best_validation_parameters"] = full_parameters.tolist()
            if (metrics["valid_samples"] < max(min_valid, required_valid_count(len(y)))
                    or not attempt["all_datasets_valid"]):
                attempt["failure_reason"] = evaluation_error or "insufficient_valid_samples_per_dataset"
            elif not optimizer_success and optimizer_name == "LM":
                attempt["failure_reason"] = "optimizer_not_converged"
            elif metrics["loss"] is None:
                attempt["failure_reason"] = "invalid_metrics"
            else:
                attempt["converged"] = True
        except Exception as exc:  # one bad expression must not stop the batch
            attempt["failure_reason"] = f"{type(exc).__name__}: {exc}"
        attempts.append(_json_value(attempt))

    successful = [row for row in attempts if row["converged"] and row["loss"] is not None]
    # Preserve the best finite fit from every restart, including a restart
    # stopped by validation patience, instead of discarding useful metrics.
    eligible = [row for row in attempts if row["loss"] is not None
                and row["optimization_loss"] is not None
                and row["valid_samples"] >= max(min_valid, required_valid_count(len(y)))
                and row["all_datasets_valid"]
                and row.get("optimizer_within_budget", True)]
    if eligible:
        best_attempt = min(
            eligible,
            key=lambda row: (
                -(row.get("validation_r2") if row.get("validation_r2") is not None
                  else -math.inf),
                row["optimization_loss"],
                int(row.get("initialization_index", 10**9)),
            ),
        )
        best = {**best_attempt, "formula_id": record["id"],
                "parameter_names": [str(item) for item in evaluator.params],
                "successful_initializations": len(successful)}
    else:
        reasons = sorted({row["failure_reason"] for row in attempts if row["failure_reason"]})
        best = {
            "formula_id": record["id"], "fit_scale": "raw", "parameters": None,
            "parameter_names": [str(item) for item in evaluator.params],
            "mse": None, "rmse": None, "r2": None, "loss": None,
            "optimization_loss": None, "pooled_r2": None, "joint_r2": None,
            "joint_r2_aggregation": aggregation, "dataset_metrics": [],
            "all_datasets_valid": False,
            "valid_samples": 0, "initialization_index": None, "converged": False,
            "optimizer_success": False, "successful_initializations": 0,
            "failure_reason": "; ".join(reasons) or "all_initializations_failed",
        }
    return attempts, _json_value(best)


def _stable_lm_seed(global_seed: int, dataset_name: str, formula_id: str) -> int:
    """Derive a process/order-independent seed for one formula fit."""
    payload = f"{int(global_seed)}\0{dataset_name}\0{formula_id}".encode()
    # A digest, rather than Python's hash(), is stable across processes/runs.
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _fit_config_for_dataset_validation(
    fitting_config: dict, dataset_name: str,
) -> dict:
    """Return a fitting config whose validation data belongs to one dataset."""
    cfg = dict(fitting_config)
    validation_X = cfg.get("_validation_X")
    validation_y = cfg.get("_validation_y")
    validation_groups = cfg.get("_validation_groups") or []
    if validation_X is None or validation_y is None:
        return cfg
    group = next(
        (item for item in validation_groups
         if str(item.get("name")) == str(dataset_name)), None
    )
    if group is None:
        # A one-dataset validation set has no useful grouping metadata to
        # match, so it is already safe to use as-is.  For multi-dataset data,
        # callers must provide matching names; run-time validation catches a
        # missing/invalid result rather than silently mixing datasets.
        if len(validation_groups) > 1:
            raise ValueError(
                f"Could not find a validation group for dataset {dataset_name}"
            )
        return cfg
    start, end = int(group["start"]), int(group["end"])
    cfg["_validation_X"] = validation_X[start:end]
    cfg["_validation_y"] = validation_y[start:end]
    cfg["_validation_groups"] = [
        dict(group, start=0, end=end - start, samples=end - start)
    ]
    return cfg


def _fit_config_for_piecewise_physical_validation(
    fitting_config: dict, segmentation_config: dict,
) -> dict:
    """Split real-data validation samples at the configured physical prior."""
    cfg = dict(fitting_config)
    validation_X = cfg.get("_validation_X")
    validation_y = cfg.get("_validation_y")
    if validation_X is None or validation_y is None:
        return cfg
    prior = _breakpoint_prior(segmentation_config)
    if prior is None:
        return cfg
    validation_X = np.asarray(validation_X)
    validation_y = np.asarray(validation_y)
    if validation_X.ndim != 2 or validation_X.shape[1] != 1:
        return cfg
    split_index = int(np.searchsorted(
        validation_X[:, 0], prior["breakpoint_value"], side="right"
    ))
    if not 0 < split_index < len(validation_X):
        return cfg
    cfg["_validation_X"] = validation_X
    cfg["_validation_y"] = validation_y
    cfg["_validation_groups"] = [
        {"name": "segment_1", "start": 0, "end": split_index,
         "samples": split_index},
        {"name": "segment_2", "start": split_index, "end": len(validation_y),
         "samples": len(validation_y) - split_index},
    ]
    return cfg


def _fit_config_for_piecewise_split_validation(
    fitting_config: dict, split_value: float,
) -> dict:
    """Partition validation observations so each piecewise fit sees its own side."""
    cfg = dict(fitting_config)
    validation_X = cfg.get("_validation_X")
    validation_y = cfg.get("_validation_y")
    if validation_X is None or validation_y is None:
        return cfg
    validation_X = np.asarray(validation_X)
    validation_y = np.asarray(validation_y)
    if (validation_X.ndim != 2 or validation_X.shape[1] != 1
            or validation_y.ndim != 1
            or len(validation_X) != len(validation_y)):
        raise ValueError("The validation set for piecewise fitting must contain equal-length 1D feature and target arrays")

    masks = (validation_X[:, 0] <= float(split_value),
             validation_X[:, 0] > float(split_value))
    X_parts = [validation_X[mask] for mask in masks]
    y_parts = [validation_y[mask] for mask in masks]
    groups = []
    offset = 0
    for index, (X_part, y_part) in enumerate(zip(X_parts, y_parts), start=1):
        groups.append({
            "name": f"segment_{index}", "start": offset,
            "end": offset + len(y_part), "samples": len(y_part),
        })
        offset += len(y_part)
    cfg["_validation_X"] = np.concatenate(X_parts, axis=0)
    cfg["_validation_y"] = np.concatenate(y_parts)
    cfg["_validation_groups"] = groups
    return cfg


def _fit_lm_task(task: tuple) -> tuple[list[dict], dict]:
    """Worker for one independent formula/data-set LM job.

    Keep this function at module scope so it can be imported by spawned worker
    processes on Windows as well as forked workers on POSIX.
    """
    (record, X, y, fitting_config, segmentation_config, piecewise,
     dataset_name, global_seed) = task
    rng = np.random.default_rng(
        _stable_lm_seed(global_seed, dataset_name, record["id"])
    )
    if piecewise:
        validation_groups = fitting_config.get("_validation_groups") or []
        if {str(item.get("name")) for item in validation_groups} != {
                "segment_1", "segment_2"}:
            fitting_config = _fit_config_for_dataset_validation(
                fitting_config, dataset_name
            )
        return fit_piecewise_formula(
            record, X, y, fitting_config, rng, segmentation_config
        )
    fitting_config = _fit_config_for_dataset_validation(
        fitting_config, dataset_name
    )
    return fit_with_restarts(record, X, y, fitting_config, rng)


def _parallel_lm_results(
    tasks: Sequence[tuple], max_workers: int | None = None,
) -> list[tuple[list[dict], dict]]:
    """Run independent formula fits in parallel, preserving input order."""
    if not tasks:
        return []
    if max_workers is None:
        # Leave one core for the parent process and avoid creating more workers
        # than formulas.  Keep the process count bounded so LM fitting does not
        # overwhelm the host on machines with many CPU cores.
        max_workers = min(
            len(tasks), DEFAULT_LM_MAX_WORKERS,
            max(1, (os.cpu_count() or 2) - 1),
        )
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # executor.map yields results in task order even when workers finish in
        # a different order, making persisted JSON/CSV ordering deterministic.
        return list(executor.map(_fit_lm_task, tasks))


def fit_formula_per_dataset(
    record: dict, X: np.ndarray, y: np.ndarray, cfg: dict, rng: np.random.Generator,
    dataset_groups: Sequence[dict],
    initial_parameters_by_dataset: dict[str, Sequence[float]] | None = None,
    initial_values_by_dataset: dict[str, dict[str, float]] | None = None,
    initial_perturbation: float = 0.0,
    warm_start_fraction: float = 1.0,
    lm_initializations: int | None = None,
    fit_rng: np.random.Generator | None = None,
) -> tuple[list[dict], dict]:
    """Fit one structure independently per dataset, then aggregate only metrics."""
    fit_cfg = dict(cfg)
    if lm_initializations is not None:
        fit_cfg["lm_initializations"] = int(lm_initializations)
    all_attempts, dataset_best = [], []
    predictions, valid_parts = [], []
    expr = sp.sympify(record["readable_expression"])
    aggregation = str(cfg.get("joint_r2_aggregation", "minimum"))
    active_rng = fit_rng if fit_rng is not None else rng
    for group in dataset_groups:
        start, end = int(group["start"]), int(group["end"])
        X_part, y_part = X[start:end], y[start:end]
        group_fit_cfg = dict(fit_cfg)
        validation_X_part = group_fit_cfg.get("_validation_X")
        validation_y_part = group_fit_cfg.get("_validation_y")
        validation_groups = group_fit_cfg.get("_validation_groups") or []
        matching_validation_group = next(
            (item for item in validation_groups
             if str(item.get("name")) == str(group.get("name"))), None
        )
        if matching_validation_group is not None:
            val_start = int(matching_validation_group["start"])
            val_end = int(matching_validation_group["end"])
            validation_X_part = validation_X_part[val_start:val_end]
            validation_y_part = validation_y_part[val_start:val_end]
        expr_for_group = expr
        warm_start = ((initial_parameters_by_dataset or {}).get(group["name"])
                      if initial_parameters_by_dataset else None)
        if warm_start is None and initial_values_by_dataset:
            previous_values = initial_values_by_dataset.get(group["name"], {})
            if previous_values:
                evaluator = FormulaEvaluator(
                    expr_for_group, X_part, float(cfg.get("max_abs_value", 1e150)),
                    float(cfg.get("complex_tolerance", 1e-12)),
                )
                low, high = map(float, cfg["initialization_range"])
                warm_start = np.asarray([
                    float(previous_values[str(symbol)])
                    if str(symbol) in previous_values
                    and np.isfinite(float(previous_values[str(symbol)]))
                    else active_rng.uniform(low, high)
                    for symbol in evaluator.params
                ], dtype=float)
        attempts, best = fit_with_restarts(
            record, X_part, y_part, group_fit_cfg, active_rng,
            initial_parameters=warm_start,
            initial_perturbation=initial_perturbation,
            warm_start_fraction=warm_start_fraction,
            validation_X=validation_X_part,
            validation_y=validation_y_part,
        )
        for attempt in attempts:
            attempt.update(dataset_name=group["name"], dataset_path=group.get("path"))
        best.update(dataset_name=group["name"], dataset_path=group.get("path"))
        all_attempts.extend(attempts)
        dataset_best.append(best)

        if best.get("parameters") is None:
            predictions.append(np.full(len(y_part), np.nan))
            valid_parts.append(np.zeros(len(y_part), dtype=bool))
        else:
            evaluator = FormulaEvaluator(
                expr_for_group, X_part, float(cfg.get("max_abs_value", 1e150)),
                float(cfg.get("complex_tolerance", 1e-12)),
            )
            prediction, valid, _ = evaluator(best["parameters"])
            predictions.append(prediction)
            valid_parts.append(valid)

    prediction = np.concatenate(predictions)
    valid = np.concatenate(valid_parts)
    metrics = compute_joint_metrics(y, prediction, valid, dataset_groups, aggregation)
    # Preserve the independently selected per-dataset fit details rather than
    # pretending that one parameter vector applies to every frequency band.
    parameters_by_dataset = {
        row["dataset_name"]: row.get("parameters") for row in dataset_best
    }
    initialization_by_dataset = {
        row["dataset_name"]: row.get("initialization_index") for row in dataset_best
    }
    successful_by_dataset = {
        row["dataset_name"]: row.get("successful_initializations", 0) for row in dataset_best
    }
    failures = [f"{row['dataset_name']}: {row.get('failure_reason')}"
                for row in dataset_best if not row.get("converged")]
    first_names = dataset_best[0].get("parameter_names", []) if dataset_best else []
    combined = {
        "formula_id": record["id"], "fit_scale": "raw",
        "parameters": parameters_by_dataset,
        "parameters_by_dataset": parameters_by_dataset,
        "parameter_names": first_names,
        "initialization_index": initialization_by_dataset,
        "initialization_index_by_dataset": initialization_by_dataset,
        "successful_initializations": successful_by_dataset,
        "successful_initializations_by_dataset": successful_by_dataset,
        "optimizer_success": all(row.get("optimizer_success", False) for row in dataset_best),
        "converged": all(row.get("converged", False) for row in dataset_best),
        "failure_reason": "; ".join(failures) if failures else None,
        "optimization_loss": sum(float(row.get("optimization_loss") or 0.0)
                                 for row in dataset_best),
        "all_datasets_valid": all(row.get("all_datasets_valid", False)
                                  for row in dataset_best),
        "dataset_fit_results": dataset_best,
        "expected_dataset_count": len(dataset_groups),
        "constants_shared_across_datasets": False,
        **metrics,
    }
    return all_attempts, _json_value(combined)


def fit_piecewise_formula(
    record: dict, X: np.ndarray, y: np.ndarray, cfg: dict,
    rng: np.random.Generator, segmentation_config: dict,
    warm_start_parameters: Sequence[float] | None = None,
    warm_start_parameters_by_dataset: dict[str, Sequence[float]] | None = None,
    preserve_split_index: int | None = None,
) -> tuple[list[dict], dict]:
    """Fit a one-variable formula on both sides of every valid split point.

    ``X`` must already be sorted by its only feature.  Each side receives an
    independent LM fit, which lets a common symbolic structure describe two
    propagation regimes with different constants.  The chosen split minimizes
    the combined raw SSE (``loss``), while the reported R² keeps the pipeline's
    normal joint aggregation setting.
    """
    if X.ndim != 2 or X.shape[1] != 1:
        raise ValueError("Piecewise fitting supports only univariate measured data")
    sample_count = len(y)
    minimum = int(segmentation_config.get("min_segment_samples", 8))
    stride = int(segmentation_config.get("candidate_stride", 1))
    perturbation = float(segmentation_config.get("warm_start_perturbation", 0.05))
    warm_starts = warm_start_parameters_by_dataset
    if warm_starts is None and warm_start_parameters is not None:
        warm_starts = {"segment_1": warm_start_parameters, "segment_2": warm_start_parameters}
    split_indices = list(range(minimum, sample_count - minimum + 1, stride))
    if preserve_split_index is not None:
        # Keep the previous stage's split as an explicit fallback candidate.
        split_indices = sorted(set(split_indices) | {int(preserve_split_index)})
    split_indices, prior = _restrict_split_indices(X, split_indices, segmentation_config)
    all_attempts, evaluated = [], []
    for split_index in split_indices:
        split_value = _split_value(X, split_index)
        groups = [
            {"name": "segment_1", "start": 0, "end": split_index},
            {"name": "segment_2", "start": split_index, "end": sample_count},
        ]
        split_fitting_config = _fit_config_for_piecewise_split_validation(
            cfg, split_value
        )
        attempts, result = fit_formula_per_dataset(
            record, X, y, split_fitting_config, rng, groups,
            initial_parameters_by_dataset=warm_starts,
            initial_perturbation=perturbation,
        )
        all_attempts.extend(attempts)
        result.update(
            piecewise=True,
            split_index=split_index,
            split_value=split_value,
            segment_sample_counts=[split_index, sample_count - split_index],
            segmentation_score=result.get("loss"),
        )
        _apply_breakpoint_regularization(result, split_value, prior)
        evaluated.append(result)
    eligible = [row for row in evaluated if row.get("loss") is not None]
    if not eligible:
        return all_attempts, _piecewise_fit_failure(
            record, "No usable breakpoint; lower min_segment_samples or check the measured data",
            len(evaluated), "fine",
        )
    best = min(eligible, key=lambda row: float(row.get("segmentation_score", row["loss"])))
    best["evaluated_split_count"] = len(evaluated)
    return all_attempts, _json_value(best)


def _coarse_split_indices(sample_count: int, segmentation_config: dict) -> list[int]:
    """Evenly select a small number of legal split indices for coarse search."""
    minimum = int(segmentation_config.get("min_segment_samples", 8))
    lower, upper = minimum, sample_count - minimum
    if lower > upper:
        return []
    requested = int(segmentation_config.get("coarse_split_count", 12))
    if requested < 1:
        raise ValueError("data.realdata.segmentation.coarse_split_count must be at least 1")
    count = min(requested, upper - lower + 1)
    return sorted({int(round(value)) for value in np.linspace(lower, upper, count)})


def _breakpoint_prior(segmentation_config: dict) -> dict | None:
    """Return the configured physical breakpoint prior in data-distance units."""
    prior = segmentation_config.get("breakpoint_prior", {}) or {}
    if not bool(prior.get("enabled", False)):
        return None
    height_t = float(prior.get("h_t_m", 25.0))
    height_r = float(prior.get("h_r_m", 4.0))
    frequency_hz = float(prior.get("carrier_frequency_hz", 5.8e9))
    speed_of_light = float(prior.get("speed_of_light_m_per_s", 299792458.0))
    meters_to_data_units = float(prior.get("meters_to_data_units", 1.0))
    if min(height_t, height_r, frequency_hz, speed_of_light,
           meters_to_data_units) <= 0:
        raise ValueError("Physical parameters in breakpoint_prior must be positive")
    wavelength_m = speed_of_light / frequency_hz
    breakpoint_m = 4.0 * height_t * height_r / wavelength_m
    return {
        "h_t_m": height_t, "h_r_m": height_r,
        "carrier_frequency_hz": frequency_hz,
        "speed_of_light_m_per_s": speed_of_light,
        "wavelength_m": wavelength_m,
        "breakpoint_m": breakpoint_m,
        "meters_to_data_units": meters_to_data_units,
        "breakpoint_value": breakpoint_m * meters_to_data_units,
        "regularization_weight": float(prior.get("regularization_weight", 0.0)),
    }


def _split_value(X: np.ndarray, split_index: int) -> float:
    return float((X[split_index - 1, 0] + X[split_index, 0]) / 2.0)


def _restrict_split_indices(
    X: np.ndarray, split_indices: Sequence[int], segmentation_config: dict,
) -> tuple[list[int], dict | None]:
    """Return the split selected by the physical prior, when enabled.

    A measured dataset can only split between two adjacent samples, so the
    physical breakpoint is represented by the legal boundary whose midpoint
    is closest to the prior value.  Once a prior is enabled this is the only
    returned index; the caller must not search the remaining boundaries.
    """
    prior = _breakpoint_prior(segmentation_config)
    indices = sorted({int(index) for index in split_indices})
    if prior is None or not indices:
        return indices, prior
    # Do not use the caller's coarse/fine candidate list here.  It may not
    # contain the boundary corresponding to the fixed physical breakpoint.
    minimum = int(segmentation_config.get("min_segment_samples", 8))
    legal_indices = list(range(minimum, len(X) - minimum + 1))
    if not legal_indices:
        return [], prior
    fixed_index = min(
        legal_indices,
        key=lambda index: abs(_split_value(X, index) - prior["breakpoint_value"]),
    )
    return [fixed_index], prior


def _apply_breakpoint_regularization(
    result: dict, split_value: float, prior: dict | None,
) -> None:
    """Record the distance to the fixed physical breakpoint."""
    if prior is None:
        return
    weight = prior["regularization_weight"]
    if weight < 0:
        raise ValueError("breakpoint_prior.regularization_weight cannot be negative")
    distance = float(split_value - prior["breakpoint_value"])
    penalty = weight * distance * distance
    raw_loss = result.get("loss")
    result.update(
        split_prior_value=prior["breakpoint_value"],
        split_prior_distance=distance,
        split_prior_penalty=penalty,
        raw_segmentation_loss=raw_loss,
        segmentation_score=(None if raw_loss is None else float(raw_loss) + penalty),
        segmentation_objective="total_residual_sse_at_fixed_breakpoint",
    )


def fit_piecewise_formula_at_splits(
    record: dict, X: np.ndarray, y: np.ndarray, cfg: dict,
    rng: np.random.Generator, split_indices: Sequence[int], stage: str,
    segmentation_config: dict | None = None,
    warm_start_parameters: Sequence[float] | None = None,
    warm_start_parameters_by_dataset: dict[str, Sequence[float]] | None = None,
    preserve_split_index: int | None = None,
) -> tuple[list[dict], dict]:
    """Fit one common structure with independent constants at supplied splits."""
    if X.ndim != 2 or X.shape[1] != 1:
        raise ValueError("Piecewise fitting supports only univariate measured data")
    all_attempts, evaluated = [], []
    perturbation = 0.05
    if warm_start_parameters is not None:
        # The caller may pass the configured value through the split wrapper;
        # this default keeps the low-level helper backwards compatible.
        perturbation = float(cfg.get("piecewise_warm_start_perturbation", 0.05))
    segmentation_config = segmentation_config or {}
    if preserve_split_index is not None:
        # The previous stage is always a legal fallback for the next stage.
        split_indices = sorted(set(split_indices) | {int(preserve_split_index)})
    split_indices, prior = _restrict_split_indices(X, split_indices, segmentation_config)
    for split_index in split_indices:
        groups = [
            {"name": "segment_1", "start": 0, "end": split_index},
            {"name": "segment_2", "start": split_index, "end": len(y)},
        ]
        warm_starts = warm_start_parameters_by_dataset
        if warm_starts is None and warm_start_parameters is not None:
            warm_starts = {"segment_1": warm_start_parameters,
                           "segment_2": warm_start_parameters}
        attempts, result = fit_formula_per_dataset(
            record, X, y, cfg, rng, groups,
            initial_parameters_by_dataset=warm_starts,
            initial_perturbation=perturbation,
        )
        all_attempts.extend(attempts)
        result.update(
            piecewise=True,
            split_index=int(split_index),
            split_value=_split_value(X, split_index),
            segment_sample_counts=[int(split_index), int(len(y) - split_index)],
            segmentation_score=result.get("loss"),
            segmentation_objective="total_residual_sse",
            segmentation_search_stage=stage,
        )
        _apply_breakpoint_regularization(result, result["split_value"], prior)
        evaluated.append(result)
    eligible = [row for row in evaluated if row.get("loss") is not None]
    if not eligible:
        return all_attempts, _piecewise_fit_failure(
            record, "No usable breakpoint; lower min_segment_samples or check the measured data",
            len(evaluated), stage,
        )
    # ``loss`` comes from compute_joint_metrics and is the pooled SSE over
    # both segments, so fine/coarse split selection minimizes total residual.
    best = min(eligible, key=lambda row: float(row.get("segmentation_score", row["loss"])))
    best["evaluated_split_count"] = len(evaluated)
    return all_attempts, _json_value(best)


def _piecewise_fit_failure(
    record: dict, reason: str, evaluated_split_count: int, stage: str,
) -> dict:
    """Return a GP-compatible result when a formula has no usable split.

    A candidate expression can be mathematically valid yet fail on every
    piecewise split because of its domain or optimizer convergence.  That is
    a candidate-level failure, not a reason to abort the whole GP run.
    """
    return _json_value({
        "formula_id": record["id"],
        "fit_scale": "raw",
        "parameters": None,
        "parameters_by_dataset": {},
        "parameter_names": [],
        "converged": False,
        "optimizer_success": False,
        "failure_reason": reason,
        "optimization_loss": None,
        "loss": None,
        "r2": None,
        "mse": None,
        "rmse": None,
        "dataset_fit_results": [],
        "piecewise": True,
        "segmentation_search_stage": stage,
        "evaluated_split_count": evaluated_split_count,
    })


def fit_coarse_piecewise_formula(
    record: dict, X: np.ndarray, y: np.ndarray, cfg: dict,
    rng: np.random.Generator, segmentation_config: dict,
    warm_start_parameters: Sequence[float] | None = None,
) -> tuple[list[dict], dict]:
    coarse_cfg = dict(cfg)
    coarse_cfg["piecewise_warm_start_perturbation"] = float(
        segmentation_config.get("warm_start_perturbation", 0.05)
    )
    return fit_piecewise_formula_at_splits(
        record, X, y, coarse_cfg, rng,
        _coarse_split_indices(len(y), segmentation_config), "coarse",
        segmentation_config=segmentation_config,
        warm_start_parameters=warm_start_parameters,
    )


def fit_fine_piecewise_formula(
    record: dict, X: np.ndarray, y: np.ndarray, cfg: dict,
    rng: np.random.Generator, segmentation_config: dict,
    warm_start_parameters_by_dataset: dict[str, Sequence[float]] | None = None,
    preserve_split_index: int | None = None,
) -> tuple[list[dict], dict]:
    fine_config = dict(segmentation_config)
    fine_config["candidate_stride"] = int(segmentation_config.get("fine_candidate_stride", 1))
    attempts, best = fit_piecewise_formula(
        record, X, y, cfg, rng, fine_config,
        warm_start_parameters_by_dataset=warm_start_parameters_by_dataset,
        preserve_split_index=preserve_split_index,
    )
    best["segmentation_search_stage"] = "fine"
    return attempts, best


def _piecewise_formula_summary(record: dict, result: dict) -> dict:
    """Keep the fit fields needed to describe one side of a formula pair."""
    return {
        "formula_id": record["id"],
        "readable_expression": record["readable_expression"],
        "sympy_expression": record.get("sympy_expression"),
        "structure_type": record.get("structure_type"),
        "parameter_names": result.get("parameter_names", []),
        "parameters": result.get("parameters"),
        "r2": result.get("r2"), "mse": result.get("mse"),
        "rmse": result.get("rmse"), "loss": result.get("loss"),
        "converged": result.get("converged", False),
        "failure_reason": result.get("failure_reason"),
    }


def _combine_piecewise_fits(record_left: dict, left: dict, X_left: np.ndarray,
                            y_left: np.ndarray, record_right: dict, right: dict,
                            X_right: np.ndarray, y_right: np.ndarray,
                            split_index: int, split_value: float) -> dict | None:
    """Evaluate two independently fitted formula structures as one piecewise model."""
    if left.get("parameters") is None or right.get("parameters") is None:
        return None
    try:
        left_eval = FormulaEvaluator(
            sp.sympify(record_left["readable_expression"]), X_left, 1e150, 1e-12
        )
        right_eval = FormulaEvaluator(
            sp.sympify(record_right["readable_expression"]), X_right, 1e150, 1e-12
        )
        left_prediction, left_valid, _ = left_eval(left["parameters"])
        right_prediction, right_valid, _ = right_eval(right["parameters"])
    except (ArithmeticError, TypeError, ValueError, OverflowError):
        return None
    metrics = compute_metrics(
        np.concatenate((y_left, y_right)),
        np.concatenate((left_prediction, right_prediction)),
        np.concatenate((left_valid, right_valid)),
    )
    if metrics["loss"] is None:
        return None
    return _json_value({
        "piecewise": True,
        "split_index": split_index,
        "split_value": split_value,
        "segment_sample_counts": [len(y_left), len(y_right)],
        "left_formula": _piecewise_formula_summary(record_left, left),
        "right_formula": _piecewise_formula_summary(record_right, right),
        "readable_expression": (
            f"if x1 <= {split_value:.12g}: {record_left['readable_expression']}; "
            f"else: {record_right['readable_expression']}"
        ),
        "segmentation_score": metrics["loss"],
        **metrics,
    })


def search_piecewise_formula_pairs(
    records: Sequence[dict], X: np.ndarray, y: np.ndarray, cfg: dict,
    rng: np.random.Generator, segmentation_config: dict,
) -> tuple[list[dict], dict[str, dict]]:
    """Fit each formula independently on each side; never form formula pairs."""
    if X.ndim != 2 or X.shape[1] != 1:
        raise ValueError("Piecewise formula search supports only univariate measured data")
    minimum = int(segmentation_config.get("min_segment_samples", 8))
    stride = int(segmentation_config.get("candidate_stride", 1))
    split_indices, prior = _restrict_split_indices(
        X, range(minimum, len(y) - minimum + 1, stride), segmentation_config
    )
    fits_by_formula: dict[str, dict] = {}
    for split_index in tqdm(split_indices, total=len(split_indices),
                            desc="Measured-data independent segment fitting (fixed breakpoint)"):
        X_parts, y_parts = (X[:split_index], X[split_index:]), (y[:split_index], y[split_index:])
        split_value = float((X[split_index - 1, 0] + X[split_index, 0]) / 2.0)
        split_fitting_config = _fit_config_for_piecewise_split_validation(
            cfg, split_value
        )
        for record in records:
            segment_fits = []
            for index, (Xs, ys) in enumerate(zip(X_parts, y_parts), start=1):
                # Validation-based restart selection and early stopping must
                # use only the validation observations belonging to this
                # segment.  Passing ``cfg`` directly would make
                # fit_with_restarts fall back to the full validation set for
                # both segment fits.
                segment_cfg = dict(split_fitting_config)
                segment_group = next(
                    item for item in segment_cfg.get("_validation_groups", [])
                    if item["name"] == f"segment_{index}"
                ) if segment_cfg.get("_validation_X") is not None else None
                if segment_group is None:
                    segment_validation_X = segment_validation_y = None
                else:
                    start, end = int(segment_group["start"]), int(segment_group["end"])
                    if start == end:
                        segment_validation_X = segment_validation_y = None
                    else:
                        segment_validation_X = segment_cfg["_validation_X"][start:end]
                        segment_validation_y = segment_cfg["_validation_y"][start:end]
                # Explicit None values prevent fit_with_restarts from
                # silently falling back to the unsliced arrays when one side
                # has no validation observations.
                segment_cfg["_validation_X"] = segment_validation_X
                segment_cfg["_validation_y"] = segment_validation_y
                _attempts, fitted = fit_with_restarts(
                    record, Xs, ys, segment_cfg, rng,
                    validation_X=segment_validation_X,
                    validation_y=segment_validation_y,
                )
                fitted = dict(fitted)
                fitted.update(dataset_name=f"segment_{index}", segment_sample_count=len(ys))
                segment_fits.append(fitted)
            if not all(item.get("loss") is not None for item in segment_fits):
                continue
            # Each parameter vector and metric belongs to its own segment.
            # The summary supports existing multi-dataset ranking only; no
            # joint predictions, pair score, or Cartesian formula search.
            fits_by_formula[record["id"]] = {
                "formula_id": record["id"], "piecewise": True,
                "independent_segment_fits": True,
                "split_index": int(split_index), "split_value": split_value,
                "split_prior_value": float(prior["breakpoint_value"]) if prior else None,
                "segment_sample_counts": [len(y_parts[0]), len(y_parts[1])],
                "parameters": {f"segment_{i+1}": item.get("parameters")
                               for i, item in enumerate(segment_fits)},
                "parameters_by_dataset": {item["dataset_name"]: item.get("parameters")
                                          for item in segment_fits},
                "parameter_names": segment_fits[0].get("parameter_names", []),
                "dataset_fit_results": segment_fits,
                "r2": float(np.mean([float(item["r2"]) for item in segment_fits])),
                "mse": float(np.mean([float(item["mse"]) for item in segment_fits])),
                "rmse": float(np.mean([float(item["rmse"]) for item in segment_fits])),
                "loss": float(sum(float(item["loss"]) for item in segment_fits)),
                "valid_samples": int(sum(int(item.get("valid_samples", 0)) for item in segment_fits)),
                "converged": all(bool(item.get("converged")) for item in segment_fits),
            }
    return [], fits_by_formula



def _select_symbol_operator_stratified(
    rows: Sequence[dict], target_count: int, ranking_config: dict,
) -> list[dict]:
    """Select a tree-ranked Top-K with optional operator quotas."""
    strata = {"log_only": [], "log_and_sin": [], "log_and_cos": []}
    for row in rows:
        try:
            expression = sp.sympify(row["readable_expression"])
            has_log = any(node.func is sp.log for node in sp.preorder_traversal(expression))
            has_sin = any(node.func is sp.sin for node in sp.preorder_traversal(expression))
            has_cos = any(node.func is sp.cos for node in sp.preorder_traversal(expression))
        except (KeyError, sp.SympifyError, TypeError, ValueError):
            has_log = has_sin = has_cos = False
        if has_log:
            if has_sin:
                strata["log_and_sin"].append(row)
            elif has_cos:
                strata["log_and_cos"].append(row)
            else:
                strata["log_only"].append(row)

    base_quota, remainder = divmod(target_count, len(strata))
    quotas = {
        group: base_quota + (index < remainder)
        for index, group in enumerate(strata)
    }
    if any(len(strata[group]) < quota for group, quota in quotas.items()):
        selected = [
            dict(row) for row in sorted(
                rows, key=lambda row: row.get("predicted_rank", 10 ** 9)
            )[:target_count]
        ]
    else:
        selected = []
        for group, quota in quotas.items():
            ranked = sorted(
                strata[group], key=lambda row: row.get("predicted_rank", 10 ** 9)
            )
            for row in ranked[:quota]:
                annotated = dict(row)
                annotated["operator_stratum"] = group
                selected.append(annotated)
        if len(selected) < target_count:
            selected = [
                dict(row) for row in sorted(
                    rows, key=lambda row: row.get("predicted_rank", 10 ** 9)
                )[:target_count]
            ]
    if len(selected) < target_count:
        if ranking_config.get("_allow_reduced_top_k", False):
            logging.warning(
                "Mode 2 tree-model Top-K reduced due to existing constraints: target=%d, actual=%d",
                target_count, len(selected),
            )
            return selected
        raise ValueError(
            "Too few formulas are available for tree-model Top-K rank selection: "
            f"selected={len(selected)}, target={target_count}."
        )
    return sorted(selected, key=lambda row: row.get("predicted_rank", 10 ** 9))


def _select_tree_top_k_for_refit(
    ranking: Sequence[dict], top_k: int,
    ranking_config: dict | None = None,
) -> list[dict]:
    """Select a fixed hybrid Top-K: best direct fits plus unseen tree ranks."""
    if top_k <= 0:
        return []
    rows = list(ranking)
    target_count = min(top_k, len(rows))
    ranking_config = ranking_config or {}
    if (ranking_config.get("ablation_enabled", False)
            and not ranking_config.get("ablation_use_operator_stratified", False)):
        return [
            dict(row) for row in sorted(
                rows, key=lambda row: row.get("predicted_rank", 10 ** 9)
            )[:target_count]
        ]
    if (ranking_config.get("ablation_enabled", False)
            and ranking_config.get("ablation_use_operator_stratified", False)):
        return _select_symbol_operator_stratified(
            rows, target_count, ranking_config,
        )

    direct_limit = int(ranking_config.get("direct_fit_top_k", 5))
    if direct_limit < 0:
        raise ValueError("ranking.direct_fit_top_k cannot be negative")
    training_rows = [
        row for row in rows if bool(row.get("is_training_formula", False))
    ]
    training_rows.sort(
        key=lambda row: (
            -(float(row["actual_fit_r2"])
              if row.get("actual_fit_r2") is not None else -math.inf),
            row.get("formula_id", ""),
        )
    )
    direct_rows = [dict(row) for row in training_rows[:direct_limit]]
    for row in direct_rows:
        row["selection_source"] = "actual_fit_r2"
    direct_ids = {row.get("formula_id") for row in direct_rows}
    unseen_rows = [
        row for row in rows
        if not bool(row.get("is_training_formula", False))
        and row.get("formula_id") not in direct_ids
    ]
    remaining = max(0, target_count - len(direct_rows))
    tree_rows = [
        dict(row) for row in sorted(
            unseen_rows, key=lambda row: row.get("predicted_rank", 10 ** 9)
        )[:remaining]
    ] if remaining else []
    for row in tree_rows:
        row["selection_source"] = "tree_model_unseen"
    selected = direct_rows + tree_rows
    if len(selected) < target_count:
        if ranking_config.get("_allow_reduced_top_k", False):
            logging.warning(
                "Mixed Top-K reduced because too few formulas are outside the training set: target=%d, actual=%d",
                target_count, len(selected),
            )
            return selected
        raise ValueError(
            "Too few formulas outside the training set for mixed Top-K: "
            f"selected={len(selected)}, target={target_count}; "
            "add candidate formulas not used for direct fitting or lower ranking.refit_top_k."
        )
    return selected


def refit_top_ranked_formulas(
    ranking: Sequence[dict], candidates: Sequence[dict], combinations: Sequence[dict],
    X: np.ndarray, y: np.ndarray, fitting_config: dict, ranking_config: dict,
    rng: np.random.Generator, dataset_groups: Sequence[dict],
    segmentation_config: dict | None = None,
    n_jobs: int | None = None,
) -> list[dict]:
    """Independently refit tree-model Top-K formulas on every configured CSV."""
    top_k = min(int(ranking_config.get("refit_top_k", 10)), len(ranking))
    if top_k <= 0:
        return []
    selected_ranking = _select_tree_top_k_for_refit(
        ranking, top_k, ranking_config,
    )
    records = {row["id"]: row for row in [*candidates, *combinations]}
    refit_config = dict(fitting_config)
    if "refit_lm_initializations" in ranking_config:
        refit_config["lm_initializations"] = int(ranking_config["refit_lm_initializations"])
    def fit_one(ranked):
        record = records[ranked["formula_id"]]
        task_rng = (
            rng if n_jobs == 1 else np.random.default_rng(
                _stable_lm_seed(
                    int(ranking_config.get("seed", 0)),
                    "top_ranked", ranked["formula_id"],
                )
            )
        )
        if segmentation_config and segmentation_config.get("enabled", True):
            _attempts, best = fit_piecewise_formula(
                record, X, y, refit_config, task_rng, segmentation_config
            )
        else:
            _attempts, best = fit_formula_per_dataset(
                record, X, y, refit_config, task_rng, dataset_groups,
                fit_rng=task_rng,
            )
        return {
            "predicted_rank": ranked["predicted_rank"],
            "selection_source": ranked.get("selection_source"),
            "operator_stratum": ranked.get("operator_stratum"),
            "complexity_layer": ranked.get("complexity_layer"),
            "complexity_value": ranked.get("complexity_value"),
            "ranking_score": ranked["ranking_score"],
            "formula_family": ranked["formula_family"],
            "formula_id": ranked["formula_id"],
            "readable_expression": ranked["readable_expression"],
            "sympy_expression": ranked["sympy_expression"],
            **best,
        }

    if n_jobs == 1:
        return [fit_one(ranked) for ranked in tqdm(
            selected_ranking, desc="Refitting top formulas per dataset"
        )]
    workers = (
        min(8, max(1, (os.cpu_count() or 2) - 1))
        if n_jobs is None or int(n_jobs) < 0 else int(n_jobs)
    )
    workers = min(max(1, workers), len(selected_ranking))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(fit_one, selected_ranking))


def reuse_top_ranked_formula_fits(
    ranking: Sequence[dict], existing_results: Sequence[dict],
    ranking_config: dict,
) -> list[dict]:
    """Reuse previously fitted LM results for the ablation Top-K stage.

    The ablation ranking is built from ``single_best`` and
    ``combination_best``.  Those rows already contain the per-dataset LM
    parameters and scores, so refitting the selected rows would only add a
    second, unnecessary optimization pass.
    """
    top_k = min(int(ranking_config.get("refit_top_k", 10)), len(ranking))
    if top_k <= 0:
        return []
    selected = _select_tree_top_k_for_refit(
        ranking, top_k, ranking_config
    )
    by_id = {str(row.get("formula_id")): row for row in existing_results}
    results = []
    missing = []
    for ranked in selected:
        source = by_id.get(str(ranked["formula_id"]))
        if source is None:
            missing.append(str(ranked["formula_id"]))
            continue
        results.append({
            **source,
            "predicted_rank": ranked["predicted_rank"],
            "selection_source": ranked.get("selection_source"),
            "operator_stratum": ranked.get("operator_stratum"),
            "complexity_layer": ranked.get("complexity_layer"),
            "complexity_value": ranked.get("complexity_value"),
            "ranking_score": ranked["ranking_score"],
            "formula_family": ranked["formula_family"],
            "fit_reused": True,
        })
    if missing:
        raise ValueError(
            "The ablation experiment could not reuse formulas from the first-round fit results: " + ", ".join(missing)
        )
    return results


def refit_ablation_top_ranked_formulas_all_datasets(
    ranking: Sequence[dict], candidates: Sequence[dict],
    combinations: Sequence[dict], X: np.ndarray, y: np.ndarray,
    fitting_config: dict, ranking_config: dict, rng: np.random.Generator,
    dataset_groups: Sequence[dict], n_jobs: int | None = 1,
    segmentation_config: dict | None = None,
) -> list[dict]:
    """Refit LM-ablation Top-K structures on every configured dataset.

    The ordinary ablation reuses per-dataset LM rows.  For the joint
    tree+GP ablation, persist a comparable result for every selected
    structure containing one real fit per frequency/dataset.
    """
    top_k = min(int(ranking_config.get("refit_top_k", 10)), len(ranking))
    if top_k <= 0:
        return []
    if ranking_config.get("ablation_use_operator_stratified", False):
        # Reuse the exact normal Top-K selector. The ablation ranking only
        # contains formulas that were genuinely LM-fitted, so the
        # mixed selector's unseen-formula branch is not applicable here.
        selected = _select_tree_top_k_for_refit(
            ranking, top_k, ranking_config=ranking_config,
        )
    else:
        selected = []
        seen_ids = set()
        for ranked in sorted(
            ranking, key=lambda row: row.get("predicted_rank", 10 ** 9)
        ):
            formula_id = str(ranked.get("formula_id"))
            if formula_id in seen_ids:
                continue
            selected.append(ranked)
            seen_ids.add(formula_id)
            if len(selected) >= top_k:
                break
    records = {row["id"]: row for row in [*candidates, *combinations]}
    refit_config = dict(fitting_config)
    if "refit_lm_initializations" in ranking_config:
        refit_config["lm_initializations"] = int(
            ranking_config["refit_lm_initializations"]
        )
    def fit_one(ranked):
        record = records.get(ranked["formula_id"])
        if record is None:
            return None
        task_rng = (
            rng if n_jobs == 1 else np.random.default_rng(
                _stable_lm_seed(
                    int(ranking_config.get("seed", 0)),
                    "ablation_top_ranked", ranked["formula_id"],
                )
            )
        )
        if segmentation_config and segmentation_config.get("enabled", True):
            _attempts, best = fit_piecewise_formula(
                record, X, y, refit_config, task_rng, segmentation_config
            )
        else:
            _attempts, best = fit_formula_per_dataset(
                record, X, y, refit_config, task_rng, dataset_groups,
                fit_rng=task_rng,
            )
            expected_datasets = {str(group["name"]) for group in dataset_groups}
            actual_datasets = {
                str(row.get("dataset_name"))
                for row in best.get("dataset_fit_results", [])
            }
            if actual_datasets != expected_datasets:
                raise ValueError(
                    "LM ablation all-band refit results are incomplete: "
                    f"actual={sorted(actual_datasets)}, "
                    f"expected={sorted(expected_datasets)}"
                )
        return {
            **best,
            "predicted_rank": ranked["predicted_rank"],
            "selection_source": ranked.get("selection_source"),
            "operator_stratum": None,
            "complexity_layer": ranked.get("complexity_layer"),
            "complexity_value": ranked.get("complexity_value"),
            "ranking_score": ranked["ranking_score"],
            "formula_family": ranked["formula_family"],
            "formula_id": ranked["formula_id"],
            "readable_expression": ranked["readable_expression"],
            "sympy_expression": ranked.get("sympy_expression"),
            "all_frequency_refit": True,
        }

    if n_jobs == 1:
        fitted = [fit_one(ranked) for ranked in tqdm(
            selected, desc="LM ablation: refitting top formulas across all frequency bands"
        )]
    else:
        workers = (
            min(8, max(1, (os.cpu_count() or 2) - 1))
            if n_jobs is None or int(n_jobs) < 0 else min(int(n_jobs), len(selected))
        )
        workers = min(max(1, workers), len(selected))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            fitted = list(executor.map(fit_one, selected))
    return [row for row in fitted if row is not None]


def refit_top_ranked_formula_pairs(
    ranking: Sequence[dict], candidates: Sequence[dict], combinations: Sequence[dict],
    X: np.ndarray, y: np.ndarray, fitting_config: dict, ranking_config: dict,
    rng: np.random.Generator, segmentation_config: dict,
    n_jobs: int | None = 1,
) -> list[dict]:
    """Refit selected formula structures independently on the two segments."""
    top_k = min(int(ranking_config.get("refit_top_k", 10)), len(ranking))
    if top_k <= 0:
        return []
    records_by_id = {row["id"]: row for row in [*candidates, *combinations]}
    selected = _select_tree_top_k_for_refit(
        ranking, top_k, ranking_config,
    )
    records = [records_by_id[row["formula_id"]] for row in selected]
    refit_config = dict(fitting_config)
    if "refit_lm_initializations" in ranking_config:
        refit_config["lm_initializations"] = int(ranking_config["refit_lm_initializations"])
    _unused_pairs, fits_by_formula = search_piecewise_formula_pairs(
        records, X, y, refit_config, rng, segmentation_config
    )
    ranking_by_id = {row["formula_id"]: row for row in selected}
    results = []
    for record in records:
        fit = fits_by_formula.get(record["id"])
        if fit is None:
            continue
        results.append({
            "predicted_rank": ranking_by_id[record["id"]].get("predicted_rank"),
            "ranking_score": ranking_by_id[record["id"]].get("ranking_score"),
            "formula_family": "single" if record in candidates else "combination",
            "formula_id": record["id"],
            "readable_expression": record["readable_expression"],
            "sympy_expression": record.get("sympy_expression"),
            "parameters": fit["parameters_by_dataset"],
            "parameters_by_dataset": fit["parameters_by_dataset"],
            "dataset_fit_results": fit["dataset_fit_results"],
            "r2": fit["r2"], "mse": fit["mse"], "rmse": fit["rmse"],
            "loss": fit["loss"], "valid_samples": fit["valid_samples"],
            "piecewise": True, "independent_segment_fits": True,
            "split_index": fit["split_index"], "split_value": fit["split_value"],
            "split_prior_value": fit.get("split_prior_value"),
        })
    return _json_value(results)



def coarse_refit_top_ranked_formulas(
    ranking: Sequence[dict], candidates: Sequence[dict], combinations: Sequence[dict],
    X: np.ndarray, y: np.ndarray, fitting_config: dict, ranking_config: dict,
    rng: np.random.Generator, segmentation_config: dict,
    warm_start_results: Sequence[dict] | None = None,
    n_jobs: int | None = 1,
) -> list[dict]:
    """Coarsely locate the best common-structure split for tree-ranked formulas."""
    top_k = min(int(ranking_config.get("refit_top_k", 10)), len(ranking))
    records = {row["id"]: row for row in [*candidates, *combinations]}
    selected_ranking = _select_tree_top_k_for_refit(
        ranking, top_k, ranking_config,
    )
    refit_config = dict(fitting_config)
    if "refit_lm_initializations" in ranking_config:
        refit_config["lm_initializations"] = int(ranking_config["refit_lm_initializations"])
    warm_starts = {
        row.get("formula_id"): next(iter((row.get("parameters_by_dataset") or {}).values()), None)
        for row in (warm_start_results or [])
    }
    global_results = {
        row.get("formula_id"): row for row in (warm_start_results or [])
    }
    results = []
    for ranked in tqdm(selected_ranking, desc="Coarse piecewise fitting of top formulas"):
        record = records[ranked["formula_id"]]
        _attempts, best = fit_coarse_piecewise_formula(
            record, X, y, refit_config, rng, segmentation_config,
            warm_start_parameters=warm_starts.get(record["id"]),
        )
        baseline = global_results.get(record["id"])
        if baseline is not None and not _result_not_worse(best, baseline):
            best = {**baseline, "segmentation_rejected": True,
                    "segmentation_rejection_reason": "coarse_r2_worse_than_global"}
        results.append({
            "predicted_rank": ranked["predicted_rank"],
            "operator_stratum": ranked.get("operator_stratum"),
            "ranking_score": ranked["ranking_score"],
            "formula_family": ranked["formula_family"],
            "formula_id": ranked["formula_id"],
            "readable_expression": ranked["readable_expression"],
            "sympy_expression": ranked["sympy_expression"],
            **best,
        })
    return results


def piecewise_prediction_difference(
    expression: str, fit_result: dict, X: np.ndarray,
) -> float | None:
    """Return the prediction jump between both formulas at the breakpoint."""
    try:
        split_index = int(fit_result.get("split_index"))
    except (TypeError, ValueError):
        return None
    if not 0 < split_index < len(X):
        return None
    dataset_rows = {
        row.get("dataset_name"): row
        for row in fit_result.get("dataset_fit_results", [])
    }
    left = dataset_rows.get("segment_1")
    right = dataset_rows.get("segment_2")
    if not left or not right:
        return None
    try:
        breakpoint = float(fit_result.get("split_value"))
    except (TypeError, ValueError):
        breakpoint = _split_value(X, split_index)
    if not math.isfinite(breakpoint):
        return None
    # Evaluate both segment fits at the same input point: the midpoint
    # represented by split_value between the adjacent measured samples.
    breakpoint_X = np.asarray(X[split_index:split_index + 1], dtype=float).copy()
    breakpoint_X[0, 0] = breakpoint
    values = []
    for row in (left, right):
        parameters = row.get("parameters")
        if parameters is None:
            return None
        evaluator = FormulaEvaluator(
            sp.sympify(expression), breakpoint_X, 1e150, 1e-12,
        )
        prediction, valid, _ = evaluator(parameters)
        if not valid[0]:
            return None
        values.append(float(prediction[0]))
    difference = abs(values[1] - values[0])
    return difference if math.isfinite(difference) else None


def _attach_piecewise_continuity_objective(
    row: dict, expression: str, X: np.ndarray,
) -> dict:
    """Attach the fitted prediction jump used by piecewise selection."""
    predicted = piecewise_prediction_difference(expression, row, X)
    row["predicted_segment_difference"] = predicted
    row["segment_prediction_difference"] = predicted
    return row


def print_pre_gp_screen_results(
    results: Sequence[dict], stratified_metadata: dict | None = None,
    X: np.ndarray | None = None, data_mode: str | None = None,
) -> None:
    """Print the selected formulas and their complexity layers before GP."""
    print("\nLayered screening complete; formulas before GP:", flush=True)
    if _is_measured_pathloss_profile(data_mode):
        print("  Fitted prediction difference across the adjacent boundary samples:", flush=True)
        if X is None:
            print("   (no usable feature data)", flush=True)
        else:
            X_values = np.asarray(X, dtype=float)
            for index, row in enumerate(results, start=1):
                split_index = row.get("split_index")
                try:
                    split_index = int(split_index)
                except (TypeError, ValueError):
                    split_index = None
                if split_index is None or not 0 < split_index < len(X_values):
                    print(f"  #{index:>3} {row.get('formula_id')}: no valid breakpoint", flush=True)
                    continue
                parameters_by_dataset = row.get("parameters_by_dataset") or {}
                left = row.get("left_formula") or {
                    "readable_expression": row.get("readable_expression"),
                    "parameters": parameters_by_dataset.get("segment_1"),
                }
                right = row.get("right_formula") or {
                    "readable_expression": row.get("readable_expression"),
                    "parameters": parameters_by_dataset.get("segment_2"),
                }

                def predict_at(formula: dict, sample_index: int) -> float | None:
                    parameters = formula.get("parameters")
                    expression = formula.get("readable_expression")
                    if parameters is None or not expression:
                        return None
                    try:
                        evaluator = FormulaEvaluator(
                            sp.sympify(expression), X_values[sample_index:sample_index + 1],
                            1e150, 1e-12,
                        )
                        prediction, valid, _ = evaluator(parameters)
                        return float(prediction[0]) if valid[0] else None
                    except (ArithmeticError, TypeError, ValueError, OverflowError,
                            sp.SympifyError):
                        return None

                y_before = predict_at(left, split_index - 1)
                y_after = predict_at(right, split_index)
                difference = (None if y_before is None or y_after is None
                              else abs(y_after - y_before))
                print(
                    f"  #{index:>3} {row.get('formula_id')}: "
                    f"split_index={split_index}, predicted y before={y_before!s}, "
                    f"predicted y after={y_after!s}, "
                    f"|predicted y before-predicted y after|={difference!s}",
                    flush=True,
                )
    if stratified_metadata and stratified_metadata.get("enabled"):
        print(
            f"  complexity layer count={stratified_metadata.get('layer_count')}, "
            f"count per layer={stratified_metadata.get('per_layer')}, "
            f"final count={stratified_metadata.get('selected_count')}, "
            f"target top-k={stratified_metadata.get('target_top_k')}",
            flush=True,
        )
    if not results:
        print("   (no coarse-screening results)", flush=True)
        return
    for index, row in enumerate(results, start=1):
        # ``r2`` is the configured joint aggregation (normally the minimum
        # segment R²).  For the coarse-screen summary, show the R² computed
        # after pooling predictions from both segments over the full dataset.
        r2 = row.get("pooled_r2", row.get("r2"))
        loss = row.get("loss")
        r2_text = "None" if r2 is None else f"{float(r2):.6f}"
        loss_text = "None" if loss is None else f"{float(loss):.6g}"
        status = "failed" if not row.get("converged", False) else "successful"
        dataset_r2 = []
        for dataset in row.get("dataset_fit_results", []) or []:
            dataset_name = dataset.get("dataset_name", dataset.get("name", "dataset"))
            dataset_value = dataset.get("r2")
            dataset_text = "None" if dataset_value is None else f"{float(dataset_value):.6f}"
            dataset_r2.append(f"{dataset_name}={dataset_text}")
        dataset_text = "; R² by dataset=" + ", ".join(dataset_r2) if dataset_r2 else ""
        print(
            f"  #{index:>3} [{status}] {row.get('formula_id')}: "
            f"complexity layer={row.get('complexity_layer', '-')}, "
            f"overall R²={r2_text}, loss={loss_text}, "
            f"{row.get('readable_expression')}{dataset_text}",
            flush=True,
        )


def select_stratified_gp_ranking(
    ranking: Sequence[dict], gp_config: dict
) -> tuple[list[dict], dict]:
    """Split by complexity, while ensuring every available operator is seeded."""
    target = int(gp_config.get("top_k", 10))
    requested_layers = int(gp_config.get("complexity_layers", 5))
    if target <= 0 or not ranking:
        return [], {
            "enabled": True, "input_count": len(ranking), "layer_count": 0,
            "per_layer": 0, "selected_count": 0, "target_top_k": target,
            "layers": [],
        }

    ordered = []
    for row in ranking:
        try:
            value = int(formula_complexity(sp.sympify(row["readable_expression"])))
        except (sp.SympifyError, TypeError, ValueError, OverflowError):
            value = 10 ** 9
        ordered.append((value, row))
    # Complexity partitioning is the first operation. Use equal-width
    # absolute complexity intervals, rather than equal formula counts.
    minimum_complexity = min(value for value, _row in ordered)
    maximum_complexity = max(value for value, _row in ordered)
    layers = [[] for _index in range(requested_layers)]
    if minimum_complexity == maximum_complexity:
        layers[0] = ordered
    else:
        width = (maximum_complexity - minimum_complexity) / requested_layers
        for value, row in ordered:
            index = int((value - minimum_complexity) / width)
            index = min(requested_layers - 1, max(0, index))
            layers[index].append((value, row))
    non_empty_layers = [
        (index, layer) for index, layer in enumerate(layers) if layer
    ]
    layer_count = len(non_empty_layers)
    if gp_config.get("_redistribute_complexity_quota", False):
        # Mode 2 keeps the requested total Top-K when possible. A sparse
        # layer contributes all available candidates; its unused quota is
        # redistributed to layers that still have spare candidates.
        target_count = min(target, len(ordered))
        base_quota, remainder = divmod(target_count, layer_count)
        layer_quotas = {
            index: min(len(layer), base_quota + (position < remainder))
            for position, (index, layer) in enumerate(non_empty_layers)
        }
        while sum(layer_quotas.values()) < target_count:
            progressed = False
            for index, layer in non_empty_layers:
                if layer_quotas[index] < len(layer):
                    layer_quotas[index] += 1
                    progressed = True
                    if sum(layer_quotas.values()) >= target_count:
                        break
            if not progressed:
                break
        quota = None
    else:
        quota = max(1, int(math.floor(float(target) / layer_count + 0.5)))
        quota = min(quota, min(len(layer) for layer in non_empty_layers))
        layer_quotas = {index: quota for index, _layer in non_empty_layers}

    def operator_labels(row: dict) -> set[str]:
        """Return wrapper labels, including labels from combination rows."""
        labels = row.get("operators") or row.get("operator")
        if isinstance(labels, str):
            labels = [labels]
        labels = {str(label) for label in (labels or []) if label is not None}
        if labels:
            return labels
        # Older ranking checkpoints may not contain the explicit wrapper.
        try:
            expression = sp.sympify(row["readable_expression"])
            has_log = any(node.func is sp.log for node in sp.preorder_traversal(expression))
            has_sin = any(node.func is sp.sin for node in sp.preorder_traversal(expression))
            has_cos = any(node.func is sp.cos for node in sp.preorder_traversal(expression))
        except (KeyError, sp.SympifyError, TypeError, ValueError):
            return {"unknown"}
        if has_log and has_sin and not has_cos:
            return {"log_abs_sin"}
        if has_log and has_cos and not has_sin:
            return {"log_abs_cos"}
        if has_log:
            return {"log_abs"}
        if has_sin:
            return {"sin"}
        if has_cos:
            return {"cos"}
        return {"identity"}

    # Reserve the best rows needed for coverage before filling ordinary
    # complexity quotas. A combination row may cover more than one operator.
    available_operators = {
        label for _index, layer in non_empty_layers
        for _value, row in layer for label in operator_labels(row)
    }
    ranked_all = sorted(
        ((index, value, row) for index, layer in non_empty_layers
         for value, row in layer),
        key=lambda item: item[2].get("predicted_rank", 10 ** 9),
    )
    uncovered = set(available_operators)
    reserved_by_layer = {index: [] for index, _layer in non_empty_layers}
    reserved_ids = set()
    operator_coverage = {}
    while uncovered:
        choices = [item for item in ranked_all if id(item[2]) not in reserved_ids]
        if not choices:
            break
        item = max(
            choices,
            key=lambda candidate: (
                len(operator_labels(candidate[2]) & uncovered),
                -candidate[2].get("predicted_rank", 10 ** 9),
            ),
        )
        layer_index, _value, row = item
        covered = operator_labels(row) & uncovered
        if not covered:
            break
        reserved_ids.add(id(row))
        reserved_by_layer[layer_index].append(row)
        uncovered -= covered
        for label in covered:
            operator_coverage[label] = row.get("formula_id")
    if uncovered:
        raise ValueError("GP initial solutions cannot cover all operators: " + ", ".join(sorted(uncovered)))
    target_count = min(target, len(ordered))
    reserved_count = sum(len(rows) for rows in reserved_by_layer.values())
    if reserved_count > target_count:
        raise ValueError("gp.top_k is smaller than the number of operators to cover; full operator coverage is impossible")
    for index, rows in reserved_by_layer.items():
        layer_quotas[index] = max(layer_quotas[index], len(rows))
    while sum(layer_quotas.values()) > target_count:
        donor = next((index for index, _layer in non_empty_layers
                      if layer_quotas[index] > len(reserved_by_layer[index])), None)
        if donor is None:
            break
        layer_quotas[donor] -= 1
    if sum(layer_quotas.values()) > target_count:
        raise ValueError(
            "Complexity strata and operator-coverage constraints cannot both be satisfied; increase gp.top_k or the number of candidate formulas"
        )

    selected = []
    layer_details = []
    for raw_layer_index, layer in non_empty_layers:
        layer_index = raw_layer_index + 1
        chosen = []
        ranked_layer = sorted(
            layer, key=lambda item: item[1].get("predicted_rank", 10 ** 9)
        )
        selected_quota = layer_quotas[raw_layer_index]
        reserved = {id(row) for row in reserved_by_layer[raw_layer_index]}
        chosen_rows = list(reserved_by_layer[raw_layer_index])
        for value, row in ranked_layer:
            if len(chosen_rows) >= selected_quota:
                break
            if id(row) not in reserved:
                chosen_rows.append(row)
        for row in chosen_rows:
            value = next(item_value for item_value, item_row in layer if item_row is row)
            annotated = dict(row)
            annotated["complexity_layer"] = layer_index
            annotated["complexity_value"] = value
            chosen.append(annotated)
        selected.extend(chosen)
        layer_details.append({
            "layer": layer_index,
            "available_count": len(layer),
            "selected_count": len(chosen),
            "complexity_min": min(value for value, _row in layer),
            "complexity_max": max(value for value, _row in layer),
            "complexity_interval_min": minimum_complexity + (
                (maximum_complexity - minimum_complexity)
                * (layer_index - 1) / requested_layers
            ),
            "complexity_interval_max": minimum_complexity + (
                (maximum_complexity - minimum_complexity)
                * layer_index / requested_layers
            ),
            "formula_ids": [row["formula_id"] for row in chosen],
            "operators": sorted({label for row in chosen for label in operator_labels(row)}),
        })
    selected.sort(key=lambda row: (
        row.get("complexity_layer", 10 ** 9),
        row.get("predicted_rank", 10 ** 9),
    ))
    return selected, {
        "enabled": True, "input_count": len(ranking),
        "layer_count": layer_count,
        "per_layer": [layer_quotas[index] for index, _layer in non_empty_layers],
        "selected_count": len(selected), "target_top_k": target,
        "available_operators": sorted(available_operators),
        "operator_coverage": operator_coverage,
        "layers": layer_details,
    }


def collect_initial_lm_mean_r2(payload: dict) -> list[dict]:
    """Aggregate all initial LM rows to one train/validation mean-R² row per formula."""
    raw_rows = [
        row for key in ("single_best", "combination_best")
        for row in payload.get(key, [])
        if isinstance(row, dict)
    ]
    grouped = {}
    for row in raw_rows:
        formula_id = row.get("formula_id")
        bucket = grouped.setdefault(formula_id, {
            "formula_id": formula_id,
            "formula_family": "combination" if row.get("combination_id")
            or row.get("combination_type") else "single",
            "readable_expression": row.get("readable_expression"),
            "datasets": {},
        })
        nested = row.get("dataset_fit_results") or [row]
        for dataset in nested:
            name = dataset.get("dataset_name", row.get("dataset_name", "dataset"))
            bucket["datasets"][str(name)] = {
                "train_r2": dataset.get("r2"),
                "validation_r2": dataset.get("validation_r2"),
            }
    output = []
    for row in grouped.values():
        datasets = list(row["datasets"].values())
        train_values = [float(item["train_r2"]) for item in datasets
                        if item["train_r2"] is not None and math.isfinite(float(item["train_r2"]))]
        validation_values = [float(item["validation_r2"]) for item in datasets
                             if item["validation_r2"] is not None and math.isfinite(float(item["validation_r2"]))]
        output.append({
            "formula_id": row["formula_id"],
            "formula_family": row["formula_family"],
            "readable_expression": row["readable_expression"],
            "mean_r2": float(np.mean(train_values)) if train_values else None,
            "validation_mean_r2": float(np.mean(validation_values)) if validation_values else None,
            "dataset_r2": row["datasets"],
        })
    return sorted(output, key=lambda row: str(row["formula_id"]))


def print_fine_search_results(results: Sequence[dict]) -> None:
    """Print final fine-search metrics using the full-dataset R²."""
    print("\nFine search complete; final results (selected by total residual, showing overall R²):", flush=True)
    if not results:
        print("   (no fine-search results available)", flush=True)
        return
    for index, row in enumerate(results, start=1):
        pooled_r2 = row.get("pooled_r2", row.get("r2"))
        loss = row.get("loss")
        r2_text = "None" if pooled_r2 is None else f"{float(pooled_r2):.8f}"
        loss_text = "None" if loss is None else f"{float(loss):.6g}"
        split = row.get("split_index")
        print(
            f"  #{index:>3} {row.get('formula_id')}: "
            f"overall R²={r2_text}, total residual={loss_text}, breakpoint={split}, "
            f"{row.get('readable_expression')}",
            flush=True,
        )


def refine_gp_piecewise_splits(
    gp_results: list[dict], X: np.ndarray, y: np.ndarray, fitting_config: dict,
    rng: np.random.Generator, segmentation_config: dict,
) -> None:
    """Refine split positions for final GP formulas while keeping one structure."""
    for row in tqdm(gp_results, desc="Final GP formula fine piecewise fitting"):
        if not row.get("piecewise", False):
            continue
        previous_result = dict(row)
        previous_split = row.get("split_index")
        record = {"id": row["formula_id"], "readable_expression": row["readable_expression"]}
        # Fine screening starts from the best parameters found during the
        # preceding coarse GP fit.  Keep exactly two LM initializations here:
        # one exact warm start and one lightly perturbed retry.
        fine_fitting_config = dict(fitting_config)
        fine_fitting_config["lm_initializations"] = 2
        warm_starts = {
            item.get("dataset_name"): item.get("parameters")
            for item in row.get("dataset_fit_results", [])
            if item.get("dataset_name") in {"segment_1", "segment_2"}
               and item.get("parameters") is not None
        }
        _attempts, best = fit_fine_piecewise_formula(
            record, X, y, fine_fitting_config, rng, segmentation_config,
            warm_start_parameters_by_dataset=warm_starts or None,
            preserve_split_index=previous_split,
        )
        if not _result_not_worse(best, previous_result):
            row["fine_segmentation_rejected"] = True
            row["fine_segmentation_rejection_reason"] = "fine_r2_worse_than_coarse"
            continue
        r2_values = [item.get("r2") for item in best.get("dataset_fit_results", [])
                     if item.get("r2") is not None]
        row.update(best)
        row.update(
            mean_r2=(float(np.mean(r2_values)) if r2_values else best.get("r2")),
            r2_std=(float(np.std(r2_values)) if r2_values else None),
            coarse_split_index=previous_split,
        )


def choose_fit_samples(candidates: Sequence[dict], combinations: Sequence[dict], cfg: dict,
                       rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Sample single and binary structures independently.

    A binary formula can be fitted without first fitting either of its
    component formulas.  Keeping the two samples independent makes
    ``single_sample_count`` and ``combination_sample_count`` control only
    their corresponding LM loops.
    """
    single_count = min(int(cfg["single_sample_count"]), len(candidates))
    combination_count = min(int(cfg["combination_sample_count"]), len(combinations))
    selected_candidates = rng.sample(list(candidates), single_count)
    sampled_combinations = rng.sample(list(combinations), combination_count)
    return selected_candidates, sampled_combinations


def choose_fit_samples_by_dataset(
    candidates: Sequence[dict], combinations: Sequence[dict], cfg: dict,
    rng: random.Random, dataset_groups: Sequence[dict],
) -> dict[str, tuple[list[dict], list[dict]]]:
    """Sample formula structures independently for each dataset."""
    return {
        group["name"]: choose_fit_samples(candidates, combinations, cfg, rng)
        for group in dataset_groups
    }


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
                             else value for key, value in row.items()})


def _existing_artifact_path(directory: Path, filename: str) -> Path:
    """Resolve an artifact name, including the legacy prefixed form."""
    path = directory / filename
    if path.exists():
        return path
    legacy_names = []
    if filename.startswith(("pareto_", "log_merge_", "constant_significance_")):
        legacy_names.append(f"gp_{filename}")
    elif filename.startswith(("tree_vs_random_", "validation_", "test_", "hypervolume_")):
        legacy_names.extend((f"mode2_{filename}", f"mode4_{filename}"))
    for legacy_name in legacy_names:
        legacy_path = directory / legacy_name
        if legacy_path.exists():
            return legacy_path
    return path



def _piecewise_segment_rows(
    formula_rows: Sequence[dict], formula_key: str, segment_name: str,
) -> list[dict]:
    """Flatten one independently fitted segment for its dedicated CSV."""
    rows = []
    for formula_row in formula_rows:
        segment_fit = next((
            item for item in (formula_row.get("dataset_fit_results") or [])
            if item.get("dataset_name") == segment_name
        ), None)
        if not isinstance(segment_fit, dict):
            continue
        rows.append({
            "segment": segment_name,
            "segment_sample_count": segment_fit.get("segment_sample_count"),
            "formula_family": formula_row.get("structure_type"),
            "formula_id": formula_row.get("formula_id"),
            "readable_expression": formula_row.get("readable_expression"),
            "sympy_expression": formula_row.get("sympy_expression"),
            "parameter_names": segment_fit.get("parameter_names"),
            "parameters": segment_fit.get("parameters"),
            "r2": segment_fit.get("r2"),
            "mse": segment_fit.get("mse"),
            "rmse": segment_fit.get("rmse"),
            "loss": segment_fit.get("loss"),
            "converged": segment_fit.get("converged"),
            "failure_reason": segment_fit.get("failure_reason"),
        })
    rows.sort(key=lambda row: (
        float("inf") if row.get("loss") is None else float(row["loss"])
    ))
    for rank, row in enumerate(rows, start=1):
        row["segment_rank"] = rank
    return rows



def save_outputs(output_dir: Path, payload: dict) -> None:
    metadata = payload.get("metadata", {})
    minimal = bool(metadata.get("minimal_output_artifacts", False))
    if metadata.get("suppress_large_artifacts", False) and not minimal:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    clean = _json_value(payload)
    clean.pop("piecewise_formula_pairs", None)
    # Internal handoff between the tree run and its random ablation; do not
    # expose the cached initial constant-test state as a result artifact.
    clean.pop("_mode2_initial_constant_state", None)
    piecewise_pairs = [*clean.get("single_best", []), *clean.get("combination_best", [])]

    # Minimal mode intentionally keeps only the training Pareto front and its
    # per-dataset fits.  Validation/HV artifacts are written by
    # save_validation_results() and the dedicated HV comparison functions.
    if minimal:
        _write_csv(output_dir / "pareto_front.csv",
                   clean.get("gp_pareto_front", []))
        flattened_gp = []
        for row in clean.get("gp_pareto_front", []):
            for dataset in row.get("dataset_fit_results", []):
                flattened_gp.append({
                    "pareto_rank": row.get("pareto_rank"),
                    "crowding_rank": row.get("crowding_rank"),
                    "crowding_distance": row.get("crowding_distance"),
                    "formula_id": row.get("formula_id"),
                    "readable_expression": row.get("readable_expression"),
                    "mean_r2": row.get("mean_r2"),
                    "r2_std": row.get("r2_std"),
                    "min_r2": row.get("min_r2"),
                    "pooled_r2": row.get("pooled_r2"),
                    "segment_prediction_difference": row.get(
                        "segment_prediction_difference"
                    ),
                    "predicted_segment_difference": row.get(
                        "predicted_segment_difference"
                    ),
                    "complexity": row.get("complexity"),
                    "dataset_name": dataset.get("dataset_name"),
                    "dataset_path": dataset.get("dataset_path"),
                    "parameter_names": dataset.get("parameter_names"),
                    "parameters": dataset.get("parameters"),
                    "r2": dataset.get("r2"),
                    "mse": dataset.get("mse"),
                    "rmse": dataset.get("rmse"),
                    "test_r2": row.get("test_r2"),
                    "test_mse": row.get("test_mse"),
                    "test_rmse": row.get("test_rmse"),
                    "loss": dataset.get("loss"),
                    "valid_samples": dataset.get("valid_samples"),
                    "initialization_index": dataset.get("initialization_index"),
                    "successful_initializations": dataset.get(
                        "successful_initializations"
                    ),
                    "converged": dataset.get("converged"),
                    "optimizer_success": dataset.get("optimizer_success"),
                    "failure_reason": dataset.get("failure_reason"),
                    "residual_autocorrelation": dataset.get(
                        "residual_autocorrelation"
                    ),
                })
        _write_csv(output_dir / "pareto_dataset_fits.csv", flattened_gp)
        _write_csv(
            output_dir / "piecewise_formula_segment_1.csv",
            _piecewise_segment_rows(
                piecewise_pairs, "left_formula", "segment_1"
            ),
        )
        _write_csv(
            output_dir / "piecewise_formula_segment_2.csv",
            _piecewise_segment_rows(
                piecewise_pairs, "right_formula", "segment_2"
            ),
        )
        # Keep the diagnostic checks attached to the retained Pareto front,
        # while omitting candidate-generation, initial-LM, ranking, and
        # Top-K artifacts from minimal-output Mode 4 seed directories.
        _write_csv(output_dir / "log_merge_checks.csv",
                   clean.get("gp_log_merge_checks", []))
        _write_csv(output_dir / "constant_significance_checks.csv",
                   clean.get("gp_constant_significance_checks", []))
        return
    if not metadata.get("suppress_results_json", False):
        (output_dir / "formula_discovery_results.json").write_text(
            json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
    mappings = {
        "bases.csv": "bases", "single_candidates.csv": "single_candidates",
        "binary_combinations.csv": "binary_combinations",
        "single_initializations.csv": "single_initializations",
        "single_best.csv": "single_best", "combination_best.csv": "combination_best",
        "structure_r2_ranking.csv": "structure_r2_ranking",
        "top_ranked_refit_results.csv": "top_ranked_refit_results",
        "pareto_front.csv": "gp_pareto_front",
        "log_merge_checks.csv": "gp_log_merge_checks",
        "constant_significance_checks.csv": "gp_constant_significance_checks",
        "top_ranked_log_merge_checks.csv": "top_ranked_log_merge_checks",
        "top_ranked_constant_significance_checks.csv": (
            "top_ranked_constant_significance_checks"
        ),
    }
    for filename, key in mappings.items():
        _write_csv(output_dir / filename, clean.get(key, []))
    _write_csv(
        output_dir / "piecewise_formula_segment_1.csv",
        _piecewise_segment_rows(
            piecewise_pairs, "left_formula", "segment_1"
        ),
    )
    _write_csv(
        output_dir / "piecewise_formula_segment_2.csv",
        _piecewise_segment_rows(
            piecewise_pairs, "right_formula", "segment_2"
        ),
    )
    ranking = clean.get("structure_r2_ranking", [])
    if ranking:
        _write_csv(output_dir / "single_formula_r2_ranking.csv", [
            row for row in ranking if row.get("formula_family") == "single"
        ])
        _write_csv(output_dir / "combination_formula_r2_ranking.csv", [
            row for row in ranking if row.get("formula_family") == "combination"
        ])
        _write_csv(output_dir / "top_ranked_formulas.csv", ranking[:100])
    flattened_refits = []
    for row in clean.get("top_ranked_refit_results", []):
        for dataset in row.get("dataset_fit_results", []):
            flattened_refits.append({
                "predicted_rank": row["predicted_rank"],
                "ranking_score": row["ranking_score"],
                "formula_id": row["formula_id"],
                "formula_family": row["formula_family"],
                "readable_expression": row["readable_expression"],
                "dataset_name": dataset.get("dataset_name"),
                "dataset_path": dataset.get("dataset_path"),
                "parameter_names": dataset.get("parameter_names"),
                "parameters": dataset.get("parameters"),
                "r2": dataset.get("r2"), "mse": dataset.get("mse"),
                "test_r2": row.get("test_r2"),
                "test_mse": row.get("test_mse"),
                "test_rmse": row.get("test_rmse"),
                "rmse": dataset.get("rmse"),
                "valid_samples": dataset.get("valid_samples"),
                "initialization_index": dataset.get("initialization_index"),
                "converged": dataset.get("converged"),
                "failure_reason": dataset.get("failure_reason"),
                "residual_autocorrelation": dataset.get("residual_autocorrelation"),
            })
    if flattened_refits:
        _write_csv(output_dir / "top_ranked_dataset_fits.csv", flattened_refits)
    flattened_gp = []
    for row in clean.get("gp_pareto_front", []):
        for dataset in row.get("dataset_fit_results", []):
            flattened_gp.append({
                "pareto_rank": row.get("pareto_rank"),
                "crowding_rank": row.get("crowding_rank"),
                "crowding_distance": row.get("crowding_distance"),
                "crowding_distance_label": row.get("crowding_distance_label"),
                "formula_id": row.get("formula_id"),
                "readable_expression": row.get("readable_expression"),
                "mean_r2": row.get("mean_r2"),
                "r2_std": row.get("r2_std"),
                "min_r2": row.get("min_r2"),
                "pooled_r2": row.get("pooled_r2"),
                "segment_prediction_difference": row.get(
                    "segment_prediction_difference"
                ),
                "complexity": row.get("complexity"),
                "dataset_name": dataset.get("dataset_name"),
                "dataset_path": dataset.get("dataset_path"),
                "parameter_names": dataset.get("parameter_names"),
                "parameters": dataset.get("parameters"),
                "r2": dataset.get("r2"),
                "mse": dataset.get("mse"),
                "rmse": dataset.get("rmse"),
                "test_r2": row.get("test_r2"),
                "test_mse": row.get("test_mse"),
                "test_rmse": row.get("test_rmse"),
                "loss": dataset.get("loss"),
                "valid_samples": dataset.get("valid_samples"),
                "initialization_index": dataset.get("initialization_index"),
                "successful_initializations": dataset.get("successful_initializations"),
                "converged": dataset.get("converged"),
                "optimizer_success": dataset.get("optimizer_success"),
                "failure_reason": dataset.get("failure_reason"),
                "residual_autocorrelation": dataset.get("residual_autocorrelation"),
            })
    if flattened_gp:
        _write_csv(output_dir / "pareto_dataset_fits.csv", flattened_gp)


def save_checkpoint(output_dir: Path, payload: dict, stage: str) -> None:
    """Disabled intermediate checkpoint persistence.

    Checkpoints are not read by the in-process pipeline; the completed run is
    persisted separately by ``save_outputs``.  Keep this function and its call
    sites as no-ops so disabling checkpoint I/O cannot alter the search,
    fitting, ranking, or final-output control flow.
    """
    return


def load_configured_data(data_config: dict):
    mode = data_config.get("mode", "csv")
    if mode not in ("csv", "realdata_mat"):
        raise ValueError("data.mode must be csv or realdata_mat")
    if mode == "realdata_mat" and not data_config.get("csv_path"):
        raise ValueError("data.csv_path is required for realdata_mat mode (CSV input)")

    loaded = load_joint_csv_data(data_config)
    if _is_measured_pathloss_profile(data_config):
        X, y, feature_names, groups = loaded
        if len(feature_names) != 1:
            raise ValueError("realdata_mat measured-pathloss input must have exactly one feature")
        # The legacy real-data reader sorted measurements stably by distance.
        # Keep that ordering while reading the preprocessed CSV instead of MAT.
        order = np.argsort(X[:, 0], kind="stable")
        X, y = X[order], y[order]
        for group in groups:
            group["profile"] = MEASURED_PATHLOSS_PROFILE
            group["feature"] = feature_names[0]
            group["sorted_by_feature"] = True
        return X, y, feature_names, groups
    return loaded


def _load_validation_data(validation_config: dict, training_data_config: dict):
    """Load an independent validation set without changing training data."""
    cfg = copy.deepcopy(validation_config)
    cfg.setdefault("features", training_data_config.get("features"))
    cfg.setdefault("target", training_data_config.get("target"))
    mode = cfg.get("mode", "csv")
    if mode == "csv":
        return load_joint_csv_data(cfg)
    raise ValueError("validation.mode must be csv")


def _split_validation_data(data, seed: int):
    """Split each configured validation dataset into validation/test halves."""
    X, y, feature_names, groups = data
    rng = np.random.default_rng(int(seed))
    validation_indices, test_indices, split_groups = [], [], []
    offset = 0
    split_metadata = []
    for group in groups:
        start, end = int(group["start"]), int(group["end"])
        indices = np.arange(start, end, dtype=int)
        rng.shuffle(indices)
        validation_count = len(indices) // 2
        validation_part = np.sort(indices[:validation_count])
        test_part = np.sort(indices[validation_count:])
        validation_indices.extend(validation_part.tolist())
        test_indices.extend(test_part.tolist())
        split_groups.append(dict(group, start=offset, end=offset + len(validation_part),
                                 samples=len(validation_part)))
        split_metadata.append({
            "name": str(group["name"]),
            "source_path": group.get("path"),
            "source_offset": start,
            "source_samples": len(indices),
            "validation_samples": len(validation_part),
            "test_samples": len(test_part),
            "validation_source_indices": validation_part.tolist(),
            "test_source_indices": test_part.tolist(),
        })
        offset += len(validation_part)
    validation_indices = np.asarray(validation_indices, dtype=int)
    test_groups, offset = [], 0
    for group in groups:
        count = len(split_metadata[len(test_groups)]["test_source_indices"])
        test_groups.append(dict(group, start=offset, end=offset + count,
                                samples=count))
        offset += count
    test_indices = np.asarray(test_indices, dtype=int)
    return (
        (X[validation_indices], y[validation_indices], feature_names, split_groups),
        (X[test_indices], y[test_indices], feature_names, test_groups),
        split_metadata,
    )


def _split_by_piecewise_segments(data, seed: int, segmentation_config: dict,
                                             train_fraction: float = 0.5,
                                             validation_fraction: float = 0.25,
                                             test_fraction: float = 0.25):
    """Split each side of the configured physical breakpoint independently."""
    fractions = (float(train_fraction), float(validation_fraction), float(test_fraction))
    if (any(value <= 0 for value in fractions)
            or not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-12)):
        raise ValueError("Automatic data-split fractions must all be positive and sum to 1")
    X, y, feature_names, groups = data
    if len(groups) != 1:
        raise ValueError("Piecewise splitting requires a single data source")
    prior = _breakpoint_prior(segmentation_config)
    if prior is None:
        raise ValueError("Piecewise splitting requires breakpoint_prior to be enabled")
    threshold = float(prior["breakpoint_value"])
    source_group = groups[0]
    source_indices = np.arange(int(source_group["start"]), int(source_group["end"]), dtype=int)
    segment_indices = (source_indices[X[source_indices, 0] <= threshold], source_indices[X[source_indices, 0] > threshold])
    rng = np.random.default_rng(int(seed))
    selected = [[], [], []]
    metadata = {"strategy": "independent_per_piecewise_segment", "seed": int(seed),
                "source_path": source_group.get("path"),
                "train_fraction": fractions[0], "validation_fraction": fractions[1],
                "test_fraction": fractions[2], "breakpoint_value": threshold, "segments": []}
    for segment_number, indices in enumerate(segment_indices, start=1):
        if len(indices) < 3:
            raise ValueError(f"segment_{segment_number} requires at least 3 samples to split")
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        train_n = max(1, int(math.floor(len(indices) * fractions[0])))
        validation_n = max(1, int(math.floor(len(indices) * fractions[1])))
        test_n = len(indices) - train_n - validation_n
        if test_n < 1:
            test_n = 1
            if train_n >= validation_n and train_n > 1: train_n -= 1
            else: validation_n -= 1
        counts = (train_n, validation_n, test_n)
        offset = 0
        segment_meta = {
            "name": f"segment_{segment_number}",
            "source_offset": int(source_group.get("start", 0)),
            "source_samples": len(indices), "counts": {}, "source_indices": {},
        }
        for label, count, partition in zip(("train", "validation", "test"), counts, selected):
            chosen = np.sort(shuffled[offset:offset + count])
            partition.extend(chosen.tolist())
            segment_meta["counts"][label] = len(chosen)
            segment_meta["source_indices"][label] = chosen.tolist()
            offset += count
        metadata["segments"].append(segment_meta)
    split_data = []
    for label, indices in zip(("train", "validation", "test"), selected):
        indices = np.asarray(sorted(indices), dtype=int)
        group = dict(source_group, start=0, end=len(indices), samples=len(indices), split=label)
        split_data.append((X[indices], y[indices], feature_names, [group]))
    return split_data[0], split_data[1], split_data[2], metadata


def _split_train_validation_test_data(
    data, seed: int, train_fraction: float = 0.5,
    validation_fraction: float = 0.25, test_fraction: float = 0.25,
):
    """Randomly split each configured source into train/validation/test sets."""
    fractions = (float(train_fraction), float(validation_fraction), float(test_fraction))
    if any(value <= 0 for value in fractions):
        raise ValueError("Automatic data-split fractions must all be positive")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Automatic data-split fractions must sum to 1")
    X, y, feature_names, groups = data
    rng = np.random.default_rng(int(seed))
    partitions = [[], [], []]
    split_metadata = []
    for group in groups:
        start, end = int(group["start"]), int(group["end"])
        indices = np.arange(start, end, dtype=int)
        if len(indices) < 3:
            raise ValueError(f"Dataset {group.get('name')} requires at least 3 samples for automatic splitting")
        rng.shuffle(indices)
        train_count = int(math.floor(len(indices) * fractions[0]))
        validation_count = int(math.floor(len(indices) * fractions[1]))
        train_count = max(1, train_count)
        validation_count = max(1, validation_count)
        test_count = len(indices) - train_count - validation_count
        if test_count < 1:
            test_count = 1
            if train_count >= validation_count and train_count > 1:
                train_count -= 1
            else:
                validation_count -= 1
        counts = (train_count, validation_count, test_count)
        offset = 0
        metadata_item = {
            "name": str(group["name"]),
            "source_path": group.get("path"),
            "source_offset": start,
            "source_samples": len(indices),
            "counts": {"train": train_count, "validation": validation_count, "test": test_count},
            "source_indices": {},
        }
        for label, count, partition in zip(
            ("train", "validation", "test"), counts, partitions
        ):
            selected = np.sort(indices[offset:offset + count])
            partition.append((group, selected))
            metadata_item["source_indices"][label] = selected.tolist()
            offset += count
        split_metadata.append(metadata_item)

    split_data = []
    for partition_index, partition in enumerate(partitions):
        X_parts, y_parts, split_groups = [], [], []
        offset = 0
        for group, indices in partition:
            X_parts.append(X[indices])
            y_parts.append(y[indices])
            split_groups.append(dict(
                group, start=offset, end=offset + len(indices),
                samples=len(indices), split=("train", "validation", "test")[partition_index],
            ))
            offset += len(indices)
        split_data.append((
            np.concatenate(X_parts), np.concatenate(y_parts),
            feature_names, split_groups,
        ))
    return split_data[0], split_data[1], split_data[2], split_metadata


def save_split_indices(
    output_directory: str | Path, split_metadata: dict | list,
    *, seed: int, config_path: str | Path | None = None,
    data_mode: str | None = None, data_profile: str | None = None,
    split_config: dict | None = None,
) -> Path:
    """Persist the exact train/validation/test source-row indices for a run."""
    output_path = Path(output_directory).resolve()
    output_roots = (
        (PROJECT_ROOT / "results").resolve(),
        (PROJECT_ROOT / "outputs").resolve(),  # legacy experiment locations
    )
    relative_output = None
    for root_index, output_root in enumerate(output_roots):
        try:
            relative_output = output_path.relative_to(output_root)
            if root_index == 0:
                # Keep new results' split manifests separate from legacy
                # outputs that share the same experiment/run names.
                relative_output = Path("results") / relative_output
            break
        except ValueError:
            continue
    if relative_output is None:
        digest = hashlib.sha1(str(output_path).encode("utf-8")).hexdigest()[:10]
        relative_output = Path("custom") / f"{output_path.name}_{digest}"
    fixed_split_seed = "seed" in (split_config or {})
    shared_experiment_split = False
    if fixed_split_seed:
        parts = relative_output.parts
        for index, part in enumerate(parts):
            if re.fullmatch(r"seed_\d+", part):
                # One split is shared by every seed in this experiment.
                relative_output = Path(*parts[:index]) if index else Path(".")
                shared_experiment_split = True
                break

    # Split indices are local to each configured source, so files from
    # different datasets can be reproduced without interpreting concatenated
    # in-memory offsets. They index rows after ProSEE's input validation and
    # invalid-value filtering.
    normalized = []
    if isinstance(split_metadata, dict) and isinstance(
        split_metadata.get("segments"), list
    ):
        segments = split_metadata["segments"]
        top_metadata = {key: value for key, value in split_metadata.items()
                        if key != "segments"}
        for segment in segments:
            offset = int(segment.get("source_offset", 0))
            normalized.append({
                **{key: value for key, value in segment.items()
                   if key != "source_offset"},
                "source_indices": {
                    label: [int(index) - offset for index in indices]
                    for label, indices in segment.get("source_indices", {}).items()
                },
            })
    else:
        top_metadata = {}
        for source in split_metadata if isinstance(split_metadata, list) else []:
            offset = int(source.get("source_offset", 0))
            local_indices = {}
            if "source_indices" in source:
                local_indices = {
                    label: [int(index) - offset for index in indices]
                    for label, indices in source["source_indices"].items()
                }
            elif "validation_source_indices" in source:
                local_indices = {
                    "validation": [int(index) - offset for index in source.get(
                        "validation_source_indices", [])],
                    "test": [int(index) - offset for index in source.get(
                        "test_source_indices", [])],
                }
            normalized.append({
                key: value for key, value in source.items()
                if key not in {"source_indices", "validation_source_indices",
                               "test_source_indices", "source_offset"}
            } | {"source_indices": local_indices})

    payload = {
        "schema_version": 1,
        "experiment_seed": int(seed),
        "split_seed": int(top_metadata.get(
            "seed", (split_config or {}).get("seed", seed)
        )),
        "data_mode": data_mode,
        "data_profile": data_profile,
        "config_path": (str(Path(config_path).resolve())
                        if config_path is not None else None),
        "output_directory": str(output_path),
        "split_config": _json_value(split_config or {}),
        "index_semantics": (
            "zero-based row positions within each source after ProSEE input "
            "validation and invalid-value filtering"
        ),
        "split_metadata": _json_value(top_metadata),
        "datasets": _json_value(normalized),
    }
    split_path = PROJECT_ROOT / "data" / "split" / relative_output / "split_indices.json"
    if shared_experiment_split and split_path.is_file():
        return split_path
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return split_path


def _validation_parameters(row: dict, validation_group: dict,
                           training_groups: Sequence[dict]):
    """Resolve frozen training parameters for one validation dataset."""
    parameters = row.get("parameters")
    if isinstance(parameters, dict):
        name = str(validation_group["name"])
        if name in parameters:
            return parameters[name]
        if len(parameters) == 1:
            return next(iter(parameters.values()))
        train_names = [str(group["name"]) for group in training_groups]
        if name in train_names and name in parameters:
            return parameters[name]
        return None
    return parameters


def _evaluate_validation_row(row: dict, X: np.ndarray, y: np.ndarray,
                             validation_groups: Sequence[dict],
                             training_groups: Sequence[dict],
                             fitting_config: dict) -> dict | None:
    """Evaluate one frozen fitted formula on all validation datasets."""
    try:
        expression = sp.sympify(row["readable_expression"])
        prediction = np.full(len(y), np.nan, dtype=float)
        valid = np.zeros(len(y), dtype=bool)

        # realdata_mat piecewise fits have two independent parameter sets and
        # a split value learned on the sorted training data.  Validation data
        # is split by that same physical boundary, rather than by the
        # original dataset name (which has no segment_1/segment_2 parameters).
        segment_rows = {
            str(item.get("dataset_name")): item
            for item in (row.get("dataset_fit_results") or [])
            if str(item.get("dataset_name")) in {"segment_1", "segment_2"}
        }
        # When a physical breakpoint prior is configured, every partition
        # must use that same physical value.  The fitted ``split_value`` is a
        # training-only sample-boundary midpoint and can differ slightly
        # because the training samples are discrete.  Without a prior, keep
        # the data-driven training split as the fallback.
        piecewise_split_value = row.get(
            "split_prior_value", row.get("split_value")
        )
        is_piecewise = (
            bool(row.get("piecewise"))
            and piecewise_split_value is not None
            and len(segment_rows) == 2
        )
        if is_piecewise:
            split_value = float(piecewise_split_value)
            for group in validation_groups:
                start, end = int(group["start"]), int(group["end"])
                x_values = np.asarray(X[start:end, 0], dtype=float)
                for segment_name, segment_mask in (
                    ("segment_1", x_values <= split_value),
                    ("segment_2", x_values > split_value),
                ):
                    indices = np.flatnonzero(segment_mask)
                    if indices.size == 0:
                        continue
                    segment = segment_rows[segment_name]
                    parameters = segment.get("parameters")
                    if parameters is None:
                        return None
                    evaluator = FormulaEvaluator(
                        expression, X[start:end][indices],
                        float(fitting_config.get("max_abs_value", 1e150)),
                        float(fitting_config.get("complex_tolerance", 1e-12)),
                    )
                    values, part_valid, _ = evaluator(parameters)
                    # Advanced indexing on a sliced array creates a copy.
                    # Write via global indices so values reach the metric arrays.
                    global_indices = start + indices
                    prediction[global_indices] = values
                    valid[global_indices] = part_valid
        else:
            for group in validation_groups:
                start, end = int(group["start"]), int(group["end"])
                parameters = _validation_parameters(row, group, training_groups)
                if parameters is None:
                    return None
                evaluator = FormulaEvaluator(
                    expression, X[start:end],
                    float(fitting_config.get("max_abs_value", 1e150)),
                    float(fitting_config.get("complex_tolerance", 1e-12)),
                )
                values, part_valid, _ = evaluator(parameters)
                prediction[start:end] = values
                valid[start:end] = part_valid
        metrics = compute_joint_metrics(
            y, prediction, valid, validation_groups,
            fitting_config.get("joint_r2_aggregation", "minimum"),
        )
        r2_values = [float(item["r2"]) for item in metrics["dataset_metrics"]
                     if item.get("r2") is not None and math.isfinite(float(item["r2"]))]
        metrics["mean_r2"] = float(np.mean(r2_values)) if r2_values else None
        metrics["r2_std"] = float(np.std(r2_values)) if r2_values else None
        metrics["min_r2"] = min(r2_values) if r2_values else None
        return _json_value(metrics)
    except (ArithmeticError, TypeError, ValueError, KeyError, sp.SympifyError):
        return None


def _metrics_for_saved_result(metrics: dict, data_mode: str) -> dict:
    """Use arithmetic per-dataset RMSE in saved CSV-mode result rows.

    The metric calculation retains the pooled RMSE for all internal uses.
    Only the RMSE value persisted in the result front is changed for CSV
    multi-dataset fits.  MAT-mode output keeps the existing pooled value.
    """
    if data_mode != "csv":
        return metrics
    dataset_metrics = metrics.get("dataset_metrics") or []
    rmse_values = []
    for item in dataset_metrics:
        try:
            value = float(item.get("rmse"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            rmse_values.append(value)
    if not rmse_values:
        return metrics
    saved_metrics = dict(metrics)
    saved_metrics["rmse"] = float(np.mean(rmse_values))
    return saved_metrics


def _validation_hv_objectives(
    data_mode: str, metric_prefix: str = "validation"
) -> tuple[tuple[str, ...], set[str]]:
    # Validation is intentionally evaluated with the same objective family as
    # the ordinary non-piecewise Mode 3 front.  A piecewise MAT run has no
    # independent validation split in the new CSV validation interface.
    if _is_measured_pathloss_profile(data_mode):
        return (f"{metric_prefix}_pooled_r2", f"{metric_prefix}_r2_std", "complexity"), {
            f"{metric_prefix}_pooled_r2"
        }
    return (f"{metric_prefix}_mean_r2", f"{metric_prefix}_r2_std", "complexity"), {
        f"{metric_prefix}_mean_r2"
    }


def _validation_hv_bounds(rows: Sequence[dict], fields: Sequence[str]):
    bounds = {}
    for field in fields:
        values = [float(row[field]) for row in rows
                  if row.get(field) is not None and math.isfinite(float(row[field]))]
        if not values:
            continue
        # Use the exact empirical range.  Callers that need a common scale
        # pass the union of all relevant histories before reaching this
        # helper; no extra floating margin is added.
        bounds[field] = (min(values), max(values))
    return bounds


def _hv_reference_parameter_rows(
    objective_fields: Sequence[str], maximize_fields: set[str], bounds: dict,
    evaluation_label: str, normalization_scope: str,
) -> tuple[list[dict], dict]:
    """Describe empirical bounds and raw/normalized HV reference coordinates."""
    rows = []
    raw_reference = {}
    normalized_reference = {}
    for field in objective_fields:
        if field not in bounds:
            continue
        lower, upper = map(float, bounds[field])
        maximize = field in maximize_fields
        direction = "maximize" if maximize else "minimize"
        raw = lower if maximize else upper
        span = upper - lower
        normalized = (
            (raw - lower) / span if maximize else (upper - raw) / span
        ) if span > 0 else None
        raw_reference[field] = raw
        normalized_reference[field] = normalized
        rows.append({
            "evaluation": evaluation_label,
            "objective": field,
            "direction": direction,
            "empirical_lower_bound": lower,
            "empirical_upper_bound": upper,
            "raw_reference": raw,
            "normalized_reference": normalized,
            "zero_span_dimension_excluded": span <= 0,
            "normalization_scope": normalization_scope,
        })
    return rows, {
        "evaluation": evaluation_label,
        "objective_order": list(objective_fields),
        "directions": {
            row["objective"]: row["direction"] for row in rows
        },
        "empirical_bounds": {
            field: {"min": float(bounds[field][0]), "max": float(bounds[field][1])}
            for field in objective_fields if field in bounds
        },
        "raw_reference": raw_reference,
        "normalized_reference": normalized_reference,
        "zero_span_objectives": [
            row["objective"] for row in rows
            if row["zero_span_dimension_excluded"]
        ],
        "normalization_scope": normalization_scope,
    }


def _write_hv_reference_parameters(
    output_dir: Path, stem: str, objective_fields: Sequence[str],
    maximize_fields: set[str], bounds: dict, evaluation_label: str,
    normalization_scope: str,
) -> dict:
    """Persist the HV reference settings beside the corresponding results."""
    rows, details = _hv_reference_parameter_rows(
        objective_fields, maximize_fields, bounds, evaluation_label,
        normalization_scope,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    json_path.write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = [
        "evaluation", "objective", "direction", "empirical_lower_bound",
        "empirical_upper_bound", "raw_reference", "normalized_reference",
        "zero_span_dimension_excluded", "normalization_scope",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    details["json_file"] = str(json_path)
    details["csv_file"] = str(csv_path)
    return details


def _attach_training_validation_mean_r2_gap(
    row: dict, metric_prefix: str = "validation",
) -> None:
    """Attach the absolute training/validation mean-R² generalization gap."""
    training = row.get("mean_r2", _mode2_mean_r2(row))
    validation = row.get(f"{metric_prefix}_mean_r2")
    try:
        training, validation = float(training), float(validation)
    except (TypeError, ValueError):
        return
    if math.isfinite(training) and math.isfinite(validation):
        row[f"{metric_prefix}_training_mean_r2_gap"] = abs(training - validation)


def _calculate_validation_hv(front: Sequence[dict], fields: Sequence[str],
                             maximize_fields: set[str], bounds: dict) -> float:
    rows = []
    for row in front:
        if not all(field in row and row[field] is not None for field in fields):
            continue
        if not all(math.isfinite(float(row[field])) for field in fields):
            continue
        tagged = dict(row)
        for field in fields:
            low, high = bounds[field]
            tagged[f"_hv_ref_{field}"] = (
                low if field in maximize_fields else high
            )
        rows.append(tagged)
    if not rows or len(bounds) != len(fields):
        return 0.0
    return _mode2_hypervolume(rows, fields, maximize_fields, bounds)


def _calculate_raw_validation_hv(front: Sequence[dict], fields: Sequence[str],
                                 maximize_fields: set[str], bounds: dict) -> float:
    """Calculate raw-unit HV using this seed's all-iteration worst corner."""
    rows = []
    for row in front:
        if not all(field in row and row[field] is not None for field in fields):
            continue
        try:
            if not all(math.isfinite(float(row[field])) for field in fields):
                continue
            tagged = dict(row)
            for field in fields:
                low, high = bounds[field]
                tagged[f"_hv_ref_{field}"] = (
                    low if field in maximize_fields else high
                )
            rows.append(tagged)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    if not rows or len(bounds) != len(fields):
        return 0.0
    # No min-max transform here: the result remains in the product units of
    # the original objectives. This per-seed raw curve is diagnostic only;
    # cross-seed comparisons use the shared normalized calculation.
    return _mode2_hypervolume(rows, fields, maximize_fields)






def save_validation_results(payload: dict, validation_config: dict,
                            X: np.ndarray, y: np.ndarray,
                            validation_groups: Sequence[dict],
                            training_groups: Sequence[dict],
                            fitting_config: dict, output_dir: Path,
                            data_mode: str,
                            experiment_variant: str = "full_feature_baseline",
                            ablation_name: str | None = None,
                            hv_bounds: dict | None = None,
                            evaluation_label: str = "validation") -> dict:
    """Evaluate every saved Mode 3 front on frozen validation data."""
    history = payload.get("mode2_hypervolume_history", [])
    if not history:
        history = [{"iteration": 0, "front": payload.get("gp_pareto_front", [])}]
    evaluated_history = []
    all_rows = []
    metric_cache = {}

    def evaluate_cached(source):
        formula_key = str(
            source.get("formula_id")
            or source.get("id")
            or source.get("readable_expression", "")
        )
        if formula_key not in metric_cache:
            metric_cache[formula_key] = _evaluate_validation_row(
                source, X, y, validation_groups, training_groups, fitting_config
            )
        return metric_cache[formula_key]

    for item in history:
        evaluated_front = []
        for source in item.get("front", []):
            row = copy.deepcopy(source)
            metrics = evaluate_cached(source)
            if metrics is not None:
                saved_metrics = _metrics_for_saved_result(metrics, data_mode)
                for key, value in saved_metrics.items():
                    row[f"{evaluation_label}_{key}"] = value
            _attach_training_validation_mean_r2_gap(row, evaluation_label)
            row[f"{evaluation_label}_complexity"] = row.get("complexity")
            if row[f"{evaluation_label}_complexity"] is None and row.get("readable_expression"):
                row[f"{evaluation_label}_complexity"] = formula_complexity(
                    sp.sympify(row["readable_expression"])
                )
            # Keep the exact names used by the HV objective selector.
            row["complexity"] = row.get(f"{evaluation_label}_complexity")
            evaluated_front.append(row)
            all_rows.append(row)
        evaluated_history.append({
            "iteration": int(item.get("iteration", 0)),
            "front": evaluated_front,
        })
    final_front = []
    for source in payload.get("gp_pareto_front", []):
        row = copy.deepcopy(source)
        metrics = evaluate_cached(source)
        if metrics is not None:
            saved_metrics = _metrics_for_saved_result(metrics, data_mode)
            for key, value in saved_metrics.items():
                row[f"{evaluation_label}_{key}"] = value
        _attach_training_validation_mean_r2_gap(row, evaluation_label)
        row[f"{evaluation_label}_complexity"] = row.get("complexity")
        if row[f"{evaluation_label}_complexity"] is None and row.get("readable_expression"):
            row[f"{evaluation_label}_complexity"] = formula_complexity(
                sp.sympify(row["readable_expression"])
            )
        row["complexity"] = row.get(f"{evaluation_label}_complexity")
        final_front.append(row)
        all_rows.append(row)
    fields, maximize_fields = _validation_hv_objectives(data_mode, evaluation_label)
    # A comparison run may provide one common set of bounds for all methods.
    # This keeps the standalone validation plot and the comparison plot on
    # exactly the same normalized HV scale.
    bounds = hv_bounds if hv_bounds is not None else _validation_hv_bounds(
        all_rows, fields
    )
    hv_rows = []
    raw_hv_rows = []
    for item in evaluated_history:
        front = item["front"]
        hv_rows.append({
            "iteration": item["iteration"],
            f"{evaluation_label}_hypervolume": _calculate_validation_hv(
                front, fields, maximize_fields, bounds
            ),
            f"{evaluation_label}_pareto_front_size": len(front),
        })
        raw_hv_rows.append({
            "iteration": item["iteration"],
            f"{evaluation_label}_raw_hypervolume": _calculate_raw_validation_hv(
                front, fields, maximize_fields, bounds
            ),
            f"{evaluation_label}_pareto_front_size": len(front),
            "raw_hypervolume_reference_bounds": {
                field: {"min": low, "max": high}
                for field, (low, high) in bounds.items()
            },
        })
    cfg = validation_config
    validation_dir = cfg.get("output_directory", "validation")
    validation_dir = Path(validation_dir)
    if not validation_dir.is_absolute():
        validation_dir = output_dir / validation_dir
    validation_dir.mkdir(parents=True, exist_ok=True)
    hypervolume_dir = validation_dir / "hypervolume"
    hypervolume_dir.mkdir(parents=True, exist_ok=True)
    # Validation filenames are intentionally fixed so runs remain comparable
    # and users only need to configure the validation data and output folder.
    results_name = f"{evaluation_label}_results.json"
    history_name = f"{evaluation_label}_history.json"
    front_name = f"{evaluation_label}_pareto_front.csv"
    dataset_fits_name = f"{evaluation_label}_pareto_dataset_fits.csv"
    hv_name = Path("hypervolume") / f"{evaluation_label}_hypervolume_by_iteration.csv"
    raw_hv_name = Path("hypervolume") / f"{evaluation_label}_raw_hypervolume_by_iteration.csv"
    hv_fronts_name = Path("hypervolume") / f"{evaluation_label}_hypervolume_fronts_by_iteration.csv"
    front_plot_name = f"{evaluation_label}_pareto_front.png"
    hv_plot_name = Path("hypervolume") / f"{evaluation_label}_hypervolume_by_iteration.png"
    result = {
        "metadata": {
            "enabled": True, "source": "frozen_training_formula_parameters",
            "experiment_variant": experiment_variant,
            "ablation_name": ablation_name,
            "training_data_path": payload.get("metadata", {}).get("data_path"),
            "evaluation_data": [dict(group) for group in validation_groups],
            "training_data": [dict(group) for group in training_groups],
            "objective_fields": list(fields),
            "maximize_fields": sorted(maximize_fields),
            "minimize_fields": [field for field in fields if field not in maximize_fields],
            "reference_point_normalized": {field: 0.0 for field in fields},
            "normalization_bounds": {
                field: {"min": low, "max": high}
                for field, (low, high) in bounds.items()
            },
            "raw_hypervolume_reference_bounds": {
                field: {"min": low, "max": high}
                for field, (low, high) in bounds.items()
            },
            "raw_hypervolume_reference_scope": (
                "this_seed_all_iterations_and_final_front"
            ),
        },
        "final_pareto_front": final_front,
        "per_iteration": hv_rows,
        "history": evaluated_history,
        "files": (
            {"pareto_front_csv": str(validation_dir / front_name),
             "hypervolume_csv": str(validation_dir / hv_name),
             "raw_hypervolume_csv": str(validation_dir / raw_hv_name),
             "hypervolume_fronts_csv": str(validation_dir / hv_fronts_name)}
            if evaluation_label == "test" else (
                {
                    "pareto_front_csv": str(validation_dir / front_name),
                    "pareto_dataset_fits_csv": str(validation_dir / dataset_fits_name),
                    "hypervolume_csv": str(validation_dir / hv_name),
                    "raw_hypervolume_csv": str(validation_dir / raw_hv_name),
                    "pareto_front_plot": str(validation_dir / front_plot_name),
                    "hypervolume_plot": str(validation_dir / hv_plot_name),
                }
                if evaluation_label == "validation" else {
                "results_json": str(validation_dir / results_name),
                "history_json": str(validation_dir / history_name),
                "pareto_front_csv": str(validation_dir / front_name),
                "pareto_dataset_fits_csv": str(validation_dir / dataset_fits_name),
                "hypervolume_csv": str(validation_dir / hv_name),
                "raw_hypervolume_csv": str(validation_dir / raw_hv_name),
                "pareto_front_plot": str(validation_dir / front_plot_name),
                "hypervolume_plot": str(validation_dir / hv_plot_name),
                }
            )
        ),
    }
    if evaluation_label not in {"test", "validation"}:
        (validation_dir / results_name).write_text(
            json.dumps(_json_value(result), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (validation_dir / history_name).write_text(
            json.dumps(_json_value({"history": evaluated_history}),
                       ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        # Held-out metrics and fronts are computed identically in memory.
        # Their detailed result/history JSON is omitted; the Pareto CSV stays.
        obsolete_names = [results_name, history_name]
        if evaluation_label == "test":
            obsolete_names.extend((
                dataset_fits_name, front_plot_name, hv_plot_name,
            ))
        for obsolete_name in obsolete_names:
            obsolete_path = validation_dir / obsolete_name
            if obsolete_path.is_file():
                obsolete_path.unlink()
    _write_csv(validation_dir / front_name, result["final_pareto_front"])
    dataset_fit_rows = []
    for row in result["final_pareto_front"]:
        for dataset in row.get(f"{evaluation_label}_dataset_metrics", []) or []:
            dataset_fit_rows.append({
                "formula_id": row.get("formula_id"),
                "readable_expression": row.get("readable_expression"),
                "complexity": row.get(f"{evaluation_label}_complexity"),
                f"{evaluation_label}_mean_r2": row.get(f"{evaluation_label}_mean_r2"),
                f"{evaluation_label}_r2_std": row.get(f"{evaluation_label}_r2_std"),
                f"{evaluation_label}_pooled_r2": row.get(f"{evaluation_label}_pooled_r2"),
                "dataset_name": dataset.get("name"),
                "dataset_path": dataset.get("path"),
                "r2": dataset.get("r2"),
                "mse": dataset.get("mse"),
                "rmse": dataset.get("rmse"),
                "loss": dataset.get("loss"),
                "valid_samples": dataset.get("valid_samples"),
            })
    # Keep only the numeric objectives needed to recalculate HV under a
    # shared experiment-wide normalization scale.  This is needed for both
    # validation and test histories; expressions, parameters, predictions,
    # and verbose iteration histories remain unpersisted.
    compact_front_rows = []
    for item in evaluated_history:
        iteration_front = item.get("front", [])
        if not iteration_front:
            compact_front_rows.append({
                "iteration": item["iteration"], "front_size": 0,
                **{field: None for field in fields},
            })
        else:
            compact_front_rows.extend({
                "iteration": item["iteration"],
                "front_size": len(iteration_front),
                **{field: row.get(field) for field in fields},
            } for row in iteration_front)
    _write_csv(validation_dir / hv_fronts_name, compact_front_rows)
    if evaluation_label == "test":
        _write_csv(validation_dir / hv_name, hv_rows)
    else:
        _write_csv(validation_dir / dataset_fits_name, dataset_fit_rows)
        _write_csv(validation_dir / hv_name, hv_rows)
    _write_csv(validation_dir / raw_hv_name, raw_hv_rows)
    try:
        if evaluation_label == "test":
            raise StopIteration
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        primary_field = (
            f"{evaluation_label}_pooled_r2" if _is_measured_pathloss_profile(data_mode)
            else f"{evaluation_label}_mean_r2"
        )
        plot_rows = [row for row in result["final_pareto_front"]
                     if row.get(primary_field) is not None
                     and row.get("validation_complexity") is not None]
        figure, axis = plt.subplots(figsize=(8, 5.5))
        axis.scatter(
            [row[f"{evaluation_label}_complexity"] for row in plot_rows],
            [row[primary_field] for row in plot_rows],
            color="tab:blue", alpha=0.75, s=42,
        )
        axis.set_xlabel("Formula complexity (lower is better)")
        axis.set_ylabel("Validation R² (higher is better)")
        axis.set_title(f"Validation Pareto front ({experiment_variant})")
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        figure.savefig(validation_dir / front_plot_name, dpi=180)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(8, 5.5))
        axis.plot(
            [item["iteration"] for item in hv_rows],
            [item[f"{evaluation_label}_hypervolume"] for item in hv_rows],
            marker="o", ms=8, linewidth=1.75, color="tab:blue",
        )
        axis.set_xlabel("Iteration", fontsize=15)
        axis.set_ylabel("Validation hypervolume", fontsize=15)
        axis.set_title(f"Validation HV by iteration ({experiment_variant})", fontsize=15)
        axis.tick_params(axis="both", labelsize=13)
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        figure.savefig(validation_dir / hv_plot_name, dpi=180)
        plt.close(figure)
    except ImportError:
        logging.warning("matplotlib is not installed; skipping validation-set visualization")
    except StopIteration:
        pass
    return result


def save_mode2_validation_comparison(
    tree_result: dict, random_result: dict, output_dir: Path,
    data_mode: str, hv_bounds: dict | None = None,
    evaluation_label: str = "validation", pysr_result: dict | None = None,
) -> dict:
    """Save the HV comparison for tree, random, and optional PySR runs.

    PySR is a one-shot baseline in the current pipeline: its saved result has
    a final Pareto front but no iterative history.  When present, represent it
    by a horizontal baseline spanning the same x-axis endpoints as the
    iterative tree/random histories.
    """
    fields, maximize_fields = _validation_hv_objectives(data_mode, evaluation_label)
    histories = {
        "tree_model": tree_result.get("history", []),
        "random_permutation": random_result.get("history", []),
    }
    # Validation uses the reporting R² window; test evaluates the frozen
    # validation-selected PySR front without test-side filtering.
    pysr_front = _pysr_front_for_hv_reporting(
        (pysr_result or {}).get("final_pareto_front", []),
        data_mode, evaluation_label,
    )
    pysr_history = (pysr_result or {}).get("history", []) or []
    has_pysr_history = len(pysr_history) > 1 and evaluation_label != "test"
    all_rows = [
        row
        for history in histories.values()
        for item in history
        for row in item.get("front", [])
    ]
    if has_pysr_history:
        all_rows.extend(
            row for item in pysr_history
            for row in _pysr_front_for_hv_reporting(
                item.get("front", []), data_mode, evaluation_label
            )
        )
    else:
        all_rows.extend(pysr_front)
    bounds = hv_bounds if hv_bounds is not None else _validation_hv_bounds(
        all_rows, fields
    )
    rows = []
    for method, history in histories.items():
        for item in history:
            front = item.get("front", [])
            rows.append({
                "method": method,
                "iteration": int(item.get("iteration", 0)),
                f"{evaluation_label}_hypervolume": _calculate_validation_hv(
                    front, fields, maximize_fields, bounds
                ),
                f"{evaluation_label}_pareto_front_size": len(front),
            })
    history_iterations = [
        int(item.get("iteration", 0))
        for history in histories.values()
        for item in history
    ]
    # Keep the final PySR report point in the CSV for numerical comparison;
    # it is intentionally not plotted.
    if pysr_front:
        start_iteration = min(history_iterations) if history_iterations else 0
        end_iteration = max(history_iterations) if history_iterations else start_iteration
        pysr_hv = _calculate_validation_hv(
            pysr_front, fields, maximize_fields, bounds
        )
        rows.append({
            "method": "pysr_baseline",
            "iteration": end_iteration,
            f"{evaluation_label}_hypervolume": pysr_hv,
            f"{evaluation_label}_pareto_front_size": len(pysr_front),
            "pysr_point_type": (
                "final_validation_selected_result" if evaluation_label == "test"
                else "final_r2_window_filtered_result"
            ),
        })
    hypervolume_dir = output_dir / "hypervolume"
    hypervolume_dir.mkdir(parents=True, exist_ok=True)
    csv_path = hypervolume_dir / f"{evaluation_label}_hypervolume_by_iteration.csv"
    _write_csv(csv_path, rows)
    plot_path = hypervolume_dir / f"{evaluation_label}_hypervolume_by_iteration.png"
    if rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Match the larger, presentation-ready dimensions used by Mode 5.
        figure, axis = plt.subplots(figsize=(11, 7))
        styles = {"tree_model": ("o", "Tree model"),
                  "random_permutation": ("s", "Random permutation")}
        for method, (marker, label) in styles.items():
            values = sorted(
                (row for row in rows if row["method"] == method),
                key=lambda row: row["iteration"],
            )
            axis.plot(
                [row["iteration"] for row in values],
                [row[f"{evaluation_label}_hypervolume"] for row in values],
                marker=marker, ms=8.0, linewidth=2.2, label=label,
            )
        axis.set_xlabel("Iteration", fontsize=15, fontweight="bold")
        axis.set_ylabel(
            f"{evaluation_label.capitalize()} hypervolume",
            fontsize=15, fontweight="bold",
        )
        axis.set_title(
            f"{evaluation_label.capitalize()} HV: tree model vs random ablation",
            fontsize=17, fontweight="bold",
        )
        axis.tick_params(axis="both", labelsize=12, width=1.2, length=5)
        axis.grid(True, alpha=0.3, linewidth=1.0)
        axis.legend(fontsize=13, frameon=False)
        figure.tight_layout()
        figure.savefig(plot_path, dpi=300)
        plt.close(figure)
    result = {
        "enabled": bool(rows),
        "objective_fields": list(fields),
        "maximize_fields": sorted(maximize_fields),
        "pysr_included": bool(pysr_front or has_pysr_history),
        "pysr_curve_type": (
            ("final_validation_selected_point" if evaluation_label == "test"
             else "final_r2_window_filtered_point") if pysr_front else None
        ),
        "pysr_hv_window": {
            "enabled": evaluation_label != "test",
            "primary_r2_gap": DEFAULT_MODE2_R2_WINDOW if evaluation_label != "test" else None,
            "scope": ("validation_selection_frozen_test_evaluation"
                       if evaluation_label == "test"
                       else "reporting_only_no_pysr_evolution_change"),
        },
        "hypervolume_csv": str(csv_path),
        "hypervolume_plot": str(plot_path),
    }
    comparison_path = hypervolume_dir / f"{evaluation_label}_comparison.json"
    comparison_path.write_text(
        json.dumps(_json_value({**result, "per_iteration": rows}),
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result

















def load_joint_csv_data(data_config: dict):
    """Load one or more CSVs and concatenate rows without adding a band feature."""
    sources = data_config.get("csv_files")
    if not sources:
        sources = [{
            "name": data_config.get("name", "dataset_1"),
            "csv_path": data_config["csv_path"],
            "features": data_config.get("features"),
            "target": data_config.get("target"),
        }]
    arrays, targets, groups = [], [], []
    expected_features = None
    offset = 0
    for index, source in enumerate(sources):
        if isinstance(source, str):
            source = {"csv_path": source}
        path = resolve_path(source["csv_path"])
        features = source.get("features", data_config.get("features"))
        target = source.get("target", data_config.get("target"))
        X_part, y_part, feature_names, target_name = load_csv_data(path, features, target)
        if expected_features is None:
            expected_features = list(feature_names)
        elif list(feature_names) != expected_features:
            raise ValueError(
                f"Feature columns and order must match across combined data: expected {expected_features}, "
                f"but {path} has {feature_names}"
            )
        name = str(source.get("name", path.stem or f"dataset_{index + 1}"))
        groups.append({"name": name, "path": str(path), "start": offset,
                       "end": offset + len(y_part), "samples": len(y_part),
                       "target": target_name})
        arrays.append(X_part)
        targets.append(y_part)
        offset += len(y_part)
    X, y = np.vstack(arrays), np.concatenate(targets)
    return X, y, expected_features, groups


def _mode2_parameter_symbols(expression: object) -> list[sp.Symbol]:
    """Return fitted constants in the same name space used by the LM code."""
    parsed = sp.sympify(expression)
    return sorted(
        [symbol for symbol in parsed.free_symbols
         if str(symbol).startswith(("a", "c", "p"))], key=str,
    )


def _mode5_periodic_zero_symbols(
    expression: object, parameter_symbols: Sequence[sp.Symbol],
) -> list[sp.Symbol]:
    """Select parameters used as standalone additive terms, including in functions.

    A symbol is excluded if any occurrence is used outside a standalone
    additive term, since replacing that symbol would also change the other
    structure. Numeric scalar factors (for example the minus sign in
    ``x - c``) do not make a product. Function arguments are inspected
    independently, so ``a*log(x + c)`` still exposes ``c`` as an additive
    parameter inside ``log``.
    """
    expression = sp.sympify(expression)
    additive_symbols: set[sp.Symbol] = set()
    other_symbols: set[sp.Symbol] = set()

    def collect_additive(node: sp.Expr, role: str = "other") -> None:
        if isinstance(node, sp.Symbol):
            if role == "addend":
                additive_symbols.add(node)
            else:
                other_symbols.add(node)
            return
        if isinstance(node, sp.Add):
            for term in node.args:
                collect_additive(term, "product" if role == "product" else "addend")
            return
        if isinstance(node, sp.Mul):
            factors = [factor for factor in node.args if not factor.is_number]
            if len(factors) <= 1:
                for factor in factors:
                    collect_additive(factor, role)
                return
            for factor in factors:
                collect_additive(
                    factor, "other" if factor.is_Function else "product"
                )
            return
        if node.is_Function:
            for argument in node.args:
                collect_additive(argument, "other")
            return
        if isinstance(node, sp.Pow):
            other_symbols.update(node.free_symbols)
            return
        for child in node.args:
            collect_additive(child, "other")

    collect_additive(expression, "other")
    return [symbol for symbol in parameter_symbols
            if symbol in additive_symbols and symbol not in other_symbols]


def _mode2_fit_row(record: dict, best: dict, ranking_row: dict | None = None) -> dict:
    """Attach structure metadata to a normal joint LM result."""
    row = {**best, "formula_id": record["id"],
           "readable_expression": record["readable_expression"],
           "sympy_expression": record.get(
               "sympy_expression", sp.srepr(sp.sympify(record["readable_expression"]))),
           "structure_type": record.get("structure_type", "mode2_exact_fit")}
    for key in ("tree_readable_expression", "source_formula_id",
                "zeroed_constants", "constant_assignments"):
        if key in record:
            row[key] = record[key]
    if ranking_row:
        for key in ("predicted_rank", "ranking_score", "formula_family", "is_training_formula"):
            if key in ranking_row:
                row[key] = ranking_row[key]
    return row


def _mode2_hypervolume(
    rows: Sequence[dict], objective_fields: Sequence[str],
    maximize_fields: set[str], objective_bounds: dict[str, tuple[float, float]] | None = None,
) -> float:
    """Return exact anchored-box HV for the supplied 2-D/3-D objectives.

    The caller provides one shared raw-value reference point per objective in
    ``_hv_ref_<field>``.  All objectives are transformed to maximization
    coordinates internally, so minimizing objectives such as complexity and
    realdata_mat's piecewise jump are handled without dropping dimensions.
    The grid-cell union is exact for the small Pareto fronts produced by
    Mode 2 and avoids an external hypervolume dependency.
    """
    if len(objective_fields) not in (2, 3):
        raise ValueError("Mode 2 HV currently supports 2 or 3 objectives")
    if objective_bounds is not None:
        # A constant objective has zero min-max span and carries no ordering
        # information.  Exclude that degenerate dimension instead of adding
        # an artificial epsilon/margin to the normalization range.
        objective_fields = tuple(
            field for field in objective_fields
            if objective_bounds[field][1] > objective_bounds[field][0]
        )
        maximize_fields = set(maximize_fields) & set(objective_fields)
        if not objective_fields:
            return 0.0
    points = []
    for row in rows:
        try:
            point = []
            for field in objective_fields:
                value = float(row[field])
                reference = float(row[f"_hv_ref_{field}"])
                if objective_bounds is not None:
                    lower, upper = objective_bounds[field]
                    span = upper - lower
                    if not math.isfinite(span) or span <= 0:
                        raise ValueError
                    if field in maximize_fields:
                        coordinate = (value - lower) / span
                        reference_coordinate = (reference - lower) / span
                    else:
                        coordinate = (upper - value) / span
                        reference_coordinate = (upper - reference) / span
                else:
                    coordinate = value if field in maximize_fields else -value
                    reference_coordinate = reference if field in maximize_fields else -reference
                if not math.isfinite(coordinate) or not math.isfinite(reference_coordinate):
                    raise ValueError
                if coordinate > reference_coordinate:
                    point.append(coordinate)
                else:
                    raise ValueError
            points.append(tuple(point))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    if not points:
        return 0.0

    references = []
    for field in objective_fields:
        reference = float(rows[0][f"_hv_ref_{field}"])
        if objective_bounds is not None:
            lower, upper = objective_bounds[field]
            span = upper - lower
            references.append(
                ((reference - lower) / span) if field in maximize_fields
                else ((upper - reference) / span)
            )
        else:
            references.append(reference if field in maximize_fields else -reference)
    # For three objectives, the old exact implementation enumerated every
    # Cartesian grid cell.  Its cost is O(n^3) for an n-point raw front, so a
    # large raw Pareto front could make the caller appear to hang while
    # printing the per-iteration HV.  Use an equivalent x-sweep instead:
    # within each x slab, calculate the exact union area of the anchored
    # y-z rectangles in O(n log n), giving O(n^2 log n) overall.
    if len(objective_fields) == 3:
        x_reference, y_reference, z_reference = references
        x_coordinates = sorted({x_reference, *[point[0] for point in points]})
        volume = 0.0
        for x_lower, x_upper in zip(x_coordinates, x_coordinates[1:]):
            slab_width = x_upper - x_lower
            if slab_width <= 0:
                continue
            active = [point for point in points if point[0] >= x_upper]
            if not active:
                continue
            active.sort(key=lambda point: point[1])
            suffix_max_z = [0.0] * len(active)
            running_max_z = z_reference
            for index in range(len(active) - 1, -1, -1):
                running_max_z = max(running_max_z, active[index][2])
                suffix_max_z[index] = running_max_z
            area = 0.0
            previous_y = y_reference
            for index, point in enumerate(active):
                y_upper = point[1]
                if y_upper > previous_y:
                    area += (y_upper - previous_y) * suffix_max_z[index]
                    previous_y = y_upper
            volume += slab_width * area
        return float(volume)

    coordinates = [sorted({references[index], *[point[index] for point in points]})
                   for index in range(len(objective_fields))]
    volume = 0.0
    ranges = [range(len(axis) - 1) for axis in coordinates]
    for cell in itertools.product(*ranges):
        lower = [coordinates[index][cell[index]]
                 for index in range(len(objective_fields))]
        upper = [coordinates[index][cell[index] + 1]
                 for index in range(len(objective_fields))]
        if any(all(point[index] >= upper[index]
                   for index in range(len(objective_fields)))
               for point in points):
            cell_volume = 1.0
            for index in range(len(objective_fields)):
                cell_volume *= upper[index] - lower[index]
            volume += cell_volume
    return float(volume)


def run_mode2_surrogate_assisted_search(
    payload: dict, config: dict, ranking_config: dict, X: np.ndarray, y: np.ndarray,
    dataset_groups: Sequence[dict], seed: int, output_dir: Path,
    is_realdata_mat: bool, piecewise_enabled: bool,
    segmentation_config: dict, n_jobs: int,
    ranking_strategy: str = "tree",
    forced_iterations: int | None = None,
    pipeline_mode: int = 3,
    selection_validation_data=None,
    report_test_data=None,
    shared_initial_constant_state: dict | None = None,
) -> dict:
    """Run the interval-layered surrogate-assisted Mode 3 search."""
    mode_cfg = dict(config.get("mode2", {}))
    gp_cfg = config.get("gp", {})
    # Admission windows use the caller's previous-round front. This threshold
    # is a main-search setting and is independent of the PySR baseline.
    r2_window_gap = max(0.0, float(mode_cfg.get(
        "validation_r2_window", DEFAULT_MODE2_R2_WINDOW
    )))
    mode4_crowding = int(pipeline_mode) == 4
    mode5_retrain_interval = max(0, int(config.get("mode5_retrain_interval", 0)))
    mode5_retrain_enabled = mode5_retrain_interval > 0
    mode5_initial_selection_enabled = (
        int(pipeline_mode) == 5
        or config.get("periodic_constant_significance_experiment_mode") == 5
    )
    # Only Mode 5 realdata_mat uses the fitted prediction jump as an
    # additional Pareto objective.  CSV mode and Mode 4 retain their
    # existing objective sets.
    mode5_realdata_piecewise_objective = (
        is_realdata_mat and mode5_initial_selection_enabled
    )
    mode5_fit_batch_rounds = max(1, int(config.get("mode5_fit_batch_rounds", 1)))
    periodic_constant_significance_interval = max(
        0, int(config.get("periodic_constant_significance_interval", 0))
    )
    periodic_constant_significance_enabled = (
        bool(config.get("periodic_constant_significance_enabled", False))
        and config.get("periodic_constant_significance_experiment_mode") in (5, 6)
        and periodic_constant_significance_interval > 0
    )
    disable_early_stopping = bool(
        config.get("mode34_disable_early_stopping", False)
    )
    for key, value in gp_cfg.items():
        if key.startswith("mode2_"):
            mode_cfg.setdefault(key[6:], value)
    top_k = max(1, int(mode_cfg.get("initial_top_k", ranking_config.get("refit_top_k", 15))))
    batch_size = max(1, int(mode_cfg.get("batch_size", 5)))
    max_iterations = 40
    max_exact_fits = mode_cfg.get("max_exact_fits")
    if max_exact_fits is not None:
        max_exact_fits = max(1, int(max_exact_fits))
    r2_drop_threshold = float(mode_cfg.get(
        "r2_drop_threshold",
        config.get("postprocessing", {}).get("merge_log_max_r2_drop", 0.01),
    ))
    if r2_drop_threshold < 0:
        raise ValueError("mode2.r2_drop_threshold cannot be negative")
    interval_layered_mode = int(pipeline_mode) == 3
    mode3_required_variable_count = (
        X.shape[1] if interval_layered_mode else 0
    )

    all_records = {row["id"]: row for row in [
        *payload.get("single_candidates", []), *payload.get("binary_combinations", [])]}
    expected_dataset_names = {str(group["name"]) for group in dataset_groups}
    # realdata_mat piecewise fitting replaces the single source dataset with
    # the two fitted segments.  Treat those segment rows as the complete fit
    # result only for this data mode; all other modes keep the original
    # dataset-name contract.
    expected_fit_dataset_names = (
        {"segment_1", "segment_2"}
        if is_realdata_mat and piecewise_enabled
        else expected_dataset_names
    )

    def has_complete_dataset_fits(row: dict | None) -> bool:
        """Require one finite R² result for every configured dataset."""
        if not row:
            return False
        dataset_rows = row.get("dataset_fit_results") or row.get("dataset_metrics")
        if (not isinstance(dataset_rows, list)
                or len(dataset_rows) != len(expected_fit_dataset_names)):
            return False
        seen_names = set()
        for dataset_row in dataset_rows:
            if not isinstance(dataset_row, dict):
                return False
            name = dataset_row.get("dataset_name", dataset_row.get("name"))
            if name is None or str(name) in seen_names:
                return False
            value = dataset_row.get("r2")
            try:
                value = float(value)
            except (TypeError, ValueError):
                return False
            if not math.isfinite(value):
                return False
            seen_names.add(str(name))
        return seen_names == expected_fit_dataset_names

    piecewise_payload = [
        row for key in ("single_best", "combination_best")
        for row in payload.get(key, [])
    ]
    if (is_realdata_mat and piecewise_enabled and piecewise_payload
            and all(bool(row.get("piecewise")) for row in piecewise_payload)):
        # Piecewise MAT fitting already returns one complete formula-level row
        # with segment_1/segment_2 fits.  Do not pass it through the legacy
        # formula×dataset regrouping below: the top-level row intentionally
        # has no dataset_name, and regrouping would discard every formula.
        for key in ("single_best", "combination_best"):
            payload[key] = [
                row for row in payload.get(key, [])
                if row.get("formula_id") in all_records
                and has_complete_dataset_fits(row)
            ]
    else:
        # Initial Mode 2 fitting is persisted one row per formula and dataset
        # by the legacy fitting loop.  Rebuild one cross-dataset row per
        # formula before ranking; unchanged for all non-piecewise modes.
        for key in ("single_best", "combination_best"):
            grouped = {}
            for row in payload.get(key, []):
                formula_id = row.get("formula_id")
                dataset_name = row.get("dataset_name")
                if formula_id is None or dataset_name is None:
                    continue
                grouped.setdefault(str(formula_id), {})[str(dataset_name)] = row
            combined_rows = []
            for formula_id, by_dataset in grouped.items():
                if set(by_dataset) != expected_dataset_names:
                    continue
                source_rows = [by_dataset[name] for name in sorted(expected_dataset_names)]
                dataset_fit_results = []
                dataset_metrics = []
                parameters_by_dataset = {}
                complete = True
                for source in source_rows:
                    fit_rows = source.get("dataset_fit_results") or [source]
                    fit_row = dict(fit_rows[0])
                    fit_row["dataset_name"] = str(source["dataset_name"])
                    fit_row["dataset_path"] = source.get("dataset_path")
                    dataset_fit_results.append(fit_row)
                    metric = dict((source.get("dataset_metrics") or [{}])[0])
                    metric.update(name=str(source["dataset_name"]),
                                  path=source.get("dataset_path"), r2=fit_row.get("r2"))
                    dataset_metrics.append(metric)
                    parameters_by_dataset[str(source["dataset_name"])] = source.get("parameters")
                    try:
                        complete = complete and math.isfinite(float(fit_row.get("r2")))
                    except (TypeError, ValueError):
                        complete = False
                if not complete:
                    continue
                combined = dict(source_rows[0])
                combined.update(
                    dataset_name="__all_datasets__",
                    dataset_path=None,
                    dataset_fit_results=dataset_fit_results,
                    dataset_metrics=dataset_metrics,
                    parameters=parameters_by_dataset,
                    parameters_by_dataset=parameters_by_dataset,
                    expected_dataset_count=len(expected_dataset_names),
                    mean_r2=float(np.mean([float(row["r2"]) for row in dataset_fit_results])),
                    all_datasets_valid=True,
                    constants_shared_across_datasets=False,
                )
                combined_rows.append(combined)
            payload[key] = combined_rows
    _annotate_mode2_training_rows([
        *payload.get("single_best", []), *payload.get("combination_best", [])
    ])
    fit_cache = {}
    fit_cache_by_expression = {}
    for row in [*payload.get("single_best", []), *payload.get("combination_best", [])]:
        _annotate_mode2_mean_r2(row)
        if row.get("formula_id") in all_records and _mode2_mean_r2(row) is not None:
            fit_cache[row["formula_id"]] = row
            fit_cache_by_expression[sp.srepr(sp.sympify(
                row["readable_expression"]))] = row
    evaluated_ids = set(fit_cache)
    initial_training_formula_count = len(evaluated_ids)
    exact_cache = dict(payload.get("mode2_exact_fit_cache", {}))
    for cached in exact_cache.values():
        _annotate_mode2_mean_r2(cached)
        expression = cached.get("readable_expression")
        if expression and has_complete_dataset_fits(cached):
            fit_cache_by_expression[sp.srepr(sp.sympify(expression))] = cached
    exact_fit_count = 0
    reserved_fit_count = 0
    cache_lock = Lock()
    single_ids = {row["id"] for row in payload.get("single_candidates", [])}
    significance_checks = []
    iterative_fitted_rows = []
    # Later rounds choose several formulas at once.  The worker count is set
    # per batch below so a batch with one formula can still use the
    # dataset-level pool without creating nested process pools.
    later_round_workers = 1
    max_later_round_workers = 1

    # Mode 2 can parallelize at both levels: formulas in a batch and zeroed
    # subsets within one formula.  Keep the total amount of LM concurrency
    # bounded so the two pools do not multiply to an unbounded thread count.
    parallel_jobs = int(n_jobs)
    if parallel_jobs == 0 or parallel_jobs < -1:
        raise ValueError("Mode 2 n_jobs must be 1, a positive integer, or -1")
    available_cpus = max(1, (os.cpu_count() or 2) - 1)
    requested_workers = available_cpus if parallel_jobs < 0 else parallel_jobs
    requested_workers = max(1, requested_workers)

    def mode2_worker_counts(task_count: int) -> tuple[int, int]:
        """Return (formula_workers, subset_workers) for nested parallelism."""
        if task_count <= 1 or requested_workers == 1:
            return 1, 1
        # Allocate the requested concurrency across both levels.  The
        # product is capped by available CPUs, while retaining parallelism at
        # both levels whenever there is enough work and hardware.
        formula_workers = min(task_count, requested_workers,
                              max(1, int(math.sqrt(available_cpus))))
        subset_workers = max(1, min(
            requested_workers // formula_workers,
            max(1, available_cpus // formula_workers),
        ))
        return formula_workers, subset_workers

    def mode2_r2_window(rows, reference_rows=None):
        """Keep rows within the R² window around the previous-round front."""
        valid = [row for row in rows if _mode2_mean_r2(row) is not None]
        if not valid:
            return []
        primary_key = "pooled_r2" if is_realdata_mat else "mean_r2"
        def primary_value(row):
            value = row.get(primary_key)
            if value is None:
                value = _mode2_mean_r2(row)
            return float(value)
        reference = reference_rows if reference_rows is not None else valid
        reference = [row for row in reference if _mode2_mean_r2(row) is not None]
        if not reference:
            reference = valid
        highest = max(primary_value(row) for row in reference)
        cutoff = highest - r2_window_gap
        return [row for row in valid if primary_value(row) >= cutoff]

    def annotate_mode2_front(rows):
        """Populate the legacy GP display metrics for Mode 2 rows."""
        for row in rows:
            dataset_rows = row.get("dataset_fit_results", []) or [row]
            values = []
            for dataset_row in dataset_rows:
                value = dataset_row.get("r2") if isinstance(dataset_row, dict) else None
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = None
                if value is not None:
                    if math.isfinite(value):
                        values.append(value)
            if values:
                row["mean_r2"] = float(np.mean(values))
                row["r2_std"] = float(np.std(values))
                row["min_r2"] = float(np.min(values))
                pooled = row.get("pooled_r2")
                row["pooled_r2"] = float(max(values) if pooled is None else pooled)
            row.setdefault("crowding_rank", 0)
            row.setdefault("crowding_distance", 0.0)
            row.setdefault("crowding_distance_label", "N/A")
            row.setdefault("complexity", formula_complexity(
                sp.sympify(row["readable_expression"])
            ))
        return rows

    validation_selection_enabled = (
        selection_validation_data is not None and not mode4_crowding
    )
    report_test_X = report_test_y = report_test_groups = None
    if report_test_data is not None:
        report_test_X, report_test_y, report_test_groups = report_test_data
    payload.setdefault("metadata", {})[
        "pareto_selection_basis"
    ] = ("validation_r2_window_incremental_nondominated_admission"
         if validation_selection_enabled else "training_objectives")
    if validation_selection_enabled:
        selection_X, selection_y, _selection_features, selection_groups = (
            selection_validation_data
        )

    def select_validation_pareto_front(rows):
        """Attach frozen validation metrics for later R²-window admission.

        Admission and dominance checks are controlled separately.  Once a
        formula has been admitted, it remains in the cumulative result set.
        """
        if not validation_selection_enabled:
            return rows
        evaluated = []
        for source in rows:
            metrics = _evaluate_validation_row(
                source, selection_X, selection_y, selection_groups,
                dataset_groups, config["fitting"],
            )
            if metrics is None:
                continue
            row = copy.deepcopy(source)
            for key, value in metrics.items():
                row[f"validation_{key}"] = value
            _attach_training_validation_mean_r2_gap(row)
            row["validation_complexity"] = formula_complexity(
                sp.sympify(row["readable_expression"])
            )
            evaluated.append(row)
        if not evaluated:
            return []
        # Keep the original training metrics/parameters and attach validation
        # metrics for traceability.
        return evaluated

    def validation_r2_window(rows, reference_rows=None):
        """Admit candidates using the previous-round validation R² window."""
        valid = [row for row in rows
                 if row.get("validation_mean_r2") is not None
                 or row.get("validation_pooled_r2") is not None]
        if not valid:
            return []
        primary_key = (
            "validation_pooled_r2" if is_realdata_mat
            else "validation_mean_r2"
        )

        def primary_value(row):
            value = row.get(primary_key)
            if value is None:
                value = row.get("validation_mean_r2")
            return float(value)

        reference = reference_rows if reference_rows is not None else valid
        reference = [row for row in reference if (
            row.get("validation_mean_r2") is not None
            or row.get("validation_pooled_r2") is not None
        )]
        if not reference:
            reference = valid
        highest = max(primary_value(row) for row in reference)
        cutoff = highest - r2_window_gap
        return [row for row in valid if primary_value(row) >= cutoff]

    def admit_incremental_validation_candidates(existing_rows, candidate_rows):
        """Return the non-dominated validation archive after this batch.

        The archive is recalculated from both existing and new rows on every
        call.  Consequently, a previously retained row is removed when a
        later candidate dominates it.
        """
        existing = list(existing_rows)
        candidates = list(candidate_rows)
        primary = (
            "validation_pooled_r2" if is_realdata_mat
            else "validation_mean_r2"
        )

        def difference_value(row):
            if not mode5_realdata_piecewise_objective:
                return 0.0
            try:
                value = float(row.get("segment_prediction_difference"))
            except (TypeError, ValueError):
                return math.inf
            return value if math.isfinite(value) else math.inf

        def dominates(left, right):
            no_worse = (
                float(left[primary]) >= float(right[primary])
                and float(left["validation_r2_std"])
                <= float(right["validation_r2_std"])
                and float(left["validation_complexity"])
                <= float(right["validation_complexity"])
                and difference_value(left) <= difference_value(right)
            )
            strictly_better = (
                float(left[primary]) > float(right[primary])
                or float(left["validation_r2_std"])
                < float(right["validation_r2_std"])
                or float(left["validation_complexity"])
                < float(right["validation_complexity"])
                or difference_value(left) < difference_value(right)
            )
            return no_worse and strictly_better

        combined = [*existing, *candidates]
        return [
            item for item in combined
            if not any(
                other is not item and dominates(other, item)
                for other in combined
            )
        ]

    def print_new_mode2_pareto_entries(rows, previous_keys, stage):
        """Print and count formulas newly entering the retained result set."""
        def display_value(value):
            try:
                return f"{float(value):.6g}"
            except (TypeError, ValueError):
                return str(value)

        current_keys = {
            sp.srepr(sp.sympify(row["readable_expression"])) for row in rows
        }
        entered = current_keys - set(previous_keys)
        if not entered:
            return current_keys, 0
        for row in rows:
            key = sp.srepr(sp.sympify(row["readable_expression"]))
            if key not in entered:
                continue
            if validation_selection_enabled:
                test_metrics = (
                    _evaluate_validation_row(
                        row, report_test_X, report_test_y, report_test_groups,
                        dataset_groups, config["fitting"],
                    )
                    if report_test_X is not None and report_test_y is not None
                    and report_test_groups is not None else None
                )
                print(
                    f"{mode_label} R²-window new solution entered ({stage}): "
                    f"formula={row.get('readable_expression')}; "
                    f"complexity={row.get('complexity')}; "
                    "training set["
                    f"mean_r2={display_value(row.get('mean_r2'))}; "
                    f"r2_std={display_value(row.get('r2_std'))}; "
                    f"min_r2={display_value(row.get('min_r2'))}; "
                    f"pooled_r2={display_value(row.get('pooled_r2'))}]; "
                    "Validation set["
                    f"mean_r2={display_value(row.get('validation_mean_r2'))}; "
                    f"r2_std={display_value(row.get('validation_r2_std'))}; "
                    f"train/validation mean_r2 gap={display_value(row.get('validation_training_mean_r2_gap'))}; "
                    f"min_r2={display_value(row.get('validation_min_r2'))}; "
                    f"pooled_r2={display_value(row.get('validation_pooled_r2'))}]; "
                    "test set["
                    f"mean_r2={display_value(test_metrics.get('mean_r2') if test_metrics else None)}; "
                    f"r2_std={display_value(test_metrics.get('r2_std') if test_metrics else None)}; "
                    f"min_r2={display_value(test_metrics.get('min_r2') if test_metrics else None)}; "
                    f"pooled_r2={display_value(test_metrics.get('pooled_r2') if test_metrics else None)}]",
                    flush=True,
                )
            else:
                test_metrics = (
                    _evaluate_validation_row(
                        row, report_test_X, report_test_y, report_test_groups,
                        dataset_groups, config["fitting"],
                    )
                    if report_test_X is not None and report_test_y is not None
                    and report_test_groups is not None else None
                )
                print(
                    f"{mode_label} Pareto new solution entered ({stage}): "
                    f"formula={row.get('readable_expression')}; "
                    f"complexity={row.get('complexity')}; "
                    f"mean_r2={display_value(row.get('mean_r2'))}; "
                    f"r2_std={display_value(row.get('r2_std'))}; "
                    f"min_r2={display_value(row.get('min_r2'))}; "
                    f"pooled_r2={display_value(row.get('pooled_r2'))}]; "
                    "test set["
                    f"mean_r2={display_value(test_metrics.get('mean_r2') if test_metrics else None)}; "
                    f"r2_std={display_value(test_metrics.get('r2_std') if test_metrics else None)}; "
                    f"min_r2={display_value(test_metrics.get('min_r2') if test_metrics else None)}; "
                    f"pooled_r2={display_value(test_metrics.get('pooled_r2') if test_metrics else None)}]",
                    flush=True,
                )
            parameters_by_dataset = row.get("parameters_by_dataset")
            if isinstance(parameters_by_dataset, dict) and parameters_by_dataset:
                names_by_dataset = {
                    item.get("dataset_name"): item.get("parameter_names", [])
                    for item in row.get("dataset_fit_results", [])
                    if item.get("dataset_name") is not None
                }
                for dataset_name, parameters in parameters_by_dataset.items():
                    print(
                        f"  Parameters [{dataset_name}]: "
                        f"names={names_by_dataset.get(dataset_name, row.get('parameter_names', []))}, "
                        f"values={parameters}",
                        flush=True,
                    )
            else:
                print(
                    f"  Parameters:  names={row.get('parameter_names', [])}, "
                    f"values={row.get('parameters')}",
                    flush=True,
                )
        return current_keys, len(entered)

    if mode4_crowding:
        # Mode 4 first peels Pareto fronts, then ranks within each front by
        # normalized NSGA-II crowding distance.  Pareto rank is encoded as the
        # dominant part of the scalar tree-training label.
        crowding_pool = list(fit_cache.values())
        # Combined multi-dataset rows may not yet carry the legacy aggregate
        # objective fields; derive them from their dataset fit rows first.
        annotate_mode2_front(crowding_pool)
        labels, pareto_ranks = _mode4_rank_crowding_labels(
            crowding_pool, is_realdata_mat, use_piecewise_difference=False
        )
        for row in crowding_pool:
            formula_id = str(row.get("formula_id"))
            row["pareto_rank"] = pareto_ranks.get(formula_id)
            row["crowding_distance"] = labels.get(formula_id, 0.0)
        payload.setdefault("metadata", {})["mode4_pareto_layer_count"] = len(set(pareto_ranks.values()))
        payload["metadata"]["mode4_crowding_normalization"] = "per_layer_finite_max_plus_one"
    if ranking_strategy not in {"tree", "random"}:
        raise ValueError("Mode 2 ranking_strategy must be tree or random")

    rank_config = dict(ranking_config)
    # realdata_mat uses pooled R² as the tree target because it is the
    # primary aggregate fit-quality metric for the shared piecewise model.
    # Other data modes retain the historical mean-R² target.
    tree_training_label = (
        "crowding_distance" if mode4_crowding
        else ("pooled_r2" if is_realdata_mat else "mean_r2")
    )
    rank_config.update(enabled=True, ablation_enabled=False,
                       training_label_field=tree_training_label)
    rank_config["_log_context"] = "Mode 4 crowding initial training" if mode4_crowding else "Mode 2 initial training"
    if ranking_strategy == "tree":
        ranking, ranking_metadata = train_and_rank(payload, rank_config, seed, output_dir)
        ranking_metadata["training_label_field"] = tree_training_label
    else:
        # Random permutation is the ablation: it consumes the same candidate
        # pool and the same exact-fit machinery, but never trains or calls a
        # tree model.  A private RNG keeps the control run reproducible.
        random_order = list(all_records.values())
        random.Random(seed).shuffle(random_order)
        ranking = []
        for predicted_rank, record in enumerate(random_order, start=1):
            ranking.append({
                **record,
                # ``train_and_rank`` exposes this canonical key; keep the
                # same schema for the random-permutation ablation so all
                # downstream Mode 2 selection code remains unchanged.
                "formula_id": record["id"],
                "predicted_rank": predicted_rank,
                "ranking_score": float(len(random_order) - predicted_rank),
                "formula_family": "single" if record["id"] in single_ids else "combination",
                "is_training_formula": record["id"] in fit_cache,
            })
        ranking_metadata = {
            "enabled": False,
            "ablation_enabled": True,
            "model_type": "random_permutation",
            "training_samples": 0,
            "ranked_formulas": len(ranking),
            "training_label_field": "none",
            "selection_basis": "seeded_random_permutation",
            "random_seed": int(seed),
        }
    payload["structure_r2_ranking"] = ranking
    payload["ranking_metadata"] = ranking_metadata

    def fit_record(record, initial_values=None, fit_seed=0,
                   dataset_parallel=False):
        if piecewise_enabled:
            rng = np.random.default_rng(int(fit_seed))
            attempts, best = fit_piecewise_formula(
                record, X, y, config["fitting"], rng, segmentation_config)
        elif dataset_parallel:
            # Match the initial LM granularity: one independent process task
            # per formula/dataset pair, followed by deterministic aggregation.
            tasks = []
            perturbation = float(config["fitting"].get(
                "zero_constant_initial_perturbation", 0.05))
            lm_initializations = int(config["fitting"].get(
                "zero_constant_lm_initializations",
                config["fitting"].get("lm_initializations", 20)))
            for group in dataset_groups:
                dataset_name = str(group["name"])
                tasks.append((
                    record, X, y, config["fitting"], group,
                    (initial_values or {}).get(dataset_name), perturbation,
                    lm_initializations, fit_seed,
                ))
            results = _parallel_mode2_dataset_results(
                tasks, max_workers=24
            )
            attempts, best = _combine_mode2_dataset_fits(
                record, X, y, config["fitting"], dataset_groups, results
            )
        else:
            rng = np.random.default_rng(int(fit_seed))
            attempts, best = fit_formula_per_dataset(
                record, X, y, config["fitting"], rng, dataset_groups,
                initial_values_by_dataset=initial_values,
                initial_perturbation=float(config["fitting"].get(
                    "zero_constant_initial_perturbation", 0.05)),
                lm_initializations=int(config["fitting"].get(
                    "zero_constant_lm_initializations",
                    config["fitting"].get("lm_initializations", 20))), fit_rng=rng)
        row = _mode2_fit_row(record, best)
        # Mode 2 owns the realdata_mat Pareto loop, so it must populate the
        # piecewise objective itself.  The legacy pipeline computes this
        # later, but mode2 deliberately bypasses that legacy post-processing.
        if is_realdata_mat:
            _attach_piecewise_continuity_objective(
                row, record["readable_expression"], X
            )
        return attempts, row

    def cache_fit(record, initial_values=None, fit_seed=0,
                  dataset_parallel=False):
        nonlocal exact_fit_count, reserved_fit_count
        formula_id = record["id"]
        expression_key = sp.srepr(sp.sympify(record["readable_expression"]))
        with cache_lock:
            if formula_id in fit_cache:
                return fit_cache[formula_id]
            if expression_key in fit_cache_by_expression:
                fit_cache[formula_id] = fit_cache_by_expression[expression_key]
                return fit_cache[formula_id]
            if formula_id in exact_cache and has_complete_dataset_fits(exact_cache[formula_id]):
                fit_cache[formula_id] = exact_cache[formula_id]
                return fit_cache[formula_id]
            if (max_exact_fits is not None
                    and exact_fit_count + reserved_fit_count >= max_exact_fits):
                return None
            reserved_fit_count += 1
        _attempts, row = fit_record(
            record, initial_values, fit_seed,
            dataset_parallel=dataset_parallel,
        )
        if not has_complete_dataset_fits(row):
            with cache_lock:
                reserved_fit_count -= 1
            return None
        _annotate_mode2_mean_r2(row)
        with cache_lock:
            reserved_fit_count -= 1
            exact_fit_count += 1
            exact_cache[formula_id] = row
            fit_cache[formula_id] = row
            fit_cache_by_expression[expression_key] = row
        if formula_id in all_records:
            target = (payload["single_best"] if formula_id in single_ids
                      else payload["combination_best"])
            if not any(item.get("formula_id") == formula_id for item in target):
                target.append(row)
        return row

    def fit_selected_formulas_initial_style(selected_tasks):
        """Fit one later-round batch with the initial formula×dataset pool."""
        nonlocal exact_fit_count, reserved_fit_count
        selected_records = []
        rows = {index: None for index, _rank_row in enumerate(selected_tasks)}
        with cache_lock:
            for index, rank_row in enumerate(selected_tasks):
                record = all_records[rank_row["formula_id"]]
                formula_id = record["id"]
                expression_key = sp.srepr(sp.sympify(record["readable_expression"]))
                cached = fit_cache.get(formula_id)
                if cached is None:
                    cached = fit_cache_by_expression.get(expression_key)
                if cached is None:
                    cached = exact_cache.get(formula_id)
                if cached is not None and has_complete_dataset_fits(cached):
                    fit_cache[formula_id] = cached
                    rows[index] = cached
                    continue
                if (max_exact_fits is not None
                        and exact_fit_count + reserved_fit_count >= max_exact_fits):
                    break
                reserved_fit_count += 1
                selected_records.append((index, rank_row, record))

        if not selected_records:
            return rows

        # Match the initial LM stage: ordinary Mode 3/4 fitting is expanded
        # into independent formula×dataset process-pool jobs.  Piecewise
        # fitting remains a formula-level job because its segmentation is
        # global rather than dataset-local.
        fitted_by_index = {}
        if piecewise_enabled:
            tasks = [(
                record, X, y, config["fitting"], segmentation_config,
                piecewise_enabled, str(dataset_groups[0]["name"]),
                _stable_lm_seed(seed, "mode2", record["id"]),
            ) for _index, _rank_row, record in selected_records]
            results = _parallel_lm_results(
                tasks, max_workers=None if requested_workers < 0 else requested_workers
            )
            for (index, _rank_row, record), (attempts, best) in zip(
                    selected_records, results):
                fitted_by_index[index] = (attempts, _mode2_fit_row(record, best))
        else:
            lm_initializations = int(config["fitting"].get(
                "zero_constant_lm_initializations",
                config["fitting"].get("lm_initializations", 20),
            ))
            tasks = []
            for _index, _rank_row, record in selected_records:
                for group in dataset_groups:
                    tasks.append((
                        record, X, y, config["fitting"], group, None, 0.05,
                        lm_initializations,
                        _stable_lm_seed(seed, "mode2", record["id"]),
                    ))
            results = _parallel_mode2_dataset_results(
                tasks, max_workers=(
                    DEFAULT_LM_MAX_WORKERS if requested_workers < 0
                    else requested_workers
                ),
            )
            dataset_count = len(dataset_groups)
            for offset, (index, _rank_row, record) in enumerate(selected_records):
                dataset_results = results[
                    offset * dataset_count:(offset + 1) * dataset_count
                ]
                attempts, best = _combine_mode2_dataset_fits(
                    record, X, y, config["fitting"], dataset_groups,
                    dataset_results,
                )
                fitted_by_index[index] = (attempts, _mode2_fit_row(record, best))

        for index, _rank_row, record in selected_records:
            attempts, row = fitted_by_index[index]
            if not has_complete_dataset_fits(row):
                with cache_lock:
                    reserved_fit_count -= 1
                continue
            _annotate_mode2_mean_r2(row)
            with cache_lock:
                reserved_fit_count -= 1
                exact_fit_count += 1
                formula_id = record["id"]
                expression_key = sp.srepr(sp.sympify(record["readable_expression"]))
                exact_cache[formula_id] = row
                fit_cache[formula_id] = row
                fit_cache_by_expression[expression_key] = row
            if formula_id in all_records:
                target = (payload["single_best"] if formula_id in single_ids
                          else payload["combination_best"])
                if not any(item.get("formula_id") == formula_id for item in target):
                    target.append(row)
            rows[index] = row
        return rows

    def mode5_zero_constant_candidates(
            source: dict, baseline: dict, parallel_workers: int | None = None,
            dataset_parallel: bool = False, reference_rows=None,
            test_phase: str = "initial"
    ) -> tuple[list[dict], list[dict]]:
        """Run the configured Mode 5 constant tests for one search phase."""
        nonlocal exact_fit_count, reserved_fit_count
        local_checks = []
        source_expression = sp.sympify(source["readable_expression"])
        source_symbols = _mode2_parameter_symbols(source_expression)
        if not source_symbols:
            return [baseline], local_checks
        constant_test_timeout = config.get(
            "mode5_constant_test_timeout_seconds", 210.0
        )
        try:
            constant_test_timeout = float(constant_test_timeout)
        except (TypeError, ValueError):
            raise ValueError(
                "mode5_constant_test_timeout_seconds must be numeric"
            )
        if constant_test_timeout < 0:
            raise ValueError(
                "mode5_constant_test_timeout_seconds cannot be negative"
            )
        constant_test_deadline = (
            time.monotonic() + constant_test_timeout
            if constant_test_timeout > 0 else None
        )
        constant_test_timed_out = False

        def initial_values_for(row: dict) -> dict:
            return {
                item.get("dataset_name", "dataset_1"): dict(zip(
                    item.get("parameter_names", []), item.get("parameters", [])))
                for item in row.get("dataset_fit_results", [])
            }

        def fit_stage(candidate_specs: dict, initial_values: dict) -> bool:
            """Complete one stage before the next stage is built."""
            nonlocal exact_fit_count, reserved_fit_count
            jobs = []
            for candidate in candidate_specs.values():
                expression_key = sp.srepr(sp.sympify(candidate["readable_expression"]))
                if (candidate["id"] not in fit_cache
                        and candidate["id"] not in exact_cache
                        and expression_key not in fit_cache_by_expression):
                    jobs.append((candidate, initial_values))
            with cache_lock:
                available = len(jobs) if max_exact_fits is None else max(
                    0, max_exact_fits - exact_fit_count - reserved_fit_count)
                jobs = jobs[:available]
                reserved_fit_count += len(jobs)
            # A formula-level deadline requires serial candidate scheduling:
            # each candidate is isolated in a child process and can therefore
            # be terminated without leaving a stuck LM thread behind.
            timed_out = False
            processed_jobs = 0
            for candidate, values in jobs:
                remaining = None
                if constant_test_deadline is not None:
                    remaining = constant_test_deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        break
                try:
                    _attempts, row = _run_mode5_callable_with_watchdog(
                        fit_record,
                        (candidate, values,
                         _stable_lm_seed(seed, "mode5", candidate["id"])),
                        {"dataset_parallel": dataset_parallel},
                        remaining,
                    )
                except TimeoutError:
                    with cache_lock:
                        reserved_fit_count -= 1
                    processed_jobs += 1
                    timed_out = True
                    break
                with cache_lock:
                    reserved_fit_count -= 1
                    if not has_complete_dataset_fits(row):
                        continue
                    exact_cache[candidate["id"]] = row
                    fit_cache[candidate["id"]] = row
                    fit_cache_by_expression[sp.srepr(sp.sympify(
                        candidate["readable_expression"]))] = row
                    exact_fit_count += 1
                processed_jobs += 1
            # Release reservations for candidates that were not started after
            # the deadline.  Completed candidates remain in the normal cache
            # and are eligible for the existing R²/Pareto selection.
            remaining_reservations = len(jobs) - processed_jobs
            if timed_out and remaining_reservations:
                with cache_lock:
                    reserved_fit_count -= remaining_reservations
            return timed_out

        def allowed(expression, candidate_id, operation, changed, zeroed,
                    before_r2) -> bool:
            if (config.get("data", {}).get("mode", "csv") == "csv"
                    and not _has_required_formula_variables(
                        expression, X.shape[1]
                    )):
                reason = "zeroing_would_use_fewer_than_required_input_variables"
            elif (bool(mode_cfg.get("require_log_abs_x1_initial", True))
                  and not _has_log_abs_x1(expression)):
                reason = "missing_required_log_abs_x1"
            elif expression.has(sp.nan, sp.zoo, sp.oo, -sp.oo, sp.I):
                reason = "invalid_symbolic_value_after_constant_assignment"
            else:
                return True
            local_checks.append({
                "formula_id": candidate_id, "source_formula_id": source["id"],
                "operation": operation, "constant_assignments": changed,
                "zeroed_constants": zeroed, "before_r2": before_r2,
                "after_r2": None, "r2_drop": None,
                "max_r2_drop": r2_drop_threshold, "accepted": False,
                "failure_reason": reason,
            })
            return False

        def make_candidate(candidate_id, expression, structure_type, changed, zeroed,
                           tree_expression=None):
            candidate = dict(
                source, id=candidate_id, formula_id=candidate_id,
                readable_expression=str(expression), sympy_expression=sp.srepr(expression),
                structure_type=structure_type, source_formula_id=source["id"],
                zeroed_constants=zeroed, constant_assignments=changed)
            if tree_expression is not None:
                candidate["tree_readable_expression"] = str(tree_expression)
            return candidate

        def candidate_row(candidate_id: str, candidate: dict):
            """Preserve this candidate's metadata when its expression was cached."""
            row = fit_cache.get(candidate_id) or fit_cache_by_expression.get(
                sp.srepr(sp.sympify(candidate["readable_expression"])))
            if row is None:
                return None
            if row.get("formula_id") == candidate_id:
                return row
            row = dict(row, formula_id=candidate_id,
                       readable_expression=candidate["readable_expression"],
                       sympy_expression=candidate["sympy_expression"],
                       structure_type=candidate["structure_type"])
            for key in ("tree_readable_expression", "source_formula_id",
                        "zeroed_constants", "constant_assignments"):
                if key in candidate:
                    row[key] = candidate[key]
            fit_cache[candidate_id] = row
            return row

        baseline_r2 = _mode2_mean_r2(baseline)
        zero_test_enabled = bool(config.get(
            f"mode5_{test_phase}_zero_test_enabled", True
        ))
        zero_symbols = source_symbols
        if test_phase == "periodic":
            zero_symbols = _mode5_periodic_zero_symbols(
                source_expression, source_symbols
            )
        zero_specs = {}
        if zero_test_enabled and zero_symbols:
            if test_phase == "periodic":
                # Periodic Mode 5 starts with each eligible constant
                # individually; accepted candidates are expanded one level
                # further below. Initial testing retains exhaustive subsets.
                assignments = (
                    tuple(0 if index == zero_index else None
                          for index in range(len(zero_symbols)))
                    for zero_index in range(len(zero_symbols))
                )
            else:
                assignments = (
                    assignment
                    for assignment in itertools.product(
                        (None, 0), repeat=len(zero_symbols)
                    )
                    if any(value == 0 for value in assignment)
                )
            for assignment in assignments:
                expression = source_expression.xreplace({
                    symbol: sp.Integer(value)
                    for symbol, value in zip(zero_symbols, assignment)
                    if value is not None
                })
                changed = {str(symbol): ("unchanged" if value is None else int(value))
                           for symbol, value in zip(zero_symbols, assignment)}
                zeroed = [name for name, value in changed.items() if value == 0]
                candidate_id = f"{source['id']}_zero_" + "_".join(
                    f"{symbol}_{'keep' if value is None else value}"
                    for symbol, value in zip(zero_symbols, assignment))
                if not allowed(expression, candidate_id, "exhaustive_zero_subset",
                               changed, zeroed, baseline_r2):
                    continue
                zero_specs[candidate_id] = make_candidate(
                    candidate_id, expression, "mode5_zero_constant_candidate", changed, zeroed)
            zero_stage_timed_out = fit_stage(zero_specs, initial_values_for(baseline))
            if zero_stage_timed_out:
                constant_test_timed_out = True

        zero_accepted = []
        for candidate_id, candidate in zero_specs.items():
            row = candidate_row(candidate_id, candidate)
            candidate_r2 = _mode2_mean_r2(row)
            drop = None if baseline_r2 is None or candidate_r2 is None else baseline_r2 - candidate_r2
            accepted = drop is not None and drop <= r2_drop_threshold
            local_checks.append({
                "formula_id": candidate_id, "source_formula_id": source["id"],
                "operation": "exhaustive_zero_subset",
                "constant_assignments": candidate["constant_assignments"],
                "zeroed_constants": candidate["zeroed_constants"],
                "before_r2": baseline_r2, "after_r2": candidate_r2,
                "r2_drop": drop, "max_r2_drop": r2_drop_threshold,
                "accepted": accepted})
            if accepted and row is not None:
                zero_accepted.append(row)

        # Periodic Mode 5 testing goes one level deeper: for every accepted
        # single-zero candidate, try zeroing each other eligible constant.
        # These pair-zero fits are always warm-started from the original
        # formula's parameters and judged against its original R².
        second_zero_specs = {}
        if test_phase == "periodic" and zero_test_enabled and not constant_test_timed_out:
            accepted_single_zero_names = {
                str(row.get("zeroed_constants", [""])[0])
                for row in zero_accepted
                if len(row.get("zeroed_constants", [])) == 1
            }
            second_zero_pairs = set()
            for first_name in accepted_single_zero_names:
                for symbol in zero_symbols:
                    second_name = str(symbol)
                    if second_name == first_name or symbol not in source_expression.free_symbols:
                        continue
                    second_zero_pairs.add(tuple(sorted((first_name, second_name))))

            symbols_by_name = {str(symbol): symbol for symbol in zero_symbols}
            for zeroed_pair in sorted(second_zero_pairs):
                zeroed = list(zeroed_pair)
                changed = {
                    str(symbol): (0 if str(symbol) in zeroed_pair else "unchanged")
                    for symbol in zero_symbols
                }
                expression = source_expression.xreplace({
                    symbols_by_name[name]: sp.Integer(0) for name in zeroed_pair
                })
                candidate_id = f"{source['id']}_zero_" + "_".join(
                    f"{name}_0" if name in zeroed_pair else f"{name}_keep"
                    for name in map(str, zero_symbols)
                )
                if not allowed(expression, candidate_id, "periodic_second_zero",
                               changed, zeroed, baseline_r2):
                    continue
                second_zero_specs[candidate_id] = make_candidate(
                    candidate_id, expression, "mode5_zero_constant_candidate",
                    changed, zeroed,
                )

            second_zero_timed_out = fit_stage(
                second_zero_specs, initial_values_for(baseline)
            )
            if second_zero_timed_out:
                constant_test_timed_out = True

        second_zero_accepted = []
        for candidate_id, candidate in second_zero_specs.items():
            row = candidate_row(candidate_id, candidate)
            candidate_r2 = _mode2_mean_r2(row)
            drop = None if baseline_r2 is None or candidate_r2 is None else baseline_r2 - candidate_r2
            accepted = drop is not None and drop <= r2_drop_threshold
            local_checks.append({
                "formula_id": candidate_id, "source_formula_id": source["id"],
                "operation": "periodic_second_zero",
                "constant_assignments": candidate["constant_assignments"],
                "zeroed_constants": candidate["zeroed_constants"],
                "before_r2": baseline_r2, "after_r2": candidate_r2,
                "r2_drop": drop, "max_r2_drop": r2_drop_threshold,
                "accepted": accepted,
            })
            if accepted and row is not None:
                second_zero_accepted.append(row)
        # Keep every accepted pure-zeroing structure for this source.  The
        # existing Pareto/validation filters will decide which of these
        # candidates survive downstream; do not discard accepted candidates
        # here merely because another accepted candidate is simpler.
        # If zeroing produces no accepted row, retain the original formula as
        # the existing fallback base.
        zero_bases = (
            (list(zero_accepted) if zero_accepted else [baseline])
            if zero_test_enabled else [baseline]
        )
        print(
            f"Constant zeroing stage retained {len(zero_bases)} bases: ",
            flush=True,
        )
        for zero_index, zero_base in enumerate(zero_bases, start=1):
            after_r2 = _mode2_mean_r2(zero_base)
            if baseline_r2 is None or after_r2 is None:
                drop_text = "NA"
                accepted_text = "NA"
            else:
                drop = float(baseline_r2) - float(after_r2)
                drop_text = f"{drop:.8g}"
                accepted_text = "True" if drop <= r2_drop_threshold else "False"
            fallback_text = "; no acceptable zeroed candidate, falling back to the original formula" if zero_base is baseline else ""
            print(
                f"  zeroed basis[{zero_index}]: "
                f"{zero_base.get('readable_expression', '<formula missing>')} "
                f"| zeroed constants={zero_base.get('zeroed_constants', [])} "
                f"| original R²={baseline_r2} "
                f"| R² after zeroing={after_r2} "
                f"| drop={drop_text} "
                f"| threshold={r2_drop_threshold:.8g} "
                f"| accepted={accepted_text}{fallback_text}",
                flush=True,
            )
        retained = []

        def register_zero_base(zero_base):
            zero_base_id = str(zero_base.get("formula_id", ""))
            zero_base_record = zero_specs.get(zero_base_id)
            if zero_base_record is None:
                return
            # Every retained zero structure is a tree-training candidate.
            if source["id"] in single_ids:
                candidate_collection, result_collection = (
                    payload["single_candidates"], payload["single_best"])
                single_ids.add(zero_base_record["id"])
            else:
                candidate_collection, result_collection = (
                    payload["binary_combinations"], payload["combination_best"])
            if not any(item.get("id") == zero_base_record["id"]
                       for item in candidate_collection):
                candidate_collection.append(zero_base_record)
            if not any(item.get("formula_id") == zero_base_record["id"]
                       for item in result_collection):
                result_collection.append(zero_base)
            all_records[zero_base_record["id"]] = zero_base_record

        for candidate in second_zero_specs.values():
            zero_specs[candidate["id"]] = candidate
        for row in second_zero_accepted:
            register_zero_base(row)
            retained.append(row)

        for zero_base in zero_bases:
            if zero_base is not baseline:
                register_zero_base(zero_base)
            retained.append(zero_base)
        if constant_test_timed_out:
            # Do not discard completed candidates.  The normal downstream
            # R²/Pareto logic normally chooses among them.  For a timed-out
            # source, make the fallback explicit: choose the best completed
            # result by the same primary R² metric, with the original
            # baseline always included as a safe fallback.
            completed_rows = [
                row for row in [baseline, *retained]
                if row is not None and _mode2_mean_r2(row) is not None
            ]
            if completed_rows:
                retained = [max(
                    completed_rows,
                    key=lambda row: (
                        float(_mode2_mean_r2(row)),
                        -float(formula_complexity(
                            sp.sympify(row.get("readable_expression", "0"))
                        )),
                    ),
                )]
            local_checks.append({
                "formula_id": f"{source['id']}_timeout",
                "source_formula_id": source["id"],
                "operation": "mode5_constant_test_timeout",
                "timeout_seconds": constant_test_timeout,
                "accepted": False,
                "failure_reason": "constant_test_timeout",
                "fallback_formula_id": (
                    retained[0].get("formula_id") if retained else None
                ),
            })
        return retained, local_checks

    def zero_candidates(source: dict, baseline: dict,
                        parallel_workers: int | None = None,
                        dataset_parallel: bool = False, reference_rows=None,
                        test_phase: str = "initial"
                        ) -> tuple[list[dict], list[dict]]:
        mode5_chain_enabled = (
            int(pipeline_mode) == 5
            or config.get("periodic_constant_significance_experiment_mode") == 5
        )
        if mode5_chain_enabled:
            return mode5_zero_constant_candidates(
                source, baseline, parallel_workers, dataset_parallel, reference_rows,
                test_phase=test_phase)
        nonlocal exact_fit_count, reserved_fit_count
        local_checks = []
        expression = sp.sympify(source["readable_expression"])
        symbols = _mode2_parameter_symbols(expression)
        if not symbols:
            return [baseline], local_checks
        initial_values = {
            item.get("dataset_name", "dataset_1"): dict(zip(
                item.get("parameter_names", []), item.get("parameters", [])))
            for item in baseline.get("dataset_fit_results", [])}
        # Test all unchanged/zero combinations only.
        zero_assignments = [
            assignment for assignment in itertools.product((None, 0), repeat=len(symbols))
            if any(value == 0 for value in assignment)
        ]
        assignments = zero_assignments
        jobs = []
        candidate_specs = {}
        for assignment in assignments:
            replacements = {
                symbol: sp.Integer(value)
                for symbol, value in zip(symbols, assignment)
                if value is not None
            }
            candidate_expression = expression.xreplace(replacements)
            changed = {
                str(symbol): ("unchanged" if value is None else int(value))
                for symbol, value in zip(symbols, assignment)
            }
            zeroed = [name for name, value in changed.items() if value == 0]
            suffix = "_".join(
                f"{name}_{'keep' if value is None else value}"
                for name, value in zip(symbols, assignment)
            )
            candidate_id = f"{source['id']}_constant_{suffix}"
            operation = "exhaustive_zero_subset"
            if (config.get("data", {}).get("mode", "csv") == "csv"
                    and not _has_required_formula_variables(
                        candidate_expression, X.shape[1]
                    )):
                local_checks.append({
                    "formula_id": candidate_id, "source_formula_id": source["id"],
                    "operation": operation, "constant_assignments": changed,
                    "zeroed_constants": zeroed, "before_r2": _mode2_mean_r2(baseline),
                    "after_r2": None, "r2_drop": None,
                    "max_r2_drop": r2_drop_threshold, "accepted": False,
                    "failure_reason": "zeroing_would_leave_at_most_one_variable",
                })
                continue
            if bool(mode_cfg.get("require_log_abs_x1_initial", True)) and not _has_log_abs_x1(candidate_expression):
                local_checks.append({
                    "formula_id": candidate_id, "source_formula_id": source["id"],
                    "operation": operation, "constant_assignments": changed,
                    "zeroed_constants": zeroed, "before_r2": _mode2_mean_r2(baseline),
                    "after_r2": None, "r2_drop": None,
                    "max_r2_drop": r2_drop_threshold, "accepted": False,
                    "failure_reason": "missing_required_log_abs_x1",
                })
                continue
            if candidate_expression.has(sp.nan, sp.zoo, sp.oo, -sp.oo, sp.I):
                local_checks.append({
                    "formula_id": candidate_id, "source_formula_id": source["id"],
                    "operation": operation, "constant_assignments": changed,
                    "zeroed_constants": zeroed, "before_r2": _mode2_mean_r2(baseline),
                    "after_r2": None, "r2_drop": None,
                    "max_r2_drop": r2_drop_threshold, "accepted": False,
                    "failure_reason": "invalid_symbolic_value_after_constant_assignment",
                })
                continue
            candidate = dict(source, id=candidate_id, formula_id=candidate_id,
                             readable_expression=str(candidate_expression),
                             sympy_expression=sp.srepr(candidate_expression),
                             structure_type="mode2_constant_candidate",
                             source_formula_id=source["id"],
                             zeroed_constants=zeroed,
                             constant_assignments=changed)
            candidate_specs[candidate_id] = candidate
            candidate_key = sp.srepr(candidate_expression)
            if (candidate_id not in fit_cache and candidate_id not in exact_cache
                    and candidate_key not in fit_cache_by_expression):
                jobs.append((candidate, initial_values))
        with cache_lock:
            available = (len(jobs) if max_exact_fits is None else
                         max(0, max_exact_fits - exact_fit_count - reserved_fit_count))
            jobs = jobs[:available]
            reserved_fit_count += len(jobs)
        active_jobs = n_jobs if parallel_workers is None else int(parallel_workers)
        workers = 1 if active_jobs == 1 else min(
            max(1, (os.cpu_count() or 2) - 1), len(jobs),
            24 if active_jobs < 0 else active_jobs)
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                fitted = list(executor.map(
                    lambda item: fit_record(
                        item[0], item[1], _stable_lm_seed(seed, "mode2", item[0]["id"]),
                        dataset_parallel=dataset_parallel), jobs))
            for (candidate, _), (_attempts, row) in zip(jobs, fitted):
                with cache_lock:
                    reserved_fit_count -= 1
                    if not has_complete_dataset_fits(row):
                        continue
                    exact_cache[candidate["id"]] = row
                    fit_cache[candidate["id"]] = row
                    fit_cache_by_expression[sp.srepr(sp.sympify(
                        candidate["readable_expression"]))] = row
                    exact_fit_count += 1
        else:
            for candidate, values in jobs:
                cache_fit(candidate, values,
                          _stable_lm_seed(seed, "mode2", candidate["id"]),
                          dataset_parallel=dataset_parallel)
        baseline_r2 = _mode2_mean_r2(baseline)
        baseline_r2 = _mode2_mean_r2(baseline)
        zero_accepted = []
        for candidate_id, candidate in candidate_specs.items():
            row = fit_cache.get(candidate_id)
            if row is None:
                row = fit_cache_by_expression.get(
                    sp.srepr(sp.sympify(candidate["readable_expression"])))
            candidate_r2 = _mode2_mean_r2(row)
            drop = (None if baseline_r2 is None or candidate_r2 is None
                    else baseline_r2 - candidate_r2)
            accepted = drop is not None and drop <= r2_drop_threshold
            local_checks.append({
                "formula_id": candidate_id, "source_formula_id": source["id"],
                "operation": "exhaustive_zero_subset",
                "constant_assignments": candidate["constant_assignments"],
                "zeroed_constants": candidate["zeroed_constants"],
                "before_r2": baseline_r2, "after_r2": candidate_r2,
                "r2_drop": drop, "max_r2_drop": r2_drop_threshold,
                "accepted": accepted,
            })
            if accepted and row is not None:
                zero_accepted.append(row)

        # Keep the legacy simplest/highest-R² representative among accepted
        # zeroed candidates. The original fit is the fallback if none survive.
        retained = []
        if zero_accepted:
            retained.append(min(zero_accepted, key=lambda row: (
                int(row.get("complexity") or formula_complexity(
                    sp.sympify(row["readable_expression"]))),
                -(float(_mode2_mean_r2(row))
                  if _mode2_mean_r2(row) is not None else -np.inf),
                row["readable_expression"],
            )))

        if not retained:
            return [baseline], local_checks
        return retained, local_checks

    require_log_abs_x1 = (
        True if interval_layered_mode
        else bool(mode_cfg.get("require_log_abs_x1_initial", True))
    )
    rejected_by_log_abs_x1 = 0
    mode3_log_constraint_rejected_count = 0
    mode3_binary_constraint_rejected_count = 0
    mode3_allowed_cache = {}

    def mode3_formula_is_allowed(row: dict) -> bool:
        """Return whether a Mode 3 formula satisfies its structural rules."""
        cache_key = row.get("formula_id", row.get("id"))
        if cache_key in mode3_allowed_cache:
            return mode3_allowed_cache[cache_key]
        try:
            expression = sp.sympify(row["readable_expression"])
            if require_log_abs_x1 and not _has_log_abs_x1(expression):
                allowed = False
            elif (mode3_required_variable_count
                  and not _has_required_formula_variables(
                      expression, mode3_required_variable_count
                  )):
                allowed = False
            else:
                allowed = True
        except (KeyError, sp.SympifyError, TypeError, ValueError):
            allowed = False
        mode3_allowed_cache[cache_key] = allowed
        return allowed

    def enforce_mode3_log_constraint(rows: Sequence[dict]) -> list[dict]:
        """Drop every Mode 3 row violating its formula constraints."""
        nonlocal mode3_log_constraint_rejected_count, mode3_binary_constraint_rejected_count
        if not interval_layered_mode:
            return list(rows)
        kept = []
        for row in rows:
            compliant = mode3_formula_is_allowed(row)
            if compliant:
                kept.append(row)
            else:
                mode3_log_constraint_rejected_count += 1
                if mode3_required_variable_count:
                    try:
                        expression = sp.sympify(row["readable_expression"])
                        binary_violation = not _has_required_formula_variables(
                            expression, mode3_required_variable_count
                        )
                    except (KeyError, sp.SympifyError, TypeError, ValueError):
                        binary_violation = True
                    if binary_violation:
                        mode3_binary_constraint_rejected_count += 1
        return kept
    # Initial Mode 2 solutions come from formulas with an exact LM fit in the
    # initial training cache.  The later ranking strategy must not determine
    # which formulas are eligible for the common initial Top-K.
    exact_fit_rows = []
    for row in ranking:
        if row["formula_id"] not in fit_cache:
            continue
        if not mode3_formula_is_allowed(row) if interval_layered_mode else (
            require_log_abs_x1 and not _has_log_abs_x1(
                sp.sympify(row["readable_expression"])
            )
        ):
            rejected_by_log_abs_x1 += 1
            continue
        exact_fit_rows.append(row)
    # Select the initial candidates by frozen validation performance.  The
    # training fit is still used to obtain parameters; validation determines
    # which 15 completed structures enter the initial Pareto stage.
    if validation_selection_enabled:
        initial_validation_scores = {}
        for rank_row in exact_fit_rows:
            formula_id = rank_row["formula_id"]
            fitted_row = fit_cache[formula_id]
            validation_metrics = _evaluate_validation_row(
                fitted_row, selection_X, selection_y, selection_groups,
                dataset_groups, config["fitting"],
            )
            if validation_metrics is None:
                continue
            scored_row = copy.deepcopy(fitted_row)
            for key, value in validation_metrics.items():
                scored_row[f"validation_{key}"] = value
            _attach_training_validation_mean_r2_gap(scored_row)
            initial_validation_scores[formula_id] = scored_row
        exact_fit_rows = [
            row for row in exact_fit_rows
            if row["formula_id"] in initial_validation_scores
            and initial_validation_scores[row["formula_id"]].get(
                "validation_pooled_r2" if is_realdata_mat else "validation_mean_r2"
            ) is not None
        ]
        primary = "validation_pooled_r2" if is_realdata_mat else "validation_mean_r2"
        exact_fit_rows.sort(key=lambda row: (
            -float(initial_validation_scores[row["formula_id"]].get(primary, -np.inf)),
            float(initial_validation_scores[row["formula_id"]].get(
                "validation_r2_std", np.inf)),
            formula_complexity(sp.sympify(row["readable_expression"])),
            row["formula_id"],
        ))
        payload["metadata"]["initial_candidate_selection"] = {
            "basis": "validation_primary_r2",
            "primary_field": primary,
            "candidate_pool": len(exact_fit_rows),
            "selected_count": min(top_k, len(exact_fit_rows)),
        }
    else:
        initial_objective = "pooled_r2" if is_realdata_mat else "mean_r2"
        exact_fit_rows.sort(key=lambda row: (
            -float(fit_cache[row["formula_id"]].get(initial_objective, -np.inf)),
            row["formula_id"],
        ))
    if mode5_initial_selection_enabled and exact_fit_rows:
        # Mode 5 has a variable-sized initial pool: keep every initial fit
        # within 0.1 R² of the best initial fit.  This selection happens
        # before constant testing; the later zeroing threshold remains the
        # independent r2_drop_threshold (0.01 by default).
        initial_objective = "pooled_r2" if is_realdata_mat else "mean_r2"
        if validation_selection_enabled:
            def initial_score(row):
                return float(initial_validation_scores[row["formula_id"]].get(
                    "validation_pooled_r2" if is_realdata_mat else "validation_mean_r2",
                    -np.inf,
                ))
            selection_basis = "validation_primary_r2_window"
        else:
            def initial_score(row):
                return float(fit_cache[row["formula_id"]].get(
                    initial_objective, -np.inf
                ))
            selection_basis = "training_primary_r2_window"
        best_initial_score = max(initial_score(row) for row in exact_fit_rows)
        initial_r2_window = 0.1
        initial_cutoff = best_initial_score - initial_r2_window
        exact_fit_rows = [
            row for row in exact_fit_rows
            if initial_score(row) >= initial_cutoff
        ]
        window_candidate_count = len(exact_fit_rows)
        # Keep the Mode 5 R²-window rule, but cap the initial pool at the
        # configured Top-K.  ``exact_fit_rows`` is already sorted by the
        # selected validation/training score above, so truncation preserves
        # the best formulas within the window.
        exact_fit_rows = exact_fit_rows[:top_k]
        payload["metadata"]["initial_candidate_selection"] = {
            "basis": selection_basis,
            "primary_field": (
                "validation_pooled_r2" if is_realdata_mat
                else "validation_mean_r2"
            ) if validation_selection_enabled else initial_objective,
            "candidate_pool": window_candidate_count,
            "selected_count": len(exact_fit_rows),
            "r2_window": initial_r2_window,
            "max_selected_count": top_k,
            "best_r2": best_initial_score,
            "r2_cutoff": initial_cutoff,
        }
    initial_sources = (
        exact_fit_rows
        if mode4_crowding or mode5_initial_selection_enabled
        else exact_fit_rows[:top_k]
    )
    # The random-permutation ablation must start from exactly the same
    # post-constant-test initial pool as the tree model.  Its ranking is still
    # rebuilt below for later search, but the initial seed pool is supplied by
    # the main run's immutable checkpoint.
    if shared_initial_constant_state is not None:
        cached_sources = shared_initial_constant_state.get("initial_sources", [])
        if cached_sources:
            initial_sources = [dict(row) for row in cached_sources]
    initial_rows = []
    # Match the later-round granularity: one formula's candidate fits are
    # dispatched as independent formula/dataset process jobs.  Keep the
    # surrounding Top-K formula and subset scheduling serial to avoid nested
    # process pools; the 24-way parallelism lives at dataset level.
    initial_formula_workers, initial_subset_workers = 1, 24

    def process_initial_formula(rank_row):
        source = all_records[rank_row["formula_id"]]
        if shared_initial_constant_state is not None:
            cached = shared_initial_constant_state.get("by_source", {}).get(
                str(source["id"])
            )
            if cached is not None:
                return (
                    str(source["id"]),
                    copy.deepcopy(cached.get("rows", [])),
                    copy.deepcopy(cached.get("checks", [])),
                )
        if mode4_crowding:
            return str(source["id"]), [fit_cache[rank_row["formula_id"]]], []
        zero_rows, checks = zero_candidates(
            source, fit_cache[rank_row["formula_id"]],
            parallel_workers=initial_subset_workers,
            dataset_parallel=True,
            test_phase="initial",
        )
        return str(source["id"]), zero_rows, checks

    # Formula-level parallelism is also enabled for the initial Top-K pool.
    # Completion order drives progress updates, while the original index and
    # source formula ID preserve deterministic aggregation and attribution.
    initial_progress_label = (
        "Mode 5 initial R²-window constant-significance test"
        if mode5_initial_selection_enabled else
        "Mode 3 initial Top-K constant-significance test"
        if int(pipeline_mode) == 3 else
        "Mode 2 initial Top-K constant-significance test"
    )
    if initial_formula_workers > 1:
        with ThreadPoolExecutor(max_workers=initial_formula_workers) as executor:
            futures = {
                executor.submit(process_initial_formula, rank_row): index
                for index, rank_row in enumerate(initial_sources)
            }
            completed = [None] * len(initial_sources)
            with tqdm(
                total=len(initial_sources),
                desc=initial_progress_label,
                unit="formula",
            ) as progress:
                for future in as_completed(futures):
                    completed[futures[future]] = future.result()
                    progress.update(1)
            initial_processed = completed
    else:
        initial_processed = [
            process_initial_formula(row)
            for row in tqdm(
                initial_sources,
                total=len(initial_sources),
                desc=initial_progress_label,
                unit="formula",
            )
        ]
    for _source_id, zero_rows, checks in initial_processed:
        initial_rows.extend(enforce_mode3_log_constraint(zero_rows))
        significance_checks.extend(checks)
    if shared_initial_constant_state is None:
        payload["_mode2_initial_constant_state"] = {
            "initial_sources": [dict(row) for row in initial_sources],
            "by_source": {
                str(source_id): {
                    "rows": copy.deepcopy(zero_rows),
                    "checks": copy.deepcopy(checks),
                }
                for (source_id, zero_rows, checks) in initial_processed
            },
        }
    if is_realdata_mat:
        for row in initial_rows:
            _attach_piecewise_continuity_objective(
                row, row.get("readable_expression", ""), X
            )
    if validation_selection_enabled:
        # The initial batch uses the same non-dominated archive update as
        # later batches.
        initial_validation_rows = select_validation_pareto_front(initial_rows)
        initial_window_rows = validation_r2_window(initial_validation_rows)
        validation_archive = admit_incremental_validation_candidates(
            [], initial_window_rows
        )
        front = list(validation_archive)
        front_metadata = {
            "selection_basis": "validation_r2_window_incremental_nondominated_admission",
            "candidate_admission": "validation_r2_window",
        }
    else:
        validation_archive = None
        front, front_metadata = filter_nondominated_initial_seeds(
            initial_rows, use_min_r2=False,
            use_piecewise_difference=mode5_realdata_piecewise_objective,
            use_pooled_r2=is_realdata_mat,
        ) if initial_rows else ([], {})
        annotate_mode2_front(front)
    payload["top_ranked_refit_results"] = []
    for rank_row in initial_sources:
        cached_row = dict(fit_cache[rank_row["formula_id"]])
        # Cached LM rows predate the ranking stage and therefore do not carry
        # the fields expected by the existing CSV serializer.
        cached_row.update(
            predicted_rank=rank_row.get("predicted_rank"),
            ranking_score=rank_row.get("ranking_score"),
            formula_family=rank_row.get("formula_family"),
            is_training_formula=rank_row.get("is_training_formula", True),
        )
        payload["top_ranked_refit_results"].append(cached_row)
    # The shared final printer expects the legacy GP presentation fields.
    # Mode 2 uses the same Pareto objectives but does not run NSGA-II
    # crowding-distance calculation, so provide deterministic display values.
    for crowding_rank, row in enumerate(front, start=1):
        row.setdefault("crowding_rank", crowding_rank)
        row.setdefault("crowding_distance", 0.0)
        row.setdefault("crowding_distance_label", "N/A")
        row.setdefault("complexity", formula_complexity(
            sp.sympify(row["readable_expression"])
        ))
    # Complexity layers are defined from the accepted zeroing-stage results.
    if mode4_crowding:
        complexity_layer_rows = list(initial_rows)
    else:
        zero_only_rows = [
            row for row in initial_rows
            if _is_zero_only_constant_candidate(row)
        ]
        if (mode5_retrain_enabled
                or config.get("periodic_constant_significance_experiment_mode") == 5):
            # Keep every accepted zero structure, while using the simplest
            # one from each initial formula to define layer boundaries.
            zero_only_rows = _select_mode5_min_complexity_zero_bases(
                zero_only_rows
            )
        if validation_selection_enabled:
            zero_only_rows = select_validation_pareto_front(zero_only_rows)
            zero_only_rows = validation_r2_window(zero_only_rows)
            complexity_layer_front = admit_incremental_validation_candidates(
                [], zero_only_rows
            )
        else:
            complexity_layer_front, _ = filter_nondominated_initial_seeds(
                zero_only_rows,
                use_min_r2=False,
                use_piecewise_difference=mode5_realdata_piecewise_objective,
                use_pooled_r2=is_realdata_mat,
            )
        complexity_layer_rows = complexity_layer_front
    initial_complexity_intervals = (
        _mode2_initial_pareto_complexity_intervals(
            all_records.values(), complexity_layer_rows,
            allow_empty_front=is_realdata_mat,
        )
    )
    payload["metadata"]["complexity_layer_basis"] = (
        "mode5_min_complexity_post_zeroing"
        if (mode5_retrain_enabled
            or config.get("periodic_constant_significance_experiment_mode") == 5)
        else "post_zeroing"
        if not mode4_crowding else "pre_constant_test_mode4"
    )
    payload["metadata"]["mode2_initial_pareto_complexity_intervals"] = (
        initial_complexity_intervals
    )
    payload["metadata"]["mode3_complexity_layer_count"] = len(
        initial_complexity_intervals
    )
    mode_label = (
        "Mode 5" if mode5_retrain_enabled else
        "Mode 4" if mode4_crowding else
        ("Mode 3" if int(pipeline_mode) == 3 else "Mode 2")
    )

    def current_training_hypervolume(rows):
        """Calculate an online, front-local normalized training HV for logging."""
        objective_fields = (
            ("pooled_r2", "complexity")
            if is_realdata_mat else ("mean_r2", "r2_std", "complexity")
        )
        valid = []
        for source in rows:
            row = dict(source)
            try:
                if row.get("complexity") is None:
                    row["complexity"] = formula_complexity(
                        sp.sympify(row["readable_expression"])
                    )
                values = [float(row[field]) for field in objective_fields]
                if all(math.isfinite(value) for value in values):
                    valid.append(row)
            except (KeyError, TypeError, ValueError, OverflowError, sp.SympifyError):
                continue
        if not valid:
            return 0.0
        bounds = {}
        for field in objective_fields:
            values = [float(row[field]) for row in valid]
            bounds[field] = (min(values), max(values))
        return _calculate_validation_hv(
            valid, objective_fields, {objective_fields[0]}, bounds
        )

    print(f"{mode_label} Initial Top-K Pareto formula-library counts in adjacent complexity intervals ((lower, upper]):", flush=True)
    if initial_complexity_intervals:
        for interval in initial_complexity_intervals:
            print(
                f"  ({interval['lower_complexity']}, "
                f"{interval['upper_complexity']}]: "
                f"{interval['formula_count']} formulas",
                flush=True,
            )
    else:
        print("  The front has fewer than two distinct complexity boundaries; no interval can be formed.", flush=True)
    previous_front_keys, _initial_entry_count = print_new_mode2_pareto_entries(
        front, set(), "Initial front"
    )
    # Significance filtering compares a whole periodic window against the
    # Pareto front at that window's start, not against the immediately
    # preceding iteration's front.
    periodic_cycle_start_front_keys = set(previous_front_keys)
    payload["gp_pareto_front"] = front
    raw_front = copy.deepcopy(front)
    initial_hypervolume = current_training_hypervolume(front)
    print(
        f"{mode_label} Initial training Pareto: front={len(front)}, HV={initial_hypervolume:.8g}",
        flush=True,
    )
    hypervolume_history = [{
        "iteration": 0,
        "front": copy.deepcopy(front),
        "hypervolume": initial_hypervolume,
    }]
    payload["gp_constant_significance_checks"] = significance_checks
    payload["top_ranked_constant_significance_checks"] = significance_checks
    evaluated_ids.update(row["formula_id"] for row in initial_sources)
    initial_global_front_keys = {
        sp.srepr(sp.sympify(row["readable_expression"])) for row in front
    }
    global_formula_change_count = 0
    cumulative_global_change_keys = set()
    pareto_entry_trigger_count = 0
    iteration = 0
    surrogate_retrain_count = 0
    pending_constant_significance_tasks = []
    periodic_constant_significance_rounds = []
    # Between surrogate retrains ``ranking`` is immutable.  Rebuilding the
    # pending list and its five complexity buckets on every iteration used to
    # scan (and sort) the complete ranking repeatedly.  Keep the same
    # rank-ordered buckets and advance a cursor only when a formula is used.
    # A retrain replaces the ranking, so the cache is rebuilt below then.
    interval_selection_cache = None

    def refresh_interval_selection_cache() -> None:
        """Cache the current rank order by interval without changing selection.

        The uncached path first removes evaluated/disallowed rows, assigns
        each remaining row to its first matching interval, and selects the
        lowest ``(predicted_rank, formula_id)`` in each bucket.  This cache
        performs exactly that work once per ranking version; a per-bucket
        cursor subsequently exposes the same next row in each iteration.
        """
        nonlocal interval_selection_cache
        layer_bounds = [(
            float(item["lower_complexity"]),
            float(item["upper_complexity"]),
        ) for item in initial_complexity_intervals]
        buckets = [[] for _ in layer_bounds]
        rejected_count = 0
        for row in ranking:
            if row["formula_id"] in evaluated_ids:
                continue
            if not mode3_formula_is_allowed(row):
                rejected_count += 1
                continue
            try:
                value = row.get("complexity")
                if value is None:
                    value = formula_complexity(sp.sympify(
                        row["readable_expression"]
                    ))
                value = int(value)
            except (KeyError, sp.SympifyError, TypeError, ValueError, OverflowError):
                continue
            for layer, (lower, upper) in enumerate(layer_bounds):
                if lower < value <= upper:
                    buckets[layer].append(row)
                    break
        for bucket in buckets:
            bucket.sort(key=lambda row: (
                row.get("predicted_rank", 10**9),
                str(row.get("formula_id", "")),
            ))
        interval_selection_cache = {
            "layer_bounds": layer_bounds,
            "buckets": buckets,
            "next_indices": [0] * len(buckets),
            "rejected_count": rejected_count,
        }

    def select_cached_interval_batch() -> tuple[list[dict], list[dict]]:
        """Return the same per-interval representatives as the old scan."""
        if interval_selection_cache is None:
            return [], []
        selected, metadata = [], []
        for index, (lower, upper) in enumerate(
                interval_selection_cache["layer_bounds"]):
            bucket = interval_selection_cache["buckets"][index]
            next_index = interval_selection_cache["next_indices"][index]
            available_count = len(bucket) - next_index
            if available_count:
                selected.append(bucket[next_index])
                interval_selection_cache["next_indices"][index] = next_index + 1
            metadata.append({
                "layer": index + 1,
                "lower_complexity": lower,
                "upper_complexity": upper,
                "available_count": available_count,
            })
        selected.sort(key=lambda row: row.get("predicted_rank", 10**9))
        return selected, metadata

    if interval_layered_mode:
        refresh_interval_selection_cache()
    mode5_prefetch_executor = None
    mode5_prefetch_queue = []
    # Periodic significance testing may run in iteration 1.  Seed its
    # reference window with the initial Pareto front before entering the loop.
    previous_front = list(front)
    iteration_limit = (max_iterations if forced_iterations is None else
                       min(max_iterations, max(0, int(forced_iterations))))
    while (iteration < iteration_limit
           and (max_exact_fits is None or exact_fit_count < max_exact_fits)
           ):
        iteration += 1
        periodic_constant_significance_due = (
            periodic_constant_significance_enabled
            and iteration % periodic_constant_significance_interval == 0
        )
        newly_entered_count = 0
        # Keep the initial surrogate ranking fixed.  Mode 2 intentionally
        prefetched_baselines_by_index = None
        # spends its budget on exact LM evaluations and Pareto updates; it
        # does not retrain the tree model after each batch.
        if interval_layered_mode:
            # Preserve the legacy per-iteration rejection accounting even
            # though disallowed rows are now filtered once per ranking.
            rejected_by_log_abs_x1 += interval_selection_cache["rejected_count"]
            selected, layer_metadata = select_cached_interval_batch()
            # Keep the legacy key for downstream diagnostics while exposing
            # the new fixed-quota layer metadata used by Mode 3/4 fitting.
            payload["metadata"]["mode3_last_complexity_intervals"] = layer_metadata
            payload["metadata"]["mode3_last_complexity_layers"] = layer_metadata
            print(
                f"{mode_label} Parallel fitting by complexity layer: "
                + "; ".join(
                    f"layer {item['layer']}=({item['lower_complexity']:.3g},"
                    f"{item['upper_complexity']:.3g}]"
                    f"/candidate {item['available_count']}"
                    for item in layer_metadata
                ), flush=True,
            )
        else:
            pending = []
            for row in ranking:
                if row["formula_id"] in evaluated_ids:
                    continue
                if require_log_abs_x1 and not _has_log_abs_x1(
                        sp.sympify(row["readable_expression"])):
                    rejected_by_log_abs_x1 += 1
                    continue
                pending.append(row)
        if not interval_layered_mode:
            selected = pending[:batch_size]
        if not selected:
            break
        selected_tasks = list(selected)
        # Phase 1: fit every original formula in this round first.  This is
        # Mode 3/4 now use the same formula×dataset process pool as the
        # initial LM stage.  Keep the older path for Mode 2 and piecewise
        # constant-significance experiments.
        if interval_layered_mode and not piecewise_enabled:
            if prefetched_baselines_by_index is None:
                baselines_by_index = fit_selected_formulas_initial_style(selected_tasks)
            else:
                baselines_by_index = prefetched_baselines_by_index
            baselines = [baselines_by_index[index]
                         for index in range(len(selected_tasks))]
            later_round_workers = min(
                len(selected_tasks) * len(dataset_groups),
                requested_workers if requested_workers > 0 else DEFAULT_LM_MAX_WORKERS,
            )
            max_later_round_workers = max(max_later_round_workers, later_round_workers)
        else:
            later_round_workers = min(len(selected_tasks), requested_workers)
            max_later_round_workers = max(
                max_later_round_workers, later_round_workers
            )
            dataset_parallel_for_formula = later_round_workers == 1
            with ThreadPoolExecutor(max_workers=min(later_round_workers, len(selected_tasks))) as executor:
                baseline_futures = {
                    executor.submit(
                        cache_fit,
                        all_records[rank_row["formula_id"]],
                        None,
                        _stable_lm_seed(seed, "mode2", rank_row["formula_id"]),
                        dataset_parallel_for_formula,
                    ): index
                    for index, rank_row in enumerate(selected_tasks)
                }
                baselines = [None] * len(selected_tasks)
                with tqdm(
                    total=len(selected_tasks),
                    desc=f"{mode_label} round {iteration}: fitting original formulas",
                    unit="formula",
                ) as progress:
                    for future in as_completed(baseline_futures):
                        baselines[baseline_futures[future]] = future.result()
                        progress.update(1)

        current_formula_tasks = [
            (all_records[rank_row["formula_id"]], baselines[index])
            for index, rank_row in enumerate(selected_tasks)
        ]
        if periodic_constant_significance_enabled:
            pending_constant_significance_tasks.extend(
                task for task in current_formula_tasks if task[1] is not None
            )
        if periodic_constant_significance_due:
            # Only test formulas that newly enter the cumulative retained set
            # during this period.  The pending queue contains all successfully
            # fitted selections, so filter it against a provisional retained
            # set made from their original (pre-significance) fits first.
            period_tasks = list(pending_constant_significance_tasks)
            period_rows = [baseline for _source, baseline in period_tasks
                           if baseline is not None]
            previous_period_keys = set(periodic_cycle_start_front_keys)
            if validation_selection_enabled:
                provisional_validation_rows = []
                for source_row in period_rows:
                    metrics = _evaluate_validation_row(
                        source_row, selection_X, selection_y,
                        selection_groups, dataset_groups, config["fitting"],
                    )
                    if metrics is None:
                        continue
                    row = copy.deepcopy(source_row)
                    for key, value in metrics.items():
                        row[f"validation_{key}"] = value
                    _attach_training_validation_mean_r2_gap(row)
                    row["validation_complexity"] = formula_complexity(
                        sp.sympify(row["readable_expression"])
                    )
                    provisional_validation_rows.append(row)
                provisional_accepted = validation_r2_window(
                    provisional_validation_rows,
                    reference_rows=previous_front,
                )
                provisional_front = admit_incremental_validation_candidates(
                    validation_archive or [], provisional_accepted
                )
            else:
                provisional_accepted = mode2_r2_window(
                    period_rows, reference_rows=previous_front
                )
                provisional_front, _ = filter_nondominated_initial_seeds(
                    [*previous_front, *provisional_accepted],
                    use_min_r2=False,
                    use_piecewise_difference=mode5_realdata_piecewise_objective,
                    use_pooled_r2=is_realdata_mat,
                )
            provisional_new_keys = {
                sp.srepr(sp.sympify(row["readable_expression"]))
                for row in provisional_front
            } - previous_period_keys
            significance_tasks = [
                task for task in period_tasks
                if sp.srepr(sp.sympify(task[0]["readable_expression"]))
                in provisional_new_keys
            ]
            # Even formulas filtered out of the significance task list have
            # already received their original baseline fit in this period;
            # mark their source IDs as evaluated so they cannot be selected
            # again in a later round.
            evaluated_ids.update(
                str(source["id"]) for source, baseline in period_tasks
                if baseline is not None
            )
        else:
            significance_tasks = current_formula_tasks

        # Phase 2: constant-significance checks are formula-level serial.
        # In Mode 5, each source completes zero-subset selection before it
        # enumerates fixed-sign (±1) assignments on the selected zeroed structure.
        # fit_record(..., dataset_parallel=True) parallelizes that candidate's
        # independent dataset fits.  This is the intended outer-serial,
        # inner-dataset-parallel scheduling and avoids competing formula-level
        # pools.
        def process_selected_formula(task):
            source, baseline = task
            if baseline is None:
                return str(source["id"]), [], []
            if (not periodic_constant_significance_enabled
                    or not periodic_constant_significance_due):
                return str(source["id"]), [baseline], []
            zero_rows, checks = zero_candidates(
                source, baseline, parallel_workers=24,
                dataset_parallel=True, reference_rows=previous_front,
                test_phase="periodic",
            )
            return str(source["id"]), zero_rows, checks

        round_rows = []
        significance_sources_by_id = {
            str(source["id"]): source
            for source, _baseline in significance_tasks
        }
        processed_tasks = [None] * len(significance_tasks)
        with tqdm(
            total=len(significance_tasks),
            desc=(
                f"Formula round {iteration}: constant-significance test for newly added formulas"
                if periodic_constant_significance_due else
                f"Mode 3 round {iteration}: original formula summary"
                if int(pipeline_mode) == 3
                  else f"Mode 2 round {iteration}: constant-significance test"),
            unit="formula",
        ) as progress:
            # Keep this loop in the main process.  Mode 5 candidate fitting
            # now uses forked watchdog children; forking from a worker thread
            # is unsafe on POSIX and can deadlock libraries such as SciPy.
            for task_index, task in enumerate(significance_tasks):
                result = process_selected_formula(task)
                processed_tasks[task_index] = result
                if periodic_constant_significance_due:
                    formula_id, rows, _checks = result
                    source = significance_sources_by_id.get(formula_id, {})
                    original_expression = source.get(
                        "readable_expression", "<original formula missing>"
                    )
                    print(
                        f"\n{mode_label} constant test complete: formula={formula_id}\n"
                        f"  original formula: {original_expression}",
                        flush=True,
                    )
                    if rows:
                        for row_index, row in enumerate(rows, start=1):
                            print(
                                f"  Updated formula [{row_index}]: "
                                f"{row.get('readable_expression', '<formula missing>')} "
                                f"| zeroed constants={row.get('zeroed_constants', [])} "
                                f"| constant assignments={row.get('constant_assignments', {})}",
                                flush=True,
                            )
                    else:
                        print("  Updated formula: (no acceptable candidate)", flush=True)
                progress.update(1)
        if periodic_constant_significance_due:
            periodic_constant_significance_rounds.append(iteration)
            pending_constant_significance_tasks.clear()
        for formula_id, rows, checks in processed_tasks:
            evaluated_ids.add(formula_id)
            round_rows.extend(rows)
            significance_checks.extend(checks)
        pareto_updated = False
        round_rows = enforce_mode3_log_constraint(round_rows)
        iterative_fitted_rows.extend(round_rows)
        if round_rows:
            if is_realdata_mat:
                for row in round_rows:
                    _attach_piecewise_continuity_objective(
                        row, row.get("readable_expression", ""), X
                    )
            # Raw front: merge all newly evaluated rows without the R²
            # admission window. This is retained for the diagnostic HV plot.
            raw_front, raw_front_metadata = filter_nondominated_initial_seeds(
                [*raw_front, *round_rows], use_min_r2=False,
                use_piecewise_difference=mode5_realdata_piecewise_objective,
                use_pooled_r2=is_realdata_mat)
            annotate_mode2_front(raw_front)
            previous_front = list(front)
            if validation_selection_enabled:
                # Evaluate validation metrics before applying the admission
                # window.  Both admission and Pareto dominance are based on
                # validation metrics; training metrics remain available only
                # for fitting/ranking traceability.
                validation_round_rows = []
                for source in round_rows:
                    metrics = _evaluate_validation_row(
                        source, selection_X, selection_y, selection_groups,
                        dataset_groups, config["fitting"],
                    )
                    if metrics is None:
                        continue
                    row = copy.deepcopy(source)
                    for key, value in metrics.items():
                        row[f"validation_{key}"] = value
                    _attach_training_validation_mean_r2_gap(row)
                    row["validation_complexity"] = formula_complexity(
                        sp.sympify(row["readable_expression"])
                    )
                    validation_round_rows.append(row)
                accepted_round_rows = validation_r2_window(
                    validation_round_rows, reference_rows=previous_front
                )
                validation_archive = admit_incremental_validation_candidates(
                    validation_archive, accepted_round_rows
                )
                front = list(validation_archive)
                front_metadata = {
                    "selection_basis": "validation_r2_window_incremental_nondominated_admission",
                    "candidate_admission": "validation_r2_window",
                }
            else:
                accepted_round_rows = mode2_r2_window(
                    round_rows, reference_rows=previous_front
                )
                front, front_metadata = filter_nondominated_initial_seeds(
                    [*previous_front, *accepted_round_rows], use_min_r2=False,
                use_piecewise_difference=mode5_realdata_piecewise_objective,
                    use_pooled_r2=is_realdata_mat)
                annotate_mode2_front(front)
            for crowding_rank, row in enumerate(front, start=1):
                row.setdefault("crowding_rank", crowding_rank)
                row.setdefault("crowding_distance", 0.0)
                row.setdefault("crowding_distance_label", "N/A")
                row.setdefault("complexity", formula_complexity(
                    sp.sympify(row["readable_expression"])
                ))
            previous_front_keys, newly_entered_count = print_new_mode2_pareto_entries(
                front, previous_front_keys, f"Round {iteration}"
            )
            if periodic_constant_significance_due:
                periodic_cycle_start_front_keys = set(previous_front_keys)
            pareto_entry_trigger_count += newly_entered_count
            pareto_updated = True
        if pareto_updated:
            after = {sp.srepr(sp.sympify(row["readable_expression"])) for row in front}
            cumulative_global_change_keys.update(
                initial_global_front_keys.symmetric_difference(after)
            )
            global_formula_change_count = len(cumulative_global_change_keys)
        updated = pareto_updated and global_formula_change_count > 0
        print(f"{mode_label}  iteration {iteration}: selected {len(selected)} items, cumulative exact fits  "
              f"{exact_fit_count}, Pareto={len(front)}, "
              f"{'updated' if updated else ('kept previous front' if not pareto_updated else 'no update')} (global changes from initial "
              f"{global_formula_change_count}; new solutions entered cumulatively "
              f"{pareto_entry_trigger_count})", flush=True)
        iteration_hypervolume = current_training_hypervolume(front)
        print(
            f"{mode_label}  iteration {iteration}  metrics: training Pareto={len(front)}, "
            f"HV={iteration_hypervolume:.8g}",
            flush=True,
        )
        payload["gp_pareto_front"] = front
        hypervolume_history.append({
            "iteration": iteration,
            "front": copy.deepcopy(front),
            "pareto_updated": pareto_updated,
            "hypervolume": iteration_hypervolume,
        })
        payload["mode2_hypervolume_history"] = hypervolume_history
        save_checkpoint(output_dir, payload, f"mode2_iteration_{iteration}")
        if mode5_retrain_enabled:
            retrain_due = (
                ranking_strategy == "tree"
                and iteration % mode5_retrain_interval == 0
            )
            print(
                f"Retraining check: round {iteration}, interval={mode5_retrain_interval}, "
                f"this round={'retrain' if retrain_due else 'no retraining'}, "
                f"completed retrains={surrogate_retrain_count} times",
                flush=True,
            )
        if (mode5_retrain_enabled and ranking_strategy == "tree"
                and iteration % mode5_retrain_interval == 0):
            retrain_config = dict(ranking_config)
            retrain_config.update(
                enabled=True,
                ablation_enabled=False,
                training_label_field=tree_training_label,
            _log_context=f"Round {iteration}: retraining on cumulative results",
            )
            ranking, ranking_metadata = train_and_rank(
                payload, retrain_config, seed + iteration, output_dir
            )
            ranking_metadata["training_label_field"] = tree_training_label
            payload["structure_r2_ranking"] = ranking
            payload["ranking_metadata"] = ranking_metadata
            surrogate_retrain_count += 1
            refresh_interval_selection_cache()
            print(
                f"Round {iteration} tree-model retraining complete: "
                f"formulas fitted so far={len(evaluated_ids)}, "
                f"training samples={ranking_metadata.get('training_samples')}, "
                f"retrain count={surrogate_retrain_count}",
                flush=True,
            )
    if mode5_prefetch_executor is not None:
        mode5_prefetch_executor.shutdown(wait=True)
    annotate_mode2_front(front)
    front = enforce_mode3_log_constraint(front)
    for crowding_rank, row in enumerate(front, start=1):
        row.setdefault("crowding_rank", crowding_rank)
        row.setdefault("crowding_distance", 0.0)
        row.setdefault("crowding_distance_label", "N/A")
        row.setdefault("complexity", formula_complexity(
            sp.sympify(row["readable_expression"])
        ))
    payload["gp_pareto_front"] = front
    payload["mode2_hypervolume_history"] = hypervolume_history
    payload["mode2_raw_pareto_front"] = raw_front
    final_training_hypervolume = current_training_hypervolume(front)
    payload["metadata"]["mode34_final_training_hypervolume"] = final_training_hypervolume
    print(
        f"{mode_label} Final training Pareto: front={len(front)}, "
        f"HV={final_training_hypervolume:.8g}, cumulative exact fits={exact_fit_count}",
        flush=True,
    )
    payload["mode2_exact_fit_cache"] = exact_cache
    if interval_layered_mode and ranking_strategy == "tree":
        layer_spearman = _mode3_complexity_layer_spearman(
            ranking, iterative_fitted_rows, initial_complexity_intervals,
        )
        payload["metadata"]["mode3_complexity_layer_spearman"] = layer_spearman
        print(f"\n{mode_label} Spearman ρ by complexity layer: tree ranking vs fitted mean_r2 ranking", flush=True)
        for item in layer_spearman:
            rho = (f"{item['spearman_rho']:.6f}"
                   if item["spearman_rho"] is not None else "N/A")
            print(
                f"  layer {item['layer']}=({item['lower_complexity']},"
                f"{item['upper_complexity']}]: fitted {item['fitted_count']}, "
                f"Spearman ρ={rho}",
                flush=True,
            )
    payload["gp_metadata"] = {
        "enabled": True, "mode": "surrogate_assisted_search",
        "ranking_strategy": ranking_strategy,
        "tree_model_trained": ranking_strategy == "tree",
        "initial_top_k": top_k, "batch_size": batch_size,
        "later_round_workers": max_later_round_workers,
        "later_round_dataset_workers": (
            24 if max_later_round_workers == 1 else 1
        ),
        "later_round_formula_scheduling": "parallel",
        "later_round_nested_process_pool": max_later_round_workers == 1,
        "later_round_fit_strategy": (
            "periodic_constant_significance_for_experiment"
            if periodic_constant_significance_enabled else
            "originals_only_no_constant_zeroing"
            if int(pipeline_mode) in (3, 4, 5, 6) else
            "originals_then_dataset_parallel_zero_subsets"
        ),
        # Mode 3 normally tests only the initial Top-K; Mode 5/6 in CSV and
        # can add the configured periodic tests for new formulas.
        "constant_significance_enabled": True,
        "initial_constant_significance_enabled": True,
        "later_constant_significance_enabled": (
            periodic_constant_significance_enabled or int(pipeline_mode) == 2
        ),
        "iterations": iteration, "max_iterations": max_iterations,
        "global_formula_change_count": global_formula_change_count,
        "pareto_entry_trigger_count": pareto_entry_trigger_count,
        "max_exact_fits": max_exact_fits, "exact_fit_count": exact_fit_count,
        "r2_drop_threshold": r2_drop_threshold, "initial_front_metadata": front_metadata,
        "interval_layered_selection": interval_layered_mode,
        "required_log_abs_x1": require_log_abs_x1,
        "rejected_by_log_abs_x1": rejected_by_log_abs_x1,
        "mode3_log_constraint_rejected_count": mode3_log_constraint_rejected_count,
        "mode3_requires_binary_formula": mode3_required_variable_count > 1,
        "mode3_required_formula_variable_count": mode3_required_variable_count,
        "mode3_binary_constraint_rejected_count": mode3_binary_constraint_rejected_count,
        "stop_reason": (
            "max_exact_fits" if (max_exact_fits is not None
                                  and exact_fit_count >= max_exact_fits) else
            "max_iterations" if iteration >= max_iterations else
            "no_unseen_candidates"
        ),
    }
    payload["metadata"].update(
        mode2="surrogate_assisted_search", pipeline_mode=int(pipeline_mode),
        mode3_constant_zeroing_disabled=(
            interval_layered_mode and not periodic_constant_significance_enabled
        ),
        mode2_iterations=iteration,
        mode2_initial_iteration_budget=40,
        mode2_initial_training_formula_count=initial_training_formula_count,
        mode2_exact_fit_count=exact_fit_count,
        # Formula usage is counted by original generated formula ID.  A
        # zero-constant candidate remains part of its source formula and must
        # not inflate this count.
        mode2_subsequent_formula_count=max(
            0, len(evaluated_ids) - initial_training_formula_count
        ),
        mode2_total_used_formula_count=len(evaluated_ids),
        mode2_evaluated_formula_count=len(evaluated_ids),
        mode2_surrogate_retrained=surrogate_retrain_count > 0,
        mode5_fit_batch_rounds=mode5_fit_batch_rounds,
        mode5_retrain_enabled=mode5_retrain_enabled,
        mode5_retrain_interval=mode5_retrain_interval,
        mode5_surrogate_retrain_count=surrogate_retrain_count,
        periodic_constant_significance_enabled=(
            periodic_constant_significance_enabled
        ),
        periodic_constant_significance_interval=(
            periodic_constant_significance_interval
        ),
        periodic_constant_significance_pending_count=(
            len(pending_constant_significance_tasks)
        ),
        periodic_constant_significance_experiment_mode=config.get(
            "periodic_constant_significance_experiment_mode"
        ),
        periodic_constant_significance_rounds=periodic_constant_significance_rounds,
        early_stopping_disabled=disable_early_stopping,
        target_iterations=40,
        mode2_ranking_strategy=ranking_strategy,
        mode2_quality_metric=(
            "pooled_r2" if is_realdata_mat else "mean_r2"
        ),
        mode5_realdata_piecewise_objective=mode5_realdata_piecewise_objective,
        mode2_r2_window_reference="dynamic_previous_front",
        mode2_r2_window_reference_count=0,
        constant_significance_enabled=True,
        constant_significance_stage=(
            "formula_periodic_new_formula_rounds_plus_initial"
            if periodic_constant_significance_enabled else
            "mode3_initial_and_mode2_exhaustive_zero_subsets"
            if interval_layered_mode else "mode2_exhaustive_zero_subsets"
        ),
        constant_significance_max_r2_drop=r2_drop_threshold)
    payload["top_ranked_constant_significance_checks"] = significance_checks
    return payload


def _mode2_compare_tree_random(
    tree_payload: dict, random_payload: dict, output_dir: Path,
) -> dict:
    """Compare the two Mode 2 searches with one shared HV reference point."""
    tree_front = list(tree_payload.get("gp_pareto_front", []))
    random_front = list(random_payload.get("gp_pareto_front", []))
    fronts = [tree_front, random_front]
    tree_history = tree_payload.get("mode2_hypervolume_history", [])
    random_history = random_payload.get("mode2_hypervolume_history", [])
    # Normalize against every saved iteration of both methods.  This gives
    # one exact empirical scale for the complete comparison history.
    normalization_rows = [
        row
        for history in (tree_history, random_history)
        for item in history
        for row in item.get("front", [])
    ]
    if not normalization_rows:
        normalization_rows = [row for front in fronts for row in front]
    is_realdata_mat = (
        _is_measured_pathloss_profile(tree_payload.get("metadata", {}).get("data_profile") or tree_payload.get("metadata", {}).get("data_mode"))
    )
    objective_fields = (
        ("pooled_r2", "complexity")
        if is_realdata_mat else ("mean_r2", "r2_std", "complexity")
    )
    maximize_fields = {objective_fields[0]}
    objective_values = {field: [] for field in objective_fields}
    for row in normalization_rows:
        try:
            row_values = {}
            for field in objective_fields:
                value = row.get(field)
                if field == "complexity" and value is None:
                    value = formula_complexity(sp.sympify(row["readable_expression"]))
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError
                row_values[field] = value
            for field, value in row_values.items():
                objective_values[field].append(value)
        except (KeyError, TypeError, ValueError, sp.SympifyError):
            continue
    # These bounds are derived after the search has finished, from every
    # saved iteration of both methods.  The same bounds are shared by both
    # methods and every iteration; online selection remains unaffected.
    normalization_bounds = {}
    references = {}
    for field in objective_fields:
        values = objective_values[field]
        if not values:
            lower, upper = (0.0, 1.0)
        else:
            lower, upper = min(values), max(values)
        normalization_bounds[field] = (lower, upper)
        references[field] = lower if field in maximize_fields else upper
    reference_parameters = _write_hv_reference_parameters(
        output_dir, "mode2_hv_reference_parameters", objective_fields,
        maximize_fields, normalization_bounds, "comparison",
        "all_iterations_tree_and_random_pareto_fronts",
    )

    def metrics(front):
        tagged = [dict(row, **{
            f"_hv_ref_{field}": references[field] for field in objective_fields
        }) for row in front]
        valid_primary = []
        for row in tagged:
            try:
                value = float(row[objective_fields[0]])
                if math.isfinite(value):
                    valid_primary.append(value)
            except (KeyError, TypeError, ValueError, sp.SympifyError):
                continue
        return {
            "pareto_front_size": len(front),
            "hypervolume": _mode2_hypervolume(
                tagged, objective_fields, maximize_fields, normalization_bounds
            ),
            "hypervolume_dimensions": len(objective_fields),
            "hypervolume_objectives": list(objective_fields),
            "hypervolume_normalized": True,
            "best_primary_objective": max(valid_primary) if valid_primary else None,
            "evaluated_formula_count": tree_payload.get("metadata", {}).get(
                "mode2_evaluated_formula_count", 0) if front is tree_front else
                random_payload.get("metadata", {}).get(
                    "mode2_evaluated_formula_count", 0),
            "iterations": (tree_payload if front is tree_front else random_payload
                            ).get("metadata", {}).get("mode2_iterations", 0),
        }

    tree_metrics, random_metrics = metrics(tree_front), metrics(random_front)
    tree_hv, random_hv = tree_metrics["hypervolume"], random_metrics["hypervolume"]
    hv_delta = tree_hv - random_hv
    if tree_hv > random_hv + 1e-12:
        conclusion = "Tree-model ranking outperforms random ordering overall (higher HV with the shared reference point)."
        winner = "tree"
    elif random_hv > tree_hv + 1e-12:
        conclusion = "Random ordering outperforms tree-model ranking overall (higher HV with the shared reference point)."
        winner = "random"
    else:
        conclusion = "The two rankings have similar HV; the current data does not indicate which is better."
        winner = "tie"
    comparison = {
        "experiment": "mode2_tree_vs_random_permutation",
        "shared_hypervolume_reference": {
            **references,
            "normalization_bounds": {
                field: {"min": bounds[0], "max": bounds[1]}
                for field, bounds in normalization_bounds.items()
            },
            "normalization_scope": "all_iterations_tree_and_random_pareto_fronts",
            "normalization_row_count": len(normalization_rows),
            "maximize": sorted(maximize_fields),
            "minimize": [field for field in objective_fields
                          if field not in maximize_fields],
            "objectives": list(objective_fields),
            "reference_parameters_file": reference_parameters["json_file"],
        },
        "tree_model": tree_metrics,
        "random_permutation": random_metrics,
        "hypervolume_delta_tree_minus_random": hv_delta,
        "winner_by_hypervolume": winner,
        "conclusion": conclusion,
        "same_initial_fits": True,
        "same_iteration_count_target": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tree_vs_random_comparison.json").write_text(
        json.dumps(_json_value(comparison), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fields = ["method", *tree_metrics.keys()]
    with (output_dir / "tree_vs_random_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"method": "tree_model", **tree_metrics})
        writer.writerow({"method": "random_permutation", **random_metrics})

    # Compare the two runs at every recorded iteration using one shared
    # normalization/reference point, so the curves remain comparable as the
    # fronts evolve.
    tree_by_iteration = {int(item["iteration"]): item.get("front", [])
                         for item in tree_history}
    random_by_iteration = {int(item["iteration"]): item.get("front", [])
                           for item in random_history}
    history_rows = []
    for iteration in sorted(set(tree_by_iteration) | set(random_by_iteration)):
        iteration_tree = tree_by_iteration.get(iteration, tree_front)
        iteration_random = random_by_iteration.get(iteration, random_front)
        def iteration_hv(front_rows):
            tagged = [dict(row, **{
                f"_hv_ref_{field}": references[field]
                for field in objective_fields
            })
                      for row in front_rows]
            return _mode2_hypervolume(
                tagged, objective_fields, maximize_fields, normalization_bounds
            )

        tree_iteration_hv = iteration_hv(iteration_tree)
        random_iteration_hv = iteration_hv(iteration_random)
        history_rows.append({
            "iteration": iteration,
            "tree_model_hypervolume": tree_iteration_hv,
            "random_permutation_hypervolume": random_iteration_hv,
            "hypervolume_delta_tree_minus_random": tree_iteration_hv - random_iteration_hv,
            "tree_pareto_front_size": len(iteration_tree),
            "random_pareto_front_size": len(iteration_random),
        })
    comparison["per_iteration_hypervolume"] = history_rows
    history_fields = list(history_rows[0]) if history_rows else [
        "iteration", "tree_model_hypervolume", "random_permutation_hypervolume",
        "hypervolume_delta_tree_minus_random", "tree_pareto_front_size",
        "random_pareto_front_size",
    ]
    with (output_dir / "hypervolume_by_iteration.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=history_fields)
        writer.writeheader()
        writer.writerows(history_rows)
    (output_dir / "tree_vs_random_comparison.json").write_text(
        json.dumps(_json_value(comparison), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if history_rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(8, 5.5))
        iterations = [row["iteration"] for row in history_rows]
        axis.plot(iterations, [row["tree_model_hypervolume"] for row in history_rows],
                  marker="o", ms=8, linewidth=1.75, label="Tree model")
        axis.plot(iterations, [row["random_permutation_hypervolume"] for row in history_rows],
                  marker="s", ms=8, linewidth=1.75, label="Random permutation")
        axis.set_xlabel("Iteration", fontsize=15)
        axis.set_ylabel("2D Hypervolume", fontsize=15)
        axis.set_title("Mode 2 HV (window-filtered): tree model vs random ablation", fontsize=15)
        axis.tick_params(axis="both", labelsize=13)
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "hypervolume_by_iteration.png", dpi=160)
        plt.close(figure)
    print("\nMode 2 automated ablation conclusion: " + conclusion, flush=True)
    print(
        f"  Tree model: HV={tree_hv:.8g}, best primary objective={tree_metrics['best_primary_objective']}; "
        f"Random ordering: HV={random_hv:.8g}, best primary objective={random_metrics['best_primary_objective']}; "
        f"HV difference (tree - random)={hv_delta:.8g}",
        flush=True,
    )
    print(f"  Comparison file: {output_dir / 'tree_vs_random_comparison.json'}", flush=True)
    print(f"  Per-iteration HV data: {output_dir / 'hypervolume_by_iteration.csv'}", flush=True)
    print(f"  Per-iteration HV plot: {output_dir / 'hypervolume_by_iteration.png'}", flush=True)
    return comparison




def _next_mode4_experiment_directory(
    output_root: Path, prefix: str = "mode4_experiment",
) -> Path:
    """Allocate the next monotonic directory for a Mode 4/5 experiment."""
    output_root.mkdir(parents=True, exist_ok=True)
    used = []
    patterns = [f"{prefix}_*"]
    # Continue the sequence after directories created by older versions that
    # included the pipeline mode in the experiment directory name.
    if prefix == "experiment":
        patterns.extend(f"mode{mode}_experiment_*" for mode in (4, 5, 6))
    for pattern in patterns:
        for path in output_root.glob(pattern):
            if not path.is_dir():
                continue
            suffix = path.name.rsplit("_", 1)[-1]
            if suffix.isdigit():
                used.append(int(suffix))
    number = max(used, default=0) + 1
    while True:
        candidate = output_root / f"{prefix}_{number:03d}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            number += 1


def _latest_mode4_experiment_directory(
    output_root: Path, prefix: str,
) -> Path | None:
    """Return the newest existing Mode 4/5/6 experiment directory."""
    candidates = []
    for path in output_root.glob(f"{prefix}_*"):
        if not path.is_dir():
            continue
        suffix = path.name.rsplit("_", 1)[-1]
        if suffix.isdigit():
            candidates.append((int(suffix), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _mode4_seed_completion_path(seed_directory: Path) -> Path:
    return seed_directory / "seed_completion.json"


def _mode4_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _load_legacy_mode4_seed_completion(
    seed_directory: Path, experiment_mode: int, seed: int,
) -> dict | None:
    """Recover completion state from pre-marker Mode 5/6 artifacts."""
    front_path = _existing_artifact_path(seed_directory, "pareto_front.csv")
    validation_path = seed_directory / "validation" / "validation_results.json"
    test_path = seed_directory / "test" / "test_results.json"
    if not (front_path.is_file() and validation_path.is_file() and test_path.is_file()):
        return None
    try:
        with front_path.open(encoding="utf-8-sig", newline="") as handle:
            source_rows = list(csv.DictReader(handle))
        validation_result = json.loads(validation_path.read_text(encoding="utf-8"))
        json.loads(test_path.read_text(encoding="utf-8"))
    except (OSError, csv.Error, json.JSONDecodeError):
        return None
    front_rows = []
    for rank, row in enumerate(source_rows, start=1):
        front_rows.append({
            "seed": int(seed), "pareto_rank": rank,
            "formula_id": row.get("formula_id"),
            "readable_expression": row.get("readable_expression"),
            "mean_r2": _mode4_float(row.get("mean_r2")),
            "r2_std": _mode4_float(row.get("r2_std")),
            "min_r2": _mode4_float(row.get("min_r2")),
            "pooled_r2": _mode4_float(row.get("pooled_r2")),
            "complexity": _mode4_float(row.get("complexity")),
            "output_directory": str(seed_directory),
        })
    if not front_rows:
        return None
    validation_front = validation_result.get("final_pareto_front", [])
    validation_r2 = [
        value for row in validation_front
        if (value := _mode4_float(row.get("validation_mean_r2"))) is not None
    ]
    history = validation_result.get("per_iteration", [])
    last_hv = history[-1].get("validation_hypervolume") if history else None
    valid_r2 = [row["mean_r2"] for row in front_rows if row["mean_r2"] is not None]
    valid_complexity = [
        row["complexity"] for row in front_rows if row["complexity"] is not None
    ]
    summary = {
        "seed": int(seed), "output_directory": str(seed_directory),
        "pareto_front_size": len(front_rows),
        "best_mean_r2": max(valid_r2) if valid_r2 else None,
        "minimum_complexity": min(valid_complexity) if valid_complexity else None,
        "final_hypervolume": last_hv,
        "best_validation_mean_r2": max(validation_r2) if validation_r2 else None,
        "pysr_best_validation_mean_r2": None,
        "pysr_best_test_mean_r2": None,
        "iterations": len(history), "run_completed": True,
    }
    for label, field in (("validation", "pysr_best_validation_mean_r2"),
                         ("test", "pysr_best_test_mean_r2")):
        for candidate in (
            seed_directory / "pysr_baseline" / "shared_refit" / f"{label}_results.json",
            seed_directory / "pysr_baseline" / f"{label}_results.json",
        ):
            if not candidate.is_file():
                continue
            try:
                result = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                break
            metric = "validation_mean_r2" if label == "validation" else "test_mean_r2"
            values = [
                value for row in result.get("final_pareto_front", [])
                if (value := _mode4_float(row.get(metric))) is not None
            ]
            summary[field] = max(values) if values else None
            break
    return {
        "status": "completed", "experiment_mode": int(experiment_mode),
        "seed": int(seed), "front_rows": front_rows,
        "summary": summary, "history": [],
        "legacy_artifacts": True,
    }


def _load_mode4_seed_completion(seed_directory: Path) -> dict | None:
    """Read a seed completion marker, returning None for incomplete seeds."""
    path = _mode4_seed_completion_path(seed_directory)
    if not path.is_file():
        return None
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if marker.get("status") != "completed":
        return None
    if not isinstance(marker.get("summary"), dict):
        return None
    if not isinstance(marker.get("front_rows"), list):
        return None
    return marker


def _write_mode4_seed_completion(
    seed_directory: Path, experiment_mode: int, seed: int,
    front_rows: list[dict], summary: dict, history: list[dict],
) -> None:
    """Persist the small, final per-seed state needed for experiment resume."""
    marker = {
        "status": "completed",
        "experiment_mode": int(experiment_mode),
        "seed": int(seed),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "front_rows": front_rows,
        "summary": summary,
        "history": history,
    }
    seed_directory.mkdir(parents=True, exist_ok=True)
    _mode4_seed_completion_path(seed_directory).write_text(
        json.dumps(_json_value(marker), ensure_ascii=False, indent=2,
                   allow_nan=False),
        encoding="utf-8",
    )


def _mode5_canonical_repeat_structure(expression: str) -> tuple[str, str, int] | None:
    """Canonicalize one fitted expression for cross-seed structure counting."""
    try:
        parsed = sp.sympify(str(expression))
    except (TypeError, ValueError, sp.SympifyError):
        return None
    parameter_names: dict[sp.Symbol, sp.Symbol] = {}
    parameter_index = 0

    def next_parameter() -> sp.Symbol:
        nonlocal parameter_index
        parameter_index += 1
        return sp.Symbol(f"C{parameter_index}")

    def normalize(node):
        if isinstance(node, sp.Symbol) and re.fullmatch(
            r"(?:a\d+|c\d+(?:_left|_right)?|p\d+|"
            r"cconst_product_\d+(?:_left|_right)?|"
            r"cconst_output_\d+(?:_left|_right)?)", node.name
        ):
            if node not in parameter_names:
                parameter_names[node] = next_parameter()
            return parameter_names[node]
        if isinstance(node, sp.Float):
            return next_parameter()
        if not node.args:
            return node
        return node.func(*(normalize(arg) for arg in node.args))

    canonical = normalize(parsed)
    return str(canonical), sp.srepr(canonical), int(formula_complexity(canonical))


def _refresh_mode5_repeated_structure_bubbles(
    experiment_dir: Path, data_mode: str, *, independent_segment_mode: bool = False,
) -> dict:
    """Rebuild cumulative repeated-structure tables and bubble charts after a seed."""
    import csv

    # Pareto rows can contain serialized fit histories well above csv's
    # default 128 KiB per-field limit, even though the plot only consumes a
    # handful of columns.  Raise the parser limit to the largest value the
    # platform's C long accepts; reduce it only when this platform overflows.
    field_limit = sys.maxsize
    while field_limit > 0:
        try:
            csv.field_size_limit(field_limit)
            break
        except OverflowError:  # pragma: no cover - platform-dependent C long
            field_limit //= 10
    if field_limit <= 0:  # pragma: no cover - defensive platform fallback
        raise RuntimeError("Could not set a valid field-size limit for the bubble-plot CSV parser")

    segment_names = ("segment_1", "segment_2") if independent_segment_mode else (None,)
    seed_directories = sorted(
        path for path in experiment_dir.glob("seed_*")
        if path.is_dir() and _load_mode4_seed_completion(path) is not None
    )
    output_metadata = {}
    for segment_name in segment_names:
        segment_root = experiment_dir / segment_name if segment_name else experiment_dir
        formula_summary_dir = segment_root / "cross_seed_formula_summary"
        formula_summary_dir.mkdir(parents=True, exist_ok=True)
        model_specs = {
            "main_model": (Path("test") / "test_pareto_front.csv", "#2878B5", "o", "Main model"),
            "pysr": (Path("pysr_baseline") / "test_pareto_front.csv", "#E07A28", "^", "PySR"),
        }
        model_records: dict[str, dict[str, list[dict]]] = {}
        for model_name, (relative_path, _color, _marker, _label) in model_specs.items():
            records_by_structure: dict[str, list[dict]] = {}
            for seed_dir in seed_directories:
                source_root = (seed_dir / segment_name) if segment_name else seed_dir
                source_path = source_root / relative_path
                if not source_path.is_file():
                    continue
                seed_label = seed_dir.name
                try:
                    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
                        rows = list(csv.DictReader(handle))
                except OSError:
                    logging.exception("Failed to read duplicate-structure data: %s", source_path)
                    continue
                except csv.Error:
                    logging.exception("Invalid CSV format; skipping potentially incomplete duplicate-structure plot: %s",
                                      source_path)
                    raise
                for row_index, row in enumerate(rows, start=1):
                    normalized = _mode5_canonical_repeat_structure(
                        row.get("readable_expression", "")
                    )
                    if normalized is None:
                        continue
                    readable, structure_key, complexity = normalized
                    occurrence = {
                        "seed": seed_label,
                        "formula_id": row.get("formula_id"),
                        "row_index": row_index,
                        "readable_expression": row.get("readable_expression"),
                        "normalized_readable_expression": readable,
                        "test_mean_r2": (
                            row.get("test_pooled_r2", row.get("test_mean_r2"))
                            if _is_measured_pathloss_profile(data_mode) else row.get("test_mean_r2")
                        ),
                        "validation_mean_r2": (
                            row.get("validation_pooled_r2", row.get("validation_mean_r2"))
                            if _is_measured_pathloss_profile(data_mode) else row.get("validation_mean_r2")
                        ),
                        "test_group_r2_std": row.get("test_group_r2_std", row.get("test_r2_std")),
                        "complexity": row.get("complexity"),
                    }
                    records_by_structure.setdefault(structure_key, []).append(occurrence)
            model_records[model_name] = records_by_structure

        summary = {
            "experiment_directory": str(experiment_dir),
            "data_mode": data_mode,
            "segment": segment_name,
            "completed_seed_count": len(seed_directories),
            "outputs": {}, "plots": {},
            "rule": {
                "source_relative_path": "test/test_pareto_front.csv",
                "repeat_definition": "same canonical structure occurs in at least two distinct seeds",
                "canonicalization": "SymPy canonical tree; alpha-rename fitted a/c/p parameters, cconst_product/cconst_output placeholders, and floating literals; preserve integer literals",
                "bubble_area_rule": "area = 72 points^2 * distinct seed frequency",
            },
        }
        for model_name, (relative_path, color, marker, label) in model_specs.items():
            records = model_records[model_name]
            output_rows = []
            for structure_key, occurrences in records.items():
                by_seed: dict[str, list[dict]] = {}
                for occurrence in occurrences:
                    by_seed.setdefault(occurrence["seed"], []).append(occurrence)
                if len(by_seed) < 2:
                    continue
                seed_means = {}
                for seed_label, seed_occurrences in by_seed.items():
                    seed_means[seed_label] = {}
                    for field in ("test_mean_r2", "validation_mean_r2", "test_group_r2_std", "complexity"):
                        values = []
                        for occurrence in seed_occurrences:
                            raw = occurrence.get(field)
                            try:
                                value = float(raw)
                            except (TypeError, ValueError):
                                continue
                            if math.isfinite(value):
                                values.append(value)
                        if values:
                            seed_means[seed_label][field] = float(np.mean(values))
                def across_seeds(field):
                    values = [values[field] for values in seed_means.values()
                              if field in values]
                    return float(np.mean(values)) if values else None
                first_occurrence = occurrences[0]
                output_rows.append({
                    "readable_expression": first_occurrence["normalized_readable_expression"],
                    "canonical_structure": structure_key,
                    "seed_count": len(by_seed),
                    "seeds": ";".join(sorted(by_seed)),
                    "occurrence_count": len(occurrences),
                    "mean_test_r2": across_seeds("test_mean_r2"),
                    "mean_validation_r2": across_seeds("validation_mean_r2"),
                    "mean_test_r2_std": across_seeds("test_group_r2_std"),
                    **({"mean_test_group_r2_std": across_seeds("test_group_r2_std")}
                       if _is_measured_pathloss_profile(data_mode) else {}),
                    "mean_complexity": across_seeds("complexity"),
                    "canonical_complexity": _mode5_canonical_repeat_structure(
                        first_occurrence["normalized_readable_expression"]
                    )[2],
                    "occurrences_json": json.dumps(occurrences, ensure_ascii=False),
                })
            output_rows.sort(key=lambda row: (
                -(row["mean_test_r2"] if row["mean_test_r2"] is not None else -math.inf),
                row["mean_complexity"] if row["mean_complexity"] is not None else math.inf,
            ))
            fields = ["readable_expression", "canonical_structure", "seed_count", "seeds",
                      "occurrence_count", "mean_test_r2", "mean_validation_r2",
                      "mean_test_r2_std", "mean_complexity", "canonical_complexity",
                      "occurrences_json"]
            if _is_measured_pathloss_profile(data_mode):
                fields.insert(fields.index("occurrences_json"), "mean_test_group_r2_std")
            csv_path = formula_summary_dir / f"{model_name}_repeated_structures.csv"
            with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(output_rows)
            summary["outputs"][model_name] = {
                "source_relative_path": str(relative_path),
                "repeated_structure_count": len(output_rows),
                "csv": str(csv_path),
            }

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            for metric_key, metric_field, metric_label, suffix in (
                ("test", "mean_test_r2", "Mean test $R^2$", "test"),
                ("validation", "mean_validation_r2", "Mean validation $R^2$", "validation"),
            ):
                figure, axis = plt.subplots(figsize=(9, 6))
                for model_name, (_path, color, marker, label) in model_specs.items():
                    plot_rows = []
                    with (formula_summary_dir / f"{model_name}_repeated_structures.csv").open(
                        "r", encoding="utf-8-sig", newline=""
                    ) as handle:
                        plot_rows = list(csv.DictReader(handle))
                    valid_rows = []
                    for row in plot_rows:
                        try:
                            x_value = float(row["mean_complexity"])
                            y_value = float(row[metric_field])
                            size = 72.0 * int(row["seed_count"])
                        except (TypeError, ValueError, KeyError):
                            continue
                        if math.isfinite(x_value) and math.isfinite(y_value):
                            valid_rows.append((x_value, y_value, size))
                    if valid_rows:
                        axis.scatter(
                            [item[0] for item in valid_rows],
                            [item[1] for item in valid_rows],
                            s=[item[2] for item in valid_rows], color=color,
                            marker=marker, alpha=0.72, edgecolors="black",
                            linewidths=0.5, label=label,
                        )
                axis.set_xlabel("Canonical formula complexity (lower is better)")
                axis.set_ylabel(metric_label)
                title = f"Repeated structures — {metric_key}"
                if segment_name:
                    title = f"{segment_name}: {title}"
                axis.set_title(title)
                axis.grid(True, alpha=0.25)
                if axis.has_data():
                    axis.legend()
                figure.tight_layout()
                stem = f"repeated_structures_{suffix}_bubble_scatter"
                png_path, pdf_path = formula_summary_dir / f"{stem}.png", formula_summary_dir / f"{stem}.pdf"
                figure.savefig(png_path, dpi=220)
                figure.savefig(pdf_path)
                if metric_key == "test":
                    figure.savefig(formula_summary_dir / "repeated_structures_bubble_scatter.png",
                                  dpi=220)
                    figure.savefig(formula_summary_dir / "repeated_structures_bubble_scatter.pdf")
                plt.close(figure)
                summary["plots"][metric_key] = {
                    "png": str(png_path), "pdf": str(pdf_path),
                    "metric_field": metric_field, "metric_label": metric_label,
                    "bubble_area_rule": "area = 72 points^2 * distinct seed frequency",
                    "model_styles": {
                        name: {"label": spec[3], "color": spec[1], "marker": spec[2]}
                        for name, spec in model_specs.items()
                    },
                }
                if metric_key == "test":
                    summary["plots"]["combined"] = {
                        "png": str(formula_summary_dir / "repeated_structures_bubble_scatter.png"),
                        "pdf": str(formula_summary_dir / "repeated_structures_bubble_scatter.pdf"),
                        "metric_field": metric_field,
                        "metric_label": metric_label,
                        "alias_of": "test",
                    }
        except ImportError:
            summary["plotting"] = {"enabled": False, "reason": "matplotlib_not_installed"}
        summary_path = formula_summary_dir / "summary.json"
        summary_path.write_text(
            json.dumps(_json_value(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        output_metadata[segment_name or "all"] = summary
    return output_metadata


def _mode4_seed_is_complete(
    seed_directory: Path, experiment_mode: int, seed: int,
) -> bool:
    marker = _load_mode4_seed_completion(seed_directory)
    if marker is None:
        marker = _load_legacy_mode4_seed_completion(
            seed_directory, experiment_mode, seed
        )
    return bool(
        marker
        and int(marker.get("experiment_mode", -1)) == int(experiment_mode)
        and int(marker.get("seed", -1)) == int(seed)
    )


def _mode4_summary_rows(payload: dict, seed: int, output_dir: Path) -> tuple[list[dict], dict]:
    """Extract publication-friendly Pareto and scalar metrics for one seed."""
    independent_segments = payload.get("metadata", {}).get(
        "independent_segment_runs"
    )
    if isinstance(independent_segments, dict):
        front_rows = []
        for segment_name, segment_summary in independent_segments.items():
            for rank, source in enumerate(segment_summary.get("front_rows", []), start=1):
                front_rows.append({
                    **source, "seed": int(seed), "segment_name": segment_name,
                    "pareto_rank": rank,
                    "output_directory": str(output_dir / segment_name),
                })
        summary = {
            "seed": int(seed), "output_directory": str(output_dir),
            "run_completed": all(
                bool(item.get("run_completed"))
                for item in independent_segments.values()
            ),
            "pareto_front_size": len(front_rows),
            "segment_summaries": {
                name: item.get("summary", {})
                for name, item in independent_segments.items()
            },
            # A scalar or a cross-segment HV is deliberately not constructed.
            "best_mean_r2": None, "best_validation_mean_r2": None,
            "pysr_best_validation_mean_r2": None,
            "pysr_best_test_mean_r2": None, "final_hypervolume": None,
            "minimum_complexity": None,
            "iterations": max((int(item.get("summary", {}).get("iterations", 0))
                                for item in independent_segments.values()), default=0),
        }
        return front_rows, summary

    front_rows = []
    for rank, row in enumerate(payload.get("gp_pareto_front", []), start=1):
        front_rows.append({
            "seed": int(seed), "pareto_rank": rank,
            "formula_id": row.get("formula_id"),
            "readable_expression": row.get("readable_expression"),
            "mean_r2": row.get("mean_r2"), "r2_std": row.get("r2_std"),
            "min_r2": row.get("min_r2"), "pooled_r2": row.get("pooled_r2"),
            "segment_prediction_difference": row.get(
                "segment_prediction_difference"
            ),
            "predicted_segment_difference": row.get(
                "predicted_segment_difference"
            ),
            "complexity": row.get("complexity"),
            "output_directory": str(output_dir),
        })
    valid_r2 = [float(row["mean_r2"]) for row in front_rows
                if row.get("mean_r2") is not None]
    valid_complexity = [float(row["complexity"]) for row in front_rows
                        if row.get("complexity") is not None]
    validation_result = {}
    validation_path = output_dir / "validation" / "validation_results.json"
    if validation_path.is_file():
        try:
            validation_result = json.loads(validation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            validation_result = {}
    else:
        front_path = output_dir / "validation" / "validation_pareto_front.csv"
        hv_path = output_dir / "validation" / "hypervolume" / "validation_hypervolume_by_iteration.csv"
        try:
            with front_path.open(encoding="utf-8-sig", newline="") as handle:
                validation_result["final_pareto_front"] = list(csv.DictReader(handle))
        except (OSError, csv.Error):
            pass
        try:
            with hv_path.open(encoding="utf-8-sig", newline="") as handle:
                validation_result["per_iteration"] = list(csv.DictReader(handle))
        except (OSError, csv.Error):
            pass
    validation_front = validation_result.get("final_pareto_front", [])
    validation_r2 = [float(row["validation_mean_r2"])
                     for row in validation_front
                     if row.get("validation_mean_r2") is not None]
    history = validation_result.get("per_iteration", [])
    last_hv = (history[-1].get("validation_hypervolume")
               if history else None)
    # Mode 4 runs the same sealed-split PySR baseline as Mode 3 for every
    # seed.  Keep its scalar metrics in the paper-level seed summary so the
    # six-seed comparison is visible without opening each seed directory.
    pysr_validation = {}
    pysr_test = {}
    for label, target in (("validation", pysr_validation), ("test", pysr_test)):
        paths = [
            output_dir / "pysr_baseline" / "shared_refit" / f"{label}_results.json",
            # Backward compatibility for experiments generated before the
            # shared-refit result was made canonical.
            output_dir / "pysr_baseline" / f"{label}_results.json",
        ]
        for path in paths:
            if not path.is_file():
                continue
            try:
                target.update(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass
            break
        if not target:
            loaded_result = _load_pysr_evaluation_result(output_dir, label)
            if loaded_result:
                target.update(loaded_result)
        if label == "test" and not target:
            front_paths = [
                output_dir / "pysr_baseline" / "shared_refit" / "test_pareto_front.csv",
                output_dir / "pysr_baseline" / "test_pareto_front.csv",
            ]
            for front_path in front_paths:
                if not front_path.is_file():
                    continue
                try:
                    with front_path.open(encoding="utf-8-sig", newline="") as handle:
                        target["final_pareto_front"] = list(csv.DictReader(handle))
                except (OSError, csv.Error):
                    pass
                break
    pysr_validation_front = pysr_validation.get("final_pareto_front", [])
    pysr_test_front = pysr_test.get("final_pareto_front", [])
    pysr_validation_r2 = [float(row["validation_mean_r2"])
                          for row in pysr_validation_front
                          if row.get("validation_mean_r2") is not None]
    pysr_test_r2 = [float(row["test_mean_r2"])
                    for row in pysr_test_front
                    if row.get("test_mean_r2") is not None]
    summary = {
        "seed": int(seed), "output_directory": str(output_dir),
        "pareto_front_size": len(front_rows),
        "best_mean_r2": max(valid_r2) if valid_r2 else None,
        "minimum_complexity": min(valid_complexity) if valid_complexity else None,
        "final_hypervolume": last_hv,
        "best_validation_mean_r2": max(validation_r2) if validation_r2 else None,
        "pysr_best_validation_mean_r2": (
            max(pysr_validation_r2) if pysr_validation_r2 else None
        ),
        "pysr_best_test_mean_r2": (
            max(pysr_test_r2) if pysr_test_r2 else None
        ),
        "iterations": payload.get("metadata", {}).get("mode2_iterations", 0),
        "run_completed": payload.get("metadata", {}).get("run_completed", False),
    }
    return front_rows, summary


def _run_mode5_independent_segment_seed(
    seed_config_path: Path, seed_directory: Path, *, n_jobs: int | None,
    serial_lm: bool,
) -> dict:
    """Run two completely isolated Mode 5 searches split at the physical prior."""
    import csv

    seed_config = json.loads(seed_config_path.read_text(encoding="utf-8"))
    data_config = seed_config.get("data", {})
    realdata_config = data_config.get("realdata", {})
    segmentation = realdata_config.get("segmentation", {}) or {}
    prior = _breakpoint_prior(segmentation)
    if prior is None:
        raise ValueError(
            "Independent Mode 5 on realdata_mat requires segmentation.breakpoint_prior to be enabled"
        )
    X, y, feature_names, _groups = load_configured_data(data_config)
    breakpoint = float(prior["breakpoint_value"])
    masks = (X[:, 0] <= breakpoint, X[:, 0] > breakpoint)
    segment_payloads = {}
    segment_summaries = {}
    for index, (segment_name, mask) in enumerate(
            zip(("segment_1", "segment_2"), masks)):
        X_segment, y_segment = X[mask], y[mask]
        if len(y_segment) < 3:
            raise ValueError(f"Independent run for {segment_name} requires at least 3 valid samples")
        segment_directory = seed_directory / segment_name
        segment_directory.mkdir(parents=True, exist_ok=True)
        csv_path = segment_directory / "segment_data.csv"
        config_path = segment_directory / "seed_config.json"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow((feature_names[0], "path_loss"))
            writer.writerows(
                (format(float(x), ".17g"), format(float(target), ".17g"))
                for x, target in zip(X_segment[:, 0], y_segment)
            )
        child_config = copy.deepcopy(seed_config)
        child_data = child_config["data"]
        child_realdata = child_data["realdata"]
        child_realdata["segmentation"] = {**segmentation, "enabled": False}
        child_realdata["independent_segment_run"] = True
        child_data["mode"] = "csv"
        child_data["csv_path"] = str(csv_path)
        child_data["features"] = [feature_names[0]]
        child_data["target"] = "path_loss"
        child_data.pop("csv_files", None)
        child_data["split"] = dict(child_data.get("split", {}))
        child_data["split"]["seed"] = int(seed_config.get("seed", 0)) + index
        child_config["output"] = {
            **child_config.get("output", {}),
            "directory": str(segment_directory),
        }
        config_path.write_text(
            json.dumps(_json_value(child_config), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        payload = run(
            config_path, n_jobs=(1 if serial_lm else n_jobs),
            pipeline_mode=3, _allow_single_mode4=True, validation_enabled=True,
        )
        # Preserve disjoint segment artifacts and make their identity explicit
        # in the parent seed summary.  No front or score is merged for search.
        tagged_front = []
        for row in payload.get("gp_pareto_front", []):
            tagged = dict(row)
            tagged["segment_name"] = segment_name
            tagged_front.append(tagged)
        payload["gp_pareto_front"] = tagged_front
        payload["metadata"]["independent_segment_name"] = segment_name
        front_rows, summary = _mode4_summary_rows(
            payload, int(seed_config.get("seed", 0)), segment_directory
        )
        for row in front_rows:
            row["segment_name"] = segment_name
        segment_payloads[segment_name] = payload
        segment_summaries[segment_name] = {
            "front_rows": front_rows, "summary": summary,
            "output_directory": str(segment_directory),
            "run_completed": bool(payload.get("metadata", {}).get("run_completed")),
        }

    combined_front = [
        row for item in segment_summaries.values()
        for row in item["front_rows"]
    ]
    return {
        "metadata": {
            "run_status": "completed", "run_completed": True,
            "mode5_method": "independent_segment_searches",
            "independent_segment_runs": segment_summaries,
            "segment_boundary": breakpoint,
            "segment_sample_counts": {
                name: int(np.sum(mask))
                for name, mask in zip(("segment_1", "segment_2"), masks)
            },
            "segment_features": list(feature_names),
            "cross_segment_selection": False,
        },
        "gp_pareto_front": combined_front,
        "mode2_hypervolume_history": [],
    }


def _run_mode5_shared_structure_seed(
    seed_config_path: Path, seed_directory: Path, *, n_jobs: int | None,
    serial_lm: bool,
) -> dict:
    """Run Mode 5 with one symbolic structure and segment-specific constants."""
    seed_config = json.loads(seed_config_path.read_text(encoding="utf-8"))
    data_config = seed_config.get("data", {})
    realdata_config = data_config.get("realdata", {})
    segmentation = realdata_config.get("segmentation", {}) or {}
    if not bool(segmentation.get("enabled", True)):
        raise ValueError("Shared-structure Mode 5 requires realdata.segmentation to be enabled")
    if _breakpoint_prior(segmentation) is None:
        raise ValueError(
            "Shared-structure Mode 5 requires segmentation.breakpoint_prior to be enabled "
            "so both methods use the same breakpoint"
        )

    child_config = copy.deepcopy(seed_config)
    child_config["output"] = {
        **child_config.get("output", {}),
        "directory": str(seed_directory),
    }
    child_realdata = child_config["data"]["realdata"]
    child_realdata["segmentation"] = {
        **segmentation,
        "independent_segment_fitting": False,
        "formula_pair_search": False,
    }
    child_realdata.pop("independent_segment_run", None)
    fitting = child_config.setdefault("fitting", {})
    # fit_formula_per_dataset is the shared-structure multi-dataset path:
    # both segments use the same expression, while parameters are stored per
    # segment.  pooled R² is the primary real-data quality objective.
    fitting["joint_r2_aggregation"] = "pooled"
    # Use the fitted prediction jump as the continuity objective.
    child_config["mode5_shared_structure_run"] = True
    child_config_path = seed_directory / "seed_config.json"
    child_config_path.parent.mkdir(parents=True, exist_ok=True)
    child_config_path.write_text(
        json.dumps(_json_value(child_config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    payload = run(
        child_config_path, n_jobs=(1 if serial_lm else n_jobs),
        pipeline_mode=3, _allow_single_mode4=True, validation_enabled=True,
    )
    payload.setdefault("metadata", {})["mode5_method"] = "shared_structure_different_parameters"
    payload["metadata"]["joint_r2_aggregation"] = "pooled"
    payload["metadata"]["continuity_objective"] = "predicted_segment_difference"
    payload["metadata"]["continuity_metric_definition"] = (
        "abs(prediction_right_formula_at_breakpoint - "
        "prediction_left_formula_at_breakpoint)"
    )
    payload["metadata"]["formula_structure_shared"] = True
    payload["metadata"]["segment_parameters_shared"] = False
    return payload


def _mode4_unified_hypervolume_records(
    seed_directories: Sequence[Path], evaluation_label: str,
    data_mode: str = "csv",
) -> tuple[list[dict], dict[str, tuple[float, float]], list[str]]:
    """Recalculate one experiment's HV with one scale shared by all seeds.

    Bounds are determined from the tree/random validation/test histories and
    the final windowed PySR fronts across all available seeds.  The resulting
    common bounds are then applied to every saved iteration and method, and
    are written back to the per-seed result/comparison artifacts.
    """
    fields, maximize_fields = _validation_hv_objectives(
        data_mode, evaluation_label
    )
    # Pareto CSV rows can contain serialized per-dataset fit histories far
    # larger than csv's conservative default field-size limit.
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    histories: dict[tuple[Path, str], list[dict]] = {}
    pysr_fronts: dict[tuple[Path, str], list[dict]] = {}
    pysr_histories: dict[tuple[Path, str], list[dict]] = {}
    final_rows = []
    source_files = []
    for seed_directory in seed_directories:
        for method in ("tree_model", "random_permutation"):
            branch = (seed_directory if method == "tree_model"
                      else seed_directory / "mode2_random_permutation_ablation")
            history_path = branch / evaluation_label / (
                f"{evaluation_label}_history.json"
            )
            compact_front_path = branch / evaluation_label / "hypervolume" / (
                f"{evaluation_label}_hypervolume_fronts_by_iteration.csv"
            )
            if not history_path.is_file():
                curve_path = branch / evaluation_label / "hypervolume" / (
                    f"{evaluation_label}_hypervolume_by_iteration.csv"
                )
                front_path = branch / evaluation_label / (
                    f"{evaluation_label}_pareto_front.csv"
                )
                try:
                    with compact_front_path.open(
                            encoding="utf-8-sig", newline="") as handle:
                        compact_front_rows = list(csv.DictReader(handle))
                except (OSError, csv.Error):
                    compact_front_rows = []
                try:
                    with curve_path.open(encoding="utf-8-sig", newline="") as handle:
                        curve = list(csv.DictReader(handle))
                except (OSError, csv.Error):
                    curve = []
                try:
                    with front_path.open(encoding="utf-8-sig", newline="") as handle:
                        final_rows.extend(list(csv.DictReader(handle)))
                except (OSError, csv.Error):
                    pass
                if compact_front_rows:
                    compact_history = {}
                    for row in compact_front_rows:
                        iteration = int(row.get("iteration", 0))
                        entry = compact_history.setdefault(
                            iteration, {"iteration": iteration, "front": []}
                        )
                        front_row = {}
                        for field in fields:
                            value = _mode4_float(row.get(field))
                            if value is not None:
                                front_row[field] = value
                        if len(front_row) == len(fields):
                            entry["front"].append(front_row)
                    history = [compact_history[key]
                               for key in sorted(compact_history)]
                else:
                    history = [{
                        "iteration": int(row.get("iteration", 0)),
                        "front": [],
                        "_precomputed_hv": row.get(
                            f"{evaluation_label}_hypervolume"
                        ),
                        "_precomputed_size": row.get(
                            f"{evaluation_label}_pareto_front_size"
                        ),
                    } for row in curve]
                if not history:
                    continue
            else:
                try:
                    payload = json.loads(history_path.read_text(encoding="utf-8"))
                    history = payload.get("history", [])
                except (OSError, json.JSONDecodeError):
                    continue
            if not history:
                continue
            histories[(seed_directory, method)] = history
            source_files.append(str(
                history_path if history_path.is_file() else
                compact_front_path if compact_front_path.is_file() else curve_path
            ))
            for history_item in history:
                final_rows.extend(history_item.get("front", []))
        pysr_path = seed_directory / "pysr_baseline" / "shared_refit" / (
            f"{evaluation_label}_results.json"
        )
        if not pysr_path.is_file():
            pysr_path = seed_directory / "pysr_baseline" / (
                f"{evaluation_label}_results.json"
            )
        if pysr_path.is_file():
            try:
                pysr_payload = json.loads(pysr_path.read_text(encoding="utf-8"))
                pysr_front = pysr_payload.get("final_pareto_front", [])
                pysr_history = pysr_payload.get("history", []) or []
            except (OSError, json.JSONDecodeError):
                pysr_front = []
                pysr_history = []
            if pysr_front:
                pysr_front = _pysr_front_for_hv_reporting(
                    pysr_front, data_mode, evaluation_label
                )
                if not pysr_front:
                    continue
                pysr_fronts[(seed_directory, "pysr_baseline")] = pysr_front
                # Keep only the final PySR front for optional reporting.  It
                # is deliberately excluded from the shared tree/random HV
                # normalization bounds.
                source_files.append(str(pysr_path))
        else:
            pysr_front_path = pysr_path.with_name(
                f"{evaluation_label}_pareto_front.csv"
            )
            if not pysr_front_path.is_file():
                pysr_front_path = seed_directory / "pysr_baseline" / (
                    f"{evaluation_label}_pareto_front.csv"
                )
            try:
                with pysr_front_path.open(encoding="utf-8-sig", newline="") as handle:
                    pysr_front = list(csv.DictReader(handle))
            except (OSError, csv.Error):
                pysr_front = []
            if pysr_front:
                pysr_front = _pysr_front_for_hv_reporting(
                    pysr_front, data_mode, evaluation_label
                )
                if pysr_front:
                    pysr_fronts[(seed_directory, "pysr_baseline")] = pysr_front
                    source_files.append(str(pysr_front_path))
    if not final_rows:
        return [], {}, source_files

    bounds = _validation_hv_bounds(final_rows, fields)
    records = []
    for (seed_directory, method), history in histories.items():
        method_rows = []
        for item in history:
            front = item.get("front", [])
            value = (
                _mode4_float(item.get("_precomputed_hv"))
                if item.get("_precomputed_hv") is not None
                else _calculate_validation_hv(front, fields, maximize_fields, bounds)
            )
            method_rows.append({
                "method": method,
                "seed": seed_directory.name,
                "iteration": int(item.get("iteration", 0)),
                f"{evaluation_label}_hypervolume": value,
                f"{evaluation_label}_pareto_front_size": (
                    int(item.get("_precomputed_size"))
                    if item.get("_precomputed_size") is not None else len(front)
                ),
            })
            records.append({
                "seed_directory": seed_directory,
                "method": method,
                **method_rows[-1],
            })

        # Keep the per-seed artifacts consistent with the experiment-level
        # normalization.  This changes only HV fields and plotting metadata.
        branch = (seed_directory if method == "tree_model"
                  else seed_directory / "mode2_random_permutation_ablation")
        results_path = branch / evaluation_label / (
            f"{evaluation_label}_results.json"
        )
        if results_path.is_file():
            try:
                result = json.loads(results_path.read_text(encoding="utf-8"))
                result["per_iteration"] = method_rows
                result.setdefault("metadata", {})["normalization_bounds"] = {
                    field: {"min": low, "max": high}
                    for field, (low, high) in bounds.items()
                }
                result["metadata"]["normalization_scope"] = (
                    "mode4_experiment_all_iterations_all_seeds_tree_and_random_methods"
                )
                results_path.write_text(
                    json.dumps(_json_value(result), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except (OSError, json.JSONDecodeError):
                pass
        _write_csv(
            branch / evaluation_label / "hypervolume" / (
                f"{evaluation_label}_hypervolume_by_iteration.csv"
            ),
            method_rows,
        )
        # The standalone per-seed curve must use the same experiment-wide
        # normalization as the cross-seed comparison, not the provisional
        # single-seed scale produced when validation first ran.
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            figure, axis = plt.subplots(figsize=(8, 5.5))
            values = sorted(method_rows, key=lambda row: row["iteration"])
            if values:
                axis.plot(
                    [row["iteration"] for row in values],
                    [row[f"{evaluation_label}_hypervolume"] for row in values],
                    marker="o", ms=4.5, linewidth=2.2,
                    color="tab:blue", label="Tree model",
                )
            axis.set_xlabel("Iteration", fontsize=15)
            axis.set_ylabel(f"{evaluation_label.capitalize()} hypervolume", fontsize=15)
            axis.tick_params(axis="both", labelsize=13)
            axis.grid(True, alpha=0.3)
            if values:
                axis.legend(fontsize=13, loc="upper left", framealpha=1.0)
            figure.tight_layout()
            figure.savefig(
                branch / evaluation_label / "hypervolume" / (
                    f"{evaluation_label}_hypervolume_by_iteration.png"
                ), dpi=220,
            )
            plt.close(figure)
        except ImportError:
            pass

    iterative_iterations = [
        int(row["iteration"])
        for row in records
        if row["method"] in {"tree_model", "random_permutation"}
    ]
    start_iteration = min(iterative_iterations) if iterative_iterations else 0
    end_iteration = max(iterative_iterations) if iterative_iterations else start_iteration
    for (seed_directory, method), front in pysr_fronts.items():
        value = _calculate_validation_hv(front, fields, maximize_fields, bounds)
        records.append({
            "seed_directory": seed_directory,
            "method": method,
            "seed": seed_directory.name,
            "iteration": end_iteration,
            f"{evaluation_label}_hypervolume": value,
            f"{evaluation_label}_pareto_front_size": len(front),
            "pysr_point_type": (
                "final_validation_selected_result" if evaluation_label == "test"
                else "final_r2_window_filtered_result"
            ),
        })

    # Regenerate the original per-seed tree/random comparison artifacts with
    # exactly the same labels and filenames as the normal pipeline.
    by_seed = {}
    for row in records:
        by_seed.setdefault(row["seed_directory"], []).append(row)
    for seed_directory, seed_rows in by_seed.items():
        comparison_rows = [
            {key: value for key, value in row.items()
             if key not in {"seed_directory"}}
            for row in seed_rows
        ]
        comparison_dir = seed_directory / evaluation_label / "hypervolume"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = comparison_dir / f"{evaluation_label}_comparison.json"
        comparison = {}
        if comparison_path.is_file():
            try:
                comparison = json.loads(
                    comparison_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                comparison = {}
        comparison["per_iteration"] = comparison_rows
        has_pysr_point = any(
            row.get("method") == "pysr_baseline" for row in comparison_rows
        )
        comparison["pysr_included"] = bool(has_pysr_point)
        comparison["pysr_curve_type"] = (
            ("final_validation_selected_point" if evaluation_label == "test"
             else "final_r2_window_filtered_point")
            if has_pysr_point else None
        )
        comparison["normalization_bounds"] = {
            field: {"min": low, "max": high}
            for field, (low, high) in bounds.items()
        }
        comparison["normalization_scope"] = (
            "mode4_experiment_all_iterations_all_seeds_tree_and_random_methods"
        )
        comparison_path.write_text(
            json.dumps(_json_value(comparison), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_csv(comparison_dir / f"{evaluation_label}_hypervolume_by_iteration.csv", comparison_rows)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            figure, axis = plt.subplots(figsize=(8, 5.5))
            styles = {"tree_model": ("o", "Tree model"),
                      "random_permutation": ("^", "Random permutation ablation")}
            for method, (marker, label) in styles.items():
                values = sorted(
                    (row for row in comparison_rows if row["method"] == method),
                    key=lambda row: row["iteration"],
                )
                if values:
                    axis.plot(
                        [row["iteration"] for row in values],
                        [row[f"{evaluation_label}_hypervolume"] for row in values],
                        marker=marker, markersize=9.4, linewidth=1.75, label=label,
                    )
            axis.set_xlabel("Iteration", fontsize=15)
            axis.set_ylabel("Validation hypervolume", fontsize=15)
            axis.tick_params(axis="both", labelsize=13)
            axis.grid(True, alpha=0.3)
            axis.legend(fontsize=13, loc="upper left", framealpha=1.0)
            figure.tight_layout()
            figure.savefig(comparison_dir / f"{evaluation_label}_hypervolume_by_iteration.png", dpi=220)
            plt.close(figure)
        except ImportError:
            pass
    return records, bounds, source_files


def _mode4_average_tree_random_hypervolume(
    seed_directories: Sequence[Path], output_dir: Path,
    evaluation_label: str = "validation", data_mode: str = "csv",
) -> dict:
    """Average unified tree/random HV by iteration across the experiment seeds."""
    records, bounds, source_files = _mode4_unified_hypervolume_records(
        seed_directories, evaluation_label, data_mode=data_mode
    )
    objective_fields, maximize_fields = _validation_hv_objectives(
        data_mode, "validation" if evaluation_label == "validation" else "test"
    )
    reference_parameters = _write_hv_reference_parameters(
        output_dir,
        f"tree_vs_random_{evaluation_label}_hv_reference_parameters",
        objective_fields, maximize_fields, bounds, evaluation_label,
        "mode4_experiment_all_iterations_all_seeds_tree_and_random_methods",
    )
    values: dict[str, dict[int, list[float]]] = {
        "tree_model": {}, "random_permutation": {}, "pysr_baseline": {}
    }
    for record in records:
        method = record["method"]
        iteration = record["iteration"]
        hypervolume = float(record[f"{evaluation_label}_hypervolume"])
        if math.isfinite(hypervolume):
            values[method].setdefault(iteration, []).append(hypervolume)

    rows = []
    for iteration in sorted({
        iteration
        for method_values in values.values()
        for iteration in method_values
    }):
        row = {"iteration": iteration}
        for method in ("tree_model", "random_permutation", "pysr_baseline"):
            samples = values[method].get(iteration, [])
            row[f"{method}_hypervolume"] = (
                float(np.mean(samples)) if samples else None
            )
            row[f"{method}_valid_seed_count"] = len(samples)
        rows.append(row)

    hypervolume_dir = output_dir / "hypervolume"
    hypervolume_dir.mkdir(parents=True, exist_ok=True)
    csv_path = hypervolume_dir / (
        f"tree_vs_random_{evaluation_label}_hypervolume_by_iteration.csv"
    )
    _write_csv(csv_path, rows)
    plot_path = hypervolume_dir / (
        f"tree_vs_random_{evaluation_label}_hypervolume_by_iteration.png"
    )
    _mode4_plot_hypervolume_rows(rows, evaluation_label, plot_path)

    return {
        "enabled": bool(rows),
        "pysr_visual_comparison_enabled": False,
        "pysr_visualization_type": None,
        "evaluation_label": evaluation_label,
        "source_seed_count": len(source_files),
        "source_comparison_files": source_files,
        "normalization_scope": (
            "mode4_experiment_all_iterations_all_seeds_tree_and_random_methods"
        ),
        "normalization_bounds": {
            field: {"min": low, "max": high}
            for field, (low, high) in bounds.items()
        },
        "zero_span_objectives": [
            field for field, (low, high) in bounds.items() if high <= low
        ],
        "hv_reference_parameters": reference_parameters,
        "csv": str(csv_path),
        "plot": str(plot_path),
    }


def _mode4_plot_hypervolume_rows(
    rows: Sequence[dict], evaluation_label: str, plot_path: Path,
) -> None:
    """Render a one-panel tree/random curve in the paper figure style."""
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5.5))
    styles = {
        "tree_model": ("o", "#1f77b4", "Tree model"),
        "random_permutation": (
            "^", "#ff7f0e", "Random permutation ablation"
        ),
    }
    has_series = False
    for method, (marker, color, label) in styles.items():
        values = []
        for row in rows:
            try:
                value = float(row.get(f"{method}_hypervolume"))
                iteration = float(row["iteration"])
            except (TypeError, ValueError, KeyError):
                continue
            if math.isfinite(value) and math.isfinite(iteration):
                values.append((iteration, value))
        values.sort(key=lambda item: item[0])
        if values:
            has_series = True
            axis.plot(
                [item[0] for item in values], [item[1] for item in values],
                marker=marker, markersize=9.4, linewidth=1.75,
                color=color, label=label,
            )
    axis.set_xlabel("Iteration", fontsize=15)
    axis.set_ylabel(f"{evaluation_label.capitalize()} hypervolume", fontsize=15)
    axis.tick_params(axis="both", labelsize=13)
    axis.grid(True, alpha=0.3)
    if has_series:
        axis.legend(fontsize=13, loc="upper left", framealpha=1.0)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=220)
    plt.close(figure)


def _refresh_mode4_average_hypervolume_artifacts(
    seed_directories: Sequence[Path], output_dir: Path, data_mode: str,
) -> tuple[dict, dict]:
    """Refresh aggregate HV artifacts from every seed available so far.

    The final call uses all seeds and therefore produces exactly the same
    filenames and calculation as before.  Calling it after each completed
    seed makes the average curves observable while a long experiment runs.
    """
    validation = _mode4_average_tree_random_hypervolume(
        seed_directories, output_dir, evaluation_label="validation",
        data_mode=data_mode,
    )
    test = _mode4_average_tree_random_hypervolume(
        seed_directories, output_dir, evaluation_label="test",
        data_mode=data_mode,
    )
    _mode4_plot_validation_test_side_by_side(seed_directories, output_dir)
    return validation, test




def _run_mode5_plot_validation_smoke_test(
    base_config: dict, experiment_dir: Path, n_jobs: int | None,
    serial_lm: bool,
) -> None:
    """Run a disposable short Mode 5 validation before the real experiment.

    This is intentionally isolated from the paper-experiment seed directories.
    It exercises tree/random comparison plus the native PySR curve with five
    search rounds and 50 PySR iterations.  Its artifacts are retained for
    inspection and are excluded from formal seed aggregation.
    """
    temp_root = experiment_dir / "mode5_validation_smoke"
    temp_root.mkdir(parents=True, exist_ok=False)
    try:
        seed_dir = temp_root / "seed_001"
        smoke_config = copy.deepcopy(base_config)
        smoke_config["seed"] = int(base_config.get("seed", 0))
        smoke_config["output"] = {
            **smoke_config.get("output", {}), "directory": str(seed_dir),
        }
        smoke_config["minimal_output_artifacts"] = False
        smoke_config["suppress_results_json"] = False
        smoke_config["mode2"] = {
            **smoke_config.get("mode2", {}), "max_iterations": 5,
        }
        smoke_config["mode34_disable_early_stopping"] = True
        smoke_config["pysr_baseline"] = {
            **smoke_config.get("pysr_baseline", {}), "niterations": 50,
        }
        smoke_config.pop("mode5_retrain_interval", None)
        smoke_config_path = temp_root / "smoke_config.json"
        smoke_config_path.write_text(
            json.dumps(_json_value(smoke_config), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            "\n[Validation] starting temporary validation: tree model/ablation 5, PySR 50; "
            "validation plots and results will be kept for inspection.", flush=True,
        )
        run(
            smoke_config_path, n_jobs=(1 if serial_lm else n_jobs),
            pipeline_mode=3, _allow_single_mode4=True,
            validation_enabled=True,
        )
        data_mode = _configured_data_profile(base_config.get("data", {}))
        _refresh_mode4_average_hypervolume_artifacts(
            [seed_dir], temp_root, data_mode
        )
        print(
            f"[Validation] smoke test complete; results retained at: {temp_root}",
            flush=True,
        )
    finally:
        # Keep the smoke artifacts; the finally block only preserves the
        # isolated lifecycle if the formal experiment raises later.
        pass


def _mode4_plot_validation_test_side_by_side(
    seed_directories: Sequence[Path], output_dir: Path,
) -> None:
    """Plot validation and test HV side by side using the unified artifacts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"tree_model": ("o", "Tree model"),
              "random_permutation": ("^", "Random permutation ablation")}
    aggregate_styles = {
        "tree_model": {"marker": "o", "color": "#1f77b4", "label": "Tree model"},
        "random_permutation": {
            "marker": "^", "color": "#ff7f0e",
            "label": "Random permutation ablation",
        },
    }

    for seed_directory in seed_directories:
        comparisons = {}
        for evaluation_label in ("validation", "test"):
            path = seed_directory / evaluation_label / "hypervolume" / f"{evaluation_label}_comparison.json"
            if not path.is_file():
                continue
            try:
                comparisons[evaluation_label] = json.loads(
                    path.read_text(encoding="utf-8")
                ).get("per_iteration", [])
            except (OSError, json.JSONDecodeError):
                continue
        if not comparisons:
            continue
        figure, axes = plt.subplots(1, 2, figsize=(16, 5.5), squeeze=False)
        for axis, evaluation_label in zip(axes[0], ("validation", "test")):
            rows = comparisons.get(evaluation_label, [])
            for method, (marker, label) in styles.items():
                values = sorted(
                    (row for row in rows if row.get("method") == method),
                    key=lambda row: float(row.get("iteration", 0)),
                )
                if values:
                    axis.plot(
                        [float(row["iteration"]) for row in values],
                        [float(row[f"{evaluation_label}_hypervolume"])
                         for row in values],
                        marker=marker, markersize=9.4, linewidth=1.75, label=label,
                    )
            axis.set_xlabel("Iteration", fontsize=15)
            axis.set_ylabel(f"{evaluation_label.capitalize()} hypervolume", fontsize=15)
            axis.tick_params(axis="both", labelsize=13)
            axis.grid(True, alpha=0.3)
            axis.legend(fontsize=13, loc="upper left", framealpha=1.0)
        figure.tight_layout()
        figure.savefig(
            seed_directory / "hypervolume" / "validation_test_hypervolume_by_iteration.png",
            dpi=220,
        )
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(16, 5.5), squeeze=False)
    has_data = False
    for axis, evaluation_label in zip(axes[0], ("validation", "test")):
        for spine in axis.spines.values():
            spine.set_linewidth(0.9)
        path = output_dir / "hypervolume" / (
            f"tree_vs_random_{evaluation_label}_hypervolume_by_iteration.csv"
        )
        # Paper experiment outputs retain the mode-specific aggregate name.
        # Read that artifact as well so regenerating the combined figure does
        # not silently produce an empty plot.
        if not path.is_file():
            path = output_dir / "hypervolume" / (
                f"mode4_tree_vs_random_{evaluation_label}_hypervolume_by_iteration.csv"
            )
        rows = []
        if path.is_file():
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        for method, style in aggregate_styles.items():
            values = sorted(
                (row for row in rows
                 if float(row.get(f"{method}_hypervolume") or "nan") ==
                 float(row.get(f"{method}_hypervolume") or "nan")),
                key=lambda row: float(row["iteration"]),
            )
            if values:
                has_data = True
                axis.plot(
                    [float(row["iteration"]) for row in values],
                    [float(row[f"{method}_hypervolume"]) for row in values],
                    marker=style["marker"], markersize=9.4,
                    linewidth=1.75, color=style["color"], label=style["label"],
                )
        axis.set_xlabel("Iteration", fontsize=15)
        axis.set_ylabel(f"{evaluation_label.capitalize()} hypervolume", fontsize=15)
        axis.tick_params(axis="both", labelsize=13)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=13, loc="upper left", framealpha=1.0)
    if has_data:
        figure.tight_layout()
        figure.savefig(
            output_dir / "hypervolume" / "tree_vs_random_validation_test_hypervolume_by_iteration.png",
            dpi=220,
        )
    plt.close(figure)


def run_mode4_paper_experiment(
    config_path: Path, *, n_jobs: int | None = None,
    serial_lm: bool = False, experiment_mode: int = 4,
    retrain_interval: int = 0, seed_count: int = 10,
    target_iterations: int | None = None,
) -> dict:
    """Run and aggregate the multi-seed Mode 4/5/6 experiment."""
    experiment_mode = int(experiment_mode)
    if experiment_mode not in (4, 5, 6):
        raise ValueError("Paper-level experiments support only Mode 4, Mode 5, or Mode 6")
    if target_iterations is None:
        target_iterations = 20 if experiment_mode == 6 else 40
    if int(seed_count) < 1 or int(target_iterations) < 1:
        raise ValueError("seed_count and target_iterations must be positive integers")
    mode_label = "Experiment"
    mode_prefix = f"mode{experiment_mode}"
    base_config = json.loads(config_path.read_text(encoding="utf-8"))
    base_seed = int(base_config.get("seed", 0))
    seeds = [base_seed + index for index in range(int(seed_count))]

    output_root = resolve_path(base_config["output"]["directory"])
    experiment_prefix = "experiment"
    # Every invocation gets a fresh experiment directory. Existing seed
    # markers and legacy artifacts are never consulted for resume decisions.
    experiment_dir = _next_mode4_experiment_directory(
        output_root, prefix=experiment_prefix
    )
    config_dir = experiment_dir / "configs"
    summary_dir = experiment_dir / "summary"
    config_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    # The main experiment runner owns only the search pipeline. PySR has a
    # separate runner and output tree.

    data_mode = _configured_data_profile(base_config.get("data", {}))
    realdata_segmentation = (
        base_config.get("data", {}).get("realdata", {}).get("segmentation", {}) or {}
    )
    # Mode 6 follows the complete Mode 5 experiment protocol.  Its only
    # intentional experimental difference here is that the random ablation
    # remains disabled below.
    mode5_protocol = experiment_mode in (5, 6)
    independent_segment_mode = bool(
        mode5_protocol
        and _is_measured_pathloss_profile(data_mode)
        and realdata_segmentation.get("enabled", True)
        and realdata_segmentation.get(
            "independent_segment_fitting",
            realdata_segmentation.get("formula_pair_search", False),
        )
    )
    dual_segment_mode = independent_segment_mode

    all_front_rows, seed_summaries = [], []
    histories = []
    for index, seed in enumerate(seeds, start=1):
        seed_dir = experiment_dir / f"seed_{index:03d}"
        seed_config = copy.deepcopy(base_config)
        seed_config["seed"] = seed
        seed_config.setdefault("fitting", {})["fit_mode"] = "refit"
        seed_config["output"] = {**seed_config.get("output", {}),
                                  "directory": str(seed_dir)}
        # Keep only the Pareto/search, validation/test, baseline, comparison,
        # and visualization artifacts for each seed.  The fitting and ranking
        # stages still run in memory, but their intermediate CSV/JSON/model
        # artifacts are intentionally not persisted.
        seed_config["minimal_output_artifacts"] = True
        seed_config["suppress_results_json"] = True
        if experiment_mode == 6:
            seed_config["disable_all_ablations"] = True
        seed_config.setdefault("mode2", {})["max_iterations"] = int(target_iterations)
        seed_config["mode34_disable_early_stopping"] = True
        if mode5_protocol:
            seed_config["mode5_retrain_interval"] = int(retrain_interval or 5)
        if mode5_protocol:
            seed_config["periodic_constant_significance_interval"] = 1
            # Use the Mode 5 protocol selector for both modes so Mode 6 also
            # receives Mode 5's initial selection and zeroing logic.
            seed_config["periodic_constant_significance_experiment_mode"] = 5
        seed_config_path = config_dir / f"seed_{index:03d}_config.json"
        seed_config_path.write_text(
            json.dumps(_json_value(seed_config), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"\n[{mode_label}] {index}/{len(seeds)}, seed={seed}: starting; "
            f"target iterations={int(target_iterations)}; Pareto-change early stopping=off",
            flush=True,
        )
        try:
            # Each seed reuses the Mode 3 search implementation.  Mode 5 adds
            # periodic surrogate retraining through seed_config above.
            if dual_segment_mode:
                independent_dir = seed_dir / "independent_segment_fitting"
                shared_dir = seed_dir / "shared_structure_fitting"
                independent_payload = _run_mode5_independent_segment_seed(
                    seed_config_path, independent_dir,
                    n_jobs=n_jobs, serial_lm=serial_lm,
                )
                shared_payload = _run_mode5_shared_structure_seed(
                    seed_config_path, shared_dir,
                    n_jobs=n_jobs, serial_lm=serial_lm,
                )
                independent_front, independent_summary = _mode4_summary_rows(
                    independent_payload, seed, independent_dir
                )
                shared_front, shared_summary = _mode4_summary_rows(
                    shared_payload, seed, shared_dir
                )
                for row in independent_front:
                    row["method"] = "independent_segment_fitting"
                for row in shared_front:
                    row["method"] = "shared_structure_fitting"
                front_rows = independent_front + shared_front
                summary = {
                    "seed": int(seed),
                    "output_directory": str(seed_dir),
                    "run_completed": bool(
                        independent_summary.get("run_completed")
                        and shared_summary.get("run_completed")
                    ),
                    "status": "completed",
                    "methods": {
                        "independent_segment_fitting": independent_summary,
                        "shared_structure_fitting": shared_summary,
                    },
                }
                for metric_name in (
                    "pareto_front_size", "best_mean_r2",
                    "best_validation_mean_r2", "pysr_best_validation_mean_r2",
                    "pysr_best_test_mean_r2", "final_hypervolume",
                    "minimum_complexity", "iterations",
                ):
                    summary[metric_name] = shared_summary.get(metric_name)
                payload = {
                    "metadata": {
                        "run_completed": summary["run_completed"],
                        "dual_segment_mode": True,
                    },
                    "gp_pareto_front": front_rows,
                    # The independent branch has no unified history; use the
                    # shared-structure history for convergence reporting.
                    "mode2_hypervolume_history": shared_payload.get(
                        "mode2_hypervolume_history", []
                    ),
                }
            else:
                payload = run(
                    seed_config_path, n_jobs=(1 if serial_lm else n_jobs),
                    pipeline_mode=3, _allow_single_mode4=True,
                    validation_enabled=True,
                )
            if not dual_segment_mode:
                front_rows, summary = _mode4_summary_rows(payload, seed, seed_dir)
            seed_history = payload.get("mode2_hypervolume_history", [])
            histories.append({
                "seed": seed,
                "history": seed_history,
            })
        except Exception as exc:
            logging.exception("%s seed=%s run failed", mode_label, seed)
            front_rows = []
            summary = {
                "seed": int(seed), "output_directory": str(seed_dir),
                "pareto_front_size": 0, "run_completed": False,
                "status": "failed", "error": str(exc),
            }
        all_front_rows.extend(front_rows)
        seed_summaries.append(summary)
        available_seed_directories = sorted(
            path for path in experiment_dir.glob("seed_*")
            if path.is_dir() and not path.name.endswith("_config.json")
        )
        if dual_segment_mode:
            shared_seed_dirs = sorted(
                path / "shared_structure_fitting"
                for path in available_seed_directories
                if (path / "shared_structure_fitting").is_dir()
            )
            shared_output = experiment_dir / "shared_structure_fitting"
            _refresh_mode4_average_hypervolume_artifacts(
                shared_seed_dirs, shared_output, data_mode
            )
            for segment_name in ("segment_1", "segment_2"):
                segment_seed_dirs = sorted(
                    path / "independent_segment_fitting" / segment_name
                    for path in available_seed_directories
                    if (path / "independent_segment_fitting" / segment_name).is_dir()
                )
                segment_output = experiment_dir / "independent_segment_fitting" / segment_name
                _refresh_mode4_average_hypervolume_artifacts(
                    segment_seed_dirs, segment_output, data_mode
                )
            final_hv_comparison_path = None
        elif independent_segment_mode:
            for segment_name in ("segment_1", "segment_2"):
                segment_seed_dirs = sorted(
                    path / segment_name for path in available_seed_directories
                    if (path / segment_name).is_dir()
                )
                segment_output = experiment_dir / segment_name
                _refresh_mode4_average_hypervolume_artifacts(
                    segment_seed_dirs, segment_output, data_mode
                )
            final_hv_comparison_path = None
        else:
            _refresh_mode4_average_hypervolume_artifacts(
                available_seed_directories, experiment_dir, data_mode
            )
            final_hv_comparison_path = None
        print(
            f"[{mode_label}] {index}/{len(seeds)}, seed={seed}: completed; "
            "seed run completed",
            flush=True,
        )

    _write_csv(summary_dir / f"{mode_prefix}_seed_summary.csv", seed_summaries)
    _write_csv(summary_dir / f"{mode_prefix}_pareto_fronts.csv", all_front_rows)
    seed_directories = sorted(
        path for path in experiment_dir.glob("seed_*")
        if path.is_dir() and not path.name.endswith("_config.json")
    )
    # Re-run the same incremental refresh once all seeds are available.  This
    # preserves the legacy final artifacts while guaranteeing their content
    # is based on the complete experiment.
    if dual_segment_mode:
        aggregate_tree_random_hv = {}
        aggregate_tree_random_test_hv = {}
        shared_seed_dirs = [
            path / "shared_structure_fitting" for path in seed_directories
            if (path / "shared_structure_fitting").is_dir()
        ]
        shared_output = summary_dir / "shared_structure_fitting"
        shared_validation_hv, shared_test_hv = _refresh_mode4_average_hypervolume_artifacts(
            shared_seed_dirs, shared_output, data_mode
        )
        aggregate_tree_random_hv["shared_structure_fitting"] = shared_validation_hv
        aggregate_tree_random_test_hv["shared_structure_fitting"] = shared_test_hv
        for segment_name in ("segment_1", "segment_2"):
            segment_seed_dirs = [
                path / "independent_segment_fitting" / segment_name
                for path in seed_directories
                if (path / "independent_segment_fitting" / segment_name).is_dir()
            ]
            segment_output = summary_dir / "independent_segment_fitting" / segment_name
            validation_hv, test_hv = _refresh_mode4_average_hypervolume_artifacts(
                segment_seed_dirs, segment_output, data_mode
            )
            aggregate_tree_random_hv[segment_name] = validation_hv
            aggregate_tree_random_test_hv[segment_name] = test_hv
        final_hv_comparison_path = None
    elif independent_segment_mode:
        aggregate_tree_random_hv = {}
        aggregate_tree_random_test_hv = {}
        final_hv_comparison_path = None
        for segment_name in ("segment_1", "segment_2"):
            segment_seed_dirs = [path / segment_name for path in seed_directories
                                 if (path / segment_name).is_dir()]
            segment_output = summary_dir / segment_name
            validation_hv, test_hv = _refresh_mode4_average_hypervolume_artifacts(
                segment_seed_dirs, segment_output, data_mode
            )
            aggregate_tree_random_hv[segment_name] = validation_hv
            aggregate_tree_random_test_hv[segment_name] = test_hv
    else:
        aggregate_tree_random_hv, aggregate_tree_random_test_hv = (
            _refresh_mode4_average_hypervolume_artifacts(
                seed_directories, summary_dir, data_mode
            )
        )
        final_hv_comparison_path = None
    numeric_metrics = ("best_mean_r2", "best_validation_mean_r2",
                       "pysr_best_validation_mean_r2",
                       "pysr_best_test_mean_r2",
                       "minimum_complexity", "final_hypervolume",
                       "pareto_front_size", "iterations")
    aggregate = {}
    for field in numeric_metrics:
        values = [float(row[field]) for row in seed_summaries
                  if row.get(field) is not None]
        if not values:
            aggregate[field] = {"count": 0, "mean": None, "std": None,
                                "min": None, "max": None,
                                "ci95_half_width": None}
            continue
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        aggregate[field] = {
            "count": len(values), "mean": mean, "std": std,
            "min": float(np.min(values)), "max": float(np.max(values)),
            "ci95_half_width": float(1.96 * std / math.sqrt(len(values))),
        }
    if independent_segment_mode:
        aggregate["segments"] = {}
        for segment_name in ("segment_1", "segment_2"):
            segment_metrics = {}
            for field in numeric_metrics:
                values = [
                    float(item.get("segment_summaries", {}).get(segment_name, {})
                          .get(field))
                    for item in seed_summaries
                    if item.get("segment_summaries", {}).get(segment_name, {})
                    .get(field) is not None
                ]
                segment_metrics[field] = {
                    "count": len(values),
                    "mean": float(np.mean(values)) if values else None,
                    "std": (float(np.std(values, ddof=1))
                            if len(values) > 1 else (0.0 if values else None)),
                    "min": float(np.min(values)) if values else None,
                    "max": float(np.max(values)) if values else None,
                }
            aggregate["segments"][segment_name] = segment_metrics
    result = {
        "experiment": f"{mode_prefix}_{int(seed_count)}seed_experiment",
        "seed_count": len(seeds), "seeds": seeds,
        "experiment_directory": str(experiment_dir),
        "seed_directories": [str(experiment_dir / f"seed_{i:03d}")
                             for i in range(1, len(seeds) + 1)],
        "seed_summary_csv": str(summary_dir / f"{mode_prefix}_seed_summary.csv"),
        "pareto_fronts_csv": str(summary_dir / f"{mode_prefix}_pareto_fronts.csv"),
        "per_seed": seed_summaries, "aggregate": aggregate,
        "tree_random_hypervolume": aggregate_tree_random_hv,
        "tree_random_test_hypervolume": aggregate_tree_random_test_hv,
    }

    # Recalculate the original per-seed convergence figure from the complete
    # saved training histories.  The online values are front-local; this
    # post-run figure uses one exact all-iteration scale for the experiment.
    training_fields = (
        ("pooled_r2", "complexity")
        if _is_measured_pathloss_profile(data_mode)
        else ("mean_r2", "r2_std", "complexity")
    )
    training_rows = [
        row
        for item in histories
        for history_item in item.get("history", [])
        for row in history_item.get("front", [])
    ]
    training_bounds = (
        _validation_hv_bounds(training_rows, training_fields)
        if training_rows and not independent_segment_mode else {}
    )
    unified_training_histories = []
    for item in histories:
        values = []
        for history_item in item.get("history", []):
            front = history_item.get("front", [])
            if training_bounds:
                values.append(_calculate_validation_hv(
                    front, training_fields, {training_fields[0]}, training_bounds
                ))
        unified_training_histories.append({"seed": item["seed"], "values": values})

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if independent_segment_mode:
            for segment_name in ("segment_1", "segment_2"):
                figure, axis = plt.subplots(figsize=(9, 6))
                for seed in seeds:
                    rows = [row for row in all_front_rows
                            if row["seed"] == seed
                            and row.get("segment_name") == segment_name
                            and row.get("complexity") is not None
                            and row.get("mean_r2") is not None]
                    axis.scatter([row["complexity"] for row in rows],
                                 [row["mean_r2"] for row in rows],
                                 s=35, alpha=0.7, label=f"seed {seed}")
                axis.set_xlabel("Formula complexity (lower is better)")
                axis.set_ylabel("Mean R² (higher is better)")
                axis.set_title(f"{segment_name}: independent Pareto fronts")
                axis.grid(True, alpha=0.25)
                if axis.has_data():
                    axis.legend(ncol=2)
                figure.tight_layout()
                pareto_plot = summary_dir / segment_name / (
                    f"{mode_prefix}_{segment_name}_pareto_fronts_by_seed.png"
                )
                pareto_plot.parent.mkdir(parents=True, exist_ok=True)
                plt.close(figure)
        else:
            figure, axis = plt.subplots(figsize=(9, 6))
            for seed in seeds:
                rows = [row for row in all_front_rows if row["seed"] == seed
                        and row.get("complexity") is not None
                        and row.get("mean_r2") is not None]
                axis.scatter([row["complexity"] for row in rows],
                             [row["mean_r2"] for row in rows], s=35, alpha=0.7,
                             label=f"seed {seed}")
            axis.set_xlabel("Formula complexity (lower is better)")
            axis.set_ylabel("Mean R² (higher is better)")
            axis.set_title("Mode 4 six-seed Pareto fronts")
            axis.grid(True, alpha=0.25)
            if axis.has_data():
                axis.legend(ncol=2)
            figure.tight_layout()
            pareto_plot = summary_dir / f"{mode_prefix}_pareto_fronts_by_seed.png"
            plt.close(figure)

        if not independent_segment_mode:
            figure, axis = plt.subplots(figsize=(9, 6))
            for item in unified_training_histories:
                values = item["values"]
                if values:
                    axis.plot(range(1, len(values) + 1), values,
                              marker="o", ms=7, linewidth=1.75, alpha=0.7,
                              label=f"seed {item['seed']}")
            axis.set_xlabel("Iteration", fontsize=15)
            axis.set_ylabel("Hypervolume", fontsize=15)
            axis.set_title("Mode 4 six-seed convergence")
            axis.tick_params(axis="both", labelsize=13)
            axis.grid(True, alpha=0.25)
            if axis.has_data():
                axis.legend(ncol=2, fontsize=12)
            figure.tight_layout()
            convergence_plot = summary_dir / f"{mode_prefix}_hypervolume_by_seed.png"
            plt.close(figure)
    except ImportError:
        result["plotting"] = {"enabled": False, "reason": "matplotlib_not_installed"}

    # Produce the standalone canonicalized formula tables as part of every
    # completed multi-seed run, using this run's fresh experiment directory.
    from canonicalize_repeated_structures import main as run_canonicalize

    main_front_paths = []
    for seed_directory in experiment_dir.glob("seed_[0-9][0-9][0-9]"):
        main_front_paths.extend((
            seed_directory / "test" / "test_pareto_front.csv",
            seed_directory / "shared_structure_fitting" / "test" / "test_pareto_front.csv",
            seed_directory / "independent_segment_fitting" / "segment_1" / "test" / "test_pareto_front.csv",
            seed_directory / "independent_segment_fitting" / "segment_2" / "test" / "test_pareto_front.csv",
            seed_directory / "segment_1" / "test" / "test_pareto_front.csv",
            seed_directory / "segment_2" / "test" / "test_pareto_front.csv",
        ))
    if any(path.is_file() for path in main_front_paths):
        print("\nGenerating main-model canonicalized results...", flush=True)
        result["canonicalized_results"] = run_canonicalize(
            experiment_dir, model="main"
        )
    else:
        result["canonicalized_results"] = {
            "skipped": True,
            "reason": "no main-model test Pareto fronts were produced",
        }
        print(
            "No main-model test Pareto fronts were produced; "
            "skipping main-model canonicalization.",
            flush=True,
        )

    summary_path = summary_dir / f"{mode_prefix}_experiment_summary.json"
    result["summary_json"] = str(summary_path)
    summary_path.write_text(
        json.dumps(_json_value(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nExperiment complete: {experiment_dir}", flush=True)
    return result


def run(
    config_path: Path = DEFAULT_CONFIG,
    symbol_operator_stratified: bool = True,
    pairwise_trigonometric_operator_mutation: bool = False,
    pca_tree_features: bool = False,
    nearest_ancestor_only: bool = False,
    numeric_features_only: bool = False,
    kernel_pca_components: int = 64,
    grouped_tree_feature_ratio: float | None = None,
    prime_experimental_feature_mode: str | None = None,
    n_jobs: int | None = None,
    parallel_lm: bool | None = None,
    pipeline_mode: int = 3,
    validation_enabled: bool | None = None,
    _allow_single_mode4: bool = False,
) -> dict:
    if int(pipeline_mode) in (5, 6) and not _allow_single_mode4:
        return run_mode4_paper_experiment(
            Path(config_path), n_jobs=n_jobs, serial_lm=(n_jobs == 1),
            experiment_mode=int(pipeline_mode),
            retrain_interval=(5 if int(pipeline_mode) == 5 else 0),
            target_iterations=(20 if int(pipeline_mode) == 6 else 40),
        )
    # Mode 5/6 seed runs deliberately reuse the Mode 3 search implementation
    # internally; Mode 3 remains an implementation detail of those modes.
    if int(pipeline_mode) in (5, 6):
        pipeline_mode = 3
    config = load_config(config_path, pipeline_mode=pipeline_mode)
    if validation_enabled is not None:
        if not validation_enabled:
            raise ValueError("Validation is required and cannot be disabled")
        config.setdefault("validation", {})["enabled"] = True
    if int(pipeline_mode) in (3, 4):
        # Keep the switch configurable for callers that need the full JSON;
        # the default Mode 3 behavior omits only formula_discovery_results.
        config["suppress_results_json"] = bool(
            config.get("suppress_results_json", True)
        )
    ranking_settings = config.setdefault("ranking", {})
    ranking_settings["feature_mode"] = "operator_primitive_matrix"
    # Ablations rank already-fitted formulas directly by LM R², but may still
    # opt into the shared operator-stratified Top-K selector.
    if int(pipeline_mode) in (2, 3, 4):
        # The surrogate modes explicitly disable operator stratification.
        ranking_settings["symbol_operator_stratified"] = False
    else:
        ranking_settings["symbol_operator_stratified"] = (
            bool(ranking_settings.get("ablation_use_operator_stratified", False))
            if ranking_settings.get("ablation_enabled", False)
            else bool(symbol_operator_stratified)
        )
    ranking_settings["pca_tree_features"] = bool(pca_tree_features)
    ranking_settings["kernel_pca_components"] = int(kernel_pca_components)
    # Mode 3 keeps the PCA dimensions, sampling, batching, and feature
    # assembly unchanged, but uses ordinary randomized PCA instead of the
    # legacy RBF Kernel PCA path.
    ranking_settings["tree_pca_method"] = "pca" if int(pipeline_mode) in (3, 4) else "kernel"
    if int(pipeline_mode) in (3, 4):
        # Modes 3/4 keep operator incidence, response SVD, and tree PCA, while
        # excluding the hand-engineered numeric structure group.
        ranking_settings["feature_ablation_remove"] = sorted(set(
            ranking_settings.get("feature_ablation_remove", []))
            | {"engineered_structure_numeric"}
        )
    # Mode 2 keeps the global-numeric tree representation, but its tree
    # products are deliberately limited to the nearest ancestor.  Preserve
    # the explicit argument for the other modes unchanged.
    ranking_settings["nearest_ancestor_only"] = bool(
        nearest_ancestor_only or numeric_features_only
    )
    if prime_experimental_feature_mode is not None:
        ranking_settings["tree_coordinates_prime_experimental_feature_mode"] = (
            str(prime_experimental_feature_mode).strip().lower()
        )
    if grouped_tree_feature_ratio is not None:
        ranking_settings["grouped_tree_feature_ratio"] = float(grouped_tree_feature_ratio)
        ranking_settings["grouped_extra_trees_enabled"] = True
    if numeric_features_only:
        ranking_settings["feature_mode"] = "tree_coordinates_global_numeric"
    seed = int(config["seed"])
    py_rng, np_rng = random.Random(seed), np.random.default_rng(seed)
    ranking_config = config.get("ranking", {})
    ranking_config["seed"] = seed
    precomputed_formula_artifact = select_precomputed_formula_artifact(config)
    if not precomputed_formula_artifact:
        raise FileNotFoundError(
            "This pipeline no longer generates formulas/features at runtime; set "
            "output.artifact in formula_generation.json and first run "
            "precompute_formula_features.py."
        )
    ranking_config["precomputed_feature_artifact"] = (
        precomputed_formula_artifact
    )
    partial_reuse = False
    if (ranking_config.get("enabled", False)
            and not ranking_config.get("ablation_enabled", False)):
        validate_ranking_dependencies(ranking_config.get("model_type", "extra_trees"))
    # Fitting observations always come from the configured CSV.  Reuse the
    # existing pipeline reader so numeric-column validation, feature mapping,
    # target selection and NaN/Inf row filtering stay consistent.
    data_cfg = config["data"]
    output_dir = resolve_path(config["output"]["directory"])
    fit_mode = config["fitting"].get("fit_mode", "refit")
    history_payload = None
    history_path = output_dir / "formula_discovery_results.json"
    legacy_history_path = output_dir / "dimensionless_formula_results.json"
    if not history_path.is_file() and legacy_history_path.is_file():
        history_path = legacy_history_path
    if fit_mode == "reuse_history" and config.get("stage", "all") != "generate":
        if not history_path.is_file():
            raise FileNotFoundError(
                f"Configuration requires reusing historical fit results, but the file does not exist: {history_path}"
            )
        history_payload = json.loads(history_path.read_text(encoding="utf-8"))
    archived_history_path = archive_previous_results(output_dir, history_path)
    X, y, feature_names, dataset_groups = load_configured_data(data_cfg)
    mode2_binary_constraint = _mode2_requires_binary_formula(
        config, pipeline_mode, feature_count=X.shape[1]
    )
    split_cfg = data_cfg.get("split", {}) or {}
    split_seed = int(split_cfg.get("seed", seed))
    mat_segmentation_cfg = data_cfg.get("realdata", {}).get("segmentation", {}) or {}
    mat_breakpoint_cfg = mat_segmentation_cfg.get("breakpoint_prior", {}) or {}
    if (_is_measured_pathloss_profile(data_cfg)
            and bool(mat_segmentation_cfg.get("enabled", False))
            and bool(mat_breakpoint_cfg.get("enabled", False))):
        train_data, validation_data, test_data, validation_split_metadata = (
            _split_by_piecewise_segments(
                (X, y, feature_names, dataset_groups), split_seed,
                data_cfg.get("realdata", {}).get("segmentation", {}) or {},
                split_cfg.get("train_fraction", 0.5),
                split_cfg.get("validation_fraction", 0.25),
                split_cfg.get("test_fraction", 0.25),
            )
        )
    else:
        train_data, validation_data, test_data, validation_split_metadata = (
            _split_train_validation_test_data(
                (X, y, feature_names, dataset_groups), split_seed,
                split_cfg.get("train_fraction", 0.5),
                split_cfg.get("validation_fraction", 0.25),
                split_cfg.get("test_fraction", 0.25),
            )
        )
    split_indices_path = save_split_indices(
        output_dir, validation_split_metadata, seed=seed,
        config_path=config_path, data_mode=data_cfg.get("mode", "csv"),
        data_profile=_configured_data_profile(data_cfg),
        split_config=split_cfg,
    )
    logging.info("Saved train/validation/test row indices to %s", split_indices_path)
    X, y, feature_names, dataset_groups = train_data
    validation_X, validation_y, validation_features, validation_groups = validation_data
    if list(validation_features) != list(feature_names):
        raise ValueError(
            "Validation feature columns and order must match the training set: "
            f"training={feature_names}, validation={validation_features}"
        )
    validation_config = config.setdefault("validation", {})
    validation_config.setdefault("enabled", True)
    validation_config.setdefault("output_directory", "validation")
    logging.info(
        "Automatic data split: train=%d, validation=%d, test=%d; random seed=%d",
        len(y), len(validation_y), len(test_data[1]), int(seed),
    )
    # Keep validation arrays in the in-process fitting configuration.  They
    # are runtime data carried through process-pool fitting tasks.
    config["fitting"]["_validation_X"] = validation_X
    config["fitting"]["_validation_y"] = validation_y
    config["fitting"]["_validation_groups"] = validation_data[3]
    # The test partition from the shared data.split path is the only test set.
    is_realdata_mat = _is_measured_pathloss_profile(data_cfg)
    segmentation_config = (data_cfg.get("realdata", {}).get("segmentation", {})
                           if is_realdata_mat else {})
    segmentation_enabled = is_realdata_mat and bool(segmentation_config.get("enabled", True))
    breakpoint_prior = _breakpoint_prior(segmentation_config) if segmentation_enabled else None
    if breakpoint_prior is not None:
        logging.info(
            "Breakpoint prior enabled: d_break=%.9g m (data-distance unit=%.9g), "
            "regularization weight=%.9g",
            breakpoint_prior["breakpoint_m"], breakpoint_prior["breakpoint_value"],
            breakpoint_prior["regularization_weight"],
        )
    fixed_breakpoint_segmentation = breakpoint_prior is not None
    coarse_gp_fine_segmentation = (
        segmentation_enabled
        and not fixed_breakpoint_segmentation
        and segmentation_config.get("strategy", "coarse_gp_fine") == "coarse_gp_fine"
    )
    # A fixed physical breakpoint has no coarse-search stage: ranked formulas
    # go directly through the one-split piecewise fitter.  The new MAT
    # coarse_gp_fine workflow keeps global ranking only when no fixed prior
    # is configured.
    piecewise_enabled = segmentation_enabled and not coarse_gp_fine_segmentation
    data_path = Path(dataset_groups[0]["path"])
    target_name = dataset_groups[0]["target"]
    started = time.time()
    logging.info("Data mode=%s, datasets=%d, samples=%d, %s -> %s (frequency-band features not added)",
                 data_cfg.get("mode", "csv"), len(dataset_groups), len(y),
                 feature_names, target_name)

    formula_artifact = load_precomputed_formula_artifact(
        precomputed_formula_artifact, config, X.shape[1]
    )
    bases = formula_artifact["bases"]
    candidates = list(formula_artifact["single_candidates"])
    combinations = list(formula_artifact["binary_combinations"])
    # The precomputed artifact keeps a canonical formula order.  Runtime
    # seeds may still explore that fixed library in different reproducible
    # orders without changing the formula definitions or feature matrix.
    py_rng.shuffle(candidates)
    py_rng.shuffle(combinations)
    if history_payload is not None:
        old_candidates = history_payload.get("single_candidates", [])
        old_combinations = history_payload.get("binary_combinations", [])
        current_structures = {
            row["id"]: row.get("readable_expression")
            for row in [*candidates, *combinations]
        }
        old_structures = {
            row["id"]: row.get("readable_expression")
            for row in [*old_candidates, *old_combinations]
        }
        if current_structures != old_structures:
            raise ValueError(
                "Historical fit results do not match the current formula structures; keep the generation configuration and seed unchanged, "
                "or set fitting.fit_mode to refit"
            )
        if not history_payload.get("single_best") and not history_payload.get("combination_best"):
            raise ValueError(f"No reusable fit results found in the historical results: {history_path}")
    payload = {
        "metadata": {"config_path": str(config_path), "seed": seed,
                     "symbol_operator_stratified": bool(
                         ranking_settings.get("symbol_operator_stratified", False)
                     ),
                     "data_path": str(data_path), "datasets": dataset_groups,
                     "joint_dataset_count": len(dataset_groups),
                     "frequency_feature_added": False,
                     "constants_shared_across_datasets": False,
                     "run_status": "running",
                     "run_completed": False,
                     "suppress_large_artifacts": bool(
                         config.get("suppress_large_artifacts", False)
                     ),
                     "suppress_results_json": bool(
                         config.get("suppress_results_json", False)
                     ),
                     "minimal_output_artifacts": bool(
                         config.get("minimal_output_artifacts", False)
                     ),
                     "previous_results_archive": str(archived_history_path)
                     if archived_history_path else None,
                     "data_mode": data_cfg.get("mode", "csv"),
                     "data_profile": _configured_data_profile(data_cfg),
                     "validation": {
                         "enabled": bool(validation_config.get("enabled", False)),
                         "config": _json_value(validation_config),
                         "split": validation_split_metadata,
                     },
                     "source_formula": dataset_groups[0].get("source_formula"),
                     "piecewise_segmentation": (segmentation_config if segmentation_enabled
                                                 else None),
                     "features": feature_names, "target": target_name, "samples": len(y)},
        "bases": bases, "single_candidates": candidates, "binary_combinations": combinations,
        "single_initializations": [], "single_best": [], "combination_best": [],
        "structure_r2_ranking": [], "ranking_metadata": {"enabled": False},
        "top_ranked_refit_results": [],
        "piecewise_formula_pairs": [],
        "gp_pareto_front": [], "gp_log_merge_checks": [],
        "top_ranked_log_merge_checks": [],
        "top_ranked_constant_significance_checks": [],
        "gp_metadata": {"enabled": False},
    }
    if history_payload is not None:
        payload["single_initializations"] = history_payload.get("single_initializations", [])
        payload["single_best"] = history_payload.get("single_best", [])
        payload["combination_best"] = history_payload.get("combination_best", [])
        payload["mode2_exact_fit_cache"] = history_payload.get(
            "mode2_exact_fit_cache", {}
        )
    early_feature_correlation_diagnostic = None
    if (ranking_config.get("enabled", False)
            and not config.get("minimal_output_artifacts", False)):
        # The candidate formulas now exist, while no LM fit or ranking-model
        # training has started.  Save this diagnostic before any fitting work.
        early_feature_correlation_diagnostic = save_feature_correlation_diagnostic(
            payload, ranking_config, seed, output_dir
        )
        payload["ranking_metadata"]["feature_correlation_diagnostic"] = (
            early_feature_correlation_diagnostic
        )
    # In history-reuse mode the directory is both the input cache and the
    # output destination.  Preserve every historical artifact until the run
    # has produced its final replacement payload.
    if history_payload is None:
        save_outputs(output_dir, payload)
    if config.get("stage", "all") == "generate":
        payload["metadata"].update(run_status="completed", run_completed=True,
                                   elapsed_seconds=time.time() - started)
        save_outputs(output_dir, payload)
        return payload

    reused_history = history_payload is not None
    # Count formula structures that actually enter the direct LM/piecewise
    # fitting stage in this run.  Keep this as a set because the same formula
    # may be fitted on multiple datasets or split points.
    direct_fit_formula_ids = set()
    if reused_history:
        old_metadata = history_payload.get("metadata", {})
        allocation_metadata = old_metadata.get("training_structure_allocations", {})
        allocations = {group["name"]: ([], []) for group in dataset_groups}
        logging.info("Reusing historical fit results: %s", history_path)
    else:
        if int(pipeline_mode) in (2, 3, 4):
            # Mode 2 needs the same initial structures in every dataset so a
            # cross-dataset mean R² can be formed for each formula.
            if int(pipeline_mode) == 4:
                total_initial = int(config["fitting"].get("single_sample_count", 40)) + int(
                    config["fitting"].get("combination_sample_count", 120)
                )
                selected_initial, layer_metadata = _select_mode4_initial_complexity_layers(
                    candidates, combinations, total_initial, layer_count=5
                )
                shared_single = [row for row in selected_initial if row in candidates]
                shared_combination = [row for row in selected_initial if row in combinations]
                config.setdefault("mode4", {})["initial_complexity_layers"] = layer_metadata
                print(
                    "Mode 4 Initial complexity-layer training capacity: "
                    + "; ".join(
                        f"layer {item['layer']}=[{item['lower']:.3g},"
                        f"{item['upper']:.3g})"
                        f"/candidate {item['available_count']}"
                        f"/eligible for training set{item['training_capacity']}"
                        for item in layer_metadata
                    ),
                    flush=True,
                )
            else:
                shared_single, shared_combination = choose_fit_samples(
                    candidates, combinations, config["fitting"], py_rng
                )
            allocations = {
                group["name"]: (list(shared_single), list(shared_combination))
                for group in dataset_groups
            }
        else:
            allocations = choose_fit_samples_by_dataset(
                candidates, combinations, config["fitting"], py_rng, dataset_groups
            )
        allocation_metadata = {}
    formula_pair_search = piecewise_enabled and bool(
        segmentation_config.get("independent_segment_fitting", segmentation_config.get("formula_pair_search", False))
    )
    # ``n_jobs`` is shared by the formula-fitting stages: initial LM fitting
    # and Top-K refitting. Constant-significance refitting is forced to
    # serial execution below. Structure feature extraction remains serial at
    # its call sites.
    if n_jobs is None:
        n_jobs = -1 if parallel_lm else 1
    n_jobs = int(n_jobs)
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs for initial LM fitting must be 1, a positive integer, or -1")
    parallel_lm = n_jobs != 1
    # Constant-significance testing is intentionally serial.  Its tests are
    # expensive LM refits, and formula-level threading can oversubscribe the
    # host while also hiding progress behind executor.map.
    constant_significance_n_jobs = 1
    payload["metadata"]["parallel_lm_enabled"] = parallel_lm
    payload["metadata"]["initial_lm_n_jobs"] = n_jobs if parallel_lm else 1
    payload["metadata"]["constant_significance_n_jobs"] = constant_significance_n_jobs
    initial_lm_started = time.perf_counter()
    if formula_pair_search and not reused_history:
        # realdata_mat has exactly one sorted one-variable group.  Search the
        # selected formula library once, then preserve each structure's best
        # same-structure fit for ranking alongside the actual formula-pair results.
        group = dataset_groups[0]
        selected_candidates, selected_combinations = allocations[group["name"]]
        selected_records = [*selected_candidates, *selected_combinations]
        direct_fit_formula_ids.update(row["id"] for row in selected_records)
        payload["piecewise_formula_pairs"], best_by_formula = search_piecewise_formula_pairs(
            selected_records, X, y, config["fitting"], np_rng, segmentation_config
        )
        for record in selected_candidates:
            best = best_by_formula.get(record["id"])
            if best is None:
                continue
            best.update(dataset_name=group["name"], dataset_path=group.get("path"),
                        training_label_scope="independent_segment_fitting")
            payload["single_best"].append({**formula_record(
                sp.sympify(record["readable_expression"]), structure_type="single_local_fit",
                candidate_id=record["id"], bases=record["bases"], operator=record["operator"],
            ), **best})
        for record in selected_combinations:
            best = best_by_formula.get(record["id"])
            if best is None:
                continue
            best.update(dataset_name=group["name"], dataset_path=group.get("path"),
                        training_label_scope="independent_segment_fitting")
            payload["combination_best"].append({**formula_record(
                sp.sympify(record["readable_expression"]), structure_type="binary_global_fit",
                combination_id=record["id"], combination_type=record["combination_type"],
                candidate_ids=record["candidate_ids"], operators=record["operators"],
                base_ids=record["base_ids"],
            ), **best})
        allocation_metadata[group["name"]] = {
            "single_formula_ids": [row["id"] for row in selected_candidates],
            "combination_formula_ids": [row["id"] for row in selected_combinations],
        }
        allocations[group["name"]] = ([], [])
        save_checkpoint(output_dir, payload, "independent_segment_fitting")
    for group in dataset_groups:
        dataset_name = group["name"]
        selected_candidates, selected_combinations = allocations[dataset_name]
        # Formula-pair search above has already performed every LM fit for
        # this real-data group.  Do not render misleading empty legacy bars.
        if formula_pair_search:
            continue
        if not reused_history:
            allocation_metadata[dataset_name] = {
                "single_formula_ids": [row["id"] for row in selected_candidates],
                "combination_formula_ids": [row["id"] for row in selected_combinations],
            }
        start, end = int(group["start"]), int(group["end"])
        X_part, y_part = X[start:end], y[start:end]
        group_fitting_config = _fit_config_for_dataset_validation(
            config["fitting"], dataset_name
        )
        if is_realdata_mat and piecewise_enabled:
            group_fitting_config = _fit_config_for_piecewise_physical_validation(
                group_fitting_config, segmentation_config
            )

        fit_queue = ([(record, False) for record in selected_candidates]
                     + [(record, True) for record in selected_combinations])
        parallel_results = None
        if parallel_lm:
            tasks = [(
                record, X_part, y_part, group_fitting_config,
                segmentation_config, piecewise_enabled, dataset_name, seed,
            ) for record, _is_combination in fit_queue]
            parallel_results = _parallel_lm_results(
                tasks, max_workers=None if n_jobs < 0 else n_jobs
            )
        progress = tqdm(
            enumerate(fit_queue),
            total=len(fit_queue),
            desc=f"{dataset_name} LM fitting",
        )
        for queue_index, (record, is_combination) in progress:
            direct_fit_formula_ids.add(record["id"])
            if parallel_results is not None:
                attempts, best = parallel_results[queue_index]
            elif piecewise_enabled:
                attempts, best = fit_piecewise_formula(
                    record, X_part, y_part, group_fitting_config, np_rng,
                    segmentation_config
                )
            else:
                attempts, best = fit_with_restarts(
                    record, X_part, y_part, group_fitting_config, np_rng
                )
            for attempt in attempts:
                attempt.update(dataset_name=dataset_name, dataset_path=group.get("path"))
            if is_realdata_mat and piecewise_enabled and best.get("piecewise"):
                # Keep the two independently validation-selected segment fits
                # as the formula-level result.  Wrapping them as one ordinary
                # dataset row would hide segment_1/segment_2 from Mode 2.
                segment_rows = [
                    dict(item) for item in (best.get("dataset_fit_results") or [])
                ]
                best.update(
                    dataset_name=dataset_name, dataset_path=group.get("path"),
                    training_label_scope="piecewise_physical_prior",
                    dataset_fit_results=segment_rows,
                    dataset_metrics=[{
                        "name": item.get("dataset_name"),
                        "path": group.get("path"),
                        "r2": item.get("r2"), "mse": item.get("mse"),
                        "rmse": item.get("rmse"), "loss": item.get("loss"),
                        "valid_samples": item.get("valid_samples"),
                    } for item in segment_rows],
                    joint_r2=best.get("r2"),
                    joint_r2_aggregation="piecewise_segments",
                    constants_shared_across_datasets=False,
                )
            else:
                dataset_result = {**best, "dataset_name": dataset_name,
                                  "dataset_path": group.get("path")}
                best.update(
                    dataset_name=dataset_name, dataset_path=group.get("path"),
                    training_label_scope="single_dataset",
                    parameters_by_dataset={dataset_name: best.get("parameters")},
                    dataset_fit_results=[dataset_result],
                    dataset_metrics=[{"name": dataset_name, "path": group.get("path"),
                                      "r2": best.get("r2"), "mse": best.get("mse"),
                                      "rmse": best.get("rmse"), "loss": best.get("loss"),
                                      "valid_samples": best.get("valid_samples")}],
                    joint_r2=best.get("r2"), joint_r2_aggregation="single_dataset",
                    constants_shared_across_datasets=False,
                )
            payload["single_initializations"].extend(attempts)
            if is_combination:
                payload["combination_best"].append({**formula_record(
                    sp.sympify(record["readable_expression"]),
                    structure_type="binary_global_fit",
                    combination_id=record["id"], combination_type=record["combination_type"],
                    candidate_ids=record["candidate_ids"], operators=record["operators"],
                    base_ids=record["base_ids"]
                ), **best})
            else:
                payload["single_best"].append({**formula_record(
                    sp.sympify(record["readable_expression"]),
                    structure_type="single_local_fit", candidate_id=record["id"],
                    bases=record["bases"], operator=record["operator"]
                ), **best})
        save_checkpoint(output_dir, payload, f"lm_fitting_{dataset_name}")
    initial_lm_elapsed_seconds = (
        0.0 if reused_history else time.perf_counter() - initial_lm_started
    )
    payload["metadata"]["initial_lm_elapsed_seconds"] = initial_lm_elapsed_seconds
    payload["metadata"]["initial_lm_timing_status"] = (
        "reused_history_no_fit" if reused_history else "completed"
    )
    print(
        "Actual elapsed time for initial LM fitting: {:.3f} seconds{}".format(
            initial_lm_elapsed_seconds,
            "(reused historical results; initial LM fitting was not run)" if reused_history else "",
        ),
        flush=True,
    )
    initial_mean_r2_rows = collect_initial_lm_mean_r2(payload)
    # Evaluate the same frozen initial-LM parameters on the independent test
    # half. This is reported for comparison only and never feeds back into
    # fitting, early stopping, or candidate selection.
    # The ordinary CSV path stores one row per formula *and dataset* in
    # ``single_best``/``combination_best``.  Aggregate those rows by formula
    # before test evaluation; otherwise a formula's later dataset row would
    # overwrite the earlier one and its parameters would be applied globally.
    initial_fit_rows = {}
    for key in ("single_best", "combination_best"):
        for source in payload.get(key, []):
            if not isinstance(source, dict) or source.get("formula_id") is None:
                continue
            formula_id = str(source["formula_id"])
            entry = initial_fit_rows.setdefault(formula_id, {
                **source, "parameters": {}, "parameters_by_dataset": {},
                "dataset_fit_results": [],
            })
            dataset_name = source.get("dataset_name")
            parameters = source.get("parameters")
            if isinstance(parameters, dict):
                entry["parameters"].update(parameters)
                entry["parameters_by_dataset"].update(
                    source.get("parameters_by_dataset", parameters)
                )
            elif dataset_name is not None:
                entry["parameters"][str(dataset_name)] = parameters
                entry["parameters_by_dataset"][str(dataset_name)] = parameters
            if source.get("dataset_fit_results"):
                entry["dataset_fit_results"].extend(source["dataset_fit_results"])
            elif dataset_name is not None:
                entry["dataset_fit_results"].append(dict(source))
    if test_data is not None:
        test_X, test_y, _test_features, test_groups = test_data
        for row in initial_mean_r2_rows:
            source = initial_fit_rows.get(str(row.get("formula_id")))
            if source is None:
                continue
            test_metrics = _evaluate_validation_row(
                source, test_X, test_y, test_groups, dataset_groups,
                config["fitting"],
            )
            if test_metrics is not None:
                row["test_r2"] = test_metrics.get("mean_r2")
    # Persist the initial LM summary in validation-performance order so the
    # most promising formulas are immediately visible in the CSV.  Missing
    # validation values are placed last; the remaining keys make ties stable.
    def _descending_finite_value(row: dict, key: str) -> float:
        try:
            value = float(row.get(key))
        except (TypeError, ValueError):
            return -math.inf
        return value if math.isfinite(value) else -math.inf

    initial_mean_r2_rows.sort(key=lambda row: (
        -_descending_finite_value(row, "validation_mean_r2"),
        -_descending_finite_value(row, "mean_r2"),
        str(row.get("formula_id", "")),
    ))
    initial_mean_r2_path = output_dir / "initial_lm_mean_r2.csv"
    initial_mean_r2_fields = [
        "formula_id", "formula_family", "readable_expression",
        "mean_r2", "validation_mean_r2", "test_r2", "dataset_r2",
    ]
    initial_mean_r2_path.parent.mkdir(parents=True, exist_ok=True)
    with initial_mean_r2_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=initial_mean_r2_fields)
        writer.writeheader()
        for row in initial_mean_r2_rows:
            writer.writerow({
                **row,
                "dataset_r2": json.dumps(
                    row.get("dataset_r2", {}), ensure_ascii=False
                ),
            })
    payload["metadata"]["initial_lm_mean_r2"] = {
        "file": str(initial_mean_r2_path),
        "formula_count": len(initial_mean_r2_rows),
        "columns": initial_mean_r2_fields,
    }
    total_formula_count = len(candidates) + len(combinations)
    direct_fit_formula_count = len(direct_fit_formula_ids)
    not_direct_fit_formula_count = max(0, total_formula_count - direct_fit_formula_count)
    not_direct_fit_formula_percent = (
        100.0 * not_direct_fit_formula_count / total_formula_count
        if total_formula_count else 0.0
    )
    print(
        "\nWith current settings, total formulas generated: {}; formulas fitted directly: {}; "
        "formulas not fitted directly: {} ({:.2f}%)".format(
            total_formula_count,
            direct_fit_formula_count,
            not_direct_fit_formula_count,
            not_direct_fit_formula_percent,
        ),
        flush=True,
    )
    payload["metadata"].update(
        elapsed_seconds=time.time() - started,
        total_formula_count=total_formula_count,
        direct_fit_formula_count=direct_fit_formula_count,
        not_direct_fit_formula_count=not_direct_fit_formula_count,
        not_direct_fit_formula_percent=not_direct_fit_formula_percent,
        fitted_single_count=(len(payload["single_best"]) if reused_history or formula_pair_search
                             else sum(len(value[0]) for value in allocations.values())),
        fitted_combination_count=(len(payload["combination_best"]) if reused_history or formula_pair_search
                                  else sum(len(value[1]) for value in allocations.values())),
        training_structure_allocations=allocation_metadata,
        fit_mode=fit_mode,
        history_source=str(history_path) if reused_history else None,
        formula_pair_search=False,
        independent_segment_fitting=bool(formula_pair_search),
        segmentation_strategy=("fixed_breakpoint" if fixed_breakpoint_segmentation
                               else ("coarse_gp_fine" if coarse_gp_fine_segmentation
                                     else ("legacy" if piecewise_enabled else "disabled"))),
    )
    # PySR is run by scripts/run_pysr_experiment.py. The main search has no
    # PySR dependency; optional comparison fields remain empty in this run.
    pysr_validation_result = None
    pysr_test_result = None

    mode2_search_completed = False
    mode2_random_validation_result = None
    initial_lm_tree_payload = None
    initial_lm_tree_ranking_config = copy.deepcopy(ranking_config)
    if int(pipeline_mode) in (2, 3, 4) and ranking_config.get("enabled", False):
        # Preserve the exact pre-Mode-2 state for the one global random
        # permutation comparison.
        initial_lm_tree_payload = copy.deepcopy(payload)
        mode2_ablation_base = copy.deepcopy(initial_lm_tree_payload)
        run_mode2_surrogate_assisted_search(
            payload, config, ranking_config, X, y, dataset_groups, seed,
            output_dir, is_realdata_mat, piecewise_enabled, segmentation_config,
            n_jobs,
            pipeline_mode=pipeline_mode,
            selection_validation_data=validation_data,
            report_test_data=(test_data[0], test_data[1], test_data[3])
            if test_data is not None else None,
        )
        control_iterations = int(payload.get("metadata", {}).get(
            "mode2_iterations", payload.get("gp_metadata", {}).get("iterations", 0)))
        ablation_control_iterations = min(
            control_iterations, int(config.get("mode2", {}).get("max_iterations", 40)))
        if (not config.get("disable_all_ablations", False)
                and not is_realdata_mat):
            random_output_dir = output_dir / "mode2_random_permutation_ablation"
            random_payload = copy.deepcopy(mode2_ablation_base)
            shared_initial_constant_state = payload.get(
                "_mode2_initial_constant_state"
            )
            if not shared_initial_constant_state:
                raise RuntimeError(
                    "Initial tree-model constant-test cache is missing; cannot start a fair random ablation"
                )
            random_payload["_mode2_initial_constant_state"] = copy.deepcopy(
                shared_initial_constant_state
            )
            run_mode2_surrogate_assisted_search(
                random_payload, config, ranking_config, X, y, dataset_groups, seed,
                random_output_dir, is_realdata_mat, piecewise_enabled,
                segmentation_config, n_jobs, ranking_strategy="random",
                forced_iterations=ablation_control_iterations, pipeline_mode=pipeline_mode,
                selection_validation_data=validation_data,
                report_test_data=(test_data[0], test_data[1], test_data[3])
                if test_data is not None else None,
                shared_initial_constant_state=(
                    random_payload["_mode2_initial_constant_state"]
                ),
            )
            save_outputs(random_output_dir, random_payload)
            comparison = _mode2_compare_tree_random(
                payload, random_payload, output_dir
            )
            if validation_data is not None:
                mode2_random_validation_result = save_validation_results(
                    random_payload, validation_config, validation_data[0],
                    validation_data[1], validation_data[3], dataset_groups,
                    config["fitting"], random_output_dir,
                    _configured_data_profile(data_cfg),
                    "mode2_random_permutation_ablation", "random_permutation",
                )
            payload["mode2_ablation"] = {
                "enabled": True,
                "method": "random_permutation",
                "random_output_directory": str(random_output_dir),
                "comparison_json": str(output_dir / "tree_vs_random_comparison.json"),
                "comparison_csv": str(output_dir / "tree_vs_random_comparison.csv"),
                "control_iterations": control_iterations,
                "comparison": comparison,
            }
        mode2_search_completed = True
        ranking_config = dict(ranking_config)
        ranking_config["enabled"] = False
    ranking_ready = True
    if (ranking_config.get("enabled", False)
            and not ranking_config.get("ablation_enabled", False)):
        available_fit_results = len(payload.get("single_best", [])) + len(
            payload.get("combination_best", [])
        )
        minimum_training_samples = int(ranking_config.get("minimum_training_samples", 20))
        if available_fit_results < minimum_training_samples:
            ranking_ready = False
            partial_reuse = history_payload is not None
            ranking_config = dict(ranking_config)
            ranking_config["enabled"] = False
            payload["ranking_metadata"] = {
                "enabled": False,
                "skipped": True,
                "skip_reason": "insufficient_reusable_fit_results",
                "available_fit_results": available_fit_results,
                "minimum_training_samples": minimum_training_samples,
            }
            logging.warning(
                "Only %d reusable fit results are available, fewer than the %d required by the tree model; "
                "skipping the tree model and GP for this run and saving the current checkpoint",
                available_fit_results, minimum_training_samples,
            )
    if ranking_config.get("enabled", False):
        if ranking_config.get("ablation_enabled", False):
            payload["structure_r2_ranking"], payload["ranking_metadata"] = (
                rank_lm_results_for_ablation(payload, ranking_config)
            )
            logging.info("Generating Top-K directly using the LM fitting objective")
        else:
            payload["structure_r2_ranking"], payload["ranking_metadata"] = train_and_rank(
                payload, ranking_config, seed, output_dir
            )
        payload["ranking_metadata"]["feature_correlation_diagnostic"] = (
            early_feature_correlation_diagnostic
        )
        if ranking_config.get("ablation_enabled", False):
            print("Tree-model ablation: skipping the tree model and ranking by LM fitted R²", flush=True)
        else:
            print(
                "Tree-model validation recall (Top-{}): {}".format(
                    payload["ranking_metadata"].get("validation", {}).get("top_k", "N/A"),
                    payload["ranking_metadata"].get("validation", {}).get("top_k_overlap"),
                ),
                flush=True,
            )
        ranking_for_refit = payload["structure_r2_ranking"]
        ranking_config_for_refit = ranking_config
        if ranking_config.get("ablation_enabled", False):
            # The ablation ranking is derived from the already-fitted LM
            # results.  Reuse those rows directly; the ordinary path below
            # remains the only path that performs Top-K refitting.
            existing_fit_results = [
                *payload.get("single_best", []),
                *payload.get("combination_best", []),
            ]
            if ranking_config.get("refit_all_datasets", False):
                payload["top_ranked_refit_results"] = (
                    refit_ablation_top_ranked_formulas_all_datasets(
                        payload["structure_r2_ranking"], candidates, combinations,
                        X, y, config["fitting"], ranking_config, np_rng,
                        dataset_groups, n_jobs=n_jobs,
                        segmentation_config=segmentation_config if piecewise_enabled else None,
                    )
                )
            else:
                payload["top_ranked_refit_results"] = reuse_top_ranked_formula_fits(
                    payload["structure_r2_ranking"], existing_fit_results,
                    ranking_config,
                )
            payload["ranking_metadata"]["top_refit_reused"] = not ranking_config.get(
                "refit_all_datasets", False
            )
            payload["ranking_metadata"]["top_refit_reuse_source"] = (
                "all_configured_datasets"
                if ranking_config.get("refit_all_datasets", False)
                else "single_best_and_combination_best"
            )
        elif formula_pair_search:
            payload["top_ranked_refit_results"] = refit_top_ranked_formula_pairs(
                payload["structure_r2_ranking"], candidates, combinations, X, y,
                config["fitting"], ranking_config, np_rng, segmentation_config,
                n_jobs=n_jobs,
            )
        else:
            payload["top_ranked_refit_results"] = refit_top_ranked_formulas(
                ranking_for_refit, candidates, combinations, X, y,
                config["fitting"], ranking_config_for_refit, np_rng, dataset_groups,
                segmentation_config if piecewise_enabled else None,
                n_jobs=n_jobs,
            )
        if coarse_gp_fine_segmentation:
            global_top_results = payload["top_ranked_refit_results"]
            payload["top_ranked_refit_results"] = coarse_refit_top_ranked_formulas(
                ranking_for_refit, candidates, combinations, X, y,
                config["fitting"], ranking_config_for_refit, np_rng, segmentation_config,
                warm_start_results=global_top_results, n_jobs=n_jobs,
            )
        print_pre_gp_screen_results(
            payload["top_ranked_refit_results"],
            payload["metadata"].get("gp_preselection"),
            X=X if is_realdata_mat else None,
            data_mode=_configured_data_profile(data_cfg),
        )
        if is_realdata_mat:
            for row in payload["top_ranked_refit_results"]:
                _attach_piecewise_continuity_objective(
                    row, row.get("readable_expression", ""), X
                )
        # Diagnostic-only visualization of the formulas selected for Top-K
        # refitting.  It does not change ranking, refitting, or GP inputs.
        if not config.get("minimal_output_artifacts", False):
            payload["ranking_metadata"]["top_k_feature_correlation_diagnostic"] = (
                save_top_k_feature_correlation_diagnostic(
                    payload["top_ranked_refit_results"], ranking_config, output_dir
                )
            )
        training_formula_ids = {
            str(formula_id)
            for formula_id in payload["ranking_metadata"].get(
                "training_formula_ids", []
            )
        }
        print("\nWhether each Top-K formula is from the tree-model training set:", flush=True)
        for rank, row in enumerate(payload["top_ranked_refit_results"], start=1):
            formula_id = str(row.get("formula_id"))
            print(
                f"  #{rank:>3} {formula_id}: "
                f"{'Yes' if formula_id in training_formula_ids else 'No'}",
                flush=True,
            )
    save_checkpoint(output_dir, payload, "ranking_and_top_refit")
    def ranked_fit_formula(
        record, initial_values_by_dataset=None, initial_perturbation=0.0,
        warm_start_fraction=1.0, lm_initializations=None, fit_rng=None,
    ):
        """Refit a ranked formula using the normal segmentation policy."""
        active_rng = fit_rng if fit_rng is not None else np_rng
        if coarse_gp_fine_segmentation:
            _global_attempts, global_best = fit_formula_per_dataset(
                record, X, y, config["fitting"], active_rng, dataset_groups,
                initial_values_by_dataset=initial_values_by_dataset,
                initial_perturbation=initial_perturbation,
                warm_start_fraction=warm_start_fraction,
                lm_initializations=lm_initializations,
                fit_rng=active_rng,
            )
            global_parameters = next(
                iter((global_best.get("parameters_by_dataset") or {}).values()), None
            )
            _attempts, best = fit_coarse_piecewise_formula(
                record, X, y, config["fitting"], active_rng, segmentation_config,
                warm_start_parameters=global_parameters,
            )
            if not _result_not_worse(best, global_best):
                best = {**global_best, "segmentation_rejected": True,
                        "segmentation_rejection_reason": "coarse_r2_worse_than_global"}
        elif piecewise_enabled:
            _attempts, best = fit_piecewise_formula(
                record, X, y, config["fitting"], active_rng, segmentation_config
            )
        else:
            _attempts, best = fit_formula_per_dataset(
                record, X, y, config["fitting"], active_rng, dataset_groups,
                initial_values_by_dataset=initial_values_by_dataset,
                initial_perturbation=initial_perturbation,
                warm_start_fraction=warm_start_fraction,
                lm_initializations=lm_initializations,
                fit_rng=active_rng,
            )
        if is_realdata_mat:
            _attach_piecewise_continuity_objective(
                best, record["readable_expression"], X
            )
        return best

    constant_significance_preprocessed = False
    # Run constant-significance checks after Top-K refitting and before final
    # selection or log-term merging.
    if (not mode2_search_completed
            and ranking_ready
            and payload.get("top_ranked_refit_results")
            and not formula_pair_search
           ):
        processed_results, checks, zeroed_count, significance_metadata = (
            preprocess_constant_significance_results(
                payload["top_ranked_refit_results"], ranked_fit_formula, config,
                allow_single_variable=(is_realdata_mat and not mode2_binary_constraint),
                feature_count=X.shape[1], seed=seed,
                n_jobs=constant_significance_n_jobs,
                apply_pareto_filter=int(pipeline_mode) != 1,
            )
        )
        payload["top_ranked_refit_results"] = processed_results
        payload["top_ranked_constant_significance_checks"] = checks
        payload["metadata"].update(
            constant_significance_enabled=True,
            constant_significance_stage="before_final_pareto",
            constant_significance_max_r2_drop=significance_metadata["max_r2_drop"],
            constant_significance_allow_single_variable=is_realdata_mat,
            constant_significance_orders=significance_metadata["orders"],
            zeroed_constant_count=zeroed_count,
            top_ranked_initial_seed_selection=significance_metadata["selection"],
        )
        constant_significance_preprocessed = True
        print(
            f"\nConstant-significance test before final front complete: zeroed {zeroed_count}  constants; "
            f"non-dominated solutions retained after both zeroing orders {len(processed_results)} items; "
            f"maximum allowed R² drop={significance_metadata['max_r2_drop']}", flush=True
        )

    # Select the final formula set for the shared post-processing steps.
    postprocess_results = None
    postprocess_fit_formula = None
    postprocess_metadata = None
    postprocess_checks_prefix = None
    if (mode2_search_completed and int(pipeline_mode) == 3
            and payload.get("gp_pareto_front")):
        # Mode 3 owns the complete surrogate-assisted search, so its final
        # Pareto front is not routed through the legacy GP post-processing
        # branch below.  Run the existing log-term merge check only now,
        # after all Mode 3 exact evaluations and Pareto updates are complete.
        # The merge therefore cannot influence surrogate ranking, search
        # iterations, stopping, or HV comparisons.
        postprocess_results = payload.get("gp_pareto_front")
        postprocess_fit_formula = ranked_fit_formula
        postprocess_metadata = payload.setdefault("gp_metadata", {})
        postprocess_checks_prefix = "gp"
    elif (payload.get("gp_metadata", {}).get("enabled")
            and not mode2_search_completed):
        postprocess_results = payload.get("gp_pareto_front")
        postprocess_fit_formula = ranked_fit_formula
        postprocess_metadata = payload["gp_metadata"]
        postprocess_checks_prefix = "gp"
    elif (not mode2_search_completed
          and payload.get("top_ranked_refit_results")
          and not formula_pair_search):
        postprocess_results = payload["top_ranked_refit_results"]
        postprocess_fit_formula = ranked_fit_formula
        postprocess_metadata = payload["metadata"]
        postprocess_checks_prefix = "top_ranked"
    # A refit is allowed to discover an improvement, but it must never make
    # the persisted top-level results worse than the previous run.  Keep the
    # new run and its archive for auditability while retaining the better
    # historical result in the active payload.
    if history_payload is not None:
        for key in ("top_ranked_refit_results", "gp_pareto_front"):
            historical = history_payload.get(key, [])
            current = payload.get(key, [])
            old_best, new_best = _best_result_r2(historical), _best_result_r2(current)
            if old_best is not None and (new_best is None or new_best < old_best - 1e-12):
                payload[key] = historical
                payload["metadata"][f"{key}_restored_from_history"] = True
                payload["metadata"][f"{key}_historical_best_r2"] = old_best
    # Historical restoration can replace the list object, so refresh the
    # selected post-processing target before running either check.
    if postprocess_checks_prefix == "gp":
        postprocess_results = payload.get("gp_pareto_front")
    elif postprocess_checks_prefix == "top_ranked":
        postprocess_results = payload.get("top_ranked_refit_results")
    # Mode 2 already performs constant testing during search, so avoid
    # repeating it on that path.
    if (postprocess_results and not constant_significance_preprocessed
            and not (mode2_search_completed and int(pipeline_mode) == 3)):
        # Constant-significance testing is mandatory for every run.
        max_r2_drop = float(config.get("postprocessing", {}).get(
            "merge_log_max_r2_drop", 0.01
        ))
        zero_constant_initial_perturbation = float(
            config.get("fitting", {}).get(
                "zero_constant_initial_perturbation", 0.05
            )
        )
        zero_constant_lm_initializations = int(
            config.get("fitting", {}).get(
                "zero_constant_lm_initializations",
                config.get("fitting", {}).get("lm_initializations", 20),
            )
        )
        processed_results, checks, zeroed_count = (
            prune_insignificant_constants(
                postprocess_results, postprocess_fit_formula, max_r2_drop,
                zero_constant_initial_perturbation,
                zero_constant_lm_initializations,
                allow_single_variable=(is_realdata_mat and not mode2_binary_constraint),
                feature_count=X.shape[1],
                seed=seed,
                n_jobs=constant_significance_n_jobs,
            )
        )
        postprocess_results[:] = processed_results
        checks_key = f"{postprocess_checks_prefix}_constant_significance_checks"
        payload[checks_key] = checks
        postprocess_metadata["constant_significance_enabled"] = True
        postprocess_metadata["constant_significance_max_r2_drop"] = max_r2_drop
        postprocess_metadata["constant_significance_allow_single_variable"] = (
            is_realdata_mat
        )
        postprocess_metadata["zeroed_constant_count"] = zeroed_count
        print(
            f"\nConstant-significance test complete: zeroed {zeroed_count}  constants; "
            f"maximum allowed R² drop={max_r2_drop}", flush=True
        )
        for row in postprocess_results:
            print(
                f"{row['formula_id']}: zeroed constants="
                f"{row.get('zeroed_constants', [])}, "
                f"formula={row['readable_expression']}", flush=True
            )
    # Log-term merging deliberately runs after constant pruning, so its
    # refit evaluates the simplified expression rather than the pre-pruned one.
    if (postprocess_results
            and (not mode2_search_completed or int(pipeline_mode) == 3)):
        max_r2_drop = float(config.get("postprocessing", {}).get(
            "merge_log_max_r2_drop", 0.01
        ))
        log_merge_n_jobs = (
            n_jobs if int(pipeline_mode) == 3 and parallel_lm else 1
        )
        merged_results, merge_checks, merged_count = (
            merge_log_formulas(
                postprocess_results, postprocess_fit_formula, max_r2_drop,
                float(config.get("postprocessing", {}).get(
                    "log_merge_initial_perturbation", 0.05
                )),
                lm_initializations=int(config.get("fitting", {}).get(
                    "log_merge_lm_initializations",
                    config.get("fitting", {}).get("lm_initializations", 20),
                )),
                seed=seed,
                n_jobs=log_merge_n_jobs,
                )
            )
        postprocess_results[:] = merged_results
        checks_key = f"{postprocess_checks_prefix}_log_merge_checks"
        payload[checks_key] = merge_checks
        postprocess_metadata["merged_log_formula_count"] = merged_count
        postprocess_metadata["log_merge_n_jobs"] = log_merge_n_jobs
        postprocess_metadata["log_merge_checks"] = merge_checks
        for check in merge_checks:
            status = "retained" if check["accepted"] else "discarded"
            check_label = "GP log-merge check" if postprocess_checks_prefix == "gp" else "Log-merge check"
            print(
                f"\n{check_label} [{status}] {check['source_formula_id']}: "
                f"\n  Before merge: {check['original_expression']}"
                f"\n  After merge: {check['merged_expression']}"
                f"\n  R²: {check['original_r2_by_dataset']} -> "
                f"{check['merged_r2_by_dataset']}"
                )
    if payload.get("gp_pareto_front"):
        if ENABLE_FINAL_PARETO_SYMPY_SIMPLIFICATION:
            simplification = simplify_pareto_front_expressions(
                payload["gp_pareto_front"]
            )
            payload.setdefault("gp_metadata", {})[
                "sympy_simplification"
            ] = simplification
            print(
                "\nFinal Pareto formula SymPy simplification: "
                f"processed {simplification['input_count']} items, "
                f"simplified  {simplification['changed_count']} items, "
                f"skipped {simplification['skipped_count']} items",
                flush=True,
            )
        else:
            payload.setdefault("gp_metadata", {})[
                "sympy_simplification"
            ] = {"enabled": False, "reason": "temporarily_disabled"}
            print("\nFinal Pareto formula SymPy simplification: temporarily disabled.", flush=True)
    if config.get("require_integer_exponents",
                  config.get("gp", {}).get("require_integer_exponents", False)):
        exponent_filter = filter_all_formula_results(payload)
        payload.setdefault("metadata", {})["integer_exponent_filter"] = exponent_filter
        if "gp_metadata" in payload:
            payload["gp_metadata"]["integer_exponent_filter"] = (
                exponent_filter.get("gp_pareto_front", {})
            )
        print(
            "\nInteger-exponent formula filter: "
            + ", ".join(
                f"{key}  retaining {detail['kept_count']}/{detail['input_count']}"
                for key, detail in exponent_filter.items()
            )
        )
    if validation_data is not None:
        if output_dir.name == "mode2_random_permutation_ablation":
            experiment_variant = "mode2_random_permutation_ablation"
            ablation_name = "random_permutation"
        else:
            experiment_variant = "full_feature_baseline"
            ablation_name = None
        validation_result = save_validation_results(
            payload, validation_config, validation_data[0], validation_data[1],
            validation_data[3], dataset_groups, config["fitting"], output_dir,
            _configured_data_profile(data_cfg), experiment_variant, ablation_name,
        )
        if mode2_random_validation_result is not None:
            # Re-write both standalone validation results with the same
            # bounds used by the tree-vs-random comparison.  The bounds must
            # be computed from both evaluated histories; otherwise each
            # standalone plot is normalized in its own coordinate system.
            hv_fields, _ = _validation_hv_objectives(_configured_data_profile(data_cfg))
            shared_validation_rows = [
                row
                for result_item in (
                    validation_result, mode2_random_validation_result,
                )
                if result_item is not None
                for history_item in result_item.get("history", [])
                for row in history_item.get("front", [])
            ]
            shared_validation_rows.extend(
                row
                for result_item in (
                    validation_result, mode2_random_validation_result,
                )
                if result_item is not None
                for row in result_item.get("final_pareto_front", [])
            )
            shared_hv_bounds = _validation_hv_bounds(
                shared_validation_rows, hv_fields
            )
            validation_result = save_validation_results(
                payload, validation_config, validation_data[0], validation_data[1],
                validation_data[3], dataset_groups, config["fitting"], output_dir,
                _configured_data_profile(data_cfg), experiment_variant, ablation_name,
                hv_bounds=shared_hv_bounds,
            )
            random_output_dir = output_dir / "mode2_random_permutation_ablation"
            mode2_random_validation_result = save_validation_results(
                random_payload, validation_config, validation_data[0],
                validation_data[1], validation_data[3], dataset_groups,
                config["fitting"], random_output_dir,
                _configured_data_profile(data_cfg),
                "mode2_random_permutation_ablation", "random_permutation",
                hv_bounds=shared_hv_bounds,
            )
        test_config = copy.deepcopy(validation_config)
        test_config["output_directory"] = "test"
        test_result = save_validation_results(
            payload, test_config, test_data[0], test_data[1], test_data[3],
            dataset_groups, config["fitting"], output_dir,
            _configured_data_profile(data_cfg), experiment_variant, ablation_name,
            evaluation_label="test",
        )
        test_random_result = None
        if mode2_random_validation_result is not None:
            test_random_result = save_validation_results(
                random_payload, test_config, test_data[0], test_data[1],
                test_data[3], dataset_groups, config["fitting"],
                output_dir / "mode2_random_permutation_ablation",
                _configured_data_profile(data_cfg),
                "mode2_random_permutation_ablation", "random_permutation",
                evaluation_label="test",
            )
            test_fields, _ = _validation_hv_objectives(
                _configured_data_profile(data_cfg), "test"
            )
            shared_test_rows = [
                row
                for result_item in (test_result, test_random_result)
                for history_item in result_item.get("history", [])
                for row in history_item.get("front", [])
            ]
            shared_test_rows.extend(
                row
                for result_item in (test_result, test_random_result)
                for row in result_item.get("final_pareto_front", [])
            )
            shared_test_bounds = _validation_hv_bounds(shared_test_rows, test_fields)
            test_result = save_validation_results(
                payload, test_config, test_data[0], test_data[1], test_data[3],
                dataset_groups, config["fitting"], output_dir,
                _configured_data_profile(data_cfg), experiment_variant, ablation_name,
                hv_bounds=shared_test_bounds, evaluation_label="test",
            )
            test_random_result = save_validation_results(
                random_payload, test_config, test_data[0], test_data[1], test_data[3],
                dataset_groups, config["fitting"],
                output_dir / "mode2_random_permutation_ablation",
                _configured_data_profile(data_cfg),
                "mode2_random_permutation_ablation", "random_permutation",
                hv_bounds=shared_test_bounds, evaluation_label="test",
            )
            save_mode2_validation_comparison(
                test_result, test_random_result, output_dir,
                _configured_data_profile(data_cfg), hv_bounds=shared_test_bounds,
                evaluation_label="test", pysr_result=pysr_test_result,
            )
        payload["metadata"]["test"] = {
            "enabled": True,
            "split": validation_split_metadata,
            "result_files": test_result["files"],
        }
        payload["metadata"]["validation"]["result_files"] = validation_result["files"]
        payload["metadata"]["validation"]["experiment_variant"] = experiment_variant
        payload["metadata"]["validation"]["ablation_name"] = ablation_name
        payload["metadata"]["validation"]["iteration_count"] = len(
            validation_result["per_iteration"]
        )
        print(
            "Validation Pareto front saved: "
            f"{validation_result['files']['pareto_front_csv']}; "
            f"per-round HV: {validation_result['files']['hypervolume_csv']}",
            flush=True,
        )
        if mode2_random_validation_result is not None:
            validation_comparison = save_mode2_validation_comparison(
                validation_result, mode2_random_validation_result, output_dir,
                _configured_data_profile(data_cfg),
                hv_bounds=shared_hv_bounds if mode2_random_validation_result is not None
                else None,
                pysr_result=pysr_validation_result,
            )
            payload["metadata"]["validation"]["mode2_comparison"] = (
                validation_comparison
            )
            print(
                "Validation tree-model/random-ablation HV comparison plot saved: "
                f"{validation_comparison['hypervolume_plot']}",
                flush=True,
            )
    payload["metadata"].update(
        run_status="partial" if partial_reuse else "completed",
        run_completed=not partial_reuse,
    )
    save_outputs(output_dir, payload)
    if payload["structure_r2_ranking"]:
        metadata = payload["ranking_metadata"]
        validation = metadata.get("validation", {})
        if metadata.get("ablation_enabled", False):
            logging.info(
                "LM ablation ranking output: %s (ranked formulas=%d, Top-K=%d)",
                output_dir / "structure_r2_ranking.csv",
                metadata["ranked_formulas"], metadata["refit_top_k"],
            )
            print("\nTop 10 LM fitted R² rankings: ")
        else:
            logging.info(
                "Tree-model prediction output: %s (model=%s, training labels=%d, ranked formulas=%d, Spearman=%s)",
                output_dir / "structure_r2_ranking.csv", metadata["model_type"],
                metadata["training_samples"], metadata["ranked_formulas"],
                validation.get("spearman"),
            )
            print("\nTop tree-model predicted frequency-band percentile rankings 10: ")
        for row in payload["structure_r2_ranking"][:10]:
            print(
                f"#{row['predicted_rank']:>3}  score={row['ranking_score']:.6f}  "
                f"{row['formula_family']:<11} {row['formula_id']}: "
                f"{row['readable_expression']}"
            )
        if payload["top_ranked_refit_results"]:
            result_label = (
                "LM ablation top formulas" if metadata.get("ablation_enabled", False)
                else ("Mode 3 initial top formulas" if metadata.get("pipeline_mode") == 3
                      else "Mode 2 initial top formulas") if metadata.get("mode2")
                else "Tree-model top formulas"
            )
            print(f"\n{result_label} per-dataset independent fit results: ")
            for row in payload["top_ranked_refit_results"]:
                print(f"\n#{row['predicted_rank']:>3} {row['formula_id']}: "
                      f"{row['readable_expression']}")
                for dataset in row["dataset_fit_results"]:
                    names = dataset.get("parameter_names") or []
                    values = dataset.get("parameters")
                    parameters = (dict(zip(names, values)) if values is not None else None)
                    r2 = dataset.get("r2")
                    r2_text = f"{r2:.8f}" if r2 is not None else "None"
                    print(
                        f"    {dataset['dataset_name']}: R²={r2_text}, "
                        f"parameters={parameters}, converged={dataset.get('converged')}, "
                        f"valid samples={dataset.get('valid_samples')}, "
                        f"failure reason={dataset.get('failure_reason')}"
                    )
        print(f"\nFull ranking: {output_dir / 'structure_r2_ranking.csv'}")
        print(f"Single-formula ranking: {output_dir / 'single_formula_r2_ranking.csv'}")
        print(f"Combination-formula ranking: {output_dir / 'combination_formula_r2_ranking.csv'}")
        print(f"Top-formula per-dataset fits:  {output_dir / 'top_ranked_dataset_fits.csv'}")
    if payload.get("gp_metadata", {}).get("enabled"):
        front_label = (("Mode 3 Pareto front from complexity-layered surrogate-assisted search:"
                        if payload.get("metadata", {}).get("pipeline_mode") == 3
                        else "Mode 2 surrogate-assisted Pareto front:")
                       if payload.get("metadata", {}).get("mode2")
                       else "Lightweight GP Pareto front:")
        print(f"\n{front_label}")
        for row in payload.get("gp_pareto_front", []):
            dataset_r2 = ", ".join(
                f"{dataset.get('dataset_name', 'dataset')}="
                f"{dataset.get('r2'):.8f}"
                for dataset in row.get("dataset_fit_results", [])
                if dataset.get("r2") is not None
            ) or "no per-dataset R² available"
            secondary_metric = (
                f"minimum R²={row.get('min_r2')!s:<12} "
                if _is_measured_pathloss_profile(data_cfg)
                else f"R² standard deviation={row.get('r2_std')!s:<12} "
            )
            difference_metric = (
                f"breakpoint prediction difference={row.get('segment_prediction_difference')!s:<12} "
                if _is_measured_pathloss_profile(data_cfg) else ""
            )
            print(
                f"#{row['crowding_rank']:>3} crowding={row.get('crowding_distance_label')!s:<8} "
                f"mean R²={row.get('mean_r2')!s:<12} "
                f"{secondary_metric}"
                f"{difference_metric}"
                f"complexity={row['complexity']:<4} {row['readable_expression']}"
            )
            print(f"    R² by dataset: {dataset_r2}")
        print(f"Pareto front results: {output_dir / 'pareto_front.csv'}")
        print(f"GP Pareto per-dataset fits: : {output_dir / 'pareto_dataset_fits.csv'}")
    if payload.get("metadata", {}).get("mode2"):
        metadata = payload["metadata"]
        generated_count = int(metadata.get(
            "total_formula_count",
            len(payload.get("single_candidates", []))
            + len(payload.get("binary_combinations", [])),
        ))
        initial_count = int(metadata.get("mode2_initial_training_formula_count", 0))
        later_count = int(metadata.get("mode2_subsequent_formula_count", 0))
        total_count = int(metadata.get(
            "mode2_total_used_formula_count", initial_count + later_count
        ))
        print(
            f"\nMode 2 initial formulas generated: {generated_count}; "
            f"formulas used: training set {initial_count} + "
            f"subsequent formulas {later_count} = total {total_count}"
        )
    logging.info("Results saved to %s", output_dir)
    return payload



_PYSR_API_NAMES = {'_apply_pysr_reproducibility_mode',
    '_pysr_structure_key',
    '_pysr_has_only_integer_exponents',
    '_pysr_power_exponent_has_numeric_constant',
    '_pysr_parameter_template',
    '_safe_pysr_expression',
    '_pysr_extra_sympy_mappings',
    '_pysr_lambdify_modules',
    '_pysr_has_x1_leaf_with_log_ancestor',
    '_pysr_nonlinear_unary_operator_names',
    '_pysr_has_nested_nonlinear_operators',
    '_pysr_has_division_by_nonlinear_structure',
    '_pysr_max_nonconstant_additive_terms',
    '_pysr_structural_penalty',
    '_pysr_nonconstant_additive_term_count',
    '_pysr_operator_name',
    '_pysr_structural_loss_function',
    '_pysr_template_structural_loss_function',
    '_pysr_prepare_equations_for_output',
    '_read_pysr_progress_log',
    '_pysr_equations_for_ipc',
    '_pysr_template_expression_for_display',
    '_pysr_fit_payload',
    '_pysr_fit_worker',
    '_run_pysr_fit_with_watchdog',
    '_pysr_metric_pair',
    '_pysr_evaluation_metrics',
    '_evaluate_pysr_expression_on_split',
    '_pysr_independent_validation_rows',
    '_pysr_independent_validation_result',
    '_save_pysr_pareto_front_csv',
    '_evaluate_frozen_pysr_front_on_test',
    '_run_pysr_shared_template_baseline',
    'run_pysr_baseline',
    'run_pysr_piecewise_baseline',
    'run_pysr_piecewise_shared_template_baseline',
    '_pysr_hv_window_front',
    '_pysr_front_for_hv_reporting',
    '_pysr_cache_signature',
    '_pysr_cache_seed_is_valid',
    '_copy_pysr_cache_to_output',
    '_save_pysr_output_to_cache',
    '_load_pysr_evaluation_result',
    '_attach_pysr_search_history',
}

def _load_pysr_module():
    from importlib import import_module
    module_name = (
        f"{__package__}.pysr.baseline"
        if __package__ else "prosee.symbolic.core.pysr.baseline"
    )
    return import_module(module_name)

def _make_pysr_delegate(name):
    def delegate(*args, **kwargs):
        return getattr(_load_pysr_module(), name)(*args, **kwargs)
    delegate.__name__ = name
    delegate.__qualname__ = name
    delegate.__module__ = __name__
    return delegate

for _pysr_name in _PYSR_API_NAMES:
    globals()[_pysr_name] = _make_pysr_delegate(_pysr_name)

def __getattr__(name):
    if name == "_PySRProgressLoggerSpec":
        return getattr(_load_pysr_module(), name)
    raise AttributeError(name)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--serial-lm", action="store_true",
        help="Use the original serial LM path and preserve the original random-number call order",
    )
    args = parser.parse_args()
    # Use the established full experiment protocol directly.

    config = load_config(args.config, pipeline_mode=5)
    # Validation is required for LM early stopping and model selection.
    if args.serial_lm:
        n_jobs = 1
    else:
        n_jobs = 24
    logging.basicConfig(level=getattr(logging, config.get("log_level", "INFO").upper()),
                        format="%(asctime)s %(levelname)s %(message)s")
    run_mode4_paper_experiment(
        args.config.resolve(), n_jobs=n_jobs, serial_lm=args.serial_lm,
        experiment_mode=5, retrain_interval=5, seed_count=10,
        target_iterations=40,
    )


if __name__ == "__main__":
    main()
