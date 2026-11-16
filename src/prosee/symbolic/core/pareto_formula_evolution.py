"""Small formula-level GP used after formula fitting."""
from __future__ import annotations

import math
import os
import random
import itertools
import itertools
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence

import numpy as np
import sympy as sp

from structure_screen.tree_search_ops import complexity, nodes_with_paths, replace_at_path, valid

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - progress remains optional
    def tqdm(iterable, **_kwargs):
        return iterable


def _formula_record(expr: sp.Expr, formula_id: str) -> dict:
    return {
        "id": formula_id,
        "readable_expression": str(expr),
        "sympy_expression": sp.srepr(expr),
        "structure_type": "lightweight_gp",
        "parameter_symbols": sorted(
            str(symbol) for symbol in expr.free_symbols
            if str(symbol).startswith(("a", "c", "p"))
        ),
    }


def _has_log_abs_x1(expr: sp.Expr) -> bool:
    """Whether ``expr`` contains log(Abs(f(x1))) for the real-data GP rule."""
    x1 = sp.Symbol("x1")
    return any(
        node.func is sp.log
        and len(node.args) == 1
        and node.args[0].func is sp.Abs
        and node.args[0].has(x1)
        for node in sp.preorder_traversal(expr)
    )


def _valid_child(expr: sp.Expr, cfg: dict, feature_count: int) -> bool:
    allowed = {sp.Symbol(f"x{i + 1}") for i in range(feature_count)}
    used = expr.free_symbols & allowed
    if not used:
        return False
    if any(str(symbol).startswith("x") and symbol not in allowed
           for symbol in expr.free_symbols):
        return False
    return valid(
        expr,
        max_depth=cfg.get("max_depth"),
        max_nodes=int(cfg.get("max_complexity", 96)),
    )


def _contains_trigonometric(expr: sp.Expr) -> bool:
    """Whether an expression contains at least one sin/cos node."""
    return any(
        node.func in (sp.sin, sp.cos)
        for node in sp.preorder_traversal(expr)
    )


def _contains_all_variables(expr: sp.Expr, feature_count: int) -> bool:
    """Return whether an expression contains every input variable."""
    required = {sp.Symbol(f"x{i + 1}") for i in range(feature_count)}
    return required <= expr.free_symbols


_NONLINEAR_OPERATOR_FUNCS = (sp.log, sp.sin, sp.cos, sp.exp, sp.Abs)


def _nonlinear_operator_constants(expr: sp.Expr) -> set[sp.Symbol]:
    """Return fitted constants associated with nonlinear operator terms.

    This protects multiplicative coefficients immediately before a nonlinear
    operator, such as ``c0*log(x1)``. Constants inside an operator argument,
    such as ``log(c1*x1)``, remain eligible for zeroing.
    """
    protected = set()
    for term in sp.Add.make_args(expr):
        nonlinear_nodes = [
            node for node in sp.preorder_traversal(term)
            if node.func in _NONLINEAR_OPERATOR_FUNCS
        ]
        if not nonlinear_nodes:
            continue
        if term.func is sp.Mul:
            for factor in term.args:
                if not any(factor is node for node in nonlinear_nodes):
                    protected.update(_parameter_symbols(factor))
    return protected


def prune_insignificant_constants(
    results: Sequence[dict], fit_formula: Callable[[dict], dict],
    max_r2_drop: float, initial_perturbation: float = 0.05,
    lm_initializations: int | None = None,
    allow_single_variable: bool = False,
    test_reverse_order: bool = False,
    feature_count: int | None = None,
    seed: int | None = None,
    n_jobs: int | None = 1,
    require_log_abs_x1: bool = False,
) -> tuple[list[dict], list[dict], int]:
    """Enumerate unchanged/zero constant assignments after refitting.

    Multi-variable formulas retain the historical guard against collapsing to
    one variable. A genuinely one-variable data mode can opt out because a
    one-variable result is its intended model class. Every eligible constant
    is tested either unchanged or set to zero.

    When an expression contains at least two top-level additive terms with a
    protected coefficient before a nonlinear operator, each eligible term is
    also refitted as an additional candidate.  Such an isolated term must
    retain every input variable; it supplements, rather than replaces, the
    original expression.
    """
    # Formula-level parallelism is safe: each formula starts from its own
    # original fit. Each formula's four-state combinations are evaluated
    # independently, so no accepted assignment changes another test.
    if n_jobs is None:
        n_jobs = 1
    n_jobs = int(n_jobs)
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs for the constant-significance test must be 1, a positive integer, or -1")
    if n_jobs != 1 and len(results) > 1:
        workers = (
            min(len(results), 24, max(1, (os.cpu_count() or 2) - 1))
            if n_jobs < 0 else min(len(results), n_jobs)
        )

        def run_one(index_and_item):
            index, item = index_and_item
            item_seed = None if seed is None else int(seed) + index
            return prune_insignificant_constants(
                [item], fit_formula, max_r2_drop, initial_perturbation,
                lm_initializations, allow_single_variable, test_reverse_order,
                feature_count, item_seed, n_jobs=1,
                require_log_abs_x1=require_log_abs_x1,
            )

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            parts = list(executor.map(run_one, enumerate(results)))
        updated = [row for part in parts for row in part[0]]
        checks = [check for part in parts for check in part[1]]
        zeroed = sum(part[2] for part in parts)
        return updated, checks, zeroed

    updated_results = []
    all_checks = []
    total_zeroed = 0

    local_rng = np.random.default_rng(seed) if seed is not None else None

    def invoke_fit(record, **kwargs):
        if local_rng is None:
            return fit_formula(record, **kwargs)
        try:
            return fit_formula(record, fit_rng=local_rng, **kwargs)
        except TypeError as exc:
            if "fit_rng" not in str(exc):
                raise
            return fit_formula(record, **kwargs)

    def dataset_r2(fit: dict) -> dict:
        rows = fit.get("dataset_fit_results") or [fit]
        return {
            row.get("dataset_name", "dataset_1"): row.get("r2")
            for row in rows
        }

    def parameter_values_by_dataset(fit: dict) -> dict[str, dict[str, float]]:
        rows = fit.get("dataset_fit_results", []) or [fit]
        values_by_dataset = {}
        for row in rows:
            names = row.get("parameter_names") or []
            values = row.get("parameters")
            if values is None or len(names) != len(values):
                continue
            values_by_dataset[row.get("dataset_name", "dataset_1")] = {
                str(name): float(value)
                for name, value in zip(names, values)
                if math.isfinite(float(value))
            }
        return values_by_dataset

    for item in tqdm(
        results,
        total=len(results),
        desc="Constant-significance test",
        unit="formula",
    ):
        original_expression = sp.sympify(item["readable_expression"])
        orders = [("forward", sorted(_parameter_symbols(original_expression), key=str))]
        if test_reverse_order and len(orders[0][1]) > 1:
            orders.append(("reverse", list(reversed(orders[0][1]))))

        for order_name, symbols in orders:
            current_expression = original_expression
            current_fit = dict(item)
            formula_checks = []
            zeroed_names = []
            formula_id = (
                item["formula_id"] if order_name == "forward"
                else f"{item['formula_id']}_zero_order_reverse"
            )

            # Split protected nonlinear addends from the original expression
            # before sequential constant zeroing can alter their arguments.
            # Otherwise a successful zero such as c1_left -> 0 changes the
            # candidate that is later isolated and the original addend is
            # lost before it ever reaches the GP seed pool.
            addends = list(sp.Add.make_args(original_expression))
            protected_addends = [
                term for term in addends
                if _nonlinear_operator_constants(term)
            ]
            if feature_count is not None and len(protected_addends) >= 2:
                for addend_index, addend in enumerate(protected_addends, start=1):
                    split_base = {
                        "formula_id": formula_id,
                        "source_formula_id": item["formula_id"],
                        "operation": "isolate_protected_nonlinear_addend",
                        "term_index": addend_index,
                        "source_expression": str(original_expression),
                        "isolated_expression": str(addend),
                        "accepted": False,
                    }
                    if not _contains_all_variables(addend, feature_count):
                        check = {
                            **split_base, "skipped": True,
                            "failure_reason": "isolated_term_missing_required_variable",
                        }
                        formula_checks.append(check)
                        all_checks.append(check)
                        continue
                    if require_log_abs_x1 and not _has_log_abs_x1(addend):
                        check = {
                            **split_base, "accepted": False, "skipped": True,
                            "failure_reason": "missing_required_log_abs_x1",
                        }
                        formula_checks.append(check)
                        all_checks.append(check)
                        continue
                    split_formula_id = f"{formula_id}_isolated_term_{addend_index}"
                    split_failure = None
                    try:
                        split_fit = invoke_fit(
                            _formula_record(addend, split_formula_id),
                            initial_values_by_dataset=parameter_values_by_dataset(item),
                            initial_perturbation=initial_perturbation,
                            warm_start_fraction=0.5,
                            lm_initializations=lm_initializations,
                        )
                    except Exception as exc:
                        split_fit = None
                        split_failure = f"{type(exc).__name__}: {exc}"
                    split_fit = split_fit or {"dataset_fit_results": [], "r2": None}
                    split_r2 = dataset_r2(split_fit)
                    accepted = bool(split_r2) and all(
                        value is not None and math.isfinite(float(value))
                        for value in split_r2.values()
                    )
                    check = {
                        **split_base, "accepted": accepted, "skipped": False,
                        "r2_by_dataset": split_r2, "failure_reason": split_failure,
                    }
                    formula_checks.append(check)
                    all_checks.append(check)
                    if accepted:
                        updated_results.append({
                            **item,
                            **split_fit,
                            "formula_id": split_formula_id,
                            "source_formula_id": item["formula_id"],
                            "constant_significance_order": order_name,
                            "readable_expression": str(addend),
                            "sympy_expression": sp.srepr(addend),
                            "complexity": complexity(addend),
                            "isolated_protected_nonlinear_addend": True,
                            "isolated_term_index": addend_index,
                            "constant_significance_tests": [check],
                            "zeroed_constants": [],
                        })

            # Enumerate unchanged/zero for every eligible constant.
            # Protected nonlinear coefficients retain the historical skip rule.
            protected_symbols = _nonlinear_operator_constants(original_expression)
            eligible_symbols = list(symbols)
            baseline_r2 = dataset_r2(item)
            accepted_candidates = []
            for assignment in itertools.product(
                (None, 0), repeat=len(eligible_symbols)
            ):
                replacements = {
                    symbol: sp.Integer(value)
                    for symbol, value in zip(eligible_symbols, assignment)
                    if value is not None
                }
                candidate_expression = original_expression.xreplace(replacements)
                changed = {
                    str(symbol): ("unchanged" if value is None else int(value))
                    for symbol, value in zip(eligible_symbols, assignment)
                }
                changed_values = {
                    name: value for name, value in changed.items()
                    if value != "unchanged"
                }
                zeroed = [name for name, value in changed.items() if value == 0]
                suffix = "_".join(
                    f"{name}_{'keep' if value is None else value}"
                    for name, value in zip(eligible_symbols, assignment)
                ) or "unchanged"
                candidate_id = (
                    formula_id if not changed_values
                    else f"{formula_id}_constant_{suffix}"
                )
                check_base = {
                    "formula_id": candidate_id,
                    "source_formula_id": item["formula_id"],
                    "constant_significance_order": order_name,
                    "operation": "exhaustive_zero_subset",
                    "constant_assignments": changed,
                    "before_expression": str(original_expression),
                    "before_r2_by_dataset": baseline_r2,
                    "max_r2_drop": max_r2_drop,
                }
                zeroed_protected = [
                    str(symbol) for symbol, value in zip(eligible_symbols, assignment)
                    if value == 0 and symbol in protected_symbols
                ]
                if zeroed_protected:
                    check = {
                        **check_base, "after_expression": str(candidate_expression),
                        "after_r2_by_dataset": {}, "r2_drop_by_dataset": {},
                        "accepted": False, "skipped": True,
                        "failure_reason": "constant_before_nonlinear_operator",
                    }
                    formula_checks.append(check)
                    all_checks.append(check)
                    continue
                remaining_variables = {
                    node for node in candidate_expression.free_symbols
                    if str(node).startswith("x")
                }
                if not allow_single_variable and len(remaining_variables) <= 1:
                    check = {**check_base, "after_expression": str(candidate_expression),
                             "after_r2_by_dataset": {}, "r2_drop_by_dataset": {},
                             "accepted": False,
                             "failure_reason": "zeroing_would_leave_at_most_one_variable"}
                    formula_checks.append(check)
                    all_checks.append(check)
                    continue
                if require_log_abs_x1 and not _has_log_abs_x1(candidate_expression):
                    check = {**check_base, "after_expression": str(candidate_expression),
                             "after_r2_by_dataset": {}, "r2_drop_by_dataset": {},
                             "accepted": False,
                             "failure_reason": "zeroing_would_break_log_abs_x1_constraint"}
                    formula_checks.append(check)
                    all_checks.append(check)
                    continue
                candidate_fit, candidate_failure = None, None
                if candidate_expression.has(sp.nan, sp.zoo, sp.oo, -sp.oo, sp.I):
                    candidate_failure = "invalid_symbolic_value_after_constant_assignment"
                elif not changed_values:
                    candidate_fit = item
                else:
                    try:
                        candidate_fit = invoke_fit(
                            _formula_record(candidate_expression, candidate_id),
                            initial_values_by_dataset=parameter_values_by_dataset(item),
                            initial_perturbation=initial_perturbation,
                            warm_start_fraction=0.5,
                            lm_initializations=lm_initializations,
                        )
                    except Exception as exc:
                        candidate_failure = f"{type(exc).__name__}: {exc}"
                candidate_fit = candidate_fit or {"dataset_fit_results": [], "r2": None}
                after_r2 = dataset_r2(candidate_fit)
                drops = {
                    name: (None if before is None or after_r2.get(name) is None
                           else float(before) - float(after_r2[name]))
                    for name, before in baseline_r2.items()
                }
                accepted = bool(drops) and all(
                    value is not None and value <= max_r2_drop
                    for value in drops.values()
                )
                check = {**check_base, "after_expression": str(candidate_expression),
                         "after_r2_by_dataset": after_r2, "r2_drop_by_dataset": drops,
                         "accepted": accepted, "failure_reason": candidate_failure}
                formula_checks.append(check)
                all_checks.append(check)
                if accepted:
                    accepted_candidates.append((candidate_expression, candidate_fit,
                                                candidate_id, changed, zeroed))

            # Preserve every accepted assignment for downstream Pareto filtering.
            for candidate_expression, candidate_fit, candidate_id, changed, zeroed in accepted_candidates:
                updated = {
                    **item, **candidate_fit,
                    "formula_id": candidate_id,
                    "source_formula_id": item["formula_id"],
                    "constant_significance_order": order_name,
                    "readable_expression": str(candidate_expression),
                    "sympy_expression": sp.srepr(candidate_expression),
                    "complexity": complexity(candidate_expression),
                    "constant_assignments": changed,
                    "constant_significance_tests": formula_checks,
                    "zeroed_constants": zeroed,
                }
                updated_results.append(updated)
                total_zeroed += len(zeroed)
            continue

            # Zeroing is sequential: an accepted zero immediately updates both
            # the expression and fit used by the next constant test.  Every
            # order starts again from the original expression and original fit.
            for symbol in symbols:
                if symbol not in current_expression.free_symbols:
                    continue
                check_base = {
                    "formula_id": formula_id,
                    "source_formula_id": item["formula_id"],
                    "constant_significance_order": order_name,
                    "constant": str(symbol),
                    "before_expression": str(current_expression),
                    "before_r2_by_dataset": dataset_r2(current_fit),
                    "max_r2_drop": max_r2_drop,
                }
                if symbol in _nonlinear_operator_constants(current_expression):
                    check = {**check_base, "after_expression": str(current_expression),
                             "after_r2_by_dataset": {}, "r2_drop_by_dataset": {},
                             "accepted": False, "skipped": True,
                             "failure_reason": "constant_before_nonlinear_operator"}
                    formula_checks.append(check)
                    all_checks.append(check)
                    continue
                candidate_expression = current_expression.xreplace({symbol: sp.Integer(0)})
                remaining_variables = {
                    node for node in candidate_expression.free_symbols
                    if str(node).startswith("x")
                }
                if not allow_single_variable and len(remaining_variables) <= 1:
                    check = {**check_base, "after_expression": str(candidate_expression),
                             "after_r2_by_dataset": {}, "r2_drop_by_dataset": {},
                             "accepted": False, "skipped": True,
                             "failure_reason": "zeroing_would_leave_at_most_one_variable"}
                    formula_checks.append(check)
                    all_checks.append(check)
                    continue
                if require_log_abs_x1 and not _has_log_abs_x1(candidate_expression):
                    check = {**check_base, "after_expression": str(candidate_expression),
                             "after_r2_by_dataset": {}, "r2_drop_by_dataset": {},
                             "accepted": False, "skipped": True,
                             "failure_reason": "zeroing_would_break_log_abs_x1_constraint"}
                    formula_checks.append(check)
                    all_checks.append(check)
                    continue
                candidate_fit, candidate_failure = None, None
                if candidate_expression.has(sp.nan, sp.zoo, sp.oo, -sp.oo, sp.I):
                    candidate_failure = "invalid_symbolic_value_after_zeroing"
                else:
                    try:
                        candidate_fit = invoke_fit(
                            _formula_record(candidate_expression, f"{formula_id}_zero_{symbol}"),
                            initial_values_by_dataset=parameter_values_by_dataset(current_fit),
                            initial_perturbation=initial_perturbation,
                            warm_start_fraction=0.5,
                            lm_initializations=lm_initializations,
                        )
                    except Exception as exc:  # one zeroing test must not stop GP
                        candidate_failure = f"{type(exc).__name__}: {exc}"
                candidate_fit = candidate_fit or {"dataset_fit_results": [], "r2": None}
                after_r2 = dataset_r2(candidate_fit)
                drops = {
                    name: (None if before is None or after_r2.get(name) is None
                           else float(before) - float(after_r2[name]))
                    for name, before in check_base["before_r2_by_dataset"].items()
                }
                accepted = bool(drops) and all(
                    value is not None and value <= max_r2_drop
                    for value in drops.values()
                )
                check = {**check_base, "after_expression": str(candidate_expression),
                         "after_r2_by_dataset": after_r2, "r2_drop_by_dataset": drops,
                         "accepted": accepted, "failure_reason": candidate_failure}
                formula_checks.append(check)
                all_checks.append(check)
                if accepted:
                    current_expression = candidate_expression
                    current_fit = candidate_fit
                    zeroed_names.append(str(symbol))
                    total_zeroed += 1

            if zeroed_names:
                mean_r2, r2_std = _r2_summary(
                    row.get("r2")
                    for row in current_fit.get("dataset_fit_results", [])
                )
                current_fit["mean_r2"] = mean_r2
                current_fit["r2_std"] = r2_std
            updated = {
                **item,
                **current_fit,
                "formula_id": formula_id,
                "source_formula_id": item["formula_id"],
                "constant_significance_order": order_name,
                "readable_expression": str(current_expression),
                "sympy_expression": sp.srepr(current_expression),
                "complexity": complexity(current_expression),
                "constant_significance_tests": formula_checks,
                "zeroed_constants": zeroed_names,
            }
            updated_results.append(updated)
    return updated_results, all_checks, total_zeroed


def filter_nondominated_initial_seeds(
    results: Sequence[dict], use_min_r2: bool = False,
    use_piecewise_difference: bool = False,
    use_pooled_r2: bool = False,
) -> tuple[list[dict], dict]:
    """Keep only non-dominated zeroed candidates for the GP seed pool."""
    candidates = []
    for row in results:
        dataset_rows = row.get("dataset_fit_results", []) or [row]
        r2_values = _finite_r2_values(item.get("r2") for item in dataset_rows)
        mean_r2, r2_std = _r2_summary(r2_values)
        if not math.isfinite(mean_r2) or not math.isfinite(r2_std):
            continue
        candidate = dict(row)
        candidate["min_r2"] = min(r2_values) if r2_values else -1e100
        pooled_r2 = candidate.get("pooled_r2", candidate.get("r2", mean_r2))
        try:
            pooled_r2 = float(pooled_r2)
        except (TypeError, ValueError):
            pooled_r2 = -1e100
        candidate["_seed_objectives"] = (
            pooled_r2 if use_pooled_r2 else mean_r2,
            0.0 if use_pooled_r2 else (candidate["min_r2"] if use_min_r2 else r2_std),
            ((lambda value: float(value) if value is not None and math.isfinite(float(value))
              else math.inf)(candidate.get("segment_prediction_difference", math.inf))
             if use_piecewise_difference else None),
            complexity(sp.sympify(row["readable_expression"])),
        )
        candidates.append(candidate)

    unique = {}
    for candidate in candidates:
        key = sp.srepr(sp.sympify(candidate["readable_expression"]))
        previous = unique.get(key)
        if previous is None:
            unique[key] = candidate
            continue
        previous_objectives = previous["_seed_objectives"]
        candidate_objectives = candidate["_seed_objectives"]
        candidate_quality = (candidate_objectives[0], candidate_objectives[1],
                             -candidate_objectives[2] if use_piecewise_difference else 0.0)
        previous_quality = (previous_objectives[0], previous_objectives[1],
                            -previous_objectives[2] if use_piecewise_difference else 0.0)
        if not use_min_r2 and not use_pooled_r2:
            candidate_quality = (candidate_objectives[0], -candidate_objectives[1],
                                 -candidate_objectives[2] if use_piecewise_difference else 0.0)
            previous_quality = (previous_objectives[0], -previous_objectives[1],
                                -previous_objectives[2] if use_piecewise_difference else 0.0)
        if candidate_quality > previous_quality:
            unique[key] = candidate
    candidates = list(unique.values())

    def dominates(left: dict, right: dict) -> bool:
        left_r2, left_secondary, left_difference, left_complexity = left["_seed_objectives"]
        right_r2, right_secondary, right_difference, right_complexity = right["_seed_objectives"]
        secondary_better_or_equal = (
            True if use_pooled_r2 else
            left_secondary >= right_secondary if use_min_r2
            else left_secondary <= right_secondary
        )
        secondary_strictly_better = (
            False if use_pooled_r2 else
            left_secondary > right_secondary if use_min_r2
            else left_secondary < right_secondary
        )
        difference_better_or_equal = (
            left_difference <= right_difference if use_piecewise_difference else True
        )
        difference_strictly_better = (
            left_difference < right_difference if use_piecewise_difference else False
        )
        return (
            left_r2 >= right_r2 and secondary_better_or_equal
            and (not use_piecewise_difference or difference_better_or_equal)
            and left_complexity <= right_complexity
            and (left_r2 > right_r2 or secondary_strictly_better
                 or (use_piecewise_difference and difference_strictly_better)
                 or left_complexity < right_complexity)
        )

    front = [
        candidate for candidate in candidates
        if not any(dominates(other, candidate)
                   for other in candidates if other is not candidate)
    ]
    front.sort(
        key=lambda row: (
            (row["_seed_objectives"][2] if use_piecewise_difference else
             (-row["_seed_objectives"][1] if use_min_r2 and not use_pooled_r2
              else row["_seed_objectives"][1])),
            row["_seed_objectives"][3],
            -row["_seed_objectives"][0], row["readable_expression"],
        )
    )
    for candidate in front:
        candidate.pop("_seed_objectives", None)
    if use_pooled_r2:
        selection_objectives = ["pooled_r2_max"]
        if use_piecewise_difference:
            selection_objectives.append("segment_prediction_difference_min")
        selection_objectives.append("complexity_min")
    elif use_piecewise_difference:
        selection_objectives = ["mean_r2_max", "min_r2_max",
                                "segment_prediction_difference_min", "complexity_min"]
    elif use_min_r2:
        selection_objectives = ["mean_r2_max", "min_r2_max", "complexity_min"]
    else:
        selection_objectives = ["mean_r2_max", "r2_std_min", "complexity_min"]
    return front, {
        "input_count": len(results),
        "deduplicated_count": len(candidates),
        "nondominated_count": len(front),
        "objectives": selection_objectives,
    }


def _stratified_initial_rows(rows: Sequence[dict], top_k: int,
                             layer_count: int) -> tuple[list[dict], dict]:
    """Select an equal quota from balanced complexity layers.

    Rows are first ordered by structural complexity and split into balanced
    layers. Each layer contributes the same number of formulas. If ``top_k``
    is not divisible by the number of non-empty layers, the quota is rounded
    to the nearest integer, so the final count is the closest reachable
    multiple of the layer count.
    """
    if not rows or top_k <= 0:
        return [], {"enabled": True, "input_count": len(rows), "layer_count": 0,
                    "per_layer": 0, "selected_count": 0, "layers": []}
    layer_count = max(1, min(int(layer_count), len(rows), int(top_k)))
    ordered = []
    for row in rows:
        try:
            expression = sp.sympify(row["readable_expression"])
            value = int(complexity(expression))
        except (sp.SympifyError, TypeError, ValueError, OverflowError):
            value = 10**9
        ordered.append((value, row))
    # Complexity partitioning comes first. Use equal-width absolute
    # complexity intervals; ranking is applied only within each interval.
    minimum_complexity = min(value for value, _row in ordered)
    maximum_complexity = max(value for value, _row in ordered)
    layers = [[] for _index in range(layer_count)]
    if minimum_complexity == maximum_complexity:
        layers[0] = ordered
    else:
        width = (maximum_complexity - minimum_complexity) / layer_count
        for value, row in ordered:
            index = int((value - minimum_complexity) / width)
            index = min(layer_count - 1, max(0, index))
            layers[index].append((value, row))
    non_empty = [(index, layer) for index, layer in enumerate(layers) if layer]
    layer_count = len(non_empty)
    quota = max(1, int(math.floor(float(top_k) / layer_count + 0.5)))
    quota = min(quota, min(len(layer) for _index, layer in non_empty))

    selected = []
    layer_metadata = []
    for raw_index, layer in non_empty:
        index = raw_index + 1
        ranked_layer = sorted(
            layer,
            key=lambda item: (
                item[1].get("predicted_rank", 10**9),
                str(item[1].get("formula_id", "")),
                str(item[1].get("readable_expression", "")),
            ),
        )
        chosen = [row for _complexity, row in ranked_layer[:quota]]
        selected.extend(chosen)
        layer_metadata.append({
            "layer": index,
            "available_count": len(layer),
            "selected_count": len(chosen),
            "complexity_min": min(int(value) for value, _row in layer),
            "complexity_max": max(int(value) for value, _row in layer),
            "complexity_interval_min": minimum_complexity + (
                (maximum_complexity - minimum_complexity)
                * raw_index / (len(layers) or 1)
            ),
            "complexity_interval_max": minimum_complexity + (
                (maximum_complexity - minimum_complexity)
                * (raw_index + 1) / (len(layers) or 1)
            ),
            "formula_ids": [row.get("formula_id") for row in chosen],
        })
    selected.sort(key=lambda row: (
        row.get("complexity_layer", 10**9),
        row.get("predicted_rank", 10**9),
        str(row.get("formula_id", "")),
        str(row.get("readable_expression", "")),
    ))
    return selected, {
        "enabled": True,
        "input_count": len(rows),
        "layer_count": layer_count,
        "per_layer": quota,
        "selected_count": len(selected),
        "target_top_k": int(top_k),
        "layers": layer_metadata,
    }


def _zero_constant(expr: sp.Expr, rng: random.Random) -> sp.Expr:
    nodes = [(path, node) for path, node in nodes_with_paths(expr)
             if node.is_Symbol and str(node).startswith(("a", "c", "p"))]
    if not nodes:
        return expr
    path, _ = rng.choice(nodes)
    return replace_at_path(expr, path, sp.Integer(0))


def _merge_duplicate_constants(expr: sp.Expr) -> sp.Expr:
    """Collapse every pure arithmetic constant subtree to one fresh parameter."""
    expression = sp.sympify(expr)
    used_names = {str(symbol) for symbol in expression.free_symbols}

    def is_constant_symbol(symbol: sp.Expr) -> bool:
        return (symbol.is_Symbol
                and str(symbol).startswith(("a", "c", "p")))

    def fresh_constant() -> sp.Symbol:
        index = 0
        while f"c{index}" in used_names:
            index += 1
        symbol = sp.Symbol(f"c{index}")
        used_names.add(str(symbol))
        return symbol

    def is_pure_constant_arithmetic(node: sp.Expr) -> bool:
        if not (node.is_Add or node.is_Mul or node.is_Pow):
            return False
        return all(
            not symbol.is_Symbol or is_constant_symbol(symbol)
            for symbol in node.free_symbols
        )

    def visit(node: sp.Expr) -> sp.Expr:
        if not node.args:
            return node
        args = tuple(visit(argument) for argument in node.args)
        rebuilt = (sp.Add(*args) if node.is_Add else
                   sp.Mul(*args) if node.is_Mul else
                   sp.Pow(*args) if node.is_Pow else node.func(*args))
        return fresh_constant() if is_pure_constant_arithmetic(rebuilt) else rebuilt

    return visit(expression)


def _replace_variable_with_constant(expr: sp.Expr, rng: random.Random) -> sp.Expr:
    nodes = [(path, node) for path, node in nodes_with_paths(expr)
             if node.is_Symbol and str(node).startswith("x")]
    if not nodes:
        return expr
    path, _ = rng.choice(nodes)
    used_names = {str(symbol) for symbol in expr.free_symbols}
    index = 0
    while f"c{index}" in used_names:
        index += 1
    replacement = sp.Symbol(f"c{index}")
    return _merge_duplicate_constants(
        replace_at_path(expr, path, replacement)
    )


def _replace_variable(expr: sp.Expr, rng: random.Random, feature_count: int) -> sp.Expr:
    nodes = [(path, node) for path, node in nodes_with_paths(expr)
             if node.is_Symbol and str(node).startswith("x")]
    if not nodes or feature_count < 2:
        return expr
    path, node = rng.choice(nodes)

    # SymPy flattens ``c*x1/x2`` into one Mul node, so an exact ``x1/x2``
    # subtree is not generally present.  Find the innermost ancestor Mul
    # where the selected input variable and another input variable have
    # opposite unit exponents, then swap that pair as one ratio mutation.
    available_variables = [sp.Symbol(f"x{i + 1}") for i in range(feature_count)]
    ratio_candidates = [
        (candidate_path, candidate, peer)
        for candidate_path, candidate in nodes_with_paths(expr)
        if candidate_path == path[:len(candidate_path)]
        and candidate.is_Mul
        and candidate.as_powers_dict().get(node, 0) in (1, -1)
        for peer in available_variables
        if peer != node
        and candidate.as_powers_dict().get(peer, 0) == -candidate.as_powers_dict()[node]
    ]
    if ratio_candidates:
        # Use the innermost ratio; variable names make a same-level choice
        # deterministic when a product contains multiple ratio factors.
        ratio_path, ratio, peer = max(
            ratio_candidates, key=lambda item: (len(item[0]), str(item[2]))
        )
        flipped = ratio.xreplace({node: peer, peer: node})
        return replace_at_path(expr, ratio_path, flipped)

    choices = [symbol for symbol in available_variables if symbol != node]
    return replace_at_path(expr, path, rng.choice(choices))


def _replace_operator(
    expr: sp.Expr,
    rng: random.Random,
    pairwise_trigonometric_only: bool = False,
) -> sp.Expr:
    nodes = []
    for path, node in nodes_with_paths(expr):
        if node.func in (sp.sin, sp.cos):
            nodes.append((path, node, "trig"))
        elif (node.func is sp.log and len(node.args) == 1
              and node.args[0].func is sp.Abs and len(node.args[0].args) == 1):
            if pairwise_trigonometric_only:
                continue
            # Treat log(Abs(x)) as one protected unary wrapper.  The nested
            # Abs/log nodes themselves are intentionally not mutation targets.
            nodes.append((path, node, "log_abs"))
        elif (node.is_Add or node.is_Mul or node.is_Pow) and len(node.args) >= 2:
            nodes.append((path, node, "binary"))
    if not nodes:
        return expr
    path, node, kind = rng.choice(nodes)
    if kind in ("trig", "log_abs"):
        argument = node.args[0]
        if kind == "log_abs":
            argument = argument.args[0]
        if pairwise_trigonometric_only:
            # Optional mode: log(Abs(.)) is not a mutation target, while
            # sin(.) and cos(.) exchange with equal probability.
            replacement = sp.cos(argument) if node.func is sp.sin else sp.sin(argument)
            return replace_at_path(expr, path, replacement)
        unary_choices = [sp.sin(argument), sp.cos(argument),
                         sp.log(sp.Abs(argument))]
        current_index = {sp.sin: 0, sp.cos: 1}.get(node.func, 2)
        replacement = rng.choice(
            unary_choices[:current_index] + unary_choices[current_index + 1:]
        )
        return replace_at_path(expr, path, replacement)
    args = node.args
    operator = rng.choice(("+", "-", "*", "/"))
    if operator == "+":
        replacement = sp.Add(*args)
    elif operator == "-":
        replacement = args[0] - sp.Add(*args[1:])
    elif operator == "*":
        replacement = sp.Mul(*args)
    else:
        replacement = args[0] / sp.Mul(*args[1:])
    return replace_at_path(expr, path, replacement)


def _parameterize_numeric_constants(expr: sp.Expr) -> sp.Expr:
    """Turn numeric constants outside exponents into fresh fit parameters.

    SymPy may expose a numeric constant after GP subtree recombination, for
    example by simplifying ``x2 / x2`` to ``1``.  Keep exponent integers as
    structural constants, but make other numeric literals optimizable.  Fresh
    ``pN`` names are chosen against every symbol already present in the
    expression so crossover cannot create parameter-name collisions.
    """
    used = set(expr.free_symbols)
    next_index = 0

    def visit(node: sp.Expr, in_exponent: bool = False) -> sp.Expr:
        nonlocal next_index
        if node.is_Number:
            if in_exponent:
                return node
            while sp.Symbol(f"p{next_index}") in used:
                next_index += 1
            parameter = sp.Symbol(f"p{next_index}")
            used.add(parameter)
            next_index += 1
            return parameter
        if not node.args:
            return node
        if node.is_Pow:
            return sp.Pow(visit(node.base), visit(node.exp, True))
        return node.func(*(visit(argument, in_exponent) for argument in node.args))

    return visit(expr)


def _top_level_log_terms(expr: sp.Expr) -> list[tuple[sp.Expr, sp.Expr, sp.Expr]]:
    """Return additive terms shaped as ``coefficient * log(argument)``."""
    terms = sp.Add.make_args(expr)
    found = []
    for term in terms:
        factors = sp.Mul.make_args(term) if term.is_Mul else (term,)
        log_factors = [factor for factor in factors if factor.func is sp.log]
        if len(log_factors) != 1:
            continue
        log_node = log_factors[0]
        coefficient_factors = [factor for factor in factors if factor != log_node]
        coefficient = sp.Mul(*coefficient_factors) if coefficient_factors else sp.Integer(1)
        found.append((term, coefficient, log_node.args[0]))
    return found


def _parameter_values(fit: dict) -> dict[sp.Symbol, float]:
    names = fit.get("parameter_names") or []
    values = fit.get("parameters")
    if values is None or len(names) != len(values):
        return {}
    return {sp.Symbol(name): float(value) for name, value in zip(names, values)}


def _fresh_parameter(expr: sp.Expr, prefix: str = "p") -> sp.Symbol:
    """Return a parameter symbol that does not collide with ``expr``."""
    used = {str(symbol) for symbol in expr.free_symbols}
    index = 0
    while f"{prefix}{index}" in used:
        index += 1
    return sp.Symbol(f"{prefix}{index}")


def _merge_log_terms(expr: sp.Expr, fit: dict) -> sp.Expr | None:
    """Merge top-level log terms when their fitted coefficient signs agree.

    The GP uses one shared structure with independently fitted parameters per
    dataset.  A merge is only valid as one shared expression when every
    dataset gives the same sign for each log coefficient.
    """
    log_terms = _top_level_log_terms(expr)
    if len(log_terms) < 2:
        return None

    dataset_fits = fit.get("dataset_fit_results") or [fit]
    signs_by_term = [[] for _ in log_terms]
    for dataset_fit in dataset_fits:
        values = _parameter_values(dataset_fit)
        for index, (_term, coefficient, _argument) in enumerate(log_terms):
            try:
                if coefficient.free_symbols - set(values):
                    return None
                value = float(coefficient.subs(values))
            except (TypeError, ValueError):
                return None
            if not math.isfinite(value) or abs(value) <= 1e-12:
                return None
            signs_by_term[index].append(1 if value > 0 else -1)

    signs = []
    for term_signs in signs_by_term:
        if not term_signs or len(set(term_signs)) != 1:
            return None
        signs.append(term_signs[0])

    terms = sp.Add.make_args(expr)
    selected = {term for term, _coefficient, _argument in log_terms}
    positive_arguments = [argument for sign, (_term, _coefficient, argument)
                          in zip(signs, log_terms) if sign > 0]
    negative_arguments = [argument for sign, (_term, _coefficient, argument)
                          in zip(signs, log_terms) if sign < 0]
    numerator = sp.Mul(*positive_arguments) if positive_arguments else None
    denominator = sp.Mul(*negative_arguments) if negative_arguments else None
    if numerator is None:
        merged_argument = sp.Pow(denominator, sp.Integer(-1))
    elif denominator is None:
        merged_argument = numerator
    else:
        merged_argument = numerator / denominator
    merged_log = sp.log(merged_argument)
    # The original terms can have different fitted magnitudes, so they cannot
    # be combined exactly into one logarithm.  Keep a fresh outer coefficient
    # and let the refit determine the best common scale for the merged term.
    merged_log = _fresh_parameter(expr) * merged_log
    merged = sp.Add(*(term for term in terms if term not in selected), merged_log)
    merged = _parameterize_numeric_constants(merged)
    return merged if sum(node.func is sp.log for node in sp.preorder_traversal(merged)) == 1 else None


def _parameter_symbols(expr: sp.Expr) -> set[sp.Symbol]:
    """Return fitted-parameter symbols, excluding input variables.

    GP formulas use ``a*`` for combination coefficients, ``c*`` for
    candidate coefficients, and ``p*`` for any parameterized constants.  A
    parameter is identified by its name rather than by its position in the
    expression, which is also how the LM evaluator later binds parameters.
    """
    return {
        symbol for symbol in expr.free_symbols
        if symbol.is_Symbol and str(symbol).startswith(("a", "c", "p"))
    }


def _rename_conflicting_parameters(
    donor: sp.Expr, occupied: set[sp.Symbol]
) -> sp.Expr:
    """Alpha-rename donor parameters that would collide with the recipient.

    A crossover copies a subtree from one fitted formula into another.  The
    copied subtree must not accidentally turn its local ``a1``/``c1`` into
    the recipient's parameter with the same name.  Rename each conflicting
    symbol consistently, so repeated occurrences inside the donor still
    refer to one parameter, while the recipient and donor parameter scopes
    remain independent.
    """
    donor_parameters = sorted(_parameter_symbols(donor), key=str)
    collisions = [symbol for symbol in donor_parameters if symbol in occupied]
    if not collisions:
        return donor

    used = set(occupied) | set(donor.free_symbols)
    substitutions = {}
    for symbol in collisions:
        base = str(symbol)
        index = 1
        fresh = sp.Symbol(f"{base}_gp{index}")
        while fresh in used:
            index += 1
            fresh = sp.Symbol(f"{base}_gp{index}")
        substitutions[symbol] = fresh
        used.add(fresh)
    return donor.xreplace(substitutions)


def _trigonometric_signature(expr: sp.Expr) -> tuple[str, ...]:
    """Return the exact sin/cos subtrees that GP must preserve."""
    return tuple(
        sp.srepr(node)
        for node in sp.preorder_traversal(expr)
        if node.func in (sp.sin, sp.cos)
    )


def _mutate_once(expr: sp.Expr, rng: random.Random, cfg: dict,
                 feature_count: int) -> sp.Expr:
    zero_probability = cfg.get("zero_constant_probability")
    if zero_probability is None:
        operation = rng.choice(("zero_constant", "variable", "operator"))
    else:
        zero_probability = float(zero_probability)
        remaining_probability = (1.0 - zero_probability) / 2.0
        operation = rng.choices(
            ("zero_constant", "variable", "operator"),
            weights=(zero_probability, remaining_probability, remaining_probability),
            k=1,
        )[0]
    if operation == "zero_constant":
        return _zero_constant(expr, rng)
    if operation == "variable":
        return _replace_variable(expr, rng, feature_count)
    return _replace_operator(
        expr, rng,
        pairwise_trigonometric_only=bool(
            cfg.get("pairwise_trigonometric_operator_mutation", False)
        ),
    )


def mutate(expr: sp.Expr, rng: random.Random, cfg: dict, feature_count: int) -> sp.Expr:
    """Mutate only outside protected trigonometric subtrees when requested."""
    if not cfg.get("protect_trigonometric_subtrees", False):
        return _mutate_once(expr, rng, cfg, feature_count)
    signature = _trigonometric_signature(expr)
    for _ in range(20):
        candidate = _mutate_once(expr, rng, cfg, feature_count)
        if _trigonometric_signature(candidate) == signature:
            return candidate
    return expr


def _crossover_once(left: sp.Expr, right: sp.Expr, rng: random.Random,
                    cfg: dict, feature_count: int) -> sp.Expr:
    left_nodes = nodes_with_paths(left)
    right_nodes = nodes_with_paths(right)
    if not left_nodes or not right_nodes:
        return left
    left_path, _ = rng.choice(left_nodes)
    _, donor = rng.choice(right_nodes)
    donor = _rename_conflicting_parameters(donor, set(left.free_symbols))
    child = replace_at_path(left, left_path, donor)
    return child if _valid_child(child, cfg, feature_count) else left


def crossover(left: sp.Expr, right: sp.Expr, rng: random.Random,
              cfg: dict, feature_count: int) -> sp.Expr:
    """Crossover while preserving the exact trigonometric subtrees."""
    if not cfg.get("protect_trigonometric_subtrees", False):
        return _crossover_once(left, right, rng, cfg, feature_count)
    signature = _trigonometric_signature(left)
    for _ in range(20):
        child = _crossover_once(left, right, rng, cfg, feature_count)
        if _trigonometric_signature(child) == signature:
            return child
    return left


def _r2_summary(values) -> tuple[float, float]:
    finite = _finite_r2_values(values)
    if not finite:
        return -1e100, 1e100

    # R² can become a very large finite negative number when a generated
    # expression fits badly.  Computing its variance directly squares that
    # number and can raise OverflowError even though every input is finite.
    # Normalize by the largest magnitude first; this also avoids cancellation
    # in the mean calculation for values with very different magnitudes.
    scale = max(abs(value) for value in finite)
    if scale == 0.0:
        return 0.0, 0.0
    normalized = [value / scale for value in finite]
    normalized_mean = math.fsum(normalized) / len(normalized)
    mean = scale * normalized_mean
    if not math.isfinite(mean):
        return -1e100, 1e100
    normalized_variance = math.fsum(
        (value - normalized_mean) ** 2 for value in normalized
    ) / len(normalized)
    std = scale * math.sqrt(normalized_variance)
    return mean, std if math.isfinite(std) else 1e100


def _finite_r2_values(values) -> list[float]:
    """Return finite R² values, ignoring missing or invalid fits."""
    finite = []
    for value in values:
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value):
            finite.append(value)
    return finite


def _adaptive_initial_r2_threshold(values, fallback: float) -> tuple[float, dict]:
    """Infer a cut between low- and high-quality source formulas with 2-means."""
    finite = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value):
            finite.append(value)
    values_array = np.asarray(finite, dtype=float)
    if values_array.size < 2 or np.ptp(values_array) <= np.finfo(float).eps:
        return float(fallback), {
            "method": "fixed_fallback",
            "reason": "fewer_than_two_distinct_r2_values",
            "sample_count": int(values_array.size),
        }

    centers = np.asarray([values_array.min(), values_array.max()], dtype=float)
    for _ in range(32):
        labels = np.abs(values_array[:, None] - centers[None, :]).argmin(axis=1)
        updated = centers.copy()
        for index in (0, 1):
            members = values_array[labels == index]
            if members.size:
                updated[index] = members.mean()
        if np.allclose(updated, centers, rtol=0.0, atol=1e-12):
            centers = updated
            break
        centers = updated
    centers.sort()
    threshold = float(np.mean(centers))
    return threshold, {
        "method": "two_means",
        "sample_count": int(values_array.size),
        "cluster_centers": centers.tolist(),
        "cluster_counts": [int(np.sum(values_array <= threshold)),
                            int(np.sum(values_array > threshold))],
    }


def _dominates(left: dict, right: dict, secondary_objective: str = "r2_std",
               use_piecewise_difference: bool = False) -> bool:
    """Compare GP objectives, using the requested secondary R² objective."""
    # realdata_mat can use pooled R² as the sole fit-quality objective.  In
    # that mode there is deliberately no second R² objective (mean/min R²).
    primary_r2 = "pooled_r2" if secondary_objective == "pooled_r2" else "mean_r2"
    if secondary_objective == "pooled_r2":
        secondary_better_or_equal = True
        secondary_strictly_better = False
    elif secondary_objective == "min_r2":
        secondary_better_or_equal = left["min_r2"] >= right["min_r2"]
        secondary_strictly_better = left["min_r2"] > right["min_r2"]
    else:
        secondary_better_or_equal = left["r2_std"] <= right["r2_std"]
        secondary_strictly_better = left["r2_std"] < right["r2_std"]
    # ``dict.get(key, default)`` does not replace an explicitly stored None.
    # Failed/unavailable piecewise evaluations are therefore normalized here
    # before doing numeric comparisons.
    def difference_value(item: dict) -> float:
        value = item.get("segment_prediction_difference")
        try:
            value = float(value)
        except (TypeError, ValueError):
            return math.inf
        return value if math.isfinite(value) else math.inf

    left_difference = difference_value(left)
    right_difference = difference_value(right)
    difference_better_or_equal = left_difference <= right_difference
    difference_strictly_better = left_difference < right_difference
    return (
        left[primary_r2] >= right[primary_r2]
        and secondary_better_or_equal
        and (not use_piecewise_difference or difference_better_or_equal)
        and left["complexity"] <= right["complexity"]
        and (
            left[primary_r2] > right[primary_r2]
            or secondary_strictly_better
            or (use_piecewise_difference and difference_strictly_better)
            or left["complexity"] < right["complexity"]
        )
    )


def pareto_front(population: Sequence[dict], secondary_objective: str = "r2_std",
                 use_piecewise_difference: bool = False) -> list[dict]:
    def difference_value(item: dict) -> float:
        value = item.get("segment_prediction_difference")
        try:
            value = float(value)
        except (TypeError, ValueError):
            return math.inf
        return value if math.isfinite(value) else math.inf

    front = [item for item in population
             if not any(_dominates(other, item, secondary_objective,
                                   use_piecewise_difference) for other in population
                        if other is not item)]
    return sorted(front, key=lambda item: (
        item["complexity"],
        difference_value(item)
        if use_piecewise_difference else 0.0,
        (-item["min_r2"] if secondary_objective == "min_r2"
         else item.get(secondary_objective, math.inf)),
        -item["pooled_r2"] if secondary_objective == "pooled_r2"
        else -item["mean_r2"],
        item["expression_key"], str(item.get("formula_id", "")),
    ))


def crowding_distances(front: Sequence[dict], secondary_objective: str = "r2_std",
                       use_piecewise_difference: bool = False) -> dict[str, float]:
    """NSGA-II crowding distance on the GP objectives."""
    distances = {item["expression_key"]: 0.0 for item in front}
    if not front:
        return distances
    objectives = (["pooled_r2"] if secondary_objective == "pooled_r2"
                  else ["mean_r2", secondary_objective])
    if use_piecewise_difference:
        objectives.append("segment_prediction_difference")
    objectives.append("complexity")

    def objective_value(item: dict, objective: str) -> float:
        value = item.get(objective)
        if objective == "segment_prediction_difference":
            try:
                value = float(value)
            except (TypeError, ValueError):
                return math.inf
            return value if math.isfinite(value) else math.inf
        return value

    for objective in objectives:
        ordered = sorted(
            front,
            key=lambda item: (
                objective_value(item, objective), item["expression_key"],
                str(item.get("formula_id", "")),
            ),
        )
        distances[ordered[0]["expression_key"]] = math.inf
        distances[ordered[-1]["expression_key"]] = math.inf
        span = (objective_value(ordered[-1], objective)
                - objective_value(ordered[0], objective))
        if len(ordered) <= 2 or span <= 0:
            continue
        for index in range(1, len(ordered) - 1):
            key = ordered[index]["expression_key"]
            if not math.isinf(distances[key]):
                distances[key] += (
                    (objective_value(ordered[index + 1], objective)
                     - objective_value(ordered[index - 1], objective)) / span
                )
    return distances


def resolve_gp_pareto_config(
    payload: dict, use_piecewise_difference: bool | None = None,
) -> dict:
    """Resolve the shared data-mode-specific GP Pareto convention.

    Keep this in the GP module so the main pipeline and ablation tooling use
    exactly the same objectives and do not drift when the convention changes.
    """
    metadata = payload.get("metadata", {})
    is_realdata_mat = metadata.get("data_mode") == "realdata_mat"
    if use_piecewise_difference is None:
        use_piecewise_difference = is_realdata_mat
    if is_realdata_mat:
        secondary_objective = "pooled_r2"
        use_piecewise_difference = False
        objective_names = ["pooled_r2_max", "complexity_min"]
    else:
        secondary_objective = "r2_std"
        objective_names = ["mean_r2_max", "r2_std_min", "complexity_min"]
    return {
        "secondary_objective": secondary_objective,
        "use_piecewise_difference": bool(use_piecewise_difference),
        "objective_names": objective_names,
    }


def apply_gp_final_hard_constraints(
    items: Sequence[dict], feature_count: int, gp_config: dict,
) -> tuple[list[dict], int]:
    """Apply the final GP hard constraints shared by all callers."""
    if gp_config.get("require_log_abs_x1", False):
        items = [
            item for item in items
            if _has_log_abs_x1(sp.sympify(item["readable_expression"]))
        ]
    rejected_by_final_variable_constraint = 0
    if gp_config.get("require_all_variables", False):
        allowed_variables = {sp.Symbol(f"x{i + 1}") for i in range(feature_count)}
        filtered = []
        for item in items:
            expression = sp.sympify(item["readable_expression"])
            if allowed_variables <= expression.free_symbols:
                filtered.append(item)
            else:
                rejected_by_final_variable_constraint += 1
        items = filtered
    return list(items), rejected_by_final_variable_constraint


def _select_population(
    population: Sequence[dict], size: int, secondary_objective: str = "r2_std",
    use_piecewise_difference: bool = False,
) -> list[dict]:
    unique = {}
    for item in population:
        unique.setdefault(item["expression_key"], item)
    remaining = list(unique.values())
    selected = []
    while remaining and len(selected) < size:
        front = pareto_front(remaining, secondary_objective, use_piecewise_difference)
        selected.extend(front[:size - len(selected)])
        front_keys = {item["expression_key"] for item in front}
        remaining = [item for item in remaining if item["expression_key"] not in front_keys]
    return selected[:size]


def merge_log_formulas(
    results: Sequence[dict], fit_formula: Callable[[dict], dict],
    max_r2_drop: float, initial_perturbation: float = 0.05,
    warm_start_fraction: float = 0.5,
    lm_initializations: int | None = None,
    seed: int | None = None,
    n_jobs: int | None = 1,
) -> tuple[list[dict], list[dict], int]:
    """Refit and accept top-level Log-term merges after other post-processing.

    Half of the LM restarts reuse the source fit as their starting point and
    the other half use independent random starts, matching constant
    significance testing.  ``initial_perturbation`` only affects the warm
    starts; it does not turn the random half into warm starts.
    """
    updated_results = [dict(item) for item in results]
    merge_checks = []
    merged_count = 0
    n_jobs = 1 if n_jobs is None else int(n_jobs)
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs for the log-merge check must be 1, a positive integer, or -1")

    def evaluate(expr: sp.Expr, formula_id: str, fit: dict | None = None) -> dict:
        record = _formula_record(expr, formula_id)
        best = fit if fit is not None else fit_formula(record)
        r2_values = _finite_r2_values(
            row.get("r2") for row in best.get("dataset_fit_results", [])
        )
        mean_r2, r2_std = _r2_summary(r2_values)
        if mean_r2 <= -1e99:
            mean_r2, r2_std = best.get("r2"), 0.0
        return {
            "expression_key": sp.srepr(expr),
            "readable_expression": str(expr),
            "formula_id": formula_id,
            "mean_r2": mean_r2,
            "r2_std": r2_std,
            "min_r2": min(r2_values) if r2_values else mean_r2,
            "complexity": complexity(expr),
            "fit": best,
        }

    def dataset_r2(fit: dict) -> dict:
        rows = fit.get("dataset_fit_results", []) or [fit]
        return {
            row.get("dataset_name", "dataset_1"): row.get("r2")
            for row in rows
        }

    def parameter_values_by_dataset(fit: dict) -> dict[str, dict[str, float]]:
        rows = fit.get("dataset_fit_results", []) or [fit]
        values_by_dataset = {}
        for row in rows:
            names = row.get("parameter_names") or []
            values = row.get("parameters")
            if values is None or len(names) != len(values):
                continue
            values_by_dataset[row.get("dataset_name", "dataset_1")] = {
                str(name): float(value)
                for name, value in zip(names, values)
                if math.isfinite(float(value))
            }
        return values_by_dataset

    def process_one(index: int, item: dict) -> tuple[int, dict | None, dict | None, int]:
        expr = sp.sympify(item["readable_expression"])
        merged_expr = _merge_log_terms(expr, item)
        if merged_expr is None:
            return index, None, None, 0
        merged_record = _formula_record(
            merged_expr, f"{item['formula_id']}_merged"
        )
        previous_values = parameter_values_by_dataset(item)
        try:
            fit_kwargs = {
                "initial_values_by_dataset": previous_values,
                "initial_perturbation": initial_perturbation,
                "warm_start_fraction": warm_start_fraction,
                "lm_initializations": lm_initializations,
            }
            if n_jobs != 1 and seed is not None:
                fit_kwargs["fit_rng"] = np.random.default_rng(
                    int(seed) + index
                )
            try:
                merged_fit = fit_formula(merged_record, **fit_kwargs)
            except TypeError as exc:
                if "fit_rng" not in str(exc):
                    raise
                fit_kwargs.pop("fit_rng", None)
                merged_fit = fit_formula(merged_record, **fit_kwargs)
            merged = evaluate(merged_expr, merged_record["id"], merged_fit)
        except TypeError:
            # Preserve compatibility with simple one-argument fit callbacks.
            merged = evaluate(merged_expr, merged_record["id"])
        original_r2 = dataset_r2(item)
        merged_r2 = dataset_r2(merged["fit"])
        r2_drop = {
            name: (None if old is None or merged_r2.get(name) is None
                   else float(old) - float(merged_r2[name]))
            for name, old in original_r2.items()
        }
        accepted = not any(
            name not in merged_r2 or old is None or new is None
            or float(new) < float(old) - max_r2_drop
            for name, old in original_r2.items()
            for new in [merged_r2.get(name)]
        )
        check = {
            "source_formula_id": item["formula_id"],
            "original_expression": item["readable_expression"],
            "merged_expression": merged["readable_expression"],
            "original_mean_r2": item.get("mean_r2"),
            "merged_mean_r2": merged.get("mean_r2"),
            "r2_drop_by_dataset": r2_drop,
            "original_r2_by_dataset": original_r2,
            "merged_r2_by_dataset": merged_r2,
            "original_complexity": item.get("complexity"),
            "merged_complexity": merged.get("complexity"),
            "accepted": accepted,
            "reason": ("r2_within_allowed_drop" if accepted
                        else "r2_drop_exceeded_allowed_threshold"),
        }
        if not accepted:
            return index, None, check, 0
        merged_r2_values = _finite_r2_values(merged_r2.values())
        updated = {
            **item,
            "expression_key": merged["expression_key"],
            "readable_expression": merged["readable_expression"],
            "formula_id": merged["formula_id"],
            "mean_r2": merged["mean_r2"],
            "r2_std": merged["r2_std"],
            "min_r2": merged["min_r2"],
            "complexity": merged["complexity"],
            **merged["fit"],
            "merged_log_terms": True,
            "source_formula_id": item["formula_id"],
        }
        return index, updated, check, 1

    indexed_items = list(enumerate(updated_results))
    if n_jobs == 1 or len(indexed_items) < 2:
        processed = [process_one(index, item) for index, item in indexed_items]
    else:
        workers = (
            min(len(indexed_items), 24, max(1, (os.cpu_count() or 2) - 1))
            if n_jobs < 0 else min(len(indexed_items), n_jobs)
        )
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            processed = list(executor.map(
                lambda pair: process_one(*pair), indexed_items
            ))
    for index, updated, check, accepted_count in processed:
        if updated is not None:
            updated_results[index] = updated
        if check is not None:
            merge_checks.append(check)
        merged_count += accepted_count
    return updated_results, merge_checks, merged_count


def run_lightweight_gp(
    payload: dict, gp_config: dict, feature_count: int,
    dataset_groups: Sequence[dict], fit_formula: Callable[[dict], dict],
    apply_log_merge: bool = True,
    piecewise_difference_fn: Callable[[str, dict], float | None] | None = None,
) -> tuple[list[dict], dict]:
    """Evolve fitted formulas and return the final Pareto front and metadata."""
    rng = random.Random(int(gp_config.get("seed", 0)))
    is_realdata_mat = payload.get("metadata", {}).get("data_mode") == "realdata_mat"
    pareto_config = resolve_gp_pareto_config(
        payload,
        use_piecewise_difference=bool(is_realdata_mat and piecewise_difference_fn is not None),
    )
    secondary_objective = pareto_config["secondary_objective"]
    use_piecewise_difference = pareto_config["use_piecewise_difference"]
    configured_threshold = float(gp_config.get("initial_r2_threshold", 0.7))
    threshold_mode = str(gp_config.get("initial_r2_threshold_mode", "fixed"))
    if threshold_mode not in ("fixed", "adaptive_cluster", "max_r2_gap"):
        raise ValueError(
            "gp.initial_r2_threshold_mode must be fixed/adaptive_cluster/max_r2_gap"
        )
    max_r2_gap = float(gp_config.get("initial_r2_max_gap", 0.1))
    if max_r2_gap < 0:
        raise ValueError("gp.initial_r2_max_gap cannot be negative")
    top_k = int(gp_config.get("top_k", 10))
    population_size = int(gp_config.get("population_size", 20))
    generations = int(gp_config.get("generations", 5))
    mutation_rate = float(gp_config.get("mutation_rate", 0.60))
    crossover_rate = float(gp_config.get("crossover_rate", 0.25))
    initial_by_key = {}
    source_pool = payload.get("top_ranked_refit_results", [])
    preselection_done = bool(gp_config.get("preselection_done", False))
    if preselection_done:
        # The tree-stage Top-K has already been applied by the caller.  Do
        # not apply another positional Top-K here: constant-significance
        # processing may produce additional variants that must all be
        # considered by the initial-R2 threshold below.
        source_rows = source_pool
        stratified_metadata = gp_config.get("preselection_metadata", {
            "enabled": True, "input_count": len(source_pool),
            "selected_count": len(source_pool), "target_top_k": top_k,
        })
    else:
        # ``source_pool`` is already the result of the tree-model Top-K
        # refit.  The GP Top-K must not truncate this pool again before the
        # initial-R2 threshold is evaluated.  In particular, constant
        # significance can add forward/reverse and isolated-term variants.
        source_rows = source_pool
        stratified_metadata = {
            "enabled": False,
            "input_count": len(source_pool),
            "selected_count": len(source_rows),
            "target_top_k": top_k,
            "selection_stage": "tree_top_k_already_applied",
        }
    rejected_by_initial_log_abs_x1 = 0
    if gp_config.get("require_log_abs_x1_initial", False):
        filtered_source_rows = []
        for row in source_rows:
            try:
                compliant = _has_log_abs_x1(
                    sp.sympify(row["readable_expression"])
                )
            except (KeyError, sp.SympifyError, TypeError, ValueError):
                compliant = False
            if compliant:
                filtered_source_rows.append(row)
            else:
                rejected_by_initial_log_abs_x1 += 1
        source_rows = filtered_source_rows
    threshold_metric = "pooled_r2" if is_realdata_mat else "mean_r2"
    source_threshold_r2 = []
    for row in source_rows:
        dataset_rows = row.get("dataset_fit_results", []) or [row]
        if is_realdata_mat:
            row_r2 = row.get("pooled_r2", row.get("r2"))
            try:
                row_r2 = float(row_r2)
            except (TypeError, ValueError):
                row_r2 = None
        else:
            row_r2, _ = _r2_summary(item.get("r2") for item in dataset_rows)
        if row_r2 is not None and math.isfinite(row_r2):
            source_threshold_r2.append(float(row_r2))
    highest_source_r2 = max(source_threshold_r2) if source_threshold_r2 else None
    if threshold_mode == "max_r2_gap" and highest_source_r2 is not None:
        threshold = highest_source_r2 - max_r2_gap
        threshold_details = {
            "method": "max_r2_gap",
            "threshold_metric": threshold_metric,
            "highest_r2": highest_source_r2,
            "max_r2_gap": max_r2_gap,
            "sample_count": len(source_threshold_r2),
        }
    elif threshold_mode == "adaptive_cluster":
        threshold, threshold_details = _adaptive_initial_r2_threshold(
            source_threshold_r2, configured_threshold
        )
        threshold_details["threshold_metric"] = threshold_metric
    else:
        threshold, threshold_details = configured_threshold, {
            "method": "fixed",
            "threshold_metric": threshold_metric,
            "configured_threshold": configured_threshold,
            "sample_count": len(source_threshold_r2),
        }
    print(
        "GP initial-population R² threshold: "
        f"{threshold:.6f} (mode={threshold_mode}, "
        f"candidate {threshold_metric} count={len(source_threshold_r2)}, "
        f"highest {threshold_metric}={('None' if highest_source_r2 is None else f'{highest_source_r2:.6f}')}, "
        f"tolerance={max_r2_gap:.6f}, method={threshold_details['method']})",
        flush=True,
    )
    rejected_by_threshold = 0
    for row in source_rows:
        dataset_rows = row.get("dataset_fit_results", []) or [row]
        if is_realdata_mat:
            row_threshold_r2 = row.get("pooled_r2", row.get("r2"))
            try:
                row_threshold_r2 = float(row_threshold_r2)
            except (TypeError, ValueError):
                row_threshold_r2 = -1e100
        else:
            row_threshold_r2, _ = _r2_summary(item.get("r2") for item in dataset_rows)
        if row_threshold_r2 < threshold:
            rejected_by_threshold += 1
            continue
        expr = sp.sympify(row["readable_expression"])
        key = sp.srepr(expr)
        item = initial_by_key.setdefault(
            key, {"expr": expr, "formula_id": row["formula_id"], "rows": []}
        )
        item["rows"].extend(row.get("dataset_fit_results", []) or [row])
        # Keep the complete source result as well as its per-dataset rows.
        # The initial GP population reuses these fits without refitting them;
        # storing only the five summary fields below used to discard MAT
        # segmentation metadata (piecewise/split/parameters_by_dataset, ...).
        item.setdefault("source_fit", dict(row))

    # Apply the GP initial-population structural guard only after the R²
    # threshold has selected the initial candidates.  This keeps the
    # threshold statistics comparable across the complete post-significance
    # source pool, while ensuring every formula actually entering GP contains
    # log(Abs(f(x1))) (the pipeline's representation of log(f(x1))).
    rejected_after_initial_log_abs_x1 = 0
    initial_log_abs_x1_filtered = {}
    for key, item in initial_by_key.items():
        if _has_log_abs_x1(item["expr"]):
            initial_log_abs_x1_filtered[key] = item
        else:
            rejected_after_initial_log_abs_x1 += 1
    initial_by_key = initial_log_abs_x1_filtered

    def evaluate(expr: sp.Expr, formula_id: str) -> dict:
        record = _formula_record(expr, formula_id)
        best = fit_formula(record)
        difference = (piecewise_difference_fn(str(expr), best)
                      if use_piecewise_difference else None)
        difference_objective = (
            float(difference) if difference is not None and math.isfinite(float(difference))
            else math.inf
        )
        r2 = best.get("r2")
        if r2 is None:
            return {"expression_key": sp.srepr(expr), "readable_expression": str(expr),
                    "formula_id": formula_id, "r2": -1e100,
                    "mean_r2": -1e100, "r2_std": 1e100, "min_r2": -1e100,
                    "pooled_r2": -1e100,
                    "complexity": complexity(expr),
                    "segment_prediction_difference": difference_objective,
                    "fit": best}
        r2_values = _finite_r2_values(
            row.get("r2") for row in best.get("dataset_fit_results", [])
        )
        mean_r2, r2_std = _r2_summary(r2_values)
        if mean_r2 <= -1e99:
            mean_r2, r2_std = float(r2), 0.0
        pooled_r2 = best.get("pooled_r2", best.get("r2", mean_r2))
        try:
            pooled_r2 = float(pooled_r2)
        except (TypeError, ValueError):
            pooled_r2 = -1e100
        return {"expression_key": sp.srepr(expr), "readable_expression": str(expr),
                "formula_id": formula_id, "r2": mean_r2,
                "mean_r2": mean_r2, "r2_std": r2_std,
                "min_r2": min(r2_values) if r2_values else mean_r2,
                "pooled_r2": pooled_r2,
                "segment_prediction_difference": difference_objective,
                "complexity": complexity(expr), "fit": best}

    # Sine formulas need a more detailed audit than the generic Top-K log:
    # constant-significance processing may create forward/reverse or isolated
    # variants, and the GP pool subsequently applies de-duplication and an R²
    # threshold.  Report the actual post-significance expression seen here.
    sin_initial_diagnostics = []

    def contains_sin(row: dict) -> bool:
        try:
            expression = sp.sympify(row.get("readable_expression", ""))
        except (KeyError, TypeError, ValueError, sp.SympifyError):
            return False
        return any(node.func is sp.sin for node in sp.preorder_traversal(expression))

    def report_sin(row: dict, entered: bool, reason: str | None = None,
                   original_id: str | None = None) -> None:
        detail = {
            "formula_id": row.get("formula_id"),
            "source_formula_id": row.get("source_formula_id", original_id),
            "original_formula_id": original_id,
            "post_significance_expression": row.get("readable_expression"),
            "entered_gp_initial_population": bool(entered),
            "reason": reason,
        }
        sin_initial_diagnostics.append(detail)
        status = "entered" if entered else "not entered"
        suffix = "" if entered else f"; reason: {reason}"
        print(
            f"Formula containing sin (after constant-significance testing) {status} GP initial population: "
            f"{row.get('formula_id')}; result: {row.get('readable_expression')}"
            f"{suffix}", flush=True,
        )

    # ``source_pool`` is the complete constant-significance-processed pool.
    # Keep the diagnostic on this pool so formulas removed by stratified
    # Top-K selection or the initial hard constraint are still shown in their
    # processed form.
    selected_source_keys = {
        (str(row.get("formula_id")), row.get("readable_expression"))
        for row in source_rows
    }
    for row in source_pool:
        if not contains_sin(row):
            continue
        try:
            expression_key = sp.srepr(sp.sympify(row["readable_expression"]))
        except (KeyError, TypeError, ValueError, sp.SympifyError):
            expression_key = None
        entered = expression_key in initial_by_key if expression_key else False
        row_key = (str(row.get("formula_id")), row.get("readable_expression"))
        if row_key not in selected_source_keys:
            reason = "GP initial-candidate Top-K/layered screening"
        elif entered:
            reason = None
        elif not _has_log_abs_x1(sp.sympify(row["readable_expression"])):
            reason = "GP initial-solution log(Abs(f(x1))) hard constraint"
        else:
            reason = "initial R² threshold filter"
        report_sin(row, entered, reason)

    if not initial_by_key:
        return [], {
            "enabled": True, "initial_population": 0, "generations": 0,
            "top_k": top_k, "initial_r2_threshold": threshold,
            "initial_r2_threshold_mode": threshold_mode,
            "initial_r2_threshold_details": threshold_details,
            "source_formula_count": len(source_rows),
            "stratified_selection": stratified_metadata,
            "rejected_by_initial_r2_threshold": rejected_by_threshold,
            "rejected_by_initial_log_abs_x1": rejected_by_initial_log_abs_x1,
            "rejected_after_initial_log_abs_x1": rejected_after_initial_log_abs_x1,
            "rejected_by_final_variable_constraint": 0,
            "best_source_mean_r2": (
                max(source_threshold_r2) if source_threshold_r2 else None
            ),
            "empty_front_reason": "no_top_ranked_formula_exceeded_initial_r2_threshold",
            "sin_initial_population_diagnostics": sin_initial_diagnostics,
        }

    # The initial population already has LM results in the current payload;
    # reuse them instead of fitting every qualifying formula a second time.
    population = []
    for item in initial_by_key.values():
        rows = item["rows"]
        r2_values = _finite_r2_values(row.get("r2") for row in rows)
        mean_r2, r2_std = _r2_summary(r2_values)
        min_r2 = min(r2_values) if r2_values else -1e100
        pooled_r2 = item.get("source_fit", {}).get("pooled_r2", mean_r2)
        try:
            pooled_r2 = float(pooled_r2)
        except (TypeError, ValueError):
            pooled_r2 = -1e100
        population.append({
            "expression_key": sp.srepr(item["expr"]),
            "readable_expression": str(item["expr"]),
            "formula_id": item["formula_id"],
            "r2": mean_r2, "mean_r2": mean_r2, "r2_std": r2_std,
            "min_r2": min_r2,
            "pooled_r2": pooled_r2,
            "segment_prediction_difference": (
                (lambda value: float(value) if value is not None and math.isfinite(float(value))
                 else math.inf)(item.get("source_fit", {}).get(
                     "segment_prediction_difference", math.inf))
                if use_piecewise_difference else None
            ),
            "complexity": complexity(item["expr"]),
            "fit": {
                # Preserve every field from a directly reused source result.
                # The per-dataset rows and GP summaries below are refreshed so
                # this is not merely a stale shallow copy.
                **item.get("source_fit", {}),
                "formula_id": item["formula_id"],
                "r2": mean_r2, "mean_r2": mean_r2, "r2_std": r2_std,
                "min_r2": min_r2,
                "pooled_r2": pooled_r2,
                "dataset_fit_results": rows,
            },
        })
    initial_count = len(population)
    initial_exclusions = []
    tracked_top_rows = gp_config.get("initial_top_ranked_rows", []) or []
    if tracked_top_rows:
        source_by_id = {str(row.get("formula_id")): row for row in source_rows}
        final_ids = {str(item.get("formula_id")) for item in population}
        for original in tracked_top_rows:
            original_id = str(original.get("formula_id"))
            if original_id in final_ids:
                continue
            # Sine formulas are reported above from ``source_rows``, which is
            # already the constant-significance-processed pool.  Do not emit
            # the raw Top-K expression here as a second, misleading diagnostic.
            if contains_sin(original):
                continue
            matches = [
                row for row in source_rows
                if str(row.get("formula_id")) == original_id
                or str(row.get("source_formula_id")) == original_id
            ]
            if not matches:
                checks = [
                    check for check in gp_config.get("constant_significance_checks", [])
                    if str(check.get("source_formula_id")) == original_id
                ]
                failure_reasons = sorted({
                    str(check.get("failure_reason"))
                    for check in checks if check.get("failure_reason")
                })
                if any(not check.get("accepted", False) and not check.get("skipped", False)
                       for check in checks):
                    failure_reasons.append("r2_drop_exceeded_max_r2_drop")
                if failure_reasons:
                    reason = "Constant-significance test: " + ", ".join(failure_reasons)
                elif checks:
                    reason = "No usable original formula or variant remained after constant-significance testing"
                else:
                    reason = "Original formula or variant not found in the GP input pool"
            else:
                reasons = []
                for row in matches:
                    expression_key = sp.srepr(sp.sympify(row["readable_expression"]))
                    if expression_key in initial_by_key:
                        reasons.append(
                            f"{row.get('formula_id')}: duplicate expression"
                        )
                    elif gp_config.get("require_log_abs_x1_initial", False) \
                            and not _has_log_abs_x1(sp.sympify(row["readable_expression"])):
                        reasons.append(
                            f"{row.get('formula_id')}: log(Abs(x1)) hard constraint"
                        )
                    else:
                        reasons.append(f"{row.get('formula_id')}: initial R² threshold filter")
                reason = "; ".join(reasons)
            detail = {
                "formula_id": original_id,
                "readable_expression": original.get("readable_expression"),
                "reason": reason,
            }
            initial_exclusions.append(detail)
            print(
                f"Original Top-K formula did not enter GP initial population: {original_id}; reason: {reason}",
                flush=True,
            )
    print(
        f"\nInitial solutions entering the GP algorithm (after deduplication and initial R² thresholding,  "
        f"{initial_count} items): ",
        flush=True,
    )
    if population:
        for index, item in enumerate(population, start=1):
            mean_r2 = item.get("mean_r2")
            r2_std = item.get("r2_std")
            mean_r2_text = (
                "None" if mean_r2 is None else f"{float(mean_r2):.6f}"
            )
            r2_std_text = "None" if r2_std is None else f"{float(r2_std):.6f}"
            print(
                f"  #{index:>3} {item['formula_id']}: "
                f"mean_R²={mean_r2_text}, R²_std={r2_std_text}, "
                f"complexity={item['complexity']}, "
                f"{item['readable_expression']}",
                flush=True,
            )
    else:
        print("   (no eligible initial solutions)", flush=True)
    # Keep every evaluated formula in a historical archive.  The breeding
    # population is intentionally allowed to change, but a good formula from
    # an earlier generation must not disappear merely because it is no longer
    # selected among the current limited-size population.
    hall_of_fame = {
        item["expression_key"]: item for item in population
    }
    while len(population) < population_size:
        population.append(rng.choice(population).copy())

    generated = 0
    generations_without_binary_elite = 0
    generation_best_pooled_r2 = []
    progress = tqdm(range(generations), desc="Lightweight GP evolution", unit="generation")
    for generation in progress:
        children = []
        child_progress = tqdm(
            total=population_size,
            desc=f"GP generation {generation + 1}/{generations}: formula fitting",
            unit="formula",
            leave=False,
        )
        while len(children) < population_size:
            parent = rng.choice(population)
            child_expr = sp.sympify(parent["readable_expression"])
            if rng.random() < crossover_rate and len(population) >= 2:
                donor = rng.choice(population)
                child_expr = crossover(
                    child_expr, sp.sympify(donor["readable_expression"]),
                    rng, gp_config, feature_count,
                )
            if rng.random() < mutation_rate:
                mutated = mutate(child_expr, rng, gp_config, feature_count)
                if _valid_child(mutated, gp_config, feature_count):
                    child_expr = mutated
            child_expr = _parameterize_numeric_constants(child_expr)
            if not _valid_child(child_expr, gp_config, feature_count):
                continue
            key = sp.srepr(child_expr)
            if key in {item["expression_key"] for item in children}:
                continue
            children.append(evaluate(child_expr, f"gp_{generation:04d}_{generated:06d}"))
            hall_of_fame.setdefault(children[-1]["expression_key"], children[-1])
            generated += 1
            if hasattr(child_progress, "update"):
                child_progress.update(1)
        if hasattr(child_progress, "close"):
            child_progress.close()
        pool = [*population, *children]
        compliant = pool
        if gp_config.get("require_log_abs_x1", False):
            compliant = [
                item for item in compliant
                if _has_log_abs_x1(sp.sympify(item["readable_expression"]))
            ]
        binary_elite_pool = [
            item for item in compliant
            if _contains_all_variables(
                sp.sympify(item["readable_expression"]), feature_count
            )
        ]
        # Only formulas containing every input variable can become elites.
        # If a generation has no such candidate, keep a fallback population
        # so unconstrained breeding can continue.
        if binary_elite_pool:
            population = _select_population(
                binary_elite_pool, population_size, secondary_objective,
                use_piecewise_difference
            )
        else:
            generations_without_binary_elite += 1
            population = _select_population(
                compliant or pool, population_size, secondary_objective,
                use_piecewise_difference
            )
        pooled_values = []
        for item in hall_of_fame.values():
            fit = item.get("fit") or {}
            value = fit.get("pooled_r2", item.get("r2"))
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                pooled_values.append(value)
        current_best_pooled_r2 = max(pooled_values) if pooled_values else None
        generation_best_pooled_r2.append(current_best_pooled_r2)
        progress_message = (
            f"GP generation {generation + 1}/{generations}: "
            f"best pooled_r2="
            f"{('None' if current_best_pooled_r2 is None else f'{current_best_pooled_r2:.8f}')}"
        )
        if hasattr(progress, "write"):
            progress.write(progress_message)
        else:
            print(progress_message, flush=True)
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(
                population=len(population),
                new_formulas=generated,
                pareto=len(pareto_front(compliant, secondary_objective,
                                        use_piecewise_difference)),
            )

    # Build the final front from all historically evaluated formulas rather
    # than only the last breeding population.  This preserves earlier
    # high-quality solutions while still applying the same Pareto objectives.
    historical_pool = list(hall_of_fame.values())
    final_pool, rejected_by_final_variable_constraint = (
        apply_gp_final_hard_constraints(historical_pool, feature_count, gp_config)
    )

    # ``final_pool`` is the complete, de-duplicated archive after all final
    # hard constraints have been applied.  Print it before Pareto selection so
    # formulas which are later dominated remain visible in the run log.
    secondary_label = {
        "min_r2": "minimum R²",
        "pooled_r2": "pooled R²",
        "r2_std": "R² standard deviation",
    }[secondary_objective]
    difference_label = ", minimize breakpoint prediction difference" if use_piecewise_difference else ""
    print(
        f"\nFinal candidates after GP evolution and before Pareto filtering (total {len(final_pool)} items; "
        f"objectives: maximize mean R², {secondary_label}"
        f"{'maximize' if secondary_objective in ('min_r2', 'pooled_r2') else 'minimize'}"
        f"{difference_label}, minimize complexity): ",
        flush=True,
    )
    if final_pool:
        for index, item in enumerate(final_pool, start=1):
            mean_r2 = item.get("mean_r2")
            secondary_value = (item.get("pooled_r2") if secondary_objective == "pooled_r2"
                               else item.get(secondary_objective))
            mean_r2_text = "None" if mean_r2 is None else f"{float(mean_r2):.6f}"
            secondary_text = (
                "None" if secondary_value is None else f"{float(secondary_value):.6f}"
            )
            difference_value = item.get("segment_prediction_difference", math.inf)
            difference_text = (
                f", breakpoint prediction difference={float(difference_value):.6f}"
                if use_piecewise_difference and math.isfinite(float(difference_value)) else ""
            )
            print(
                f"  #{index:>3} {item['formula_id']}: "
                f"mean R²={mean_r2_text}, {secondary_label}={secondary_text}"
                f"{difference_text}, "
                f"complexity={item['complexity']}, {item['readable_expression']}",
                flush=True,
            )
    else:
        print("   (no candidates remain after final constraint filtering)", flush=True)
    front = pareto_front(final_pool, secondary_objective, use_piecewise_difference)
    distances = crowding_distances(front, secondary_objective, use_piecewise_difference)
    front = sorted(
        front,
        key=lambda item: (
            not math.isinf(distances[item["expression_key"]]),
            -distances[item["expression_key"]],
            (-item["pooled_r2"] if secondary_objective == "pooled_r2"
             else -item["mean_r2"]),
            item.get("segment_prediction_difference", math.inf)
            if use_piecewise_difference else 0.0,
            (-item["min_r2"] if secondary_objective == "min_r2" else item.get(secondary_objective, math.inf)),
            item["complexity"],
            item["expression_key"], str(item.get("formula_id", "")),
        ),
    )
    output = []
    for rank, item in enumerate(front, start=1):
        crowding = distances[item["expression_key"]]
        output.append({
            "pareto_rank": rank, "crowding_rank": rank,
            "crowding_distance": crowding,
            "crowding_distance_label": "∞" if math.isinf(crowding) else crowding,
            "is_boundary_solution": math.isinf(crowding),
            "formula_id": item["formula_id"],
            "readable_expression": item["readable_expression"],
            "r2": None if item["mean_r2"] <= -1e99 else item["mean_r2"],
            "mean_r2": None if item["mean_r2"] <= -1e99 else item["mean_r2"],
            "r2_std": None if item["r2_std"] >= 1e99 else item["r2_std"],
            "min_r2": None if item["min_r2"] <= -1e99 else item["min_r2"],
            "pooled_r2": None if item["pooled_r2"] <= -1e99 else item["pooled_r2"],
            "segment_prediction_difference": (
                (item.get("segment_prediction_difference")
                 if math.isfinite(float(item.get("segment_prediction_difference", math.inf)))
                 else None)
                if use_piecewise_difference else None
            ),
            "complexity": item["complexity"], **item["fit"],
        })
    merge_log_max_r2_drop = float(gp_config.get("merge_log_max_r2_drop", 0.01))
    if apply_log_merge:
        output, merge_checks, merged_count = merge_log_formulas(
            output, fit_formula, merge_log_max_r2_drop
        )
    else:
        merge_checks, merged_count = [], 0
    metadata = {
        "enabled": True, "top_k": top_k,
        "initial_r2_threshold": threshold,
        "initial_r2_threshold_mode": threshold_mode,
        "initial_r2_threshold_details": threshold_details,
        "source_formula_count": len(source_rows),
        "stratified_selection": stratified_metadata,
        "rejected_by_initial_r2_threshold": rejected_by_threshold,
        "rejected_by_initial_log_abs_x1": rejected_by_initial_log_abs_x1,
        "rejected_after_initial_log_abs_x1": rejected_after_initial_log_abs_x1,
        "rejected_by_final_variable_constraint": rejected_by_final_variable_constraint,
        "initial_top_ranked_exclusions": initial_exclusions,
        "sin_initial_population_diagnostics": sin_initial_diagnostics,
        "best_source_mean_r2": (
            max(source_threshold_r2) if source_threshold_r2 else None
        ),
        "objectives": pareto_config["objective_names"],
        "initial_population": initial_count, "population_size": population_size,
        "generations": generations, "generated_formulas": generated,
        "generations_without_binary_elite": generations_without_binary_elite,
        "best_pooled_r2_by_generation": generation_best_pooled_r2,
        "pareto_front_size": len(output),
        "historical_formula_count": len(hall_of_fame),
        "merged_log_formula_count": merged_count,
        "merge_log_max_r2_drop": merge_log_max_r2_drop,
        "log_merge_checks": merge_checks,
        "seed": int(gp_config.get("seed", 0)),
        "hard_constraints": (
            (["contains_log_abs_of_x1"]
             if gp_config.get("require_log_abs_x1", False) else [])
            + (["final_pareto_contains_all_variables"]
               if gp_config.get("require_all_variables", False) else [])
        ),
    }
    return output, metadata
