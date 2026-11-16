"""Optional four-variable, dimensionally typed GP pipeline.

This module is deliberately separate from :mod:`structure_screen.tree_search_ops`.  The
legacy GP path therefore remains byte-for-byte unchanged when the feature is
disabled in the unified configuration.
"""
from __future__ import annotations

import json
import logging
import random
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import sympy as sp

from .encoding import canonical_structure
from .tree_search_ops import (CandidateBatcher, complexity, contains_disabled_operator,
                 deduplicate_exact,
                 has_transcendental_function, nodes_with_paths, pareto_front,
                 replace_at_path, valid)
from .lm import fit_constants, has_invalid_symbolic_values


Dimension = Tuple[Fraction, ...]
SUPPORTED_UNARY = (sp.log, sp.sin, sp.cos, sp.exp)


def _fraction(value) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, str):
        return Fraction(value)
    return Fraction(str(value))


def normalize_dimensions(values: Sequence) -> Dimension:
    return tuple(_fraction(value) for value in values)


def load_dimension_spec(config: dict) -> tuple[Dict[sp.Symbol, Dimension], Dimension]:
    raw_variables = config["variable_dimensions"]
    target = normalize_dimensions(config["target_dimension"])
    if not target:
        raise ValueError("dimensional_gp.target_dimension 不能为空")
    dimensions = {}
    for index in range(1, 5):
        name = f"x{index}"
        if name not in raw_variables:
            raise ValueError(f"dimensional_gp.variable_dimensions 缺少 {name}")
        dimension = normalize_dimensions(raw_variables[name])
        if len(dimension) != len(target):
            raise ValueError(f"{name} 的量纲向量长度与目标量纲不一致")
        dimensions[sp.Symbol(name)] = dimension
    return dimensions, target


def zero_dimension(size: int) -> Dimension:
    return (Fraction(0),) * size


def infer_dimension(expr: sp.Expr, variable_dimensions: Dict[sp.Symbol, Dimension]):
    """Return an exact dimension vector, or ``None`` for an illegal tree."""
    if not variable_dimensions:
        raise ValueError("variable_dimensions 不能为空")
    size = len(next(iter(variable_dimensions.values())))
    zero = zero_dimension(size)

    def visit(node):
        if node.is_Number:
            return zero
        if node.is_Symbol:
            return variable_dimensions.get(node)
        if node.is_Add:
            dimensions = [visit(argument) for argument in node.args]
            if any(value is None for value in dimensions) or len(set(dimensions)) != 1:
                return None
            return dimensions[0]
        if node.is_Mul:
            result = list(zero)
            for argument in node.args:
                dimension = visit(argument)
                if dimension is None:
                    return None
                result = [left + right for left, right in zip(result, dimension)]
            return tuple(result)
        if node.is_Pow:
            base_dimension = visit(node.base)
            if base_dimension is None or not node.exp.is_Number:
                return None
            try:
                exponent = _fraction(node.exp)
            except (TypeError, ValueError, ZeroDivisionError):
                return None
            return tuple(value * exponent for value in base_dimension)
        if node.func == sp.Abs and len(node.args) == 1:
            return visit(node.args[0])
        if node.func in SUPPORTED_UNARY and len(node.args) == 1:
            argument_dimension = visit(node.args[0])
            return zero if argument_dimension == zero else None
        return None

    return visit(sp.sympify(expr))


def dimensionally_valid(expr, variable_dimensions, target_dimension=None) -> bool:
    dimension = infer_dimension(sp.sympify(expr), variable_dimensions)
    return dimension is not None and (
        target_dimension is None or dimension == normalize_dimensions(target_dimension)
    )


def expand_to_four_variables(X: np.ndarray):
    """Pad a 1D--3D input matrix with deterministic unit-valued variables."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not 1 <= X.shape[1] <= 4:
        raise ValueError(f"X 必须有 1~4 列，实际 shape={X.shape}")
    original_count = X.shape[1]
    if original_count == 4:
        return X.copy(), {}
    padded = np.column_stack((X, np.ones((len(X), 4 - original_count), dtype=float)))
    substitutions = {f"x{index}": 1 for index in range(original_count + 1, 5)}
    return padded, substitutions


def restore_original_formula(formula, original_variable_count: int) -> str:
    if not 1 <= original_variable_count <= 4:
        raise ValueError("original_variable_count 必须在 1~4 之间")
    expr = sp.sympify(formula)
    substitutions = {sp.Symbol(f"x{index}"): 1
                     for index in range(original_variable_count + 1, 5)}
    restored = sp.simplify(sp.powsimp(sp.trigsimp(expr.subs(substitutions)), force=True))
    return str(restored)


def load_deduplicated_native_4d(path: Path, variable_dimensions,
                                validate_dimensions=True):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = payload.get("generation_metadata", {})
    values = metadata.get("native_4d")
    if not values:
        raise ValueError(
            f"{path} 不含 generation_metadata.native_4d；"
            "请先用 stage=generate/all 重新生成严格无量纲公式"
        )
    unique = {}
    rejected = 0
    required_native_dimension = zero_dimension(len(next(iter(variable_dimensions.values()))))
    for text in values:
        try:
            expr = sp.sympify(text)
            if not expr.free_symbols:
                rejected += 1
                continue
            if (validate_dimensions
                    and infer_dimension(expr, variable_dimensions) != required_native_dimension):
                rejected += 1
                continue
            key = canonical_structure(expr) if validate_dimensions else sp.srepr(expr)
            unique.setdefault(key, expr)
        except Exception:
            rejected += 1
    logging.info("四元量纲 GP native 基元：输入=%d，结构去重=%d，非法=%d",
                 len(values), len(unique), rejected)
    if not unique:
        raise ValueError(f"未能从 {path} 加载任何量纲合法的 native 四元基元")
    return list(unique.values())


def _validate_ratio_basis(variable_dimensions, target_dimension):
    """Validate the assumptions behind F(x1/x4, x2/x4, x3/x4)."""
    expected_symbols = {sp.Symbol(f"x{index}") for index in range(1, 5)}
    if set(variable_dimensions) != expected_symbols:
        raise ValueError("ratio_basis_only 要求变量恰好为 x1、x2、x3、x4")
    dimensions = {variable_dimensions[symbol] for symbol in expected_symbols}
    if len(dimensions) != 1:
        raise ValueError("ratio_basis_only 要求 x1、x2、x3、x4 具有相同量纲")
    required_target = zero_dimension(len(next(iter(dimensions))))
    if target_dimension != required_target:
        raise ValueError("ratio_basis_only 只适用于无量纲目标")


def _choose_ratio_terminal(rng, variable_dimensions, cfg):
    """Choose c*xi/x4, i=1..3, with an independently fitted constant c."""
    denominator = sp.Symbol("x4")
    numerator = rng.choice([sp.Symbol("x1"), sp.Symbol("x2"), sp.Symbol("x3")])
    low = float(cfg.get("constant_min", -2.0))
    high = float(cfg.get("constant_max", 2.0))
    coefficient = sp.Float(rng.uniform(low, high))
    expr = coefficient * numerator / denominator
    return expr, infer_dimension(expr, variable_dimensions)


def _choose_unchecked_terminal(rng, native_pool, variable_dimensions, cfg):
    """Choose a trusted terminal without computing its dimension."""
    native_probability = float(cfg.get("native_terminal_probability", 0.45))
    variable_probability = float(cfg.get("variable_terminal_probability", 0.40))
    draw = rng.random()
    low = float(cfg.get("constant_min", -2.0))
    high = float(cfg.get("constant_max", 2.0))
    if cfg.get("ratio_basis_only", False):
        ratio_probability = float(cfg.get(
            "ratio_terminal_probability",
            min(1.0, native_probability + variable_probability),
        ))
        if draw < ratio_probability:
            denominator = sp.Symbol("x4")
            numerator = rng.choice([sp.Symbol("x1"), sp.Symbol("x2"), sp.Symbol("x3")])
            return sp.Float(rng.uniform(low, high)) * numerator / denominator
        return sp.Float(rng.uniform(low, high))
    if native_pool and draw < native_probability:
        return rng.choice(native_pool)
    if draw < native_probability + variable_probability:
        return sp.Symbol(rng.choice([str(symbol) for symbol in variable_dimensions]))
    return sp.Float(rng.uniform(low, high))


def _choose_terminal(rng, native_pool, variable_dimensions, cfg):
    native_probability = float(cfg.get("native_terminal_probability", 0.45))
    variable_probability = float(cfg.get("variable_terminal_probability", 0.40))
    draw = rng.random()
    if cfg.get("ratio_basis_only", False):
        ratio_probability = float(cfg.get(
            "ratio_terminal_probability",
            min(1.0, native_probability + variable_probability),
        ))
        if draw < ratio_probability:
            return _choose_ratio_terminal(rng, variable_dimensions, cfg)
        low = float(cfg.get("constant_min", -2.0))
        high = float(cfg.get("constant_max", 2.0))
        expr = sp.Float(rng.uniform(low, high))
        return expr, infer_dimension(expr, variable_dimensions)
    if native_pool and draw < native_probability:
        expr = rng.choice(native_pool)
    elif draw < native_probability + variable_probability:
        expr = rng.choice(list(variable_dimensions))
    else:
        low = float(cfg.get("constant_min", -2.0))
        high = float(cfg.get("constant_max", 2.0))
        expr = sp.Float(rng.uniform(low, high))
    return expr, infer_dimension(expr, variable_dimensions)


def random_dimensional_tree(rng, native_pool, variable_dimensions, cfg, level=0,
                            max_depth=None):
    """Build a tree whose every node is dimensionally legal by construction."""
    max_depth = int(cfg.get("initial_max_depth", 4) if max_depth is None else max_depth)
    terminal_probability = float(cfg.get("terminal_probability", 0.30))
    if level >= max_depth or rng.random() < terminal_probability:
        return _choose_terminal(rng, native_pool, variable_dimensions, cfg)

    unary_probability = float(cfg.get("unary_probability", 0.25))
    if rng.random() < unary_probability:
        child, child_dimension = random_dimensional_tree(
            rng, native_pool, variable_dimensions, cfg, level + 1, max_depth
        )
        zero = zero_dimension(len(child_dimension))
        disabled = {str(name).strip().lower()
                    for name in cfg.get("disabled_operators", ["exp", "sqrt"])}
        names = [str(name).strip().lower() for name in
                 cfg.get("unary_operators", ["log", "sin", "cos", "exp"])
                 if str(name).strip().lower() not in disabled]
        if names:
            name = rng.choice(names)
            functions = {
                "log": sp.log, "sin": sp.sin, "cos": sp.cos,
                "exp": sp.exp, "sqrt": sp.sqrt,
            }
            if name not in functions:
                if name not in ("log_sin", "log_cos"):
                    raise ValueError(f"不支持的量纲 GP 一元算子：{name}")
            if name == "sqrt":
                return sp.sqrt(child), tuple(value / 2 for value in child_dimension)
            if child_dimension == zero:
                if name == "log_sin":
                    return sp.log(sp.Abs(sp.sin(child)) + sp.Float(.01)), zero
                if name == "log_cos":
                    return sp.log(sp.Abs(sp.cos(child)) + sp.Float(.01)), zero
                argument = sp.Abs(child) if name == "log" else child
                return functions[name](argument), zero
        return child, child_dimension

    left, left_dimension = random_dimensional_tree(
        rng, native_pool, variable_dimensions, cfg, level + 1, max_depth
    )
    right, right_dimension = random_dimensional_tree(
        rng, native_pool, variable_dimensions, cfg, level + 1, max_depth
    )
    operations = tuple(cfg.get("binary_operators", ["+", "-", "*", "/"]))
    operation = rng.choice(operations)
    if operation in ("+", "-") and left_dimension != right_dimension:
        legal_fallbacks = [name for name in operations if name in ("*", "/")]
        operation = rng.choice(legal_fallbacks or ["*"])
    if operation == "+": return left + right, left_dimension
    if operation == "-": return left - right, left_dimension
    if operation == "*":
        return left * right, tuple(a + b for a, b in zip(left_dimension, right_dimension))
    if operation == "/":
        return left / right, tuple(a - b for a, b in zip(left_dimension, right_dimension))
    raise ValueError(f"不支持的量纲 GP 二元算子：{operation}")


def random_unchecked_tree(rng, native_pool, variable_dimensions, cfg, level=0,
                          max_depth=None):
    """Fast tree grammar for trusted bases; never calls dimension inference."""
    max_depth = int(cfg.get("initial_max_depth", 4) if max_depth is None else max_depth)
    if (level >= max_depth
            or rng.random() < float(cfg.get("terminal_probability", 0.30))):
        return _choose_unchecked_terminal(rng, native_pool, variable_dimensions, cfg)

    if rng.random() < float(cfg.get("unary_probability", 0.25)):
        child = random_unchecked_tree(
            rng, native_pool, variable_dimensions, cfg, level + 1, max_depth
        )
        disabled = {str(name).strip().lower()
                    for name in cfg.get("disabled_operators", ["exp", "sqrt"])}
        names = [str(name).strip().lower() for name in
                 cfg.get("unary_operators", ["log", "sin", "cos", "exp"])
                 if str(name).strip().lower() not in disabled]
        if not names:
            return child
        name = rng.choice(names)
        if name == "log": return sp.log(sp.Abs(child))
        if name == "sin": return sp.sin(child)
        if name == "cos": return sp.cos(child)
        if name == "exp": return sp.exp(child)
        if name == "sqrt": return sp.sqrt(child)
        if name == "log_sin": return sp.log(sp.Abs(sp.sin(child)) + sp.Float(.01))
        if name == "log_cos": return sp.log(sp.Abs(sp.cos(child)) + sp.Float(.01))
        raise ValueError(f"不支持的量纲 GP 一元算子：{name}")

    left = random_unchecked_tree(
        rng, native_pool, variable_dimensions, cfg, level + 1, max_depth
    )
    right = random_unchecked_tree(
        rng, native_pool, variable_dimensions, cfg, level + 1, max_depth
    )
    operation = rng.choice(tuple(cfg.get("binary_operators", ["+", "-", "*", "/"])))
    if operation == "+": return left + right
    if operation == "-": return left - right
    if operation == "*": return left * right
    if operation == "/": return left / right
    raise ValueError(f"不支持的量纲 GP 二元算子：{operation}")


def eligible_dimensional_candidate(expr, cfg, variable_dimensions, target_dimension):
    if contains_disabled_operator(expr, cfg.get("disabled_operators", ["exp", "sqrt"])):
        return False
    if (cfg.get("require_transcendental", False)
            and not has_transcendental_function(expr)):
        return False
    return (valid(expr, None, int(cfg.get("max_formula_complexity", 20)))
            and infer_dimension(expr, variable_dimensions) == target_dimension)


def eligible_unchecked_candidate(expr, cfg):
    """Fast candidate checks unrelated to dimensions or symbolic equivalence."""
    if contains_disabled_operator(expr, cfg.get("disabled_operators", ["exp", "sqrt"])):
        return False
    if (cfg.get("require_transcendental", False)
            and not has_transcendental_function(expr)):
        return False
    return valid(expr, None, int(cfg.get("max_formula_complexity", 20)))


def _nodes_with_dimensions(expr, variable_dimensions):
    result = []
    for path, node in nodes_with_paths(expr):
        dimension = infer_dimension(node, variable_dimensions)
        if dimension is not None:
            result.append((path, node, dimension))
    return result


def dimensional_crossover(left, right, rng, cfg, variable_dimensions, target_dimension):
    left_nodes = _nodes_with_dimensions(left, variable_dimensions)
    right_nodes = _nodes_with_dimensions(right, variable_dimensions)
    rng.shuffle(left_nodes)
    for target_path, _, target_dimension_at_node in left_nodes:
        donors = [node for _, node, dimension in right_nodes
                  if dimension == target_dimension_at_node]
        if not donors:
            continue
        child = replace_at_path(left, target_path, rng.choice(donors))
        if eligible_dimensional_candidate(child, cfg, variable_dimensions, target_dimension):
            return child
    return left


def dimensional_mutation(expr, rng, cfg, native_pool, variable_dimensions, target_dimension):
    nodes = _nodes_with_dimensions(expr, variable_dimensions)
    if not nodes:
        return expr
    target_path, _, required = rng.choice(nodes)
    for _ in range(int(cfg.get("mutation_subtree_attempts", 30))):
        replacement, dimension = random_dimensional_tree(
            rng, native_pool, variable_dimensions, cfg, max_depth=2
        )
        if dimension != required:
            continue
        child = replace_at_path(expr, target_path, replacement)
        if eligible_dimensional_candidate(child, cfg, variable_dimensions, target_dimension):
            return child
    return expr


def unchecked_crossover(left, right, rng, cfg):
    """Fast crossover without annotating either parent tree with dimensions."""
    target_path, _ = rng.choice(nodes_with_paths(left))
    _, donor = rng.choice(nodes_with_paths(right))
    child = replace_at_path(left, target_path, donor)
    return child if eligible_unchecked_candidate(child, cfg) else left


def unchecked_delete_subtree_mutation(expr, rng, cfg, native_pool,
                                      variable_dimensions):
    """Replace a non-root subtree with a trusted terminal, only if simpler."""
    original_complexity = complexity(expr)
    candidates = [
        (path, node) for path, node in nodes_with_paths(expr, include_root=False)
        if complexity(node) > 1
    ]
    if not candidates:
        return expr
    attempts = int(cfg.get("fast_delete_attempts", 8))
    for _ in range(max(1, attempts)):
        target_path, target = rng.choice(candidates)
        terminal = _choose_unchecked_terminal(
            rng, native_pool, variable_dimensions, cfg
        )
        # Avoid rebuilding the full formula when this replacement cannot make
        # the selected subtree smaller.
        if complexity(terminal) >= complexity(target):
            continue
        child = replace_at_path(expr, target_path, terminal)
        if (complexity(child) < original_complexity
                and eligible_unchecked_candidate(child, cfg)):
            return child
    return expr


def unchecked_mutation(expr, rng, cfg, native_pool, variable_dimensions):
    """Fast subtree mutation without dimension matching or retry-by-dimension."""
    if rng.random() < float(cfg.get("delete_subtree_probability", 0.0)):
        return unchecked_delete_subtree_mutation(
            expr, rng, cfg, native_pool, variable_dimensions
        )
    nodes = nodes_with_paths(expr)
    if not nodes:
        return expr
    attempts = int(cfg.get("fast_mutation_attempts", 3))
    for _ in range(max(1, attempts)):
        target_path, _ = rng.choice(nodes)
        replacement = random_unchecked_tree(
            rng, native_pool, variable_dimensions, cfg, max_depth=2
        )
        child = replace_at_path(expr, target_path, replacement)
        if eligible_unchecked_candidate(child, cfg):
            return child
    return expr


def generate_dimensional_target_free(cfg, native_pool, variable_dimensions,
                                     target_dimension, seed=42,
                                     candidate_callback=None):
    """Generate target-free formulas using only the typed grammar and native bases."""
    dimension_checks = bool(cfg.get("enable_dimension_checks", True))
    if dimension_checks and cfg.get("ratio_basis_only", False):
        _validate_ratio_basis(variable_dimensions, target_dimension)
    rng = random.Random(seed)
    population_size = int(cfg["population_size"])
    maximum_attempts = population_size * int(cfg.get("initialization_attempt_factor", 200))

    def initialize_population():
        population, attempts = [], 0
        while len(population) < population_size and attempts < maximum_attempts:
            attempts += 1
            if dimension_checks:
                expr, _ = random_dimensional_tree(
                    rng, native_pool, variable_dimensions, cfg
                )
                eligible = eligible_dimensional_candidate(
                    expr, cfg, variable_dimensions, target_dimension
                )
            else:
                expr = random_unchecked_tree(
                    rng, native_pool, variable_dimensions, cfg
                )
                eligible = eligible_unchecked_candidate(expr, cfg)
            if eligible:
                population.append(expr)
        if not population:
            raise RuntimeError("无法创建四元 GP 初始种群")
        if len(population) < population_size:
            logging.warning(
                "四元 GP 初始种群不足：%d/%d",
                len(population), population_size,
            )
        return population

    population = initialize_population()

    batch_size = int(cfg.get("candidates_per_fit", 0))
    max_generations = int(cfg.get("max_generations", cfg.get("generations", 20)))
    batcher = CandidateBatcher(
        batch_size, candidate_callback,
        cfg.get("deduplication_scope", "global_structure"),
    )
    restart_after_screening = (
        bool(cfg.get("restart_population_after_screening", False))
        and candidate_callback is not None
    )

    def is_eligible(expr):
        if not dimension_checks:
            return eligible_unchecked_candidate(expr, cfg)
        return eligible_dimensional_candidate(
            expr, cfg, variable_dimensions, target_dimension
        )

    for generation in range(max_generations):
        eligible = [expr for expr in population if is_eligible(expr)]
        batches_before = batcher.batches_emitted
        if batcher.add(
                eligible, generation + 1,
                stop_after_emit=restart_after_screening):
            logging.info("四元量纲 GP 达到配置目标：generation=%d", generation + 1)
            return batcher.last_batch
        if (restart_after_screening
                and batcher.batches_emitted > batches_before):
            batcher.discard_pending()
            if generation + 1 >= max_generations:
                return batcher.last_batch
            population = initialize_population()
            logging.info(
                "四元 GP 完成第 %d 个筛选批次，已重新随机初始化种群",
                batcher.batches_emitted,
            )
            continue
        next_population = []
        for _ in range(population_size):
            left = rng.choice(population)
            child = left
            if rng.random() < float(cfg.get("crossover_probability", 0.65)):
                if dimension_checks:
                    child = dimensional_crossover(
                        left, rng.choice(population), rng, cfg,
                        variable_dimensions, target_dimension,
                    )
                else:
                    child = unchecked_crossover(
                        left, rng.choice(population), rng, cfg
                    )
            if rng.random() < float(cfg.get("mutation_probability", 0.80)):
                if dimension_checks:
                    child = dimensional_mutation(
                        child, rng, cfg, native_pool,
                        variable_dimensions, target_dimension,
                    )
                else:
                    child = unchecked_mutation(
                        child, rng, cfg, native_pool, variable_dimensions
                    )
            next_population.append(child)
        population = next_population
        logging.info(
            "四元量纲 GP generation=%d/%d screened=%d batches=%d "
            "dimension_checks=%s",
            generation + 1, max_generations,
            batcher.total_emitted, batcher.batches_emitted, dimension_checks,
        )
    eligible = [expr for expr in population if is_eligible(expr)]
    if batcher.add(eligible, max_generations):
        return batcher.last_batch
    batcher.flush(max_generations)
    return batcher.last_batch


def _regression_metrics(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    error = y - prediction
    scale = float(np.max(np.abs(error))) if error.size else 0.0
    if not np.isfinite(scale):
        return {"mse": float("inf"), "rmse": float("inf")}
    if scale == 0.0:
        return {"mse": 0.0, "rmse": 0.0}
    with np.errstate(over="ignore", invalid="ignore"):
        normalized_mse = float(np.mean(np.square(error / scale)))
    limit = np.sqrt(np.finfo(float).max / max(normalized_mse, 1.0))
    mse = float("inf") if scale > limit else scale * scale * normalized_mse
    return {"mse": mse, "rmse": float(np.sqrt(mse))}


def run_dimensional_gp(predictor, X, y, output_path, cfg, lm_cfg,
                       native_formulas_path, dimensional_cfg, seed=42,
                       original_variable_count=None):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    original_count = (X.shape[1] if original_variable_count is None
                      else int(original_variable_count))
    if not 1 <= original_count <= 4 or original_count > X.shape[1]:
        raise ValueError(
            f"original_variable_count={original_count} 与 X.shape={X.shape} 不匹配"
        )
    X4, _ = expand_to_four_variables(X)
    X_original = X[:, :original_count]
    substitutions = {f"x{index}": 1 for index in range(original_count + 1, 5)}
    dimension_checks = bool(cfg.get("enable_dimension_checks", True))
    if dimension_checks:
        variable_dimensions, target_dimension = load_dimension_spec(dimensional_cfg)
    else:
        # Fast grammar needs symbol names only.  Do not parse or validate the
        # dimension specification when recursive dimension checks are disabled.
        variable_dimensions = {sp.Symbol(f"x{index}"): () for index in range(1, 5)}
        target_dimension = ()
    if cfg.get("ratio_basis_only", False):
        if dimension_checks:
            _validate_ratio_basis(variable_dimensions, target_dimension)
        native_pool = []
        logging.info(
            "四元 GP 比值基模式：含变量终端仅使用 "
            "c*x1/x4、c*x2/x4、c*x3/x4；dimension_checks=%s",
            dimension_checks,
        )
    else:
        native_pool = load_deduplicated_native_4d(
            native_formulas_path, variable_dimensions,
            validate_dimensions=dimension_checks,
        )
    fitted_results = []
    invalid_after_restore = 0
    lm_failed = 0
    fit_count = 0
    callback_invocations = 0
    target_r2 = float(cfg.get("target_r2", 1.0))
    target_complexity = int(cfg.get("target_complexity", 0))
    top_k = int(cfg.get("fit_top_k_per_batch", 2))
    threshold = cfg.get("r2_threshold")

    def screen_and_fit(batch, generation):
        nonlocal fit_count, callback_invocations, invalid_after_restore, lm_failed
        callback_invocations += 1
        # Also protects the legacy/custom-generator fallback path.
        batch = deduplicate_exact(batch)
        formulas = [str(expr) for expr in batch]
        scores = np.asarray(predictor(formulas), dtype=float)
        selected = sorted(
            ((formula, float(score)) for formula, score in zip(formulas, scores)
             if np.isfinite(score) and score <= 1.0
             and (threshold is None or score >= threshold)),
            key=lambda pair: pair[1], reverse=True,
        )[:top_k]
        reached = False
        for formula_4d, predicted_r2 in selected:
            fit_count += 1
            index = fit_count
            try:
                restored_formula = restore_original_formula(formula_4d, original_count)
                restored_expr = sp.sympify(restored_formula)
            except Exception:
                invalid_after_restore += 1
                continue
            if (not restored_expr.free_symbols
                    or has_invalid_symbolic_values(restored_expr)):
                invalid_after_restore += 1
                continue
            try:
                fitted_4d = fit_constants(
                    formula_4d, X4, y, lm_cfg["restarts"], lm_cfg["max_nfev"], seed + index
                )
                fitted_original = fit_constants(
                    restored_formula, X_original, y, lm_cfg["restarts"], lm_cfg["max_nfev"],
                    seed + 100000 + index,
                )
            except Exception:
                fitted_4d = fitted_original = None
            if not fitted_4d or not fitted_original:
                lm_failed += 1
                continue
            scores_to_check = (fitted_4d["true_r2"], fitted_original["true_r2"])
            if not all(np.isfinite(score) and score <= 1.0 for score in scores_to_check):
                continue
            formula_id = f"dim4_{index:06d}"
            four_metrics = _regression_metrics(y, fitted_4d["prediction"])
            original_metrics = _regression_metrics(y, fitted_original["prediction"])
            restored_optimized_4d = restore_original_formula(
                fitted_4d["optimized_formula"], original_count
            )
            row = {
                "mapping_id": formula_id,
                "structure": canonical_structure(formula_4d),
                "complexity": complexity(sp.sympify(formula_4d)),
                "predicted_r2": predicted_r2,
                "true_r2": float(fitted_original["true_r2"]),
                "generation": generation,
                "derived_variable_substitutions": substitutions,
                "four_dimensional": {
                    "formula": formula_4d,
                    "optimized_formula": fitted_4d["optimized_formula"],
                    "predicted_r2": predicted_r2,
                    "true_r2": float(fitted_4d["true_r2"]),
                    **four_metrics,
                    "optimized_constants": fitted_4d["constants"],
                },
                "original_variables": {
                    "formula": restored_formula,
                    "optimized_formula": fitted_original["optimized_formula"],
                    "true_r2": float(fitted_original["true_r2"]),
                    **original_metrics,
                    "optimized_constants": fitted_original["constants"],
                },
                "correspondence": {
                    "mapping_id": formula_id,
                    "four_dimensional_formula": formula_4d,
                    "restored_original_formula": restored_formula,
                    "optimized_four_dimensional_formula": fitted_4d["optimized_formula"],
                    "restored_from_optimized_4d_formula": restored_optimized_4d,
                    "independently_fitted_original_formula": fitted_original["optimized_formula"],
                    "substitutions": substitutions,
                },
            }
            fitted_results.append(row)
            reached = reached or (
                row["true_r2"] >= target_r2
                and row["complexity"] <= target_complexity
            )
        return reached

    candidates = generate_dimensional_target_free(
        cfg, native_pool, variable_dimensions, target_dimension, seed,
        candidate_callback=screen_and_fit,
    )
    # Compatibility for custom/legacy generators that do not invoke the callback.
    if callback_invocations == 0:
        batch_size = max(1, int(cfg.get("candidates_per_fit", len(candidates) or 1)))
        for offset in range(0, len(candidates), batch_size):
            if screen_and_fit(candidates[offset:offset + batch_size], None):
                break

    final = pareto_front(fitted_results)
    if cfg.get("pareto_max_results"):
        final = final[:int(cfg["pareto_max_results"])]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info(
        "四元 GP：原变量=%d，补全=%s，native=%d，last_batch=%d，"
        "NN/LM拟合=%d，置1后非法=%d，LM失败=%d，"
        "双公式拟合有效=%d，Pareto=%d，dimension_checks=%s",
        original_count, substitutions, len(native_pool), len(candidates),
        fit_count, invalid_after_restore, lm_failed, len(fitted_results), len(final),
        dimension_checks,
    )
    return final
