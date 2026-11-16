"""Fixed-order structural features for symbolic formula ranking models."""
from __future__ import annotations

import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import sympy as sp


FIT_PARAMETER = re.compile(r"^(?:a|c|p)\d+(?:_(?:left|right))?$")
OPERATORS = ("add", "subtract", "multiply", "divide", "power", "log", "sin", "cos", "abs")
VARIABLES = ("x1", "x2", "x3", "x4")
# The formula generator's wrapper vocabulary.  It lives here so generation
# and ranking encoders use one ordered definition without a circular import.
FORMULA_WRAPPERS = (
    "identity", "log_abs", "sin", "cos", "log_abs_sin", "log_abs_cos",
)
CATEGORICAL_STRUCTURE_FEATURES = (
    "root_operator", "root_left_operator", "root_right_operator", "parent_child_signature",
)


def node_token(node: sp.Expr) -> str:
    if node.is_Symbol:
        return "FIT_CONST" if FIT_PARAMETER.fullmatch(str(node)) else f"VAR:{node}"
    if node.is_Integer:
        return f"INT:{int(node)}"
    if node.is_Number:
        return "NUMBER"
    return f"OP:{node.func.__name__}/{len(node.args)}"


def operator_name(node: sp.Expr) -> str:
    if node.is_Add:
        return "add"
    if node.is_Mul:
        return "multiply"
    if node.is_Pow:
        return "power"
    if node.func == sp.log:
        return "log"
    if node.func == sp.sin:
        return "sin"
    if node.func == sp.cos:
        return "cos"
    if node.func == sp.Abs:
        return "abs"
    if node.is_Symbol:
        return "fit_constant" if FIT_PARAMETER.fullmatch(str(node)) else "variable"
    if node.is_Number:
        return "number"
    return node.func.__name__.lower()


def preorder_node_sequence(expr: sp.Expr, max_nodes: int) -> list[str]:
    tokens = [node_token(node) for node in sp.preorder_traversal(expr)]
    return (tokens[:max_nodes] + ["<PAD>"] * max_nodes)[:max_nodes]


def _subtree_stats(node: sp.Expr) -> tuple[int, int]:
    if not node.args:
        return 1, 0
    child = [_subtree_stats(arg) for arg in node.args]
    return 1 + sum(item[0] for item in child), 1 + max(item[1] for item in child)


def _is_negative_term(node: sp.Expr) -> bool:
    return bool(node.could_extract_minus_sign())


def _is_negative_exponent(node: sp.Expr) -> bool:
    if not node.is_Pow or node.exp.free_symbols:
        return False
    try:
        return bool(float(node.exp) < 0)
    except (TypeError, ValueError, OverflowError):
        return False


def numeric_feature_names(max_depth: int) -> list[str]:
    names = ["node_count", "tree_depth", "max_consecutive_unary_depth"]
    names.extend(f"depth_{depth}_node_count" for depth in range(max_depth + 1))
    names.append("depth_overflow_node_count")
    names.extend(("left_subtree_node_count", "right_subtree_node_count",
                  "left_right_node_count_difference", "left_subtree_depth",
                  "right_subtree_depth", "left_right_depth_difference"))
    for operator in OPERATORS:
        names.extend((f"operator_{operator}_count", f"operator_{operator}_ratio"))
    for variable in VARIABLES:
        names.extend((
            f"{variable}_count", f"{variable}_max_depth", f"{variable}_inside_log",
            f"{variable}_inside_trig", f"{variable}_inside_denominator",
            f"{variable}_inside_power_exponent",
        ))
    return names


@dataclass(frozen=True)
class FormulaStructureFeatureExtractor:
    max_nodes: int = 96
    max_depth: int = 12

    @property
    def categorical_names(self) -> list[str]:
        return [f"node_{index:03d}" for index in range(self.max_nodes)] + list(
            CATEGORICAL_STRUCTURE_FEATURES
        )

    @property
    def numeric_names(self) -> list[str]:
        return numeric_feature_names(self.max_depth)

    @property
    def feature_names(self) -> list[str]:
        return self.categorical_names + self.numeric_names

    @property
    def categorical_indices(self) -> list[int]:
        return list(range(len(self.categorical_names)))

    def extract_one(self, formula: str) -> tuple[list[str], list[float]]:
        expr = sp.sympify(formula)
        sequence = preorder_node_sequence(expr, self.max_nodes)
        root_args = list(expr.args)
        root_left = operator_name(root_args[0]) if root_args else "none"
        root_right = operator_name(root_args[1]) if len(root_args) > 1 else "none"

        total_nodes, tree_depth = _subtree_stats(expr)
        if root_args:
            left_nodes, left_depth = _subtree_stats(root_args[0])
            if len(root_args) == 1:
                right_nodes, right_depth = 0, 0
            else:
                right_stats = [_subtree_stats(arg) for arg in root_args[1:]]
                right_nodes = sum(item[0] for item in right_stats)
                right_depth = max(item[1] for item in right_stats)
        else:
            left_nodes = right_nodes = left_depth = right_depth = 0

        depth_counts = [0] * (self.max_depth + 1)
        depth_overflow = 0
        operator_counts = {name: 0 for name in OPERATORS}
        variable_stats = {
            name: {"count": 0, "max_depth": 0, "log": 0, "trig": 0,
                   "denominator": 0, "exponent": 0}
            for name in VARIABLES
        }
        parent_child_counts: dict[str, int] = {}
        max_unary_chain = 0

        def walk(node: sp.Expr, depth: int, unary_chain: int, *, inside_log: bool,
                 inside_trig: bool, inside_denominator: bool,
                 inside_exponent: bool) -> None:
            nonlocal depth_overflow, max_unary_chain
            if depth <= self.max_depth:
                depth_counts[depth] += 1
            else:
                depth_overflow += 1
            name = operator_name(node)
            if name in operator_counts:
                operator_counts[name] += 1
            if node.is_Add:
                operator_counts["subtract"] += sum(_is_negative_term(arg) for arg in node.args)
            if _is_negative_exponent(node):
                operator_counts["divide"] += 1
            is_unary = name in {"log", "sin", "cos", "abs"}
            current_chain = unary_chain + 1 if is_unary else 0
            max_unary_chain = max(max_unary_chain, current_chain)
            if node.is_Symbol and str(node) in variable_stats:
                stats = variable_stats[str(node)]
                stats["count"] += 1
                stats["max_depth"] = max(stats["max_depth"], depth)
                stats["log"] |= int(inside_log)
                stats["trig"] |= int(inside_trig)
                stats["denominator"] |= int(inside_denominator)
                stats["exponent"] |= int(inside_exponent)
            parent = operator_name(node)
            for index, child in enumerate(node.args):
                child_name = operator_name(child)
                key = f"{parent}>{child_name}"
                parent_child_counts[key] = parent_child_counts.get(key, 0) + 1
                child_denominator = inside_denominator or (
                    node.is_Pow and index == 0 and _is_negative_exponent(node)
                )
                child_exponent = inside_exponent or (node.is_Pow and index == 1)
                walk(
                    child, depth + 1, current_chain,
                    inside_log=inside_log or name == "log",
                    inside_trig=inside_trig or name in {"sin", "cos"},
                    inside_denominator=child_denominator,
                    inside_exponent=child_exponent,
                )

        walk(expr, 0, 0, inside_log=False, inside_trig=False,
             inside_denominator=False, inside_exponent=False)
        signature = "|".join(
            f"{key}:{parent_child_counts[key]}" for key in sorted(parent_child_counts)
        ) or "none"
        categories = sequence + [operator_name(expr), root_left, root_right, signature]

        numeric: list[float] = [float(total_nodes), float(tree_depth), float(max_unary_chain)]
        numeric.extend(map(float, depth_counts))
        numeric.append(float(depth_overflow))
        numeric.extend((float(left_nodes), float(right_nodes), float(left_nodes - right_nodes),
                        float(left_depth), float(right_depth), float(left_depth - right_depth)))
        denominator = max(1, total_nodes)
        for operator in OPERATORS:
            count = float(operator_counts[operator])
            numeric.extend((count, count / denominator))
        for variable in VARIABLES:
            stats = variable_stats[variable]
            numeric.extend((float(stats["count"]), float(stats["max_depth"]),
                            float(stats["log"]), float(stats["trig"]),
                            float(stats["denominator"]), float(stats["exponent"])))
        array = np.nan_to_num(np.asarray(numeric, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if len(array) != len(self.numeric_names):
            raise RuntimeError("Structure feature names do not match the numeric dimensions")
        return [str(value) for value in categories], array.tolist()

    def transform(
        self, formulas: Sequence[str], n_jobs: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract formula rows in stable order; ``n_jobs=1`` is legacy mode."""
        formulas = list(formulas)
        if n_jobs == 1 or len(formulas) < 2:
            extracted = [self.extract_one(formula) for formula in formulas]
        else:
            workers = (
                min(8, max(1, (os.cpu_count() or 2) - 1))
                if n_jobs is None or int(n_jobs) < 0 else int(n_jobs)
            )
            workers = min(max(1, int(workers)), len(formulas))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                extracted = list(executor.map(self.extract_one, formulas))
        categorical = [item[0] for item in extracted]
        numeric = [item[1] for item in extracted]
        return np.asarray(categorical, dtype=object), np.asarray(numeric, dtype=np.float32)
