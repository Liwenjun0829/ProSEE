"""Rank generated formulas from fixed-order syntax-tree and engineered features."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import sympy as sp
from scipy.sparse import csr_matrix, hstack
from scipy.stats import rankdata, spearmanr

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - progress remains optional
    def tqdm(iterable, **_kwargs):
        return iterable

from formula_structure_features import (
    FORMULA_WRAPPERS,
    OPERATORS,
    VARIABLES,
    FormulaStructureFeatureExtractor,
    node_token,
    operator_name,
)


def generate_formula_value_matrix(
    formula: str, grid_size: int = 100, x_min: float = 0.0, x_max: float = 100.0,
) -> np.ndarray:
    """Evaluate a formula on a regular ``x1``/``x2`` grid.

    All free symbols other than ``x1`` and ``x2`` are treated as fitted
    parameters and fixed to one.  Evaluation failures deliberately produce an
    all-NaN matrix; ``normalize_value_matrix`` then turns it into a safe,
    uninformative zero matrix instead of letting one invalid formula abort a
    ranking run.
    """
    grid_size = int(grid_size)
    if grid_size < 2:
        raise ValueError("response_svd_grid_size must be at least 2")
    axis = np.linspace(float(x_min), float(x_max), grid_size, dtype=float)
    x1_values, x2_values = np.meshgrid(axis, axis, indexing="ij")
    try:
        expression = sp.sympify(formula)
        x1, x2 = sp.Symbol("x1"), sp.Symbol("x2")
        substitutions = {
            symbol: sp.Integer(1)
            for symbol in expression.free_symbols
            if symbol not in {x1, x2}
        }
        expression = expression.subs(substitutions)
        evaluator = sp.lambdify((x1, x2), expression, modules="numpy")
        with np.errstate(all="ignore"):
            values = np.asarray(evaluator(x1_values, x2_values), dtype=float)
        if values.ndim == 0:
            values = np.full((grid_size, grid_size), float(values), dtype=float)
        else:
            values = np.broadcast_to(values, (grid_size, grid_size)).astype(float, copy=True)
        return values
    # A generated formula can contain an unsupported SymPy function or a
    # lambdify implementation that raises a library-specific exception.
    # Treat every formula-local evaluation failure as invalid numeric output.
    except Exception:  # noqa: BLE001 - this is an intentional fault boundary
        return np.full((grid_size, grid_size), np.nan, dtype=float)


def normalize_value_matrix(value_matrix: np.ndarray) -> np.ndarray:
    """Min-max normalize a response matrix while safely handling invalid values."""
    values = np.asarray(value_matrix, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros(values.shape, dtype=np.float32)
    finite_values = values[finite]
    lower, upper = float(np.min(finite_values)), float(np.max(finite_values))
    filled = np.where(finite, values, float(np.median(finite_values)))
    scale = upper - lower
    if not math.isfinite(scale) or scale <= np.finfo(float).eps:
        return np.zeros(values.shape, dtype=np.float32)
    normalized = (filled - lower) / scale
    return np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def extract_svd_features(value_matrix: np.ndarray, feature_count: int = 20) -> np.ndarray:
    """Return a fixed-width, zero-padded vector of singular values."""
    feature_count = int(feature_count)
    if feature_count < 1:
        raise ValueError("response_svd_feature_count must be at least 1")
    try:
        singular_values = np.linalg.svd(
            np.nan_to_num(value_matrix, nan=0.0, posinf=0.0, neginf=0.0),
            compute_uv=False,
        )
    except np.linalg.LinAlgError:
        singular_values = np.empty(0, dtype=float)
    features = np.zeros(feature_count, dtype=np.float32)
    count = min(feature_count, len(singular_values))
    features[:count] = np.nan_to_num(singular_values[:count], nan=0.0,
                                     posinf=0.0, neginf=0.0)
    return features


class FormulaResponseSVDEncoder:
    """Encode formulas by SVD singular values of their normalized responses."""

    def __init__(self, grid_size: int = 100, feature_count: int = 20):
        self.grid_size = int(grid_size)
        self.feature_count = int(feature_count)
        if self.grid_size < 2:
            raise ValueError("response_svd_grid_size must be at least 2")
        if self.feature_count < 1:
            raise ValueError("response_svd_feature_count must be at least 1")
        self.feature_names = [f"response_svd_singular_value_{index + 1}"
                              for index in range(self.feature_count)]

    def fit(self, _formulas: Sequence[str]):
        return self

    def transform(self, formulas: Sequence[str], n_jobs: int | None = None) -> np.ndarray:
        formulas = list(formulas)

        def encode(formula: str) -> np.ndarray:
            return extract_svd_features(
                normalize_value_matrix(generate_formula_value_matrix(
                    formula, grid_size=self.grid_size,
                )), self.feature_count,
            )

        if n_jobs == 1 or len(formulas) < 2:
            features = [encode(formula) for formula in formulas]
        else:
            workers = (
                min(8, max(1, (os.cpu_count() or 2) - 1))
                if n_jobs is None or int(n_jobs) < 0 else int(n_jobs)
            )
            workers = min(max(1, int(workers)), len(formulas))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                features = list(executor.map(encode, formulas))
        return np.asarray(features, dtype=np.float32)


class OperatorPrimitiveMatrixEncoder:
    """Encode generator records as operator-instance × primitive matrices.

    The input is intentionally a generated-formula record, not an expression
    string: ``operator``/``operators`` and ``base_ids`` are the generator's
    authoritative classification.  This prevents nested wrappers from being
    counted as both their composite and their component operators.
    """

    rows_per_operator = 2
    merged_operator_categories = ("identity", "log_abs", "trig", "log_abs_trig")
    merged_operator_map = {
        "identity": "identity", "log_abs": "log_abs",
        "sin": "trig", "cos": "trig",
        "log_abs_sin": "log_abs_trig", "log_abs_cos": "log_abs_trig",
    }

    def __init__(self, base_records: Sequence[dict], rows_per_operator: int | None = None):
        if rows_per_operator is not None:
            rows_per_operator = int(rows_per_operator)
            if rows_per_operator < 1:
                raise ValueError("rows_per_operator must be a positive integer")
        self.rows_per_operator = (
            self.__class__.rows_per_operator
            if rows_per_operator is None else rows_per_operator
        )
        self.operator_categories = tuple(FORMULA_WRAPPERS)
        self.base_ids = tuple(str(row["id"]) for row in base_records)
        self.base_expressions = tuple(
            str(row["readable_expression"]) for row in base_records
        )
        if len(set(self.base_ids)) != len(self.base_ids):
            raise ValueError("Formula base IDs must be unique")
        self.base_indices = {base_id: index for index, base_id in enumerate(self.base_ids)}
        self.operator_indices = {
            operator: index for index, operator in enumerate(self.operator_categories)
        }
        self.expression_vectors: dict[str, np.ndarray] = {}

    @property
    def matrix_shape(self) -> tuple[int, int]:
        return (len(self.operator_categories) * self.rows_per_operator, len(self.base_ids))

    @property
    def merged_matrix_shape(self) -> tuple[int, int]:
        return (len(self.merged_operator_categories) * self.rows_per_operator,
                len(self.base_ids))

    @property
    def combined_matrix_feature_dimension(self) -> int:
        return int(np.prod(self.matrix_shape))

    @property
    def feature_names(self) -> list[str]:
        return [
            f"operator_primitive_{operator}_{occurrence}_{base_id}"
            for operator in self.operator_categories
            for occurrence in range(1, self.rows_per_operator + 1)
            for base_id in self.base_ids
        ]

    @property
    def merged_feature_names(self) -> list[str]:
        return [
            f"merged_operator_primitive_{operator}_{occurrence}_{base_id}"
            for operator in self.merged_operator_categories
            for occurrence in range(1, self.rows_per_operator + 1)
            for base_id in self.base_ids
        ]

    def _operator_instances(self, record: dict) -> list[tuple[str, Sequence[str]]]:
        """Return ordered wrapper instances from the generation metadata."""
        if record.get("operator") is not None:
            base_ids = record.get("base_ids") or ()
            # Auxiliary fitting seeds (for example the envelope bridge) are
            # not generated from formula primitives, so they have no
            # incidence entry in this generator-specific representation.
            if str(record["operator"]) not in self.operator_indices and not base_ids:
                return []
            return [(str(record["operator"]), base_ids)]
        operators = record.get("operators") or ()
        base_groups = record.get("base_ids") or ()
        if len(operators) != len(base_groups):
            if operators or base_groups:
                raise ValueError("The operators and base_ids lengths in the combination formula do not match")
            return []
        return [(str(operator), base_ids or ())
                for operator, base_ids in zip(operators, base_groups)]

    def transform(self, records: Sequence[dict]) -> np.ndarray:
        result = np.zeros((len(records), np.prod(self.matrix_shape)), dtype=np.float32)
        for record_index, record in enumerate(records):
            matrix = result[record_index].reshape(self.matrix_shape)
            occurrences = {operator: 0 for operator in self.operator_categories}
            for operator, base_ids in self._operator_instances(record):
                if operator not in self.operator_indices:
                    raise ValueError(f"Unknown formula-generation operator: {operator}")
                occurrence = occurrences[operator]
                occurrences[operator] += 1
                if occurrence >= self.rows_per_operator:
                    continue
                row = self.operator_indices[operator] * self.rows_per_operator + occurrence
                for base_id in base_ids:
                    try:
                        matrix[row, self.base_indices[str(base_id)]] = 1.0
                    except KeyError as exc:
                        raise ValueError(f"Formula references an undefined formula base: {base_id}") from exc
        return result

    def transform_merged(self, records: Sequence[dict]) -> np.ndarray:
        """Return the second matrix with similar trigonometric operators merged."""
        result = np.zeros((len(records), np.prod(self.merged_matrix_shape)), dtype=np.float32)
        operator_indices = {
            operator: index for index, operator in enumerate(self.merged_operator_categories)
        }
        for record_index, record in enumerate(records):
            matrix = result[record_index].reshape(self.merged_matrix_shape)
            occurrences = {operator: 0 for operator in self.merged_operator_categories}
            for operator, base_ids in self._operator_instances(record):
                merged_operator = self.merged_operator_map.get(operator)
                if merged_operator is None:
                    continue
                occurrence = occurrences[merged_operator]
                occurrences[merged_operator] += 1
                if occurrence >= self.rows_per_operator:
                    continue
                row = operator_indices[merged_operator] * self.rows_per_operator + occurrence
                for base_id in base_ids:
                    try:
                        matrix[row, self.base_indices[str(base_id)]] = 1.0
                    except KeyError as exc:
                        raise ValueError(f"Formula references an undefined formula base: {base_id}") from exc
        return result

    def register_records(self, records: Sequence[dict], vectors: np.ndarray | None = None) -> None:
        """Persist vectors keyed by SymPy's structural representation.

        A loaded scorer receives expression strings only.  Generated formulas
        are therefore resolved through their parsed ``srepr`` rather than a
        formatting-dependent string comparison.
        """
        if vectors is None:
            vectors = self.transform(records)
        for record, vector in zip(records, vectors):
            key = sp.srepr(sp.sympify(record["readable_expression"]))
            self.expression_vectors[key] = np.asarray(
                vector, dtype=np.float32
            ).copy()

    def transform_formulas(self, formulas: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(formulas), self.combined_matrix_feature_dimension), dtype=np.float32)
        for index, formula in enumerate(formulas):
            key = sp.srepr(sp.sympify(formula))
            known = self.expression_vectors.get(key)
            if known is not None:
                vectors[index] = known
        return vectors

class GroupedExtraTreesRegressor:
    """Extra-randomized forest with separate tree/numeric feature groups."""

    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1,
                 tree_feature_count=0, tree_feature_ratio=0.5,
                 random_state=None, n_jobs=None):
        self.n_estimators = int(n_estimators)
        self.max_depth = None if max_depth is None else int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.tree_feature_count = int(tree_feature_count)
        self.tree_feature_ratio = float(tree_feature_ratio)
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _build_tree(self, X, y, indices, depth, rng, importance, sample_weight):
        weights = sample_weight[indices]
        weight_total = float(np.sum(weights))
        prediction = float(np.dot(weights, y[indices]) / max(weight_total, 1e-12))
        leaf = (prediction, None, None, None, None)
        if (len(indices) < 2 * self.min_samples_leaf
                or (self.max_depth is not None and depth >= self.max_depth)
                or np.all(y[indices] == y[indices[0]])):
            return leaf
        numeric = np.arange(self.tree_feature_count, X.shape[1], dtype=int)
        tree_count = min(
            self.tree_feature_count,
            max(1, int(round(self.tree_feature_count * self.tree_feature_ratio))),
        ) if self.tree_feature_count else 0
        tree = (rng.choice(self.tree_feature_count, size=tree_count, replace=False)
                if tree_count else np.empty(0, dtype=int))
        candidates = np.concatenate((tree, numeric))
        parent_sum = float(np.dot(weights, y[indices]))
        parent_sse = float(np.dot(weights, y[indices] ** 2)) - parent_sum ** 2 / max(weight_total, 1e-12)
        best = None
        for feature in candidates:
            values = X[indices, feature]
            lower, upper = float(np.min(values)), float(np.max(values))
            if not upper > lower:
                continue
            threshold = float(rng.uniform(lower, upper))
            left = indices[values <= threshold]
            right = indices[values > threshold]
            if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                continue
            left_sum = float(np.dot(sample_weight[left], y[left]))
            right_sum = float(np.dot(sample_weight[right], y[right]))
            left_weight = float(np.sum(sample_weight[left]))
            right_weight = float(np.sum(sample_weight[right]))
            score = (
                float(np.dot(sample_weight[left], y[left] ** 2))
                - left_sum ** 2 / max(left_weight, 1e-12)
                + float(np.dot(sample_weight[right], y[right] ** 2))
                - right_sum ** 2 / max(right_weight, 1e-12)
            )
            if best is None or score < best[0]:
                best = (score, int(feature), threshold, left, right)
        if best is None or best[0] >= parent_sse - 1e-12:
            return leaf
        score, feature, threshold, left, right = best
        importance[feature] += (parent_sse - score) / len(indices)
        return (prediction, feature, threshold,
                self._build_tree(X, y, left, depth + 1, rng, importance, sample_weight),
                self._build_tree(X, y, right, depth + 1, rng, importance, sample_weight))

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("GroupedExtraTreesRegressor input dimensions do not match")
        if not 0 <= self.tree_feature_count <= X.shape[1]:
            raise ValueError("Invalid number of tree-coordinate features")
        if self.min_samples_leaf < 1 or self.n_estimators < 1:
            raise ValueError("n_estimators and min_samples_leaf must be at least 1")
        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype=np.float64)
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if sample_weight.shape != (len(y),) or np.any(~np.isfinite(sample_weight)) \
                or np.any(sample_weight < 0) or not np.any(sample_weight > 0):
            raise ValueError("sample_weight must be 1D, non-negative, and contain at least one positive weight")
        if not 0.0 < self.tree_feature_ratio <= 1.0:
            raise ValueError("tree_feature_ratio must satisfy 0 < ratio <= 1")
        self.n_features_in_ = X.shape[1]
        self.feature_importances_ = np.zeros(X.shape[1], dtype=np.float64)
        indices = np.arange(len(y), dtype=int)
        rng = np.random.default_rng(self.random_state)
        self.trees_ = [self._build_tree(X, y, indices, 0, rng,
                                         self.feature_importances_, sample_weight)
                       for _ in range(self.n_estimators)]
        total = float(np.sum(self.feature_importances_))
        if total > 0:
            self.feature_importances_ /= total
        return self

    @staticmethod
    def _predict_tree(tree, X):
        prediction, feature, threshold, left, right = tree
        output = np.full(len(X), prediction, dtype=np.float64)
        if feature is None:
            return output
        left_mask = X[:, feature] <= threshold
        if np.any(left_mask):
            output[left_mask] = GroupedExtraTreesRegressor._predict_tree(
                left, X[left_mask]
            )
        if np.any(~left_mask):
            output[~left_mask] = GroupedExtraTreesRegressor._predict_tree(
                right, X[~left_mask]
            )
        return output

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.mean([self._predict_tree(tree, X) for tree in self.trees_], axis=0)


class TreeCoordinateEncoder:
    """Encode a formula tree as padded node rows.

    Legacy mode emits nodes in preorder.  Layered mode emits nodes by depth,
    reserving a fixed number of slots for each depth. ``x`` is the normalized
    inorder position of the node, ``y`` is normalized depth, ``symbol`` is a
    vocabulary code, and ``path`` is the binary heap-style path code
    (root=1, left=2, right=3).
    Ancestor-product rows additionally include operator composition and
    variable-occurrence ratios for the node's subtree.
    Zero is reserved for padding.
    """

    def __init__(
        self, max_nodes: int = 96, ancestor_products: bool = False,
        operator_count_mode: str = "count", max_nodes_per_level: int | None = None,
        nearest_ancestor_only: bool = False,
        include_composition_features: bool = True,
    ):
        self.max_nodes = int(max_nodes)
        self.ancestor_products = bool(ancestor_products)
        self.nearest_ancestor_only = bool(nearest_ancestor_only)
        self.include_composition_features = bool(include_composition_features)
        self.max_nodes_per_level = (
            None if max_nodes_per_level is None else int(max_nodes_per_level)
        )
        if self.max_nodes_per_level is not None and self.max_nodes_per_level < 1:
            raise ValueError("tree_coordinates maximum nodes per level must be at least 1")
        self.max_tree_levels = 0
        self.layer_node_caps: list[int] = []
        self.operator_count_mode = str(operator_count_mode).lower()
        if self.operator_count_mode not in {"count", "ratio"}:
            raise ValueError("operator_count_mode must be count or ratio")
        self.max_ancestor_count = 0
        self.symbol_codes = {"<PAD>": 0}
        self.standardization_means = None
        self.standardization_scales = None

    @staticmethod
    def layer_node_counts(formula: str) -> list[int]:
        """Count SymPy AST nodes at every zero-based tree depth."""
        counts: list[int] = []

        def visit(node: sp.Expr, depth: int) -> None:
            while len(counts) <= depth:
                counts.append(0)
            counts[depth] += 1
            for child in node.args:
                visit(child, depth + 1)

        visit(sp.sympify(formula), 0)
        return counts

    def compatibility_reason(self, formula: str) -> str | None:
        """Return why a formula cannot use this fitted layered encoding."""
        counts = self.layer_node_counts(formula)
        if (self.max_nodes_per_level is not None
                and any(count > self.max_nodes_per_level for count in counts)):
            return "nodes_per_level_exceeded"
        if not self.max_nodes_per_level:
            return None
        if len(counts) > self.max_tree_levels:
            return "tree_levels_exceeded"
        if any(count > self.layer_node_caps[depth]
               for depth, count in enumerate(counts)):
            return "layer_feature_capacity_exceeded"
        return None

    def fit(self, formulas):
        formulas = [str(formula) for formula in formulas]
        if not formulas:
            raise ValueError("tree_coordinates training formulas cannot be empty")
        if self.max_nodes_per_level is not None:
            counts_by_formula = [self.layer_node_counts(formula) for formula in formulas]
            if any(any(count > self.max_nodes_per_level for count in counts)
                   for counts in counts_by_formula):
                raise ValueError("A training formula contains a syntax tree that exceeds the per-level node limit")
            self.max_tree_levels = max(len(counts) for counts in counts_by_formula)
            self.layer_node_caps = [
                max(counts[depth] if depth < len(counts) else 0
                    for counts in counts_by_formula)
                for depth in range(self.max_tree_levels)
            ]
            # A layered representation reserves exactly the observed maximum
            # slots at each depth, never the legacy global max_nodes slots.
            self.max_nodes = sum(self.layer_node_caps)
        tokens = sorted({node_token(node)
                         for formula in formulas
                         for node in sp.preorder_traversal(sp.sympify(formula))})
        self.symbol_codes.update({token: index for index, token in enumerate(tokens, 1)})
        if self.ancestor_products:
            self.max_ancestor_count = 0
            all_vectors = []
            for formula in formulas:
                def visit(node, depth):
                    self.max_ancestor_count = max(self.max_ancestor_count, depth)
                    for child in node.args:
                        visit(child, depth + 1)
                visit(sp.sympify(formula), 0)
                coordinates = self._coordinate_rows(formula)
                all_vectors.extend(
                    self._expanded_node_vector(row, operator_counts, variable_ratios)
                    for row, _, operator_counts, variable_ratios in coordinates
                )
            matrix = np.asarray(all_vectors, dtype=np.float32)
            self.standardization_means = np.mean(matrix, axis=0).astype(np.float32)
            scales = np.std(matrix, axis=0).astype(np.float32)
            self.standardization_scales = np.where(scales > 0, scales, 1.0).astype(np.float32)
        return self

    def _coordinate_rows(self, formula):
        expr = sp.sympify(formula)
        operator_indices = {name: index for index, name in enumerate(OPERATORS)}
        operator_counts_cache = {}
        variable_counts_cache = {}

        def subtree_operator_counts(node):
            cached = operator_counts_cache.get(node)
            if cached is not None:
                return cached
            counts = np.zeros(len(OPERATORS), dtype=np.float32)
            name = operator_name(node)
            if name in operator_indices:
                counts[operator_indices[name]] += 1.0
            if node.is_Add:
                counts[operator_indices["subtract"]] += sum(
                    bool(arg.could_extract_minus_sign()) for arg in node.args
                )
            if node.is_Pow and not node.exp.free_symbols:
                try:
                    if float(node.exp) < 0:
                        counts[operator_indices["divide"]] += 1.0
                except (TypeError, ValueError, OverflowError):
                    pass
            for child in node.args:
                counts += subtree_operator_counts(child)
            operator_counts_cache[node] = counts
            return counts

        subtree_operator_counts(expr)

        def subtree_variable_counts(node):
            cached = variable_counts_cache.get(node)
            if cached is not None:
                return cached
            counts = np.zeros(len(VARIABLES), dtype=np.float32)
            if node.is_Symbol and str(node) in VARIABLES:
                counts[VARIABLES.index(str(node))] += 1.0
            for child in node.args:
                counts += subtree_variable_counts(child)
            variable_counts_cache[node] = counts
            return counts

        subtree_variable_counts(expr)
        leaf_count = max(1, sum(1 for node in sp.preorder_traversal(expr)
                                if not node.args))
        rows = []

        def walk(node, depth, path, left_leaf, ancestor_indices):
            if not node.args:
                center = left_leaf
            else:
                child_counts = [sum(1 for item in sp.preorder_traversal(child)
                                    if not item.args) for child in node.args]
                center = left_leaf + (sum(child_counts) - 1) / 2.0
            current_index = len(rows)
            coordinate = (center / max(1, leaf_count - 1),
                          depth / max(1, self.max_tree_levels - 1
                                      if self.max_nodes_per_level is not None
                                      else self.max_nodes - 1),
                          self.symbol_codes.get(node_token(node), 0), path)
            operator_values = subtree_operator_counts(node).copy()
            if self.operator_count_mode == "ratio":
                total_operators = float(np.sum(operator_values))
                if total_operators > 0:
                    operator_values /= total_operators
                else:
                    operator_values.fill(0.0)
            variable_values = subtree_variable_counts(node).copy()
            total_variables = float(np.sum(variable_values))
            if total_variables > 0:
                variable_values /= total_variables
            else:
                variable_values.fill(0.0)
            rows.append((coordinate, tuple(ancestor_indices), operator_values,
                         variable_values))
            cursor = left_leaf
            for index, child in enumerate(node.args):
                child_count = sum(1 for item in sp.preorder_traversal(child)
                                  if not item.args)
                walk(child, depth + 1, path * 2 + index, cursor,
                     ancestor_indices + (current_index,))
                cursor += child_count

        walk(expr, 0, 1, 0, ())
        return rows

    def _expanded_node_vector(self, row, operator_counts, variable_ratios):
        x, y, symbol, path = row
        one_hot = np.zeros(len(self.symbol_codes) - 1, dtype=np.float32)
        if symbol:
            one_hot[int(symbol) - 1] = 1.0
        values = [np.concatenate((np.asarray((x, y), dtype=np.float32), one_hot))]
        if self.include_composition_features:
            values.extend((operator_counts, variable_ratios))
        return np.concatenate(values)

    def _one(self, formula):
        if self.max_nodes_per_level is not None:
            reason = self.compatibility_reason(formula)
            if reason is not None:
                raise ValueError(f"Formula does not satisfy the layered tree-coordinate encoding constraints: {reason}")
            return self._one_layered(formula)
        if self.ancestor_products:
            coordinates = self._coordinate_rows(formula)
            vectors = [((self._expanded_node_vector(
                            row, operator_counts, variable_ratios
                        )
                        - self.standardization_means)
                        / self.standardization_scales).astype(np.float32)
                         for row, _, operator_counts, variable_ratios in coordinates]
            rows = np.zeros((self.max_nodes, self.feature_width), dtype=np.float32)
            for index, (vector, (_, ancestor_indices, _, _)) in enumerate(
                    zip(vectors[:self.max_nodes], coordinates[:self.max_nodes])):
                selected_ancestor_indices = (
                    ancestor_indices[-1:] if self.nearest_ancestor_only else ancestor_indices
                )
                ancestors = [vectors[ancestor_index]
                             for ancestor_index in selected_ancestor_indices]
                values = [vector]
                values.extend(vector * ancestor for ancestor in ancestors)
                flattened = np.concatenate(values)
                rows[index, :len(flattened)] = flattened
            return rows.reshape(-1)

        expr = sp.sympify(formula)
        nodes = []
        leaf_count = max(1, sum(1 for node in sp.preorder_traversal(expr)
                               if not node.args))

        def walk(node, depth, path, left_leaf, right_leaf, parent_row, ancestors):
            if not node.args:
                center = left_leaf
                next_leaf = left_leaf + 1
            else:
                child_counts = [sum(1 for item in sp.preorder_traversal(child)
                                    if not child.args) for child in node.args]
                center = left_leaf + (sum(child_counts) - 1) / 2.0
                next_leaf = left_leaf + sum(child_counts)
            current_row = (center / max(1, leaf_count - 1),
                           depth / max(1, self.max_nodes - 1),
                           self.symbol_codes.get(node_token(node), 0), path)
            if self.ancestor_products:
                current_vector = np.asarray(current_row, dtype=np.float32)
                products = tuple(
                    float(value)
                    for ancestor in ancestors
                    for value in current_vector * np.asarray(ancestor, dtype=np.float32)
                )
                products += (0.0,) * (4 * (self.max_ancestor_count - len(ancestors)))
                row = current_row + products
            else:
                row = current_row + parent_row
            nodes.append(row)
            cursor = left_leaf
            for index, child in enumerate(node.args):
                child_count = sum(1 for item in sp.preorder_traversal(child)
                                  if not item.args)
                walk(child, depth + 1, path * 2 + index + 0, cursor,
                     cursor + child_count, current_row, ancestors + (current_row,))
                cursor += child_count

        # The path formula above gives the first child path=2 and the second
        # child path=3; for n-ary SymPy nodes this remains deterministic.
        walk(expr, 0, 1, 0, leaf_count, (0, 0, 0, 0), ())
        row_width = (4 * (1 + self.max_ancestor_count)
                     if self.ancestor_products else 8)
        rows = np.zeros((self.max_nodes, row_width), dtype=np.float32)
        for index, row in enumerate(nodes[:self.max_nodes]):
            rows[index] = row
        return rows.reshape(-1)

    def _one_layered(self, formula: str) -> np.ndarray:
        """Encode nodes in depth-major slots with dynamic per-layer widths."""
        coordinates = self._coordinate_rows(formula)
        by_depth: list[list[tuple[int, tuple]]] = [
            [] for _ in range(self.max_tree_levels)
        ]
        for old_index, item in enumerate(coordinates):
            by_depth[len(item[1])].append((old_index, item))
        coordinate_by_old = {
            old_index: np.asarray(item[0], dtype=np.float32)
            for old_index, item in enumerate(coordinates)
        }
        rows = np.zeros((self.max_nodes, self.feature_width), dtype=np.float32)
        offset = 0
        for layer, capacity in zip(by_depth, self.layer_node_caps):
            # Padding belongs to its own level, so a shallow formula cannot
            # shift every deeper level into earlier feature slots.
            for within_level, (old_index, (coordinate, ancestors, operator_counts,
                                           variable_ratios)) in enumerate(layer):
                slot = offset + within_level
                if self.ancestor_products:
                    vector = ((self._expanded_node_vector(
                        coordinate, operator_counts, variable_ratios
                    ) - self.standardization_means) / self.standardization_scales).astype(np.float32)
                    values = [vector]
                    selected_ancestor_indices = (
                        ancestors[-1:] if self.nearest_ancestor_only else ancestors
                    )
                    values.extend(vector * ((self._expanded_node_vector(
                        coordinates[ancestor][0], coordinates[ancestor][2], coordinates[ancestor][3]
                    ) - self.standardization_means) / self.standardization_scales)
                                  for ancestor in selected_ancestor_indices)
                    flattened = np.concatenate(values)
                    rows[slot, :len(flattened)] = flattened
                else:
                    parent = (coordinate_by_old[ancestors[-1]] if ancestors
                              else np.zeros(4, dtype=np.float32))
                    rows[slot] = np.concatenate((np.asarray(coordinate, dtype=np.float32), parent))
            offset += capacity
        return rows.reshape(-1)

    def transform(self, formulas, n_jobs: int | None = None):
        formulas = list(formulas)
        if n_jobs == 1 or len(formulas) < 2:
            features = [self._one(formula) for formula in formulas]
        else:
            workers = (
                min(8, max(1, (os.cpu_count() or 2) - 1))
                if n_jobs is None or int(n_jobs) < 0 else int(n_jobs)
            )
            workers = min(max(1, int(workers)), len(formulas))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                features = list(executor.map(self._one, formulas))
        return np.asarray(features, dtype=np.float32)

    @property
    def feature_width(self):
        if self.ancestor_products:
            # When only the nearest ancestor is requested, do not retain
            # padded columns for all deeper ancestor levels.  This makes the
            # switch reduce the actual model input width, rather than merely
            # filling the unused columns with zeros.
            ancestor_slots = 1 if self.nearest_ancestor_only else self.max_ancestor_count
            return len(self._ancestor_field_names) * (1 + ancestor_slots)
        return 8

    @property
    def _ancestor_field_names(self):
        fields = ("x", "y") + tuple(
            f"symbol_onehot_{index:03d}"
            for index in range(1, len(self.symbol_codes))
        )
        if self.include_composition_features:
            fields += tuple(
                f"operator_{operator}_{self.operator_count_mode}"
                for operator in OPERATORS
            ) + tuple(f"variable_{variable}_ratio" for variable in VARIABLES)
        return fields

    @property
    def feature_names(self):
        if self.ancestor_products:
            ancestor_count = (
                1 if self.nearest_ancestor_only else self.max_ancestor_count
            )
            fields = self._ancestor_field_names + tuple(
                f"ancestor_{ancestor:03d}_product_{field}"
                for ancestor in range(1, ancestor_count + 1)
                for field in self._ancestor_field_names
            )
        else:
            fields = ("x", "y", "symbol", "path", "parent_x", "parent_y",
                      "parent_symbol", "parent_path")
        return [f"node_{node:03d}_{field}"
                for node in range(self.max_nodes)
                for field in fields]


class ExperimentalPrimeTreeCoordinateEncoder(TreeCoordinateEncoder):
    """Experimental tree-coordinate encoder using primes for node symbols.

    This intentionally leaves :class:`TreeCoordinateEncoder` untouched.  It
    replaces only its ``symbol_onehot_*`` fields with one ``symbol_prime``
    field.  For each ancestor slot, non-symbol fields retain the existing
    elementwise product, while the symbol field is ``log(product(primes))``
    along the node-to-ancestor path.  Summing logarithms is mathematically
    identical to multiplying primes first and then taking ``log`` but avoids
    overflow for deep trees.
    """

    _PRIME_FIELD_INDEX = 2
    # Stable assignments for the vocabulary emitted by the formula generator.
    # Additional observed tokens receive the next unused primes in sorted-token
    # order and are persisted with the fitted encoder.
    FIXED_SYMBOL_PRIMES = {
        "FIT_CONST": 2,
        "VAR:x1": 3, "VAR:x2": 5, "VAR:x3": 7, "VAR:x4": 11,
        "INT:-2": 13, "INT:-1": 17, "INT:0": 19, "INT:1": 23,
        "INT:2": 29, "NUMBER": 31,
        "OP:Abs/1": 37, "OP:Add/2": 41, "OP:Add/3": 43,
        "OP:Mul/2": 47, "OP:Mul/3": 53, "OP:Pow/2": 59,
        "OP:cos/1": 61, "OP:log/1": 67, "OP:sin/1": 71,
    }

    @staticmethod
    def _first_primes(count: int) -> list[int]:
        """Return a deterministic, reusable sequence of the first primes."""
        primes: list[int] = []
        candidate = 2
        while len(primes) < count:
            limit = int(math.isqrt(candidate))
            if all(candidate % divisor for divisor in range(2, limit + 1)):
                primes.append(candidate)
            candidate += 1
        return primes

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.symbol_primes = {"<PAD>": 0}

    def fit(self, formulas):
        # The base implementation retains its stable token vocabulary,
        # layered capacities, and standardization of all non-prime fields.
        super().fit(formulas)
        tokens = sorted(token for token in self.symbol_codes if token != "<PAD>")
        assigned = {
            token: self.FIXED_SYMBOL_PRIMES[token]
            for token in tokens if token in self.FIXED_SYMBOL_PRIMES
        }
        unknown_tokens = [token for token in tokens if token not in assigned]
        used_primes = set(self.FIXED_SYMBOL_PRIMES.values())
        available_primes = [prime for prime in self._first_primes(
            len(used_primes) + len(unknown_tokens) + 16
        ) if prime not in used_primes]
        assigned.update(zip(unknown_tokens, available_primes))
        self.symbol_primes = {"<PAD>": 0, **assigned}

        # ``super().fit`` calculated this column before the prime mapping was
        # populated.  Preserve raw primes in the output rather than z-scoring
        # them; coordinate and composition fields keep their original
        # standardization/product behavior.
        if self.standardization_means is not None:
            self.standardization_means[self._PRIME_FIELD_INDEX] = 0.0
            self.standardization_scales[self._PRIME_FIELD_INDEX] = 1.0
        return self

    def _prime_for_symbol_code(self, symbol_code: int | float) -> float:
        if not symbol_code:
            return 0.0
        inverse_codes = {code: token for token, code in self.symbol_codes.items()}
        return float(self.symbol_primes.get(inverse_codes[int(symbol_code)], 0))

    def _expanded_node_vector(self, row, operator_counts, variable_ratios):
        x, y, symbol, _path = row
        values = [np.asarray((x, y, self._prime_for_symbol_code(symbol)), dtype=np.float32)]
        if self.include_composition_features:
            values.extend((operator_counts, variable_ratios))
        return np.concatenate(values)

    def _standardized_vector(self, raw_vector: np.ndarray) -> np.ndarray:
        vector = ((raw_vector - self.standardization_means)
                  / self.standardization_scales).astype(np.float32)
        # Raw prime code is categorical identity, not a continuous z-score.
        vector[self._PRIME_FIELD_INDEX] = raw_vector[self._PRIME_FIELD_INDEX]
        return vector

    def _encoded_rows(self, formula: str) -> tuple[list[tuple], list[np.ndarray]]:
        coordinates = self._coordinate_rows(formula)
        raw_vectors = [self._expanded_node_vector(row, operators, variables)
                       for row, _, operators, variables in coordinates]
        return coordinates, [self._standardized_vector(vector) for vector in raw_vectors]

    def _node_feature_row(self, node_index: int, coordinates: list[tuple],
                          vectors: list[np.ndarray]) -> np.ndarray:
        _coordinate, ancestor_indices, _operators, _variables = coordinates[node_index]
        node_vector = vectors[node_index]
        values = [node_vector]
        node_prime = float(node_vector[self._PRIME_FIELD_INDEX])
        path_log_prime = math.log(node_prime) if node_prime > 0 else 0.0
        prime_path_valid = node_prime > 0
        selected_ancestor_indices = (
            ancestor_indices[-1:] if self.nearest_ancestor_only else ancestor_indices
        )
        for ancestor_index in selected_ancestor_indices:
            ancestor = vectors[ancestor_index]
            product = node_vector * ancestor
            # This is log(p_node * p_parent * ... * p_ancestor), evaluated as
            # a sum of logs to keep very deep trees finite.
            ancestor_prime = float(ancestor[self._PRIME_FIELD_INDEX])
            if prime_path_valid and ancestor_prime > 0:
                path_log_prime += math.log(ancestor_prime)
            else:
                path_log_prime = 0.0
                prime_path_valid = False
            product[self._PRIME_FIELD_INDEX] = path_log_prime
            values.append(product)
        return np.concatenate(values)

    def _one(self, formula):
        if self.max_nodes_per_level is not None:
            reason = self.compatibility_reason(formula)
            if reason is not None:
                raise ValueError(f"Formula does not satisfy the layered tree-coordinate encoding constraints: {reason}")
            return self._one_layered(formula)
        coordinates, vectors = self._encoded_rows(formula)
        rows = np.zeros((self.max_nodes, self.feature_width), dtype=np.float32)
        for index in range(min(self.max_nodes, len(coordinates))):
            flattened = self._node_feature_row(index, coordinates, vectors)
            rows[index, :len(flattened)] = flattened
        return rows.reshape(-1)

    def _one_layered(self, formula: str) -> np.ndarray:
        coordinates, vectors = self._encoded_rows(formula)
        by_depth: list[list[int]] = [[] for _ in range(self.max_tree_levels)]
        for old_index, item in enumerate(coordinates):
            by_depth[len(item[1])].append(old_index)
        rows = np.zeros((self.max_nodes, self.feature_width), dtype=np.float32)
        offset = 0
        for layer, capacity in zip(by_depth, self.layer_node_caps):
            for within_level, old_index in enumerate(layer):
                flattened = self._node_feature_row(old_index, coordinates, vectors)
                rows[offset + within_level, :len(flattened)] = flattened
            offset += capacity
        return rows.reshape(-1)

    @property
    def _ancestor_field_names(self):
        fields = ("x", "y", "symbol_prime")
        if self.include_composition_features:
            fields += tuple(f"operator_{operator}_{self.operator_count_mode}"
                            for operator in OPERATORS)
            fields += tuple(f"variable_{variable}_ratio" for variable in VARIABLES)
        return fields

    @property
    def feature_names(self):
        ancestor_count = (
            1 if self.nearest_ancestor_only else self.max_ancestor_count
        )
        fields = self._ancestor_field_names + tuple(
            (f"ancestor_{ancestor:03d}_log_prime_product"
             if field == "symbol_prime"
             else f"ancestor_{ancestor:03d}_product_{field}")
            for ancestor in range(1, ancestor_count + 1)
            for field in self._ancestor_field_names
        )
        return [f"node_{node:03d}_{field}"
                for node in range(self.max_nodes)
                for field in fields]


def _canonical_model_type(model_type: str) -> str:
    model_type = str(model_type).lower()
    return model_type


def validate_ranking_dependencies(model_type: str) -> None:
    """Fail before costly LM fitting when the selected ranker is unavailable."""
    try:
        import joblib  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Structure ranking requires scikit-learn/joblib; use the project's py39 environment") from exc
    model_type = _canonical_model_type(model_type)
    if model_type == "xgboost":
        try:
            import xgboost  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("XGBoost ranking requires xgboost; use the project's py39 environment") from exc


def syntax_node_token(node: sp.Expr) -> str:
    """Backward-compatible public node tokenizer."""
    return node_token(node)


def preorder_node_sequence(formula: str, max_nodes: int) -> list[str]:
    extractor = FormulaStructureFeatureExtractor(max_nodes=max_nodes)
    categorical, _ = extractor.extract_one(formula)
    return categorical[:max_nodes]


def _training_percentile_examples_with_r2(
    payload: dict, label_field: str = "r2",
) -> list[tuple[str, float, float, str]]:
    """Convert fitted R² values to within-dataset percentile labels.

    All fitted single and combination formulas share one ranking pool inside
    each dataset.  The best R² receives 100 and the worst receives 0.
    Repeated formula observations from different datasets remain separate
    training examples.
    """
    by_dataset: dict[str, list[tuple[str, float]]] = {}
    for key in ("single_best", "combination_best"):
        for row in payload.get(key, []):
            formula_id, value = row.get("formula_id"), row.get(label_field)
            if formula_id is None or value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                dataset_name = str(row.get("dataset_name") or "__all_datasets__")
                by_dataset.setdefault(dataset_name, []).append((str(formula_id), value))

    examples = []
    for observations in by_dataset.values():
        values = np.asarray([value for _, value in observations], dtype=float)
        ranks = rankdata(values, method="average")
        denominator = max(1, len(observations) - 1)
        dataset_name = next(
            name for name, values_for_dataset in by_dataset.items()
            if values_for_dataset is observations
        )
        examples.extend(
            (formula_id, float(100.0 * (rank - 1.0) / denominator),
             float(value), dataset_name)
            for (formula_id, value), rank in zip(observations, ranks)
        )
    return examples


def _training_percentile_examples(payload: dict, label_field: str = "r2") -> list[tuple[str, float]]:
    return [
        (formula_id, percentile)
        for formula_id, percentile, _r2, _dataset in
        _training_percentile_examples_with_r2(payload, label_field)
    ]


def _training_labels(payload: dict, label_field: str = "r2") -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for formula_id, percentile in _training_percentile_examples(payload, label_field):
        grouped.setdefault(formula_id, []).append(percentile)
    # For display, use the conservative percentile when one formula was fit
    # on multiple datasets. Training itself retains every observation below.
    return {formula_id: min(values) for formula_id, values in grouped.items()}


def _training_examples(payload: dict,
                       label_field: str = "r2") -> list[tuple[str, float]]:
    return _training_percentile_examples(payload, label_field)


def _training_actual_r2(payload: dict) -> dict[str, float]:
    """Return the conservative actual LM R² for every fitted formula.

    This is intentionally separate from the percentile labels used to train
    the structure ranker.  The hybrid selector needs the measured fit quality
    when choosing its small direct-fit set.
    """
    values: dict[str, list[float]] = {}
    for key in ("single_best", "combination_best"):
        for row in payload.get(key, []):
            formula_id = row.get("formula_id")
            if formula_id is None:
                continue
            value = row.get("pooled_r2", row.get("r2"))
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.setdefault(str(formula_id), []).append(value)
    return {formula_id: min(scores) for formula_id, scores in values.items()}




def _formula_pool(payload: dict) -> list[dict]:
    pool = []
    for family, key in (("single", "single_candidates"), ("combination", "binary_combinations")):
        for row in payload.get(key, []):
            pool.append({
                "formula_id": row["id"], "formula_family": family,
                "readable_expression": row["readable_expression"],
                "tree_readable_expression": row.get(
                    "tree_readable_expression", row["readable_expression"]
                ),
                "sympy_expression": row["sympy_expression"],
                "structure_type": row["structure_type"],
                "source_formula_id": row.get("source_formula_id"),
                "combination_type": row.get("combination_type"),
                "operator": row.get("operator"), "operators": row.get("operators"),
                "base_ids": row.get("base_ids"),
            })
    return pool



def _make_model(config: dict, seed: int):
    model_type = _canonical_model_type(config.get("model_type", "extra_trees"))
    n_estimators = int(config.get("n_estimators", 400))
    n_jobs = int(config.get("n_jobs", -1))
    if model_type == "extra_trees":
        if config.get("grouped_extra_trees_enabled", False):
            model = GroupedExtraTreesRegressor(
                n_estimators=n_estimators, max_depth=config.get("max_depth"),
                min_samples_leaf=int(config.get("min_samples_leaf", 2)),
                tree_feature_count=int(config.get("grouped_tree_feature_count", 0)),
                tree_feature_ratio=float(config.get("grouped_tree_feature_ratio", 0.5)),
                random_state=seed, n_jobs=n_jobs,
            )
            return model_type, model
        from sklearn.ensemble import ExtraTreesRegressor
        model = ExtraTreesRegressor(
            n_estimators=n_estimators, max_depth=config.get("max_depth"),
            min_samples_leaf=int(config.get("min_samples_leaf", 2)),
            max_features=config.get("max_features", 1.0), random_state=seed, n_jobs=n_jobs,
        )
    elif model_type == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(
            objective="reg:squarederror", n_estimators=n_estimators,
            max_depth=int(config.get("max_depth") or 6),
            learning_rate=float(config.get("learning_rate", 0.05)),
            subsample=float(config.get("subsample", 0.9)),
            colsample_bytree=float(config.get("colsample_bytree", 0.9)),
            reg_lambda=float(config.get("reg_lambda", 1.0)), random_state=seed, n_jobs=n_jobs,
        )
    else:
        raise ValueError(
            "ranking.model_type must be extra_trees or xgboost"
        )
    return model_type, model


def _fit(model, features, indices, labels, sample_weights=None):
    if sample_weights is None:
        model.fit(features[indices], labels)
    else:
        weights = np.asarray(sample_weights, dtype=float)
        if weights.shape != (len(labels),):
            raise ValueError("sample_weights length must match labels")
        model.fit(features[indices], labels, sample_weight=weights)


def _rank_metrics(y_true: np.ndarray, prediction: np.ndarray, top_k: int) -> dict:
    if len(y_true) < 2:
        return {"spearman": None, "top_k_overlap": None, "samples": len(y_true)}
    rho = float(spearmanr(y_true, prediction).statistic)
    k = min(max(1, top_k), len(y_true))
    true_top, predicted_top = set(np.argsort(-y_true)[:k]), set(np.argsort(-prediction)[:k])
    return {"spearman": rho if math.isfinite(rho) else None, "top_k": k,
            "top_k_overlap": len(true_top & predicted_top) / k, "samples": len(y_true)}


def _save_importance(path: Path, names: list[str], values) -> None:
    rows = sorted(zip(names, map(float, values)), key=lambda item: item[1], reverse=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("rank", "feature", "importance"))
        writer.writerows((rank, name, value) for rank, (name, value) in enumerate(rows, 1))


def _feature_importance_categories(names: Sequence[str], values) -> dict[str, float]:
    """Aggregate ExtraTrees importance by the four operator-mode groups."""
    return {
        category: stats["total"]
        for category, stats in _feature_importance_category_stats(names, values).items()
    }


def _feature_importance_category_stats(
    names: Sequence[str], values,
) -> dict[str, dict[str, float | int]]:
    """Return total, count, and per-feature mean importance by group."""
    categories = {
        category: {"total": 0.0, "count": 0}
        for category in (
            "operator_primitive_matrix", "response_svd",
            "engineered_structure_numeric", "tree_kernel_pca",
        )
    }
    engineered_prefixes = (
        "node_count", "tree_depth", "max_consecutive_unary_depth",
        "depth_", "left_subtree_", "right_subtree_",
        "left_right_", "operator_", "x1_", "x2_", "x3_", "x4_",
    )
    for name, value in zip(names, values):
        name = str(name)
        if name.startswith("operator_primitive_") or name.startswith("merged_operator_primitive_"):
            category = "operator_primitive_matrix"
        elif name.startswith("response_svd_"):
            category = "response_svd"
        elif name.startswith("tree_kernel_pca_component_"):
            category = "tree_kernel_pca"
        elif name.startswith(engineered_prefixes):
            category = "engineered_structure_numeric"
        else:
            continue
        categories[category]["total"] += float(value)
        categories[category]["count"] += 1
    for stats in categories.values():
        stats["mean"] = (
            stats["total"] / stats["count"] if stats["count"] else 0.0
        )
    return categories




def _feature_group(name: str) -> str | None:
    name = str(name)
    if name.startswith(("operator_primitive_", "merged_operator_primitive_")):
        return "operator_primitive_matrix"
    if name.startswith("response_svd_"):
        return "response_svd"
    if name.startswith("tree_kernel_pca_component_"):
        return "tree_kernel_pca"
    engineered_prefixes = (
        "node_count", "tree_depth", "max_consecutive_unary_depth",
        "depth_", "left_subtree_", "right_subtree_", "left_right_",
        "operator_", "x1_", "x2_", "x3_", "x4_",
    )
    if name.startswith(engineered_prefixes):
        return "engineered_structure_numeric"
    return None


def _apply_feature_group_ablation(
    features: np.ndarray, names: Sequence[str], config: dict,
) -> tuple[np.ndarray, list[str]]:
    """Remove requested feature groups while preserving the feature order."""
    removed = {
        str(value).strip().lower()
        for value in config.get("feature_ablation_remove", [])
    }
    if not removed:
        return features, list(names)
    valid = {
        "operator_primitive_matrix", "response_svd",
        "engineered_structure_numeric", "tree_kernel_pca",
    }
    unknown = removed - valid
    if unknown:
        raise ValueError(f"Unknown tree-feature ablation categories: {sorted(unknown)}")
    keep = np.asarray([
        _feature_group(name) not in removed for name in names
    ], dtype=bool)
    if not np.any(keep):
        raise ValueError("No features remain after tree-feature ablation")
    return np.asarray(features)[:, keep], [name for name, flag in zip(names, keep) if flag]


def _resolve_feature_mode(config: dict) -> str:
    mode = str(config.get("feature_mode", "operator_primitive_matrix")).lower()
    aliases = {
        "default": "model_default",
        "tree": "tree_coordinates",
        "tree_coordinate": "tree_coordinates",
        "tree_coordinates_prime": "tree_coordinates_prime_experimental",
        "prime_experimental": "tree_coordinates_prime_experimental",
        "response_svd": "value_matrix_svd",
        "numeric_response_svd": "value_matrix_svd",
        "value_svd": "value_matrix_svd",
        "global_numeric_tree": "tree_coordinates_global_numeric",
        "operator_primitive": "operator_primitive_matrix",
        "operator_primitive_incidence": "operator_primitive_matrix",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"model_default", "tree_coordinates",
                    "tree_coordinates_prime_experimental", "value_matrix_svd",
                    "tree_coordinates_global_numeric", "operator_primitive_matrix"}:
        raise ValueError(
            "ranking.feature_mode must be model_default, tree_coordinates, "
            "tree_coordinates_prime_experimental, tree_coordinates_global_numeric, "
            "value_matrix_svd, or operator_primitive_matrix"
        )
    return mode


def _is_explicit_tree_coordinate_mode(feature_mode: str) -> bool:
    return feature_mode in {"tree_coordinates", "tree_coordinates_prime_experimental",
                            "tree_coordinates_global_numeric"}


def _is_global_numeric_tree_mode(feature_mode: str) -> bool:
    return feature_mode == "tree_coordinates_global_numeric"


def _resolve_prime_feature_variant(
    config: dict, feature_mode: str, prompt_user: bool = False,
) -> str:
    """Choose the representation inside the experimental prime mode only."""
    if feature_mode != "tree_coordinates_prime_experimental":
        return "tree"
    configured_value = str(config.get(
        "tree_coordinates_prime_experimental_feature_mode",
        config.get("prime_experimental_feature_mode", "both"),
    )).lower().strip()
    value = configured_value
    if prompt_user and sys.stdin.isatty():
        try:
            value = input(
                "Experimental tree-coordinate feature mode [tree/numeric/both]; press Enter to use "
                f"{configured_value or 'tree'}: "
            ).strip().lower() or configured_value
        except (EOFError, OSError):
            value = configured_value
    aliases = {
        "tree_coordinates": "tree", "tree_coordinate": "tree", "original": "tree",
        "numeric": "numeric", "value": "numeric", "value_matrix_svd": "numeric",
        "response_svd": "numeric", "svd": "numeric",
        "both": "both", "combined": "both", "tree_and_numeric": "both",
    }
    value = aliases.get(value, value)
    if value not in {"tree", "numeric", "both"}:
        raise ValueError(
            "ranking.tree_coordinates_prime_experimental_feature_mode must be "
            "tree, numeric, or both"
        )
    return value


def _tree_coordinate_encoder_class(feature_mode: str):
    return (ExperimentalPrimeTreeCoordinateEncoder
            if feature_mode in {"tree_coordinates_prime_experimental",
                                "tree_coordinates_global_numeric"}
            else TreeCoordinateEncoder)


def _fit_tree_feature_pca(
    features: np.ndarray,
    components: int = 64,
    fit_samples: int = 5000,
    random_state: int = 0,
    transform_batch_size: int = 1024,
    method: str = "kernel",
):
    """Fit the configured tree-feature PCA.

    Keep the requested dimension bounded by the available feature matrix so
    small formula pools remain valid. For large formula pools, fit on a
    reproducible random subset and transform the complete pool in batches.
    Mode 3 selects ordinary randomized PCA; the legacy Kernel PCA path is
    retained for the other modes.
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or not features.shape[0] or not features.shape[1]:
        raise ValueError("Kernel PCA input must be a non-empty 2D feature matrix")
    requested = int(components)
    if requested < 1:
        raise ValueError("kernel_pca_components must be at least 1")
    fit_samples = int(fit_samples)
    if fit_samples < 1:
        raise ValueError("kernel_pca_fit_samples must be at least 1")
    transform_batch_size = int(transform_batch_size)
    if transform_batch_size < 1:
        raise ValueError("kernel_pca_transform_batch_size must be at least 1")
    method = str(method).strip().lower()
    if method not in {"kernel", "pca"}:
        raise ValueError("tree_pca_method must be kernel or pca")

    sample_count = min(fit_samples, features.shape[0])
    if sample_count < features.shape[0]:
        rng = np.random.default_rng(int(random_state))
        sample_indices = np.sort(
            rng.choice(features.shape[0], size=sample_count, replace=False)
        )
        fit_features = features[sample_indices]
    else:
        fit_features = features

    n_components = min(requested, sample_count, features.shape[1])
    if method == "pca":
        from sklearn.decomposition import PCA

        pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=int(random_state),
        )
    else:
        from sklearn.decomposition import KernelPCA

        pca = KernelPCA(
            n_components=n_components,
            kernel="rbf",
            gamma=None,
            fit_inverse_transform=False,
            eigen_solver="auto",
            remove_zero_eig=True,
        )
    pca.fit(fit_features)

    output_width = (
        pca.eigenvectors_.shape[1]
        if hasattr(pca, "eigenvectors_")
        else pca.components_.shape[0]
    )
    transformed = np.empty(
        (features.shape[0], output_width), dtype=np.float64
    )
    for start in range(0, features.shape[0], transform_batch_size):
        end = min(start + transform_batch_size, features.shape[0])
        transformed[start:end] = pca.transform(features[start:end])
    return pca, transformed


def _precomputed_feature_descriptor(
    formulas: Sequence[str], config: dict, feature_mode: str,
    prime_variant: str, include_operator_tree_pca: bool,
) -> dict:
    """Return the exact descriptor used to validate a frozen feature pack."""
    return {
        "formulas": list(formulas),
        "feature_mode": feature_mode,
        "max_nodes": int(config.get("max_nodes", 96)),
        "tree_max_nodes_per_level": config.get("tree_max_nodes_per_level"),
        "nearest_ancestor_only": bool(config.get("nearest_ancestor_only", False)),
        "tree_coordinate_operator_count_mode": str(
            config.get("tree_coordinate_operator_count_mode", "ratio")
        ),
        "prime_feature_mode": prime_variant,
        "pca_tree_features": bool(config.get("pca_tree_features", False)),
        "kernel_pca_components": int(config.get("kernel_pca_components", 64)),
        "response_svd_grid_size": int(config.get("response_svd_grid_size", 100)),
        "response_svd_feature_count": int(
            config.get("response_svd_feature_count", 20)
        ),
        "include_engineered_structure_numeric": bool(
            config.get("include_engineered_structure_numeric", True)
        ),
        "operator_primitive_tree_pca": include_operator_tree_pca,
        "tree_pca_method": str(config.get("tree_pca_method", "kernel")).lower(),
        "operator_primitive_numeric_encoding": "response_svd_plus_engineered_v2",
    }


def build_precomputed_feature_artifact(
    pool: Sequence[dict], payload: dict, config: dict, seed: int,
) -> dict:
    """Build the frozen feature representation used by the current pipeline.

    The precomputation command intentionally supports the current production
    feature mode (operator primitive matrix plus tree PCA).  Adding another
    feature mode must be explicit so a pack can never silently contain a
    representation different from the one expected by the pipeline.
    """
    import joblib  # noqa: F401  # validates the runtime dependency early

    pool = list(pool)
    formulas = [str(row["readable_expression"]) for row in pool]
    feature_mode = _resolve_feature_mode(config)
    include_operator_tree_pca = (
        feature_mode == "operator_primitive_matrix"
        and bool(config.get("operator_primitive_tree_pca_enabled", True))
    )
    if feature_mode != "operator_primitive_matrix" or not include_operator_tree_pca:
        raise ValueError(
            "The precomputation script currently supports only operator_primitive_matrix + tree PCA; "
            f"current feature_mode={feature_mode!r}, "
            f"tree_pca={include_operator_tree_pca}"
        )
    prime_variant = _resolve_prime_feature_variant(
        config, feature_mode, prompt_user=False
    )
    extractor = FormulaStructureFeatureExtractor(
        max_nodes=int(config.get("max_nodes", 96)),
        max_depth=int(config.get("engineered_max_depth", 12)),
    )
    categorical, numeric = extractor.transform(formulas, n_jobs=1)
    base_records = payload.get("bases", [])
    matrix_encoder = OperatorPrimitiveMatrixEncoder(base_records)
    matrix_features = matrix_encoder.transform(pool)
    response_encoder = FormulaResponseSVDEncoder(
        grid_size=int(config.get("response_svd_grid_size", 100)),
        feature_count=int(config.get("response_svd_feature_count", 20)),
    )
    response_features = response_encoder.transform(formulas, n_jobs=1)
    include_engineered_numeric = bool(
        config.get("include_engineered_structure_numeric", True)
    )
    feature_blocks = [matrix_features, response_features]
    if include_engineered_numeric:
        feature_blocks.append(numeric)
    base_features = np.concatenate(feature_blocks, axis=1)
    tree_encoder = TreeCoordinateEncoder(
        int(config.get("max_nodes", 96)),
        ancestor_products=True,
        operator_count_mode=str(
            config.get("tree_coordinate_operator_count_mode", "ratio")
        ),
        nearest_ancestor_only=bool(config.get("nearest_ancestor_only", False)),
    ).fit(formulas)
    raw_tree_features = tree_encoder.transform(formulas, n_jobs=1)
    tree_feature_pca, tree_features = _fit_tree_feature_pca(
        raw_tree_features,
        config.get("kernel_pca_components", 64),
        config.get("kernel_pca_fit_samples", 5000),
        int(seed),
        config.get("kernel_pca_transform_batch_size", 1024),
        method=config.get("tree_pca_method", "pca"),
    )
    features = np.concatenate((base_features, tree_features), axis=1)
    model_feature_names = matrix_encoder.feature_names + response_encoder.feature_names
    if include_engineered_numeric:
        model_feature_names += extractor.numeric_names
    model_feature_names += [
        f"tree_kernel_pca_component_{index + 1:03d}"
        for index in range(tree_features.shape[1])
    ]
    descriptor = _precomputed_feature_descriptor(
        formulas, config, feature_mode, prime_variant, include_operator_tree_pca
    )
    return {
        "version": 2,
        "formula_ids": [str(row["id"]) for row in pool],
        "cache_descriptor": descriptor,
        "features": np.asarray(features, dtype=np.float32),
        # Categorical node tokens are strings such as ``OP:Mul/2``; preserve
        # them as an object array for the ranking output instead of coercing
        # them to numeric values.
        "categorical_features": np.asarray(categorical, dtype=object),
        "model_feature_names": list(model_feature_names),
        "tree_encoder": tree_encoder,
        "response_svd_encoder": response_encoder,
        "operator_primitive_encoder": matrix_encoder,
        "tree_feature_pca": tree_feature_pca,
        "precompute_seed": int(seed),
    }


def load_precomputed_feature_artifact(
    path: str | Path, pool: Sequence[dict], config: dict,
) -> dict:
    """Load and strictly validate a frozen feature representation."""
    import joblib

    artifact_path = Path(path)
    if not artifact_path.is_absolute():
        artifact_path = Path(__file__).resolve().parent / artifact_path
    if not artifact_path.is_file():
        raise FileNotFoundError(
            "Precomputed formula/feature artifact does not exist; run "
            f"precompute_formula_features_bivariate.py：{artifact_path}"
        )
    try:
        artifact = joblib.load(artifact_path)
    except Exception as exc:
        raise RuntimeError(f"Could not read precomputed formula/feature artifact: {artifact_path}") from exc
    if not isinstance(artifact, dict) or int(artifact.get("version", -1)) != 2:
        raise ValueError(f"Unsupported precomputed artifact version: {artifact_path}")
    pool = list(pool)
    expected_ids = []
    for row in pool:
        formula_id = row.get("id", row.get("formula_id"))
        if formula_id is None:
            raise ValueError("Current formula-pool record is missing the id/formula_id field")
        expected_ids.append(str(formula_id))
    artifact_ids = [str(value) for value in artifact.get("formula_ids", [])]
    if set(artifact_ids) != set(expected_ids) or len(artifact_ids) != len(expected_ids):
        raise ValueError(
            "The formula ID set in the precomputed features does not match the current formula library; "
            "regenerate the precomputed artifact using the current configuration."
        )
    feature_mode = _resolve_feature_mode(config)
    include_operator_tree_pca = (
        feature_mode == "operator_primitive_matrix"
        and bool(config.get("operator_primitive_tree_pca_enabled", True))
    )
    prime_variant = _resolve_prime_feature_variant(
        config, feature_mode, prompt_user=False
    )
    expected_descriptor = _precomputed_feature_descriptor(
        [str(row["readable_expression"]) for row in pool],
        config, feature_mode, prime_variant, include_operator_tree_pca,
    )
    artifact_descriptor = artifact.get("cache_descriptor")
    # Feature extraction is also configured independently during
    # precomputation.  Keep the descriptor for provenance, but validate the
    # formula list and matrix below instead of comparing it with the runtime
    # pipeline ranking config.
    descriptor_matches = isinstance(artifact_descriptor, dict)
    if descriptor_matches:
        actual_formulas = sorted(artifact_descriptor.get("formulas", []))
        expected_formulas = sorted(expected_descriptor.get("formulas", []))
        descriptor_matches = actual_formulas == expected_formulas
    if not descriptor_matches:
        raise ValueError(
            "The formula-expression list in the precomputed features does not match the current formula library; "
            "rerun the precomputation script with the final formula-generation configuration."
        )
    features = np.asarray(artifact.get("features"), dtype=np.float32)
    categorical = np.asarray(artifact.get("categorical_features"), dtype=object)
    names = artifact.get("model_feature_names")
    if features.ndim != 2 or features.shape[0] != len(pool):
        raise ValueError("The number of rows in the precomputed feature matrix does not match the number of formulas")
    if categorical.ndim != 2 or categorical.shape[0] != len(pool):
        raise ValueError("The number of rows in the precomputed node features does not match the number of formulas")
    if not isinstance(names, list) or len(names) != features.shape[1]:
        raise ValueError("The number of precomputed feature names does not match the number of feature-matrix columns")
    if artifact_ids != expected_ids:
        row_by_id = {formula_id: index for index, formula_id in enumerate(artifact_ids)}
        reorder = [row_by_id[formula_id] for formula_id in expected_ids]
        features = features[reorder]
        categorical = categorical[reorder]
    artifact["formula_ids"] = expected_ids
    artifact["features"] = features
    artifact["categorical_features"] = categorical
    artifact["artifact_path"] = str(artifact_path)
    return artifact


def _diagnostic_features(pool: list[dict], config: dict, n_jobs: int | None = 1):
    """Return exactly the numeric feature representation used by the mode."""
    max_nodes = int(config.get("max_nodes", 96))
    max_depth = int(config.get("engineered_max_depth", 12))
    formulas = [row.get("tree_readable_expression", row["readable_expression"]) for row in pool]
    feature_mode = _resolve_feature_mode(config)
    model_type = _canonical_model_type(config.get("model_type", "extra_trees"))
    if feature_mode == "operator_primitive_matrix":
        base_records = config.get("_operator_primitive_base_records")
        if base_records is None:
            raise ValueError(
                "operator_primitive_matrix requires bases records from the formula-generation stage"
            )
        matrix_encoder = OperatorPrimitiveMatrixEncoder(base_records)
        response_encoder = FormulaResponseSVDEncoder(
            grid_size=int(config.get("response_svd_grid_size", 100)),
            feature_count=int(config.get("response_svd_feature_count", 20)),
        )
        numeric = response_encoder.transform(formulas, n_jobs=n_jobs)
        matrix = matrix_encoder.transform(pool)
        return (np.concatenate((matrix, numeric), axis=1),
                matrix_encoder.feature_names + response_encoder.feature_names,
                "operator_primitive_matrix_plus_normalized_value_matrix_svd")
    if config.get("grouped_tree_feature_ratio") is not None:
        config["grouped_extra_trees_enabled"] = model_type == "extra_trees"
    prime_variant = _resolve_prime_feature_variant(config, feature_mode, prompt_user=False)
    use_response_svd = (
        feature_mode == "value_matrix_svd"
        or (feature_mode == "tree_coordinates_prime_experimental"
            and prime_variant in {"numeric", "both"})
    )
    use_tree_coordinates = (
        (_is_explicit_tree_coordinate_mode(feature_mode)
         and (feature_mode != "tree_coordinates_prime_experimental"
              or prime_variant in {"tree", "both"}))
    )
    if use_response_svd:
        encoder = FormulaResponseSVDEncoder(
            grid_size=int(config.get("response_svd_grid_size", 100)),
            feature_count=int(config.get("response_svd_feature_count", 20)),
        )
        response_features = encoder.transform(formulas, n_jobs=n_jobs)
        if use_tree_coordinates and feature_mode == "tree_coordinates_prime_experimental":
            tree_encoder = _tree_coordinate_encoder_class(feature_mode)(
                max_nodes, ancestor_products=True,
                operator_count_mode=str(config.get("tree_coordinate_operator_count_mode", "ratio")),
                nearest_ancestor_only=bool(config.get("nearest_ancestor_only", False)),
            ).fit(formulas)
            tree_features = tree_encoder.transform(formulas, n_jobs=n_jobs)
            return (np.concatenate((tree_features, response_features), axis=1),
                    tree_encoder.feature_names + encoder.feature_names,
                    "tree_coordinates_prime_experimental_combined")
        return response_features, encoder.feature_names, "normalized_value_matrix_svd"
    if use_tree_coordinates:
        encoder = _tree_coordinate_encoder_class(feature_mode)(
            max_nodes, ancestor_products=_is_explicit_tree_coordinate_mode(feature_mode),
            operator_count_mode=(
                str(config.get("tree_coordinate_operator_count_mode", "ratio"))
                if _is_explicit_tree_coordinate_mode(feature_mode) else "count"
            ),
            nearest_ancestor_only=bool(config.get("nearest_ancestor_only", False)),
            include_composition_features=not _is_global_numeric_tree_mode(feature_mode),
        ).fit(formulas)
        features = encoder.transform(formulas, n_jobs=n_jobs)
        feature_names = encoder.feature_names
        if _is_global_numeric_tree_mode(feature_mode):
            extractor = FormulaStructureFeatureExtractor(max_nodes=max_nodes, max_depth=max_depth)
            _, numeric = extractor.transform(formulas, n_jobs=n_jobs)
            features = np.concatenate((features, numeric), axis=1)
            feature_names = feature_names + extractor.numeric_names
        return (features, feature_names,
                "tree_coordinates_prime_experimental"
                if isinstance(encoder, ExperimentalPrimeTreeCoordinateEncoder)
                else ("tree_coordinates_global_numeric"
                      if _is_global_numeric_tree_mode(feature_mode)
                      else "tree_coordinates"))
    extractor = FormulaStructureFeatureExtractor(max_nodes=max_nodes, max_depth=max_depth)
    _, numeric = extractor.transform(formulas, n_jobs=n_jobs)
    return numeric, extractor.numeric_names, "engineered_numeric"


def save_feature_correlation_diagnostic(
    payload: dict, config: dict, seed: int, output_dir: Path
) -> dict:
    """Save Pearson correlations between sampled formulas' standardized vectors.

    This is deliberately diagnostic-only: it re-extracts features from the
    formula pool and does not alter model fitting, ranking, or random states
    used by the existing pipeline.
    """
    diagnostic_config = config.get("feature_correlation", {})
    if not diagnostic_config.get("enabled", False):
        return {"enabled": False}

    pool = _formula_pool(payload)
    sample_size = max(2, int(diagnostic_config.get("sample_size", 100)))
    sample_size = min(sample_size, len(pool))
    if sample_size < 2:
        return {"enabled": True, "status": "skipped", "reason": "fewer_than_two_formulas"}

    rng = np.random.default_rng(int(diagnostic_config.get("seed", seed)))
    sampled_indices = np.sort(rng.choice(len(pool), size=sample_size, replace=False))
    diagnostic_config_for_mode = dict(config)
    diagnostic_config_for_mode["_operator_primitive_base_records"] = payload.get("bases", [])
    all_features, feature_names, feature_encoding = _diagnostic_features(
        pool, diagnostic_config_for_mode
    )
    numeric_features = all_features[sampled_indices]
    # Standardize each engineered feature across the sampled formulas before
    # comparing formulas.  Constant features use a unit scale and therefore
    # become zero rather than producing NaN values.
    vectors = numeric_features.astype(float)
    means = np.mean(vectors, axis=0, keepdims=True)
    scales = np.std(vectors, axis=0, keepdims=True)
    safe_scales = np.where(scales > 0, scales, 1.0)
    standardized = (vectors - means) / safe_scales
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.corrcoef(standardized, rowvar=True)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 1.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    formula_ids = [pool[int(index)]["formula_id"] for index in sampled_indices]
    formula_expressions = [
        pool[int(index)]["readable_expression"] for index in sampled_indices
    ]
    formula_positions = {formula_id: position
                         for position, formula_id in enumerate(formula_ids)}
    matrix_path = output_dir / "sampled_formula_feature_correlation.csv"
    with matrix_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["formula_id"] + formula_ids)
        writer.writerows([formula_id] + [f"{value:.8f}" for value in row]
                         for formula_id, row in zip(formula_ids, correlation))

    sample_path = output_dir / "sampled_formula_feature_correlation_formulas.csv"
    with sample_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("formula_id", "formula_family", "readable_expression"))
        writer.writerows((pool[int(index)]["formula_id"],
                          pool[int(index)]["formula_family"],
                          pool[int(index)]["readable_expression"])
                         for index in sampled_indices)

    heatmap_path = output_dir / "sampled_formula_feature_correlation_heatmap.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure_size = max(12.0, min(28.0, sample_size * 0.28))
        figure, axis = plt.subplots(figsize=(figure_size, figure_size))
        image = axis.imshow(correlation, cmap="coolwarm", vmin=-1.0, vmax=1.0,
                            aspect="auto", interpolation="nearest")
        axis.set_xticks(np.arange(sample_size))
        axis.set_yticks(np.arange(sample_size))
        axis.set_xticklabels(formula_ids, rotation=90, fontsize=6)
        axis.set_yticklabels(formula_ids, fontsize=6)
        axis.set_xlabel("Sampled formula ID")
        axis.set_ylabel("Sampled formula ID")
        axis.set_title(f"Pearson correlation between standardized formula feature vectors (n={sample_size})")
        figure.colorbar(image, ax=axis, label="Pearson correlation",
                        fraction=0.046, pad=0.04)
        figure.tight_layout()
        figure.savefig(heatmap_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    except ImportError as exc:
        logging.warning("Could not plot feature-correlation heatmap: matplotlib is not installed: %s", exc)
        return {
            "enabled": True, "status": "csv_only", "sampled_formulas": sample_size,
            "matrix_path": str(matrix_path), "sample_path": str(sample_path),
            "heatmap_path": None,
        }

    pairs = []
    for left in range(sample_size):
        for right in range(left + 1, sample_size):
            pairs.append((abs(float(correlation[left, right])),
                          float(correlation[left, right]),
                          formula_ids[left], formula_ids[right]))
    pairs.sort(reverse=True)
    top_pairs = [
        {
            "formula_a": left,
            "formula_a_expression": formula_expressions[formula_ids.index(left)],
            "formula_b": right,
            "formula_b_expression": formula_expressions[formula_ids.index(right)],
            "correlation": value,
            "feature_mse": float(np.mean(
                (standardized[formula_positions[left]]
                 - standardized[formula_positions[right]]) ** 2
            )),
        }
        for _, value, left, right in pairs[:10]
    ]
    logging.info(
        "Formula-feature Pearson correlation diagnostic complete: sampled formulas=%d, features=%d, mode=%s, heatmap=%s",
        sample_size, len(feature_names), feature_encoding, heatmap_path,
    )
    for pair in top_pairs:
        logging.info(
            "  Formula Pearson correlation: %s <-> %s = %.6f, all-feature MSE = %.6f\n"
            "    Formula A: %s\n"
            "    Formula B: %s",
            pair["formula_a"], pair["formula_b"], pair["correlation"],
            pair["feature_mse"],
            pair["formula_a_expression"], pair["formula_b_expression"],
        )
    return {
        "enabled": True, "status": "completed", "sampled_formulas": sample_size,
        "sampled_formula_ids": formula_ids,
        "numeric_feature_count": len(feature_names),
        "feature_mode": feature_encoding,
        "matrix_path": str(matrix_path), "sample_path": str(sample_path),
        "heatmap_path": str(heatmap_path),
        "standardization": "featurewise_zscore_across_sampled_formulas",
        "top_absolute_correlations": top_pairs,
    }


def save_top_k_feature_correlation_diagnostic(
    top_results: Sequence[dict], config: dict, output_dir: Path,
) -> dict:
    """Visualize feature-vector correlations among the selected Top-K formulas.

    This is diagnostic-only.  It uses the same feature representation as the
    configured ranking model and does not feed the result back into ranking,
    refitting, GP, or any other pipeline decision.
    """
    diagnostic_config = config.get("feature_correlation", {})
    if not diagnostic_config.get("enabled", False):
        return {"enabled": False}
    rows = list(top_results or [])
    if len(rows) < 2:
        return {"enabled": True, "status": "skipped",
                "reason": "fewer_than_two_top_k_formulas",
                "selected_formulas": len(rows)}

    normalized_rows = []
    for index, row in enumerate(rows):
        expression = row.get("readable_expression")
        if not expression:
            continue
        normalized_rows.append({
            "formula_id": str(row.get("formula_id", f"top_k_{index + 1:03d}")),
            "formula_family": row.get("formula_family", row.get("structure_type", "unknown")),
            "readable_expression": expression,
            "operator": row.get("operator"),
            "operators": row.get("operators"),
            "base_ids": row.get("base_ids"),
        })
    if len(normalized_rows) < 2:
        return {"enabled": True, "status": "skipped",
                "reason": "fewer_than_two_parseable_top_k_formulas",
                "selected_formulas": len(normalized_rows)}

    try:
        features, feature_names, feature_encoding = _diagnostic_features(
            normalized_rows, config
        )
    except (TypeError, ValueError, sp.SympifyError) as exc:
        logging.warning("Could not extract Top-K tree-feature correlations: %s", exc)
        return {"enabled": True, "status": "skipped",
                "reason": "feature_extraction_failed", "error": str(exc)}

    vectors = np.asarray(features, dtype=float)
    means = np.mean(vectors, axis=0, keepdims=True)
    scales = np.std(vectors, axis=0, keepdims=True)
    standardized = (vectors - means) / np.where(scales > 0, scales, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.corrcoef(standardized, rowvar=True)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 1.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    formula_ids = [row["formula_id"] for row in normalized_rows]
    matrix_path = output_dir / "top_k_tree_feature_correlation.csv"
    with matrix_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["formula_id"] + formula_ids)
        writer.writerows(
            [formula_id] + [f"{value:.8f}" for value in row]
            for formula_id, row in zip(formula_ids, correlation)
        )

    formula_path = output_dir / "top_k_tree_feature_correlation_formulas.csv"
    with formula_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("formula_id", "formula_family", "readable_expression"))
        writer.writerows(
            (row["formula_id"], row["formula_family"], row["readable_expression"])
            for row in normalized_rows
        )

    heatmap_path = output_dir / "top_k_tree_feature_correlation_heatmap.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure_size = max(8.0, min(24.0, len(formula_ids) * 0.7))
        figure, axis = plt.subplots(figsize=(figure_size, figure_size))
        image = axis.imshow(correlation, cmap="coolwarm", vmin=-1.0, vmax=1.0,
                            aspect="auto", interpolation="nearest")
        axis.set_xticks(np.arange(len(formula_ids)))
        axis.set_yticks(np.arange(len(formula_ids)))
        axis.set_xticklabels(formula_ids, rotation=90, fontsize=7)
        axis.set_yticklabels(formula_ids, fontsize=7)
        axis.set_xlabel("Top-K formula ID")
        axis.set_ylabel("Top-K formula ID")
        axis.set_title(
            "Pearson correlation of Top-K formula tree-feature vectors "
            f"(n={len(formula_ids)}, encoding={feature_encoding})"
        )
        figure.colorbar(image, ax=axis, label="Pearson correlation",
                        fraction=0.046, pad=0.04)
        figure.tight_layout()
        figure.savefig(heatmap_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
    except ImportError as exc:
        logging.warning("Could not plot Top-K feature-correlation heatmap: matplotlib is not installed: %s", exc)
        heatmap_path = None

    return {
        "enabled": True,
        "status": "completed" if heatmap_path else "csv_only",
        "selected_formulas": len(formula_ids),
        "formula_ids": formula_ids,
        "numeric_feature_count": len(feature_names),
        "feature_names": feature_names,
        "feature_encoding": feature_encoding,
        "matrix_path": str(matrix_path),
        "formula_path": str(formula_path),
        "heatmap_path": str(heatmap_path) if heatmap_path else None,
        "standardization": "featurewise_zscore_across_top_k_formulas",
    }


def train_and_rank(payload: dict, config: dict, seed: int, output_dir: Path) -> tuple[list[dict], dict]:
    """Train on within-dataset percentile labels and rank all formulas."""
    validate_ranking_dependencies(config.get("model_type", "extra_trees"))
    import joblib
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import OneHotEncoder

    pool = _formula_pool(payload)
    # Mode 5 may append zeroed/±1 derived formulas to the payload for exact
    # evaluation.  The tree ranker must remain trained on the original frozen
    # formula library, whose precomputed feature artifact does not contain
    # those derived IDs.
    original_pool = [
        row for row in pool
        if not row.get("source_formula_id")
        and not str(row.get("structure_type", "")).startswith("mode5_")
    ]
    derived_count = len(pool) - len(original_pool)
    if derived_count:
        print(
            f"Tree-model training uses the original formula library: excluding {derived_count} zeroed or ±1-derived formulas, "
            f"retaining {len(original_pool)} original formulas.",
            flush=True,
        )
    pool = original_pool
    allowed_ids = {str(row["formula_id"]) for row in pool}
    label_field = str(config.get("training_label_field", "r2"))
    labels = {
        formula_id: value for formula_id, value in
        _training_labels(payload, label_field).items()
        if str(formula_id) in allowed_ids
    }
    examples = [
        item for item in _training_examples(payload, label_field)
        if str(item[0]) in allowed_ids
    ]
    actual_r2 = {
        formula_id: value for formula_id, value in _training_actual_r2(payload).items()
        if str(formula_id) in allowed_ids
    }
    feature_mode = _resolve_feature_mode(config)
    active_feature_ablation = sorted({
        str(value).strip().lower()
        for value in config.get("feature_ablation_remove", [])
    })
    if active_feature_ablation:
        print(
            "Tree-model feature ablation enabled: removed="
            + ", ".join(active_feature_ablation),
            flush=True,
        )
    if feature_mode == "operator_primitive_matrix":
        # Later Top-K diagnostics use the same fixed primitive order.
        config["_operator_primitive_base_records"] = list(payload.get("bases", []))
    # Ranking samples are equally weighted in every feature mode.  The
    # optional _fit sample_weight plumbing remains available for callers, but
    # the pipeline deliberately does not activate it.
    sample_weights = None
    sample_weight_metadata = {"enabled": False, "strategy": "uniform"}
    model_type = _canonical_model_type(config.get("model_type", "extra_trees"))
    include_operator_tree_pca = (
        feature_mode == "operator_primitive_matrix"
        and bool(config.get("operator_primitive_tree_pca_enabled", True))
    )
    # The experimental mode is intentionally selected at runtime.  The
    # configuration value is only a default for non-interactive execution or
    # an empty/invalid terminal response.
    # Main pipeline runs must be non-interactive; retain the resolver's
    # prompt_user interface for external callers while selecting the
    # configured/default (both) variant here.
    prime_variant = _resolve_prime_feature_variant(config, feature_mode, prompt_user=False)
    use_response_svd = (
        feature_mode == "value_matrix_svd"
        or (feature_mode == "tree_coordinates_prime_experimental"
            and prime_variant in {"numeric", "both"})
    )
    use_tree_coordinates = (
        (_is_explicit_tree_coordinate_mode(feature_mode)
         and (feature_mode != "tree_coordinates_prime_experimental"
              or prime_variant in {"tree", "both"}))
    )
    layered_tree_metadata = {"enabled": False}
    tree_encoder = None
    response_svd_encoder = None
    tree_feature_pca = None
    operator_primitive_encoder = None
    if use_tree_coordinates:
        pool_index = {row["formula_id"]: index for index, row in enumerate(pool)}
        # Derive a separate capacity for every tree depth from the complete
        # formula pool. This makes the layered encoder elastic.
        counts_by_formula = [
            TreeCoordinateEncoder.layer_node_counts(row["readable_expression"])
            for row in pool
        ]
        max_tree_levels = max((len(counts) for counts in counts_by_formula), default=0)
        dynamic_layer_caps = [
            max(
                (counts[depth] if depth < len(counts) else 0)
                for counts in counts_by_formula
            )
            for depth in range(max_tree_levels)
        ]
        configured_nodes_per_level = config.get("tree_max_nodes_per_level", None)
        nodes_per_level = (
            max(1, int(configured_nodes_per_level))
            if configured_nodes_per_level is not None
            else max(dynamic_layer_caps, default=1)
        )
        training_formula_ids = sorted({formula_id for formula_id, _value in examples})
        if not training_formula_ids:
            raise ValueError("No formulas are available to train the tree-coordinate ranker")
        encoder_formulas = [
            row["readable_expression"] for row in pool
            if configured_nodes_per_level is None
            or all(count <= nodes_per_level
                   for count in TreeCoordinateEncoder.layer_node_counts(
                       row["readable_expression"]))
        ]
        if not encoder_formulas:
            raise ValueError("No training formulas satisfy tree_max_nodes_per_level")
        tree_encoder = _tree_coordinate_encoder_class(feature_mode)(
            int(config.get("max_nodes", 96)),
            ancestor_products=_is_explicit_tree_coordinate_mode(feature_mode),
            operator_count_mode=(
                str(config.get("tree_coordinate_operator_count_mode", "ratio"))
                if _is_explicit_tree_coordinate_mode(feature_mode) else "count"
            ),
            nearest_ancestor_only=bool(config.get("nearest_ancestor_only", False)),
            include_composition_features=not _is_global_numeric_tree_mode(feature_mode),
            max_nodes_per_level=nodes_per_level,
        ).fit(encoder_formulas)
        retained_pool, discarded_reasons = [], {}
        for row in pool:
            reason = tree_encoder.compatibility_reason(row["readable_expression"])
            if reason is None:
                retained_pool.append(row)
            else:
                discarded_reasons[reason] = discarded_reasons.get(reason, 0) + 1
        pool = retained_pool
        labels = {formula_id: value for formula_id, value in labels.items()
                  if formula_id in {row["formula_id"] for row in pool}}
        layered_tree_metadata = {
            "enabled": True,
            "max_nodes_per_level": nodes_per_level,
            "max_nodes_per_level_mode": "auto_from_formula_pool",
            "tree_levels": tree_encoder.max_tree_levels,
            "nodes_per_level": tree_encoder.layer_node_caps,
            "feature_node_slots": tree_encoder.max_nodes,
            "discarded_training_formulas": 0,
            "discarded_formulas": sum(discarded_reasons.values()),
            "discarded_reasons": discarded_reasons,
        }
        print(
            "Tree-coordinate layered encoding: per-layer node caps (auto-determined from formula pool)={}, dynamic layer count={}, nodes per layer={}, "
            "total node slots={}; discarded training formulas={}, discarded formulas to rank={}".format(
                tree_encoder.layer_node_caps, tree_encoder.max_tree_levels,
                tree_encoder.layer_node_caps, tree_encoder.max_nodes,
                0, sum(discarded_reasons.values()),
            ), flush=True,
        )
        for depth, cap in enumerate(tree_encoder.layer_node_caps):
            print(f"   round  {depth} layer node limit: {cap}", flush=True)
    minimum = int(config.get("minimum_training_samples", 20))
    if len(examples) < minimum:
        raise ValueError(f"Only {len(examples)} real frequency-band percentile-ranking samples are available; at least {minimum} are required")
    pool_index = {row["formula_id"]: index for index, row in enumerate(pool)}
    sample_formula_indices = np.asarray([pool_index[formula_id] for formula_id, _ in examples])
    labeled_y = np.asarray([value for _, value in examples])
    sample_groups = np.asarray([formula_id for formula_id, _ in examples])

    formulas = [row.get("tree_readable_expression", row["readable_expression"]) for row in pool]
    precomputed_path = config.get("precomputed_feature_artifact")
    precomputed_feature_cache = None
    if precomputed_path:
        precomputed_feature_cache = load_precomputed_feature_artifact(
            precomputed_path, pool, config
        )
        # Keep only the lightweight metadata object needed by the ranking
        # output; all feature values and node sequences come from the pack.
        extractor = FormulaStructureFeatureExtractor(
            max_nodes=int(config.get("max_nodes", 96)),
            max_depth=int(config.get("engineered_max_depth", 12)),
        )
        categorical = precomputed_feature_cache["categorical_features"]
        numeric = None
    else:
        extractor = FormulaStructureFeatureExtractor(
            max_nodes=int(config.get("max_nodes", 96)),
            max_depth=int(config.get("engineered_max_depth", 12)),
        )
        categorical, numeric = extractor.transform(formulas, n_jobs=1)
    encoder = None
    # Keep one shared cache for all pipeline output directories.  The store
    # contains multiple entries keyed by the complete formula/encoder
    # signature, so different binary-formula pools can coexist in one file.
    # The cache format stores only the final model input matrix.  Use a new
    # filename so an older cache containing the large ``raw_features`` matrix
    # is never loaded and decompressed by this version.
    feature_cache_path = (
        PROJECT_ROOT / "artifacts" / "reference_features"
        / "structure_r2_feature_cache_features_only.joblib"
    )
    cache_descriptor = {
        "formulas": formulas,
        "feature_mode": feature_mode,
        "max_nodes": int(config.get("max_nodes", 96)),
        "tree_max_nodes_per_level": config.get("tree_max_nodes_per_level"),
        "nearest_ancestor_only": bool(config.get("nearest_ancestor_only", False)),
        "tree_coordinate_operator_count_mode": str(
            config.get("tree_coordinate_operator_count_mode", "ratio")
        ),
        "prime_feature_mode": prime_variant,
        "pca_tree_features": bool(config.get("pca_tree_features", False)),
        "kernel_pca_components": int(config.get("kernel_pca_components", 64)),
        "response_svd_grid_size": int(config.get("response_svd_grid_size", 100)),
        "response_svd_feature_count": int(
            config.get("response_svd_feature_count", 20)
        ),
        "operator_primitive_tree_pca": include_operator_tree_pca,
        "tree_pca_method": str(config.get("tree_pca_method", "kernel")).lower(),
        "operator_primitive_numeric_encoding": "response_svd_plus_engineered_v2",
    }
    cache_key = hashlib.sha256(json.dumps(
        cache_descriptor, ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")).hexdigest()
    feature_cache = precomputed_feature_cache
    feature_cache_store = {}
    cacheable_tree_features = use_tree_coordinates or include_operator_tree_pca
    if cacheable_tree_features and feature_cache is None:
        try:
            cached_store = joblib.load(feature_cache_path)
            if isinstance(cached_store, dict) and isinstance(
                    cached_store.get("entries"), dict):
                feature_cache_store = cached_store["entries"]
            elif isinstance(cached_store, dict) and cached_store.get("cache_key"):
                # Migrate the previous single-entry cache format in memory.
                feature_cache_store = {
                    cached_store["cache_key"]: cached_store
                }
            cached = feature_cache_store.get(cache_key)
            if (isinstance(cached, dict)
                    and cached.get("features") is not None
                    and cached.get("model_feature_names") is not None):
                feature_cache = cached
                print(f"Reusing tree-coordinate feature cache: {feature_cache_path}", flush=True)
        except (OSError, EOFError, ValueError, AttributeError,
                ModuleNotFoundError, ImportError):
            feature_cache = None

    if feature_cache is not None:
        features = np.asarray(feature_cache["features"], dtype=np.float32)
        model_feature_names = list(feature_cache["model_feature_names"])
        tree_feature_pca = feature_cache.get("tree_feature_pca")
        if feature_cache.get("tree_encoder") is not None:
            tree_encoder = feature_cache["tree_encoder"]
        if feature_cache.get("response_svd_encoder") is not None:
            response_svd_encoder = feature_cache["response_svd_encoder"]
        if feature_cache.get("operator_primitive_encoder") is not None:
            operator_primitive_encoder = feature_cache["operator_primitive_encoder"]
    elif feature_mode == "operator_primitive_matrix":
        operator_primitive_encoder = OperatorPrimitiveMatrixEncoder(payload.get("bases", []))
        matrix_features = operator_primitive_encoder.transform(pool)
        operator_primitive_encoder.register_records(pool, matrix_features)
        response_svd_encoder = FormulaResponseSVDEncoder(
            grid_size=int(config.get("response_svd_grid_size", 100)),
            feature_count=int(config.get("response_svd_feature_count", 20)),
        )
        response_features = response_svd_encoder.transform(formulas, n_jobs=1)
        # Keep both learned numerical representations: response SVD and the
        # original engineered structure statistics.
        base_features = np.concatenate((matrix_features, response_features, numeric), axis=1)
        model_feature_names = (
            operator_primitive_encoder.feature_names
            + response_svd_encoder.feature_names
            + extractor.numeric_names
        )
        if include_operator_tree_pca:
            tree_encoder = TreeCoordinateEncoder(
                int(config.get("max_nodes", 96)),
                ancestor_products=True,
                operator_count_mode=str(config.get(
                    "tree_coordinate_operator_count_mode", "ratio")),
                nearest_ancestor_only=bool(config.get(
                    "nearest_ancestor_only", False)),
            ).fit(formulas)
            raw_tree_features = tree_encoder.transform(formulas, n_jobs=1)
            tree_feature_pca, tree_features = _fit_tree_feature_pca(
                raw_tree_features,
                config.get("kernel_pca_components", 64),
                config.get("kernel_pca_fit_samples", 5000),
                seed,
                config.get("kernel_pca_transform_batch_size", 1024),
                method=config.get("tree_pca_method", "kernel"),
            )
            features = np.concatenate((base_features, tree_features), axis=1)
            del raw_tree_features
            model_feature_names += [
                f"tree_kernel_pca_component_{index + 1:03d}"
                for index in range(tree_features.shape[1])
            ]
        else:
            features = base_features
    elif use_response_svd:
        response_svd_encoder = FormulaResponseSVDEncoder(
            grid_size=int(config.get("response_svd_grid_size", 100)),
            feature_count=int(config.get("response_svd_feature_count", 20)),
        ).fit(formulas)
        response_features = response_svd_encoder.transform(formulas, n_jobs=1)
        if use_tree_coordinates and feature_mode == "tree_coordinates_prime_experimental":
            tree_features = tree_encoder.transform(formulas, n_jobs=1)
            if bool(config.get("pca_tree_features", False)):
                tree_feature_pca, tree_features = _fit_tree_feature_pca(
                    tree_features,
                    config.get("kernel_pca_components", 64),
                    config.get("kernel_pca_fit_samples", 5000),
                    seed,
                    config.get("kernel_pca_transform_batch_size", 1024),
                    method=config.get("tree_pca_method", "kernel"),
                )
            features = np.concatenate((tree_features, response_features), axis=1)
            tree_feature_names = (
                [f"tree_kernel_pca_component_{index + 1:03d}"
                 for index in range(tree_features.shape[1])]
                if tree_feature_pca is not None else tree_encoder.feature_names
            )
            model_feature_names = tree_feature_names + response_svd_encoder.feature_names
        else:
            features = response_features
            model_feature_names = response_svd_encoder.feature_names
    elif use_tree_coordinates:
        features = tree_encoder.transform(formulas, n_jobs=1)
        if _is_global_numeric_tree_mode(feature_mode):
            features = np.concatenate((features, numeric), axis=1)
            model_feature_names = tree_encoder.feature_names + extractor.numeric_names
        if bool(config.get("pca_tree_features", False)):
            tree_feature_pca, features = _fit_tree_feature_pca(
                features,
                config.get("kernel_pca_components", 64),
                config.get("kernel_pca_fit_samples", 5000),
                seed,
                config.get("kernel_pca_transform_batch_size", 1024),
                method=config.get("tree_pca_method", "kernel"),
            )
            model_feature_names = [f"tree_kernel_pca_component_{index + 1:03d}"
                                   for index in range(features.shape[1])]
        else:
            if not _is_global_numeric_tree_mode(feature_mode):
                model_feature_names = tree_encoder.feature_names
    else:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
        encoded = encoder.fit_transform(categorical)
        features = hstack((encoded, csr_matrix(numeric)), format="csr")
        model_feature_names = list(encoder.get_feature_names_out(extractor.categorical_names)) + extractor.numeric_names

    if cacheable_tree_features and feature_cache is None:
        # Cache the complete representation before feature ablation. All mode3
        # ablation branches share this entry and remove columns afterward.
        cache_payload = {
            "cache_key": cache_key,
            "cache_descriptor": cache_descriptor,
            "features": np.asarray(features, dtype=np.float32),
            "model_feature_names": list(model_feature_names),
            "tree_encoder": tree_encoder,
            "response_svd_encoder": response_svd_encoder,
            "tree_feature_pca": tree_feature_pca,
            "operator_primitive_encoder": operator_primitive_encoder,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        feature_cache_store[cache_key] = cache_payload
        joblib.dump({
            "version": 2,
            "entries": feature_cache_store,
        }, feature_cache_path, compress=3)
        print(f"Saving tree-coordinate feature cache: {feature_cache_path}", flush=True)

    # Apply feature-group ablation after loading/building the complete
    # representation, so all mode3 branches reuse identical features,
    # encoders, PCA fit, and feature ordering.
    features, model_feature_names = _apply_feature_group_ablation(
        features, model_feature_names, config
    )

    if config.get("grouped_extra_trees_enabled", False):
        config["grouped_tree_feature_count"] = len(tree_encoder.feature_names)

    validation_fraction = float(config.get("validation_fraction", 0.2))
    sample_indices = np.arange(len(examples))
    if 0 < validation_fraction < 1 and len(examples) >= 10 and len(set(sample_groups)) >= 2:
        splitter = GroupShuffleSplit(n_splits=5, test_size=validation_fraction, random_state=seed)
        train_samples, validation_samples = next(splitter.split(
            sample_indices, labeled_y, groups=sample_groups
        ))
    else:
        train_samples, validation_samples = sample_indices, np.asarray([], dtype=int)

    model_type, _ = _make_model(config, seed)
    train_feature_indices = sample_formula_indices[train_samples]
    train_y = labeled_y[train_samples]
    validation_feature_indices = sample_formula_indices[validation_samples]
    validation_true = labeled_y[validation_samples]
    _, validation_model = _make_model(config, seed)
    _fit(validation_model, features, train_feature_indices, train_y,
         sample_weights[train_samples] if sample_weights is not None else None)
    if len(validation_samples):
        validation_prediction = np.asarray(validation_model.predict(
            features[validation_feature_indices]
        ))
        validation_metrics = _rank_metrics(
            validation_true, validation_prediction,
            int(config.get("validation_top_k", 10))
        )
    else:
        validation_metrics = {"spearman": None, "top_k_overlap": None, "samples": 0}
    validation_metrics["primary_metric"] = (
        "top_k_overlap" if model_type == "xgboost" else "spearman"
    )

    _, final_model = _make_model(config, seed)
    _fit(final_model, features, sample_formula_indices, labeled_y, sample_weights)
    scores = np.nan_to_num(np.asarray(final_model.predict(features), dtype=float),
                           nan=-1e100, posinf=1e100, neginf=-1e100)
    order = np.argsort(-scores, kind="stable")
    training_formula_ids = {
        str(pool[int(training_index)]["formula_id"])
        for training_index in sample_formula_indices
    }
    ranking, family_ranks = [], {"single": 0, "combination": 0}
    for rank, index in enumerate(order, start=1):
        row, formula_id = pool[int(index)], pool[int(index)]["formula_id"]
        family_ranks[row["formula_family"]] += 1
        ranking.append({
            "predicted_rank": rank,
            "predicted_family_rank": family_ranks[row["formula_family"]],
            "ranking_score": float(scores[index]),
            "actual_fitted_percentile": labels.get(formula_id),
            "has_percentile_label": formula_id in labels,
            "actual_fit_r2": actual_r2.get(str(formula_id)),
            "is_training_formula": formula_id in training_formula_ids,
            "node_sequence": categorical[index, :extractor.max_nodes].tolist(), **row,
        })

    raw_feature_names = list(model_feature_names)
    output_dir.mkdir(parents=True, exist_ok=True)
    persist_ranker_artifacts = not bool(
        config.get("suppress_large_artifacts", False)
        or config.get("minimal_output_artifacts", False)
    )
    model_path = output_dir / "structure_r2_ranker.joblib"
    feature_path = output_dir / "structure_r2_features.json"
    importance_path = output_dir / "structure_r2_feature_importance.csv"
    training_formula_path = output_dir / "structure_r2_training_formulas.csv"
    # Save the exact formula rows consumed by the ranker.  This is kept as a
    # lightweight CSV even in minimal-output Mode 4 runs so the tree training
    # set remains auditable without persisting another feature matrix.
    validation_training_sample_indices = set(map(int, train_samples))
    with training_formula_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "sample_index", "pool_index", "formula_id", "formula_family",
            "structure_type", "readable_expression", "training_label",
            "actual_fit_r2", "used_by_validation_model", "used_by_final_model",
        ))
        writer.writeheader()
        for sample_index, (pool_index_value, target) in enumerate(
                zip(sample_formula_indices, labeled_y)):
            formula = pool[int(pool_index_value)]
            formula_id = str(formula["formula_id"])
            writer.writerow({
                "sample_index": sample_index,
                "pool_index": int(pool_index_value),
                "formula_id": formula_id,
                "formula_family": formula.get("formula_family"),
                "structure_type": formula.get("structure_type"),
                "readable_expression": formula.get("readable_expression"),
                "training_label": float(target),
                "actual_fit_r2": actual_r2.get(formula_id),
                "used_by_validation_model": sample_index in validation_training_sample_indices,
                "used_by_final_model": True,
            })
    importance_feature_names = model_feature_names
    if persist_ranker_artifacts:
        joblib.dump({"model": final_model, "encoder": encoder, "extractor": extractor,
                     "tree_encoder": tree_encoder,
                     "response_svd_encoder": response_svd_encoder,
                     "tree_feature_pca": tree_feature_pca,
                     "operator_primitive_encoder": operator_primitive_encoder,
                     "feature_mode": feature_mode,
                     "feature_ablation_remove": sorted({
                         str(value).strip().lower()
                         for value in config.get("feature_ablation_remove", [])
                     }),
                     "model_type": model_type,
                     "selected_feature_indices": (selected_indices.tolist()
                                                   if use_tree_coordinates and _is_explicit_tree_coordinate_mode(feature_mode)
                                                   else None)}, model_path)
    importance = final_model.feature_importances_
    if persist_ranker_artifacts:
        _save_importance(importance_path, importance_feature_names, importance)
    importance_category_stats = _feature_importance_category_stats(
        importance_feature_names, importance
    )
    importance_categories = {
        category: stats["total"]
        for category, stats in importance_category_stats.items()
    }
    if operator_primitive_encoder is not None:
        print("ExtraTrees importance of four feature groups (total/share/mean per feature): ", flush=True)
        for category, stats in importance_category_stats.items():
            total = float(stats["total"])
            mean = float(stats["mean"])
            print(
                f"  {category}: total={total:.8f} ({total:.2%}), "
                f"feature count={int(stats['count'])}, mean={mean:.8f}",
                flush=True,
            )
    if (tree_encoder is not None and response_svd_encoder is not None
            and feature_mode == "tree_coordinates_prime_experimental"):
        feature_encoding = "tree_coordinates_prime_experimental_plus_normalized_value_matrix_svd"
    else:
        feature_encoding = (
        ("operator_primitive_matrix_plus_normalized_value_matrix_svd_plus_engineered_numeric_plus_tree_pca"
         if include_operator_tree_pca
         else "operator_primitive_matrix_plus_normalized_value_matrix_svd_plus_engineered_numeric")
        if operator_primitive_encoder is not None
        else
        ("layered_tree_coordinates_with_global_numeric_features"
         if _is_global_numeric_tree_mode(feature_mode)
         else "layered_tree_coordinates_with_experimental_prime_log_ancestor_products"
         if isinstance(tree_encoder, ExperimentalPrimeTreeCoordinateEncoder)
         else "layered_tree_coordinates_with_standardized_onehot_ancestor_products"
         if tree_encoder.ancestor_products
         else "layered_tree_coordinates")
        if tree_encoder is not None
        else "normalized_value_matrix_svd"
        if response_svd_encoder is not None
        else "preorder_categories_plus_engineered_structure_features"
        )
    feature_payload = {
        "raw_feature_names": raw_feature_names,
        "raw_feature_dimension": len(raw_feature_names),
        "categorical_feature_names": extractor.categorical_names,
        "numeric_feature_names": (response_svd_encoder.feature_names
                                  if response_svd_encoder is not None
                                  else extractor.numeric_names),
        "engineered_numeric_feature_names": extractor.numeric_names,
        "model_feature_names": model_feature_names,
        "model_feature_dimension": len(model_feature_names),
        "sample_weighting": sample_weight_metadata,
        "layered_tree_coordinates": layered_tree_metadata,
        "tree_coordinate_operator_count_mode": (
            tree_encoder.operator_count_mode
            if tree_encoder is not None and tree_encoder.ancestor_products else None
        ),
        "tree_coordinate_variable_ratio_features": bool(
            tree_encoder is not None and tree_encoder.ancestor_products
            and tree_encoder.include_composition_features
        ),
        "global_tree_numeric_features": bool(
            _is_global_numeric_tree_mode(feature_mode)
        ),
        "operator_primitive_matrix": ({
            "operator_categories": list(operator_primitive_encoder.operator_categories),
            "rows_per_operator": operator_primitive_encoder.rows_per_operator,
            "base_ids": list(operator_primitive_encoder.base_ids),
            "base_expressions": list(operator_primitive_encoder.base_expressions),
            "matrix_shape": list(operator_primitive_encoder.matrix_shape),
            "matrix_feature_dimension": int(np.prod(operator_primitive_encoder.matrix_shape)),
            "combined_matrix_feature_dimension": operator_primitive_encoder.combined_matrix_feature_dimension,
            "numeric_feature_dimension": len(response_svd_encoder.feature_names),
            "engineered_numeric_feature_dimension": len(extractor.numeric_names),
            "tree_pca_enabled": bool(include_operator_tree_pca),
            "tree_pca_method": str(config.get("tree_pca_method", "kernel")).lower(),
            "tree_pca_feature_dimension": (
                int(len(getattr(tree_feature_pca, "eigenvalues_", getattr(
                    tree_feature_pca, "singular_values_", []))))
                if include_operator_tree_pca and tree_feature_pca is not None else 0
            ),
        } if operator_primitive_encoder is not None else None),
        "nearest_ancestor_only": bool(
            tree_encoder is not None and tree_encoder.nearest_ancestor_only
        ),
        "tree_feature_pca": ({
            "enabled": True,
            "method": str(config.get("tree_pca_method", "kernel")).lower(),
            "kernel": "rbf" if str(config.get("tree_pca_method", "kernel")).lower() == "kernel" else None,
            "requested_components": int(config.get("kernel_pca_components", 64)),
            "input_dimension": int(tree_feature_pca.n_features_in_),
            "output_dimension": int(len(getattr(tree_feature_pca, "eigenvalues_", getattr(
                tree_feature_pca, "singular_values_", [])))),
        } if tree_feature_pca is not None else {"enabled": False}),
        "response_svd": ({
            "grid_size": response_svd_encoder.grid_size,
            "feature_count": response_svd_encoder.feature_count,
            "normalization": "finite_value_min_max_with_invalid_median_fill",
            "prime_experimental_feature_mode": (
                prime_variant
                if feature_mode == "tree_coordinates_prime_experimental" else None
            ),
        } if response_svd_encoder is not None else None),
        "tree_coordinates_prime_experimental_feature_mode": (
            prime_variant if feature_mode == "tree_coordinates_prime_experimental" else None
        ),
        "feature_cache": ({
            "enabled": True,
            "path": str(feature_cache_path),
            "cache_key": cache_key,
            "hit": feature_cache is not None,
            "model_feature_dimension": int(features.shape[1]),
        } if cacheable_tree_features else {"enabled": False}),
        "feature_importance_categories": importance_categories,
        "feature_importance_category_stats": importance_category_stats,
    }
    encoder_payload = {
        "encoding": feature_encoding,
        "max_nodes": (tree_encoder.max_nodes if tree_encoder is not None
                      else extractor.max_nodes),
        "response_svd_grid_size": (response_svd_encoder.grid_size
                                   if response_svd_encoder is not None else None),
        "response_svd_feature_count": (response_svd_encoder.feature_count
                                        if response_svd_encoder is not None else None),
        "engineered_max_depth": extractor.max_depth,
        "tree_max_nodes_per_level": (tree_encoder.max_nodes_per_level
                                       if tree_encoder is not None else None),
        "tree_levels": (tree_encoder.max_tree_levels
                          if tree_encoder is not None else None),
        "tree_nodes_per_level": (tree_encoder.layer_node_caps
                                   if tree_encoder is not None else None),
        "tree_node_fields": (tree_encoder._ancestor_field_names
                              if tree_encoder is not None and tree_encoder.ancestor_products
                              else (["x", "y", "symbol", "path"]
                                    if tree_encoder is not None else None)),
        "legacy_tree_node_fields": (["x", "y", "symbol", "path",
                                      "parent_x", "parent_y", "parent_symbol", "parent_path"]
                                     if tree_encoder is not None and not tree_encoder.ancestor_products
                                     else None),
        "ancestor_product_fields": (tree_encoder._ancestor_field_names
                                     if tree_encoder is not None and tree_encoder.ancestor_products
                                     else None),
        "max_ancestor_count": (tree_encoder.max_ancestor_count
                                if tree_encoder is not None and tree_encoder.ancestor_products
                                else None),
        "standardization_means": (tree_encoder.standardization_means.tolist()
                                   if tree_encoder is not None and tree_encoder.ancestor_products
                                   else None),
        "standardization_scales": (tree_encoder.standardization_scales.tolist()
                                   if tree_encoder is not None and tree_encoder.ancestor_products
                                   else None),
        "sample_weighting": sample_weight_metadata,
        "layered_tree_coordinates": layered_tree_metadata,
        "symbol_codes": (tree_encoder.symbol_codes if tree_encoder is not None else None),
        "symbol_primes": (tree_encoder.symbol_primes
                          if isinstance(tree_encoder, ExperimentalPrimeTreeCoordinateEncoder)
                          else None),
        "categories_by_feature": ([list(map(str, values)) for values in encoder.categories_]
                                  if encoder is not None else None),
        **feature_payload,
    }
    if persist_ranker_artifacts:
        feature_path.write_text(
            json.dumps(feature_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # Retain the existing encoder metadata filename for downstream compatibility.
        (output_dir / "structure_r2_encoder.json").write_text(
            json.dumps(encoder_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    metadata = {
        "enabled": True, "model_type": model_type,
        "feature_mode": feature_mode,
        "feature_encoding": feature_encoding,
        "max_nodes": (tree_encoder.max_nodes if tree_encoder is not None
                      else extractor.max_nodes),
        "training_samples": len(examples),
        "training_formula_ids": sorted({
            str(pool[int(index)]["formula_id"])
            for index in sample_formula_indices
        }),
        "unique_labeled_formulas": len(labels),
        "ranked_formulas": len(pool), "training_target": "within_dataset_r2_percentile",
        "raw_feature_dimension": len(raw_feature_names),
        "model_feature_dimension": len(model_feature_names),
        "sample_weighting": sample_weight_metadata,
        "layered_tree_coordinates": layered_tree_metadata,
        "validation": validation_metrics,
        "primary_metric": validation_metrics["primary_metric"],
        "model_path": str(model_path) if persist_ranker_artifacts else None,
        "encoder_path": (str(output_dir / "structure_r2_encoder.json")
                          if persist_ranker_artifacts else None),
        "feature_list_path": str(feature_path) if persist_ranker_artifacts else None,
        "feature_importance_path": (str(importance_path)
                                     if persist_ranker_artifacts else None),
        "training_formula_path": str(training_formula_path),
        "ranker_artifacts_persisted": persist_ranker_artifacts,
    }
    primary_metric = validation_metrics["primary_metric"]
    log_context = config.get("_log_context")
    logging.info(
        "Structure percentile ranking complete%s: %s, training=%d, ranked=%d, features=%d, primary metric=%s, value=%s",
        f" ({log_context})" if log_context else "",
        model_type, len(examples), len(pool), len(model_feature_names),
        primary_metric, validation_metrics.get(primary_metric),
    )
    return ranking, metadata


def load_structure_ranker_scorer(model_path: Path):
    """Load a persisted ranker and return a scorer for new formula strings.

    The returned callable uses exactly the feature encoders saved with the
    fitted ranker, so GP-generated structures can be scored without any LM
    fitting or ranker retraining.
    """
    import joblib

    bundle = joblib.load(model_path)
    model = bundle["model"]
    encoder = bundle.get("encoder")
    extractor = bundle["extractor"]
    tree_encoder = bundle.get("tree_encoder")
    response_svd_encoder = bundle.get("response_svd_encoder")
    tree_feature_pca = bundle.get("tree_feature_pca")
    operator_primitive_encoder = bundle.get("operator_primitive_encoder")
    feature_mode = bundle.get("feature_mode", "model_default")
    model_type = bundle["model_type"]
    selected = bundle.get("selected_feature_indices")
    selected_indices = (None if selected is None
                        else np.asarray(selected, dtype=int))

    def score(formulas: Sequence[str]) -> np.ndarray:
        formulas = [str(formula) for formula in formulas]
        compatible = np.ones(len(formulas), dtype=bool)
        if tree_encoder is not None and tree_encoder.max_nodes_per_level is not None:
            compatible = np.asarray([
                tree_encoder.compatibility_reason(formula) is None
                for formula in formulas
            ], dtype=bool)
        scores = np.full(len(formulas), -1e100, dtype=float)
        if not np.any(compatible):
            return scores
        active_formulas = [formula for formula, keep in zip(formulas, compatible) if keep]
        # Engineered structure statistics remain the fourth feature group in
        # operator-matrix mode, so they must be regenerated for new formulas.
        categorical, numeric = extractor.transform(active_formulas, n_jobs=1)
        if operator_primitive_encoder is not None:
            matrix_features = operator_primitive_encoder.transform_formulas(active_formulas)
            response_features = response_svd_encoder.transform(active_formulas, n_jobs=1)
            features = np.concatenate((matrix_features, response_features, numeric), axis=1)
            if tree_encoder is not None and tree_feature_pca is not None:
                tree_features = tree_encoder.transform(active_formulas, n_jobs=1)
                features = np.concatenate((
                    features, tree_feature_pca.transform(tree_features),
                ), axis=1)
            features, _ = _apply_feature_group_ablation(
                features, (
                    operator_primitive_encoder.feature_names
                    + response_svd_encoder.feature_names
                    + extractor.numeric_names
                    + ([
                        f"tree_kernel_pca_component_{index + 1:03d}"
                        for index in range(tree_feature_pca.eigenvectors_.shape[1])
                    ] if tree_feature_pca is not None else [])
                ),
                {"feature_ablation_remove": bundle.get("feature_ablation_remove", [])},
            )
            if selected_indices is not None:
                features = features[:, selected_indices]
        elif response_svd_encoder is not None:
            response_features = response_svd_encoder.transform(active_formulas, n_jobs=1)
            if tree_encoder is not None:
                tree_features = tree_encoder.transform(active_formulas, n_jobs=1)
                if tree_feature_pca is not None:
                    tree_features = tree_feature_pca.transform(tree_features)
                features = np.concatenate((
                    tree_features, response_features,
                ), axis=1)
            else:
                features = response_features
            if selected_indices is not None:
                features = features[:, selected_indices]
        elif tree_encoder is not None:
            features = tree_encoder.transform(active_formulas, n_jobs=1)
            if feature_mode == "tree_coordinates_global_numeric":
                _, numeric = extractor.transform(active_formulas, n_jobs=1)
                features = np.concatenate((features, numeric), axis=1)
            if tree_feature_pca is not None:
                features = tree_feature_pca.transform(features)
            if selected_indices is not None:
                features = features[:, selected_indices]
        else:
            features = hstack((encoder.transform(categorical), csr_matrix(numeric)), format="csr")
        scores[compatible] = np.nan_to_num(
            np.asarray(model.predict(features), dtype=float),
            nan=-1e100, posinf=1e100, neginf=-1e100,
        )
        return scores

    # Tree-coordinate encoders with ancestor products have a fixed maximum
    # ancestor count learned from the training formulas.  GP must reject
    # deeper candidates before calling ``transform``; otherwise a row can
    # contain more ancestor products than the allocated feature width.
    if tree_encoder is not None and tree_encoder.ancestor_products:
        score.max_formula_depth = int(tree_encoder.max_ancestor_count)
    if tree_encoder is not None and tree_encoder.max_nodes_per_level is not None:
        score.is_formula_compatible = lambda formula: (
            tree_encoder.compatibility_reason(str(formula)) is None
        )

    return score
