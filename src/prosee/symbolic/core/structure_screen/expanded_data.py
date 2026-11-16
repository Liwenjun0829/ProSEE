"""Family perturbation and LM-fitted formula-combination dataset builder."""
import csv
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import sympy as sp

from .encoding import canonical_structure


EPS = 1e-12


@dataclass
class FamilyMember:
    formula: str
    constants: List[float]
    prediction: np.ndarray
    fitted_prediction: np.ndarray
    fitted_r2: float
    correlation: float


def _numbers(expr):
    """Return perturbable constants, excluding every constant in an exponent."""
    numbers = []

    def visit(node):
        if node.is_Number:
            numbers.append(node)
            return
        if node.is_Pow:
            # An exponent defines formula structure, so family construction must
            # keep the entire exponent subtree unchanged.
            visit(node.base)
            return
        for argument in node.args:
            visit(argument)

    visit(expr)
    return numbers


def _safe_prediction(expr, X, variables):
    try:
        fn = sp.lambdify(variables, expr, "numpy")
        with np.errstate(all="ignore"):
            columns = [int(str(symbol)[1:]) - 1 for symbol in variables]
            if any(i < 0 or i >= X.shape[1] for i in columns):
                return None
            value = np.asarray(fn(*[X[:, i] for i in columns]), dtype=float)
        if value.ndim == 0:
            value = np.full(X.shape[0], float(value))
        value = value.reshape(-1)
        if value.shape != (X.shape[0],) or not np.all(np.isfinite(value)) or np.max(np.abs(value)) > 1e150:
            return None
        return value
    except Exception:
        return None


def _affine_fit(prediction, y):
    """Fit intercept and scale so each family member has a comparable fitted R²."""
    if np.std(prediction) <= EPS:
        return None
    A = np.column_stack((np.ones(len(y)), prediction))
    try:
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        fitted = A @ coef
    except np.linalg.LinAlgError:
        return None
    den = float(np.sum((y - np.mean(y)) ** 2))
    score = 1.0 - float(np.sum((y - fitted) ** 2)) / den if den > EPS else np.nan
    if not np.isfinite(score) or score > 1.0 + 1e-9:
        return None
    return fitted, min(score, 1.0)


def _safe_corr(a, b):
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or np.std(a) <= EPS or np.std(b) <= EPS:
        return np.nan
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else np.nan


def build_family(formula, X, y, rng, size=8, scale=0.2):
    expr = sp.sympify(formula)
    nums = _numbers(expr)
    base = np.asarray([float(n) for n in nums], dtype=float)
    variables = sorted(expr.free_symbols, key=lambda s: s.name)
    family = []
    for index in range(max(1, size)):
        constants = base if index == 0 else base + rng.normal(0.0, scale * np.maximum(np.abs(base), 1.0), len(base))
        candidate = expr.xreplace(dict(zip(nums, map(sp.Float, constants))))
        pred = _safe_prediction(candidate, X, variables)
        fitted = _affine_fit(pred, y) if pred is not None else None
        if fitted is None:
            continue
        corr = _safe_corr(fitted[0], y)
        if np.isfinite(corr):
            family.append(FamilyMember(str(candidate), constants.tolist(), pred, fitted[0], fitted[1], corr))
    return family


def estimated_r2(predictions, y, rcond=1e-10):
    """Compute r.T @ pinv(R) @ r with constant/NaN columns removed."""
    good = [np.asarray(p, float) for p in predictions if np.all(np.isfinite(p)) and np.std(p) > EPS]
    if not good or np.std(y) <= EPS:
        return None
    matrix = np.column_stack(good)
    R = np.corrcoef(matrix, rowvar=False)
    R = np.atleast_2d(np.asarray(R, float))
    r = np.asarray([_safe_corr(matrix[:, i], y) for i in range(matrix.shape[1])], float)
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(r)):
        return None
    try:
        # Hermitian mode improves stability for the symmetric correlation matrix.
        value = float(r.T @ np.linalg.pinv(R, rcond=rcond, hermitian=True) @ r)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return float(np.clip(value, 0.0, 1.0))


def best_family_combination(families, y, max_product=50000):
    sizes = [len(f) for f in families]
    if not sizes or 0 in sizes or int(np.prod(sizes)) > max_product:
        return None
    best = None
    for members in itertools.product(*families):
        score = estimated_r2([m.fitted_prediction for m in members], y)
        if score is not None and (best is None or score > best[0]):
            best = (score, members)
    return best


def load_base_formulas(path):
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8-sig", newline="")))
    seen, output = set(), []
    for row in rows:
        formula = row.get("formula") or row.get("expr")
        try:
            key = canonical_structure(formula)
            if formula and key not in seen:
                seen.add(key); output.append(formula)
        except Exception:
            continue
    return output


def build_expanded_records(csv_path, X, y, output_path, seed=42, family_size=8,
                           perturb_scale=0.2, max_terms=2, max_bases=80,
                           max_combinations=10000):
    rng = np.random.default_rng(seed)
    bases = load_base_formulas(csv_path)
    if max_bases is not None and max_bases < len(bases):
        bases = list(rng.choice(bases, size=max_bases, replace=False))
    family_map = {f: build_family(f, X, y, rng, family_size, perturb_scale) for f in bases}
    bases = [f for f in bases if family_map[f]]
    records = []
    for width in range(1, max(1, max_terms) + 1):
        for combo in itertools.islice(itertools.combinations(bases, width), max_combinations):
            best = best_family_combination([family_map[f] for f in combo], y)
            if best is None:
                continue
            score, members = best
            records.append({
                "formula": " + ".join(f"({f})" for f in combo),
                "base_formulas": list(combo), "estimated_r2": score, "best_members": [m.formula for m in members],
                "member_constants": [m.constants for m in members], "member_fitted_r2": [m.fitted_r2 for m in members],
                "member_correlations": [m.correlation for m in members], "combination_size": width,
            })
    payload = {"seed": seed, "family_size": family_size, "perturb_scale": perturb_scale,
               "r2_fitting_method": "LM", "max_nfev": max_nfev,
               "base_count": len(bases), "records": records}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("expanded bases=%d records=%d", len(bases), len(records))
    return records
