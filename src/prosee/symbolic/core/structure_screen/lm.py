import numpy as np
import sympy as sp
from scipy.optimize import least_squares

from .encoding import parse_formula


def r2_score(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if pred.shape != y.shape or not np.all(np.isfinite(y)) or not np.all(np.isfinite(pred)):
        return float("-inf")

    def safe_sum_squares(values):
        values = np.asarray(values, dtype=float)
        scale = float(np.max(np.abs(values))) if values.size else 0.0
        if not np.isfinite(scale):
            return float("inf")
        if scale == 0.0:
            return 0.0
        with np.errstate(over="ignore", invalid="ignore"):
            normalized = float(np.sum(np.square(values / scale)))
        if not np.isfinite(normalized):
            return float("inf")
        limit = np.sqrt(np.finfo(float).max / max(normalized, 1.0))
        return float("inf") if scale > limit else scale * scale * normalized

    with np.errstate(over="ignore", invalid="ignore"):
        numerator = safe_sum_squares(y - pred)
        denominator = safe_sum_squares(y - np.mean(y))
    if denominator <= 0 or not np.isfinite(denominator) or not np.isfinite(numerator):
        return float("-inf") if numerator > 0 else float("nan")
    return float(1.0 - numerator / denominator)


def has_invalid_symbolic_values(expr):
    """Reject expressions that SymPy/NumPy cannot safely turn into real code."""
    return bool(expr.has(sp.nan, sp.zoo, sp.oo, -sp.oo, sp.I))


def parameterize_constants(expr):
    """Replace numeric constants with parameters, except inside exponents."""
    params = []

    def visit(node, in_exponent=False):
        if node.is_Number:
            if in_exponent:
                return node
            param = sp.Symbol(f"p{len(params)}")
            params.append(param)
            return param
        if not node.args:
            return node
        if node.is_Pow:
            return sp.Pow(visit(node.base), visit(node.exp, True))
        return node.func(*(visit(arg, in_exponent) for arg in node.args))

    return visit(expr), params


def fit_constants(formula, X, y, restarts=5, max_nfev=300, seed=42):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or len(X) != len(y) or not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        return None
    try:
        expr = parse_formula(formula)
        if has_invalid_symbolic_values(expr):
            return None
        template, params = parameterize_constants(expr)
        if has_invalid_symbolic_values(template):
            return None
    except Exception:
        return None
    variables = sorted(template.free_symbols - set(params), key=lambda s: s.name)
    try:
        columns = []
        for symbol in variables:
            name = str(symbol)
            if not name.startswith("x") or not name[1:].isdigit():
                return None
            column = int(name[1:]) - 1
            if column < 0 or column >= X.shape[1]:
                return None
            columns.append(column)
        fn = sp.lambdify([*variables, *params], template, "numpy")
    except Exception:
        # Includes SymPy printer failures such as ComplexInfinity/zoo.
        return None
    rng, best = np.random.default_rng(seed), None

    def predict(p):
        with np.errstate(all="ignore"):
            raw = np.asarray(fn(*[X[:, column] for column in columns], *p))
        if np.iscomplexobj(raw):
            if np.any(np.abs(np.imag(raw)) > 1e-12):
                raise ValueError("complex prediction")
            raw = np.real(raw)
        v = np.asarray(raw, dtype=float)
        return np.full(len(y), float(v)) if v.ndim == 0 else v.reshape(-1)

    def residual(p):
        try:
            v = predict(p)
            if (v.shape != y.shape or not np.all(np.isfinite(v))
                    or (v.size and np.max(np.abs(v)) >= 1e150)):
                return np.full(len(y), 1e6)
            with np.errstate(over="ignore", invalid="ignore"):
                difference = v - y
            return difference if np.all(np.isfinite(difference)) else np.full(len(y), 1e6)
        except Exception:
            return np.full(len(y), 1e6)

    if not params:
        try:
            pred = predict([])
            score = r2_score(y, pred)
        except Exception:
            return None
        return ({"true_r2": score, "constants": [], "optimized_formula": str(template),
                 "prediction": pred} if np.isfinite(score) else None)
    starts = [rng.uniform(-2, 2, len(params)) for _ in range(restarts)]
    for p0 in starts:
        try:
            result = least_squares(residual, p0, method="lm", max_nfev=max_nfev)
            pred = predict(result.x)
            score = r2_score(y, pred)
            if np.isfinite(score) and (best is None or score > best["true_r2"]):
                best = {"true_r2": score, "constants": result.x.tolist(), "optimized_formula": str(template.subs(dict(zip(params, result.x)))), "prediction": pred}
        except Exception:
            continue
    return best
