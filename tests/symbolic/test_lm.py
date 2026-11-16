import numpy as np
import sympy as sp
import warnings

from structure_screen.lm import fit_constants, parameterize_constants, r2_score


def test_exponent_constants_are_not_parameterized():
    x1 = sp.Symbol("x1")
    template, params = parameterize_constants(sp.Float(0.7) * x1 ** sp.Float(2.3))
    assert len(params) == 1
    assert any(
        node.is_Pow and node.exp == sp.Float(2.3)
        for node in sp.preorder_traversal(template)
    )


def test_fitting_keeps_exponent_fixed():
    X = np.linspace(0.5, 2.0, 40).reshape(-1, 1)
    y = 3.0 * X[:, 0] ** 2.3
    fitted = fit_constants("0.7*x1**2.3", X, y, restarts=3, max_nfev=200)
    optimized = sp.sympify(fitted["optimized_formula"])
    power = next(node for node in sp.preorder_traversal(optimized) if node.is_Pow)
    assert abs(float(power.exp) - 2.3) < 1e-12
    assert fitted["true_r2"] > 0.999999


def test_complex_infinity_formula_fails_safely_before_lambdify():
    X = np.linspace(1.0, 2.0, 20).reshape(-1, 1)
    y = X[:, 0]
    assert fit_constants("zoo*x1", X, y) is None
    assert fit_constants("x1/(1-1)", X, y) is None


def test_r2_extreme_prediction_does_not_emit_overflow_warning():
    y = np.arange(10.0)
    prediction = np.full(10, 1e308)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        score = r2_score(y, prediction)
    assert score == float("-inf")
    assert not any(issubclass(item.category, RuntimeWarning) for item in caught)
