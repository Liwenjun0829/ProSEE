import numpy as np
import sympy as sp

from structure_screen.legacy_gp_adapter import _predict, _valid


def test_legacy_prediction_and_validation_are_finite():
    X = np.column_stack((np.linspace(1.0, 2.0, 20), np.linspace(2.0, 5.0, 20) ** 1.2))
    expr = sp.sympify("x1/x2")
    prediction = _predict(expr, X)
    assert prediction is not None
    assert np.all(np.isfinite(prediction))
    assert _valid(expr, X, set(), 20) == expr


def test_legacy_validation_rejects_unknown_variables():
    X = np.ones((10, 2))
    assert _valid(sp.Symbol("x3"), X, set(), 20) is None
