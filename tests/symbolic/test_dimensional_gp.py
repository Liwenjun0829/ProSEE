import json
import random
from pathlib import Path

import numpy as np
import sympy as sp

from structure_screen import dimensional_gp as dgp


def dimensions():
    return {sp.Symbol(f"x{i}"): (dgp.Fraction(1),) for i in range(1, 5)}


def test_expand_to_four_variables_adds_unit_derived_columns():
    X = np.arange(10.0).reshape(5, 2)
    expanded, substitutions = dgp.expand_to_four_variables(X)
    assert expanded.shape == (5, 4)
    assert np.array_equal(expanded[:, :2], X)
    assert np.all(expanded[:, 2:] == 1.0)
    assert substitutions == {"x3": 1, "x4": 1}


def test_dimensional_rules_cover_arithmetic_and_transcendentals():
    dims = dimensions()
    zero, one, two = (dgp.Fraction(0),), (dgp.Fraction(1),), (dgp.Fraction(2),)
    assert dgp.infer_dimension(sp.sympify("x1+x2"), dims) == one
    assert dgp.infer_dimension(sp.sympify("x1*x2"), dims) == two
    assert dgp.infer_dimension(sp.sympify("x1/x2"), dims) == zero
    assert dgp.infer_dimension(sp.sympify("x1+1"), dims) is None
    assert dgp.infer_dimension(sp.sympify("log(x1)"), dims) is None
    assert dgp.infer_dimension(sp.sympify("sin(x1)"), dims) is None
    assert dgp.infer_dimension(sp.sympify("exp(x1)"), dims) is None
    assert dgp.infer_dimension(sp.sympify("log(x1/x2)"), dims) == zero
    assert dgp.infer_dimension(sp.sympify("cos(x1/x2)"), dims) == zero
    assert dgp.dimensionally_valid(sp.sympify("x1/x2"), dims, [0])
    assert not dgp.dimensionally_valid(sp.sympify("x1*x2"), dims, [0])


def test_restore_formula_substitutes_only_derived_variables():
    restored = sp.sympify(dgp.restore_original_formula("x1*x3/(x2*x4)", 2))
    assert sp.simplify(restored - sp.Symbol("x1") / sp.Symbol("x2")) == 0
    assert restored.free_symbols == {sp.Symbol("x1"), sp.Symbol("x2")}


def test_native_4d_loading_is_structurally_deduplicated(tmp_path):
    path = tmp_path / "formulas.json"
    path.write_text(json.dumps({
        "native": ["x1/x2"],
        "wrapped": [],
        "generation_metadata": {
            "native_4d": ["2*x1/x2", "9*x1/x2", "x1+x2"]
        },
    }), encoding="utf-8")
    loaded = dgp.load_deduplicated_native_4d(path, dimensions())
    assert len(loaded) == 1
    assert dgp.infer_dimension(loaded[0], dimensions()) == (dgp.Fraction(0),)


def test_ratio_basis_mode_uses_only_three_scaled_ratios_or_constants():
    dims = dimensions()
    cfg = {
        "ratio_basis_only": True,
        "ratio_terminal_probability": 1.0,
        "constant_min": 0.5,
        "constant_max": 1.5,
    }
    terminals = [
        dgp._choose_terminal(random.Random(seed), [sp.sympify("x1/x2")], dims, cfg)[0]
        for seed in range(30)
    ]
    allowed_ratios = {
        sp.Symbol("x1") / sp.Symbol("x4"),
        sp.Symbol("x2") / sp.Symbol("x4"),
        sp.Symbol("x3") / sp.Symbol("x4"),
    }
    for terminal in terminals:
        coefficient, ratio = terminal.as_coeff_Mul()
        assert coefficient != 1
        assert ratio in allowed_ratios
        assert dgp.infer_dimension(terminal, dims) == (dgp.Fraction(0),)


def test_ratio_basis_mode_rejects_invalid_dimensional_assumptions():
    dims = dimensions()
    bad_dims = dict(dims)
    bad_dims[sp.Symbol("x4")] = (dgp.Fraction(2),)
    with np.testing.assert_raises_regex(ValueError, "相同量纲"):
        dgp.generate_dimensional_target_free(
            {"ratio_basis_only": True}, [], bad_dims, (dgp.Fraction(0),)
        )
    with np.testing.assert_raises_regex(ValueError, "无量纲目标"):
        dgp.generate_dimensional_target_free(
            {"ratio_basis_only": True}, [], dims, (dgp.Fraction(1),)
        )


def test_saved_results_keep_both_formula_metrics_and_mapping(monkeypatch, tmp_path):
    formulas_path = tmp_path / "formulas.json"
    formulas_path.write_text(json.dumps({
        "native": ["0.5*x1"], "wrapped": [],
        "generation_metadata": {"native_4d": ["0.5*x1/x3"]},
    }), encoding="utf-8")
    candidate = sp.sympify("0.5*x1/x3")
    collapses_to_infinity = sp.sympify("x1/(x3-1)")
    monkeypatch.setattr(
        dgp, "generate_dimensional_target_free",
        lambda *args, **kwargs: [collapses_to_infinity, candidate],
    )
    X_original = np.column_stack((np.linspace(1.0, 3.0, 30), np.linspace(2.0, 4.0, 30)))
    X, _ = dgp.expand_to_four_variables(X_original)
    y = 2.0 * X_original[:, 0]
    output = tmp_path / "results.json"
    cfg = {
        "prescreen_top_k": 5, "pareto_max_results": 5,
        "population_size": 5, "generations": 1,
    }
    dimensional_cfg = {
        "variable_dimensions": {f"x{i}": [1] for i in range(1, 5)},
        "target_dimension": [0],
    }
    results = dgp.run_dimensional_gp(
        lambda formulas: np.asarray([.95, .9]), X, y, output,
        cfg, {"restarts": 3, "max_nfev": 100}, formulas_path,
        dimensional_cfg, 42, original_variable_count=2,
    )
    assert len(results) == 1
    saved = json.loads(output.read_text(encoding="utf-8"))
    row = saved[0]
    assert row["mapping_id"] == row["correspondence"]["mapping_id"]
    assert "restored_from_optimized_4d_formula" in row["correspondence"]
    assert "independently_fitted_original_formula" in row["correspondence"]
    assert row["derived_variable_substitutions"] == {"x3": 1, "x4": 1}
    assert "x3" in row["four_dimensional"]["formula"]
    assert "x3" not in row["original_variables"]["formula"]
    for name in ("four_dimensional", "original_variables"):
        assert {"formula", "optimized_formula", "true_r2", "mse", "rmse"} <= set(row[name])
        assert row[name]["true_r2"] > .999
