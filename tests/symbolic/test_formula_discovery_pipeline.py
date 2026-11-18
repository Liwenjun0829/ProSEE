import json
import random

import numpy as np
import pytest
import sympy as sp
from scipy.io import savemat

from formula_discovery_pipeline import (
    build_base_records,
    build_candidate_records,
    build_combination_records,
    fit_with_restarts,
    fit_formula_per_dataset,
    load_joint_csv_data,
    refit_top_ranked_formulas,
    reuse_top_ranked_formula_fits,
    refit_top_ranked_formula_pairs,
    choose_fit_samples_by_dataset,
    apply_wrapper,
    run,
    fit_piecewise_formula,
    load_realdata_mat,
    _evaluate_validation_row,
    search_piecewise_formula_pairs,
    add_gp_residual_autocorrelation,
    fit_coarse_piecewise_formula,
    piecewise_prediction_difference,
    _select_tree_top_k_for_refit,
    rank_lm_results_for_ablation,
    _breakpoint_prior,
    _restrict_split_indices,
    has_only_integer_exponents,
    filter_gp_integer_exponents,
    formula_row_has_only_integer_exponents,
    compute_joint_metrics,
    _apply_pipeline_mode,
    _pysr_structural_loss_function,
    _pysr_structural_penalty,
    _pysr_hv_window_front,
    select_stratified_gp_ranking,
    _has_at_least_two_formula_variables,
    _mode2_requires_binary_formula,
    _mode2_initial_pareto_complexity_intervals,
    select_precomputed_formula_artifact,
    DEFAULT_CONFIG,
    load_config,
    _array_data_sha256,
    _mode2_pareto_rows,
    _select_mode3_complexity_interval_batch,
    _select_mode5_min_complexity_zero_bases,
    _load_mode4_seed_completion,
    _mode4_seed_is_complete,
    _write_mode4_seed_completion,
)


def test_mode4_seed_completion_marker_round_trips(tmp_path):
    seed_dir = tmp_path / "seed_001"
    front_rows = [{"seed": 1518, "formula_id": "f1"}]
    summary = {"seed": 1518, "run_completed": True}
    history = [{"iteration": 1, "front": front_rows}]

    assert not _mode4_seed_is_complete(seed_dir, 5, 1518)
    _write_mode4_seed_completion(
        seed_dir, 5, 1518, front_rows, summary, history
    )
    marker = _load_mode4_seed_completion(seed_dir)

    assert _mode4_seed_is_complete(seed_dir, 5, 1518)
    assert marker["status"] == "completed"
    assert marker["experiment_mode"] == 5
    assert marker["seed"] == 1518
    assert marker["front_rows"] == front_rows
    assert marker["summary"] == summary
    assert marker["history"] == history
    assert not _mode4_seed_is_complete(seed_dir, 6, 1518)
    assert not _mode4_seed_is_complete(seed_dir, 5, 1519)


def test_precomputed_artifact_is_selected_by_data_mode():
    config = {
        "data": {"mode": "csv"},
        "precomputed_formula_feature_artifact": "csv.joblib",
        "precomputed_formula_feature_artifact_realdata_mat": "mat.joblib",
    }
    assert select_precomputed_formula_artifact(config) == "csv.joblib"
    config["data"]["mode"] = "realdata_mat"
    assert select_precomputed_formula_artifact(config) == "mat.joblib"


def test_realdata_empty_pareto_front_has_no_complexity_interval_error():
    library = [{"readable_expression": "x1"}]
    assert _mode2_initial_pareto_complexity_intervals(
        library, [], allow_empty_front=True
    ) == []


def test_realdata_piecewise_validation_uses_training_split_value():
    X = np.arange(1.0, 11.0).reshape(-1, 1)
    y = np.where(X[:, 0] <= 5.5, 2.0 * X[:, 0], 5.0 * X[:, 0])
    row = {
        "readable_expression": "c1*x1",
        "piecewise": True,
        "split_value": 5.5,
        "dataset_fit_results": [
            {"dataset_name": "segment_1", "parameters": [2.0]},
            {"dataset_name": "segment_2", "parameters": [5.0]},
        ],
    }
    metrics = _evaluate_validation_row(
        row, X, y,
        [{"name": "measured_pathloss", "start": 0, "end": len(y)}],
        [{"name": "measured_pathloss", "start": 0, "end": len(y)}],
        {"max_abs_value": 1e100, "complex_tolerance": 1e-12},
    )
    assert metrics is not None
    assert metrics["pooled_r2"] == pytest.approx(1.0)


def test_realdata_piecewise_validation_prefers_physical_breakpoint():
    X = np.arange(1.0, 11.0).reshape(-1, 1)
    y = np.where(X[:, 0] <= 5.0, 2.0 * X[:, 0], 5.0 * X[:, 0])
    row = {
        "readable_expression": "c1*x1",
        "piecewise": True,
        "split_value": 5.5,
        "split_prior_value": 5.0,
        "dataset_fit_results": [
            {"dataset_name": "segment_1", "parameters": [2.0]},
            {"dataset_name": "segment_2", "parameters": [5.0]},
        ],
    }
    metrics = _evaluate_validation_row(
        row, X, y,
        [{"name": "measured_pathloss", "start": 0, "end": len(y)}],
        [{"name": "measured_pathloss", "start": 0, "end": len(y)}],
        {"max_abs_value": 1e100, "complex_tolerance": 1e-12},
    )
    assert metrics is not None
    assert metrics["pooled_r2"] == pytest.approx(1.0)


def test_piecewise_prediction_difference_evaluates_both_fits_at_breakpoint():
    X = np.arange(1.0, 11.0).reshape(-1, 1)
    fit_result = {
        "split_index": 5,
        "split_value": 5.5,
        "dataset_fit_results": [
            {"dataset_name": "segment_1", "parameters": [2.0]},
            {"dataset_name": "segment_2", "parameters": [5.0]},
        ],
    }

    # 5*x evaluated on both sides at x=5.5 differs by 3*5.5=16.5.
    # Evaluating at the adjacent samples instead would give 20.
    assert piecewise_prediction_difference("c1*x1", fit_result, X) == pytest.approx(16.5)


def test_lm_rejects_domain_restricted_formula_below_coverage_threshold():
    X = np.arange(-4.0, 5.0).reshape(-1, 1)
    y = X[:, 0].copy()
    attempts, best = fit_with_restarts(
        {"id": "restricted", "readable_expression": "c1*log(x1)"}, X, y,
        {
            "lm_initializations": 2,
            "max_nfev": 50,
            "initialization_range": [-2, 2],
            "minimum_valid_fraction": 0.9,
        },
        np.random.default_rng(13),
    )

    assert all(not row["all_datasets_valid"] for row in attempts)
    assert best["parameters"] is None
from generate_zero_sum_bases import (
    generate_zero_sum_bases,
    generate_exponent_vectors,
    project_four_variable_exponents,
)
from pareto_formula_evolution import (
    crossover, mutate, _has_log_abs_x1, _parameterize_numeric_constants,
    _merge_duplicate_constants, _replace_operator, _replace_variable,
    _replace_variable_with_constant,
    _merge_log_terms, _rename_conflicting_parameters,
    _parameter_symbols,
    prune_insignificant_constants,
    merge_log_formulas,
    _adaptive_initial_r2_threshold,
    _valid_child, run_lightweight_gp,
)
from structure_screen.tree_search_ops import valid
from structure_r2_ranker import OperatorPrimitiveMatrixEncoder, TreeCoordinateEncoder


def test_pipeline_mode_does_not_mutate_gp_settings():
    config = {
        "gp": {
            "stratified_selection": False,
            "complexity_layers": 3,
            "top_k": 15,
        },
        "ranking": {"symbol_operator_stratified": True},
    }
    original_gp = json.loads(json.dumps(config["gp"]))
    mode_three = _apply_pipeline_mode(config, 3)
    assert mode_three["gp"] == original_gp
    assert mode_three["mode2"]["interval_layered_selection"] is True
    assert mode_three["mode2"]["max_iterations"] == 40
    assert "max_training_samples" not in mode_three["ranking"]
    assert mode_three["ranking"]["symbol_operator_stratified"] is False


def test_mode2_binary_zeroing_constraint_excludes_realdata_mat():
    assert _mode2_requires_binary_formula({"data": {"mode": "csv"}}, 2)


def test_mode2_binary_zeroing_constraint_applies_only_to_csv():
    assert not _mode2_requires_binary_formula({"data": {"mode": "realdata_mat"}}, 2)
    assert _has_at_least_two_formula_variables(sp.sympify("x1*x2"))
    assert not _has_at_least_two_formula_variables(sp.sympify("x1"))


def test_mode5_complexity_basis_keeps_simplest_zero_result_per_source():
    rows = [
        {"source_formula_id": "source_a", "formula_id": "a_complex",
         "readable_expression": "x1*x2", "complexity": 6, "mean_r2": 0.99},
        {"source_formula_id": "source_a", "formula_id": "a_simple",
         "readable_expression": "x1 + x2", "complexity": 4, "mean_r2": 0.80},
        {"source_formula_id": "source_b", "formula_id": "b_tie_low_r2",
         "readable_expression": "x1 + x2", "complexity": 4, "mean_r2": 0.80},
        {"source_formula_id": "source_b", "formula_id": "b_tie_high_r2",
         "readable_expression": "x1 + x2", "complexity": 4, "mean_r2": 0.90},
    ]

    selected = _select_mode5_min_complexity_zero_bases(rows)

    assert [row["formula_id"] for row in selected] == [
        "a_simple", "b_tie_high_r2"
    ]


def test_mode2_realdata_pareto_uses_pooled_r2_but_other_modes_use_mean_r2():
    rows = [
        {
            "formula_id": "pooled_better",
            "readable_expression": "x1",
            "pooled_r2": 0.90,
            "segment_prediction_difference": 1.0,
            "dataset_fit_results": [
                {"r2": 0.80}, {"r2": 0.80},
            ],
        },
        {
            "formula_id": "mean_better",
            "readable_expression": "x2",
            "pooled_r2": 0.70,
            "segment_prediction_difference": 1.0,
            "dataset_fit_results": [
                {"r2": 0.85}, {"r2": 0.85},
            ],
        },
    ]

    realdata_front = _mode2_pareto_rows(
        rows, {"data": {"mode": "realdata_mat"}}
    )
    csv_front = _mode2_pareto_rows(rows, {"data": {"mode": "csv"}})

    assert [row["formula_id"] for row in realdata_front] == ["pooled_better"]
    assert [row["formula_id"] for row in csv_front] == ["mean_better"]


def test_mode2_initial_pareto_complexity_intervals_are_lower_exclusive():
    library = [
        {"readable_expression": "x1", "complexity": 1},
        {"readable_expression": "x1*x2", "complexity": 2},
        {"readable_expression": "x1 + x2", "complexity": 3},
        {"readable_expression": "x1**2 + x2", "complexity": 4},
        {"readable_expression": "x1**3 + x2", "complexity": 4},
    ]
    front = [
        {"readable_expression": "c0*x1", "complexity": 2},
        {"readable_expression": "c0*x1*x2", "complexity": 4},
        {"readable_expression": "c1*x1*x2", "complexity": 4},
    ]

    assert _mode2_initial_pareto_complexity_intervals(library, front) == [
        {"lower_complexity": 0, "upper_complexity": 2, "formula_count": 1},
        {"lower_complexity": 2, "upper_complexity": 4, "formula_count": 3},
    ]


def test_mode3_selects_one_pending_formula_per_pareto_interval():
    rows = [
        {"formula_id": "low", "complexity": 1, "predicted_rank": 3},
        {"formula_id": "mid", "complexity": 3, "predicted_rank": 1},
        {"formula_id": "high", "complexity": 4, "predicted_rank": 2},
    ]
    intervals = [
        {"lower_complexity": 0, "upper_complexity": 2},
        {"lower_complexity": 2, "upper_complexity": 4},
    ]

    selected, metadata = _select_mode3_complexity_interval_batch(rows, intervals)

    assert [row["formula_id"] for row in selected] == ["mid", "low"]
    assert [item["available_count"] for item in metadata] == [1, 2]


def test_constant_zeroing_does_not_fit_unary_mode2_csv_candidate():
    source = {
        "formula_id": "source",
        "readable_expression": "c0*x1*x2 + c1*x1",
        "dataset_fit_results": [{
            "dataset_name": "dataset_1", "r2": 0.9,
            "parameters": [1.0, 1.0], "parameter_names": ["c0", "c1"],
        }],
    }
    fitted_expressions = []

    def fit_formula(record, **_kwargs):
        fitted_expressions.append(record["readable_expression"])
        return {
            "dataset_fit_results": [{
                "dataset_name": "dataset_1", "r2": 0.9,
                "parameters": [1.0], "parameter_names": ["c0"],
            }],
        }

    prune_insignificant_constants(
        [source], fit_formula, max_r2_drop=0.01, feature_count=2,
        allow_single_variable=False,
    )
    assert all(
        _has_at_least_two_formula_variables(sp.sympify(expression))
        for expression in fitted_expressions
    )


def test_refit_top_k_uses_predicted_rank_for_all_formulas():
    ranking = [
        {
            "formula_id": f"f{index}",
            "readable_expression": f"x{index % 2 + 1}",
            "predicted_rank": index,
            "is_training_formula": index < 6,
            "actual_fit_r2": [0.2, 0.95, 0.4, 0.8, 0.7, 0.6, None, None][index],
        }
        for index in range(8)
    ]
    selected = _select_tree_top_k_for_refit(ranking, 4, ranking_config={})
    assert [row["formula_id"] for row in selected] == ["f0", "f1", "f2", "f3"]


def test_gp_stratified_selection_covers_each_available_operator():
    ranking = [
        {
            "formula_id": f"f{index}",
            "readable_expression": "x1",
            "predicted_rank": index,
            "operator": operator,
        }
        for index, operator in enumerate(
            ("identity", "log_abs", "sin", "cos", "log_abs_sin", "log_abs_cos")
        )
    ]
    selected, metadata = select_stratified_gp_ranking(
        ranking, {"top_k": 6, "complexity_layers": 3}
    )
    assert len(selected) == 6
    assert metadata["available_operators"] == [
        "cos", "identity", "log_abs", "log_abs_cos", "log_abs_sin", "sin"
    ]
    assert set(metadata["operator_coverage"]) == set(metadata["available_operators"])
    assert {row["operator"] for row in selected} == set(metadata["available_operators"])


def test_gp_stratified_selection_rejects_top_k_that_cannot_cover_operators():
    ranking = [
        {"formula_id": "identity", "readable_expression": "x1",
         "predicted_rank": 1, "operator": "identity"},
        {"formula_id": "log", "readable_expression": "x1",
         "predicted_rank": 2, "operator": "log_abs"},
    ]
    with pytest.raises(ValueError, match="覆盖全部算子"):
        select_stratified_gp_ranking(
            ranking, {"top_k": 1, "complexity_layers": 1}
        )


def _ranking_formula(formula_id, expression, rank):
    return {
        "formula_id": formula_id,
        "readable_expression": expression,
        "predicted_rank": rank,
        "ranking_score": 1.0 / rank,
        "formula_family": "single",
        "sympy_expression": expression,
    }


def test_constant_pruning_isolates_multivariable_protected_nonlinear_addends():
    source = {
        "formula_id": "source",
        "readable_expression": (
            "c0*log(Abs(x1*x2)) + c1*sin(x1 + x2)"
        ),
        "dataset_fit_results": [{
            "dataset_name": "dataset_1", "r2": 0.9,
            "parameters": [1.0, 2.0], "parameter_names": ["c0", "c1"],
        }],
    }

    def fit_formula(record, **_kwargs):
        return {
            "formula_id": record["id"], "r2": 0.8,
            "dataset_fit_results": [{
                "dataset_name": "dataset_1", "r2": 0.8,
                "parameters": [1.0, 2.0],
                "parameter_names": ["c0", "c1"],
            }],
        }

    candidates, checks, _zeroed = prune_insignificant_constants(
        [source], fit_formula, max_r2_drop=0.01, feature_count=2,
    )

    isolated = [
        row for row in candidates
        if row.get("isolated_protected_nonlinear_addend")
    ]
    assert len(isolated) == 2
    assert {row["readable_expression"] for row in isolated} == {
        "c0*log(Abs(x1*x2))", "c1*sin(x1 + x2)",
    }
    assert all(
        {sp.Symbol("x1"), sp.Symbol("x2")}
        <= sp.sympify(row["readable_expression"]).free_symbols
        for row in isolated
    )
    assert sum(
        check.get("operation") == "isolate_protected_nonlinear_addend"
        and check["accepted"]
        for check in checks
    ) == 2

    one_variable_source = {
        **source,
        "readable_expression": "c0*log(Abs(x1)) + c1*sin(x2)",
    }
    rejected_candidates, rejected_checks, _ = prune_insignificant_constants(
        [one_variable_source], fit_formula, max_r2_drop=0.01, feature_count=2,
    )
    assert not any(
        row.get("isolated_protected_nonlinear_addend")
        for row in rejected_candidates
    )
    assert sum(
        check.get("failure_reason") == "isolated_term_missing_required_variable"
        for check in rejected_checks
    ) == 2


def test_constant_pruning_splits_original_addends_before_zeroing():
    source = {
        "formula_id": "source",
        "readable_expression": (
            "a1*log(Abs(c1_left*x1*x2 + c2_left*x1)) + "
            "a2*log(Abs(c1_right/x2 + c2_right*x1/x2))"
        ),
        "dataset_fit_results": [{
            "dataset_name": "dataset_1", "r2": 0.9,
            "parameters": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "parameter_names": [
                "a1", "a2", "c1_left", "c1_right", "c2_left", "c2_right",
            ],
        }],
    }

    def fit_formula(record, **_kwargs):
        return {
            "formula_id": record["id"], "r2": 0.9,
            "dataset_fit_results": [{
                "dataset_name": "dataset_1", "r2": 0.9,
                "parameters": [1.0, 2.0],
                "parameter_names": ["a1", "c1_left"],
            }],
        }

    candidates, _checks, _zeroed = prune_insignificant_constants(
        [source], fit_formula, max_r2_drop=0.01, feature_count=2,
    )
    isolated = {
        row["readable_expression"] for row in candidates
        if row.get("isolated_protected_nonlinear_addend")
    }
    assert isolated == {
        "a1*log(Abs(c1_left*x1*x2 + c2_left*x1))",
        "a2*log(Abs(c1_right/x2 + c2_right*x1/x2))",
    }


def test_log_merge_uses_half_warm_and_half_random_initializations():
    source = {
        "formula_id": "source",
        "readable_expression": "c0*log(Abs(x1)) + c1*log(Abs(x2))",
        "mean_r2": 0.9,
        "complexity": 6,
        "dataset_fit_results": [{
            "dataset_name": "dataset_1", "r2": 0.9,
            "parameters": [1.0, -1.0], "parameter_names": ["c0", "c1"],
        }],
    }
    calls = []

    def fit_formula(record, **kwargs):
        calls.append(kwargs)
        return {
            "dataset_fit_results": [{
                "dataset_name": "dataset_1", "r2": 0.9,
                "parameters": [1.0], "parameter_names": ["p0"],
            }],
        }

    _results, checks, count = merge_log_formulas(
        [source], fit_formula, max_r2_drop=0.01, lm_initializations=7,
    )

    assert count == 1
    assert checks[0]["accepted"]
    assert calls[0]["warm_start_fraction"] == 0.5
    assert calls[0]["lm_initializations"] == 7


def test_lm_ablation_ignores_tree_top_k_constraints():
    ranking = [
        _ranking_formula("first", "x1", 1),
        _ranking_formula("second", "x2", 2),
        _ranking_formula("third", "x1 + x2", 3),
    ]
    selected = _select_tree_top_k_for_refit(
        ranking,
        3,
        ranking_config={
            "ablation_enabled": True,
            "symbol_operator_stratified": True,
        },
    )

    assert [row["formula_id"] for row in selected] == ["first", "second", "third"]


def test_lm_ablation_can_reuse_operator_stratified_top_k_selector():
    ranking = [
        _ranking_formula("log_first", "log(Abs(x1))", 1),
        _ranking_formula("log_second", "log(Abs(x2))", 2),
        _ranking_formula("log_trig_first", "log(Abs(sin(x1)))", 3),
        _ranking_formula("log_trig_second", "log(Abs(cos(x2)))", 4),
    ]
    selected = _select_tree_top_k_for_refit(
        ranking,
        4,
        ranking_config={
            "ablation_enabled": True,
            "ablation_use_operator_stratified": True,
            "symbol_operator_stratified": True,
        },
    )

    assert [row["formula_id"] for row in selected] == [
        "log_first", "log_second", "log_trig_first", "log_trig_second"
    ]
    assert [row["operator_stratum"] for row in selected] == [
        "log_only", "log_only", "log_and_trig", "log_and_trig"
    ]


def test_lm_ablation_ranks_by_r2():
    payload = {
        "single_best": [
            {"formula_id": "low_r2", "readable_expression": "x1", "r2": 0.2},
            {"formula_id": "high_r2", "readable_expression": "x2", "r2": 0.8},
        ],
        "combination_best": [],
    }
    ranking, metadata = rank_lm_results_for_ablation(payload, {"refit_top_k": 2})

    assert [row["formula_id"] for row in ranking] == ["high_r2", "low_r2"]
    assert [row["ranking_score"] for row in ranking] == [0.8, 0.2]
    assert metadata["selection_basis"] == "direct_lm_r2"


def test_tree_coordinates_include_parent_coordinates():
    encoder = TreeCoordinateEncoder(max_nodes=8).fit(["x1 + x2"])
    features = encoder.transform(["x1 + x2"])[0].reshape(8, 8)

    assert features.shape == (8, 8)
    assert np.all(features[0, 4:] == 0.0)  # root has no parent
    assert np.allclose(features[1, 4:], features[0, :4])
    assert np.allclose(features[2, 4:], features[0, :4])
    assert len(encoder.feature_names) == 8 * 8


def test_tree_coordinates_use_ancestor_products_in_explicit_mode():
    encoder = TreeCoordinateEncoder(max_nodes=8, ancestor_products=True).fit(
        ["sin(x1) + x2"]
    )
    features = encoder.transform(["sin(x1) + x2"])[0].reshape(8, -1)

    assert encoder.max_ancestor_count == 2
    assert features.shape == (8, encoder.feature_width)
    assert encoder.feature_width == 18
    assert not any(name.endswith("_path") for name in encoder.feature_names)
    assert features[0, :4].tolist() == pytest.approx(features[0, :4].tolist())
    assert np.all(features[0, len(encoder._ancestor_field_names):] == 0.0)


def test_mode_two_tree_coordinates_use_only_nearest_ancestor_without_composition():
    encoder = TreeCoordinateEncoder(
        max_nodes=8,
        ancestor_products=True,
        nearest_ancestor_only=True,
        include_composition_features=False,
    ).fit(["sin(cos(x1))"])
    features = encoder.transform(["sin(cos(x1))"])[0].reshape(8, -1)

    assert encoder.nearest_ancestor_only
    assert not encoder.include_composition_features
    field_width = len(encoder._ancestor_field_names)
    assert field_width == 5
    assert encoder.feature_width == field_width * 2
    assert features.shape == (8, encoder.feature_width)
    assert not any("operator_" in name or "variable_" in name
                   for name in encoder._ancestor_field_names)
    # The grandchild has one current-node vector plus one nearest-ancestor
    # product; the remaining ancestor-product slot is zero padded.
    assert np.any(features[2, field_width:2 * field_width])
    assert np.all(features[2, 2 * field_width:] == 0.0)


def test_operator_primitive_matrix_uses_generator_metadata_and_fixed_slots():
    bases = [
        {"id": f"base_{index:06d}", "readable_expression": f"x1**{index + 1}"}
        for index in range(8)
    ]
    encoder = OperatorPrimitiveMatrixEncoder(bases)
    records = [
        {"operator": "identity", "base_ids": ["base_000000"]},
        {"operator": "log_abs_sin", "base_ids": ["base_000001", "base_000002"]},
        {"operators": ["identity", "log_abs"], "base_ids": [
            ["base_000003"], ["base_000004"],
        ]},
    ]
    matrix = encoder.transform(records).reshape(len(records), *encoder.matrix_shape)
    merged = encoder.transform_merged(records).reshape(
        len(records), *encoder.merged_matrix_shape
    )

    assert encoder.matrix_shape == (12, 8)
    assert matrix[0, 0, 0] == 1.0  # identity_1
    assert np.count_nonzero(matrix[0]) == 1
    # The composite occupies only log_abs_sin, never log_abs or sin.
    composite_row = 4 * 2
    assert matrix[1, composite_row, 1:3].tolist() == [1.0, 1.0]
    assert not np.any(matrix[1, 2:4])  # log_abs rows
    assert not np.any(matrix[1, 4:8])  # sin/cos rows
    assert matrix[2, 0, 3] == 1.0      # identity_1
    assert matrix[2, 2, 4] == 1.0      # log_abs_1
    assert not np.any(matrix[2, 1])    # unused identity_2 remains zero
    assert encoder.merged_matrix_shape == (8, 8)
    assert merged[1, 6, 1] == 1.0     # merged log_abs_trig_1
    assert not np.any(merged[1, 2:6])  # no separate sin/cos rows


def test_gp_integer_exponent_filter():
    assert has_only_integer_exponents("c1*x1**2 + sin(x1)")
    assert has_only_integer_exponents("c1*x1**(-1)")
    assert not has_only_integer_exponents("sqrt(x1)")
    assert not has_only_integer_exponents("x1**x2")
    assert not has_only_integer_exponents("x1**2.0")

    kept, report = filter_gp_integer_exponents([
        {"formula_id": "good", "readable_expression": "c1*x1**2"},
        {"formula_id": "bad", "readable_expression": "x1**0.5"},
    ])
    assert [row["formula_id"] for row in kept] == ["good"]
    assert report["rejected_count"] == 1
    assert formula_row_has_only_integer_exponents({
        "readable_expression": "if x1 <= 1: x1**(-1); else: x1**2",
        "left_formula": {"readable_expression": "x1**(-1)"},
        "right_formula": {"readable_expression": "x1**2"},
    })


def test_pysr_structural_loss_is_tree_aware_and_rejects_tunable_power_constants():
    cfg = {
        "structural_loss_during_search": True,
        "structural_loss_penalty": 1e300,
        "require_all_features": True,
        "require_log_ancestor_for_x1": True,
        "forbid_bfgs_constants_in_power_exponents": True,
        "unary_operators": [
            "sin", "cos", "log_abs(x) = log(abs(x))",
            "log_abs_sin(x) = log(abs(sin(x)))",
        ],
        "binary_operators": ["+", "-", "*", "/", "^"],
    }
    loss = _pysr_structural_loss_function(cfg, ["x1", "x2"])
    assert loss is not None
    assert "function pysr_structural_loss" in loss
    assert "node.op in [3, 4]" in loss
    assert "node.op in [5]" in loss
    assert "for feature_index in 1:2" in loss
    assert "has_tunable_power_exponent" in loss

    power_only_cfg = {
        "forbid_bfgs_constants_in_power_exponents": True,
    }
    assert _pysr_structural_penalty("x1**2.0", power_only_cfg) == 1e300
    assert _pysr_structural_penalty("x1**2", power_only_cfg) == 0.0


def test_pysr_reporting_window_uses_fifteen_percent_r2_gap():
    rows = [
        {"formula_id": "best", "validation_mean_r2": 1.0},
        {"formula_id": "boundary", "validation_mean_r2": 0.85},
        {"formula_id": "outside", "validation_mean_r2": 0.849},
    ]
    selected = _pysr_hv_window_front(rows, "csv", "validation")
    assert [row["formula_id"] for row in selected] == ["best", "boundary"]


def test_pysr_additive_term_limit_is_at_most_two():
    cfg = {
        "max_nonconstant_additive_terms": 2,
        "structural_loss_during_search": True,
        "binary_operators": ["+", "-", "*", "/"],
    }
    assert _pysr_structural_penalty("x1", cfg) == 0.0
    assert _pysr_structural_penalty("x1 + x2", cfg) == 0.0
    assert _pysr_structural_penalty("x1 + x2 + x3", cfg) == 1e300
    loss = _pysr_structural_loss_function(cfg, ["x1", "x2"])
    assert "nonconstant_additive_terms(tree) > 2" in loss


def test_gp_parameter_symbols_are_processed_in_stable_order():
    expression = sp.sympify("a1*x1 + a2*x2 + c1*x1 + p0")

    assert [str(symbol) for symbol in
            sorted(_parameter_symbols(expression), key=str)] == [
                "a1", "a2", "c1", "p0"
            ]


def small_config():
    return {
        "bases": {"arities": [2], "max_abs_exponent": 2},
        "candidates": {
            "max_count": 12,
            "two_base_ratio": 0.25,
            "wrappers": ["identity", "log_abs", "sin"],
            "max_sampling_attempts": 2000,
        },
        "combinations": {
            "max_count": 20,
            "types": ["add", "multiply"],
            "max_sampling_attempts": 2000,
        },
    }


def test_lower_arity_bases_project_four_variable_pool_and_deduplicate():
    source = generate_exponent_vectors(4, max_abs=1)
    expected = sorted({alpha[:2] for alpha in source if any(alpha[:2])})
    projected = project_four_variable_exponents(2, max_abs=1)
    bases = generate_zero_sum_bases(2, max_abs=1)

    assert projected == expected
    assert len(projected) == len(set(projected))
    assert [
        tuple(int(expr.as_powers_dict().get(sp.Symbol(f"x{i}"), 0))
              for i in range(1, 3))
        for expr in bases
    ] == expected
    assert any(sum(alpha) != 0 for alpha in projected)


def test_binary_bases_include_pure_reciprocals_and_ratios():
    projected = project_four_variable_exponents(2, max_abs=2)
    assert (-1, 0) in projected
    assert (0, -1) in projected
    assert (-2, 0) in projected
    assert (0, -2) in projected
    assert (-1, 1) in projected
    assert (1, -1) in projected


def test_candidates_and_combinations_have_independent_parameters():
    config = small_config()
    bases = build_base_records(config, arity=2)
    candidates = build_candidate_records(bases, config, random.Random(4))
    combinations = build_combination_records(candidates, config, random.Random(5))
    repeated_bases = build_base_records(config, arity=2)
    repeated_candidates = build_candidate_records(
        repeated_bases, config, random.Random(4)
    )
    repeated_combinations = build_combination_records(
        repeated_candidates, config, random.Random(5)
    )

    assert candidates
    assert combinations
    assert bases == repeated_bases
    assert candidates == repeated_candidates
    assert combinations == repeated_combinations
    for candidate in candidates:
        assert candidate["parameter_symbols"]
        assert len(candidate["parameter_symbols"]) == candidate["base_count"]
    for combination in combinations:
        names = set(combination["parameter_symbols"])
        assert "a1" in names
        if combination["combination_type"] == "add":
            assert "a2" in names
        assert any(name.endswith("_left") for name in names)
        assert any(name.endswith("_right") for name in names)


def test_simplified_numeric_constants_become_fresh_gp_parameters():
    expr = sp.sympify("a1*log(Abs(c1*x1 + x2/x2)) + p0 + x1**-1")

    parameterized = _parameterize_numeric_constants(expr)

    assert parameterized.has(sp.Symbol("p1"))
    assert not parameterized.has(sp.Integer(1))
    assert parameterized.has(sp.Pow(sp.Symbol("x1"), sp.Integer(-1)))


def test_log_merge_uses_fitted_coefficient_signs():
    expr = sp.sympify("a1*log(x1) + a2*log(x2)")
    fit = {
        "dataset_fit_results": [{
            "parameter_names": ["a1", "a2"],
            "parameters": [2.0, -3.0],
        }]
    }

    merged = _merge_log_terms(expr, fit)

    assert merged == sp.Symbol("p0") * sp.log(sp.Symbol("x1") / sp.Symbol("x2"))


def test_combinations_support_configurable_term_count():
    config = small_config()
    config["combinations"]["max_count"] = 40
    config["combinations"]["max_terms"] = 3
    bases = build_base_records(config, arity=2)
    candidates = build_candidate_records(bases, config, random.Random(14))
    combinations = build_combination_records(candidates, config, random.Random(15))

    assert combinations
    assert all(2 <= row["combination_term_count"] <= 3
               for row in combinations)
    ternary = [row for row in combinations
               if row["combination_term_count"] == 3]
    assert ternary
    for row in ternary:
        assert len(row["candidate_ids"]) == 3
        assert len(row["operators"]) == 3
        assert all(f"_term{index}" in row["readable_expression"]
                   for index in range(1, 4))


def test_gp_crossover_renames_conflicting_donor_parameters():
    left = sp.sympify("a1*x1 + a2*x2")
    donor = sp.sympify("a1 + c1_left*x1")

    renamed = _rename_conflicting_parameters(donor, set(left.free_symbols))

    assert renamed.has(sp.Symbol("a1_gp1"))
    assert not renamed.has(sp.Symbol("a1"))
    assert renamed.has(sp.Symbol("c1_left"))

    # Repeated occurrences in a donor remain tied to the same fresh
    # parameter, rather than creating inconsistent names within the subtree.
    repeated = _rename_conflicting_parameters(
        sp.sympify("a1*x1 + a1*x2"), set(left.free_symbols)
    )
    assert repeated.has(sp.Symbol("a1_gp1"))
    assert repeated.coeff(sp.Symbol("x1")) == sp.Symbol("a1_gp1")
    assert repeated.coeff(sp.Symbol("x2")) == sp.Symbol("a1_gp1")


def test_gp_final_front_keeps_historical_initial_solution():
    initial_expr = "c1*log(Abs(x1))"
    payload = {
        "top_ranked_refit_results": [
            {
                "formula_id": "initial_best",
                "readable_expression": initial_expr,
                "dataset_fit_results": [{
                    "dataset_name": "dataset_1", "r2": 0.99,
                    "parameters": [2.0], "parameter_names": ["c1"],
                }],
            },
            {
                "formula_id": "initial_second",
                "readable_expression": "c2*log(Abs(x1)) + sin(x1)",
                "dataset_fit_results": [{
                    "dataset_name": "dataset_1", "r2": 0.2,
                    "parameters": [1.0], "parameter_names": ["c2"],
                }],
            },
        ],
    }

    def fit_formula(record):
        # Every generated child is deliberately worse than the initial
        # solution.  The historical archive must still retain the initial one.
        return {
            "formula_id": record["id"], "r2": 0.1,
            "dataset_fit_results": [{
                "dataset_name": "dataset_1", "r2": 0.1,
                "parameters": [0.1], "parameter_names": ["c1"],
            }],
        }

    front, metadata = run_lightweight_gp(
        payload,
        {"top_k": 2, "initial_r2_threshold": 0.0,
         "initial_r2_threshold_mode": "max_r2_gap",
         "initial_r2_max_gap": 0.1,
         "population_size": 2, "generations": 1,
         "mutation_rate": 0.0, "crossover_rate": 1.0, "seed": 3},
        feature_count=1, dataset_groups=[{"name": "dataset_1"}],
        fit_formula=fit_formula,
    )

    assert any(row["formula_id"] == "initial_best" for row in front)
    assert metadata["historical_formula_count"] >= 1
    assert metadata["initial_r2_threshold"] == pytest.approx(0.89)
    assert metadata["initial_r2_threshold_details"]["method"] == "max_r2_gap"


def test_gp_initial_r2_threshold_sees_all_post_significance_rows():
    payload = {
        "top_ranked_refit_results": [
            {
                "formula_id": "seed_1",
                "readable_expression": "c1*log(Abs(x1))",
                "dataset_fit_results": [{"dataset_name": "dataset_1", "r2": 0.90}],
            },
            {
                "formula_id": "seed_2",
                "readable_expression": "c2*log(Abs(x1)) + x1**2",
                "dataset_fit_results": [{"dataset_name": "dataset_1", "r2": 0.85}],
            },
        ],
    }

    _front, metadata = run_lightweight_gp(
        payload,
        {"top_k": 1, "initial_r2_threshold_mode": "max_r2_gap",
         "initial_r2_max_gap": 0.10, "population_size": 2,
         "generations": 0, "seed": 1},
        feature_count=1, dataset_groups=[{"name": "dataset_1"}],
        fit_formula=lambda record: {"r2": 0.0, "dataset_fit_results": []},
    )

    assert metadata["source_formula_count"] == 2
    assert metadata["initial_population"] == 2


def test_gp_reuse_preserves_source_fit_metadata():
    payload = {
        "top_ranked_refit_results": [{
            "formula_id": "initial_piecewise",
                "readable_expression": "c1*log(Abs(x1)) + c2*x1**-1",
            "piecewise": True,
            "split_index": 12,
            "split_value": 7.5,
            "segment_sample_counts": [12, 8],
            "parameters_by_dataset": {
                "segment_1": [1.0, 2.0], "segment_2": [3.0, 4.0]
            },
            "dataset_fit_results": [{
                "dataset_name": "segment_1", "r2": 0.9,
                "parameters": [1.0, 2.0],
                "parameter_names": ["c1", "c2"],
            }, {
                "dataset_name": "segment_2", "r2": 0.8,
                "parameters": [3.0, 4.0],
                "parameter_names": ["c1", "c2"],
            }],
        }],
    }

    def fit_formula(record):
        return {"r2": 0.1, "dataset_fit_results": []}

    front, _metadata = run_lightweight_gp(
        payload,
        {"top_k": 1, "initial_r2_threshold": 0.0,
         "population_size": 1, "generations": 0, "seed": 1},
        feature_count=1, dataset_groups=[{"name": "realdata"}],
        fit_formula=fit_formula,
    )

    reused = next(row for row in front
                  if row["formula_id"] == "initial_piecewise")
    assert reused["piecewise"] is True
    assert reused["split_index"] == 12
    assert reused["split_value"] == 7.5
    assert reused["parameters_by_dataset"]["segment_2"] == [3.0, 4.0]


def test_adaptive_initial_r2_threshold_uses_two_means():
    threshold, details = _adaptive_initial_r2_threshold(
        [-0.4, -0.2, 0.82, 0.9], fallback=0.4
    )
    assert details["method"] == "two_means"
    assert threshold == pytest.approx(0.28)
    assert details["cluster_counts"] == [2, 2]


def test_adaptive_initial_r2_threshold_falls_back_for_constant_values():
    threshold, details = _adaptive_initial_r2_threshold([0.4, 0.4], fallback=0.35)
    assert threshold == pytest.approx(0.35)
    assert details["method"] == "fixed_fallback"


def test_breakpoint_prior_calculates_and_fixes_realdata_split():
    prior = _breakpoint_prior({"breakpoint_prior": {
        "enabled": True, "h_t_m": 25, "h_r_m": 4,
        "carrier_frequency_hz": 5.8e9,
        "meters_to_data_units": 0.001,
    }})
    assert prior["breakpoint_m"] == pytest.approx(7738.687008597128)
    assert prior["breakpoint_value"] == pytest.approx(7.738687008597128)
    X = np.arange(1.0, 21.0).reshape(-1, 1)
    indices, _ = _restrict_split_indices(
        X, [5, 10, 15], {"min_segment_samples": 5, "breakpoint_prior": {
            "enabled": True, "h_t_m": 25, "h_r_m": 4,
            "carrier_frequency_hz": 5.8e9,
            "meters_to_data_units": 0.001,
            }},
    )
    assert indices == [7]

    # The fixed breakpoint is selected even when the caller's coarse list
    # does not contain that boundary.
    indices, _ = _restrict_split_indices(
        X, [5, 10, 15], {"min_segment_samples": 5, "breakpoint_prior": {
            "enabled": True, "h_t_m": 25, "h_r_m": 4,
            "carrier_frequency_hz": 5.8e9,
            "meters_to_data_units": 0.001,
        }},
    )
    assert indices == [7]


def test_lm_keeps_every_initialization_and_best_result():
    x1 = np.linspace(1.0, 4.0, 30)
    x2 = np.linspace(2.0, 3.0, 30)
    X = np.column_stack((x1, x2))
    y = 2.5 * x1 / x2
    record = {
        "id": "candidate_test",
        "readable_expression": "c1*x1/x2",
    }
    cfg = {
        "lm_initializations": 4,
        "max_nfev": 100,
        "initialization_range": [-2, 2],
        "minimum_valid_fraction": 1.0,
        "max_abs_value": 1e100,
    }
    attempts, best = fit_with_restarts(record, X, y, cfg, np.random.default_rng(9))

    assert len(attempts) == 4
    assert all(row["initial_parameters"] for row in attempts)
    assert best["converged"]
    assert best["r2"] > 0.999999
    assert best["initialization_index"] in range(4)


def test_best_finite_result_is_kept_when_lm_hits_evaluation_limit():
    x1 = np.linspace(1.0, 3.0, 20)
    x2 = np.linspace(2.0, 4.0, 20)
    X = np.column_stack((x1, x2))
    record = {"id": "limited", "readable_expression": "sin(c1*x1/x2)"}
    cfg = {
        "lm_initializations": 2,
        "max_nfev": 1,
        "initialization_range": [-2, 2],
        "minimum_valid_fraction": 1.0,
        "max_abs_value": 1e100,
    }
    _, best = fit_with_restarts(
        record, X, np.sin(1.5 * x1 / x2), cfg, np.random.default_rng(3)
    )

    assert best["parameters"] is not None
    assert best["valid_samples"] == len(X)
    assert best["loss"] is not None
    assert not best["converged"]


def test_multiple_csv_files_are_concatenated_without_band_feature(tmp_path):
    header = "distance,height,target\n"
    first = tmp_path / "freq1.csv"
    second = tmp_path / "freq2.csv"
    first.write_text(header + "1,2,3\n2,3,4\n3,4,5\n", encoding="utf-8")
    second.write_text(header + "4,5,8\n5,6,9\n6,7,10\n", encoding="utf-8")

    X, y, names, groups = load_joint_csv_data({
        "features": ["distance", "height"], "target": "target",
        "csv_files": [
            {"name": "f1", "csv_path": str(first)},
            {"name": "f2", "csv_path": str(second)},
        ],
    })

    assert X.shape == (6, 2)
    assert names == ["distance", "height"]
    assert [group["name"] for group in groups] == ["f1", "f2"]
    assert [(group["start"], group["end"]) for group in groups] == [(0, 3), (3, 6)]
    assert not any("freq" in name or "band" in name for name in names)


def test_joint_structure_uses_independent_parameters_and_minimum_dataset_r2():
    x1 = np.tile(np.linspace(1, 3, 20), 2)
    x2 = np.ones_like(x1)
    X = np.column_stack((x1, x2))
    y = np.concatenate((2 * x1[:20], 4 * x1[20:]))
    groups = [
        {"name": "f1", "start": 0, "end": 20},
        {"name": "f2", "start": 20, "end": 40},
    ]
    record = {"id": "shared", "readable_expression": "c1*x1/x2"}
    cfg = {
        "lm_initializations": 3, "max_nfev": 100,
        "initialization_range": [-2, 2], "minimum_valid_fraction": 1.0,
        "max_abs_value": 1e100,
        "joint_residual_weighting": "equal_dataset",
    }
    _, best = fit_formula_per_dataset(
        record, X, y, cfg, np.random.default_rng(8), groups
    )

    assert best["converged"]
    assert not best["constants_shared_across_datasets"]
    assert set(best["parameters_by_dataset"]) == {"f1", "f2"}
    assert not np.isclose(best["parameters_by_dataset"]["f1"][0],
                          best["parameters_by_dataset"]["f2"][0])
    assert len(best["dataset_metrics"]) == 2
    assert best["r2"] == min(row["r2"] for row in best["dataset_metrics"])
    assert best["joint_r2_minimum"] == best["r2"]
    assert best["joint_r2_mean"] == pytest.approx(
        np.mean([row["r2"] for row in best["dataset_metrics"]])
    )
    assert best["joint_r2_pooled"] == best["pooled_r2"]


def test_top_ranked_formula_is_refitted_on_every_dataset():
    x1 = np.tile(np.linspace(1, 3, 12), 2)
    X = np.column_stack((x1, np.ones_like(x1)))
    y = np.concatenate((2 * x1[:12], 5 * x1[12:]))
    groups = [{"name": "f1", "start": 0, "end": 12},
              {"name": "f2", "start": 12, "end": 24}]
    candidate = {
        "id": "candidate_000000", "readable_expression": "c1*x1/x2",
        "sympy_expression": "c1*x1/x2", "structure_type": "single_candidate",
    }
    ranking = [{
        "predicted_rank": 1, "ranking_score": 0.8, "formula_family": "single",
        "formula_id": candidate["id"], "readable_expression": candidate["readable_expression"],
        "sympy_expression": candidate["sympy_expression"],
    }]
    fitting = {"lm_initializations": 2, "max_nfev": 100,
               "initialization_range": [-1, 1], "minimum_valid_fraction": 1.0,
               "max_abs_value": 1e100}

    results = refit_top_ranked_formulas(
        ranking, [candidate], [], X, y, fitting, {"refit_top_k": 1},
        np.random.default_rng(2), groups,
    )

    assert len(results) == 1
    assert len(results[0]["dataset_fit_results"]) == 2
    assert results[0]["dataset_fit_results"][0]["r2"] > 0.999999
    assert results[0]["dataset_fit_results"][1]["r2"] > 0.999999
    assert results[0]["parameters_by_dataset"]["f1"] != results[0]["parameters_by_dataset"]["f2"]


def test_ablation_top_ranked_formula_reuses_existing_lm_fit():
    ranking = [{
        "predicted_rank": 1, "ranking_score": 0.9, "formula_family": "single",
        "formula_id": "candidate_000000", "readable_expression": "c1*x1",
        "sympy_expression": "c1*x1",
    }]
    existing = [{
        "formula_id": "candidate_000000",
        "readable_expression": "c1*x1",
        "parameters": [2.0],
        "parameter_names": ["c1"],
        "dataset_fit_results": [{"dataset_name": "dataset_1", "r2": 1.0}],
    }]

    results = reuse_top_ranked_formula_fits(
        ranking, existing, {"refit_top_k": 1}
    )

    assert results[0]["parameters"] == [2.0]
    assert results[0]["dataset_fit_results"][0]["r2"] == 1.0
    assert results[0]["fit_reused"] is True


def test_independent_random_sampling_can_overlap_between_datasets():
    candidates = [{"id": f"c{i}"} for i in range(8)]
    allocations = choose_fit_samples_by_dataset(
        candidates, [],
        {"single_sample_count": 4, "combination_sample_count": 0},
        random.Random(12), [{"name": "f1"}, {"name": "f2"}],
    )
    first = {row["id"] for row in allocations["f1"][0]}
    second = {row["id"] for row in allocations["f2"][0]}

    assert first != second
    assert first & second  # overlap is allowed, not forced or removed


def test_single_and_combination_sampling_are_independent():
    candidates = [{"id": f"c{i}"} for i in range(20)]
    combinations = [
        {"id": f"z{i}", "candidate_ids": [f"c{i}", f"c{i + 1}"]}
        for i in range(19)
    ]
    selected, sampled = choose_fit_samples_by_dataset(
        candidates, combinations,
        {"single_sample_count": 3, "combination_sample_count": 10},
        random.Random(12), [{"name": "f1"}],
    )["f1"]

    assert len(selected) == 3
    assert len(sampled) == 10


def test_all_candidate_log_wrappers_use_natural_log():
    x = sp.Symbol("x")
    for wrapper in ("log_abs", "log_abs_sin", "log_abs_cos"):
        expr = apply_wrapper(x, wrapper)
        function = sp.lambdify(x, expr, "numpy")
        argument = np.e if wrapper == "log_abs" else (
            np.arcsin(0.1) if wrapper == "log_abs_sin" else np.arccos(0.1)
        )
        expected = 1.0 if wrapper == "log_abs" else np.log(0.1)
        assert np.isclose(float(function(argument)), expected)


def test_realdata_mat_loader_sorts_distance_and_uses_configured_keys(tmp_path):
    path = tmp_path / "measurements.mat"
    savemat(path, {"distance_meas": [[30.0], [10.0], [20.0]],
                  "pathloss_meas": [[90.0], [50.0], [70.0]]})

    X, y, names, groups = load_realdata_mat({"realdata": {"mat_path": str(path)}})

    assert names == ["distance"]
    assert X[:, 0].tolist() == [10.0, 20.0, 30.0]
    assert y.tolist() == [50.0, 70.0, 90.0]
    assert groups[0]["sorted_by_feature"]


def test_piecewise_fit_enumerates_split_and_selects_lowest_total_loss():
    x = np.arange(1.0, 21.0)
    X = x.reshape(-1, 1)
    y = np.where(x <= 10, 2.0 * x, 5.0 * x)
    record = {"id": "piecewise", "readable_expression": "c1*x1"}
    fitting = {"lm_initializations": 2, "max_nfev": 100,
               "initialization_range": [-1, 1], "minimum_valid_fraction": 1.0,
               "max_abs_value": 1e100, "_joint_r2_primary": "pooled"}

    attempts, best = fit_piecewise_formula(
        record, X, y, fitting, np.random.default_rng(11),
        {"enabled": True, "min_segment_samples": 5, "candidate_stride": 1},
    )

    assert attempts
    assert best["piecewise"]
    assert best["split_index"] == 10
    assert best["split_value"] == 10.5
    assert best["segment_sample_counts"] == [10, 10]
    assert best["loss"] < 1e-10
    assert best["parameters_by_dataset"]["segment_1"] == pytest.approx([2.0])
    assert best["parameters_by_dataset"]["segment_2"] == pytest.approx([5.0])


def test_coarse_piecewise_fit_uses_a_small_evenly_spaced_split_set():
    x = np.arange(1.0, 21.0)
    X = x.reshape(-1, 1)
    y = np.where(x <= 10, 2.0 * x, 5.0 * x)
    record = {"id": "piecewise", "readable_expression": "c1*x1"}
    fitting = {"lm_initializations": 1, "max_nfev": 100,
               "initialization_range": [-1, 1], "minimum_valid_fraction": 1.0,
               "max_abs_value": 1e100}

    _attempts, best = fit_coarse_piecewise_formula(
        record, X, y, fitting, np.random.default_rng(12),
        {"min_segment_samples": 5, "coarse_split_count": 3},
    )

    assert best["segmentation_search_stage"] == "coarse"
    assert best["evaluated_split_count"] == 3
    assert best["split_index"] == 10


def test_piecewise_fit_returns_candidate_failure_when_no_split_is_usable():
    x = np.arange(1.0, 11.0)
    X = x.reshape(-1, 1)
    y = np.ones_like(x)
    record = {"id": "invalid_piecewise", "readable_expression": "log(c1*x1)"}
    fitting = {"lm_initializations": 1, "max_nfev": 1,
               "initialization_range": [-1, 1], "minimum_valid_fraction": 1.0,
               "max_abs_value": 1e100}

    _attempts, best = fit_coarse_piecewise_formula(
        record, X, y, fitting, np.random.default_rng(12),
        {"min_segment_samples": 6, "coarse_split_count": 3},
    )

    assert best["converged"] is False
    assert best["r2"] is None
    assert best["evaluated_split_count"] == 0


def test_pipeline_trains_realdata_mat_with_piecewise_results(tmp_path):
    mat_path = tmp_path / "measurements.mat"
    x = np.arange(1.0, 21.0)
    savemat(mat_path, {"distance_meas": x.reshape(-1, 1),
                      "pathloss_meas": np.where(x <= 10, 2 * x, 5 * x).reshape(-1, 1)})
    config = {
        "stage": "all", "seed": 5,
        "data": {"mode": "realdata_mat", "realdata": {
                "mat_path": str(mat_path),
                "segmentation": {"enabled": True, "min_segment_samples": 5,
                                 "strategy": "legacy", "candidate_stride": 1,
                                 "formula_pair_search": True,
                             "segment_formula_top_k": 2, "top_pair_results": 2},
        }},
        "bases": {"arities": [1], "max_abs_exponent": 1},
        "candidates": {"max_count": 2, "two_base_ratio": 0,
                        "wrappers": ["identity"], "max_sampling_attempts": 50},
        "combinations": {"max_count": 1, "types": ["add"], "max_sampling_attempts": 50},
        "fitting": {"fit_mode": "refit", "single_sample_count": 2,
                    "combination_sample_count": 1, "lm_initializations": 1,
                    "max_nfev": 100, "initialization_range": [-1, 1],
                    "minimum_valid_fraction": 1.0, "max_abs_value": 1e100},
        "output": {"directory": str(tmp_path / "output")},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    payload = run(config_path)

    assert payload["metadata"]["data_mode"] == "realdata_mat"
    assert payload["metadata"]["piecewise_segmentation"]["min_segment_samples"] == 5
    assert payload["single_best"]
    assert all(row["piecewise"] for row in payload["single_best"])
    assert {row["split_index"] for row in payload["single_best"]} == {10}
    assert payload["piecewise_formula_pairs"]
    assert "left_formula" in payload["piecewise_formula_pairs"][0]
    assert "right_formula" in payload["piecewise_formula_pairs"][0]


def test_mat_coarse_gp_fine_strategy_ranks_global_formulas_then_coarse_refits(tmp_path):
    mat_path = tmp_path / "measurements.mat"
    x = np.arange(1.0, 25.0)
    savemat(mat_path, {"distance_meas": x.reshape(-1, 1),
                      "pathloss_meas": np.where(x <= 12, 2 * x, 5 * x).reshape(-1, 1)})
    config = {
        "stage": "all", "seed": 6,
        "data": {"mode": "realdata_mat", "realdata": {
            "mat_path": str(mat_path),
            "segmentation": {"enabled": True, "strategy": "coarse_gp_fine",
                             "min_segment_samples": 5, "coarse_split_count": 3,
                             "fine_candidate_stride": 1},
        }},
        "bases": {"arities": [1], "max_abs_exponent": 1},
        "candidates": {"max_count": 2, "two_base_ratio": 0,
                        "wrappers": ["identity"], "max_sampling_attempts": 50},
        "combinations": {"max_count": 1, "types": ["add"], "max_sampling_attempts": 50},
        "fitting": {"fit_mode": "refit", "single_sample_count": 2,
                    "combination_sample_count": 1, "lm_initializations": 1,
                    "max_nfev": 100, "initialization_range": [-1, 1],
                    "minimum_valid_fraction": 1.0, "max_abs_value": 1e100},
        "ranking": {"enabled": True, "model_type": "extra_trees",
                    "minimum_training_samples": 2, "validation_fraction": 0.0,
                    "refit_top_k": 2, "n_estimators": 10, "min_samples_leaf": 1,
                    "max_nodes": 20, "engineered_max_depth": 6},
        "gp": {"enabled": False},
        "output": {"directory": str(tmp_path / "output")},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    payload = run(config_path)

    assert all("piecewise" not in row for row in payload["single_best"])
    assert payload["top_ranked_refit_results"]
    assert all(row["segmentation_search_stage"] == "coarse"
               for row in payload["top_ranked_refit_results"])
    assert payload["metadata"]["segmentation_strategy"] == "coarse_gp_fine"


def test_piecewise_formula_pair_search_allows_different_segment_structures():
    x = np.arange(1.0, 21.0)
    X = x.reshape(-1, 1)
    y = np.concatenate((2.0 * x[:10], 100.0 / x[10:]))
    records = [
        {"id": "line", "readable_expression": "c1*x1", "structure_type": "single_candidate"},
        {"id": "inverse", "readable_expression": "c1/x1", "structure_type": "single_candidate"},
    ]
    fitting = {"lm_initializations": 2, "max_nfev": 100,
               "initialization_range": [-1, 1], "minimum_valid_fraction": 1.0,
               "max_abs_value": 1e100}

    pairs, per_formula = search_piecewise_formula_pairs(
        records, X, y, fitting, np.random.default_rng(17),
        {"enabled": True, "min_segment_samples": 5, "candidate_stride": 1,
         "segment_formula_top_k": 2, "top_pair_results": 3},
    )

    assert pairs
    assert pairs[0]["split_index"] == 10
    assert pairs[0]["left_formula"]["formula_id"] == "line"
    assert pairs[0]["right_formula"]["formula_id"] == "inverse"
    assert pairs[0]["loss"] < 1e-10
    assert set(per_formula) == {"line", "inverse"}


def test_top_ranked_refit_uses_formula_pair_search_for_realdata():
    x = np.arange(1.0, 21.0)
    X = x.reshape(-1, 1)
    y = np.concatenate((2.0 * x[:10], 100.0 / x[10:]))
    candidates = [
        {"id": "line", "readable_expression": "c1*x1", "sympy_expression": "c1*x1",
         "structure_type": "single_candidate"},
        {"id": "inverse", "readable_expression": "c1/x1", "sympy_expression": "c1/x1",
         "structure_type": "single_candidate"},
    ]
    ranking = [
        {"formula_id": "line", "ranking_score": 0.9, "predicted_rank": 1,
         "formula_family": "single", "readable_expression": "c1*x1", "sympy_expression": "c1*x1"},
        {"formula_id": "inverse", "ranking_score": 0.8, "predicted_rank": 2,
         "formula_family": "single", "readable_expression": "c1/x1", "sympy_expression": "c1/x1"},
    ]
    fitting = {"lm_initializations": 2, "max_nfev": 100,
               "initialization_range": [-1, 1], "minimum_valid_fraction": 1.0,
               "max_abs_value": 1e100}

    results = refit_top_ranked_formula_pairs(
        ranking, candidates, [], X, y, fitting, {"refit_top_k": 2},
        np.random.default_rng(22),
        {"enabled": True, "min_segment_samples": 5, "candidate_stride": 1,
         "segment_formula_top_k": 2, "top_pair_results": 2},
    )

    assert results[0]["left_formula"]["formula_id"] == "line"
    assert results[0]["right_formula"]["formula_id"] == "inverse"
    assert results[0]["formula_family"] == "piecewise_pair"


def test_gp_residual_autocorrelation_is_attached_per_csv_dataset():
    x = np.arange(1.0, 7.0)
    X = x.reshape(-1, 1)
    residual = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    y = 2.0 * x + residual
    gp_results = [{
        "readable_expression": "c1*x1",
        "dataset_fit_results": [{"dataset_name": "csv_1", "parameters": [2.0]}],
    }]

    add_gp_residual_autocorrelation(
        gp_results, X, y, [{"name": "csv_1", "start": 0, "end": len(y)}],
        {"max_abs_value": 1e100, "complex_tolerance": 1e-12},
    )

    diagnostics = gp_results[0]["dataset_fit_results"][0]["residual_autocorrelation"]
    assert diagnostics["samples"] == len(y)
    assert diagnostics["durbin_watson"] == pytest.approx(20 / 6)
    assert diagnostics["lag_1"] == pytest.approx(-1.0)


def test_lightweight_gp_operations_keep_formulas_legal():
    expr = sp.sympify("c1*x1/x2 + c2*log(Abs(x1))")
    cfg = {"max_complexity": 96, "max_depth": 20}
    for seed in range(8):
        child = mutate(expr, random.Random(seed), cfg, 2)
        assert valid(child, max_depth=20, max_nodes=96)
        crossed = crossover(expr, sp.sympify("c3*sin(x1*x2)"),
                            random.Random(seed), cfg, 2)
        assert valid(crossed, max_depth=20, max_nodes=96)


def test_gp_operator_mutation_swaps_sin_cos_and_log_abs():
    expr = sp.sympify("sin(x1) + log(Abs(x2))")
    seen = set()
    for seed in range(40):
        mutated = _replace_operator(expr, random.Random(seed))
        seen.add(mutated)
    assert sp.sympify("cos(x1) + log(Abs(x2))") in seen
    assert sp.sympify("sin(x1) + sin(x2)") in seen
    assert sp.sympify("log(Abs(x1)) + log(Abs(x2))") in seen


def test_gp_operator_mutation_keeps_binary_operator_mutation_separate():
    expr = sp.sympify("sin(x1) + x2")
    seen = set()
    for seed in range(20):
        mutated = _replace_operator(expr, random.Random(seed))
        seen.add(mutated)
    assert any(value == sp.sympify("x2 - sin(x1)") for value in seen)
    assert any(value == sp.sympify("sin(x1)*x2") for value in seen)
    assert any(value == sp.sympify("x2/sin(x1)") for value in seen)


def test_gp_operator_mutation_can_limit_unary_mutation_to_sin_cos_pair():
    sin_expr = sp.sympify("sin(x1)")
    cos_expr = sp.sympify("cos(x1)")
    log_expr = sp.sympify("log(Abs(x1))")
    sin_seen = {
        _replace_operator(
            sin_expr, random.Random(seed), pairwise_trigonometric_only=True
        )
        for seed in range(20)
    }
    cos_seen = {
        _replace_operator(
            cos_expr, random.Random(seed), pairwise_trigonometric_only=True
        )
        for seed in range(20)
    }
    assert sin_seen == {cos_expr}
    assert cos_seen == {sin_expr}
    assert {
        _replace_operator(
            log_expr, random.Random(seed), pairwise_trigonometric_only=True
        )
        for seed in range(20)
    } == {log_expr}


def test_gp_variable_mutation_flips_a_flattened_quotient():
    expr = sp.sympify("c1*x1/x2 + x3")
    mutated = _replace_variable(expr, random.Random(0), 3)
    assert mutated == sp.sympify("c1*x2/x1 + x3")


def test_gp_variable_mutation_flips_a_ratio_inside_nested_wrappers():
    expr = sp.sympify("log(Abs(sin(c2_right_gp1*x1/x2)))")

    mutated = _replace_variable(expr, random.Random(0), 2)

    assert mutated == sp.sympify("log(Abs(sin(c2_right_gp1*x2/x1)))")


def test_gp_variable_to_constant_uses_a_fresh_constant_symbol():
    expr = sp.sympify("c0*x1 + c1*x2")
    mutated = _replace_variable_with_constant(expr, random.Random(0))

    assert mutated == sp.sympify("c0*x1 + c3")
    assert sp.Symbol("c2") not in expr.free_symbols
    assert sp.Symbol("c3") not in expr.free_symbols


def test_gp_merges_repeated_constant_symbols_in_arithmetic_nodes():
    c1 = sp.Symbol("c1")
    raw = sp.Add(sp.Symbol("x1"),
                 sp.Add(c1, c1, evaluate=False), evaluate=False)

    assert _merge_duplicate_constants(raw) == sp.sympify("x1 + c0")


def test_gp_constant_merging_collapses_all_arithmetic_forms():
    for raw in (
        sp.Add(sp.Symbol("c1"), sp.Symbol("c1"), evaluate=False),
        sp.Mul(sp.Symbol("c1"), sp.Symbol("c1"), evaluate=False),
        sp.Pow(sp.Symbol("c1"), sp.Symbol("c1"), evaluate=False),
    ):
        merged = _merge_duplicate_constants(raw)
        assert merged.is_Symbol
        assert str(merged).startswith("c")


def test_realdata_gp_log_abs_x1_constraint_is_checked_at_selection_not_mutation():
    plain = sp.sympify("c1*x1")
    compliant = sp.sympify("log(Abs(c1*x1))")
    cfg = {"max_complexity": 30, "require_log_abs_x1": True}

    assert not _has_log_abs_x1(plain)
    assert _has_log_abs_x1(compliant)
    # Normal GP evolution can retain a plain parent; Pareto/elite filtering
    # performs the hard constraint at the end of a generation.
    assert _valid_child(plain, cfg, 1)


def test_pysr_reproducible_config_contract(tmp_path):
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    loaded = load_config(DEFAULT_CONFIG)
    assert loaded["postprocessing"] == {
        "merge_log_max_r2_drop": 0.01,
        "log_merge_initial_perturbation": 0.05,
    }
    assert "gp" not in loaded
    assert "pysr_baseline" not in loaded

    pysr_path = DEFAULT_CONFIG.parent / "pysr_baseline.json"
    pysr_config = json.loads(pysr_path.read_text(encoding="utf-8"))
    config["pysr_baseline"] = pysr_config["pysr_baseline"]
    combined_path = tmp_path / "pipeline_with_pysr.json"
    combined_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    loaded_with_pysr = load_config(combined_path)
    pysr = loaded_with_pysr["pysr_baseline"]
    assert pysr["reproducible"] is True
    assert pysr["deterministic"] is True
    assert pysr["parallelism"] == "serial"
    assert pysr["timeout_in_seconds"] is None

    config["pysr_baseline"]["reproducibility_mode"] = "invalid"
    invalid_path = tmp_path / "invalid_pysr_config.json"
    invalid_path.write_text(
        json.dumps(config, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="reproducibility_mode"):
        load_config(invalid_path)


def test_legacy_gp_log_merge_settings_migrate_to_postprocessing(tmp_path):
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config.pop("postprocessing")
    config["gp"] = {
        "merge_log_max_r2_drop": 0.025,
        "log_merge_initial_perturbation": 0.08,
    }
    legacy_path = tmp_path / "legacy_postprocessing_config.json"
    legacy_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_config(legacy_path)

    assert loaded["postprocessing"] == {
        "merge_log_max_r2_drop": 0.025,
        "log_merge_initial_perturbation": 0.08,
    }
    assert "gp" not in loaded



def test_array_data_sha256_is_stable_and_value_sensitive():
    first = _array_data_sha256(np.arange(6).reshape(2, 3), np.array([1.0]))
    second = _array_data_sha256(np.arange(6).reshape(2, 3), np.array([1.0]))
    changed = _array_data_sha256(np.arange(6).reshape(2, 3), np.array([2.0]))
    assert first == second
    assert first != changed
