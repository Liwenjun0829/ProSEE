import math

import numpy as np
import pytest

from structure_r2_ranker import (
    ExperimentalPrimeTreeCoordinateEncoder,
    FormulaResponseSVDEncoder,
    TreeCoordinateEncoder,
    extract_svd_features,
    generate_formula_value_matrix,
    load_structure_ranker_scorer,
    normalize_value_matrix,
    preorder_node_sequence,
    train_and_rank,
)


def test_value_matrix_svd_handles_parameters_and_invalid_values():
    matrix = generate_formula_value_matrix("c1*log(x1 - x2)/x2", grid_size=8)
    normalized = normalize_value_matrix(matrix)
    features = extract_svd_features(normalized, feature_count=5)

    assert matrix.shape == (8, 8)
    assert normalized.shape == (8, 8)
    assert np.all(np.isfinite(normalized))
    assert features.shape == (5,)
    assert np.all(np.isfinite(features))

    encoder = FormulaResponseSVDEncoder(grid_size=8, feature_count=5)
    assert encoder.transform(["c1*x1 + c2*x2"]).shape == (1, 5)


def test_experimental_prime_operator_combinations_are_distinct():
    formulas = ["sin(cos(x1))", "sin(log(x1))"]
    encoder = ExperimentalPrimeTreeCoordinateEncoder(
        max_nodes=8, ancestor_products=True,
    ).fit(formulas)
    field_width = len(encoder._ancestor_field_names)
    prime_product_index = field_width + encoder._PRIME_FIELD_INDEX
    first, second = encoder.transform(formulas).reshape(2, 8, -1)

    # The second node is cos/log and its first ancestor is the sin root.
    first_value = first[1, prime_product_index]
    second_value = second[1, prime_product_index]
    assert first_value != second_value
    assert first_value == pytest.approx(math.log(
        encoder.symbol_primes["OP:sin/1"] * encoder.symbol_primes["OP:cos/1"]
    ))
    assert second_value == pytest.approx(math.log(
        encoder.symbol_primes["OP:sin/1"] * encoder.symbol_primes["OP:log/1"]
    ))


def test_experimental_prime_encoding_can_use_nearest_ancestor_only():
    encoder = ExperimentalPrimeTreeCoordinateEncoder(
        max_nodes=8, ancestor_products=True, nearest_ancestor_only=True,
    ).fit(["sin(cos(x1))"])
    features = encoder.transform(["sin(cos(x1))"])

    assert encoder.nearest_ancestor_only
    assert encoder.feature_width == len(encoder._ancestor_field_names) * 2
    assert features.shape == (1, len(encoder.feature_names))


def test_experimental_prime_encoding_is_finite_for_deep_tree():
    formula = "sin(" * 120 + "x1" + ")" * 120
    encoder = ExperimentalPrimeTreeCoordinateEncoder(
        max_nodes=128, ancestor_products=True,
    ).fit([formula])

    features = encoder.transform([formula])

    assert encoder.max_ancestor_count == 120
    assert np.all(np.isfinite(features))


def test_preorder_encoding_normalizes_fit_constants_but_keeps_structure():
    sequence = preorder_node_sequence("a1*sin(c1*x1/x2**2)", max_nodes=20)

    assert sequence[0].startswith("OP:")
    assert sequence.count("FIT_CONST") == 2
    assert "VAR:x1" in sequence
    assert "VAR:x2" in sequence
    assert "INT:-2" in sequence
    assert len(sequence) == 20


def test_extra_trees_ranks_all_formulas_from_real_r2_labels(tmp_path):
    candidates = []
    fitted = []
    for index in range(12):
        formula = f"c1*x1**{index + 1}/x2"
        formula_id = f"candidate_{index:06d}"
        candidates.append({
            "id": formula_id,
            "readable_expression": formula,
            "sympy_expression": formula,
            "structure_type": "single_candidate",
            "operator": "identity",
            "base_ids": [f"base_{index}"],
        })
        if index < 8:
            fitted.append({"formula_id": formula_id, "r2": index / 10})
    payload = {
        "single_candidates": candidates,
        "binary_combinations": [],
        "single_best": fitted,
        "combination_best": [],
    }
    # The same structure fitted on another dataset remains a separate
    # supervision row, while grouped validation keeps it in one split.
    payload["single_best"].append({"formula_id": "candidate_000000", "r2": 0.35})
    config = {
        "model_type": "extra_trees",
        "max_nodes": 24,
        "minimum_training_samples": 6,
        "validation_fraction": 0.25,
        "n_estimators": 20,
        "min_samples_leaf": 1,
        "n_jobs": 1,
    }

    ranking, metadata = train_and_rank(payload, config, 11, tmp_path)

    assert len(ranking) == len(candidates)
    assert [row["predicted_rank"] for row in ranking] == list(range(1, 13))
    assert all(np.isfinite(row["ranking_score"]) for row in ranking)
    assert metadata["training_samples"] == 9
    assert metadata["unique_labeled_formulas"] == 8
    assert metadata["raw_feature_dimension"] > config["max_nodes"]
    assert (tmp_path / "structure_r2_ranker.joblib").exists()
    assert (tmp_path / "structure_r2_encoder.json").exists()
    assert (tmp_path / "structure_r2_features.json").exists()
    assert (tmp_path / "structure_r2_feature_importance.csv").exists()


def test_value_matrix_svd_mode_ranks_and_persists_scorer(tmp_path):
    candidates, fitted = [], []
    for index in range(6):
        formula_id = f"candidate_{index:06d}"
        candidates.append({
            "id": formula_id, "readable_expression": f"c1*x1**{index + 1} + c2*x2",
            "sympy_expression": f"c1*x1**{index + 1} + c2*x2",
            "structure_type": "single_candidate", "operator": "identity",
            "base_ids": [f"base_{index}"],
        })
        fitted.append({"formula_id": formula_id, "r2": index / 10})
    payload = {"single_candidates": candidates, "binary_combinations": [],
               "single_best": fitted, "combination_best": []}
    config = {
        "model_type": "extra_trees", "feature_mode": "value_matrix_svd",
        "response_svd_grid_size": 8, "response_svd_feature_count": 4,
        "minimum_training_samples": 6, "validation_fraction": 0.0,
        "n_estimators": 10, "n_jobs": 1,
    }

    ranking, metadata = train_and_rank(payload, config, 3, tmp_path)

    assert len(ranking) == len(candidates)
    assert metadata["feature_encoding"] == "normalized_value_matrix_svd"
    assert metadata["raw_feature_dimension"] == 4
    scorer = load_structure_ranker_scorer(tmp_path / "structure_r2_ranker.joblib")
    scores = scorer(["c9*log(x1 - x2)/x2", candidates[0]["readable_expression"]])
    assert np.all(np.isfinite(scores))


@pytest.mark.parametrize("tree_pca", [False, True])
def test_operator_matrix_uses_svd_and_reloads_consistently(tmp_path, monkeypatch, tree_pca):
    import json
    import structure_r2_ranker as ranker

    monkeypatch.setattr(ranker, "__file__", str(tmp_path / "ranker.py"))
    candidates = [
        {"id": f"candidate_{i}", "readable_expression": f"c1*x1**{i + 1}/x2",
         "sympy_expression": f"c1*x1**{i + 1}/x2",
         "structure_type": "single_candidate", "operator": "identity",
         "base_ids": [f"base_{i}"]}
        for i in range(6)
    ]
    payload = {
        "bases": [{"id": f"base_{i}", "readable_expression": f"x1**{i + 1}/x2"}
                  for i in range(6)],
        "single_candidates": candidates, "binary_combinations": [],
        "single_best": [{"formula_id": row["id"], "r2": i / 10}
                        for i, row in enumerate(candidates)],
        "combination_best": [],
    }
    config = {
        "model_type": "extra_trees", "feature_mode": "operator_primitive_matrix",
        "operator_primitive_tree_pca_enabled": tree_pca,
        "response_svd_grid_size": 8, "response_svd_feature_count": 4,
        "kernel_pca_components": 2, "max_nodes": 16,
        "minimum_training_samples": 6, "validation_fraction": 0.0,
        "n_estimators": 10, "n_jobs": 1,
    }
    ranking, _ = train_and_rank(payload, config, 3, tmp_path)
    features = json.loads((tmp_path / "structure_r2_features.json").read_text())
    assert features["numeric_feature_names"] == FormulaResponseSVDEncoder(8, 4).feature_names
    assert "node_count" in features["model_feature_names"]
    assert features["engineered_numeric_feature_names"]
    assert features["operator_primitive_matrix"]["numeric_feature_dimension"] == 4
    scorer = load_structure_ranker_scorer(tmp_path / "structure_r2_ranker.joblib")
    np.testing.assert_allclose(
        scorer([row["readable_expression"] for row in ranking]),
        [row["ranking_score"] for row in ranking],
    )
    # A second training pass exercises the serialized feature cache too.
    cached_ranking, _ = train_and_rank(payload, config, 3, tmp_path)
    np.testing.assert_allclose(
        [row["ranking_score"] for row in cached_ranking],
        [row["ranking_score"] for row in ranking],
    )


def test_layered_tree_coordinates_limit_width_and_discard_overflow(tmp_path):
    candidates, fitted = [], []
    for index in range(6):
        formula_id = f"candidate_{index:06d}"
        formula = "c1*sin(x1) + c2*x2"
        candidates.append({
            "id": formula_id, "readable_expression": formula,
            "sympy_expression": formula, "structure_type": "single_candidate",
            "operator": "sin", "base_ids": [f"base_{index}"],
        })
        fitted.append({"formula_id": formula_id, "r2": index / 10})
    overflow_formula = "x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9"
    candidates.append({
        "id": "overflow", "readable_expression": overflow_formula,
        "sympy_expression": overflow_formula, "structure_type": "single_candidate",
        "operator": "identity", "base_ids": ["overflow_base"],
    })
    payload = {
        "single_candidates": candidates, "binary_combinations": [],
        "single_best": fitted, "combination_best": [],
    }
    config = {
        "model_type": "extra_trees", "feature_mode": "tree_coordinates",
        "max_nodes": 96, "tree_max_nodes_per_level": 8,
        "minimum_training_samples": 6, "validation_fraction": 0.0,
    }

    ranking, metadata = train_and_rank(payload, config, 13, tmp_path)

    layered = metadata["layered_tree_coordinates"]
    assert layered["max_nodes_per_level"] == 8
    assert layered["tree_levels"] == 4
    assert layered["nodes_per_level"] == [1, 2, 4, 1]
    assert layered["feature_node_slots"] == 8
    assert layered["discarded_formulas"] == 1
    assert "overflow" not in {row["formula_id"] for row in ranking}

    encoder = TreeCoordinateEncoder(max_nodes_per_level=8).fit(
        ["c1*sin(x1) + c2*x2"]
    )
    assert encoder.transform(["c1*sin(x1) + c2*x2"]).shape == (1, 8 * 8)
    scorer = load_structure_ranker_scorer(tmp_path / "structure_r2_ranker.joblib")
    assert scorer([overflow_formula])[0] == -1e100
