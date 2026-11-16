import numpy as np

from formula_structure_features import FormulaStructureFeatureExtractor, OPERATORS, VARIABLES


def feature_map(extractor, formula):
    categorical, numeric = extractor.extract_one(formula)
    return dict(zip(extractor.categorical_names, categorical)), dict(zip(extractor.numeric_names, numeric))


def test_fixed_feature_order_and_operator_hierarchy_features():
    extractor = FormulaStructureFeatureExtractor(max_nodes=16, max_depth=5)
    categories, values = feature_map(extractor, "log(Abs(sin(c1*x1/x2**2))) + cos(x2)")

    assert extractor.feature_names == extractor.categorical_names + extractor.numeric_names
    assert extractor.categorical_indices == list(range(len(extractor.categorical_names)))
    assert categories["root_operator"] == "add"
    assert categories["parent_child_signature"] != "none"
    assert values["operator_add_count"] == 1
    assert values["operator_divide_count"] == 1
    assert values["operator_power_count"] == 1
    assert values["operator_log_count"] == 1
    assert values["operator_sin_count"] == 1
    assert values["operator_cos_count"] == 1
    assert values["operator_abs_count"] == 1
    assert values["max_consecutive_unary_depth"] >= 3


def test_variable_context_and_batch_values_are_safe():
    extractor = FormulaStructureFeatureExtractor(max_nodes=12, max_depth=3)
    _, values = feature_map(extractor, "log(Abs(x1/x2)) + sin(x3) + x1**x4")
    categorical, numeric = extractor.transform(["x1/x2", "log(Abs(x1))"])

    assert values["x1_inside_log"] == 1
    assert values["x2_inside_denominator"] == 1
    assert values["x3_inside_trig"] == 1
    assert values["x4_inside_power_exponent"] == 1
    assert categorical.shape == (2, len(extractor.categorical_names))
    assert numeric.shape == (2, len(extractor.numeric_names))
    assert np.all(np.isfinite(numeric))
    for operator in OPERATORS:
        assert f"operator_{operator}_count" in extractor.numeric_names
    for variable in VARIABLES:
        assert f"{variable}_count" in extractor.numeric_names
