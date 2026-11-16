import random
import numpy as np
import sympy as sp
from pathlib import Path
from structure_screen import tree_search_ops
from structure_screen.encoding import canonical_structure, TreeEncoder
from structure_screen.tree_search_ops import (complexity, delete_subtree_mutation,
                                 eligible_candidate, generate_target_free,
                                 pareto_front, valid)

def test_constants_are_deduplicated():
    assert canonical_structure("2.1*x1 + 3") == canonical_structure("9.7*x1 + 8")


def test_reciprocal_complexity_counts_the_power_subtree():
    assert complexity(sp.sympify("c1*x1")) == 3
    assert complexity(sp.sympify("c1/x1")) == 5
    assert complexity(sp.sympify("c1*x1*x2")) == 4
    assert complexity(sp.sympify("c1*x1/x2")) == 6

def test_encoder_padding_and_truncation():
    e=TreeEncoder(5).fit(["x1+x2"]); assert len(e.encode("x1+x2+x3+x4"))==5

def test_delete_subtree_is_valid_and_usually_smaller():
    original=sp.sympify("(x1+x2)*(x1/(x2+1)+x1*x2)"); reduced=0
    for seed in range(100):
        out=delete_subtree_mutation(original,random.Random(seed),7,63,("x1","x2")); assert valid(out,7,63); assert sp.sympify(str(out)) is not None; reduced += complexity(out)<complexity(original)
    # A scaled variable contributes Mul + Float + Symbol, so some replacements
    # preserve rather than reduce the node count.
    assert reduced >= 40

def test_delete_subtree_variable_terminal_has_a_numeric_multiplier():
    original=sp.sympify("x1+x2")
    for seed in range(100):
        out=delete_subtree_mutation(original,random.Random(seed),7,63,("x1","x2"))
        scaled=[node for node in sp.preorder_traversal(out)
                if node.is_Mul and node.free_symbols
                and any(isinstance(arg,sp.Float) for arg in node.args)]
        if scaled:
            return
    raise AssertionError("no scaled variable terminal was generated")

def test_target_free_gp_prioritizes_log_and_stays_valid():
    cfg={"population_size":40,"generations":2,"max_formula_complexity":15,
         "delete_subtree_probability":.2,"mutation_probability":.8,
         "crossover_probability":.65,"log_probability":.8,"trig_probability":0.0,
         "require_transcendental":True}
    formulas=generate_target_free(cfg,["x1","x2"],42)
    assert formulas and all(valid(f,None,15) for f in formulas)
    assert all(f.has(sp.log) for f in formulas)

def test_transcendental_function_is_a_hard_candidate_requirement():
    cfg={"require_transcendental":True,"max_formula_complexity":15,
         "disabled_operators": []}
    x1 = sp.Symbol("x1")
    assert eligible_candidate(sp.log(x1),cfg)
    assert eligible_candidate(sp.sin(x1),cfg)
    assert eligible_candidate(sp.cos(x1),cfg)
    assert eligible_candidate(sp.exp(x1),cfg)
    assert not eligible_candidate(sp.Symbol("x1")+sp.Symbol("x2"),cfg)
    assert eligible_candidate(
        sp.Symbol("x1") + sp.Symbol("x2"),
        {"require_transcendental":False,"max_formula_complexity":15},
    )

def test_generated_log_has_a_numeric_multiplier():
    import random
    from structure_screen.tree_search_ops import random_tree
    expr=random_tree(random.Random(7),["x1"],max_depth=1,log_probability=1.0)
    log_node=next(iter(expr.find(sp.log)))
    assert expr.is_Mul and any(arg.is_Number and arg != 0 for arg in expr.args)
    assert log_node in expr.args

def test_random_tree_can_generate_sin_and_cos_nodes():
    from structure_screen.tree_search_ops import random_tree
    generated=[]
    for seed in range(50):
        expr=random_tree(random.Random(seed),["x1"],max_depth=2,
                         log_probability=0.0,trig_probability=1.0)
        generated.append(expr)
    assert any(expr.has(sp.sin) for expr in generated)
    assert any(expr.has(sp.cos) for expr in generated)
    assert all(complexity(expr) <= 20 for expr in generated)
    assert all(not expr.has(sp.nan,sp.zoo,sp.oo,-sp.oo) for expr in generated)

def test_variable_terminal_has_a_numeric_multiplier():
    from structure_screen.tree_search_ops import random_tree
    expr=random_tree(random.Random(1),["x1"],max_depth=0)
    assert expr.has(sp.Symbol("x1")) and expr.is_Mul
    assert any(isinstance(arg,sp.Float) and arg != 0 for arg in expr.args)

def test_pareto_maximizes_r2_and_minimizes_complexity():
    rows=[{"true_r2":.8,"complexity":10},{"true_r2":.7,"complexity":12},
          {"true_r2":.75,"complexity":5},{"true_r2":.9,"complexity":20}]
    front=pareto_front(rows)
    assert rows[1] not in front
    assert {(.75,5),(.8,10),(.9,20)} == {(x["true_r2"],x["complexity"]) for x in front}

def test_every_generated_formula_obeys_global_complexity_limit():
    cfg={"population_size":50,"generations":3,"max_formula_complexity":15,
         "delete_subtree_probability":.2,"mutation_probability":1.0,
         "crossover_probability":1.0,"log_probability":.5,"require_transcendental":True}
    formulas=generate_target_free(cfg,["x1","x2"],11)
    assert formulas and max(map(complexity,formulas)) <= 15


def test_iterative_screen_fits_nn_top_two_and_stops_on_joint_target(monkeypatch, tmp_path):
    batches = [
        [sp.sympify("x1"), sp.sympify("x1+x2"), sp.sympify("x1*x2")],
        [sp.sympify("sin(x1)"), sp.sympify("cos(x1)"), sp.sympify("log(x1)")],
        [sp.sympify("x1/x2")],
    ]
    scores = {
        "x1": .1, "x1 + x2": .8, "x1*x2": .9,
        "sin(x1)": .7, "cos(x1)": .6, "log(x1)": .95,
        "x1/x2": .99,
    }
    fitted = []

    def fake_generate(cfg, variables, seed, candidate_callback=None):
        emitted = []
        for generation, batch in enumerate(batches, 1):
            emitted.extend(batch)
            if candidate_callback(batch, generation):
                break
        return emitted

    def fake_fit(formula, X, y, restarts, max_nfev, seed):
        fitted.append(formula)
        r2 = .995 if formula == "log(x1)" else .8
        return {"true_r2": r2, "constants": [],
                "optimized_formula": formula, "prediction": y.copy()}

    monkeypatch.setattr(tree_search_ops, "generate_target_free", fake_generate)
    monkeypatch.setattr(tree_search_ops, "fit_constants", fake_fit)
    cfg = {
        "candidates_per_fit": 3, "fit_top_k_per_batch": 2,
        "target_r2": .99, "target_complexity": 3, "max_generations": 10,
    }
    X = np.arange(12.0).reshape(6, 2) + 1
    y = np.arange(6.0)
    tree_search_ops.run_gp(lambda formulas: [scores[text] for text in formulas], X, y,
              tmp_path / "results.json", cfg, {"restarts": 1, "max_nfev": 10})
    assert fitted == ["x1*x2", "x1 + x2", "log(x1)", "sin(x1)"]
    assert "x1/x2" not in fitted


def test_run_gp_exactly_deduplicates_before_predictor_and_lm(monkeypatch, tmp_path):
    x1, x2 = sp.symbols("x1 x2")
    predictor_batches = []
    fitted = []

    monkeypatch.setattr(
        tree_search_ops, "generate_target_free",
        lambda cfg, variables, seed, candidate_callback=None: [
            x1 + x2, x1 + x2, x1*x2
        ],
    )
    monkeypatch.setattr(
        tree_search_ops, "fit_constants",
        lambda formula, X, y, *args: fitted.append(formula) or {
            "true_r2": .5, "constants": [],
            "optimized_formula": formula, "prediction": y.copy(),
        },
    )

    def predictor(formulas):
        predictor_batches.append(list(formulas))
        return np.linspace(.9, .8, len(formulas))

    cfg = {
        "deduplication_scope": "batch_exact",
        "candidates_per_fit": 10,
        "fit_top_k_per_batch": 2,
        "target_r2": 1.0,
        "target_complexity": 1,
    }
    X = np.ones((6, 2))
    y = np.arange(6.0)
    tree_search_ops.run_gp(
        predictor, X, y, tmp_path / "results.json", cfg,
        {"restarts": 1, "max_nfev": 1}, variables=["x1", "x2"],
    )

    assert predictor_batches == [["x1 + x2", "x1*x2"]]
    assert fitted == ["x1 + x2", "x1*x2"]


def test_generation_never_exceeds_max_generations():
    callback_generations = []
    cfg = {
        "population_size": 20, "max_generations": 3, "candidates_per_fit": 1,
        "max_formula_complexity": 20, "delete_subtree_probability": .2,
        "mutation_probability": .8, "crossover_probability": .65,
        "require_transcendental": False,
    }
    generated = generate_target_free(
        cfg, ["x1", "x2"], 42,
        candidate_callback=lambda batch, generation: callback_generations.append(generation) or False,
    )
    assert generated
    assert callback_generations
    assert max(callback_generations) <= cfg["max_generations"]
