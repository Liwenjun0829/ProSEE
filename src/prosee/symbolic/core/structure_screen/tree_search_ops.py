"""Target-free GP generation followed by neural screening and LM/Pareto selection."""
import json
import logging
import random
from pathlib import Path

import numpy as np
import sympy as sp

from .encoding import canonical_structure
from .lm import fit_constants


TRANSCENDENTAL_FUNCTIONS = (sp.log, sp.sin, sp.cos, sp.exp)


def has_transcendental_function(expr):
    """Whether a formula contains at least one supported transcendental function."""
    return any(expr.has(function) for function in TRANSCENDENTAL_FUNCTIONS)


def contains_disabled_operator(expr, disabled_operators):
    """Check configurable forbidden operators on a canonical SymPy tree.

    Function names such as exp/sqrt/log/sin/cos/abs are supported directly.
    ``sqrt`` is represented by SymPy as ``Pow(base, 1/2)`` and therefore gets
    an explicit check. ``pow``/``**`` can be used to reject every power node.
    """
    disabled = {str(name).strip().lower() for name in (disabled_operators or [])}
    if not disabled:
        return False
    aliases = {
        "exponential": "exp", "ln": "log", "square_root": "sqrt",
        "**": "pow", "power": "pow",
    }
    disabled = {aliases.get(name, name) for name in disabled}
    for node in sp.preorder_traversal(sp.sympify(expr)):
        if node.is_Pow:
            if "pow" in disabled:
                return True
            if "sqrt" in disabled and node.exp == sp.Rational(1, 2):
                return True
        function_name = node.func.__name__.lower()
        if function_name in disabled:
            return True
    return False


def depth(expr):
    return 1 + max((depth(arg) for arg in expr.args), default=0)


def complexity(expr):
    """Return structural complexity by counting every SymPy tree node."""
    expression = sp.sympify(expr)

    def count(node):
        return 1 + sum(count(argument) for argument in node.args)

    return count(expression)


def raw_tree_key(expr):
    """Exact key for the already-built SymPy tree; performs no simplification."""
    return sp.srepr(expr)


def deduplicate_exact(expressions):
    """Deduplicate one batch by its raw tree only, preserving input order."""
    seen = set()
    result = []
    for expr in expressions:
        key = raw_tree_key(expr)
        if key not in seen:
            seen.add(key)
            result.append(expr)
    return result


class CandidateBatcher:
    """Bounded candidate buffering with configurable legacy compatibility.

    ``batch_exact`` keeps no cross-generation history.  ``global_structure``
    retains the former constant-normalized global structure deduplication and
    is available only for compatibility with older experiments.
    """
    def __init__(self, batch_size, callback=None, deduplication_scope="batch_exact"):
        if deduplication_scope not in ("batch_exact", "global_structure"):
            raise ValueError(
                "gp.deduplication_scope must be batch_exact or global_structure"
            )
        self.batch_size = int(batch_size)
        self.callback = callback
        self.scope = deduplication_scope
        self.pending = []
        self.history = set() if self.scope == "global_structure" else None
        self.last_batch = []
        self.total_emitted = 0
        self.batches_emitted = 0

    def _deduplicate(self, expressions):
        if self.scope == "batch_exact":
            return deduplicate_exact(expressions)
        result = []
        for expr in expressions:
            key = canonical_structure(expr)
            if key not in self.history:
                self.history.add(key)
                result.append(expr)
        return result

    def _emit(self, expressions, generation):
        batch = self._deduplicate(expressions)
        if not batch:
            return False
        self.last_batch = batch
        self.total_emitted += len(batch)
        self.batches_emitted += 1
        return bool(self.callback and self.callback(batch, generation))

    def add(self, expressions, generation, stop_after_emit=False):
        expressions = list(expressions)
        if not expressions:
            return False
        if self.callback is None or self.batch_size <= 0:
            return self._emit(expressions, generation)
        self.pending.extend(expressions)
        while len(self.pending) >= self.batch_size:
            raw_batch = self.pending[:self.batch_size]
            del self.pending[:self.batch_size]
            if self._emit(raw_batch, generation):
                return True
            if stop_after_emit:
                return False
        return False

    def discard_pending(self):
        """Drop candidates belonging to a population that is being restarted."""
        self.pending.clear()

    def flush(self, generation):
        if not self.pending:
            return False
        raw_batch, self.pending = self.pending, []
        return self._emit(raw_batch, generation)


def nodes_with_paths(expr, include_root=True):
    """Return every occurrence as ``(child-index path, node)``.

    Paths distinguish structurally equal/cached SymPy nodes, which a node
    object or an xreplace mapping cannot do.
    """
    result = []

    def visit(node, path):
        if include_root or path:
            result.append((path, node))
        for index, argument in enumerate(node.args):
            visit(argument, path + (index,))

    visit(sp.sympify(expr), ())
    return result


def replace_at_path(expr, path, replacement):
    """Replace exactly one subtree occurrence selected by its argument path."""
    path = tuple(path)
    if not path:
        return sp.sympify(replacement)
    index = path[0]
    if index < 0 or index >= len(expr.args):
        raise IndexError(f"invalid subtree path {path} for {expr}")
    arguments = list(expr.args)
    arguments[index] = replace_at_path(arguments[index], path[1:], replacement)
    return expr.func(*arguments)


def valid(expr, max_depth=None, max_nodes=15):
    try:
        expr = expr if isinstance(expr, sp.Basic) else sp.sympify(expr)
        return (bool(expr.free_symbols) and not expr.has(sp.nan, sp.zoo, sp.oo, -sp.oo)
                and (max_depth is None or depth(expr) <= max_depth)
                and complexity(expr) <= max_nodes)
    except Exception:
        return False


def _scaled_variable(rng, variables, coefficient_range=(.25, 2.0)):
    """Create a variable terminal with a non-zero leading coefficient."""
    low, high = coefficient_range
    coefficient = rng.uniform(low, high) * (-1.0 if rng.random() < .5 else 1.0)
    return sp.Float(coefficient) * sp.Symbol(rng.choice(variables))


def delete_subtree_mutation(expr, rng, max_depth=7, max_nodes=63,
                            variables=("x1", "x2"), coefficient_range=(.25, 2.0)):
    nodes = nodes_with_paths(expr, include_root=False)
    if not nodes:
        return expr
    terminal = (_scaled_variable(rng, variables, coefficient_range)
                if rng.random() < .8 else sp.Float(rng.uniform(-2, 2)))
    path, _ = rng.choice(nodes)
    changed = replace_at_path(expr, path, terminal)
    return changed if valid(changed, max_depth, max_nodes) else expr


def _protected_log(child, rng, coefficient_range):
    """Build the scaled protected-log form used by GP-generated subtrees."""
    low, high = coefficient_range
    coefficient = rng.uniform(low, high) * (-1.0 if rng.random() < .5 else 1.0)
    return sp.Float(coefficient) * sp.log(sp.Abs(child) + sp.Float(.01))


def random_tree(rng, variables, level=0, max_depth=5, log_probability=.30,
                log_coefficient_range=(.25, 2.0), trig_probability=.20,
                nested_log_trig_probability=0.0):
    if level >= max_depth or rng.random() < .30:
        return (_scaled_variable(rng, variables, log_coefficient_range)
                if rng.random() < .70 else sp.Float(rng.uniform(-2, 2)))
    if (nested_log_trig_probability > 0
            and rng.random() < nested_log_trig_probability):
        child = random_tree(
            rng, variables, level + 1, max_depth, log_probability,
            log_coefficient_range, trig_probability,
            nested_log_trig_probability,
        )
        trig_child = rng.choice((sp.sin, sp.cos))(child)
        return _protected_log(trig_child, rng, log_coefficient_range)
    if rng.random() < log_probability:
        child = random_tree(
            rng, variables, level + 1, max_depth, log_probability,
            log_coefficient_range, trig_probability,
            nested_log_trig_probability,
        )
        # Protected log is favoured and remains real for almost all finite inputs.
        return _protected_log(child, rng, log_coefficient_range)
    if rng.random() < trig_probability:
        child = random_tree(
            rng, variables, level + 1, max_depth, log_probability,
            log_coefficient_range, trig_probability,
            nested_log_trig_probability,
        )
        return rng.choice((sp.sin, sp.cos))(child)
    left = random_tree(
        rng, variables, level + 1, max_depth, log_probability,
        log_coefficient_range, trig_probability, nested_log_trig_probability,
    )
    right = random_tree(
        rng, variables, level + 1, max_depth, log_probability,
        log_coefficient_range, trig_probability, nested_log_trig_probability,
    )
    return rng.choice((lambda: left + right, lambda: left - right, lambda: left * right,
                       lambda: left / (right + sp.Float(.01))))()


def mutate(expr, rng, cfg, variables):
    limit = int(cfg.get("max_formula_complexity", 15))
    if rng.random() < cfg["delete_subtree_probability"]:
        coefficient_range = (cfg.get("log_coefficient_min", .25),
                             cfg.get("log_coefficient_max", 2.0))
        return delete_subtree_mutation(expr, rng, None, limit, variables, coefficient_range)
    target_path, _ = rng.choice(nodes_with_paths(expr))
    coefficient_range = (cfg.get("log_coefficient_min", .25),
                         cfg.get("log_coefficient_max", 2.0))
    child = random_tree(rng, variables, max_depth=2,
                        log_probability=cfg.get("log_probability", .30),
                        log_coefficient_range=coefficient_range,
                        trig_probability=cfg.get("trig_probability", .20),
                        nested_log_trig_probability=cfg.get(
                            "nested_log_trig_probability", 0.0
                        ))
    changed = replace_at_path(expr, target_path, child)
    return changed if valid(changed, None, limit) else expr


def crossover(left, right, rng, cfg):
    target_path, _ = rng.choice(nodes_with_paths(left))
    _, donor = rng.choice(nodes_with_paths(right))
    child = replace_at_path(left, target_path, donor)
    limit = int(cfg.get("max_formula_complexity", 15))
    return child if valid(child, None, limit) else left


def eligible_candidate(expr, cfg):
    """Hard constraints for formulas exposed to screening and persistence."""
    if contains_disabled_operator(expr, cfg.get("disabled_operators", ["exp", "sqrt"])):
        return False
    if (cfg.get("require_transcendental", False)
            and not has_transcendental_function(expr)):
        return False
    return valid(expr, None, int(cfg.get("max_formula_complexity", 15)))


def pareto_front(results):
    """Non-dominated solutions: maximize true R², minimize node complexity."""
    front = []
    for item in results:
        dominated = any(
            other["true_r2"] >= item["true_r2"]
            and other["complexity"] <= item["complexity"]
            and (other["true_r2"] > item["true_r2"] or other["complexity"] < item["complexity"])
            for other in results
        )
        if not dominated:
            front.append(item)
    return sorted(front, key=lambda item: (item["complexity"], -item["true_r2"]))


def generate_target_free(cfg, variables, seed, candidate_callback=None):
    """Generate formulas without X, y, correlations, fitness, or neural scores."""
    rng = random.Random(seed)
    coefficient_range = (cfg.get("log_coefficient_min", .25),
                         cfg.get("log_coefficient_max", 2.0))
    limit = int(cfg.get("max_formula_complexity", 15))
    def initialize_population():
        population = []
        attempts = 0
        while (len(population) < cfg["population_size"]
               and attempts < cfg["population_size"] * 50):
            attempts += 1
            expr = random_tree(
                rng, variables, max_depth=4,
                log_probability=cfg.get("log_probability", .30),
                log_coefficient_range=coefficient_range,
                trig_probability=cfg.get("trig_probability", .20),
                nested_log_trig_probability=cfg.get(
                    "nested_log_trig_probability", 0.0
                ),
            )
            if valid(expr, None, limit):
                population.append(expr)
        if not population:
            raise RuntimeError(
                "unable to create an initial GP population within complexity limit"
            )
        return population

    population = initialize_population()
    batch_size = int(cfg.get("candidates_per_fit", 0))
    max_generations = int(cfg.get("max_generations", cfg.get("generations", 20)))
    batcher = CandidateBatcher(
        batch_size, candidate_callback,
        cfg.get("deduplication_scope", "global_structure"),
    )
    restart_after_screening = (
        bool(cfg.get("restart_population_after_screening", False))
        and candidate_callback is not None
    )

    for generation in range(max_generations):
        eligible = [expr for expr in population if eligible_candidate(expr, cfg)]
        batches_before = batcher.batches_emitted
        if batcher.add(
                eligible, generation + 1,
                stop_after_emit=restart_after_screening):
            logging.info("target-free GP reached configured target at generation=%d",
                         generation + 1)
            return batcher.last_batch
        if (restart_after_screening
                and batcher.batches_emitted > batches_before):
            batcher.discard_pending()
            if generation + 1 >= max_generations:
                return batcher.last_batch
            population = initialize_population()
            logging.info(
                "target-free GP restarted population after screening batch=%d",
                batcher.batches_emitted,
            )
            continue
        pool = population
        population = []
        for _ in range(cfg["population_size"]):
            left = rng.choice(pool)
            if rng.random() < cfg.get("crossover_probability", .65):
                child = crossover(left, rng.choice(pool), rng, cfg)
            else:
                child = left
            if rng.random() < cfg.get("mutation_probability", .80):
                child = mutate(child, rng, cfg, variables)
            population.append(child)
        logging.info(
            "target-free GP generation=%d/%d screened=%d batches=%d",
            generation + 1, max_generations,
            batcher.total_emitted, batcher.batches_emitted,
        )
    eligible = [expr for expr in population if eligible_candidate(expr, cfg)]
    if batcher.add(eligible, max_generations):
        return batcher.last_batch
    batcher.flush(max_generations)
    return batcher.last_batch


def run_gp(predictor, X, y, output_path, cfg, lm_cfg, seed=42, variables=None):
    variables = variables or [f"x{i + 1}" for i in range(X.shape[1])]
    fitted_results = []
    # Only the legacy mode retains cross-batch screening history.  The fast
    # default relies exclusively on exact deduplication inside each batch.
    screened_structures = (
        set() if cfg.get("deduplication_scope", "global_structure")
        == "global_structure" else None
    )
    fit_count = 0
    callback_invocations = 0
    target_r2 = float(cfg.get("target_r2", 1.0))
    target_complexity = int(cfg.get("target_complexity", 0))
    top_k = int(cfg.get("fit_top_k_per_batch", 2))
    threshold = cfg.get("r2_threshold")

    def screen_and_fit(batch, generation):
        nonlocal fit_count, callback_invocations
        callback_invocations += 1
        # Keep this guard even though the built-in generator already batches:
        # legacy/custom generators must also deduplicate before NN and LM.
        batch = deduplicate_exact(batch)
        formulas = [str(expr) for expr in batch]
        scores = np.asarray(predictor(formulas), dtype=float)
        ranked = sorted(
            ((formula, float(score)) for formula, score in zip(formulas, scores)
             if np.isfinite(score) and score <= 1.0
             and (threshold is None or score >= threshold)),
            key=lambda pair: pair[1], reverse=True,
        )[:top_k]
        reached = False
        for formula, predicted in ranked:
            structure = canonical_structure(formula)
            if screened_structures is not None:
                if structure in screened_structures:
                    continue
                screened_structures.add(structure)
            fit_count += 1
            fitted = fit_constants(
                formula, X, y, lm_cfg["restarts"], lm_cfg["max_nfev"], seed + fit_count
            )
            if fitted and np.isfinite(fitted["true_r2"]) and fitted["true_r2"] <= 1.0:
                row = {
                    "structure": structure, "formula": formula,
                    "predicted_r2": predicted, "true_r2": fitted["true_r2"],
                    "optimized_constants": fitted["constants"],
                    "optimized_formula": fitted["optimized_formula"],
                    "complexity": complexity(sp.sympify(formula)),
                    "generation": generation,
                }
                fitted_results.append(row)
                reached = reached or (
                    row["true_r2"] >= target_r2
                    and row["complexity"] <= target_complexity
                )
        return reached

    candidates = generate_target_free(
        cfg, variables, seed, candidate_callback=screen_and_fit
    )
    # Compatibility for custom/legacy generators that do not invoke the callback.
    if callback_invocations == 0:
        batch_size = max(1, int(cfg.get("candidates_per_fit", len(candidates) or 1)))
        for offset in range(0, len(candidates), batch_size):
            if screen_and_fit(candidates[offset:offset + batch_size], None):
                break

    final = pareto_front(fitted_results)
    if cfg.get("pareto_max_results"):
        final = final[:int(cfg["pareto_max_results"])]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("GP last_batch=%d NN/LM fitted=%d LM_valid=%d pareto=%d",
                 len(candidates), fit_count, len(fitted_results), len(final))
    return final
