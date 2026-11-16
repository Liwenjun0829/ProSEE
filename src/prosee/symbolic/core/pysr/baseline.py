"""PySR baseline implementation, isolated from the main formula pipeline."""
from __future__ import annotations

# The baseline shares numerical helpers and constants with the main pipeline.
# It is imported lazily by the pipeline, after that module has initialized.
from .. import formula_discovery_pipeline as _pipeline

for _name, _value in vars(_pipeline).items():
    if not _name.startswith("__"):
        globals().setdefault(_name, _value)

# Extracted from formula_discovery_pipeline.py (_apply_pysr_reproducibility_mode)
def _apply_pysr_reproducibility_mode(config: dict) -> dict:
    """Resolve the one-switch PySR execution profile in a config."""
    pysr_config = config.setdefault("pysr_baseline", {})
    mode = str(pysr_config.get("reproducibility_mode", "multithreaded")).strip().lower()
    if mode == "multithreaded":
        pysr_config.update(
            reproducible=False,
            deterministic=False,
            parallelism="multithreading",
            timeout_in_seconds=3600,
        )
    elif mode == "reproducible":
        pysr_config.update(
            reproducible=True,
            deterministic=True,
            parallelism="serial",
            timeout_in_seconds=None,
        )
    else:
        raise ValueError(
            "pysr_baseline.reproducibility_mode must be 'multithreaded' or 'reproducible'"
        )
    return config

# Extracted from formula_discovery_pipeline.py (_pysr_structure_key)
def _pysr_structure_key(expression: str) -> str:
    """Return a stable key after replacing fitted numeric constants.

    Only explicit floating-point leaves emitted by PySR are treated as fitted
    constants.  SymPy creates exact ``Integer``/``Rational`` nodes while
    canonicalising an expression (notably ``x*x -> x**2``); those are part of
    the structure and must never be made dataset-specific parameters.
    """
    expression_obj = sp.sympify(str(expression))
    counter = 0

    def replace_constants(node):
        nonlocal counter
        if isinstance(node, sp.Float):
            replacement = sp.Symbol(f"C{counter}")
            counter += 1
            return replacement
        if not node.args:
            return node
        return node.func(*(replace_constants(arg) for arg in node.args))

    return sp.srepr(replace_constants(expression_obj))

# Extracted from formula_discovery_pipeline.py (_pysr_has_only_integer_exponents)
def _pysr_has_only_integer_exponents(expression: str | sp.Expr) -> bool:
    """Return whether every explicit power exponent is integer-valued."""
    expression_obj = sp.sympify(expression)
    for power in expression_obj.atoms(sp.Pow):
        exponent = power.exp
        if exponent.is_Integer:
            continue
        if exponent.is_Rational and exponent.q == 1:
            continue
        if exponent.is_Float and math.isclose(
            float(exponent), round(float(exponent)), rel_tol=0.0, abs_tol=1e-12
        ):
            continue
        return False
    return True

# Extracted from formula_discovery_pipeline.py (_pysr_power_exponent_has_numeric_constant)
def _pysr_power_exponent_has_numeric_constant(
    expression: str | sp.Expr,
) -> bool:
    """Return whether an exported power exponent contains a float constant.

    A PySR ``^`` exponent represented by a floating literal is a tunable
    constant in the normal PySR search.  Integer literals cannot be reliably
    distinguished from fitted constants after CSV/SymPy export, so this is a
    conservative output-time safeguard only; the Julia objective below is the
    authoritative in-search check.
    """
    expression_obj = sp.sympify(expression)
    for power in expression_obj.atoms(sp.Pow):
        if any(isinstance(node, sp.Float)
               for node in sp.preorder_traversal(power.exp)):
            return True
    return False

# Extracted from formula_discovery_pipeline.py (_pysr_parameter_template)
def _pysr_parameter_template(
    expression: str,
) -> tuple[sp.Expr, list[sp.Symbol], list[float]]:
    """Parameterise explicit PySR floats while preserving the exact tree.

    Exact integers and rationals are structural syntax, rather than tunable
    constants.  This distinction prevents, for example, a structural square
    from becoming a dataset-specific real exponent during refitting.
    """
    expression_obj = sp.sympify(str(expression))
    # Integer-valued floating exponents are structural integers, not tunable
    # constants.  This keeps fixed-structure refitting within the same
    # integer-exponent search space as the main model.
    def normalize_integer_powers(node):
        if isinstance(node, sp.Pow):
            base = normalize_integer_powers(node.base)
            exponent = normalize_integer_powers(node.exp)
            if isinstance(exponent, sp.Float) and math.isclose(
                float(exponent), round(float(exponent)), rel_tol=0.0, abs_tol=1e-12
            ):
                exponent = sp.Integer(round(float(exponent)))
            return sp.Pow(base, exponent)
        if not node.args:
            return node
        return node.func(*(normalize_integer_powers(arg) for arg in node.args))

    expression_obj = normalize_integer_powers(expression_obj)
    parameters = []
    initial_values = []

    def replace_constants(node):
        if isinstance(node, sp.Float):
            parameter = sp.Symbol(f"p{len(parameters)}")
            parameters.append(parameter)
            initial_values.append(float(node))
            return parameter
        if not node.args:
            return node
        return node.func(*(replace_constants(arg) for arg in node.args))

    return replace_constants(expression_obj), parameters, initial_values

# Extracted from formula_discovery_pipeline.py (_safe_pysr_expression)
def _safe_pysr_expression(row: dict) -> str | None:
    """Extract a PySR guess expression from an equations CSV row."""
    for key in ("sympy_format", "equation", "lambda_format"):
        value = row.get(key)
        if value is None:
            continue
        value = str(value).strip()
        if not value or value.startswith("PySRFunction"):
            continue
        try:
            sp.sympify(value)
        except (TypeError, ValueError, sp.SympifyError):
            continue
        return value
    return None

# Extracted from formula_discovery_pipeline.py (_pysr_extra_sympy_mappings)
def _pysr_extra_sympy_mappings() -> dict[str, Any]:
    """Map the protected wrappers used by the main formula library."""
    return {
        "log_abs": lambda value: sp.log(sp.Abs(value)),
        "log_abs_sin": lambda value: sp.log(sp.Abs(sp.sin(value))),
        "log_abs_cos": lambda value: sp.log(sp.Abs(sp.cos(value))),
    }

# Extracted from formula_discovery_pipeline.py (_pysr_lambdify_modules)
def _pysr_lambdify_modules() -> list[object]:
    """Numerical implementations for PySR's protected unary wrappers."""
    return [{
        "log_abs": lambda value: np.log(np.abs(value)),
        "log_abs_sin": lambda value: np.log(np.abs(np.sin(value))),
        "log_abs_cos": lambda value: np.log(np.abs(np.cos(value))),
    }, "numpy"]

# Extracted from formula_discovery_pipeline.py (_pysr_has_x1_leaf_with_log_ancestor)
def _pysr_has_x1_leaf_with_log_ancestor(
    expression: sp.Expr, x1_name: str = "x1",
) -> bool:
    """Require at least one x1 leaf to have a log node among its ancestors."""
    x1 = sp.Symbol(str(x1_name))
    log_function_names = {"log", "log_abs", "log_abs_sin", "log_abs_cos"}

    def visit(node: sp.Expr, has_log_ancestor: bool) -> bool:
        current_has_log = (
            has_log_ancestor
            or node.func is sp.log
            or getattr(node.func, "__name__", "") in log_function_names
        )
        if node == x1 and not node.args:
            return current_has_log
        return any(visit(child, current_has_log) for child in node.args)

    return visit(sp.sympify(expression), False)

# Extracted from formula_discovery_pipeline.py (_pysr_nonlinear_unary_operator_names)
def _pysr_nonlinear_unary_operator_names(baseline_cfg: dict) -> set[str]:
    """Return configured nonlinear unary operators as atomic operators."""
    return {
        _pysr_operator_name(operator)
        for operator in baseline_cfg.get("unary_operators", [])
    }

# Extracted from formula_discovery_pipeline.py (_pysr_has_nested_nonlinear_operators)
def _pysr_has_nested_nonlinear_operators(
    expression: str | sp.Expr, baseline_cfg: dict,
) -> bool:
    """Check whether a nonlinear unary operator contains another below it.

    Configured operators are treated as primitives, so log_abs_sin and
    log_abs_cos each count as one operator and their internal composition is
    exempt from the nesting rule.
    """
    nonlinear_names = _pysr_nonlinear_unary_operator_names(baseline_cfg)
    expression_obj = sp.sympify(expression)

    def is_nonlinear(node: sp.Expr) -> bool:
        return getattr(node.func, "__name__", str(node.func)) in nonlinear_names

    def contains_nonlinear(node: sp.Expr) -> bool:
        return is_nonlinear(node) or any(contains_nonlinear(arg) for arg in node.args)

    def has_nested(node: sp.Expr) -> bool:
        if is_nonlinear(node) and any(contains_nonlinear(arg) for arg in node.args):
            return True
        return any(has_nested(arg) for arg in node.args)

    return has_nested(expression_obj)

# Extracted from formula_discovery_pipeline.py (_pysr_has_division_by_nonlinear_structure)
def _pysr_has_division_by_nonlinear_structure(expression: str | sp.Expr) -> bool:
    """Return whether a division has a nonlinear denominator subtree.

    Only configured unary function calls count as nonlinear. Arithmetic
    operators, including products and powers, are traversed but do not by
    themselves make a subtree nonlinear.
    """
    expression_obj = sp.sympify(expression)

    def is_nonlinear(node: sp.Expr) -> bool:
        if node.is_Atom:
            return False
        if node.is_Function:
            return True
        return any(is_nonlinear(arg) for arg in node.args)

    def contains_forbidden_division(node: sp.Expr) -> bool:
        if node.is_Pow and node.exp.is_number and node.exp < 0:
            if is_nonlinear(node.base):
                return True
        return any(contains_forbidden_division(arg) for arg in node.args)

    return contains_forbidden_division(expression_obj)

# Extracted from formula_discovery_pipeline.py (_pysr_max_nonconstant_additive_terms)
def _pysr_max_nonconstant_additive_terms(baseline_cfg: dict) -> int | None:
    """Return the maximum allowed top-level nonconstant additive terms."""
    value = baseline_cfg.get("max_nonconstant_additive_terms")
    if value is None:
        # Backward compatibility: the old field is now interpreted as a maximum.
        value = baseline_cfg.get("required_nonconstant_additive_terms")
    if value is None:
        return None
    value = int(value)
    if value < 0:
        raise ValueError("max_nonconstant_additive_terms must be non-negative")
    return value

# Extracted from formula_discovery_pipeline.py (_pysr_structural_penalty)
def _pysr_structural_penalty(
    expression: str, baseline_cfg: dict, x1_name: str = "x1",
    required_feature_names: Sequence[str] | None = None,
) -> float:
    """Return a large loss for a PySR tree violating structural constraints."""
    require_log = bool(baseline_cfg.get("require_log_ancestor_for_x1", False))
    require_all_features = bool(
        baseline_cfg.get("require_all_features", False)
    )
    forbid_bfgs_power_constants = bool(
        baseline_cfg.get("forbid_bfgs_constants_in_power_exponents", False)
    ) and bool(baseline_cfg.get("should_optimize_constants", True))
    max_additive_terms = _pysr_max_nonconstant_additive_terms(baseline_cfg)
    forbid_nested_nonlinear = bool(
        baseline_cfg.get("forbid_nested_nonlinear_operators", False)
    )
    forbid_division_by_nonlinear = bool(
        baseline_cfg.get("forbid_division_by_nonlinear_structure", False)
    )
    if (not require_log and not require_all_features
            and not forbid_bfgs_power_constants
            and not forbid_nested_nonlinear
            and not forbid_division_by_nonlinear
            and max_additive_terms is None):
        return 0.0
    try:
        expression_obj = sp.sympify(expression)
        if (forbid_bfgs_power_constants
                and _pysr_power_exponent_has_numeric_constant(expression_obj)):
            return 1e300
        if (forbid_nested_nonlinear
                and _pysr_has_nested_nonlinear_operators(
                    expression_obj, baseline_cfg)):
            return 1e300
        if (forbid_division_by_nonlinear
                and _pysr_has_division_by_nonlinear_structure(expression_obj)):
            return 1e300
        if require_log and not _pysr_has_x1_leaf_with_log_ancestor(
            expression_obj, x1_name=x1_name,
        ):
            return 1e300
        if require_all_features:
            names = tuple(str(name) for name in (required_feature_names or ()))
            symbols = {sp.Symbol(name) for name in names}
            if symbols and not symbols.issubset(expression_obj.free_symbols):
                return 1e300
        if (max_additive_terms is not None
                and _pysr_nonconstant_additive_term_count(expression_obj)
                > max_additive_terms):
            return 1e300
    except (TypeError, ValueError, sp.SympifyError):
        return 1e300
    return 0.0

# Extracted from formula_discovery_pipeline.py (_pysr_nonconstant_additive_term_count)
def _pysr_nonconstant_additive_term_count(expression: sp.Expr) -> int:
    """Count top-level +/- terms, ignoring pure constant terms."""
    expression = sp.sympify(expression)
    terms = expression.args if expression.is_Add else (expression,)
    return sum(not term.is_number for term in terms)

# Extracted from formula_discovery_pipeline.py (_pysr_operator_name)
def _pysr_operator_name(operator: str) -> str:
    """Extract a Julia operator name from a PySR operator specification."""
    specification = str(operator).strip()
    match = re.match(r"([A-Za-z_]\w*)\s*\(", specification)
    return match.group(1) if match else specification

# Extracted from formula_discovery_pipeline.py (_pysr_structural_loss_function)
def _pysr_structural_loss_function(
    baseline_cfg: dict, feature_names: Sequence[str],
) -> str | None:
    """Build the Julia objective used to enforce tree constraints in search.

    ``loss_function`` runs for every candidate evaluation inside
    SymbolicRegression.jl.  It must be Julia source rather than a Python
    callback so parallel PySR populations remain compiled and process-safe.
    """
    require_log = bool(baseline_cfg.get("require_log_ancestor_for_x1", False))
    require_all_features = bool(baseline_cfg.get("require_all_features", False))
    forbid_bfgs_power_constants = bool(
        baseline_cfg.get("forbid_bfgs_constants_in_power_exponents", False)
    ) and bool(baseline_cfg.get("should_optimize_constants", True))
    max_additive_terms = _pysr_max_nonconstant_additive_terms(baseline_cfg)
    forbid_nested_nonlinear = bool(
        baseline_cfg.get("forbid_nested_nonlinear_operators", False)
    )
    forbid_division_by_nonlinear = bool(
        baseline_cfg.get("forbid_division_by_nonlinear_structure", False)
    )
    enabled = bool(baseline_cfg.get("structural_loss_during_search", False))
    if not enabled or not (
        require_log or require_all_features or forbid_bfgs_power_constants
        or forbid_nested_nonlinear or forbid_division_by_nonlinear
        or max_additive_terms is not None
    ):
        return None

    unary_operators = list(baseline_cfg.get("unary_operators", []))
    binary_operators = list(baseline_cfg.get("binary_operators", [
        "+", "-", "*", "/",
    ]))
    log_operator_names = {"log", "log_abs", "log_abs_sin", "log_abs_cos"}
    log_operator_indices = [
        index + 1 for index, operator in enumerate(unary_operators)
        if _pysr_operator_name(operator) in log_operator_names
    ]
    power_operator_indices = [
        index + 1 for index, operator in enumerate(binary_operators)
        if str(operator).strip() == "^"
    ]
    nonlinear_unary_operator_indices = [
        index + 1 for index, _operator in enumerate(unary_operators)
    ]
    division_operator_indices = [
        index + 1 for index, operator in enumerate(binary_operators)
        if str(operator).strip() in {"/", "div"}
    ]
    additive_operator_indices = [
        index + 1 for index, operator in enumerate(binary_operators)
        if str(operator).strip() in {"+", "-"}
    ]
    penalty = float(baseline_cfg.get("structural_loss_penalty", 1e300))
    if not math.isfinite(penalty) or penalty <= 0.0:
        raise ValueError(
            "pysr_baseline.structural_loss_penalty must be a finite positive number"
        )
    all_features_check = ""
    if require_all_features:
        all_features_check = "\n".join([
            f"    for feature_index in 1:{len(feature_names)}",
            f"        if !has_feature(tree, feature_index); return L({penalty!r}); end",
            "    end",
        ])
    nested_nonlinear_check = ""
    if forbid_nested_nonlinear and nonlinear_unary_operator_indices:
        nested_nonlinear_check = "\n".join([
            "    function contains_nonlinear_unary(node)",
            "        if node.degree == 0; return false; end",
            f"        if node.degree == 1 && node.op in {nonlinear_unary_operator_indices!r}; return true; end",
            "        return contains_nonlinear_unary(node.l) || (node.degree == 2 && contains_nonlinear_unary(node.r))",
            "    end",
            "    function has_nested_nonlinear_unary(node)",
            "        if node.degree == 0; return false; end",
            f"        if node.degree == 1 && node.op in {nonlinear_unary_operator_indices!r} && contains_nonlinear_unary(node.l); return true; end",
            "        return has_nested_nonlinear_unary(node.l) || (node.degree == 2 && has_nested_nonlinear_unary(node.r))",
            "    end",
            f"    if has_nested_nonlinear_unary(tree); return L({penalty!r}); end",
        ])
    nonlinear_denominator_check = ""
    if forbid_division_by_nonlinear and division_operator_indices:
        nonlinear_denominator_check = "\n".join([
            "    function is_nonlinear(node)",
            "        if node.degree == 0; return false; end",
            f"        if node.degree == 1 && node.op in {nonlinear_unary_operator_indices!r}; return true; end",
            "        if node.degree == 1; return is_nonlinear(node.l); end",
            "        return is_nonlinear(node.l) || is_nonlinear(node.r)",
            "    end",
            "    function has_nonlinear_denominator(node)",
            "        if node.degree == 0; return false; end",
            f"        if node.degree == 2 && node.op in {division_operator_indices!r} && is_nonlinear(node.r); return true; end",
            "        return has_nonlinear_denominator(node.l) || (node.degree == 2 && has_nonlinear_denominator(node.r))",
            "    end",
            f"    if has_nonlinear_denominator(tree); return L({penalty!r}); end",
        ])
    additive_terms_check = ""
    if max_additive_terms is not None:
        additive_terms_check = "\n".join([
            "    function nonconstant_additive_terms(node)",
            "        if node.degree == 0",
            "            return node.constant ? 0 : 1",
            "        elseif node.degree == 2 && node.op in "
            f"{additive_operator_indices!r}",
            "            return nonconstant_additive_terms(node.l) + nonconstant_additive_terms(node.r)",
            "        end",
            "        return 1",
            "    end",
            f"    if nonconstant_additive_terms(tree) > {int(max_additive_terms)}; return L({penalty!r}); end",
        ])
    # ``feature`` and ``op`` are one-based in SymbolicRegression.jl.  The
    # explicit recursive walk avoids allocating expression copies per tree.
    return f'''
function pysr_structural_loss(tree, dataset::Dataset{{T,L}}, options)::L where {{T,L}}
    function has_feature(node, feature_index)
        if node.degree == 0
            return !node.constant && node.feature == feature_index
        elseif node.degree == 1
            return has_feature(node.l, feature_index)
        end
        return has_feature(node.l, feature_index) || has_feature(node.r, feature_index)
    end
    function has_x1_with_log_ancestor(node, under_log)
        current_under_log = under_log || (node.degree == 1 && node.op in {log_operator_indices!r})
        if node.degree == 0
            return !node.constant && node.feature == 1 && current_under_log
        elseif node.degree == 1
            return has_x1_with_log_ancestor(node.l, current_under_log)
        end
        return has_x1_with_log_ancestor(node.l, current_under_log) || has_x1_with_log_ancestor(node.r, current_under_log)
    end
    function has_constant(node)
        if node.degree == 0
            return node.constant
        elseif node.degree == 1
            return has_constant(node.l)
        end
        return has_constant(node.l) || has_constant(node.r)
    end
    function has_tunable_power_exponent(node)
        if node.degree == 0
            return false
        elseif node.degree == 1
            return has_tunable_power_exponent(node.l)
        end
        return ((node.op in {power_operator_indices!r}) && has_constant(node.r)) || has_tunable_power_exponent(node.l) || has_tunable_power_exponent(node.r)
    end

    {'if !has_x1_with_log_ancestor(tree, false); return L(' + repr(penalty) + '); end' if require_log else ''}
    {all_features_check}
    {'if has_tunable_power_exponent(tree); return L(' + repr(penalty) + '); end' if forbid_bfgs_power_constants else ''}
    {nested_nonlinear_check}
    {nonlinear_denominator_check}
    {additive_terms_check}
    prediction, completed = eval_tree_array(tree, dataset.X, options)
    if !completed
        return L(Inf)
    end
    return sum(abs2, prediction .- dataset.y) / length(dataset.y)
end
'''

# Extracted from formula_discovery_pipeline.py (_pysr_template_structural_loss_function)
def _pysr_template_structural_loss_function(
    baseline_cfg: dict, feature_names: Sequence[str],
) -> str | None:
    """Apply the ordinary tree constraints to template ``f`` while scoring
    the complete template expression (including its category parameters).
    """
    source = _pysr_structural_loss_function(baseline_cfg, feature_names)
    if source is None:
        return None
    source = source.replace(
        "function pysr_structural_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}",
        "function pysr_template_structural_loss(expression, dataset::Dataset{T}, options::Options) where T",
    )
    source = source.replace(
        "function pysr_template_structural_loss(expression, dataset::Dataset{T}, options::Options) where T\n",
        "function pysr_template_structural_loss(expression, dataset::Dataset{T}, options::Options) where T\n    tree = get_tree(expression.trees.f)\n",
    )
    source = source.replace("return L(", "return T(")
    source = source.replace(
        "prediction, completed = eval_tree_array(tree, dataset.X, options)",
        "prediction, completed = eval_tree_array(expression, dataset.X, options)",
    )
    source = source.replace(
        "return sum(abs2, prediction .- dataset.y) / length(dataset.y)",
        "residual = abs2.(prediction .- dataset.y)\n"
        "    if dataset.weights === nothing\n"
        "        return sum(residual) / length(dataset.y)\n"
        "    end\n"
        "    return sum(dataset.weights .* residual) / sum(dataset.weights)",
    )
    return source

# Extracted from formula_discovery_pipeline.py (_pysr_prepare_equations_for_output)
def _pysr_prepare_equations_for_output(
    equations, baseline_cfg: dict, x1_name: str = "x1",
    required_feature_names: Sequence[str] | None = None,
):
    """Annotate violating PySR equations with a large structural loss."""
    if not (
        bool(baseline_cfg.get("require_log_ancestor_for_x1", False))
        or bool(baseline_cfg.get("require_all_features", False))
        or bool(baseline_cfg.get(
            "forbid_bfgs_constants_in_power_exponents", False
        ))
        or bool(baseline_cfg.get("forbid_nested_nonlinear_operators", False))
        or bool(baseline_cfg.get(
            "forbid_division_by_nonlinear_structure", False
        ))
        or _pysr_max_nonconstant_additive_terms(baseline_cfg) is not None
    ):
        return equations
    prepared = equations.copy()
    prepared["raw_loss"] = prepared["loss"]
    prepared["structural_penalty"] = 0.0
    for index, row in prepared.iterrows():
        expression = _safe_pysr_expression(row)
        penalty = (
            _pysr_structural_penalty(
                expression, baseline_cfg, x1_name, required_feature_names
            )
            if expression is not None else 1e300
        )
        prepared.at[index, "structural_penalty"] = penalty
        if penalty > 0.0:
            prepared.at[index, "loss"] = penalty
    return prepared


# Extracted from formula_discovery_pipeline.py (_PySRProgressLoggerSpec)
class _PySRProgressLoggerSpec:
    """Small PySR logger adapter which records Pareto snapshots to TSV.

    The logger is observational only: it runs from PySR's native Julia
    callback and does not alter mutation, selection, population, or stopping
    behavior.  It is intentionally defined without importing PySR at module
    import time because PySR is optional and the fit runs in a watchdog child.
    """

    def __init__(self, path: str, log_interval: int = 1):
        self.path = str(path)
        self.log_interval = max(1, int(log_interval))

    def create_logger(self):
        from pysr.julia_import import jl

        jl.seval("using Base64")
        jl.seval(r"""
            if !isdefined(SymbolicRegression, :CodexProgressLogger)
                @eval SymbolicRegression begin
                    import Logging
                    import DynamicExpressions
                    import .LoggingModule: get_logger, logging_callback!
                    mutable struct CodexProgressLogger <: AbstractSRLogger
                        path::String
                        log_interval::Int
                        step::Base.RefValue{Int}
                    end
                    get_logger(logger::CodexProgressLogger) = Logging.NullLogger()
                    function logging_callback!(logger::CodexProgressLogger;
                            state, datasets, ropt, options)
                        logger.step[] += 1
                        open(logger.path, "a") do io
                            println(io, "#progress\t", logger.step[])
                        end
                        if mod(logger.step[] - 1, logger.log_interval) != 0
                            return nothing
                        end
                        open(logger.path, "a") do io
                            for (output_index, (hall, dataset)) in enumerate(
                                    zip(state.halls_of_fame, datasets))
                                dominating = HallOfFameModule.calculate_pareto_frontier(hall)
                                for member in dominating
                                    complexity = ComplexityModule.compute_complexity(
                                        member, options)
                                    equation = DynamicExpressions.string_tree(
                                        member.tree, options;
                                        variable_names=dataset.variable_names)
                                    encoded = Main.Base64.base64encode(
                                        Vector{UInt8}(codeunits(equation)))
                                    println(io, logger.step[], "\t", output_index,
                                        "\t", complexity, "\t", member.loss, "\t", encoded)
                                end
                            end
                        end
                        return nothing
                    end
                end
            end
        """)
        make_logger = jl.seval(
            "(path, interval) -> SymbolicRegression.CodexProgressLogger(path, interval, Ref(0))"
        )
        return make_logger(self.path, self.log_interval)

    def write_hparams(self, logger, hparams):
        return None

    def close(self, logger):
        return None

# Extracted from formula_discovery_pipeline.py (_read_pysr_progress_log)
def _read_pysr_progress_log(path: str | Path | None) -> list[dict]:
    """Read the optional native PySR Hall-of-Fame progress log."""
    if not path:
        return []
    path = Path(path)
    if not path.is_file():
        return []
    import base64

    rows = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t", 4)
            if len(parts) != 5:
                continue
            try:
                rows.append({
                    "step": int(parts[0]),
                    "output": int(parts[1]),
                    "complexity": int(parts[2]),
                    "loss": float(parts[3]),
                    "equation": base64.b64decode(parts[4]).decode("utf-8"),
                })
            except (TypeError, ValueError, UnicodeDecodeError):
                continue
    except OSError:
        return []
    return rows

# Extracted from formula_discovery_pipeline.py (_pysr_equations_for_ipc)
def _pysr_equations_for_ipc(equations):
    """Remove live Julia callables before sending PySR equations across processes.

    TemplateExpressionSpec exports ``lambda_format`` as a Julia-backed
    callable. Returning the complete DataFrame through a multiprocessing
    queue makes the receiving process try to deserialize that Julia object,
    which fails because its Julia runtime belongs to the child process.
    The pipeline only needs the human-readable equation and scalar metrics.
    """
    if equations is None:
        return None
    if isinstance(equations, list):
        return [_pysr_equations_for_ipc(frame) for frame in equations]
    if not hasattr(equations, "columns"):
        return equations

    result = equations.drop(
        columns=["lambda_format", "jax_format", "torch_format"],
        errors="ignore",
    ).copy()
    safe_types = (str, int, float, bool, type(None), np.generic)
    for column in result.columns:
        if result[column].dtype == object:
            result[column] = result[column].map(
                lambda value: value if isinstance(value, safe_types) else str(value)
            )
    return result

# Extracted from formula_discovery_pipeline.py (_pysr_template_expression_for_display)
def _pysr_template_expression_for_display(
    expression: str, feature_names: Sequence[str], parameter_names: Sequence[str],
) -> str:
    """Render PySR's numbered template inputs using pipeline names.

    PySR numbers the combined template inputs in order: physical features
    first, followed by the indexed parameter vectors. For example, with two
    features ``#1``/``#2`` are those features and ``#3``... are ``p1``...
    parameter vectors.
    """
    input_names = {index + 1: str(name)
                   for index, name in enumerate(feature_names)}
    parameter_offset = len(feature_names)
    input_names.update({
        parameter_offset + index + 1: f"{name}[dataset_id]"
        for index, name in enumerate(parameter_names)
    })

    def replace_input(match: re.Match) -> str:
        input_index = int(match.group(1))
        return input_names.get(input_index, match.group(0))

    return re.sub(r"#(\d+)\b", replace_input, str(expression))

# Extracted from formula_discovery_pipeline.py (_pysr_fit_payload)
def _pysr_fit_payload(
    X: np.ndarray, y: np.ndarray, feature_names: Sequence[str], kwargs: dict,
    prediction_inputs: dict[str, np.ndarray],
) -> dict:
    """Run one PySR fit and return only serializable artifacts.

    Keeping the model object inside the child process is intentional: a
    PySRRegressor owns a Julia runtime and cannot be safely pickled back to
    the parent process.
    """
    from importlib.metadata import PackageNotFoundError, version

    from pysr import PySRRegressor

    model_kwargs = dict(kwargs)
    progress_log_path = model_kwargs.pop("_progress_log_path", None)
    fit_weights = model_kwargs.pop("_fit_weights", None)
    template_spec_config = model_kwargs.pop("_template_expression_spec", None)
    equation_prediction_mode = bool(
        model_kwargs.pop("_predict_all_equations", False)
    )
    if progress_log_path:
        model_kwargs["logger_spec"] = _PySRProgressLoggerSpec(
            str(progress_log_path),
            int(model_kwargs.pop("_progress_log_interval", 1)),
        )
    # Lambdas are not pickleable under the ``spawn`` start method.  Rebuild
    # this mapping in the child rather than falling back to unsafe ``fork``.
    model_kwargs["extra_sympy_mappings"] = _pysr_extra_sympy_mappings()
    if template_spec_config is not None:
        from pysr import TemplateExpressionSpec

        model_kwargs["expression_spec"] = TemplateExpressionSpec(
            expressions=list(template_spec_config["expressions"]),
            variable_names=list(template_spec_config["variable_names"]),
            parameters=dict(template_spec_config["parameters"]),
            combine=str(template_spec_config["combine"]),
        )
    model = PySRRegressor(**model_kwargs)
    model.fit(
        X, y, variable_names=list(feature_names), weights=fit_weights
    )
    progress = _read_pysr_progress_log(progress_log_path)
    try:
        pysr_version = version("pysr")
    except PackageNotFoundError:  # pragma: no cover - import already succeeded
        pysr_version = None
    runtime = {
        "python_version": sys.version,
        "pysr_version": pysr_version,
        "numpy_version": np.__version__,
        "scipy_version": version("scipy"),
        "sympy_version": sp.__version__,
        "pandas_version": version("pandas"),
    }
    try:
        from pysr.julia_import import jl
        runtime.update({
            "julia_version": str(jl.seval("string(VERSION)")),
            "symbolic_regression_version": str(jl.seval(
                "string(Base.pkgversion(SymbolicRegression))"
            )),
            "julia_threads": int(jl.seval("Threads.nthreads()")),
        })
    except Exception as exc:  # metadata must never invalidate a completed fit
        runtime["julia_runtime_metadata_error"] = f"{type(exc).__name__}: {exc}"
    predictions = {
        name: np.asarray(model.predict(values), dtype=float).reshape(-1)
        for name, values in prediction_inputs.items()
    }
    equation_predictions = None
    if equation_prediction_mode:
        equation_predictions = {}
        equations = getattr(model, "equations_", None)
        for equation_index in range(0 if equations is None else len(equations)):
            equation_predictions[str(equation_index)] = {
                name: np.asarray(
                    model.predict(values, index=equation_index), dtype=float
                ).reshape(-1)
                for name, values in prediction_inputs.items()
            }
    selected_equation = None
    if template_spec_config is None:
        selected_equation = str(model.sympy())
    elif getattr(model, "equations_", None) is not None and len(model.equations_):
        selected_equation = str(model.equations_.iloc[0].get("equation", ""))
    return {
        "predictions": predictions,
        "equation_predictions": equation_predictions,
        "equations": _pysr_equations_for_ipc(
            getattr(model, "equations_", None)
        ),
        "equation": selected_equation,
        "search_progress": progress,
        "runtime": runtime,
    }

# Extracted from formula_discovery_pipeline.py (_pysr_fit_worker)
def _pysr_fit_worker(
    result_queue, X: np.ndarray, y: np.ndarray, feature_names: Sequence[str],
    kwargs: dict, prediction_inputs: dict[str, np.ndarray],
) -> None:
    """Process entry point for a watchdog-protected PySR fit."""
    iteration_timeout = kwargs.pop("_pysr_iteration_timeout_seconds", None)
    progress_path = kwargs.pop("_pysr_progress_path", None)
    if iteration_timeout is not None:
        os.environ["PYSR_ITERATION_TIMEOUT_SECONDS"] = str(float(iteration_timeout))
    if progress_path is not None:
        os.environ["PYSR_PROGRESS_LOG_PATH"] = str(progress_path)
    # PySR can start Julia child processes.  A dedicated process group lets
    # the parent terminate that whole group on a hard timeout on POSIX.
    if os.name == "posix":
        try:
            os.setsid()
        except OSError:
            pass
    try:
        result_queue.put(("ok", _pysr_fit_payload(
            X, y, feature_names, kwargs, prediction_inputs
        )))
    except BaseException as exc:  # report import/Julia failures to the parent
        result_queue.put((
            "error",
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        ))

# Extracted from formula_discovery_pipeline.py (_run_pysr_fit_with_watchdog)
def _run_pysr_fit_with_watchdog(
    X: np.ndarray, y: np.ndarray, feature_names: Sequence[str], kwargs: dict,
    prediction_inputs: dict[str, np.ndarray], baseline_cfg: dict,
) -> dict:
    """Run one PySR fit with parent-enforced total and idle deadlines.

    ``timeout_in_seconds`` is forwarded to PySR as before, but it is not a
    reliable parent-side guarantee when Julia is stuck during startup, native
    code, or shutdown. The watchdog can also enforce a maximum period without
    progress-log updates, including when the reproducible profile disables
    PySR's total wall-clock timeout.
    """
    if not bool(baseline_cfg.get("process_watchdog_enabled", True)):
        return _pysr_fit_payload(X, y, feature_names, kwargs, prediction_inputs)

    configured_timeout = baseline_cfg.get("timeout_in_seconds")
    iteration_timeout = baseline_cfg.get(
        "process_watchdog_iteration_timeout_seconds", 160
    )
    startup_timeout = float(baseline_cfg.get(
        "process_watchdog_startup_timeout_seconds", 1800
    ))
    if configured_timeout is None and iteration_timeout is None:
        return _pysr_fit_payload(X, y, feature_names, kwargs, prediction_inputs)
    timeout = None if configured_timeout is None else float(configured_timeout)
    iteration_timeout = (
        None if iteration_timeout is None else float(iteration_timeout)
    )
    grace = float(baseline_cfg.get("process_watchdog_grace_seconds", 60.0))
    if ((timeout is not None and timeout <= 0)
            or (iteration_timeout is not None and iteration_timeout <= 0)
            or startup_timeout <= 0 or grace < 0):
        raise ValueError(
            "PySR watchdog timeout must be positive; "
            "process_watchdog_grace_seconds cannot be negative"
        )
    total_deadline = (
        None if timeout is None else time.monotonic() + timeout + grace
    )
    idle_deadline = (
        time.monotonic() + startup_timeout
        if iteration_timeout is not None else None
    )
    progress_path = kwargs.get("_progress_log_path")
    try:
        initial_stat = Path(progress_path).stat() if progress_path else None
        last_progress_signature = (
            (initial_stat.st_mtime_ns, initial_stat.st_size)
            if initial_stat is not None else None
        )
    except OSError:
        last_progress_signature = None
    progress_seen = False
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_pysr_fit_worker,
        args=(
            result_queue, X, y, list(feature_names),
            {
                **kwargs,
                "_pysr_iteration_timeout_seconds": baseline_cfg.get(
                    "pysr_iteration_timeout_seconds", 100
                ),
                "_pysr_progress_path": kwargs.get("_progress_log_path"),
            }, prediction_inputs,
        ),
        daemon=False,
    )
    process.start()
    try:
        while True:
            now = time.monotonic()
            deadlines = [
                deadline for deadline in (total_deadline, idle_deadline)
                if deadline is not None
            ]
            wait_seconds = min(
                1.0, max(0.01, min(deadlines) - now)
            ) if deadlines else 1.0
            try:
                status, payload = result_queue.get(timeout=wait_seconds)
                break
            except queue.Empty:
                pass

            if iteration_timeout is not None and progress_path:
                try:
                    current_stat = Path(progress_path).stat()
                    current_signature = (
                        current_stat.st_mtime_ns, current_stat.st_size
                    )
                except OSError:
                    current_signature = None
                if (current_signature is not None
                        and current_signature != last_progress_signature):
                    last_progress_signature = current_signature
                    progress_seen = True
                    idle_deadline = time.monotonic() + iteration_timeout

            now = time.monotonic()
            if total_deadline is not None and now >= total_deadline:
                timeout_reason = f"total timeout of {timeout + grace:g}s"
            elif idle_deadline is not None and now >= idle_deadline:
                timeout_reason = (
                    f"no progress produced within {startup_timeout:g}s after startup"
                    if not progress_seen else
                    f"no new search progress for {iteration_timeout:g}s"
                )
            else:
                if not process.is_alive():
                    try:
                        status, payload = result_queue.get(timeout=1.0)
                    except queue.Empty as exc:
                        raise RuntimeError(
                            f"PySR child process exited early with exitcode={process.exitcode} "
                            "and returned no result"
                        ) from exc
                    break
                continue

            if process.is_alive():
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except (OSError, ProcessLookupError):
                        process.terminate()
                else:
                    process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join()
            raise TimeoutError(f"PySR child process exceeded {timeout_reason} and was terminated")
    finally:
        result_queue.close()
        result_queue.join_thread()

    process.join(timeout=10)
    if process.is_alive():
        process.terminate()
        process.join()
    if status != "ok":
        raise RuntimeError(f"PySR child process failed: {payload}")
    return payload


# Extracted from formula_discovery_pipeline.py (_pysr_metric_pair)
def _pysr_metric_pair(target: np.ndarray, prediction: np.ndarray) -> tuple[float | None, float | None]:
    """Return finite-sample R² and RMSE for a fixed-structure fit."""
    target = np.asarray(target, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    valid = np.isfinite(target) & np.isfinite(prediction)
    if int(np.sum(valid)) < 2:
        return None, None
    residual = target[valid] - prediction[valid]
    total = float(np.sum((target[valid] - np.mean(target[valid])) ** 2))
    r2 = 1.0 - float(np.sum(residual ** 2)) / total if total > 0 else None
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    return r2, rmse

# Extracted from formula_discovery_pipeline.py (_pysr_evaluation_metrics)
def _pysr_evaluation_metrics(
    target: np.ndarray, prediction: np.ndarray,
) -> dict:
    """Evaluate a PySR prediction using only the finite-value rule.

    PySR is a symbolic-regression baseline, not a masked-prediction model.
    Held-out scores therefore require at least two finite paired samples, but
    do not impose the main pipeline's training-time 90% validity threshold.
    Keeping this rule explicit prevents an evaluation-only coverage threshold
    from making the baseline incomparable to ordinary held-out regression.
    """
    target = np.asarray(target, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    valid = np.isfinite(target) & np.isfinite(prediction)
    valid_count = int(np.sum(valid))
    total_count = int(target.size)
    required_count = 2
    r2, rmse = _pysr_metric_pair(target, prediction)
    return {
        "r2": r2 if valid_count >= required_count else None,
        "rmse": rmse if valid_count >= required_count else None,
        "valid_samples": valid_count,
        "total_samples": total_count,
        "minimum_required_valid_samples": required_count,
        "valid_fraction": (float(valid_count) / total_count if total_count else 0.0),
        "valid": bool(valid_count >= required_count and r2 is not None),
    }

# Extracted from formula_discovery_pipeline.py (_evaluate_pysr_expression_on_split)
def _evaluate_pysr_expression_on_split(
    expression: str, X: np.ndarray, y: np.ndarray,
    feature_names: Sequence[str],
) -> dict | None:
    """Evaluate one fixed PySR expression without changing its constants."""
    try:
        symbols = [sp.Symbol(str(name)) for name in feature_names]
        local_symbols = {str(symbol): symbol for symbol in symbols}
        local_symbols.update(_pysr_extra_sympy_mappings())
        parsed = sp.sympify(expression, locals=local_symbols)
        function = sp.lambdify(symbols, parsed, modules=_pysr_lambdify_modules())
        target = np.asarray(y, dtype=float).reshape(-1)
        prediction = np.asarray(
            function(*np.asarray(X, dtype=float).T), dtype=float
        ).reshape(-1)
        if prediction.size == 1 and target.size != 1:
            prediction = np.full(target.shape, float(prediction[0]), dtype=float)
        if prediction.size != target.size:
            return None
        return _pysr_evaluation_metrics(target, prediction)
    except (ArithmeticError, FloatingPointError, TypeError, ValueError,
            KeyError, sp.SympifyError, OverflowError):
        return None

# Extracted from formula_discovery_pipeline.py (_pysr_independent_validation_rows)
def _pysr_independent_validation_rows(
    datasets: Sequence[dict], baseline_cfg: dict, validation_data,
    feature_names: Sequence[str], evaluation_label: str = "validation",
) -> list[dict]:
    """Score every native PySR Pareto equation on the frozen validation split."""
    if validation_data is None:
        return []
    evaluation_X, evaluation_y, evaluation_groups = validation_data
    groups_by_name = {
        str(group.get("name")): group for group in evaluation_groups
    }
    rows = []
    for dataset_index, item in enumerate(datasets, start=1):
        if item.get("status") != "completed":
            continue
        group = groups_by_name.get(str(item.get("dataset_name")))
        equations_path = item.get("equations_csv")
        if group is None or not equations_path:
            continue
        try:
            with Path(str(equations_path)).open(
                newline="", encoding="utf-8"
            ) as handle:
                equation_rows = list(csv.DictReader(handle))
        except (OSError, csv.Error):
            continue
        start, end = int(group["start"]), int(group["end"])
        for row_index, equation_row in enumerate(equation_rows):
            expression = _safe_pysr_expression(equation_row)
            if expression is None:
                continue
            if _pysr_structural_penalty(
                expression, baseline_cfg, feature_names[0], feature_names
            ) > 0.0:
                continue
            metrics = _evaluate_pysr_expression_on_split(
                expression, evaluation_X[start:end], evaluation_y[start:end],
                feature_names,
            )
            if not metrics or not metrics.get("valid"):
                continue
            try:
                complexity = int(formula_complexity(sp.sympify(expression)))
            except (TypeError, ValueError, sp.SympifyError):
                complexity = None
            rows.append({
                "formula_id": (
                    f"pysr_{item.get('dataset_name', dataset_index)}_"
                    f"{row_index:06d}"
                ),
                "readable_expression": expression,
                f"{evaluation_label}_mean_r2": metrics["r2"],
                f"{evaluation_label}_r2_std": 0.0,
                f"{evaluation_label}_mean_rmse": metrics["rmse"],
                f"{evaluation_label}_complexity": complexity,
                "complexity": complexity,
                "dataset_name": item.get("dataset_name"),
                "source_equations_csv": str(equations_path),
                "source_row_index": row_index,
                "pysr_loss": equation_row.get("loss"),
                "formula_family": "pysr_native_pareto",
            })
    return rows

# Extracted from formula_discovery_pipeline.py (_pysr_independent_validation_result)
def _pysr_independent_validation_result(
    datasets: Sequence[dict], baseline_cfg: dict, validation_data,
    feature_names: Sequence[str], evaluation_label: str = "validation",
    data_mode: str = "csv",
) -> dict:
    """Build the canonical windowed front from native PySR Pareto equations."""
    rows = _pysr_independent_validation_rows(
        datasets, baseline_cfg, validation_data, feature_names, evaluation_label,
    )
    rows = _pysr_hv_window_front(
        rows, data_mode, evaluation_label,
        r2_window=max(0.0, float(baseline_cfg.get(
            "final_validation_r2_window", DEFAULT_PYSR_R2_WINDOW
        ))),
    )
    return {
        "metadata": {
            "experiment_variant": "pysr_baseline",
            "evaluation_label": evaluation_label,
            "selection": "native_pareto_validation_r2_window",
            "final_validation_r2_window": max(0.0, float(baseline_cfg.get(
                "final_validation_r2_window", DEFAULT_PYSR_R2_WINDOW
            ))),
        },
        "final_pareto_front": rows,
        "history": [{"iteration": 0, "front": rows}],
        "per_iteration": [],
    }

# Extracted from formula_discovery_pipeline.py (_save_pysr_pareto_front_csv)
def _save_pysr_pareto_front_csv(
    result: dict | None, result_directory: Path, evaluation_label: str,
) -> Path | None:
    """Write a PySR evaluation front using the pipeline's standard CSV writer."""
    if not result:
        return None
    rows = [copy.deepcopy(row) for row in result.get("final_pareto_front", [])]
    for rank, row in enumerate(rows, start=1):
        row.setdefault("pareto_rank", rank)
    path = result_directory / f"{evaluation_label}_pareto_front.csv"
    _write_csv(path, rows)
    return path

# Extracted from formula_discovery_pipeline.py (_evaluate_frozen_pysr_front_on_test)
def _evaluate_frozen_pysr_front_on_test(
    validation_result: dict | None, test_data, feature_names: Sequence[str],
) -> dict | None:
    """Evaluate the validation-selected independent PySR front on sealed test data.

    This is reporting only: expressions and constants come from the validation
    result, and no test-side filtering or refitting is performed.
    """
    if not validation_result or test_data is None:
        return None
    test_X, test_y, test_groups = test_data
    if len(test_groups) != 1:
        return None
    test_X = np.asarray(test_X)
    test_y = np.asarray(test_y, dtype=float).reshape(-1)
    evaluated_rows = []
    for source in validation_result.get("final_pareto_front", []):
        metrics = _evaluate_pysr_expression_on_split(
            str(source.get("readable_expression", "")), test_X, test_y,
            feature_names,
        )
        if not metrics or not metrics.get("valid"):
            continue
        row = copy.deepcopy(source)
        row.update({
            "test_mean_r2": metrics["r2"],
            "test_pooled_r2": metrics["r2"],
            "test_r2_std": 0.0,
            "test_min_r2": metrics["r2"],
            "test_mean_rmse": metrics["rmse"],
            "test_complexity": source.get("complexity"),
            "test_valid_samples": metrics["valid_samples"],
            "test_total_samples": metrics["total_samples"],
            "test_valid_fraction": metrics["valid_fraction"],
        })
        evaluated_rows.append(row)
    return {
        "metadata": {
            **(validation_result.get("metadata", {}) or {}),
            "evaluation_label": "test",
            "selection_split": "validation",
            "test_data_used_for_selection": False,
            "test_evaluation": "frozen_validation_selected_expressions",
            "test_dataset": [dict(group) for group in test_groups],
        },
        "final_pareto_front": evaluated_rows,
        "history": [{"iteration": 0, "front": evaluated_rows}],
        "per_iteration": [],
    }

# Extracted from formula_discovery_pipeline.py (_run_pysr_shared_template_baseline)
def _run_pysr_shared_template_baseline(
    baseline_dir: Path, X: np.ndarray, y: np.ndarray,
    feature_names: Sequence[str], dataset_groups: Sequence[dict],
    config: dict, seed: int, validation_data=None, test_data=None,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Jointly evolve one PySR template with category-specific parameter vectors.

    Dataset identity is only exposed to the template for indexing its parameter
    vectors. The searched expression receives the physical input columns and
    those indexed parameters, never the dataset ID as an ordinary feature.
    """
    baseline_cfg = config.get("pysr_baseline", {}) or {}
    is_realdata_mat = _is_measured_pathloss_profile(config.get("data", {}))
    template_cfg = baseline_cfg.get("shared_template", {}) or {}
    baseline_dir.mkdir(parents=True, exist_ok=True)
    names = [str(group["name"]) for group in dataset_groups]
    if len(set(names)) != len(names):
        raise ValueError("PySR shared_template requires unique dataset names")
    category_by_name = {name: index + 1 for index, name in enumerate(names)}
    physical_names = [f"x{index + 1}" for index in range(len(feature_names))]
    template_names = [*physical_names, "dataset_id"]
    parameter_count = int(template_cfg.get("parameter_count", 4))
    if parameter_count < 1:
        raise ValueError("pysr_baseline.shared_template.parameter_count must be at least 1")
    parameter_names = [f"p{index + 1}" for index in range(parameter_count)]
    combine_args = [*physical_names]
    combine_args.extend(f"{name}[dataset_id]" for name in parameter_names)
    template_spec = {
        "expressions": ["f"],
        "variable_names": template_names,
        "parameters": {name: len(names) for name in parameter_names},
        "combine": f"f({', '.join(combine_args)})",
    }

    def with_categories(values: np.ndarray, groups: Sequence[dict]) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(feature_names):
            raise ValueError("The number of PySR shared_template feature columns does not match the training set")
        category = np.empty((len(values), 1), dtype=float)
        assigned = np.zeros(len(values), dtype=bool)
        for group in groups:
            start, end = int(group["start"]), int(group["end"])
            name = str(group["name"])
            if name not in category_by_name:
                raise ValueError(f"Data partition contains a dataset not seen in training: {name}")
            category[start:end, 0] = category_by_name[name]
            assigned[start:end] = True
        if not np.all(assigned):
            raise ValueError("PySR shared_template dataset groups do not cover all samples")
        return np.column_stack([values, category])

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    prediction_inputs = {"training": with_categories(X, dataset_groups)}
    fit_weights = np.zeros(len(y), dtype=float)
    for group in dataset_groups:
        start, end = int(group["start"]), int(group["end"])
        count = end - start
        if count < 1:
            raise ValueError(f"PySR shared_template dataset is empty: {group.get('name')}")
        fit_weights[start:end] = len(y) / (len(dataset_groups) * count)
    evaluation_arrays: dict[str, tuple[np.ndarray, np.ndarray, Sequence[dict]]] = {}
    for label, data in (("validation", validation_data), ("test", test_data)):
        if data is None:
            continue
        eval_X, eval_y, eval_groups = data[:3]
        prediction_inputs[label] = with_categories(eval_X, eval_groups)
        evaluation_arrays[label] = (
            np.asarray(eval_X, dtype=float),
            np.asarray(eval_y, dtype=float).reshape(-1), eval_groups,
        )

    common_kwargs = {
        "niterations": int(baseline_cfg.get("niterations", 100)),
        "populations": int(baseline_cfg.get("populations", 31)),
        "population_size": int(baseline_cfg.get("population_size", 27)),
        "maxsize": int(baseline_cfg.get("maxsize", 30)),
        "binary_operators": list(baseline_cfg.get(
            "binary_operators", ["+", "-", "*", "/"])),
        "unary_operators": list(baseline_cfg.get("unary_operators", [])),
        "model_selection": str(baseline_cfg.get("model_selection", "best")),
        "verbosity": int(baseline_cfg.get("verbosity", 1)),
        "random_state": int(seed),
        "_template_expression_spec": template_spec,
        "_predict_all_equations": True,
        "_fit_weights": fit_weights,
        "_progress_log_path": str(baseline_dir / "shared_template_progress.tsv"),
        "_progress_log_interval": int(baseline_cfg.get("progress_log_interval", 1)),
    }
    structural_loss = _pysr_template_structural_loss_function(
        baseline_cfg, physical_names
    )
    if structural_loss is not None:
        common_kwargs["loss_function_expression"] = structural_loss
    for key in (
        "optimizer_algorithm", "optimizer_nrestarts", "optimizer_iterations",
        "optimizer_f_calls_limit", "should_optimize_constants",
        "optimize_probability", "deterministic", "parallelism", "precision",
        "timeout_in_seconds",
    ):
        if key in baseline_cfg:
            common_kwargs[key] = baseline_cfg[key]

    try:
        fit_payload = _run_pysr_fit_with_watchdog(
            prediction_inputs["training"], y, template_names, common_kwargs,
            prediction_inputs, baseline_cfg,
        )
    except Exception as exc:
        logging.warning("PySR multi-file shared-template baseline failed: %s", exc)
        metadata = {
            "enabled": True, "available": True,
            "fit_scope": "joint_template_shared_structure_category_parameters",
            "status": "failed", "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "dataset_names": names,
            "shared_template": template_spec,
        }
        metadata_path = baseline_dir / "pysr_baseline.json"
        metadata["metadata_json"] = str(metadata_path)
        metadata_path.write_text(
            json.dumps(_json_value(metadata), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return metadata, {}
    equations = fit_payload.get("equations")
    if equations is None or len(equations) == 0:
        metadata = {
            "enabled": True, "available": True,
            "fit_scope": "joint_template_shared_structure_category_parameters",
            "status": "completed_without_equations", "datasets": [],
            "runtime": fit_payload.get("runtime", {}),
        }
        metadata_path = baseline_dir / "pysr_baseline.json"
        metadata["metadata_json"] = str(metadata_path)
        metadata_path.write_text(json.dumps(_json_value(metadata), ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata, {}

    equations_path = baseline_dir / "shared_template_equations.csv"
    equations.to_csv(equations_path, index=False)
    equation_predictions = fit_payload.get("equation_predictions") or {}
    validation_candidates = []
    train_predictions: dict[str, np.ndarray] = {}
    for position, (_equation_index, equation_row) in enumerate(equations.iterrows()):
        index = int(position)
        candidate_predictions = equation_predictions.get(str(index), {})
        complexity_value = equation_row.get("complexity")
        try:
            complexity = int(complexity_value)
        except (TypeError, ValueError, OverflowError):
            complexity = 1
        candidate_repr = str(equation_row.get("julia_expression", "")).strip()
        expression_text = str(equation_row.get("equation", "")).strip()
        if not expression_text or expression_text.lower() == "nan":
            expression_text = candidate_repr
        if not expression_text or expression_text.lower() == "nan":
            expression_text = f"TemplateExpressionSpec candidate {index}"
        readable_expression = _pysr_template_expression_for_display(
            expression_text, feature_names, parameter_names
        )
        row = {
            "formula_id": f"pysr_shared_template_{index:06d}",
            "readable_expression": readable_expression,
            "complexity": complexity,
            "formula_family": "pysr_template_shared_structure",
            "source_equations_csv": str(equations_path),
            "source_row_index": index,
            "pysr_loss": equation_row.get("loss"),
            "dataset_parameters": "optimized category-specific template vectors",
            "template_candidate_repr": candidate_repr,
        }
        if "validation" not in evaluation_arrays:
            continue
        validation_X, validation_y, validation_groups = evaluation_arrays["validation"]
        validation_pred = np.asarray(candidate_predictions.get("validation", []), dtype=float)
        per_dataset = []
        if validation_pred.size != len(validation_y):
            continue
        for group in validation_groups:
            start, end = int(group["start"]), int(group["end"])
            metrics = _pysr_evaluation_metrics(
                validation_y[start:end], validation_pred[start:end]
            )
            if not metrics.get("valid"):
                continue
            per_dataset.append({
                "dataset_name": str(group["name"]),
                "validation_r2": metrics["r2"],
                "validation_rmse": metrics["rmse"],
                "validation_valid_samples": metrics["valid_samples"],
                "validation_total_samples": metrics["total_samples"],
            })
        if not per_dataset:
            continue
        row.update({
            "validation_mean_r2": float(np.mean([item["validation_r2"] for item in per_dataset])),
            "validation_r2_std": float(np.std([item["validation_r2"] for item in per_dataset])),
            "validation_mean_rmse": float(np.mean([item["validation_rmse"] for item in per_dataset])),
            "validation_pooled_r2": _pysr_metric_pair(validation_y, validation_pred)[0],
            "validation_complexity": complexity,
            "dataset_fit_results": per_dataset,
        })
        validation_candidates.append(row)

    selected_training_prediction = np.asarray(
        fit_payload.get("predictions", {}).get("training", []), dtype=float
    )
    if selected_training_prediction.size == len(y):
        for group in dataset_groups:
            start, end = int(group["start"]), int(group["end"])
            train_predictions[str(group["name"])] = selected_training_prediction[start:end]

    # Keep every candidate with valid validation metrics. The R² window is
    # temporarily disabled for shared-template PySR so low-scoring results
    # remain available for inspection and comparison.
    validation_front = validation_candidates
    validation_result = {
        "metadata": {
            "experiment_variant": "pysr_shared_template",
            "fit_scope": "joint_training_data_shared_formula_structure",
            "evaluation_label": "validation",
            "selection": "all_candidates_with_valid_validation_metrics",
            "dataset_names": names,
            "template": template_spec,
            "parameter_count_per_dataset": parameter_count,
        },
        "final_pareto_front": validation_front,
        "history": [{"iteration": 0, "front": validation_front}],
        "per_iteration": [],
    }
    validation_path = baseline_dir / "validation_results.json"
    if validation_path.is_file():
        validation_path.unlink()
    _save_pysr_pareto_front_csv(validation_result, baseline_dir, "validation")

    if "test" in evaluation_arrays and validation_front:
        test_X, test_y, test_groups = evaluation_arrays["test"]
        test_candidates = []
        for source in validation_front:
            index = int(source["source_row_index"])
            test_pred = np.asarray(
                equation_predictions.get(str(index), {}).get("test", []), dtype=float
            )
            if test_pred.size != len(test_y):
                continue
            per_dataset = []
            for group in test_groups:
                start, end = int(group["start"]), int(group["end"])
                metrics = _pysr_evaluation_metrics(test_y[start:end], test_pred[start:end])
                if metrics.get("valid"):
                    per_dataset.append({
                        "dataset_name": str(group["name"]),
                        "test_r2": metrics["r2"], "test_rmse": metrics["rmse"],
                        "test_valid_samples": metrics["valid_samples"],
                        "test_total_samples": metrics["total_samples"],
                    })
            if not per_dataset:
                continue
            item = copy.deepcopy(source)
            # Keep per-group averages for diagnostics, and persist a
            # sample-weighted pooled RMSE for whole-holdout comparisons.
            pooled_rmse = None
            pooled_rmse_groups = [
                (float(entry["test_rmse"]), int(entry["test_valid_samples"]))
                for entry in per_dataset
                if entry.get("test_rmse") is not None
                and entry.get("test_valid_samples") is not None
                and int(entry["test_valid_samples"]) > 0
            ]
            pooled_valid_count = int(sum(count for _, count in pooled_rmse_groups))
            if (len(pooled_rmse_groups) == len(per_dataset)
                    and pooled_valid_count == int(np.sum(
                        np.isfinite(test_y) & np.isfinite(test_pred)))):
                pooled_rmse = math.sqrt(sum(
                    count * rmse * rmse for rmse, count in pooled_rmse_groups
                ) / pooled_valid_count)
            item.update({
                "test_mean_r2": float(np.mean([entry["test_r2"] for entry in per_dataset])),
                "test_pooled_r2": _pysr_metric_pair(test_y, test_pred)[0],
                "test_r2_std": float(np.std([entry["test_r2"] for entry in per_dataset])),
                "test_mean_rmse": float(np.mean([entry["test_rmse"] for entry in per_dataset])),
                "test_complexity": source["complexity"],
                "test_dataset_fit_results": per_dataset,
            })
            if is_realdata_mat:
                item.update({
                    "test_group_r2_std": float(np.std([entry["test_r2"] for entry in per_dataset])),
                    "test_group_mean_r2": float(np.mean([entry["test_r2"] for entry in per_dataset])),
                    "test_group_mean_rmse": float(np.mean([entry["test_rmse"] for entry in per_dataset])),
                    "test_pooled_rmse": pooled_rmse,
                })
            test_candidates.append(item)
        test_result = {
            "metadata": {
                **validation_result["metadata"], "evaluation_label": "test",
                "selection_split": "validation", "test_data_used_for_selection": False,
            },
            "final_pareto_front": test_candidates,
            "history": [{"iteration": 0, "front": test_candidates}],
            "per_iteration": [],
        }
        test_path = baseline_dir / "test_results.json"
        if test_path.is_file():
            test_path.unlink()
        _save_pysr_pareto_front_csv(test_result, baseline_dir, "test")

    metadata = {
        "enabled": True, "available": True,
        "fit_scope": "joint_template_shared_structure_category_parameters",
        "feature_names": list(feature_names), "dataset_names": names,
        "datasets": [{
            "dataset_name": name, "samples": int(group["end"] - group["start"]),
            "status": "completed", "equations_csv": str(equations_path),
            "equation_count": int(len(equations)),
        } for name, group in zip(names, dataset_groups)],
        "completed_datasets": len(names),
        "shared_template": {
            **template_spec,
            "parameter_count_per_dataset": parameter_count,
            "structural_loss_during_search": structural_loss is not None,
        },
        "structural_constraints_applied": structural_loss is not None,
        "structural_loss_function": (
            "loss_function_expression_on_shared_f" if structural_loss is not None else None
        ),
        "training_objective": "equal_weight_mean_squared_error_per_dataset",
        "runtime": fit_payload.get("runtime", {}),
        "search_progress": fit_payload.get("search_progress", []),
        "process_watchdog": {
            "enabled": bool(baseline_cfg.get("process_watchdog_enabled", True)),
            "pysr_iteration_timeout_seconds": baseline_cfg.get(
                "pysr_iteration_timeout_seconds", 100
            ),
            "process_fallback_no_progress_timeout_seconds": baseline_cfg.get(
                "process_watchdog_iteration_timeout_seconds", 160
            ),
            "startup_timeout_seconds": baseline_cfg.get(
                "process_watchdog_startup_timeout_seconds", 1800
            ),
        },
        "reproducibility": {
            "random_state": int(seed),
            "training_data_sha256": _array_data_sha256(
                prediction_inputs["training"], y
            ),
            "pysr_config_sha256": hashlib.sha256(json.dumps(
                _json_value({"baseline": baseline_cfg, "template": template_spec}),
                ensure_ascii=False, sort_keys=True,
            ).encode("utf-8")).hexdigest(),
            "validation_data_sha256": (
                _array_data_sha256(prediction_inputs["validation"], evaluation_arrays["validation"][1])
                if "validation" in evaluation_arrays else None
            ),
            "test_data_sha256": (
                _array_data_sha256(prediction_inputs["test"], evaluation_arrays["test"][1])
                if "test" in evaluation_arrays else None
            ),
            "pipeline_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    metadata_path = baseline_dir / "pysr_baseline.json"
    metadata["metadata_json"] = str(metadata_path)
    metadata_path.write_text(json.dumps(_json_value(metadata), ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata, train_predictions

# Extracted from formula_discovery_pipeline.py (run_pysr_baseline)
def run_pysr_baseline(
    output_dir: Path, X: np.ndarray, y: np.ndarray,
    feature_names: Sequence[str], dataset_groups: Sequence[dict],
    config: dict, seed: int, validation_data=None, test_data=None,
    baseline_output_dir: Path | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Fit an independent baseline for one file or a shared template for many."""
    baseline_cfg = config.get("pysr_baseline", {}) or {}
    baseline_dir = baseline_output_dir or (output_dir / "pysr_baseline")

    def save_status(metadata: dict) -> tuple[dict, dict[str, np.ndarray]]:
        baseline_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = baseline_dir / "pysr_baseline.json"
        metadata["metadata_json"] = str(metadata_path)
        metadata_path.write_text(
            json.dumps(_json_value(metadata), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return metadata, {}

    if not bool(baseline_cfg.get("enabled", True)):
        return save_status({"enabled": False, "reason": "disabled"})
    try:
        from pysr import PySRRegressor
    except ImportError:
        return save_status({"enabled": False, "available": False,
                            "reason": "pysr_not_installed"})

    if len(dataset_groups) > 1:
        if not bool((baseline_cfg.get("shared_template", {}) or {}).get(
                "enabled", True)):
            return save_status({
                "enabled": False,
                "reason": "shared_template_disabled_for_multiple_datasets",
                "dataset_count": len(dataset_groups),
            })
        return _run_pysr_shared_template_baseline(
            baseline_dir, X, y, feature_names, dataset_groups,
            config, seed, validation_data, test_data,
        )

    baseline_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "enabled": True, "available": True,
        "fit_scope": "independent_per_dataset_file",
        "feature_names": list(feature_names), "datasets": [],
    }
    predictions: dict[str, np.ndarray] = {}
    common_kwargs = {
        "niterations": int(baseline_cfg.get("niterations", 100)),
        "populations": int(baseline_cfg.get("populations", 31)),
        "population_size": int(baseline_cfg.get("population_size", 27)),
        "maxsize": int(baseline_cfg.get("maxsize", 30)),
        "binary_operators": list(baseline_cfg.get(
        "binary_operators", ["+", "-", "*", "/"])),
        "unary_operators": list(baseline_cfg.get(
            "unary_operators", [])),
        "model_selection": str(baseline_cfg.get("model_selection", "best")),
        "verbosity": int(baseline_cfg.get("verbosity", 1)),
        "random_state": int(seed),
    }
    structural_loss = _pysr_structural_loss_function(
        baseline_cfg, feature_names
    )
    if structural_loss is not None:
        common_kwargs["loss_function"] = structural_loss
    metadata["structural_loss_during_search"] = {
        "enabled": structural_loss is not None,
        "penalty": float(baseline_cfg.get("structural_loss_penalty", 1e300)),
        "require_all_features": bool(baseline_cfg.get(
            "require_all_features", False
        )),
        "require_log_ancestor_for_x1": bool(baseline_cfg.get(
            "require_log_ancestor_for_x1", False
        )),
        "forbid_bfgs_constants_in_power_exponents": bool(
            baseline_cfg.get("forbid_bfgs_constants_in_power_exponents", False)
        ),
        "forbid_nested_nonlinear_operators": bool(
            baseline_cfg.get("forbid_nested_nonlinear_operators", False)
        ),
        "forbid_division_by_nonlinear_structure": bool(
            baseline_cfg.get("forbid_division_by_nonlinear_structure", False)
        ),
        "max_nonconstant_additive_terms": _pysr_max_nonconstant_additive_terms(
            baseline_cfg
        ),
    }
    # Make PySR's constant optimizer explicit in the saved configuration.
    # These are PySR defaults, but recording them avoids version-dependent
    # ambiguity in experiment artifacts.
    for key in (
        "optimizer_algorithm", "optimizer_nrestarts",
        "optimizer_iterations", "optimizer_f_calls_limit",
        "should_optimize_constants", "optimize_probability",
        "deterministic", "parallelism", "precision",
        "timeout_in_seconds",
    ):
        if key in baseline_cfg:
            common_kwargs[key] = baseline_cfg[key]
    reproducibility_descriptor = json.dumps(
        _json_value(baseline_cfg), ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")
    metadata["reproducibility"] = {
        "requested": bool(baseline_cfg.get("reproducible", False)),
        "random_state": int(seed),
        "deterministic": bool(common_kwargs.get("deterministic", False)),
        "parallelism": str(common_kwargs.get("parallelism", "multithreading")),
        "stop_condition": (
            "niterations_or_no_progress_timeout"
            if common_kwargs.get("timeout_in_seconds") is None
            and baseline_cfg.get(
                "process_watchdog_iteration_timeout_seconds", 160
            )
            else "niterations" if common_kwargs.get("timeout_in_seconds") is None
            else "wall_clock_or_niterations"
        ),
        "niterations": int(common_kwargs["niterations"]),
        "timeout_in_seconds": common_kwargs.get("timeout_in_seconds"),
        "pysr_iteration_timeout_seconds": baseline_cfg.get(
            "pysr_iteration_timeout_seconds", 100
        ),
        "process_watchdog_iteration_timeout_seconds": baseline_cfg.get(
            "process_watchdog_iteration_timeout_seconds", 160
        ),
        "process_watchdog_startup_timeout_seconds": baseline_cfg.get(
            "process_watchdog_startup_timeout_seconds", 1800
        ),
        "pysr_config_sha256": hashlib.sha256(reproducibility_descriptor).hexdigest(),
        "training_data_sha256": _array_data_sha256(X, y),
        "pipeline_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if validation_data is not None:
        metadata["reproducibility"]["validation_data_sha256"] = (
            _array_data_sha256(validation_data[0], validation_data[1])
        )
    if test_data is not None:
        metadata["reproducibility"]["test_data_sha256"] = (
            _array_data_sha256(test_data[0], test_data[1])
        )
    for dataset_index, group in enumerate(dataset_groups):
        name = str(group["name"])
        safe_name = "".join(
            char if char.isalnum() or char in "-_" else "_" for char in name
        ) or f"dataset_{dataset_index + 1}"
        start, end = int(group["start"]), int(group["end"])
        X_part, y_part = np.asarray(X[start:end]), np.asarray(y[start:end])
        item = {"dataset_name": name, "samples": int(len(y_part))}
        try:
            common_kwargs["_progress_log_path"] = str(
                baseline_dir / f"{dataset_index:02d}_{safe_name}_progress.tsv"
            )
            common_kwargs["_progress_log_interval"] = int(
                baseline_cfg.get("progress_log_interval", 1)
            )
            prediction_inputs = {"training": X_part}
            evaluation_group = None
            if validation_data is not None:
                evaluation_X, _evaluation_y, evaluation_groups = validation_data
                evaluation_group = next(
                    (candidate for candidate in evaluation_groups
                     if str(candidate.get("name")) == name), None
                )
                if evaluation_group is not None:
                    v_start, v_end = (int(evaluation_group["start"]),
                                      int(evaluation_group["end"]))
                    prediction_inputs["validation"] = np.asarray(
                        evaluation_X[v_start:v_end]
                    )
            fit_payload = _run_pysr_fit_with_watchdog(
                X_part, y_part, feature_names, common_kwargs, prediction_inputs,
                baseline_cfg,
            )
            metadata.setdefault("runtime", fit_payload.get("runtime", {}))
            predicted = fit_payload["predictions"]["training"]
            predictions[name] = predicted
            valid = np.isfinite(predicted) & np.isfinite(y_part)
            if np.sum(valid) >= 2:
                residual = y_part[valid] - predicted[valid]
                ss_res = float(np.sum(residual ** 2))
                centered = y_part[valid] - np.mean(y_part[valid])
                ss_tot = float(np.sum(centered ** 2))
                item["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
                item["rmse"] = float(np.sqrt(np.mean(residual ** 2)))
            # Stage 1 is still a structure-discovery/search phase.  Keep the
            # sealed test split inaccessible until stage 2 has selected the
            # validation Pareto front.
            for evaluation_label, evaluation_data in (("validation", validation_data),):
                if evaluation_data is None or evaluation_group is None:
                    continue
                _evaluation_X, evaluation_y, _evaluation_groups = evaluation_data
                v_start, v_end = (int(evaluation_group["start"]),
                                  int(evaluation_group["end"]))
                evaluation_pred = fit_payload["predictions"][evaluation_label]
                evaluation_target = np.asarray(evaluation_y[v_start:v_end])
                evaluation_valid = (np.isfinite(evaluation_pred)
                                    & np.isfinite(evaluation_target))
                if np.sum(evaluation_valid) >= 2:
                    residual = (evaluation_target[evaluation_valid]
                                - evaluation_pred[evaluation_valid])
                    centered = (evaluation_target[evaluation_valid]
                                - np.mean(evaluation_target[evaluation_valid]))
                    total = float(np.sum(centered ** 2))
                    item[f"{evaluation_label}_r2"] = (
                        1.0 - float(np.sum(residual ** 2)) / total
                        if total > 0 else None
                    )
                    item[f"{evaluation_label}_rmse"] = float(
                        np.sqrt(np.mean(residual ** 2))
                    )
            equations = fit_payload["equations"]
            if equations is not None:
                equations_path = baseline_dir / (
                    f"{dataset_index:02d}_{safe_name}_equations.csv"
                )
                _pysr_prepare_equations_for_output(
                    equations, baseline_cfg, feature_names[0], feature_names
                ).to_csv(equations_path, index=False)
                item["equations_csv"] = str(equations_path)
            item["equation"] = fit_payload["equation"]
            item["search_progress"] = fit_payload.get("search_progress", [])
            item["status"] = "completed"
        except Exception as exc:  # PySR/Julia errors must not break Mode 3.
            item.update({"status": "failed", "error": str(exc)})
            logging.warning("PySR baseline failed for dataset %s: %s", name, exc)
        metadata["datasets"].append(item)
    metadata["completed_datasets"] = sum(
        item.get("status") == "completed" for item in metadata["datasets"]
    )
    metadata_path = baseline_dir / "pysr_baseline.json"
    metadata["metadata_json"] = str(metadata_path)
    metadata_path.write_text(
        json.dumps(_json_value(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Search uses training data and validation selects the frozen front;
    # test metrics are added only after that front has been selected.
    for evaluation_label in ("validation",):
        # The native equations CSV preserves the full PySR Pareto output.
        # The canonical baseline exposes every validation-valid equation in
        # the same fixed R² window used by the tree and random methods.
        evaluation_result = _pysr_independent_validation_result(
            metadata["datasets"], baseline_cfg, validation_data, feature_names,
            evaluation_label, _configured_data_profile(config.get("data", {})),
        )
        # Keep the independent artifact for auditability.  With shared stage 2
        # disabled, the same validation-only result is also the canonical PySR
        # baseline consumed by the tree/HV comparison.
        for obsolete_name in (
            f"independent_{evaluation_label}_results.json",
            f"{evaluation_label}_results.json",
        ):
            obsolete_path = baseline_dir / obsolete_name
            if obsolete_path.is_file():
                obsolete_path.unlink()
        _save_pysr_pareto_front_csv(
            evaluation_result, baseline_dir, evaluation_label
        )
        # Once the validation front is frozen, evaluate those exact expressions
        # on test data for standalone runs as well as the integrated pipeline.
        if evaluation_label == "validation" and test_data is not None:
            independent_test_result = _evaluate_frozen_pysr_front_on_test(
                evaluation_result,
                (test_data[0], test_data[1], test_data[2]), feature_names,
            )
            if independent_test_result is not None:
                obsolete_test_results = baseline_dir / "test_results.json"
                if obsolete_test_results.is_file():
                    obsolete_test_results.unlink()
                _save_pysr_pareto_front_csv(
                    independent_test_result, baseline_dir, "test"
                )
    return metadata, predictions

# Extracted from formula_discovery_pipeline.py (run_pysr_piecewise_baseline)
def run_pysr_piecewise_baseline(
    output_dir: Path, X: np.ndarray, y: np.ndarray,
    feature_names: Sequence[str], config: dict, seed: int,
    segmentation_config: dict, main_payload: dict,
    validation_data=None, test_data=None,
) -> dict:
    """Run separate PySR searches on the two sides of one shared split."""
    baseline_dir = output_dir / "pysr_baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "enabled": True,
        "fit_scope": "independent_per_piecewise_segment",
        "feature_names": list(feature_names),
        "segments": {},
    }
    minimum = int(segmentation_config.get("min_segment_samples", 8))
    split_candidates, split_prior = _restrict_split_indices(
        X, range(minimum, len(y) - minimum + 1), segmentation_config
    )
    if split_prior is not None and split_candidates:
        split_index = split_candidates[0]
    else:
        main_results = [
            *main_payload.get("piecewise_formula_pairs", []),
            *main_payload.get("single_best", []),
            *main_payload.get("combination_best", []),
        ]
        scored = []
        for row in main_results:
            try:
                index = int(row["split_index"])
                score = float(row.get("segmentation_score", row.get("loss")))
            except (KeyError, TypeError, ValueError):
                continue
            if minimum <= index <= len(y) - minimum and math.isfinite(score):
                scored.append((score, index))
        if scored:
            split_index = min(scored)[1]
        elif split_candidates:
            split_index = split_candidates[len(split_candidates) // 2]
        else:
            metadata.update(enabled=False, reason="no_legal_piecewise_split")
            (baseline_dir / "piecewise_segments.json").write_text(
                json.dumps(_json_value(metadata), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return metadata
    split_value = _split_value(X, split_index)
    metadata.update({
        "split_index": int(split_index),
        "split_value": float(split_value),
        "split_source": "physical_breakpoint_prior" if split_prior is not None else "main_model_or_median_legal_split",
        # Both independent PySR searches use the main-model seed. PySR requires
        # an unsigned 32-bit random_state; ordinary seeds remain unchanged.
        "main_model_seed": int(seed),
        "pysr_random_state": int(seed) % (2 ** 32),
    })
    segment_specs = (
        ("segment_1", slice(0, split_index)),
        ("segment_2", slice(split_index, len(y))),
    )
    for segment_index, (segment_name, train_slice) in enumerate(segment_specs):
        X_segment = np.asarray(X[train_slice])
        y_segment = np.asarray(y[train_slice])
        source_path = main_payload.get("metadata", {}).get("data_path") or "realdata_mat"
        group = {
            "name": segment_name,
            "path": f"{source_path}#{segment_name}",
            "start": 0, "end": len(y_segment), "samples": len(y_segment),
            "target": "path_loss",
        }
        validation_segment = None
        if validation_data is not None:
            validation_X, validation_y = validation_data[0], validation_data[1]
            mask = (validation_X[:, 0] <= split_value) if segment_index == 0 else (validation_X[:, 0] > split_value)
            if int(np.sum(mask)) >= 2:
                validation_segment = (
                    np.asarray(validation_X[mask]), np.asarray(validation_y[mask]),
                    [{**group, "start": 0, "end": int(np.sum(mask)), "samples": int(np.sum(mask))}],
                )
        test_segment = None
        if test_data is not None:
            test_X, test_y = test_data[0], test_data[1]
            mask = (test_X[:, 0] <= split_value) if segment_index == 0 else (test_X[:, 0] > split_value)
            if int(np.sum(mask)) >= 2:
                test_segment = (
                    np.asarray(test_X[mask]), np.asarray(test_y[mask]),
                    [{**group, "start": 0, "end": int(np.sum(mask)), "samples": int(np.sum(mask))}],
                )
        segment_metadata, _predictions = run_pysr_baseline(
            output_dir, X_segment, y_segment, feature_names, [group], config,
            int(seed) % (2 ** 32),
            validation_data=validation_segment, test_data=test_segment,
            baseline_output_dir=baseline_dir / segment_name,
        )
        segment_metadata["main_model_seed"] = int(seed)
        segment_metadata["pysr_random_state"] = int(seed) % (2 ** 32)
        metadata["segments"][segment_name] = segment_metadata
    metadata["enabled"] = any(
        item.get("enabled") for item in metadata["segments"].values()
    )
    metadata["metadata_json"] = str(baseline_dir / "pysr_baseline.json")
    (baseline_dir / "piecewise_segments.json").write_text(
        json.dumps(_json_value(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (baseline_dir / "pysr_baseline.json").write_text(
        json.dumps(_json_value(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata

# Extracted from formula_discovery_pipeline.py (run_pysr_piecewise_shared_template_baseline)
def run_pysr_piecewise_shared_template_baseline(
    output_dir: Path, X: np.ndarray, y: np.ndarray,
    feature_names: Sequence[str], config: dict, seed: int,
    segmentation_config: dict, main_payload: dict,
    validation_data=None, test_data=None,
) -> dict:
    """Fit one shared PySR formula structure with separate segment parameters."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    minimum = int(segmentation_config.get("min_segment_samples", 8))
    split_candidates, split_prior = _restrict_split_indices(
        X, range(minimum, len(y) - minimum + 1), segmentation_config
    )
    if split_prior is not None and split_candidates:
        split_index = split_candidates[0]
    else:
        main_results = [
            *main_payload.get("piecewise_formula_pairs", []),
            *main_payload.get("single_best", []),
            *main_payload.get("combination_best", []),
        ]
        scored = []
        for row in main_results:
            try:
                index = int(row["split_index"])
                score = float(row.get("segmentation_score", row.get("loss")))
            except (KeyError, TypeError, ValueError):
                continue
            if minimum <= index <= len(y) - minimum and math.isfinite(score):
                scored.append((score, index))
        if scored:
            split_index = min(scored)[1]
        elif split_candidates:
            split_index = split_candidates[len(split_candidates) // 2]
        else:
            baseline_dir = output_dir / "pysr_baseline"
            baseline_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "enabled": False, "reason": "no_legal_piecewise_split",
                "fit_scope": "joint_template_shared_structure_category_parameters",
                "segmentation_mode": "shared_breakpoint_two_segments",
            }
            metadata_path = baseline_dir / "pysr_baseline.json"
            metadata["metadata_json"] = str(metadata_path)
            metadata_path.write_text(json.dumps(
                _json_value(metadata), ensure_ascii=False, indent=2
            ), encoding="utf-8")
            return metadata

    split_value = _split_value(X, split_index)

    def split_groups(values_X, values_y, groups=None):
        values_X = np.asarray(values_X, dtype=float)
        values_y = np.asarray(values_y, dtype=float).reshape(-1)
        source_groups = groups or [{
            "name": "realdata_mat", "start": 0, "end": len(values_y),
        }]
        parts_X, parts_y, result_groups = [], [], []
        offset = 0
        for segment_index, segment_name in enumerate(("segment_1", "segment_2")):
            mask = ((values_X[:, 0] <= split_value) if segment_index == 0
                    else (values_X[:, 0] > split_value))
            for group in source_groups:
                start, end = int(group["start"]), int(group["end"])
                local_mask = mask[start:end]
                part_X, part_y = values_X[start:end][local_mask], values_y[start:end][local_mask]
                if not len(part_y):
                    continue
                parts_X.append(part_X)
                parts_y.append(part_y)
                result_groups.append({
                    "name": segment_name,
                    "path": f"{group.get('path', 'realdata_mat')}#{segment_name}",
                    "start": offset, "end": offset + len(part_y),
                    "samples": len(part_y), "target": group.get("target", "path_loss"),
                })
                offset += len(part_y)
        if not result_groups:
            return np.empty((0, values_X.shape[1])), np.empty(0), []
        # Keep one group for each physical segment, combining source groups if
        # a caller ever supplies more than one MAT-backed input group.
        merged_X, merged_y, merged_groups = [], [], []
        offset = 0
        for segment_name in ("segment_1", "segment_2"):
            indices = [i for i, group in enumerate(result_groups)
                       if group["name"] == segment_name]
            if not indices:
                continue
            segment_X = np.concatenate([parts_X[i] for i in indices], axis=0)
            segment_y = np.concatenate([parts_y[i] for i in indices], axis=0)
            merged_X.append(segment_X)
            merged_y.append(segment_y)
            first_group = result_groups[indices[0]]
            merged_groups.append({
                **first_group, "start": offset, "end": offset + len(segment_y),
                "samples": len(segment_y),
            })
            offset += len(segment_y)
        return np.concatenate(merged_X), np.concatenate(merged_y), merged_groups

    train_X, train_y, train_groups = split_groups(X, y)
    segmented_validation = None
    if validation_data is not None:
        val_X, val_y = validation_data[:2]
        val_groups = validation_data[3] if len(validation_data) > 3 else None
        segmented_validation = (*split_groups(val_X, val_y, val_groups),)
        segmented_validation = (segmented_validation[0], segmented_validation[1],
                                segmented_validation[2])
    segmented_test = None
    if test_data is not None:
        test_X, test_y = test_data[:2]
        test_groups = test_data[3] if len(test_data) > 3 else None
        segmented_test = split_groups(test_X, test_y, test_groups)

    pysr_config = copy.deepcopy(config)
    pysr_config.setdefault("pysr_baseline", {})["enabled"] = True
    metadata, _predictions = run_pysr_baseline(
        output_dir, train_X, train_y, feature_names, train_groups,
        pysr_config, int(seed) % (2 ** 32),
        validation_data=segmented_validation, test_data=segmented_test,
    )
    metadata.update({
        "segmentation_mode": "shared_breakpoint_two_segments",
        "split_index": int(split_index), "split_value": float(split_value),
        "split_source": ("physical_breakpoint_prior" if split_prior is not None
                         else "main_model_or_median_legal_split"),
        "main_model_seed": int(seed), "pysr_random_state": int(seed) % (2 ** 32),
        "segment_names": [group["name"] for group in train_groups],
    })
    metadata_path = output_dir / "pysr_baseline" / "pysr_baseline.json"
    metadata["metadata_json"] = str(metadata_path)
    metadata_path.write_text(json.dumps(
        _json_value(metadata), ensure_ascii=False, indent=2
    ), encoding="utf-8")
    return metadata

# Extracted from formula_discovery_pipeline.py (_pysr_hv_window_front)
def _pysr_hv_window_front(
    rows: Sequence[dict], data_mode: str, evaluation_label: str,
    reference_rows: Sequence[dict] | None = None,
    r2_window: float = DEFAULT_PYSR_R2_WINDOW,
) -> list[dict]:
    """Return the PySR rows eligible for HV/plots under Mode 2's R² window.

    This deliberately runs *after* PySR has completed its native search and
    does not alter either its Hall-of-Fame evolution or its saved selected
    front.  It is solely the reporting counterpart of the Mode 2/3 admission
    rule: retain candidates whose active primary R² is within the configured
    fairness window of the best
    candidate available in that PySR snapshot.  When a reference front is
    supplied, it uses the previous-round best exactly like Mode 2/3 candidate
    admission; otherwise the current rows form the reference set.
    """
    primary = (
        f"{evaluation_label}_pooled_r2" if _is_measured_pathloss_profile(data_mode)
        else f"{evaluation_label}_mean_r2"
    )
    valid = []
    for row in rows:
        try:
            value = float(row[primary])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            valid.append(row)
    if not valid:
        return []
    reference = reference_rows if reference_rows is not None else valid
    reference_valid = []
    for row in reference:
        try:
            value = float(row[primary])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            reference_valid.append(row)
    if not reference_valid:
        reference_valid = valid
    best = max(float(row[primary]) for row in reference_valid)
    cutoff = best - max(0.0, float(r2_window))
    return [row for row in valid if float(row[primary]) >= cutoff]

# Extracted from formula_discovery_pipeline.py (_pysr_front_for_hv_reporting)
def _pysr_front_for_hv_reporting(
    rows: Sequence[dict], data_mode: str, evaluation_label: str,
) -> list[dict]:
    """Return PySR rows allowed to enter a reported HV calculation.

    Validation is the selection split. Test evaluates the frozen validation-selected
    front as-is and must never be re-filtered by test R2.
    """
    if str(evaluation_label) == "test":
        return list(rows or [])
    return _pysr_hv_window_front(rows, data_mode, evaluation_label)

# Extracted from formula_discovery_pipeline.py (_pysr_cache_signature)
def _pysr_cache_signature(config: dict) -> str:
    """Return a stable cache namespace for data/PySR settings.

    The seed is deliberately excluded: each seed gets its own child directory
    inside this namespace, while changing the data or PySR configuration starts
    a separate cache and cannot silently reuse incompatible results.
    """
    payload = {
        "data": config.get("data", {}),
        "pysr": config.get("pysr", config.get("pysr_baseline", {})),
        "target": config.get("target", {}),
    }
    encoded = json.dumps(_json_value(payload), ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]

# Extracted from formula_discovery_pipeline.py (_pysr_cache_seed_is_valid)
def _pysr_cache_seed_is_valid(cache_seed_dir: Path) -> bool:
    """Check that a cached seed contains a readable shared validation result."""
    candidates = [
        cache_seed_dir / "pysr_baseline" / "shared_refit" / "validation_results.json",
        cache_seed_dir / "pysr_baseline" / "validation_results.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("final_pareto_front"), list):
            return True
    for path in (
        cache_seed_dir / "pysr_baseline" / "shared_refit" / "validation_pareto_front.csv",
        cache_seed_dir / "pysr_baseline" / "validation_pareto_front.csv",
    ):
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                if next(csv.DictReader(handle), None) is not None:
                    return True
        except (OSError, csv.Error):
            continue
    return False

# Extracted from formula_discovery_pipeline.py (_copy_pysr_cache_to_output)
def _copy_pysr_cache_to_output(cache_seed_dir: Path, output_dir: Path) -> None:
    source = cache_seed_dir / "pysr_baseline"
    target = output_dir / "pysr_baseline"
    if not source.is_dir():
        raise FileNotFoundError(f"PySR cache directory does not exist: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)

# Extracted from formula_discovery_pipeline.py (_save_pysr_output_to_cache)
def _save_pysr_output_to_cache(output_dir: Path, cache_seed_dir: Path) -> None:
    source = output_dir / "pysr_baseline"
    if not source.is_dir():
        logging.warning("PySR output directory does not exist; cannot cache: %s", source)
        return
    cache_seed_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, cache_seed_dir / "pysr_baseline", dirs_exist_ok=True)

# Extracted from formula_discovery_pipeline.py (_load_pysr_evaluation_result)
def _load_pysr_evaluation_result(output_dir: Path, evaluation_label: str) -> dict | None:
    """Load a saved PySR front used by the pipeline comparison."""
    paths = [
        output_dir / "pysr_baseline" / "shared_refit"
        / f"{evaluation_label}_results.json",
        output_dir / "pysr_baseline" / f"{evaluation_label}_results.json",
    ]
    for path in paths:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        if metadata.get("experiment_variant") in {
            "pysr_shared_fixed_structure_bfgs", "pysr_shared_template",
            "pysr_baseline",
        }:
            return payload
    # Result JSONs are intentionally omitted; the final front CSV carries the
    # frozen candidates needed by downstream test evaluation and summaries.
    for result_dir in (
        output_dir / "pysr_baseline" / "shared_refit",
        output_dir / "pysr_baseline",
    ):
        front_path = result_dir / f"{evaluation_label}_pareto_front.csv"
        if not front_path.is_file():
            continue
        try:
            with front_path.open(encoding="utf-8-sig", newline="") as handle:
                front = list(csv.DictReader(handle))
        except (OSError, csv.Error):
            continue
        metadata = {}
        for metadata_path in (
            output_dir / "pysr_baseline" / "pysr_baseline.json",
            result_dir / "pysr_baseline.json",
        ):
            if not metadata_path.is_file():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = {}
            break
        return {
            "metadata": metadata,
            "final_pareto_front": front,
            "history": [{"iteration": 0, "front": front}],
            "per_iteration": [],
        }
    return None

# Extracted from formula_discovery_pipeline.py (_attach_pysr_search_history)
def _attach_pysr_search_history(
    result: dict | None, pysr_metadata: dict | None,
    X: np.ndarray, y: np.ndarray, groups: Sequence[dict],
    feature_names: Sequence[str], evaluation_label: str,
    data_mode: str = "csv",
) -> dict | None:
    """Evaluate native PySR Hall-of-Fame snapshots on a frozen split.

    The snapshots are observational outputs from PySR's logger.  The PySR
    search itself is not restarted or chunked, so this does not alter its
    iteration behavior.
    """
    if not result or not pysr_metadata:
        return result
    progress_rows = []
    for dataset in pysr_metadata.get("datasets", []):
        progress_rows.extend(dataset.get("search_progress", []) or [])
    if not progress_rows or not groups:
        return result

    local_symbols = {str(name): sp.Symbol(str(name))
                     for name in feature_names}
    local_symbols.update({f"x{index}": sp.Symbol(str(name))
                          for index, name in enumerate(feature_names)})
    local_symbols.update(_pysr_extra_sympy_mappings())
    by_step: dict[int, dict[str, dict]] = {}
    for item in progress_rows:
        equation = str(item.get("equation", "")).strip()
        if not equation:
            continue
        key = (int(item.get("step", 0)), equation)
        by_step.setdefault(key[0], {})[equation] = item

    history = []
    prefix = str(evaluation_label)
    previous_window_front: list[dict] = []
    for step in sorted(by_step):
        candidates = []
        for equation, item in by_step[step].items():
            try:
                expression = sp.sympify(equation, locals=local_symbols)
                function = sp.lambdify(
                    [local_symbols[str(name)] for name in feature_names],
                    expression, modules=_pysr_lambdify_modules(),
                )
                values, targets = [], []
                dataset_r2 = []
                for group in groups:
                    start, end = int(group["start"]), int(group["end"])
                    prediction = np.asarray(
                        function(*np.asarray(X[start:end], dtype=float).T),
                        dtype=float,
                    ).reshape(-1)
                    target = np.asarray(y[start:end], dtype=float).reshape(-1)
                    # SymPy returns a scalar array for constant PySR
                    # expressions.  Treat it as a constant prediction over
                    # the whole dataset; reject any other shape mismatch so
                    # one malformed snapshot cannot abort the run.
                    if prediction.size == 1 and target.size != 1:
                        prediction = np.full(target.shape, prediction.item(), dtype=float)
                    if prediction.size != target.size:
                        continue
                    valid = np.isfinite(prediction) & np.isfinite(target)
                    if np.sum(valid) < 2:
                        continue
                    residual = target[valid] - prediction[valid]
                    centered = target[valid] - np.mean(target[valid])
                    total = float(np.sum(centered ** 2))
                    if total <= 0:
                        continue
                    dataset_r2.append(1.0 - float(np.sum(residual ** 2)) / total)
                if not dataset_r2:
                    continue
                complexity = int(item.get("complexity"))
                candidates.append({
                    "formula_id": f"pysr_progress_{step}_{len(candidates)}",
                    "readable_expression": equation,
                    f"{prefix}_mean_r2": float(np.mean(dataset_r2)),
                    f"{prefix}_r2_std": float(np.std(dataset_r2)),
                    "complexity": complexity,
                    f"{prefix}_complexity": complexity,
                    "formula_family": "pysr_progress",
                })
            except (ArithmeticError, TypeError, ValueError, KeyError,
                    sp.SympifyError, OverflowError):
                continue
        if not candidates:
            continue
        # This mirrors Mode 2/3's previous-front R² admission window, but is
        # calculated from logger snapshots only.  PySR has already evolved
        # independently; neither its populations nor its Hall of Fame are
        # mutated here.
        candidates = _pysr_hv_window_front(
            candidates, data_mode, prefix,
            reference_rows=previous_window_front,
        )
        if not candidates:
            history.append({"iteration": int(step), "front": []})
            continue
        primary = f"{prefix}_mean_r2"
        spread = f"{prefix}_r2_std"
        front = []
        for candidate in candidates:
            dominated = False
            for other in candidates:
                no_worse = (
                    other[primary] >= candidate[primary]
                    and other[spread] <= candidate[spread]
                    and other["complexity"] <= candidate["complexity"]
                )
                strict = (
                    other[primary] > candidate[primary]
                    or other[spread] < candidate[spread]
                    or other["complexity"] < candidate["complexity"]
                )
                if no_worse and strict:
                    dominated = True
                    break
            if not dominated:
                front.append(candidate)
        history.append({"iteration": int(step), "front": front})
        previous_window_front = front
    if len(history) >= 1:
        result["history"] = history
        result["per_iteration"] = []
        result.setdefault("metadata", {})["search_history_source"] = (
            "pysr_native_hall_of_fame_logger"
        )
        result["metadata"]["search_history_is_frozen_evaluation"] = True
        result["metadata"]["native_progress_snapshot_count"] = len(by_step)
        result["metadata"]["eligible_progress_snapshot_count"] = len(history)
        result["metadata"]["hv_reporting_window"] = {
            "enabled": True,
            "primary_r2_gap": DEFAULT_PYSR_R2_WINDOW,
            "reference": "previous_snapshot_front",
            "affects_pysr_evolution": False,
        }
    return result

def run_pysr_stage(
    main_output_dir: Path, pysr_output_directory: str | Path | None,
    X: np.ndarray, y: np.ndarray, feature_names: Sequence[str],
    dataset_groups: Sequence[dict], config: dict, seed: int,
    data_cfg: dict, is_realdata_mat: bool, segmentation_enabled: bool,
    segmentation_config: dict, payload: dict, pipeline_mode: int,
    validation_data, test_data, validation_X: np.ndarray,
    validation_y: np.ndarray, validation_groups: Sequence[dict],
) -> dict:
    """Run and persist the PySR baseline in its own output directory."""
    output_dir = (Path(pysr_output_directory) if pysr_output_directory
                  else Path(main_output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    independent_segment_child = bool(
        is_realdata_mat
        and (data_cfg.get("realdata", {}) or {}).get("independent_segment_run", False)
    )
    # PySR runs before the tree search and supplies the online HV baseline.
    pysr_validation_result = None
    pysr_test_result = None
    pysr_metadata = {}
    pysr_cache_dir = None
    cache_reused = False
    # realdata_mat uses the multi-dataset shared-template path, with the two
    # physical segments treated as datasets that share one formula structure.
    pysr_supported_data_mode = (
        int(pipeline_mode) in (3, 4)
        and data_cfg.get("mode", "csv") in ("csv", "realdata_mat")
    )
    independent_segment_child = bool(
        is_realdata_mat
        and (data_cfg.get("realdata", {}) or {}).get("independent_segment_run", False)
    )
    pysr_enabled = bool(
        (config.get("pysr_baseline", {}) or {}).get("enabled", False)
    )
    if pysr_supported_data_mode and not pysr_enabled:
        pysr_metadata = {
            "enabled": False,
            "reason": "disabled",
            "data_mode": data_cfg.get("mode", "csv"),
        }
        baseline_dir = output_dir / "pysr_baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = baseline_dir / "pysr_baseline.json"
        pysr_metadata["metadata_json"] = str(metadata_path)
        metadata_path.write_text(
            json.dumps(_json_value(pysr_metadata), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        payload["metadata"]["pysr_baseline"] = pysr_metadata
        print("pysr_baseline.enabled=false, skipping PySR.", flush=True)
    elif (pysr_supported_data_mode and is_realdata_mat
            and segmentation_enabled and not independent_segment_child
            and len(dataset_groups) == 1):
        pysr_metadata = run_pysr_piecewise_shared_template_baseline(
            output_dir, X, y, feature_names, config, seed,
            segmentation_config, payload,
            validation_data=validation_data,
            test_data=test_data,
        )
        payload["metadata"]["pysr_baseline"] = pysr_metadata
        pysr_validation_result = _load_pysr_evaluation_result(
            output_dir, "validation"
        )
        pysr_test_result = _load_pysr_evaluation_result(output_dir, "test")
        pysr_result_dir = output_dir / "pysr_baseline"
        print(
            f"PySR seed={seed}: fit both measured-data segments using the shared formula structure and breakpoint; results at {pysr_result_dir}",
            flush=True,
        )
    elif pysr_supported_data_mode:
        pysr_cache_cfg = config.get("pysr_cache", {}) or {}
        pysr_cache_dir = Path(str(pysr_cache_cfg.get("directory", ""))) \
            if pysr_cache_cfg.get("directory") else None
        cache_reused = bool(
            len(dataset_groups) == 1
            and
            pysr_cache_cfg.get("reuse", False)
            and pysr_cache_dir is not None
            and _pysr_cache_seed_is_valid(pysr_cache_dir)
        )
        if cache_reused:
            _copy_pysr_cache_to_output(pysr_cache_dir, output_dir)
            copied_baseline = output_dir / "pysr_baseline"
            for obsolete_name in (
                "validation_results.json", "independent_validation_results.json",
            ):
                for obsolete_path in copied_baseline.rglob(obsolete_name):
                    if obsolete_path.is_file():
                        obsolete_path.unlink()
            metadata_path = output_dir / "pysr_baseline" / "pysr_baseline.json"
            if metadata_path.is_file():
                try:
                    loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if isinstance(loaded_metadata, dict):
                        pysr_metadata = loaded_metadata
                        # The cached metadata points to the original run;
                        # rewrite it to this experiment's copied artifact.
                        pysr_metadata["metadata_json"] = str(metadata_path)
                except (OSError, json.JSONDecodeError):
                    logging.warning("Could not read reused PySR metadata: %s", metadata_path)
            # A cache may have been created before the canonical PySR front
            # retained every window-eligible native equation.  Rebuild that
            # derived validation artifact from the cached equations CSV.
            if (pysr_metadata.get("fit_scope") !=
                    "joint_template_shared_structure_category_parameters"
                    and validation_data is not None and pysr_metadata.get("datasets")):
                rebuilt_validation = _pysr_independent_validation_result(
                    pysr_metadata["datasets"], config.get("pysr_baseline", {}),
                    validation_data, feature_names,
                    data_mode=_configured_data_profile(config.get("data", {})),
                )
                _save_pysr_pareto_front_csv(
                    rebuilt_validation, output_dir / "pysr_baseline", "validation"
                )
            print(f"PySR seed={seed}: reusing cached results ({pysr_cache_dir})", flush=True)
        else:
            pysr_metadata, _pysr_predictions = run_pysr_baseline(
                output_dir, X, y, feature_names, dataset_groups, config, seed,
                validation_data=(validation_X, validation_y, validation_groups),
                test_data=(test_data[0], test_data[1], test_data[3]),
            )
            if (len(dataset_groups) == 1 and pysr_cache_dir is not None
                    and pysr_cache_cfg.get("save", True)):
                _save_pysr_output_to_cache(output_dir, pysr_cache_dir)
        pysr_validation_result = _load_pysr_evaluation_result(
            output_dir, "validation"
        )
        pysr_test_result = _load_pysr_evaluation_result(output_dir, "test")
        if (validation_data is not None and pysr_metadata.get("fit_scope") !=
                "joint_template_shared_structure_category_parameters"):
            pysr_validation_result = _attach_pysr_search_history(
                pysr_validation_result, pysr_metadata,
                validation_data[0], validation_data[1], validation_data[3],
                feature_names, "validation", _configured_data_profile(data_cfg),
            )
        if (test_data is not None and pysr_metadata.get("fit_scope") !=
                "joint_template_shared_structure_category_parameters"):
            if pysr_test_result is not None:
                pysr_test_result = _attach_pysr_search_history(
                    pysr_test_result, pysr_metadata,
                    test_data[0], test_data[1], test_data[3],
                    feature_names, "test", _configured_data_profile(data_cfg),
                )
            elif pysr_validation_result is not None:
                # Independent PySR search deliberately leaves test sealed.
                # Evaluate its already validation-selected equations here,
                # without refitting constants or selecting on test metrics.
                pysr_test_result = _evaluate_frozen_pysr_front_on_test(
                    pysr_validation_result,
                    (test_data[0], test_data[1], test_data[3]), feature_names,
                )
        for evaluation_label, evaluation_result in (
            ("validation", pysr_validation_result),
            ("test", pysr_test_result),
        ):
            if not evaluation_result:
                continue
            shared_result_path = (
                output_dir / "pysr_baseline" / "shared_refit"
                / f"{evaluation_label}_results.json"
            )
            result_path = (
                shared_result_path if shared_result_path.is_file()
                else output_dir / "pysr_baseline" / f"{evaluation_label}_results.json"
            )
            result_path.parent.mkdir(parents=True, exist_ok=True)
            if evaluation_label in {"test", "validation"}:
                if result_path.is_file():
                    result_path.unlink()
            else:
                result_path.write_text(
                    json.dumps(_json_value(evaluation_result),
                               ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            _save_pysr_pareto_front_csv(
                evaluation_result, result_path.parent, evaluation_label
            )
        plot_paths = []
        if ENABLE_PYSR_VISUAL_COMPARISON:
            plot_paths = save_formula_mode_gp_plots(
                output_dir, payload.get("gp_pareto_front", []), X, y,
                dataset_groups, config["fitting"], baseline_predictions=None,
            )
        payload["metadata"]["formula_mode_visualization"] = {
            "enabled": bool(plot_paths),
            "plot_count": len(plot_paths),
            "plots": plot_paths,
            "directory": str(output_dir / "formula_mode_plots"),
            "colors": {"actual_data": "tab:blue", "gp_formula": "tab:orange"},
            "pysr_overlay": False,
            "pysr_comparison_disabled": not ENABLE_PYSR_VISUAL_COMPARISON,
        }
        pysr_metadata["visualization"] = {
            "enabled": bool(plot_paths),
            "plots": plot_paths,
            "directory": str(output_dir / "formula_mode_plots"),
            "overlay": "same_gp_formula_figures",
            "comparison_disabled": not ENABLE_PYSR_VISUAL_COMPARISON,
        }
        pysr_metadata["cache_reused"] = bool(cache_reused)
        payload["metadata"]["pysr_baseline"] = pysr_metadata
        metadata_path = pysr_metadata.get("metadata_json")
        if metadata_path:
            Path(metadata_path).write_text(
                json.dumps(_json_value(pysr_metadata), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    elif pysr_supported_data_mode:
        pysr_metadata = {
            "enabled": False,
            "reason": "skipped_multiple_dataset_files",
            "dataset_count": len(dataset_groups),
        }
        payload["metadata"]["pysr_baseline"] = pysr_metadata
        print(
            f"PySR seed={seed}: found {len(dataset_groups)} input files, skipping PySR.",
            flush=True,
        )
    elif is_realdata_mat:
        pysr_metadata = {
            "enabled": False,
            "reason": "disabled_for_measured_pathloss_profile",
            "data_mode": data_cfg.get("mode", "csv"),
            "data_profile": _configured_data_profile(data_cfg),
        }
        payload["metadata"]["pysr_baseline"] = pysr_metadata
        print("Measured path-loss profile: skipping PySR under current settings.", flush=True)
    pysr_hv_monitor_bounds = None
    pysr_hv_baseline = None
    hv_cache_path = (
        pysr_cache_dir / "pysr_hv_parameters.json"
        if pysr_cache_dir is not None else None
    )
    hv_cache_loaded = False
    if (cache_reused and pysr_validation_result is not None
            and hv_cache_path is not None and hv_cache_path.is_file()):
        try:
            cached_hv = json.loads(hv_cache_path.read_text(encoding="utf-8"))
            cached_baseline = cached_hv.get("pysr_hv_baseline")
            cached_bounds = cached_hv.get("pysr_hv_monitor_bounds")
            if (cached_baseline is not None and isinstance(cached_bounds, dict)
                    and cached_bounds):
                pysr_hv_baseline = float(cached_baseline)
                pysr_hv_monitor_bounds = cached_bounds
                hv_cache_loaded = True
                print(f"PySR seed={seed}: reusing cached HV parameters ({hv_cache_path})", flush=True)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            logging.warning("PySR HV cache is invalid; recomputing: %s", hv_cache_path)

    if pysr_validation_result is not None:
        profile = _configured_data_profile(data_cfg)
        monitor_fields, monitor_maximize = _validation_hv_objectives(
            profile, "validation")
        monitor_front = _pysr_hv_window_front(
            pysr_validation_result.get("final_pareto_front", []),
            profile, "validation")
        # This snapshot is the immutable PySR quality reference for all
        # subsequent Mode2/Mode5 admission windows.
        if not hv_cache_loaded:
            monitor_rows = list(monitor_front)
            for item in pysr_validation_result.get("history", []) or []:
                monitor_rows.extend(_pysr_hv_window_front(
                    item.get("front", []), profile, "validation"))
            candidate_bounds = _validation_hv_bounds(monitor_rows, monitor_fields)
            if len(candidate_bounds) == len(monitor_fields) and monitor_front:
                pysr_hv_monitor_bounds = candidate_bounds
                pysr_hv_baseline = _calculate_validation_hv(
                    monitor_front, monitor_fields, monitor_maximize, candidate_bounds)
            else:
                logging.warning("PySR validation-set HV monitoring unavailable: complete normalization bounds are missing")
        if pysr_hv_baseline is not None:
            print(f"PySR validation baseline HV (fixed online normalization scale)={pysr_hv_baseline:.8g}", flush=True)

    if (hv_cache_path is not None and pysr_hv_baseline is not None
            and isinstance(pysr_hv_monitor_bounds, dict)):
        try:
            hv_cache_path.parent.mkdir(parents=True, exist_ok=True)
            hv_cache_path.write_text(json.dumps(_json_value({
                "pysr_hv_baseline": pysr_hv_baseline,
                "pysr_hv_monitor_bounds": pysr_hv_monitor_bounds,
                "source": "pysr_validation_shared_refit",
            }), ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            logging.warning("Could not save PySR HV parameter cache: %s", exc)

    return {
        "validation_result": pysr_validation_result,
        "test_result": pysr_test_result,
        "metadata": pysr_metadata,
        "cache_dir": pysr_cache_dir,
        "cache_reused": cache_reused,
        "hv_monitor_bounds": pysr_hv_monitor_bounds,
        "hv_baseline": pysr_hv_baseline,
        "hv_cache_path": hv_cache_path,
    }
