#!/usr/bin/env python3
"""Find repeated Main-model and PySR symbolic structures across experiment seeds."""
import csv
import json
import re
import sys
import math
import statistics
from collections import defaultdict
from pathlib import Path

import sympy as sp

# Standalone default; the pipeline passes its newly completed experiment path.
EXPERIMENT_DIR = Path('/home/lwj/另一种NN模型/artifacts/expri2/mode5_experiment_059')
ROOT = EXPERIMENT_DIR
OUT = ROOT / 'cross_seed_formula_summary' / 'main'
# ``pN`` is emitted by the independent realdata_mat branch for some fitted
# prefactors.  Treat it exactly like the older ``aN``/``cN`` parameter names;
# otherwise it is incorrectly retained as a structural symbol.
PARAMETER_RE = re.compile(
    r'^(?:a\d+|c\d+(?:_(?:left|right))?|p\d+'
    r'|cconst_product_\d+(?:_(?:left|right))?'
    r'|cconst_output_\d+(?:_(?:left|right))?'
    r'|__canon_const_\d+)$'
)
PYSR_NUMBER_RE = re.compile(
    r'(?<![A-Za-z0-9_.])(?:\d+(?:\.\d*)?|\.\d+)'
    r'(?:[eE][+-]?\d+)?(?![A-Za-z0-9_.])'
)
csv.field_size_limit(sys.maxsize)
_CANON_CONST_COUNTER = 0


def is_parameter(node):
    if isinstance(node, sp.Symbol):
        return bool(PARAMETER_RE.fullmatch(str(node)))
    # All floating-point literals are treated like fitted parameters. Equal
    # literal values share an identity within an expression; integer literals
    # remain fixed structural tokens.
    return isinstance(node, sp.Float)


def first_present(*values):
    """Return the first non-empty metric, preserving valid zero values."""
    return next((value for value in values if value not in (None, '')), None)


def parameter_identity(node):
    if isinstance(node, sp.Symbol):
        return f'SYMBOL:{node}'
    return f'FLOAT:{sp.sstr(node)}'


def merge_parameter_products(expr):
    """Merge multiple direct fitted-parameter factors into one free constant."""
    if not expr.args:
        return expr
    args = tuple(merge_parameter_products(arg) for arg in expr.args)
    if expr.func is sp.Mul:
        parameter_factors = [arg for arg in args if is_parameter(arg)]
        if len(parameter_factors) >= 2:
            merged = _new_canonical_constant(
                sp.Mul(*parameter_factors, evaluate=False))
            args = tuple(arg for arg in args if not is_parameter(arg)) + (merged,)
        return sp.Mul(*args, evaluate=False)
    try:
        return expr.func(*args)
    except (TypeError, ValueError):
        return expr


def build_sorted_tree(node):
    """Build a tree sorted at every node; a/c parameter names are ignored for sorting."""
    if is_parameter(node):
        return ('PARAM', parameter_identity(node))
    if isinstance(node, sp.Symbol):
        return ('SYMBOL', str(node))
    if node.is_Number:
        return ('NUMBER', str(node))
    args = list(node.args)
    # The sign of a product containing a free fitted parameter is absorbed
    # into that parameter (it may be positive or negative), so +/- variants
    # share one structural representation. Signs without such a parameter stay.
    # A standalone -1 is removed after being absorbed; retaining Abs(-1)=1
    # would create redundant output such as log(Abs(C1*x1))*1*C2.
    if node.func is sp.Mul and any(is_parameter(arg) for arg in args):
        cleaned_args = []
        for arg in args:
            if arg.is_Number and arg.is_negative:
                if arg == -1:
                    continue
                arg = sp.Abs(arg)
            cleaned_args.append(arg)
        args = cleaned_args
    # Sign normalization above can remove every factor except one parameter.
    # Do not preserve the resulting unevaluated Mul(parameter) wrapper: it is
    # the same structure as the parameter itself and otherwise causes
    # equivalent formulas from different seeds to receive different keys.
    if node.func is sp.Mul and len(args) == 1:
        return build_sorted_tree(args[0])
    children = [build_sorted_tree(arg) for arg in args]
    # Only commutative operators reorder operands. Powers and functions keep
    # their semantic argument order. Parameter identities are restored later
    # by left-to-right alpha-renaming.
    if node.func in (sp.Add, sp.Mul):
        children.sort(key=lambda child: json.dumps(
            sort_signature(child), ensure_ascii=False, separators=(',', ':')))
    return ('NODE', node.func.__name__, tuple(children))


def sort_signature(tree):
    kind = tree[0]
    if kind == 'PARAM':
        return ('PARAM',)
    if kind in ('SYMBOL', 'NUMBER'):
        return tree
    return ('NODE', tree[1], tuple(sort_signature(child) for child in tree[2]))



def _is_variable_symbol(symbol):
    return isinstance(symbol, sp.Symbol) and not is_parameter(symbol)


def _is_constant_expression(expr):
    return not any(_is_variable_symbol(symbol) for symbol in expr.free_symbols)


def normalize_log_quotients(expr):
    """Rewrite a constant-over-expression log as one signed log of a product."""
    if not expr.args:
        return expr
    args = tuple(normalize_log_quotients(arg) for arg in expr.args)
    try:
        rebuilt = expr.func(*args)
    except (TypeError, ValueError):
        rebuilt = expr
    if rebuilt.func is not sp.log or len(rebuilt.args) != 1:
        return rebuilt
    absolute = rebuilt.args[0]
    if absolute.func is not sp.Abs or len(absolute.args) != 1:
        return rebuilt
    inside = absolute.args[0]
    denominator, numerators = [], []
    for factor in sp.Mul.make_args(inside):
        if (factor.func is sp.Pow and len(factor.args) == 2
                and factor.args[1].is_number and bool(factor.args[1] < 0)):
            denominator.append(sp.Pow(factor.args[0], -factor.args[1], evaluate=False))
        else:
            numerators.append(factor)
    if not denominator:
        return rebuilt
    numerator = sp.Mul(*numerators, evaluate=False) if numerators else sp.Integer(1)
    # log|c/expr| = -log|expr/c|. Fold the reciprocal constant into C and
    # preserve one log node rather than expanding into a sum of logs.
    if not _is_constant_expression(numerator):
        return rebuilt
    if numerator == 1:
        # Avoid inventing a free multiplier for the exact unit numerator.
        positive_expression = sp.Mul(*denominator, evaluate=False)
    else:
        reciprocal = _new_canonical_constant(sp.Pow(numerator, -1, evaluate=False))
        positive_expression = sp.Mul(reciprocal, *denominator, evaluate=False)
    return sp.Mul(-1, sp.log(sp.Abs(positive_expression, evaluate=False), evaluate=False), evaluate=False)


def _new_canonical_constant(_expression):
    """Represent a derived constant factor with the same anonymous C syntax."""
    global _CANON_CONST_COUNTER
    _CANON_CONST_COUNTER += 1
    return sp.Symbol(f'__canon_const_{_CANON_CONST_COUNTER}')


def prepare_pysr_expression(text):
    """Extract a PySR formula and make every numeric literal a free parameter."""
    expression = str(text or '').strip()
    # Shared-template exports serialize parameter vectors after the formula:
    # ``f = <expression>; p1 = [...]; ...``. Only the f expression defines
    # symbolic structure; the fitted vectors are metadata, not formula terms.
    expression = expression.split(';', 1)[0].strip()
    if '=' in expression:
        left, right = expression.split('=', 1)
        if left.strip() in {'f', 'y', 'output'}:
            expression = right.strip()

    # Shared-template parameters are vectors indexed by dataset_id in the
    # fitting implementation. For structure comparison, treat each referenced
    # vector slot as an ordinary free parameter (p1, p2, ...); the index is not
    # part of the symbolic formula structure.
    expression = re.sub(
        r'\b(p\d+)\s*\[\s*dataset_id\s*\]', r'\1', expression
    )

    def replace_number(_match):
        return str(_new_canonical_constant(_match.group(0)))

    return PYSR_NUMBER_RE.sub(replace_number, expression)


def pysr_sympy_locals(expression):
    """Provide PySR's protected-log aliases and indexed template parameters."""
    names = set(re.findall(
        r'(?<![A-Za-z0-9_])(?:a\d+|c\d+(?:_(?:left|right))?|p\d+'
        r'|cconst_product_\d+(?:_(?:left|right))?'
        r'|cconst_output_\d+(?:_(?:left|right))?|__canon_const_\d+)'
        r'(?![A-Za-z0-9_])', expression,
    ))
    local_symbols = {
        name: (sp.IndexedBase(name) if re.fullmatch(r'p\d+', name)
               and f'{name}[' in expression else sp.Symbol(name, real=True))
        for name in names
    }

    def log_abs(value):
        return sp.log(sp.Abs(value, evaluate=False), evaluate=False)

    def log_abs_sin(value):
        return log_abs(sp.sin(value, evaluate=False))

    def log_abs_cos(value):
        return log_abs(sp.cos(value, evaluate=False))

    local_symbols.update({
        'log_abs': log_abs,
        'log_abs_sin': log_abs_sin,
        'log_abs_cos': log_abs_cos,
    })
    return local_symbols


def _extract_abs_log_term(term):
    factors = list(sp.Mul.make_args(term))
    logs = [factor for factor in factors
            if factor.func is sp.log and len(factor.args) == 1
            and factor.args[0].func is sp.Abs
            and len(factor.args[0].args) == 1]
    if len(logs) != 1:
        return None
    log_term = logs[0]
    coefficient_factors = list(factors)
    coefficient_factors.remove(log_term)
    coefficient = sp.Mul(*coefficient_factors)
    argument = log_term.args[0].args[0]
    return coefficient, argument


def absorb_additive_constants_into_log(expr):
    """Absorb standalone additive constants into a deterministic log term."""
    if not expr.args:
        return expr
    args = tuple(absorb_additive_constants_into_log(arg) for arg in expr.args)
    try:
        rebuilt = expr.func(*args)
    except (TypeError, ValueError):
        rebuilt = expr
    if rebuilt.func is not sp.Add:
        return rebuilt
    terms = list(sp.Add.make_args(rebuilt))
    constants = [term for term in terms if _is_constant_expression(term)]
    if not constants:
        return rebuilt
    log_candidates = []
    for term in terms:
        if term in constants:
            continue
        extracted = _extract_abs_log_term(term)
        if extracted is None:
            continue
        _coefficient, argument = extracted
        if any(_is_variable_symbol(symbol) for symbol in argument.free_symbols):
            tree = build_sorted_tree(term)
            key = json.dumps(sort_signature(tree), ensure_ascii=False,
                             separators=(',', ':'))
            log_candidates.append((key, term, extracted))
    if not log_candidates:
        return rebuilt
    _, selected_term, (coefficient, argument) = min(log_candidates, key=lambda item: item[0])
    if coefficient == 0:
        return rebuilt
    constant_sum = sp.Add(*constants)
    if constant_sum == 0:
        return rebuilt
    if coefficient == 1:
        ratio = constant_sum
    elif coefficient == -1:
        ratio = -constant_sum
    else:
        ratio = sp.Mul(constant_sum, sp.Pow(coefficient, -1, evaluate=False),
                        evaluate=False)
    # If f already has a constant multiplier, leave the offset untouched:
    # that multiplier already provides the desired canonical C*f form.
    argument_factors = sp.Mul.make_args(argument)
    if any(_is_constant_expression(factor) for factor in argument_factors):
        return rebuilt
    exp_factor = _new_canonical_constant(sp.exp(ratio, evaluate=False))
    new_argument = sp.Mul(exp_factor, argument, evaluate=False)
    new_log = sp.log(sp.Abs(new_argument, evaluate=False), evaluate=False)
    new_term = sp.Mul(coefficient, new_log, evaluate=False)
    remaining = [term for term in terms if term not in constants and term != selected_term]
    return sp.Add(*remaining, new_term, evaluate=False)

def alpha_rename_tree(tree):
    names = {}

    def visit(item):
        kind = item[0]
        if kind == 'PARAM':
            original = item[1]
            if original not in names:
                names[original] = f'C{len(names) + 1}'
            return ('SYMBOL', names[original])
        if kind in ('SYMBOL', 'NUMBER'):
            return item
        return ('NODE', item[1], tuple(visit(child) for child in item[2]))

    return visit(tree)


def canonical_node_count(tree):
    """Count every node in the normalized tree, matching project node_count."""
    if tree[0] in ('SYMBOL', 'NUMBER'):
        return 1
    return 1 + sum(canonical_node_count(child) for child in tree[2])


def render_structure(tree):
    kind = tree[0]
    if kind in ('SYMBOL', 'NUMBER'):
        return tree[1]
    return f"{tree[1]}(" + ', '.join(render_structure(child) for child in tree[2]) + ')'


def render_readable(tree):
    """Render normalized trees with SymPy-like infix notation."""
    kind = tree[0]
    if kind in ('SYMBOL', 'NUMBER'):
        return tree[1], 4

    name, children = tree[1], tree[2]
    if name == 'Add':
        rendered = []
        for child in children:
            negative = False
            magnitude = child
            if child[0] == 'NODE' and child[1] == 'Mul':
                factors = list(child[2])
                minus_index = next((i for i, factor in enumerate(factors)
                                    if factor == ('NUMBER', '-1')), None)
                if minus_index is not None:
                    factors.pop(minus_index)
                    negative = True
                    magnitude = (factors[0] if len(factors) == 1
                                 else ('NODE', 'Mul', tuple(factors)))
            elif child == ('NUMBER', '-1'):
                negative = True
                magnitude = ('NUMBER', '1')
            text, precedence = render_readable(magnitude)
            if precedence < 1:
                text = f'({text})'
            if not rendered:
                rendered.append(('-' if negative else '') + text)
            else:
                rendered.append((' - ' if negative else ' + ') + text)
        return ''.join(rendered), 1

    if name == 'Mul':
        factors = list(children)
        negative = False
        minus_index = next((i for i, factor in enumerate(factors)
                            if factor == ('NUMBER', '-1')), None)
        if minus_index is not None:
            factors.pop(minus_index)
            negative = True
        parts = []
        for factor in factors:
            text, precedence = render_readable(factor)
            if precedence < 2:
                text = f'({text})'
            parts.append(text)
        product = '*'.join(parts) if parts else '1'
        return ('-' if negative else '') + product, 2

    if name == 'Pow':
        base, base_precedence = render_readable(children[0])
        exponent, exponent_precedence = render_readable(children[1])
        if base_precedence < 3 or (children[0][0] == 'NUMBER' and base.startswith('-')):
            base = f'({base})'
        if exponent_precedence < 4:
            exponent = f'({exponent})'
        return f'{base}**{exponent}', 3

    args = [render_readable(child)[0] for child in children]
    return f"{name}({', '.join(args)})", 4


def read_fronts(relative_path, route, *, pysr=False, pooled_metrics=False):
    occurrences = defaultdict(list)
    missing = []
    for seed_dir in sorted(ROOT.glob('seed_*')):
        if not seed_dir.is_dir():
            continue
        relative_paths = (relative_path if isinstance(relative_path, (list, tuple))
                          else [relative_path])
        path = next((seed_dir / item for item in relative_paths
                     if (seed_dir / item).is_file()), seed_dir / relative_paths[0])
        if not path.is_file():
            missing.append(str(path.relative_to(ROOT)))
            continue
        with path.open('r', encoding='utf-8-sig', newline='') as handle:
            rows = csv.DictReader(handle)
            for row_index, row in enumerate(rows, start=1):
                expression = (row.get('readable_expression') or '').strip()
                if not expression:
                    continue
                if pysr:
                    expression = prepare_pysr_expression(expression)
                    local_symbols = pysr_sympy_locals(expression)
                    # Do not evaluate while parsing: distinct constants remain
                    # distinct parameter slots even when their values match.
                    parsed = sp.sympify(expression, locals=local_symbols,
                                        evaluate=False)
                else:
                    parameter_names = set(re.findall(
                        r'(?<![A-Za-z0-9_])(?:a\d+|c\d+(?:_(?:left|right))?|p\d+'
                        r'|cconst_product_\d+(?:_(?:left|right))?'
                        r'|cconst_output_\d+(?:_(?:left|right))?)(?![A-Za-z0-9_])',
                        expression,
                    ))
                    local_symbols = {name: sp.Symbol(name, real=True)
                                     for name in parameter_names}
                    parsed = sp.sympify(expression, locals=local_symbols)
                parsed = merge_parameter_products(parsed)
                parsed = normalize_log_quotients(parsed)
                parsed = absorb_additive_constants_into_log(parsed)
                parsed = merge_parameter_products(parsed)
                tree = build_sorted_tree(parsed)
                normalized_tree = alpha_rename_tree(tree)
                canonical = render_structure(normalized_tree)
                readable_expression = render_readable(normalized_tree)[0]
                occurrences[canonical].append({
                    'seed': seed_dir.name,
                    'formula_id': row.get('formula_id') or row.get('id') or '',
                    'row_index': row_index,
                    'readable_expression': readable_expression,
                    'normalized_readable_expression': readable_expression,
                    'normalized_tree': normalized_tree,
                    # Canonical summaries compare whole holdouts. Prefer
                    # pooled metrics when fronts also contain group averages.
                    'test_mean_r2': first_present(
                        row.get('test_pooled_r2') if pooled_metrics else None,
                        row.get('test_mean_r2')),
                    'test_mean_rmse': first_present(
                        row.get('test_pooled_rmse') if pooled_metrics else None,
                        row.get('test_mean_rmse'), row.get('test_rmse')),
                    'validation_mean_r2': first_present(
                        row.get('validation_pooled_r2') if pooled_metrics else None,
                        row.get('validation_mean_r2')),
                    'validation_mean_rmse': first_present(
                        row.get('validation_pooled_rmse') if pooled_metrics else None,
                        row.get('validation_mean_rmse'), row.get('validation_rmse')),
                    'test_r2_std': first_present(
                        row.get('test_group_r2_std') if pooled_metrics else None,
                        row.get('test_r2_std')),
                    **({'test_group_r2_std': first_present(
                        row.get('test_group_r2_std'), row.get('test_r2_std'))}
                       if pooled_metrics else {}),
                    'complexity': row.get('test_complexity') or row.get('complexity'),
                    # Piecewise diagnostics.  Independent segment runs do
                    # not have these fields, so they remain empty there;
                    # shared-structure realdata_mat runs do.
                    'segment_prediction_difference': row.get('segment_prediction_difference'),
                })
    if missing:
        print(f'[{route}] missing {len(missing)} front file(s): ' + ', '.join(missing),
              file=sys.stderr)
    repeated = []
    for canonical, items in occurrences.items():
        seeds = sorted({item['seed'] for item in items})
        if len(seeds) < 2:
            continue
        per_seed = defaultdict(list)
        for item in items:
            per_seed[item['seed']].append(item)

        # A seed can contain multiple Pareto rows for the same canonical
        # structure. Select that seed's validation-best row first; then give
        # each seed one equal-weight contribution to every reported metric.
        seed_best_items = []
        for seed_items in per_seed.values():
            eligible = [item for item in seed_items
                        if _finite_float(item.get('validation_mean_r2')) is not None]
            if eligible:
                # Formula ID is the sole tie-break after validation R²;
                # row index only stabilizes duplicate IDs.
                seed_best_items.append(min(eligible, key=lambda item: (
                    -_finite_float(item.get('validation_mean_r2')),
                    str(item.get('formula_id') or ''),
                    item['row_index'],
                )))

        def seed_average(field):
            values = [_finite_float(item.get(field)) for item in seed_best_items]
            values = [value for value in values if value is not None]
            return sum(values) / len(values) if values else None

        repeated.append({
            'readable_expression': items[0]['normalized_readable_expression'],
            'canonical_structure': canonical,
            'seed_count': len(seeds),
            'seeds': ';'.join(seeds),
            'occurrence_count': len(items),
            'mean_test_r2': seed_average('test_mean_r2'),
            'mean_test_rmse': seed_average('test_mean_rmse'),
            'mean_validation_r2': seed_average('validation_mean_r2'),
            'mean_validation_rmse': seed_average('validation_mean_rmse'),
            'mean_test_r2_std': seed_average('test_r2_std'),
            **({'mean_test_group_r2_std': seed_average('test_r2_std')}
               if pooled_metrics else {}),
            'mean_complexity': canonical_node_count(items[0]['normalized_tree']),
            'canonical_complexity': canonical_node_count(items[0]['normalized_tree']),
            'mean_source_complexity': seed_average('complexity'),
            'mean_segment_prediction_difference': seed_average(
                'segment_prediction_difference'),
            'occurrences_json': json.dumps(
                [{key: value for key, value in item.items() if key != 'normalized_tree'} for item in items],
                ensure_ascii=False, separators=(',', ':')), 
        })
    repeated.sort(key=lambda row: (-row['seed_count'], -row['occurrence_count'],
                                   row['canonical_structure']))
    all_results = []
    for canonical, items in occurrences.items():
        seeds = sorted({item['seed'] for item in items})
        for item in items:
            all_results.append({
                key: value for key, value in item.items()
                if key != 'normalized_tree'
            } | {
                'canonical_structure': canonical,
                'seed_count': len(seeds),
                'seeds': ';'.join(seeds),
                'occurrence_count': len(items),
                'is_repeated_structure': len(seeds) >= 2,
                'canonical_complexity': canonical_node_count(
                    items[0]['normalized_tree']),
            })
    all_results.sort(key=lambda row: (row['seed'], row['row_index'],
                                      row['canonical_structure']))
    return repeated, all_results, missing


def detect_data_mode():
    """Infer the data mode from per-seed configs, with an artifact fallback."""
    modes = set()
    for config_path in sorted(ROOT.glob('seed_*_config.json')):
        try:
            config = json.loads(config_path.read_text(encoding='utf-8'))
            data_config = config.get('data', {})
            mode = data_config.get('mode')
            profile = data_config.get('profile')
            if profile == 'measured_pathloss':
                modes.add('realdata_mat')
            if mode:
                modes.add(str(mode))
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    if 'realdata_mat' in modes:
        return 'realdata_mat'
    if modes:
        return sorted(modes)[0]
    # Older experiment directories may not have a copied config.  The
    # dedicated real-data branch has stable directory names, so use those as
    # a conservative fallback.
    if any((ROOT / seed.name / 'shared_structure_fitting').is_dir()
           for seed in ROOT.glob('seed_*') if seed.is_dir()):
        return 'realdata_mat'
    return 'csv'


def read_sources(sources, *, pooled_metrics=False):
    """Read source routes and return repeated summaries and all occurrences."""
    generated_rows = {}
    all_generated_rows = {}
    source_metadata = {}
    for route, relative_path in sources.items():
        rows, all_rows, missing = read_fronts(
            relative_path, route, pysr=route.startswith('pysr'),
            pooled_metrics=pooled_metrics,
        )
        relative_paths = (relative_path if isinstance(relative_path, (list, tuple))
                          else [relative_path])
        display_paths = [path.as_posix() for path in relative_paths]
        generated_rows[route] = rows
        all_generated_rows[route] = all_rows
        source_metadata[route] = {
            'source_relative_path': (display_paths[0] if len(display_paths) == 1
                                     else display_paths),
            'repeated_structure_count': len(rows),
            'result_count': len(all_rows),
            'missing_seed_files': missing,
        }
    return generated_rows, all_generated_rows, source_metadata


def write_csv(path, rows, *, pooled_metrics=False):
    fields = ['readable_expression', 'canonical_structure', 'seed_count', 'seeds',
              'occurrence_count', 'mean_test_r2', 'mean_test_rmse',
              'mean_validation_r2', 'mean_validation_rmse', 'mean_test_r2_std',
              'mean_complexity', 'canonical_complexity', 'mean_source_complexity',
              'mean_segment_prediction_difference',
              'occurrences_json']
    if pooled_metrics:
        fields.insert(fields.index('mean_complexity'), 'mean_test_group_r2_std')
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_all_results_csv(path, rows, *, pooled_metrics=False):
    fields = [
        'seed', 'formula_id', 'row_index', 'readable_expression',
        'normalized_readable_expression', 'canonical_structure',
        'is_repeated_structure', 'seed_count', 'seeds', 'occurrence_count',
        'test_mean_r2', 'test_mean_rmse', 'validation_mean_r2',
        'validation_mean_rmse', 'test_r2_std', 'complexity',
        'canonical_complexity', 'segment_prediction_difference',
    ]
    if pooled_metrics:
        fields.insert(fields.index('complexity'), 'test_group_r2_std')
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _t_critical_975(df):
    """Return the two-sided 95% Student-t critical value."""
    if df <= 0:
        return None
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except ImportError:
        # Cornish-Fisher expansion around the normal quantile keeps this
        # reporting step dependency-free when SciPy is unavailable.
        z = 1.959963984540054
        v = float(df)
        return (z + (z**3 + z) / (4 * v)
                + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * v**2)
                + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z)
                / (384 * v**3))


def _metric_summary(values):
    values = [value for value in values if value is not None]
    count = len(values)
    if not count:
        return {'count': 0, 'mean': None, 'std': None, 'ci95': [None, None]}
    mean = statistics.mean(values)
    std = statistics.stdev(values) if count > 1 else None
    if count > 1:
        half_width = _t_critical_975(count - 1) * std / math.sqrt(count)
        ci95 = [mean - half_width, mean + half_width]
    else:
        ci95 = [None, None]
    return {'count': count, 'mean': mean, 'std': std, 'ci95': ci95}


def analyze_validation_selected_repeated_structures(
        generated_rows, all_generated_rows, *, data_mode):
    """Select validation-best per-seed formulas among canonical repeats only."""
    selected_by_route = {}
    for route, repeated_rows in generated_rows.items():
        eligible_structures = {
            row['canonical_structure'] for row in repeated_rows
        }
        per_seed = defaultdict(list)
        for row in all_generated_rows.get(route, []):
            if not row.get('is_repeated_structure'):
                continue
            validation_r2 = _finite_float(row.get('validation_mean_r2'))
            if validation_r2 is None:
                continue
            per_seed[row['seed']].append((validation_r2, row))

        chosen = {}
        for seed, candidates in per_seed.items():
            # A repeated structure is matched using the canonicalized tree;
            # parameter names and literal normalization therefore follow the
            # exact same rules as the existing repeated-structure tables.
            candidates = [item for item in candidates
                          if item[1]['canonical_structure'] in eligible_structures]
            if not candidates:
                continue
            candidates.sort(key=lambda item: (
                -item[0], item[1].get('formula_id', ''),
                item[1].get('canonical_structure', ''),
            ))
            validation_r2, row = candidates[0]
            test_r2 = _finite_float(row.get('test_mean_r2'))
            selected = {
                'route': route,
                'seed': seed,
                'formula_id': row.get('formula_id', ''),
                'canonical_structure': row['canonical_structure'],
                'readable_expression': row.get('normalized_readable_expression', ''),
                'validation_mean_r2': validation_r2,
                'test_mean_r2': test_r2,
                'validation_mean_rmse': _finite_float(row.get('validation_mean_rmse')),
                'test_mean_rmse': _finite_float(row.get('test_mean_rmse')),
                'complexity': _finite_float(row.get('canonical_complexity')),
            }
            chosen[seed] = selected
        selected_by_route[route] = chosen

    if data_mode == 'realdata_mat':
        candidate_route_pairs = [
            ('realdata_mat_shared_structure', 'pysr_shared_structure'),
            ('realdata_mat_independent_segment_1', 'pysr_independent_segment_1'),
            ('realdata_mat_independent_segment_2', 'pysr_independent_segment_2'),
        ]
    else:
        candidate_route_pairs = [('main_model', 'pysr_baseline')]
    route_pairs = [
        (left, right) for left, right in candidate_route_pairs
        if left in selected_by_route and right in selected_by_route
    ]

    route_summaries = {}
    for route, chosen in selected_by_route.items():
        structure_counts = defaultdict(int)
        for row in chosen.values():
            structure_counts[row['canonical_structure']] += 1
        repeated_structures = [count for count in structure_counts.values()
                               if count > 1]
        repeated_seed_count = sum(repeated_structures)
        route_summaries[route] = {
            'selected_seed_count': len(chosen),
            'validation_mean_r2': _metric_summary(
                [row['validation_mean_r2'] for row in chosen.values()]),
            'test_mean_r2': _metric_summary(
                [row['test_mean_r2'] for row in chosen.values()]),
            'test_mean_rmse': _metric_summary(
                [row['test_mean_rmse'] for row in chosen.values()]),
            'selected_repeated_structure_count': len({
                row['canonical_structure'] for row in chosen.values()
            }),
            'repeated_structure_type_count': len(repeated_structures),
            'selected_seeds_in_repeated_structures': repeated_seed_count,
            'selected_seed_repeat_rate': (
                repeated_seed_count / len(chosen) if chosen else None
            ),
        }

    comparisons = {}
    for left, right in route_pairs:
        if left not in selected_by_route or right not in selected_by_route:
            continue
        left_by_seed = selected_by_route[left]
        right_by_seed = selected_by_route[right]
        paired = []
        for seed in sorted(set(left_by_seed) & set(right_by_seed)):
            left_score = left_by_seed[seed]['test_mean_r2']
            right_score = right_by_seed[seed]['test_mean_r2']
            if left_score is not None and right_score is not None:
                paired.append({
                    'seed': seed,
                    'left_test_mean_r2': left_score,
                    'right_test_mean_r2': right_score,
                    'difference_left_minus_right': left_score - right_score,
                })
        difference_summary = _metric_summary(
            [row['difference_left_minus_right'] for row in paired])
        comparisons[f'{left}_minus_{right}'] = {
            'left_route': left,
            'right_route': right,
            'paired_seed_count': len(paired),
            'test_r2_difference': difference_summary,
        }

    return {
        'analysis': 'validation-best formula per seed, restricted to canonical structures occurring in at least two distinct seeds',
        'selection_metric': 'validation_mean_r2',
        'reported_test_metric': 'test_mean_r2',
        'complexity_limit': None,
        'structure_eligibility_source': 'Existing canonicalizer source fronts; currently the test Pareto fronts. Eligibility is therefore post-hoc and test-informed.',
        'confidence_interval': 'two-sided 95% Student-t interval across seeds; paired intervals use within-seed differences',
        'routes': route_summaries,
        'paired_comparisons': comparisons,
    }


def write_validation_selected_repeated_structure_summary(path, analysis):
    """Save only aggregate model and paired-difference statistics as CSV."""
    fields = [
        'comparison', 'sample_count', 'test_r2_mean_or_difference',
        'test_r2_std', 'ci95_lower', 'ci95_upper',
    ]
    route_labels = {
        'main_model': 'main_model',
        'pysr_baseline': 'pysr_baseline',
        'realdata_mat_shared_structure': 'main_model_shared_structure',
        'pysr_shared_structure': 'pysr_shared_structure',
        'realdata_mat_independent_segment_1': 'main_model_segment_1',
        'pysr_independent_segment_1': 'pysr_segment_1',
        'realdata_mat_independent_segment_2': 'main_model_segment_2',
        'pysr_independent_segment_2': 'pysr_segment_2',
    }
    rows = []

    def row_for(label, metric):
        ci = metric.get('ci95', [None, None])
        return {
            'comparison': label,
            'sample_count': metric.get('count'),
            'test_r2_mean_or_difference': metric.get('mean'),
            'test_r2_std': metric.get('std'),
            'ci95_lower': ci[0],
            'ci95_upper': ci[1],
        }

    for route, summary in analysis.get('routes', {}).items():
        rows.append(row_for(
            route_labels.get(route, route), summary['test_mean_r2'],
        ))
    for comparison in analysis.get('paired_comparisons', {}).values():
        left = route_labels.get(comparison['left_route'], comparison['left_route'])
        right = route_labels.get(comparison['right_route'], comparison['right_route'])
        rows.append(row_for(
            f'{left}_minus_{right}_paired_difference',
            comparison['test_r2_difference'],
        ))

    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)



def plot_repeated_structures(rows_by_model, metric_field='mean_test_r2',
                             metric_label='Mean test $R^2$', suffix='test'):
    """Save a shared-scale bubble scatter plot for both model routes."""
    import matplotlib.pyplot as plt

    styles = {
        'main_model': {'label': 'Main model', 'color': '#2878B5', 'marker': 'o'},
        'pysr_baseline': {'label': 'PySR', 'color': '#E07A28', 'marker': '^'},
        'pysr_shared_structure': {
            'label': 'PySR shared structure', 'color': '#E07A28', 'marker': '^',
        },
        'pysr_independent_segment_1': {
            'label': 'PySR segment 1', 'color': '#E07A28', 'marker': '^',
        },
        'pysr_independent_segment_2': {
            'label': 'PySR segment 2', 'color': '#E07A28', 'marker': '^',
        },
        'realdata_mat_shared_structure': {
            'label': 'realdata_mat shared structure', 'color': '#2878B5', 'marker': 'o',
        },
        'realdata_mat_independent_segment_1': {
            'label': 'realdata_mat segment 1', 'color': '#E07A28', 'marker': '^',
        },
        'realdata_mat_independent_segment_2': {
            'label': 'realdata_mat segment 2', 'color': '#6A4C93', 'marker': 's',
        },
    }
    # Bubble area (points squared) is linear in distinct-seed frequency and
    # uses the same global scale for both methods.
    area_per_seed = 72.0
    fig, ax = plt.subplots(figsize=(8.2, 5.8), constrained_layout=True)
    all_counts = sorted({int(row['seed_count'])
                         for rows in rows_by_model.values() for row in rows})

    for route, rows in rows_by_model.items():
        style = styles.get(route, {
            'label': route, 'color': '#4C956C', 'marker': 'D',
        })
        points = []
        for row in rows:
            try:
                x = float(row['mean_complexity'])
                y = float(row[metric_field])
                count = int(row['seed_count'])
            except (TypeError, ValueError, KeyError):
                continue
            if not all(v == v and abs(v) != float('inf') for v in (x, y)):
                continue
            points.append((x, y, count))
        points.sort(key=lambda point: (point[0], point[1], point[2]))
        if not points:
            continue
        xs, ys, counts = zip(*points)
        ax.scatter(xs, ys, s=[area_per_seed * n for n in counts],
                   c=style['color'], marker=style['marker'], label=style['label'],
                   alpha=0.78, edgecolors='white', linewidths=0.9, zorder=3)
        # Alternate annotation offsets to keep labels legible around nearby points.
        offsets = [(7, 7), (7, -12), (-7, 8), (-7, -12), (0, 12), (0, -16)]
        for index, (x, y, count) in enumerate(points):
            dx, dy = offsets[index % len(offsets)]
            ax.annotate(str(count), (x, y), xytext=(dx, dy),
                        textcoords='offset points', fontsize=8, color='#30343B',
                        ha='left' if dx >= 0 else 'right', va='center', zorder=4)

    ax.set_xlabel('Canonical node-count complexity')
    ax.set_ylabel(metric_label)
    ax.set_title('Repeated structures across seeds')
    ax.grid(True, color='#D9DEE5', linewidth=0.7, alpha=0.8, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if all_counts:
        legend_counts = sorted(set([all_counts[0], all_counts[len(all_counts)//2], all_counts[-1]]))
        size_handles = [ax.scatter([], [], s=area_per_seed * n, color='#8B929C',
                                   alpha=0.65, edgecolors='white', linewidths=0.8,
                                   label=str(n)) for n in legend_counts]
        size_legend = ax.legend(handles=size_handles, title='Seed frequency',
                                frameon=False, loc='lower right', labelspacing=1.2,
                                handletextpad=0.8)
        ax.add_artist(size_legend)
    stem = 'repeated_structures_bubble_scatter' if not suffix else f'repeated_structures_{suffix}_bubble_scatter'
    png_path = OUT / f'{stem}.png'
    pdf_path = OUT / f'{stem}.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return {'png': str(png_path), 'pdf': str(pdf_path),
            'metric_field': metric_field, 'metric_label': metric_label,
            'bubble_area_rule': 'area = 72 points^2 * distinct seed frequency; shared across models',
            'model_styles': styles}

def main(experiment_dir=None, model='main'):
    global ROOT, OUT, _CANON_CONST_COUNTER
    model = str(model).strip().lower()
    if model not in {'main', 'pysr', 'both'}:
        raise ValueError("model must be 'main', 'pysr', or 'both'")
    ROOT = Path(experiment_dir).resolve() if experiment_dir is not None else EXPERIMENT_DIR
    OUT = ROOT / 'cross_seed_formula_summary' / model
    _CANON_CONST_COUNTER = 0
    OUT.mkdir(parents=True, exist_ok=True)
    data_mode = detect_data_mode()
    # Keep Main-model and PySR summaries separate. The two realdata_mat forms
    # are deliberately separate routes:
    # shared symbolic structure with segment-specific constants, and formulas
    # fitted independently on segment 1/segment 2.
    if data_mode == 'realdata_mat':
        main_sources = {
            'realdata_mat_shared_structure': Path(
                'shared_structure_fitting/test/test_pareto_front.csv'),
            'realdata_mat_independent_segment_1': Path(
                'independent_segment_fitting/segment_1/test/test_pareto_front.csv'),
            'realdata_mat_independent_segment_2': Path(
                'independent_segment_fitting/segment_2/test/test_pareto_front.csv'),
        }
        pysr_sources = {
            'pysr_shared_structure': [
                Path('shared_structure_fitting/pysr_baseline/shared_refit/test_pareto_front.csv'),
                Path('shared_structure_fitting/pysr_baseline/test_pareto_front.csv'),
            ],
            'pysr_independent_segment_1': [
                Path('independent_segment_fitting/segment_1/pysr_baseline/shared_refit/test_pareto_front.csv'),
                Path('independent_segment_fitting/segment_1/pysr_baseline/test_pareto_front.csv'),
            ],
            'pysr_independent_segment_2': [
                Path('independent_segment_fitting/segment_2/pysr_baseline/shared_refit/test_pareto_front.csv'),
                Path('independent_segment_fitting/segment_2/pysr_baseline/test_pareto_front.csv'),
            ],
        }
    else:
        main_sources = {
            'main_model': Path('test/test_pareto_front.csv'),
        }
        pysr_sources = {
            'pysr_baseline': [
                Path('pysr_baseline/shared_refit/test_pareto_front.csv'),
                Path('pysr_baseline/test_pareto_front.csv'),
            ],
        }
    if model == 'main':
        sources = main_sources
    elif model == 'pysr':
        sources = pysr_sources
    else:
        sources = {**main_sources, **pysr_sources}
    summary = {'experiment_directory': str(ROOT), 'data_mode': data_mode,
               'outputs': {}, 'rule': {
        'model_scope': model,
        'files': ('only the requested Main-model or PySR result routes are read'
                  if model != 'both' else
                  'each seed test_pareto_front.csv; PySR routes are read from pysr_baseline/'),
        'operand_order': 'sort Add/Mul children only; preserve Pow and function argument order',
        'parameter_symbols': 'aN, cN[_left|_right], pN, cconst_product_N[_left|_right], and cconst_output_N[_left|_right] alpha-renamed C1, C2, ... left-to-right; explicit product signs are absorbed into a free parameter',
        'floating_literals': 'main-model floating literals and every PySR numeric literal (including integers) are alpha-renamed C1, C2, ...; each PySR literal occurrence gets a distinct parameter before canonicalization',
        'pysr_template_rows': 'only the expression after f= is canonicalized; trailing fitted parameter vectors are treated as metadata',
        'readable_expression_format': 'SymPy-like infix expression using normalized parameter names',
        'integer_literals': 'preserved unless they are standalone additive constants absorbed into a variable log',
        'log_quotients': 'log(Abs(1/expr)) becomes -log(Abs(expr)); for other constant numerators, use -log(Abs(C*expr))',
        'standalone_additive_constants': 'an additive offset may be absorbed into a first canonical log as A*log(Abs(C*f)); logs whose argument already has a constant multiplier are left as-is',
        'parameter_products': 'multiple direct fitted-parameter factors in one product are merged into one free canonical constant',
        'complexity': 'mean_complexity is recomputed by counting every node in the canonicalized expression tree (same node_count rule as formula_discovery_pipeline); mean_source_complexity retains the source Pareto complexity for reference',
        'repeat_definition': 'same canonical structure appears in at least two distinct seeds',
        'comparison_metrics': ('For each canonical structure, select the validation-R2-best occurrence within each seed, then average each metric across those selected seed rows with equal seed weight. Test R2/RMSE prefer pooled holdout fields; Validation R2 prefers pooled fields; validation RMSE uses a pooled field when present and otherwise retains the saved group mean. mean_test_group_r2_std is averaged across the selected seed rows.'
                              if data_mode == 'realdata_mat' else
                              'For each canonical structure, select the validation-R2-best occurrence within each seed, then average each metric across those selected seed rows with equal seed weight.'),
    }}
    pooled_metrics = data_mode == 'realdata_mat'
    generated_rows, all_generated_rows, source_metadata = read_sources(
        sources, pooled_metrics=pooled_metrics
    )
    for route, relative_path in sources.items():
        rows = generated_rows[route]
        output_path = OUT / f'{route}_repeated_structures.csv'
        write_csv(output_path, rows, pooled_metrics=pooled_metrics)
        all_output_path = OUT / f'{route}_all_results.csv'
        write_all_results_csv(all_output_path, all_generated_rows[route],
                              pooled_metrics=pooled_metrics)
        summary['outputs'][route] = {
            **source_metadata[route], 'csv': str(output_path),
            'all_results_csv': str(all_output_path),
        }
        print(f'{route}: {len(rows)} repeated structures -> {output_path}')
        print(f'{route}: {len(all_generated_rows[route])} total results -> {all_output_path}')
    repeated_selection = analyze_validation_selected_repeated_structures(
        generated_rows, all_generated_rows, data_mode=data_mode,
    )
    comparison_csv_path = OUT / 'validation_selected_repeated_structure_summary.csv'
    write_validation_selected_repeated_structure_summary(
        comparison_csv_path, repeated_selection,
    )
    print('Validation-selected repeated-structure aggregate saved -> '
          f'{comparison_csv_path}')
    test_plot = plot_repeated_structures(
        generated_rows, metric_field='mean_test_r2',
        metric_label='Mean test $R^2$', suffix='test')
    validation_plot = plot_repeated_structures(
        generated_rows, metric_field='mean_validation_r2',
        metric_label='Mean validation $R^2$', suffix='validation')
    summary['plots'] = {'test': test_plot, 'validation': validation_plot}
    print(f"Test bubble plots saved -> {test_plot['png']} and {test_plot['pdf']}")
    print(f"Validation bubble plots saved -> {validation_plot['png']} and {validation_plot['pdf']}")
    summary_path = OUT / 'summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
                            encoding='utf-8')
    print(f'summary -> {summary_path}')
    return summary


if __name__ == '__main__':
    main()
