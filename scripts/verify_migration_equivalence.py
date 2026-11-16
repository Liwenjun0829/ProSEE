#!/usr/bin/env python3
"""Verify that the migrated numerical kernel matches its recorded source."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "MIGRATION_MANIFEST.json"
REFERENCE_RENAMES = {
    "pareto_formula_evolution.py": "lightweight_formula_gp.py",
    "structure_screen/tree_search_ops.py": "structure_screen/gp.py",
    "canonicalize_repeated_structures.py": "canonicalize_mode5_main_model.py",
    "formula_discovery_pipeline.py": "dimensionless_formula_pipeline.py",
    "generate_zero_sum_bases.py": "generate_dimensionless_bases.py",
}
REFERENCE_IMPORT_UPDATES = {
    "formula_discovery_pipeline.py": (
        (b"ROOT = Path(__file__).resolve().parent\n",
         b"ROOT = Path(__file__).resolve().parent\nPROJECT_ROOT = ROOT.parents[3]\n"),
        (b'DEFAULT_CONFIG = ROOT / "dimensionless_formula_config.json"',
         b'DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "formula_discovery" / "pipeline.json"'),
        (b'DEFAULT_FORMULA_GENERATION_CONFIG = ROOT / "dimensionless_formula_generation_config.json"',
         b'DEFAULT_FORMULA_GENERATION_CONFIG = (\n    PROJECT_ROOT / "configs" / "formula_discovery" / "formula_generation.json"\n)'),
        (b"from structure_screen.gp import complexity as formula_complexity",
         b"from structure_screen.tree_search_ops import complexity as formula_complexity"),
        (b"from unified_symbolic_pipeline import load_csv_data, resolve_path",
         b"from data_io import load_csv_data, resolve_path"),
        (b"from generate_dimensionless_bases import generate_dimensionless_bases",
         b"from generate_zero_sum_bases import generate_zero_sum_bases"),
        (b"generate_dimensionless_bases(", b"generate_zero_sum_bases("),
        (b"dimensionless_formula_results", b"formula_discovery_results"),
        (b"projected_dimensionless_monomial", b"projected_zero_sum_monomial"),
        (b"symbolic_dimensionless_constant", b"symbolic_constant_base"),
        (b"dimensionless parameter", b"fitted symbolic constant"),
        (b"No nontrivial dimensionless bases were generated",
         b"No nontrivial zero-sum bases were generated"),
        (b"variable/dimensionless base", b"variable/base"),
        (b"Dimensionless basis -> candidates -> binary formulas -> two-stage LM fitting.",
         b"Formula bases -> candidates -> binary formulas -> two-stage LM fitting."),
        (b'if row.get("structure_type") == "symbolic_constant_base"',
         b'if row.get("structure_type") in {\n            "symbolic_constant_base", "symbolic_dimensionless_constant"\n        }'),
        (b'history_path = output_dir / "formula_discovery_results.json"\n    if fit_mode',
         b'history_path = output_dir / "formula_discovery_results.json"\n    legacy_history_path = output_dir / "dimensionless_formula_results.json"\n    if not history_path.is_file() and legacy_history_path.is_file():\n        history_path = legacy_history_path\n    if fit_mode'),
        (b"from lightweight_formula_gp import (", b"from pareto_formula_evolution import ("),
        (b"from canonicalize_mode5_main_model import main as run_canonicalize",
         b"from canonicalize_repeated_structures import main as run_canonicalize"),
        (b"precompute_dimensionless_formula_features.py", b"precompute_formula_features.py"),
        (b"dimensionless_formula_generation_config.json", b"formula_generation.json"),
    ),
    "generate_zero_sum_bases.py": (
        (b"Generate dimensionless bases with the project's original projection logic.",
         b"Generate monomial bases whose exponent vectors sum to zero."),
        (b"Generate dimensionless bases", b"Generate zero-sum bases"),
        (b"generate_dimensionless_bases", b"generate_zero_sum_bases"),
        (b"dimensionless_bases_", b"zero_sum_bases_"),
        (b"dimensionless monomial bases", b"zero-sum monomial bases"),
        (b"dimensionless bases", b"zero-sum bases"),
    ),
    "pareto_formula_evolution.py": (
        (b"from structure_screen.gp import ", b"from structure_screen.tree_search_ops import "),
        (b"after dimensionless formula fitting", b"after formula fitting"),
    ),
    "structure_screen/dimensional_gp.py": (
        (b"structure_screen.gp", b"structure_screen.tree_search_ops"),
        (b"from .gp import ", b"from .tree_search_ops import "),
    ),
    "structure_screen/legacy_gp_adapter.py": (
        (b"from .gp import ", b"from .tree_search_ops import "),
    ),
    "structure_screen/__init__.py": (
        (b"Formula-structure learning and GP screening utilities.",
         b"Formula-structure representation, learned ranking, tree search, and fitting utilities."),
    ),
    "structure_screen/cli.py": (
        (b'from typing import Sequence\n',
         b'from typing import Sequence\n\nPROJECT_ROOT = Path(__file__).resolve().parents[4]\nDEFAULT_CONFIG = PROJECT_ROOT / "configs" / "structure_screen" / "legacy.json"\n'),
        (b'parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))',
         b'parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)'),
        (b'from .gp import run_gp', b'from .tree_search_ops import run_gp'),
    ),
    "structure_r2_ranker.py": (
        (b"Dimensionless base IDs", b"Formula base IDs"),
        (b"not generated from dimensionless primitives", b"not generated from formula primitives"),
        (b"undefined dimensionless base", b"undefined formula base"),
        (b'precompute_dimensionless_formula_features.py', b'precompute_formula_features.py'),
        (b'from scipy.stats import rankdata, spearmanr\n',
         b'from scipy.stats import rankdata, spearmanr\n\nPROJECT_ROOT = next(\n    parent for parent in Path(__file__).resolve().parents\n    if (parent / "pyproject.toml").is_file()\n)\n'),
        (b'feature_cache_path = (Path(__file__).resolve().parent\n                          / "structure_r2_feature_cache_features_only.joblib")',
         b'feature_cache_path = (\n        PROJECT_ROOT / "artifacts" / "reference_features"\n        / "structure_r2_feature_cache_features_only.joblib"\n    )'),
    ),
    "canonicalize_repeated_structures.py": (
        (b"Find repeated Main-model and PySR symbolic structures across Mode 5 seeds.",
         b"Find repeated Main-model and PySR symbolic structures across experiment seeds."),
        (b"same node_count rule as dimensionless_formula_pipeline",
         b"same node_count rule as formula_discovery_pipeline"),
        (b"dimensionless_formula_results", b"formula_discovery_results"),
    ),
}

EXTRACTED_FUNCTIONS = ("resolve_path", "validate_xy", "load_csv_data")


def _function_source(path: Path, names: tuple[str, ...]) -> dict[str, str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    return {
        node.name: ast.get_source_segment(source, node) or ""
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-root",
        type=Path,
        help="optional legacy source root for a direct byte comparison",
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    core_root = PROJECT_ROOT / manifest["core_destination"]
    failures: list[str] = []

    for relative_name, expected_hash in manifest["files"].items():
        migrated_path = core_root / relative_name
        if not migrated_path.is_file():
            failures.append(f"missing migrated file: {migrated_path}")
            continue
        actual_hash = file_sha256(migrated_path)
        if actual_hash != expected_hash:
            failures.append(
                f"migrated hash mismatch: {relative_name} "
                f"({actual_hash} != {expected_hash})"
            )
        if args.reference_root is not None:
            if relative_name == "data_io.py":
                reference_path = (
                    args.reference_root.expanduser().resolve()
                    / "unified_symbolic_pipeline.py"
                )
                if not reference_path.is_file():
                    failures.append(f"missing reference file: {reference_path}")
                else:
                    try:
                        expected_functions = _function_source(reference_path, EXTRACTED_FUNCTIONS)
                        actual_functions = _function_source(migrated_path, EXTRACTED_FUNCTIONS)
                        if expected_functions != actual_functions:
                            failures.append("extracted data_io functions differ from reference")
                    except (SyntaxError, UnicodeError) as exc:
                        failures.append(f"cannot compare extracted data_io functions: {exc}")
                continue
            reference_name = REFERENCE_RENAMES.get(relative_name, relative_name)
            reference_path = args.reference_root.expanduser().resolve() / reference_name
            if not reference_path.is_file():
                failures.append(f"missing reference file: {reference_path}")
            else:
                reference_bytes = reference_path.read_bytes()
                for old, new in REFERENCE_IMPORT_UPDATES.get(relative_name, ()):
                    reference_bytes = reference_bytes.replace(old, new)
                if hashlib.sha256(reference_bytes).hexdigest() != actual_hash:
                    failures.append(f"reference bytes differ: {relative_name}")

    if failures:
        print("Migration equivalence: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    source_root = PROJECT_ROOT / "src"
    sys.path.insert(0, str(source_root))
    from prosee.symbolic import configuration, data, evaluation, fitting, formulas

    required_symbols = {
        configuration: ("load_config",),
        data: ("load_configured_data", "load_joint_csv_data"),
        evaluation: ("compute_metrics", "compute_joint_metrics"),
        fitting: ("fit_with_restarts", "fit_formula_per_dataset"),
        formulas: ("build_base_records", "build_candidate_records"),
    }
    missing_symbols = [
        f"{module.__name__}.{name}"
        for module, names in required_symbols.items()
        for name in names
        if not hasattr(module, name)
    ]
    if missing_symbols:
        print("Migration equivalence: FAIL")
        for name in missing_symbols:
            print(f"- missing public API: {name}")
        return 1

    print(f"Migration equivalence: PASS ({len(manifest['files'])} core files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
