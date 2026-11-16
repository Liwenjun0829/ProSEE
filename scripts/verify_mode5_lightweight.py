#!/usr/bin/env python3
"""Run five original-parameter rounds for archived experiments 055, 056, 058–060.

The script derives temporary configs from each experiment's saved seed config,
runs only seed 001 and five search rounds, disables the optional PySR baseline,
and compares validation/test hypervolume and Pareto-front sizes at iterations
0–5 to the archived per-iteration records. Outputs go to /tmp by default.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = PROJECT_ROOT / "src" / "prosee" / "symbolic" / "core"
REFERENCE_ROOT = Path(os.environ.get(
    "PROSEE_LEGACY_PIPELINE_ROOT",
    "/home/lwj/正式代码/另一种NN模型",
)).expanduser().resolve()
PROSEE_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from formula_discovery_pipeline import (  # noqa: E402
    _calculate_validation_hv,
    _validation_hv_objectives,
    run_mode4_paper_experiment,
)

DEFAULT_OUTPUT_ROOT = Path("/tmp/prosee_archived_experiment_verification")
SIGNED_CANDIDATE_MARKERS = (
    "zero_then_signed",
    "signed_best_r2",
    "signed_r2_floor",
    "exhaustive_constant_assignment",
)
PROSEE_CSV_SOURCES: dict[int, dict[str, Path]] = {
    55: {
        "takeoff": PROSEE_PROCESSED_ROOT / "A2G" / "standardized_pathloss_dataset_measured_0123_takeoff.csv",
        "landing": PROSEE_PROCESSED_ROOT / "A2G" / "standardized_pathloss_dataset_measured_0123_landing.csv",
    },
    56: {
        "cruise": PROSEE_PROCESSED_ROOT / "A2G" / "standardized_pathloss_dataset_measured_0123_cruise.csv",
    },
    58: {
        "freq1": PROSEE_PROCESSED_ROOT / "Square1" / "standardized_pathloss_dataset_f1p4GHz_V.csv",
        "freq2": PROSEE_PROCESSED_ROOT / "Square1" / "standardized_pathloss_dataset_f28p0GHz_V.csv",
        "freq3": PROSEE_PROCESSED_ROOT / "Square1" / "standardized_pathloss_dataset_f3p5GHz_V.csv",
    },
    59: {
        "freq1": PROJECT_ROOT / "data" / "legacy" / "experiment_inputs" / "class1_all_freq1.csv",
        "freq2": PROJECT_ROOT / "data" / "legacy" / "experiment_inputs" / "class1_all_freq2.csv",
        "freq3": PROJECT_ROOT / "data" / "legacy" / "experiment_inputs" / "class1_all_freq3.csv",
    },
}


def build_lightweight_config(experiment: int, output_dir: Path) -> dict[str, Any]:
    """Copy the saved experiment config, preserving algorithm parameters."""
    source = REFERENCE_ROOT / "artifacts" / "expri2" / f"mode5_experiment_{experiment:03d}" / "seed_001_config.json"
    if not source.is_file():
        raise FileNotFoundError(f"Saved experiment config not found: {source}")

    config = json.loads(source.read_text(encoding="utf-8"))
    config.pop("gpu", None)
    config["output"] = {**config.get("output", {}), "directory": str(output_dir)}

    # Rebase datasets that were moved to ProSEE. Experiment 059 uses the
    # preserved legacy Block1 exports because ProSEE's current freq2/freq3
    # class-1 exports contain different file_name membership.
    for csv_source in config.get("data", {}).get("csv_files", []):
        source_name = csv_source.get("name")
        replacement = PROSEE_CSV_SOURCES.get(experiment, {}).get(source_name)
        if replacement is not None:
            if not replacement.is_file():
                raise FileNotFoundError(f"ProSEE dataset not found: {replacement}")
            csv_source["csv_path"] = str(replacement)

    # Keep the saved algorithm, fitting, and ranking parameters. The runner
    # below reduces the work to one seed and five search rounds.
    config.setdefault("pysr_baseline", {})["enabled"] = False

    # These fields are obsolete and must not be carried into the smoke config.
    for key in (
        "mode5_enable_signed_constant_test",
        "mode5_initial_signed_constant_test_enabled",
        "mode5_periodic_signed_constant_test_enabled",
    ):
        config.pop(key, None)
    return config


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Iteration artifact not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _mode5_branches(experiment: int) -> list[tuple[str, Path]]:
    """Return the saved per-seed branches that define the iteration baseline."""
    seed_root = REFERENCE_ROOT / "artifacts" / "expri2" / f"mode5_experiment_{experiment:03d}" / "seed_001"
    if experiment == 60:
        return [
            ("shared_structure", seed_root / "shared_structure_fitting"),
            ("segment_1", seed_root / "independent_segment_fitting" / "segment_1"),
            ("segment_2", seed_root / "independent_segment_fitting" / "segment_2"),
        ]
    return [("main", seed_root)]


def _compare_iteration_csv(
    reference_path: Path,
    actual_path: Path,
    *,
    metric_name: str,
    iteration_ids: tuple[int, ...] = tuple(range(6)),
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Compare saved validation/test HV and front sizes for iterations 0–5."""
    ref_rows = _read_csv_rows(reference_path)
    actual_rows = _read_csv_rows(actual_path)

    def index(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
        return {
            int(row["iteration"]): row
            for row in rows
            if row.get("method", "tree_model") == "tree_model"
            and row.get("iteration", "").isdigit()
        }

    reference, actual = index(ref_rows), index(actual_rows)
    comparisons = []
    for iteration in iteration_ids:
        if iteration not in reference or iteration not in actual:
            raise ValueError(
                f"{metric_name}: missing iteration {iteration} "
                f"(reference={iteration in reference}, actual={iteration in actual})"
            )
        expected, observed = reference[iteration], actual[iteration]
        hv_key = f"{metric_name}_hypervolume"
        size_key = f"{metric_name}_pareto_front_size"
        expected_hv, observed_hv = float(expected[hv_key]), float(observed[hv_key])
        expected_size, observed_size = int(expected[size_key]), int(observed[size_key])
        hv_matches = math.isclose(expected_hv, observed_hv, rel_tol=rtol, abs_tol=atol)
        size_matches = expected_size == observed_size
        comparisons.append({
            "iteration": iteration,
            "reference_hypervolume": expected_hv,
            "actual_hypervolume": observed_hv,
            "hypervolume_matches": hv_matches,
            "reference_pareto_front_size": expected_size,
            "actual_pareto_front_size": observed_size,
            "pareto_front_size_matches": size_matches,
        })
    return {
        "reference": str(reference_path),
        "actual": str(actual_path),
        "iterations": comparisons,
        "passed": all(
            item["hypervolume_matches"] and item["pareto_front_size_matches"]
            for item in comparisons
        ),
    }


def compare_saved_iterations(experiment: int, actual_seed_root: Path) -> list[dict[str, Any]]:
    """Compare rounds using the archived experiment-wide HV normalization."""
    results = []
    for branch_name, reference_branch in _mode5_branches(experiment):
        actual_branch = actual_seed_root
        if experiment == 60:
            actual_branch = actual_seed_root / (
                "shared_structure_fitting" if branch_name == "shared_structure"
                else f"independent_segment_fitting/{branch_name}"
            )
        for metric_name in ("validation", "test"):
            filename = f"mode2_{metric_name}_hypervolume_by_iteration.csv"
            reference_path = reference_branch / filename
            with (reference_branch / f"mode2_{metric_name}_comparison.json").open(
                "r", encoding="utf-8"
            ) as handle:
                comparison_metadata = json.load(handle)
            bounds = {
                field: (float(values["min"]), float(values["max"]))
                for field, values in comparison_metadata["normalization_bounds"].items()
            }
            data_mode = "realdata_mat" if experiment == 60 else "csv"
            fields, maximize_fields = _validation_hv_objectives(
                data_mode, metric_name
            )
            reference_rows = _read_csv_rows(reference_path)
            expected_by_iteration = {
                int(row["iteration"]): row
                for row in reference_rows
                if row.get("method") == "tree_model"
            }
            actual_path = actual_branch / metric_name / "hypervolume" / (
                f"{metric_name}_hypervolume_fronts_by_iteration.csv"
            )
            actual_rows = _read_csv_rows(actual_path)
            actual_fronts: dict[int, list[dict[str, float]]] = {}
            for row in actual_rows:
                iteration = int(row["iteration"])
                actual_fronts.setdefault(iteration, []).append({
                    field: float(row[field]) for field in fields
                })
            comparisons = []
            for iteration in range(6):
                if iteration not in expected_by_iteration or iteration not in actual_fronts:
                    raise ValueError(
                        f"{metric_name}: missing iteration {iteration} "
                        f"(reference={iteration in expected_by_iteration}, "
                        f"actual={iteration in actual_fronts})"
                    )
                expected = expected_by_iteration[iteration]
                actual_front = actual_fronts[iteration]
                expected_hv = float(expected[f"{metric_name}_hypervolume"])
                actual_hv = _calculate_validation_hv(
                    actual_front, fields, maximize_fields, bounds
                )
                expected_size = int(expected[f"{metric_name}_pareto_front_size"])
                actual_size = len(actual_front)
                comparisons.append({
                    "iteration": iteration,
                    "reference_hypervolume": expected_hv,
                    "actual_hypervolume": actual_hv,
                    "hypervolume_matches": math.isclose(
                        expected_hv, actual_hv, rel_tol=1e-5, abs_tol=1e-6
                    ),
                    "reference_pareto_front_size": expected_size,
                    "actual_pareto_front_size": actual_size,
                    "pareto_front_size_matches": expected_size == actual_size,
                })
            comparison = {
                "reference": str(reference_path),
                "actual_fronts": str(actual_path),
                "normalization_bounds": bounds,
                "normalization_source": str(
                    reference_branch / f"mode2_{metric_name}_comparison.json"
                ),
                "iterations": comparisons,
                "passed": all(
                    item["hypervolume_matches"]
                    and item["pareto_front_size_matches"]
                    for item in comparisons
                ),
            }
            comparison["branch"] = branch_name
            comparison["metric"] = metric_name
            results.append(comparison)
    return results


def assess_initial_solution_import(experiment: int) -> dict[str, Any]:
    """Check whether archived initial-fit summaries contain reusable parameters."""
    seed_root = REFERENCE_ROOT / "artifacts" / "expri2" / f"mode5_experiment_{experiment:03d}" / "seed_001"
    assessments = []
    for branch_name, branch_root in _mode5_branches(experiment):
        if experiment == 60:
            branch_root = seed_root / (
                "shared_structure_fitting" if branch_name == "shared_structure"
                else f"independent_segment_fitting/{branch_name}"
            )
        summary_path = branch_root / "initial_lm_mean_r2.csv"
        if not summary_path.is_file():
            assessments.append({"branch": branch_name, "available": False})
            continue
        with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
        # Historical files store aggregate scores and formulas, not fitted
        # per-dataset parameter values required to initialize the optimizer.
        assessments.append({
            "branch": branch_name,
            "available": True,
            "columns": fields,
            "reused": False,
            "reason": "summary has aggregate R²/formula fields but no per-dataset fitted parameter values",
        })
    return {"reused": False, "branches": assessments}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help=f"base directory for temporary configs, logs, and outputs (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--experiments", type=int, nargs="+", choices=[55, 56, 58, 59, 60],
        default=[55, 56, 58, 59, 60], metavar="055, 056, 058–060",
        help="subset of experiments to run; defaults to all five",
    )
    parser.add_argument(
        "--lm-jobs", type=int, default=None,
        help="parallel LM workers (default: each saved config's ranking.n_jobs)",
    )
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.output_root.expanduser().resolve() / stamp
    run_root.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []

    for experiment in args.experiments:
        label = f"{experiment:03d}"
        experiment_root = run_root / label
        experiment_root.mkdir(parents=True)
        config_path = experiment_root / "reference_check_config.json"
        log_path = experiment_root / "run.log"
        config = build_lightweight_config(experiment, experiment_root / "output")
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        lm_jobs = args.lm_jobs or int(config.get("ranking", {}).get("n_jobs", 24))
        if lm_jobs < 2:
            parser.error("--lm-jobs must be at least 2 to preserve the parallel seeded-fit path")
        print(f"[{label}] running with LM n_jobs={lm_jobs}; log: {log_path}", flush=True)

        result: dict[str, Any] = {
            "experiment": label,
            "config": str(config_path),
            "log": str(log_path),
            "status": "failed",
            "initial_solution_import": assess_initial_solution_import(experiment),
        }
        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                with redirect_stdout(log_file), redirect_stderr(log_file):
                    summary = run_mode4_paper_experiment(
                        config_path,
                        n_jobs=lm_jobs,
                        serial_lm=False,
                        experiment_mode=5,
                        retrain_interval=5,
                        seed_count=1,
                        target_iterations=5,
                    )
            summary_path = Path(summary["summary_json"])
            saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            seed_rows = saved_summary.get("per_seed", [])
            completed = bool(seed_rows) and all(
                row.get("run_completed") is True for row in seed_rows
            )
            if not completed:
                details = [
                    row.get("error") or row.get("status")
                    for row in seed_rows
                    if row.get("run_completed") is not True
                ]
                raise RuntimeError(
                    "experiment summary contains an incomplete seed"
                    + (f": {details}" if details else "")
                )
            seed_directories = saved_summary.get("seed_directories", [])
            if not seed_directories:
                raise RuntimeError("experiment summary has no seed output directory")
            iteration_comparisons = compare_saved_iterations(
                experiment, Path(seed_directories[0])
            )
            log_text = log_path.read_text(encoding="utf-8")
            signed_markers = [
                marker for marker in SIGNED_CANDIDATE_MARKERS
                if marker in log_text
            ]
            if signed_markers:
                raise RuntimeError(
                    "removed signed-candidate markers appeared in the log: "
                    + ", ".join(signed_markers)
                )
            comparison_passed = all(
                item["passed"] for item in iteration_comparisons
            )
            passed = comparison_passed and not signed_markers
            result.update(
                status="passed" if passed else "failed",
                summary=str(summary_path),
                completed_seeds=len(seed_rows),
                iteration_comparisons=iteration_comparisons,
                signed_candidate_markers=signed_markers,
            )
            if not comparison_passed:
                result["error"] = "one or more saved iteration metrics differ"
            elif signed_markers:
                result["error"] = "removed signed-candidate markers appeared in the log"
            print(f"[{label}] {'PASS' if passed else 'FAIL'}", flush=True)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write("\n\nVerification exception:\n")
                log_file.write(traceback.format_exc())
            print(f"[{label}] FAIL: {result['error']}", flush=True)
        results.append(result)

    report = {
        "run_directory": str(run_root),
        "mode": 5,
        "seed_count_per_experiment": 1,
        "target_iterations": 5,
        "lm_n_jobs": lm_jobs,
        "pysr_enabled": False,
        "iteration_comparison": "validation/test HV and Pareto front size at iterations 0–5, recalculating actual HV with each archived experiment-wide normalization scale",
        "results": results,
        "all_passed": all(item["status"] == "passed" for item in results),
    }
    report_path = run_root / "verification_summary.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Summary: {report_path}", flush=True)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
