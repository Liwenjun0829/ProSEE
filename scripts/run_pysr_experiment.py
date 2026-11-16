#!/usr/bin/env python3
"""Run only the PySR baseline for one experiment configuration.

Example:
    python scripts/run_pysr_experiment.py \\
        --config configs/experiments/a2g_takeoff_landing.json

The matching sidecar is inferred from ``configs/experiments/pysr`` unless
``--pysr-config`` is supplied explicitly. Results and PySR-only canonicalized
artifacts are written under a fresh ``results/experiments/<scene>/pysr/run_NNN/`` tree.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from prosee.symbolic.core import formula_discovery_pipeline as pipeline  # noqa: E402
from prosee.symbolic.core import canonicalize_repeated_structures as canonicalizer  # noqa: E402
from prosee.symbolic.core.pysr import baseline as pysr_baseline  # noqa: E402


def run_pysr_experiment(
    config_path: Path, pysr_config_path: Path, *, seed_count: int = 10,
) -> Path:
    config_path = config_path.resolve()
    pysr_config_path = pysr_config_path.resolve()
    sidecar = json.loads(pysr_config_path.read_text(encoding="utf-8"))
    pysr_config = sidecar.get("pysr_baseline", sidecar)
    if not isinstance(pysr_config, dict):
        raise ValueError(f"PySR sidecar must contain a JSON object: {pysr_config_path}")
    if not bool(pysr_config.get("enabled", False)):
        raise ValueError(f"PySR is disabled in {pysr_config_path}")

    config = pipeline.load_config(config_path, pipeline_mode=3)
    config["pysr_baseline"] = copy.deepcopy(pysr_config)
    config = pipeline._apply_pysr_reproducibility_mode(config)
    main_output_root = pipeline.resolve_path(config["output"]["directory"])
    scenario_root = (
        main_output_root.parent
        if main_output_root.name == "main" else main_output_root
    )
    experiment_root = scenario_root / "pysr"
    run_dir = pipeline._next_mode4_experiment_directory(
        experiment_root, prefix="run"
    )
    base_seed = int(config.get("seed", 0))
    split_cfg = config.get("data", {}).get("split", {}) or {}
    data_cfg = config["data"]
    segmentation_cfg = (
        data_cfg.get("realdata", {}).get("segmentation", {}) or {}
    )
    is_realdata_mat = pipeline._is_measured_pathloss_profile(data_cfg)

    for index in range(int(seed_count)):
        seed = base_seed + index
        seed_config = copy.deepcopy(config)
        seed_config["seed"] = seed
        seed_config["pysr_cache"] = {"reuse": False, "save": False}
        train_seed = int(split_cfg.get("seed", seed))
        loaded = pipeline.load_configured_data(data_cfg)
        if (
            is_realdata_mat
            and bool(segmentation_cfg.get("enabled", False))
            and bool((segmentation_cfg.get("breakpoint_prior", {}) or {}).get("enabled", False))
        ):
            train_data, validation_data, test_data, _split_metadata = (
                pipeline._split_by_piecewise_segments(
                    loaded, train_seed, segmentation_cfg,
                    split_cfg.get("train_fraction", 0.5),
                    split_cfg.get("validation_fraction", 0.25),
                    split_cfg.get("test_fraction", 0.25),
                )
            )
        else:
            train_data, validation_data, test_data, _split_metadata = (
                pipeline._split_train_validation_test_data(
                    loaded, train_seed,
                    split_cfg.get("train_fraction", 0.5),
                    split_cfg.get("validation_fraction", 0.25),
                    split_cfg.get("test_fraction", 0.25),
                )
            )
        X, y, feature_names, dataset_groups = train_data
        seed_output = run_dir / f"seed_{index + 1:03d}"
        seed_output.mkdir(parents=True, exist_ok=True)
        split_indices_path = pipeline.save_split_indices(
            seed_output, _split_metadata, seed=seed,
            config_path=config_path, data_mode=data_cfg.get("mode", "csv"),
            data_profile=pipeline._configured_data_profile(data_cfg),
            split_config=split_cfg,
        )
        print(f"Saved train/validation/test row indices to {split_indices_path}", flush=True)
        seed_config["output"] = {
            **seed_config.get("output", {}), "directory": str(seed_output),
        }
        (run_dir / f"seed_{index + 1:03d}_config.json").write_text(
            json.dumps(seed_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        payload = {"metadata": {"seed": seed, "config_path": str(config_path)}}
        result = pysr_baseline.run_pysr_stage(
            seed_output, seed_output, X, y, feature_names, dataset_groups,
            seed_config, seed, data_cfg, is_realdata_mat,
            bool(is_realdata_mat and segmentation_cfg.get("enabled", False)),
            segmentation_cfg, payload, 3, validation_data, test_data,
            validation_data[0], validation_data[1], validation_data[3],
        )
        print(
            f"PySR seed={seed} complete: {result['metadata'].get('metadata_json', 'no metadata path')}",
            flush=True,
        )
    has_test_front = any(
        (seed_dir / "pysr_baseline").rglob("test_pareto_front.csv")
        for seed_dir in run_dir.glob("seed_[0-9][0-9][0-9]")
        if (seed_dir / "pysr_baseline").is_dir()
    )
    if has_test_front:
        print("Canonicalizing PySR results only...", flush=True)
        canonicalizer.main(run_dir, model="pysr")
    else:
        print(
            "No PySR test Pareto fronts were produced; skipping PySR canonicalization.",
            flush=True,
        )
    print(f"PySR experiment complete: {run_dir}", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="main experiment config")
    parser.add_argument("--pysr-config", type=Path, help="PySR sidecar config")
    parser.add_argument("--seed-count", type=int, default=10)
    args = parser.parse_args()
    if args.seed_count < 1:
        parser.error("--seed-count must be positive")
    sidecar = args.pysr_config or (
        PROJECT_ROOT / "configs" / "experiments" / "pysr" / args.config.name
    )
    run_pysr_experiment(args.config, sidecar, seed_count=args.seed_count)


if __name__ == "__main__":
    main()
