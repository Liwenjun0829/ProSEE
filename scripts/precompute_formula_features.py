#!/usr/bin/env python3
"""Generate the frozen formula library and feature matrix for the pipeline.

This is the only command that is allowed to generate the formula pool or fit
the formula-feature encoders.  ``formula_discovery_pipeline.py`` loads the
resulting Joblib package and exits if it is absent or incompatible.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = PROJECT_ROOT / "src" / "prosee" / "symbolic" / "core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from formula_discovery_pipeline import (
    ROOT,
    build_base_records,
    build_candidate_records,
    build_combination_records,
    formula_generation_signature,
)
from structure_r2_ranker import build_precomputed_feature_artifact
from data_io import resolve_path

def _load_generation_config(generation_config_path: Path) -> tuple[dict, dict]:
    """Load all formula-pool and precomputed-feature settings from one file."""
    generation_config = json.loads(
        generation_config_path.resolve().read_text(encoding="utf-8")
    )
    config = {key: generation_config[key] for key in (
        "seed", "bases", "candidates", "combinations")}
    config["ranking"] = dict(generation_config["features"])
    return config, generation_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="预生成并冻结公式库、公式特征和特征编码器"
    )
    parser.add_argument("--generation-config", type=Path,
                        default=PROJECT_ROOT / "configs" / "formula_discovery" / "formula_generation.json",
                        help="独立的公式库/特征生成配置")
    parser.add_argument("--output", type=Path, default=None,
                        help="Joblib 输出路径；默认读取配置中的预计算文件路径")
    args = parser.parse_args()

    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError(
            "预计算需要 joblib；请使用与 formula_discovery_pipeline 相同的环境运行"
        ) from exc

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    config, generation_config = _load_generation_config(args.generation_config)
    ranking_config = config["ranking"]
    formula_seed = int(config["seed"])
    feature_seed = int(generation_config["feature_seed"])

    output_value = (
        args.output
        or generation_config.get("output", {}).get("artifact")
    )
    if not output_value:
        raise ValueError(
            "请通过 --output 或 generation config 的 output.artifact 指定输出文件"
        )
    output_path = Path(output_value)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # This artifact is the two-variable formula/feature library. Its symbolic
    # inputs are x1 and x2, so generation does not need to load a dataset.
    feature_count = 2
    feature_names = ["x1", "x2"]
    import random
    rng = random.Random(formula_seed)
    bases = build_base_records(
        config, feature_count,
        include_symbolic_constant_base=bool(
            generation_config["include_symbolic_constant_base"]
        ),
    )
    candidates = build_candidate_records(bases, config, rng)
    # Store a canonical order in the frozen artifact.  Runtime pipeline seeds
    # perform the reproducible exploration shuffle after loading it.
    candidates.sort(key=lambda row: (str(row["readable_expression"]), str(row["sympy_expression"])))
    for index, record in enumerate(candidates):
        record["id"] = f"candidate_{index:06d}"
    combinations = build_combination_records(
        candidates, config, rng, shuffle_records=False
    )
    pool = [*candidates, *combinations]
    payload = {"bases": bases}

    logging.info(
        "公式库生成完成：bases=%d，single=%d，combinations=%d，总计=%d",
        len(bases), len(candidates), len(combinations), len(pool),
    )
    feature_artifact = build_precomputed_feature_artifact(
        pool, payload, config["ranking"], feature_seed
    )
    artifact = {
        **feature_artifact,
        "formula_generation_signature": formula_generation_signature(
            config, feature_count,
            include_symbolic_constant_base=bool(
                generation_config["include_symbolic_constant_base"]
            ),
        ),
        "include_symbolic_constant_base": bool(
            generation_config["include_symbolic_constant_base"]
        ),
        "formula_generation_seed": formula_seed,
        "feature_precompute_seed": feature_seed,
        "generation_config_path": str(args.generation_config.resolve()),
        "feature_names": list(feature_names),
        "bases": bases,
        "single_candidates": candidates,
        "binary_combinations": combinations,
    }

    temporary_path = output_path.with_name(output_path.name + ".tmp")
    joblib.dump(artifact, temporary_path, compress=3)
    temporary_path.replace(output_path)

    formula_json_path = output_path.with_suffix(".formulas.json")
    formula_json_path.write_text(
        json.dumps({
            "version": 2,
            "formula_generation_signature": artifact["formula_generation_signature"],
            "include_symbolic_constant_base": bool(
                generation_config["include_symbolic_constant_base"]
            ),
            "formula_generation_seed": formula_seed,
            "feature_precompute_seed": feature_seed,
            "generation_config_path": str(args.generation_config.resolve()),
            "feature_count": feature_count,
            "feature_names": list(feature_names),
            "bases": bases,
            "single_candidates": candidates,
            "binary_combinations": combinations,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logging.info("Joblib 公式/特征文件：%s", output_path)
    logging.info("可读公式 JSON 文件：%s", formula_json_path)
    logging.info("特征矩阵形状：%s", tuple(feature_artifact["features"].shape))


if __name__ == "__main__":
    main()
