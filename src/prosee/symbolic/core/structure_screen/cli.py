"""Shared CSV loader and command-line entry point for structure screening."""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "structure_screen" / "legacy.json"

import numpy as np


def data(
    path: str | Path,
    target: str,
    features: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Load selected numeric feature/target columns from a headed CSV file."""
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        missing = [name for name in [*features, target] if name not in headers]
        if missing:
            raise ValueError(f"CSV 缺少列 {missing}；可用列：{headers}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV 没有数据行：{csv_path}")
    X = np.asarray([[float(row[name]) for name in features] for row in rows], dtype=float)
    y = np.asarray([float(row[target]) for row in rows], dtype=float)
    finite = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if int(finite.sum()) < 3:
        raise ValueError("CSV 清理 NaN/Inf 后不足 3 条有效数据")
    return X[finite], y[finite]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练公式结构模型或执行 GP/LM 筛选")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--input", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    screen_parser = commands.add_parser("screen")
    screen_parser.add_argument("--model-dir", type=Path, required=True)
    screen_parser.add_argument("--data", type=Path, required=True)
    screen_parser.add_argument("--features", nargs="+", required=True)
    screen_parser.add_argument("--target", required=True)
    screen_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "train":
        from .train import train
        result = train(args.input, args.output_dir, config)
    else:
        from .tree_search_ops import run_gp
        from .train import load_predictor
        X, y = data(args.data, args.target, args.features)
        result = run_gp(
            load_predictor(args.model_dir), X, y, args.output,
            config["gp"], config["lm"], config.get("seed", 42),
            variables=[f"x{i + 1}" for i in range(X.shape[1])],
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
