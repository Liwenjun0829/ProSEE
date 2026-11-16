"""Shared dataset loading and project-relative path utilities."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent


def resolve_path(value: str | Path, base: Path = ROOT) -> Path:
    """Resolve relative configuration paths against this script directory."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def validate_xy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or not 1 <= X.shape[1] <= 4:
        raise ValueError(f"X must be a 2D array with 1–4 columns; got shape={X.shape}")
    if len(X) != len(y) or len(y) < 3:
        raise ValueError("X and y must have the same number of rows and at least 3 samples")
    finite = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if not np.all(finite):
        logging.warning("Discarded %d rows containing NaN/Inf values", int(np.sum(~finite)))
        X, y = X[finite], y[finite]
    if len(y) < 3:
        raise ValueError("Fewer than 3 valid samples remain after removing NaN/Inf values")
    if np.std(y) == 0:
        raise ValueError("Target column cannot contain only identical values")
    return X, y


def load_csv_data(path: Path, features: Sequence[str] | None, target: str | None):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        rows = list(reader)
    if not headers or not rows:
        raise ValueError(f"CSV is empty or has no header: {path}")
    target = target or ("path_loss" if "path_loss" in headers else headers[-1])
    if target not in headers:
        raise ValueError(f"Target column {target!r} is not in the CSV; available columns: {headers}")
    if features:
        feature_names = list(features)
    else:
        # Real-world CSV files often also contain file names, IDs or labels.
        # Infer only columns whose non-empty values are all numeric.
        feature_names = []
        for name in headers:
            if name == target:
                continue
            try:
                if rows and all(str(row[name]).strip() != "" and
                                np.isfinite(float(row[name])) for row in rows):
                    feature_names.append(name)
            except (TypeError, ValueError):
                continue
    if not 1 <= len(feature_names) <= 4:
        raise ValueError(f"There must be 1–4 feature columns; currently selected automatically/explicitly: {feature_names}")
    missing = [name for name in feature_names if name not in headers]
    if missing:
        raise ValueError(f"CSV is missing feature columns: {missing}")
    try:
        X = np.asarray([[float(row[name]) for name in feature_names] for row in rows])
        y = np.asarray([float(row[target]) for row in rows])
    except (TypeError, ValueError) as exc:
        raise ValueError("Selected feature/target columns must be numeric") from exc
    X, y = validate_xy(X, y)
    return X, y, feature_names, target
