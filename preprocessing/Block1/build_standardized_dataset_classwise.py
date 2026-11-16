"""Build class-wise LHS path-loss datasets from XGBoost residual labels.

Workflow: select an XGBoost regressor by validation RMSE -> make out-of-fold
residual labels -> use those labels directly as the final Class-1/Class-2
assignment -> LHS sample each class independently -> visualize one manually
entered path-loss formula per class.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import qmc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import sympy as sp


FEATURES = ["tx_rx_distance", "rx_height"]
TARGET = "path_loss"
FREQUENCIES = ("f1p4GHz_V", "f5p9GHz_V", "f28p0GHz_V")

# Edit these two strings directly.  d = Tx-Rx distance and h = Rx height.
# Supported operators/functions: +, -, *, /, ** (or ^), log, exp, sqrt, abs.
MANUAL_CLASS_FORMULAS = {
    1: "60 + 0.25*d + 0.02*h",
    2: "65 + 0.18*d + 0.03*h",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create classified Class-1/Class-2 CSV datasets.")
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--input-root", type=Path, default=root / "data/raw/Block1")
    parser.add_argument("--tx-pose", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=root / "data/processed/Block1")
    parser.add_argument("--datasets", nargs="+", choices=FREQUENCIES, default=list(FREQUENCIES))
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--oof-folds", type=int, default=5)
    parser.add_argument("--mad-z", type=float, default=5.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def raw_frequency_rows(folder: Path, tx_pose: Path) -> pd.DataFrame:
    """Integrate NPZ channel coefficients and receiver positions into model features."""
    with tx_pose.open("r", encoding="utf-8") as f:
        pose = json.load(f)
    p = pose["world_transform"]["position"]
    tx = np.array([p["x"], p["y"], p["z"]], dtype=float)
    files = sorted(folder.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No NPZ files found: {folder}")
    rows = []
    skipped = 0
    for path in files:
        try:
            with np.load(path, allow_pickle=False) as z:
                if not {"a_real", "a_imag", "uav_pos"}.issubset(z.files):
                    skipped += 1
                    continue
                rx = np.asarray(z["uav_pos"], dtype=float)
                rx = rx[:3] if rx.ndim == 1 else rx.reshape(-1, rx.shape[-1])[0, :3]
                power = np.sum(np.asarray(z["a_real"], dtype=float)**2 + np.asarray(z["a_imag"], dtype=float)**2)
            if not np.isfinite(power) or power <= 0 or not np.isfinite(rx).all():
                skipped += 1
                continue
            rows.append({"file_name": path.name, "rx_height": float(rx[2]),
                         "tx_height": float(tx[2]), "path_loss": float(-10*np.log10(power)),
                         "tx_rx_distance": float(np.linalg.norm(rx-tx))})
        except (OSError, ValueError, KeyError, IndexError):
            skipped += 1
    print(f"{folder.name}: {len(rows)}/{len(files)} valid raw records ({skipped} skipped)")
    return pd.DataFrame(rows)


def make_xgb(params: dict[str, int | float], seed: int) -> XGBRegressor:
    return XGBRegressor(objective="reg:squarederror", n_jobs=1, random_state=seed,
                        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, **params)


def choose_xgb(x: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> tuple[dict[str, int | float], float]:
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=args.validation_size, random_state=args.random_state)
    candidates = [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr}
        # Kept identical to visualize_kmeans_clusters.py.
        for n in (200, 400, 800) for d in (3, 6, 9) for lr in (0.03, 0.05, 0.10)
    ]
    scored = []
    for params in candidates:
        model = make_xgb(params, args.random_state).fit(x_train, y_train)
        scored.append((float(np.sqrt(mean_squared_error(y_val, model.predict(x_val)))), params))
    best_rmse, best_params = min(scored, key=lambda item: item[0])
    return best_params, best_rmse


def xgb_oof_prediction(x: np.ndarray, y: np.ndarray, params: dict[str, int | float], args: argparse.Namespace) -> np.ndarray:
    pred = np.full(len(y), np.nan)
    folds = KFold(n_splits=args.oof_folds, shuffle=True, random_state=args.random_state)
    for fold, (train, test) in enumerate(folds.split(x)):
        pred[test] = make_xgb(params, args.random_state + fold).fit(x[train], y[train]).predict(x[test])
    return pred


def high_error_labels(error: np.ndarray, mad_z: float) -> tuple[np.ndarray, float]:
    median = float(np.median(error))
    mad = float(np.median(np.abs(error - median)))
    threshold = median + mad_z * 1.4826 * mad if mad else float(np.quantile(error, 0.95))
    return (error > threshold).astype(int), threshold


def fit_reference_boundary(x: np.ndarray, xgb_class2: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    """Fit a descriptive LR boundary only; it never changes the XGBoost labels."""
    model = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=seed)),
    ]).fit(x, xgb_class2)
    scale, lr = model.named_steps["scale"], model.named_steps["lr"]
    coefficient = lr.coef_[0] / scale.scale_
    intercept = float(lr.intercept_[0] - np.sum(lr.coef_[0] * scale.mean_ / scale.scale_))
    return coefficient, intercept


def lhs_indices(x: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Two-dimensional LHS targets, mapped to unique nearest observed X rows."""
    n = len(x)
    if count >= n:
        return np.arange(n)
    low, high = x.min(axis=0), x.max(axis=0)
    targets = qmc.LatinHypercube(d=2, seed=seed).random(count) * (high - low) + low
    # Standardize coordinates before nearest-neighbour matching so both axes matter.
    scale = np.maximum(x.std(axis=0), 1e-12)
    tree = cKDTree(x / scale)
    _, neighbours = tree.query(targets / scale, k=min(32, n))
    selected: list[int] = []
    used: set[int] = set()
    for candidates in np.atleast_2d(neighbours):
        for idx in np.atleast_1d(candidates):
            idx = int(idx)
            if idx not in used:
                selected.append(idx); used.add(idx); break
    if len(selected) < count:
        remaining = np.setdiff1d(np.arange(n), np.array(selected, dtype=int), assume_unique=False)
        rng = np.random.default_rng(seed)
        selected.extend(rng.choice(remaining, size=count - len(selected), replace=False).tolist())
    return np.asarray(selected, dtype=int)


def manual_formula(expression: str, x: np.ndarray) -> tuple[callable, np.ndarray, str]:
    """Parse a user-supplied formula; this function never estimates coefficients."""
    d, h = sp.symbols("d h")
    allowed = {"d": d, "h": h, "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt, "abs": sp.Abs}
    try:
        symbolic = sp.sympify(expression.replace("^", "**"), locals=allowed)
        unknown = symbolic.free_symbols.difference({d, h})
        if unknown:
            raise ValueError(f"unknown variables: {', '.join(map(str, unknown))}")
        function = sp.lambdify((d, h), symbolic, modules="numpy")
        prediction = np.asarray(function(x[:, 0], x[:, 1]), dtype=float)
        if prediction.ndim == 0:
            prediction = np.full(len(x), float(prediction))
        if prediction.shape != (len(x),) or not np.isfinite(prediction).all():
            raise ValueError("formula must produce one finite prediction per row")
    except Exception as exc:
        raise ValueError(f"Invalid manual formula {expression!r}: {exc}") from exc
    return function, prediction, str(symbolic)


def plot_xgb_classes(x: np.ndarray, labels: np.ndarray, coefficient: np.ndarray, intercept: float, output: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.scatter(x[labels == 0, 0], x[labels == 0, 1], s=7, alpha=.35, c="tab:blue", label="Class 1")
    ax.scatter(x[labels == 1, 0], x[labels == 1, 1], s=12, alpha=.85, c="tab:red", label="Class 2 (XGBoost OOF high residual)")
    distance = np.linspace(x[:, 0].min(), x[:, 0].max(), 300)
    if abs(coefficient[1]) > 1e-12:
        height = -(intercept + coefficient[0] * distance) / coefficient[1]
        ax.plot(distance, height, color="black", linewidth=2, label="Logistic reference boundary (not used for class)")
    ax.set(title="XGBoost final classes with logistic reference boundary", xlabel="Tx-Rx distance (m)", ylabel="Rx height (m)")
    ax.legend(); fig.tight_layout(); fig.savefig(output, dpi=dpi, bbox_inches="tight"); plt.close(fig)


def plot_response_distribution(data: pd.DataFrame, output: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, feature, label in zip(axes, FEATURES, ["Tx-Rx distance (m)", "Rx height (m)"]):
        for cls, color in [(1, "tab:blue"), (2, "tab:red")]:
            part = data[data["final_class"] == cls]
            ax.scatter(part[feature], part[TARGET], s=8, alpha=.42, c=color, label=f"Class {cls}")
        ax.set(xlabel=label, ylabel="Path loss (dB)", title=f"Path loss vs. {label}"); ax.legend()
    fig.tight_layout(); fig.savefig(output, dpi=dpi, bbox_inches="tight"); plt.close(fig)


def plot_formula(data: pd.DataFrame, cls: int, predict, formula: str, r2: float, rmse: float, output: Path, dpi: int) -> None:
    x = data[FEATURES].to_numpy(float); y = data[TARGET].to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, index, label in zip(axes, [0, 1], ["Tx-Rx distance (m)", "Rx height (m)"]):
        order = np.argsort(x[:, index]); line_x = x[order, index]
        probe = np.tile(np.median(x, axis=0), (len(x), 1)); probe[:, index] = line_x
        ax.scatter(x[:, index], y, s=9, alpha=.45, c="tab:blue", label=f"Class {cls} data")
        ax.plot(line_x, predict(probe[:, 0], probe[:, 1]), c="tab:orange", lw=2, label="Manual formula (other X at median)")
        ax.set(xlabel=label, ylabel="Path loss (dB)"); ax.legend()
    fig.suptitle(f"Class {cls} formula fit | R²={r2:.4f}, RMSE={rmse:.4f} dB\n{formula}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, .88]); fig.savefig(output, dpi=dpi, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.mad_z <= 0 or not 0 < args.validation_size < 1: raise ValueError("Invalid MAD-z or validation-size argument.")
    tx_pose = args.tx_pose or args.input_root / "RoofLidar_world_pose.json"
    if not tx_pose.is_file(): raise FileNotFoundError(f"Transmitter pose not found: {tx_pose}")
    # Check all inputs before beginning the potentially long model fitting.
    for name in args.datasets:
        folder = args.input_root / name
        if not folder.is_dir() or not any(folder.glob("*.npz")):
            raise FileNotFoundError(f"Missing NPZ frequency folder: {folder}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for freq_index, name in enumerate(args.datasets):
        raw = raw_frequency_rows(args.input_root / name, tx_pose)
        numeric = raw[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce")
        valid = np.isfinite(numeric.to_numpy(float)).all(axis=1)
        data = raw.loc[valid].copy().reset_index(drop=True)
        x, y = numeric.loc[valid, FEATURES].to_numpy(float), numeric.loc[valid, TARGET].to_numpy(float)
        if len(data) < args.oof_folds * 2:
            raise ValueError(f"Too few finite rows for OOF prediction in {name}: {len(data)}")

        # Use a distinct but deterministic seed for each carrier frequency.
        seed = args.random_state + freq_index
        local_args = argparse.Namespace(**vars(args))
        local_args.random_state = seed
        params, validation_rmse = choose_xgb(x, y, local_args)
        oof_pred = xgb_oof_prediction(x, y, params, local_args)
        residual = np.abs(y - oof_pred)
        class2, threshold = high_error_labels(residual, args.mad_z)
        data["xgboost_oof_prediction"] = oof_pred
        data["xgboost_oof_absolute_error"] = residual
        data["final_class"] = class2 + 1

        # Keep outputs for each band together and identify their frequency in each filename.
        output_dir = args.output_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_dir / f"{name}_all_final_classified.csv", index=False, encoding="utf-8-sig")
        for cls in (1, 2):
            data.loc[data["final_class"] == cls].to_csv(
                output_dir / f"{name}_class{cls}_all.csv", index=False, encoding="utf-8-sig")
        print(f"{name}: saved {len(data)} classified records to {output_dir}; "
              f"class counts={dict(zip(*np.unique(data['final_class'], return_counts=True)))}; "
              f"residual threshold={threshold:.6g}, validation RMSE={validation_rmse:.6g}")


if __name__ == "__main__":
    main()
