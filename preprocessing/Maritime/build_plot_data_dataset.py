"""Convert the measured Maritime MAT arrays into ProSEE's CSV schema.

The output preserves the existing realdata_mat loader's row policy exactly:
remove rows with non-finite distance/path-loss values, then stable-sort by
distance. Values use 17 significant digits so the CSV round-trips to the same
IEEE-754 floats.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "Maritime" / "plot_data2.mat"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "Maritime" / "plot_data2.csv"
FIELDS = ("distance", "path_loss")


def build_dataset(
    input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT,
) -> tuple[np.ndarray, np.ndarray]:
    mat_data = loadmat(str(input_path))
    required = ("distance_meas", "pathloss_meas")
    missing = [name for name in required if name not in mat_data]
    if missing:
        raise KeyError(f"MAT file is missing required arrays: {missing}")

    distance = np.asarray(mat_data[required[0]], dtype=float).squeeze()
    path_loss = np.asarray(mat_data[required[1]], dtype=float).squeeze()
    if (distance.ndim != 1 or path_loss.ndim != 1
            or len(distance) != len(path_loss)):
        raise ValueError("distance_meas and pathloss_meas must be equal-length 1D arrays")

    valid = np.isfinite(distance) & np.isfinite(path_loss)
    distance, path_loss = distance[valid], path_loss[valid]
    if len(distance) < 3:
        raise ValueError("At least 3 valid measured samples are required")
    order = np.argsort(distance, kind="stable")
    distance, path_loss = distance[order], path_loss[order]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(
            (format(float(x), ".17g"), format(float(y), ".17g"))
            for x, y in zip(distance, path_loss)
        )
    return distance, path_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    distance, _path_loss = build_dataset(args.input, args.output)
    print(f"Wrote {len(distance)} rows to {args.output}")


if __name__ == "__main__":
    main()
