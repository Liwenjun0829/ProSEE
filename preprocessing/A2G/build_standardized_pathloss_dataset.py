"""Build a standardized path-loss CSV from measured_dataset__0123.xlsx.

The standardized schema is copied from the existing dataset:
file_name, rx_height, tx_height, path_loss, tx_rx_distance

The workbook's first two columns are labeled lat/lon.  Although the task
description lists longitude before latitude, this script follows the explicit
labels in the workbook because the values are in the valid latitude/longitude
ranges and the receiver row uses the same order.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA = ["file_name", "rx_height", "tx_height", "path_loss", "tx_rx_distance"]
EARTH_RADIUS_M = 6_371_008.8  # mean Earth radius, IUGG value
FREQUENCY_HZ = 3.60e9  # fixed by the task; not a column in the reference schema
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "A2G"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "A2G"


def haversine_distance_m(lat1, lon1, lat2, lon2):
    """Return WGS84-coordinate horizontal distance using the haversine method."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_dataset(
    input_path: Path,
    output_path: Path = PROCESSED_DIR / "standardized_pathloss_dataset_measured_0123.csv",
    takeoff_output_path: Path = PROCESSED_DIR / "standardized_pathloss_dataset_measured_0123_takeoff.csv",
    cruise_output_path: Path = PROCESSED_DIR / "standardized_pathloss_dataset_measured_0123_cruise.csv",
    landing_output_path: Path = PROCESSED_DIR / "standardized_pathloss_dataset_measured_0123_landing.csv",
) -> pd.DataFrame:
    """Build the complete dataset and its flight-phase CSVs."""

    raw = pd.read_excel(input_path, header=None)
    if raw.shape[1] != 155:
        raise ValueError(f"Expected 155 columns, found {raw.shape[1]}")

    # Excel row 3 (zero-based index 2) stores the receiver GPS and height.
    rx_lat = float(raw.iloc[2, 0])
    rx_lon = float(raw.iloc[2, 1])
    rx_height = float(raw.iloc[2, 2])

    # Excel row 4 onward is measurement data; columns 5-155 are 151 CIR bins.
    data = raw.iloc[3:].copy()
    lat = pd.to_numeric(data.iloc[:, 0], errors="raise").to_numpy(dtype=float)
    lon = pd.to_numeric(data.iloc[:, 1], errors="raise").to_numpy(dtype=float)
    tx_height = pd.to_numeric(data.iloc[:, 2], errors="raise").to_numpy(dtype=float)
    cir_db = data.iloc[:, 4:155].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if cir_db.shape[1] != 151:
        raise ValueError(f"Expected 151 CIR bins, found {cir_db.shape[1]}")

    # Convert every CIR bin from dB to linear power before summing.
    power_sum = np.sum(np.power(10.0, cir_db / 10.0), axis=1)
    path_loss = -10.0 * np.log10(power_sum)

    horizontal_distance = haversine_distance_m(lat, lon, rx_lat, rx_lon)
    distance = np.sqrt(horizontal_distance**2 + (tx_height - rx_height) ** 2)

    result = pd.DataFrame(
        {
            "file_name": [f"{input_path.stem}_row_{i:06d}" for i in range(1, len(data) + 1)],
            "rx_height": rx_height,
            "tx_height": tx_height,
            "path_loss": path_loss,
            "tx_rx_distance": distance,
        },
        columns=SCHEMA,
    )

    # Keep the original time order for the flight-phase files.
    phase_counts = {"takeoff": 100, "cruise": 717, "landing": 234}
    expected_count = sum(phase_counts.values())
    if len(result) != expected_count:
        raise ValueError(
            f"Expected {expected_count} measured samples for phase split, found {len(result)}"
        )
    takeoff = result.iloc[: phase_counts["takeoff"]].reset_index(drop=True)
    cruise_end = phase_counts["takeoff"] + phase_counts["cruise"]
    cruise = result.iloc[phase_counts["takeoff"] : cruise_end].reset_index(drop=True)
    landing = result.iloc[cruise_end:].reset_index(drop=True)

    if [len(takeoff), len(cruise), len(landing)] != list(phase_counts.values()):
        raise RuntimeError("Flight-phase split produced unexpected sample counts")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, float_format="%.12g")

    for output_path, phase_data in (
        (takeoff_output_path, takeoff),
        (cruise_output_path, cruise),
        (landing_output_path, landing),
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        phase_data.to_csv(output_path, index=False, float_format="%.12g")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, nargs="?", default=RAW_DIR / "measured_dataset__0123.xlsx"
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=PROCESSED_DIR / "standardized_pathloss_dataset_measured_0123.csv",
    )
    parser.add_argument(
        "--takeoff-output",
        type=Path,
        default=PROCESSED_DIR / "standardized_pathloss_dataset_measured_0123_takeoff.csv",
        help="Output CSV for the first 100 samples",
    )
    parser.add_argument(
        "--cruise-output",
        type=Path,
        default=PROCESSED_DIR / "standardized_pathloss_dataset_measured_0123_cruise.csv",
        help="Output CSV for samples 101-817",
    )
    parser.add_argument(
        "--landing-output",
        type=Path,
        default=PROCESSED_DIR / "standardized_pathloss_dataset_measured_0123_landing.csv",
        help="Output CSV for the final 234 samples",
    )
    args = parser.parse_args()
    result = build_dataset(
        args.input,
        args.output,
        takeoff_output_path=args.takeoff_output,
        cruise_output_path=args.cruise_output,
        landing_output_path=args.landing_output,
    )

    print("output columns:", list(result.columns))
    print("dataset shape:", result.shape)
    print("takeoff shape: (100, 5)")
    print("cruise shape: (717, 5)")
    print("landing shape: (234, 5)")
    print("phase total samples: 1051")
    print("first 5 rows:")
    print(result.head(5).to_string(index=False))
    print("distance min/max/mean:", result["tx_rx_distance"].min(), result["tx_rx_distance"].max(), result["tx_rx_distance"].mean())
    print("path_loss min/max/mean:", result["path_loss"].min(), result["path_loss"].max(), result["path_loss"].mean())
    print("contains NaN:", bool(result.isna().any().any()))
    print("contains Inf:", bool(np.isinf(result.select_dtypes(include=[np.number])).to_numpy().any()))


if __name__ == "__main__":
    main()
