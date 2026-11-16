
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def check_valid_flags(npz_folder: Path) -> dict:
    """Check that every NPZ file has a non-empty boolean ``valid`` field of True values."""
    npz_files = sorted(npz_folder.glob("*.npz"))
    report = {
        "folder": npz_folder,
        "file_count": len(npz_files),
        "entry_count": 0,
        "missing_field": [],
        "read_errors": [],
        "false_entries": [],
    }

    for path in npz_files:
        try:
            with np.load(path, allow_pickle=False) as data:
                if "valid" not in data.files:
                    report["missing_field"].append(path.name)
                    continue
                valid = np.asarray(data["valid"])
                if valid.dtype.kind != "b":
                    report["read_errors"].append(
                        (path.name, f"valid has non-boolean dtype {valid.dtype}")
                    )
                    continue
                report["entry_count"] += valid.size
                false_count = int(np.count_nonzero(~valid))
                if false_count:
                    report["false_entries"].append((path.name, false_count))
        except Exception as exc:
            report["read_errors"].append((path.name, str(exc)))

    return report


def validate_datasets(input_root: Path, datasets: list[str]) -> None:
    """Validate all selected datasets before processing any of them."""
    reports = []
    for dataset in datasets:
        folder = input_root / dataset
        if not folder.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")
        report = check_valid_flags(folder)
        if report["file_count"] == 0:
            raise RuntimeError(f"No NPZ files found in dataset folder: {folder}")
        reports.append((dataset, report))

    invalid_reports = []
    for dataset, report in reports:
        print(f"Validity check for {dataset}: {report['file_count']} NPZ files, "
              f"{report['entry_count']} valid entries checked.")
        if (report["entry_count"] == 0 or report["missing_field"]
                or report["read_errors"] or report["false_entries"]):
            invalid_reports.append((dataset, report))

    if invalid_reports:
        print("Invalid data detected; processing has been stopped.")
        for dataset, report in invalid_reports:
            print(
                f"{dataset}: {len(report['missing_field'])} files missing 'valid', "
                f"{len(report['read_errors'])} read/type errors, "
                f"{len(report['false_entries'])} files containing False."
            )
            for name, error in report["read_errors"][:20]:
                print(f"  {name}: {error}")
            for name, count in report["false_entries"][:20]:
                print(f"  {name}: {count} False value(s)")
            if report["missing_field"]:
                print("  Files missing 'valid' (first 20): "
                      + ", ".join(report["missing_field"][:20]))
        raise RuntimeError("Dataset validity check failed. Fix invalid input data and retry.")

    print("All data are valid; processing all datasets.")


def load_tx_position(tx_pose_path: Path) -> np.ndarray:
    """Load fixed transmitter position from RoofLidar world pose JSON file."""
    with tx_pose_path.open("r", encoding="utf-8") as f:
        pose = json.load(f)

    pos = pose["world_transform"]["position"]
    return np.array([pos["x"], pos["y"], pos["z"]], dtype=np.float64)


def compute_path_loss(a_real: np.ndarray, a_imag: np.ndarray) -> float:
    """Compute path loss: -10 * log10(sum(|a|^2))."""
    power_sum = np.sum(a_real.astype(np.float64) ** 2 + a_imag.astype(np.float64) ** 2)
    if not np.isfinite(power_sum) or power_sum <= 0:
        return np.nan
    return float(-10.0 * np.log10(power_sum))


def parse_uav_position(uav_pos: np.ndarray) -> np.ndarray:
    """Extract a single receiver position [x, y, z] from npz field."""
    arr = np.asarray(uav_pos, dtype=np.float64)

    if arr.ndim == 1 and arr.shape[0] >= 3:
        return arr[:3]

    if arr.ndim >= 2 and arr.shape[-1] >= 3:
        flat = arr.reshape(-1, arr.shape[-1])
        return flat[0, :3]

    raise ValueError(f"uav_pos shape not supported: {arr.shape}")


def integrate_npz_folder(npz_folder: Path, tx_pose_path: Path) -> list:
    """Read all npz files and integrate valid records."""
    tx_pos = load_tx_position(tx_pose_path)
    tx_height = float(tx_pos[2])

    rows = []
    npz_files = sorted(npz_folder.glob("*.npz"))

    # Count records rejected for each reason.
    count_missing_keys = 0
    count_path_loss_invalid = 0
    count_uav_pos_error = 0
    count_exception = 0

    for npz_file in npz_files:
        try:
            with np.load(npz_file) as data:
                required_keys = {"a_real", "a_imag", "uav_pos"}
                if not required_keys.issubset(data.files):
                    count_missing_keys += 1
                    continue

                a_real = data["a_real"]
                a_imag = data["a_imag"]
                try:
                    rx_pos = parse_uav_position(data["uav_pos"])
                except Exception:
                    count_uav_pos_error += 1
                    continue

            path_loss = compute_path_loss(a_real, a_imag)
            if not np.isfinite(path_loss):
                count_path_loss_invalid += 1
                continue

            distance = float(np.linalg.norm(rx_pos - tx_pos))
            rx_height = float(rx_pos[2])

            rows.append(
                {
                    "file_name": npz_file.name,
                    "rx_height": rx_height,
                    "tx_height": tx_height,
                    "path_loss": path_loss,
                    "tx_rx_distance": distance,
                }
            )
        except Exception:
            count_exception += 1
            continue

    total = len(npz_files)
    print(f"Total files: {total}")
    print(f"Files missing required fields: {count_missing_keys}")
    print(f"Receiver position parse errors: {count_uav_pos_error}")
    print(f"Invalid path loss values: {count_path_loss_invalid}")
    print(f"Other processing errors: {count_exception}")
    print(f"Valid records: {len(rows)}")

    return rows


def save_standardized_csv(rows: list, output_path: Path, include_anomaly_columns: bool = False) -> None:
    """Save integrated records to a standardized CSV file."""
    fieldnames = ["file_name", "rx_height", "tx_height", "path_loss", "tx_rx_distance"]
    if include_anomaly_columns:
        fieldnames.extend(["anomaly_score", "is_outlier", "anomaly_method"])

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
def main() -> None:
    parser = argparse.ArgumentParser(description="Integrate multi_path npz data to standardized CSV.")
    parser.add_argument(
        "--input_root",
        type=Path,
        default=None,
        help="Root containing the three frequency folders (defaults to data/raw/Square1).",
    )
    parser.add_argument(
        "--tx_pose",
        type=Path,
        default=None,
        help="Fixed transmitter world pose file (defaults to data/raw/Square1/RoofLidar_world_pose.json).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for output CSV files (defaults to data/processed/Square1).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["f1p4GHz_V", "f28p0GHz_V", "f3p5GHz_V"],
        help="Frequency subdirectories to process.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    input_root = args.input_root or project_root / "data" / "raw" / "Square1"
    tx_pose = args.tx_pose or input_root / "RoofLidar_world_pose.json"
    output_dir = args.output_dir or project_root / "data" / "processed" / "Square1"

    if not tx_pose.is_file():
        raise FileNotFoundError(f"tx pose file not found: {tx_pose}")

    validate_datasets(input_root, args.datasets)
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        npz_folder = input_root / dataset
        if not npz_folder.is_dir():
            raise FileNotFoundError(f"npz folder not found: {npz_folder}")

        rows = integrate_npz_folder(npz_folder, tx_pose)
        if not rows:
            raise RuntimeError(f"No valid records found in {npz_folder}.")

        output_path = output_dir / f"standardized_pathloss_dataset_{dataset}.csv"
        save_standardized_csv(rows, output_path)
        print(f"Saved {len(rows)} records to: {output_path}")






if __name__ == "__main__":
    main()






