from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parses command line arguments for time-series filtering."""
    parser = argparse.ArgumentParser(description="Filter ionospheric station time-grid CSV files.")
    parser.add_argument("--input-dir", default="normalized_2024_2025_exploration_v0_1_15min_by_station")
    parser.add_argument("--config", default="configs/time_series_filtering.json")
    parser.add_argument("--output-dir", default="normalized_2024_2025_exploration_v0_1_15min_filtered_by_station")
    parser.add_argument("--station", default=None, help="Optional station code for a quick run.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Loads filtering settings from JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def rolling_mad(values: pd.Series, window: int) -> pd.Series:
    """Computes rolling median absolute deviation."""
    median = values.rolling(window=window, center=True, min_periods=1).median()
    deviation = (values - median).abs()
    return deviation.rolling(window=window, center=True, min_periods=1).median()


def hampel_filter(values: pd.Series, window: int, n_sigma: float) -> tuple[pd.Series, pd.Series]:
    """Replaces local outliers with rolling median values."""
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.rolling(window=window, center=True, min_periods=1).median()
    mad = rolling_mad(numeric, window)
    scale = 1.4826 * mad
    outliers = (numeric - median).abs() > (n_sigma * scale)
    outliers = outliers & scale.notna() & (scale > 0)
    filtered = numeric.mask(outliers, median)
    return filtered, outliers


def rolling_median_filter(values: pd.Series, window: int) -> pd.Series:
    """Applies short rolling median smoothing."""
    numeric = pd.to_numeric(values, errors="coerce")
    smoothed = numeric.rolling(window=window, center=True, min_periods=1).median()
    return smoothed.where(numeric.notna(), np.nan)


def filter_column(values: pd.Series, config: dict[str, Any]) -> tuple[pd.Series, int]:
    """Applies configured filters to one numeric column."""
    current = pd.to_numeric(values, errors="coerce")
    changed_points = 0

    hampel = config.get("methods", {}).get("hampel", {})
    if hampel.get("enabled", False):
        current, outliers = hampel_filter(
            current,
            window=int(hampel.get("window", 9)),
            n_sigma=float(hampel.get("n_sigma", 3.0)),
        )
        changed_points += int(outliers.sum())

    rolling = config.get("methods", {}).get("rolling_median", {})
    if rolling.get("enabled", False):
        before = current.copy()
        current = rolling_median_filter(current, window=int(rolling.get("window", 3)))
        changed_points += int(((before != current) & before.notna() & current.notna()).sum())

    return current, changed_points


def filter_station_file(path: Path, output_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Filters one station CSV and writes the result."""
    frame = pd.read_csv(path)
    columns = [column for column in config.get("columns", []) if column in frame.columns]
    suffix = str(config.get("filtered_suffix", "_filtered"))
    keep_original = bool(config.get("keep_original_columns", True))
    replace_columns = bool(config.get("replace_columns", True))

    summary: dict[str, Any] = {
        "station_file": path.name,
        "rows": int(len(frame)),
        "filtered_columns": [],
        "changed_points": {},
    }

    for column in columns:
        original = pd.to_numeric(frame[column], errors="coerce")
        filtered, changed_points = filter_column(original, config)
        if keep_original:
            frame[f"{column}_raw"] = original
        if replace_columns:
            frame[column] = filtered
        else:
            frame[f"{column}{suffix}"] = filtered
        summary["filtered_columns"].append(column)
        summary["changed_points"][column] = int(changed_points)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    summary["output"] = str(output_path)
    return summary


def copy_small_root_files(input_dir: Path, output_dir: Path) -> None:
    """Copies small root CSV/Markdown files that describe the normalized dataset."""
    for name in ["station_coverage.csv", "time_normalization_report.md"]:
        source = input_dir / name
        if source.exists():
            target = output_dir / name
            target.write_bytes(source.read_bytes())


def main() -> None:
    """Runs filtering for all station files or for one selected station."""
    args = parse_args()
    input_dir = Path(args.input_dir)
    station_dir = input_dir / "stations"
    output_dir = Path(args.output_dir)
    output_station_dir = output_dir / "stations"
    config = load_config(Path(args.config))

    paths = sorted(station_dir.glob("*_time_grid.csv"))
    if args.station:
        paths = [path for path in paths if path.name == f"{args.station}_time_grid.csv"]
    if not paths:
        raise FileNotFoundError(f"No station CSV files found in {station_dir}.")

    summaries = []
    for path in paths:
        output_path = output_station_dir / path.name
        summary = filter_station_file(path, output_path, config)
        summaries.append(summary)
        total_changed = sum(summary["changed_points"].values())
        print(f"{path.stem}: filtered_columns={len(summary['filtered_columns'])} changed_points={total_changed}")

    output_dir.mkdir(parents=True, exist_ok=True)
    copy_small_root_files(input_dir, output_dir)
    (output_dir / "filtering_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "filtering_manifest.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    report = pd.DataFrame(
        [
            {
                "station_file": item["station_file"],
                "rows": item["rows"],
                "filtered_columns": len(item["filtered_columns"]),
                "changed_points_total": sum(item["changed_points"].values()),
            }
            for item in summaries
        ]
    )
    report.to_csv(output_dir / "filtering_summary.csv", index=False)


if __name__ == "__main__":
    main()
