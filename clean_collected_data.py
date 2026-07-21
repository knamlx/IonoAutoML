from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


MISSING_VALUES = {"", "nan", "na", "n/a", "null", "none", "---", "__"}
DEFAULT_ROW_MISSING_THRESHOLD = 0.30
DEFAULT_COLUMN_MISSING_THRESHOLD = 0.45


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a soft-cleaned copy of collected ionosphere datasets.")
    parser.add_argument("--input-dir", default="data_2024_2025_indices", help="Directory with processed CSV files.")
    parser.add_argument("--giro-raw-dir", default="data_2024_2025_giro/raw/giro", help="Directory with raw GIRO txt files.")
    parser.add_argument("--output-dir", default="cleaned_2024_2025", help="Directory for cleaned data and report.")
    parser.add_argument("--row-threshold", type=float, default=DEFAULT_ROW_MISSING_THRESHOLD)
    parser.add_argument("--column-threshold", type=float, default=DEFAULT_COLUMN_MISSING_THRESHOLD)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    processed_dir = input_dir / "processed"
    output_dir = Path(args.output_dir)
    cleaned_processed = output_dir / "processed"
    cleaned_processed.mkdir(parents=True, exist_ok=True)

    reports = []
    for csv_path in sorted(processed_dir.glob("*.csv")):
        reports.append(
            clean_csv(
                csv_path,
                cleaned_processed / csv_path.name,
                row_threshold=args.row_threshold,
                column_threshold=args.column_threshold,
            )
        )

    giro_report = analyze_giro_raw(Path(args.giro_raw_dir))
    report_path = output_dir / "cleaning_report.md"
    write_report(report_path, reports, giro_report, args.row_threshold, args.column_threshold)
    print(f"Wrote cleaned data to {cleaned_processed}")
    print(f"Wrote report to {report_path}")


def clean_csv(csv_path: Path, output_path: Path, row_threshold: float, column_threshold: float) -> dict[str, Any]:
    rows, columns = read_csv(csv_path)
    report: dict[str, Any] = {
        "file": csv_path.name,
        "input_rows": len(rows),
        "input_columns": len(columns),
        "output_rows": 0,
        "output_columns": 0,
        "dropped_rows": 0,
        "dropped_columns": [],
        "filled": {},
        "missing_before": {},
        "missing_after": {},
        "output_file": str(output_path),
    }
    if not rows or not columns:
        write_csv(output_path, [], [])
        return report

    report["missing_before"] = missing_percentages(rows, columns)
    columns_to_keep = choose_columns(rows, columns, column_threshold)
    report["dropped_columns"] = [column for column in columns if column not in columns_to_keep]

    projected_rows = [{column: row.get(column, "") for column in columns_to_keep} for row in rows]
    filtered_rows = [
        row for row in projected_rows if row_missing_fraction(row, columns_to_keep) <= row_threshold
    ]
    report["dropped_rows"] = len(projected_rows) - len(filtered_rows)

    fill_report = apply_known_fills(csv_path.name, filtered_rows, columns_to_keep)
    report["filled"] = fill_report
    report["missing_after"] = missing_percentages(filtered_rows, columns_to_keep)
    report["output_rows"] = len(filtered_rows)
    report["output_columns"] = len(columns_to_keep)
    write_csv(output_path, filtered_rows, columns_to_keep)
    write_json(output_path.with_suffix(".json"), filtered_rows, csv_path.stem)
    return report


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        columns = list(reader.fieldnames or [])
        return list(reader), columns


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]], name: str) -> None:
    payload = {"dataset": name, "records": rows}
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def is_missing(value: Any) -> bool:
    return value is None or str(value).strip().lower() in MISSING_VALUES


def row_missing_fraction(row: dict[str, Any], columns: list[str]) -> float:
    if not columns:
        return 1.0
    return sum(1 for column in columns if is_missing(row.get(column))) / len(columns)


def missing_percentages(rows: list[dict[str, Any]], columns: list[str]) -> dict[str, float]:
    if not rows:
        return {}
    result = {}
    for column in columns:
        missing = sum(1 for row in rows if is_missing(row.get(column)))
        result[column] = round(missing / len(rows) * 100, 2)
    return result


def choose_columns(rows: list[dict[str, Any]], columns: list[str], threshold: float) -> list[str]:
    keep = []
    for column in columns:
        missing = sum(1 for row in rows if is_missing(row.get(column)))
        fraction = missing / len(rows) if rows else 1.0
        if fraction <= threshold:
            keep.append(column)
    return keep


def apply_known_fills(file_name: str, rows: list[dict[str, Any]], columns: list[str]) -> dict[str, int]:
    filled: dict[str, int] = {}
    if file_name == "geophysical_indices.csv" and "status" in columns:
        count = 0
        for row in rows:
            if is_missing(row.get("status")):
                row["status"] = "unknown"
                count += 1
        filled["status_unknown"] = count

    if file_name == "omni_solar_wind.csv" and "flow_speed_km_s" in columns:
        filled["flow_speed_km_s_carried_forward"] = carry_forward_numeric_by_time(rows, "time_utc", "flow_speed_km_s")
    return filled


def carry_forward_numeric_by_time(rows: list[dict[str, Any]], time_column: str, value_column: str) -> int:
    rows.sort(key=lambda row: parse_time(row.get(time_column)) or datetime.max)
    last_value = None
    filled = 0
    for row in rows:
        value = parse_float(row.get(value_column))
        if value is not None:
            last_value = value
        elif last_value is not None and parse_time(row.get(time_column)) is not None:
            row[value_column] = last_value
            filled += 1
    return filled


def parse_time(value: Any) -> datetime | None:
    if is_missing(value):
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def parse_float(value: Any) -> float | None:
    if is_missing(value):
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def analyze_giro_raw(raw_dir: Path) -> dict[str, Any]:
    files = sorted(raw_dir.rglob("*.txt"))
    station_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"files": 0, "data_lines": 0, "years": set()})
    files_without_data = 0
    for path in files:
        station = "unknown"
        year = "unknown"
        for part in path.parts:
            if part.startswith("station="):
                station = part.split("=", 1)[1]
            elif part.startswith("year="):
                year = part.split("=", 1)[1]
        data_lines = count_giro_data_lines(path)
        if data_lines == 0:
            files_without_data += 1
        station_stats[station]["files"] += 1
        station_stats[station]["data_lines"] += data_lines
        station_stats[station]["years"].add(year)

    zero_data_stations = sorted(station for station, stats in station_stats.items() if stats["data_lines"] == 0)
    missing_2025 = sorted(station for station, stats in station_stats.items() if "2025" not in stats["years"])
    data_stations = sorted(
        ({"station": station, "data_lines": stats["data_lines"], "files": stats["files"]} for station, stats in station_stats.items() if stats["data_lines"] > 0),
        key=lambda item: item["data_lines"],
        reverse=True,
    )
    return {
        "raw_files": len(files),
        "files_without_data": files_without_data,
        "stations_total": len(station_stats),
        "stations_with_data": len(data_stations),
        "stations_without_data": len(zero_data_stations),
        "zero_data_stations": zero_data_stations,
        "missing_2025": missing_2025,
        "top_data_stations": data_stations[:15],
    }


def count_giro_data_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                count += 1
    return count


def write_report(
    path: Path,
    reports: list[dict[str, Any]],
    giro_report: dict[str, Any],
    row_threshold: float,
    column_threshold: float,
) -> None:
    lines = [
        "# Cleaning report",
        "",
        f"- Row auto-drop threshold: more than {row_threshold:.0%} missing values.",
        f"- Column auto-drop threshold: more than {column_threshold:.0%} missing values.",
        "- Original files are not modified.",
        "",
        "## Processed CSV",
    ]
    for report in reports:
        lines.extend(
            [
                "",
                f"### {report['file']}",
                f"- Input: {report['input_rows']} rows, {report['input_columns']} columns.",
                f"- Output: {report['output_rows']} rows, {report['output_columns']} columns.",
                f"- Dropped rows: {report['dropped_rows']}.",
                f"- Dropped columns: {', '.join(report['dropped_columns']) or 'none'}.",
                f"- Filled values: {format_dict(report['filled'])}.",
                f"- Output file: `{report['output_file']}`.",
            ]
        )
        before = top_missing(report["missing_before"])
        after = top_missing(report["missing_after"])
        lines.append(f"- Top missing before: {before}.")
        lines.append(f"- Top missing after: {after}.")

    lines.extend(
        [
            "",
            "## Raw GIRO coverage",
            f"- Raw files: {giro_report['raw_files']}.",
            f"- Files without measurement rows: {giro_report['files_without_data']}.",
            f"- Stations total: {giro_report['stations_total']}.",
            f"- Stations with data: {giro_report['stations_with_data']}.",
            f"- Stations without data: {giro_report['stations_without_data']}.",
            f"- Stations missing 2025 files: {', '.join(giro_report['missing_2025']) or 'none'}.",
            "",
            "### Zero-data stations",
            ", ".join(giro_report["zero_data_stations"]) or "none",
            "",
            "### Top GIRO stations by measurement rows",
        ]
    )
    for station in giro_report["top_data_stations"]:
        lines.append(f"- {station['station']}: {station['data_lines']} rows across {station['files']} files")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def top_missing(values: dict[str, float]) -> str:
    if not values:
        return "none"
    top = sorted(values.items(), key=lambda item: item[1], reverse=True)[:5]
    return ", ".join(f"{name} {percent:.2f}%" for name, percent in top if percent > 0) or "none"


def format_dict(values: dict[str, Any]) -> str:
    if not values:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in values.items())


if __name__ == "__main__":
    main()
