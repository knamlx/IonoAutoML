from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


MISSING_MARKERS = {"", "---", "__", "nan", "na", "null", "none"}
GIRO_MEASUREMENTS = [
    "foF2",
    "foF1",
    "foE",
    "foEs",
    "fbEs",
    "foEa",
    "foP",
    "fxI",
    "MUF_D",
    "M_D",
    "hF2",
    "hF",
    "hE",
    "hEs",
    "hEa",
    "hP",
    "hmF2",
    "hmF1",
    "hmE",
    "halfNm",
    "yF2",
    "yF1",
    "yE",
    "B0",
    "B1",
    "D1",
    "TEC",
    "FF",
    "FE",
    "QF",
    "QE",
    "fmin",
    "fminF",
    "fminE",
    "fminEs",
    "foF2p",
]
CORE_GIRO_COLUMNS = ["foF2", "foF1", "foE", "MUF_D", "hmF2", "TEC", "fmin", "foF2p"]
CORE_GIRO_POSITIONS = {name: GIRO_MEASUREMENTS.index(name) for name in CORE_GIRO_COLUMNS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize collected GIRO and index data to one time grid.")
    parser.add_argument("--giro-raw-dir", default="data_2024_2025_giro/raw/giro")
    parser.add_argument("--processed-dir", default="cleaned_2024_2025/processed")
    parser.add_argument("--output-dir", default="normalized_2024_2025")
    parser.add_argument("--freq", default="auto", help="Pandas frequency, for example 5min. Use auto to infer.")
    parser.add_argument("--min-cs", type=float, default=50.0, help="Soft GIRO confidence cutoff.")
    parser.add_argument(
        "--max-interpolate-gap",
        default="30min",
        help="Deprecated compatibility option. GIRO values are not interpolated.",
    )
    parser.add_argument("--top-stations", type=int, default=0, help="Normalize only top N stations by raw data rows. 0 means all.")
    parser.add_argument("--split-by-station", action="store_true", help="Also write one analytical CSV per station.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_station_stats = scan_giro_station_stats(Path(args.giro_raw_dir))
    include_stations = choose_top_stations(raw_station_stats, args.top_stations)
    if include_stations is not None:
        print(f"Using top {len(include_stations)} stations by raw coverage: {', '.join(include_stations)}")

    print("Reading GIRO raw files...")
    giro = read_giro_raw(Path(args.giro_raw_dir), include_stations=include_stations, raw_station_stats=raw_station_stats)
    if giro.empty:
        raise SystemExit("No GIRO measurement rows found.")

    coverage = build_station_coverage(giro)
    excluded = coverage[coverage["data_rows"] == 0]["station"].tolist()
    active_stations = sorted(giro["station"].dropna().unique().tolist())
    freq = infer_frequency(giro["time_utc"]) if args.freq == "auto" else args.freq
    print(f"Inferred/selected frequency: {freq}")
    print(f"Active GIRO stations: {len(active_stations)}")

    print("Normalizing GIRO to time grid...")
    normalized_giro = normalize_giro(giro, freq, args.min_cs, args.max_interpolate_gap)
    normalized_giro.to_csv(output_dir / "giro_time_grid.csv", index=False)

    print("Preparing index features...")
    processed_dir = Path(args.processed_dir)
    indices = prepare_index_features(processed_dir, freq)

    print("Merging features by time...")
    analytical = normalized_giro.merge(indices, on="time_utc", how="left")
    analytical.to_csv(output_dir / "analytical_time_grid.csv", index=False)
    station_files = write_station_datasets(analytical, output_dir) if args.split_by_station else []

    coverage.to_csv(output_dir / "station_coverage.csv", index=False)
    write_report(
        output_dir / "time_normalization_report.md",
        giro,
        normalized_giro,
        analytical,
        coverage,
        excluded,
        active_stations,
        freq,
        args.min_cs,
        args.max_interpolate_gap,
        station_files,
    )
    print(f"Wrote normalized outputs to {output_dir}")


def scan_giro_station_stats(raw_dir: Path) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "data_lines": 0})
    for path in sorted(raw_dir.rglob("*.txt")):
        station = get_path_value(path, "station")
        stats[station]["files"] += 1
        data_lines = 0
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and not stripped.upper().startswith("ERROR:"):
                    data_lines += 1
        stats[station]["data_lines"] += data_lines
    return dict(stats)


def choose_top_stations(raw_station_stats: dict[str, dict[str, int]], top_stations: int) -> set[str] | None:
    if top_stations <= 0:
        return None
    ranked = sorted(
        ((station, stats["data_lines"]) for station, stats in raw_station_stats.items() if stats["data_lines"] > 0),
        key=lambda item: item[1],
        reverse=True,
    )
    return {station for station, _ in ranked[:top_stations]}


def read_giro_raw(
    raw_dir: Path,
    include_stations: set[str] | None = None,
    raw_station_stats: dict[str, dict[str, int]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    raw_station_stats = raw_station_stats or {}
    for path in sorted(raw_dir.rglob("*.txt")):
        station = get_path_value(path, "station")
        if include_stations is not None and station not in include_stations:
            continue
        year = get_path_value(path, "year")
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or stripped.upper().startswith("ERROR:"):
                    continue
                parsed = parse_giro_line(stripped)
                if parsed:
                    parsed["station"] = station
                    parsed["year"] = year
                    rows.append(parsed)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["time_utc"])
    numeric_columns = ["CS", *CORE_GIRO_COLUMNS]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df.attrs["raw_station_stats"] = dict(raw_station_stats)
    return df.sort_values(["station", "time_utc"])


def parse_giro_line(line: str) -> dict[str, Any] | None:
    parts = re.split(r"\s+", line.strip())
    if len(parts) < 3:
        return None
    row: dict[str, Any] = {"time_utc": parts[0], "CS": to_number(parts[1])}
    for measurement, measurement_index in CORE_GIRO_POSITIONS.items():
        token_index = 2 + measurement_index * 2
        row[measurement] = to_number(parts[token_index]) if token_index < len(parts) else None
    return row


def to_number(value: Any) -> float | None:
    value = missing_to_none(value)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def missing_to_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in MISSING_MARKERS:
        return None
    return text


def get_path_value(path: Path, key: str) -> str:
    prefix = f"{key}="
    for part in path.parts:
        if part.startswith(prefix):
            return part.split("=", 1)[1]
    return ""


def infer_frequency(times: pd.Series) -> str:
    rounded = pd.to_datetime(times, utc=True).sort_values().drop_duplicates()
    deltas = rounded.diff().dropna().dt.total_seconds()
    deltas = deltas[(deltas > 0) & (deltas <= 3600)]
    if deltas.empty:
        return "5min"
    minute_deltas = (deltas / 60).round().astype(int)
    minute_deltas = minute_deltas[minute_deltas > 0]
    if minute_deltas.empty:
        return "5min"
    mode_minutes = Counter(minute_deltas.tolist()).most_common(1)[0][0]
    return f"{mode_minutes}min"


def normalize_giro(df: pd.DataFrame, freq: str, min_cs: float, max_interpolate_gap: str) -> pd.DataFrame:
    filtered = df[(df["CS"].isna()) | (df["CS"] == -1) | (df["CS"] >= min_cs)].copy()
    numeric_columns = [column for column in ["CS", *CORE_GIRO_COLUMNS] if column in filtered.columns]
    pieces = []
    for station, station_df in filtered.groupby("station", sort=True):
        station_df = station_df.sort_values("time_utc").drop_duplicates("time_utc", keep="last")
        station_df = station_df.set_index("time_utc")
        resampled = station_df[numeric_columns].resample(freq).mean()
        original_counts = station_df.resample(freq).size().reindex(resampled.index, fill_value=0)
        resampled["station"] = station
        resampled["giro_rows_in_interval"] = original_counts.astype(int)
        resampled["had_original_giro_row"] = original_counts.gt(0)
        pieces.append(resampled.reset_index())
    if not pieces:
        return pd.DataFrame()
    result = pd.concat(pieces, ignore_index=True)
    result["interval"] = make_interval(result["time_utc"], freq)
    ordered = ["time_utc", "interval", "station", "had_original_giro_row", "giro_rows_in_interval", *numeric_columns]
    return result[[column for column in ordered if column in result.columns]]


def make_interval(times: pd.Series, freq: str) -> pd.Series:
    delta = pd.to_timedelta(freq)
    starts = pd.to_datetime(times, utc=True)
    ends = starts + delta
    return starts.dt.strftime("%H:%M") + "-" + ends.dt.strftime("%H:%M")


def prepare_index_features(processed_dir: Path, freq: str) -> pd.DataFrame:
    frames = []
    gfz_path = processed_dir / "geophysical_indices.csv"
    if gfz_path.exists() and gfz_path.stat().st_size > 2:
        gfz = pd.read_csv(gfz_path)
        gfz["time_utc"] = pd.to_datetime(gfz["time_utc"], utc=True, errors="coerce")
        gfz["value"] = pd.to_numeric(gfz["value"], errors="coerce")
        gfz = gfz.dropna(subset=["time_utc", "index"])
        gfz = gfz.pivot_table(index="time_utc", columns="index", values="value", aggfunc="mean")
        frames.append(gfz.add_prefix("gfz_"))

    omni_path = processed_dir / "omni_solar_wind.csv"
    if omni_path.exists() and omni_path.stat().st_size > 2:
        omni = pd.read_csv(omni_path)
        omni["time_utc"] = pd.to_datetime(omni["time_utc"], utc=True, errors="coerce")
        for column in ["bz_gsm_nT", "dst_nT", "flow_speed_km_s", "proton_density_n_cc"]:
            if column in omni.columns:
                omni[column] = pd.to_numeric(omni[column], errors="coerce")
        keep = [column for column in ["bz_gsm_nT", "dst_nT", "flow_speed_km_s", "proton_density_n_cc"] if column in omni.columns]
        omni = omni.dropna(subset=["time_utc"]).set_index("time_utc")[keep].sort_index()
        frames.append(omni)

    if not frames:
        return pd.DataFrame({"time_utc": []})

    resampled = [frame.sort_index().resample(freq).ffill() for frame in frames]
    merged = pd.concat(resampled, axis=1).sort_index().ffill().reset_index()
    return merged.rename(columns={"index": "time_utc"})


def write_station_datasets(analytical: pd.DataFrame, output_dir: Path) -> list[Path]:
    station_dir = output_dir / "stations"
    station_dir.mkdir(parents=True, exist_ok=True)
    station_files = []
    for station, station_df in analytical.groupby("station", sort=True):
        path = station_dir / f"{station}_time_grid.csv"
        station_df = station_df.sort_values("time_utc").reset_index(drop=True)
        station_df.to_csv(path, index=False)
        station_files.append(path)
    return station_files


def build_station_coverage(giro: pd.DataFrame) -> pd.DataFrame:
    raw_stats = giro.attrs.get("raw_station_stats", {})
    if giro.empty:
        rows = [{"station": station, "data_rows": stats["data_lines"], "raw_files": stats["files"], "start_utc": pd.NaT, "end_utc": pd.NaT} for station, stats in raw_stats.items()]
        return pd.DataFrame(rows, columns=["station", "data_rows", "raw_files", "start_utc", "end_utc"])
    coverage = (
        giro.groupby("station")
        .agg(data_rows=("time_utc", "size"), start_utc=("time_utc", "min"), end_utc=("time_utc", "max"))
        .reset_index()
    )
    coverage["raw_files"] = coverage["station"].map(lambda station: raw_stats.get(station, {}).get("files", 0))
    existing = set(coverage["station"])
    zero_rows = [
        {"station": station, "data_rows": stats["data_lines"], "raw_files": stats["files"], "start_utc": pd.NaT, "end_utc": pd.NaT}
        for station, stats in raw_stats.items()
        if station not in existing
    ]
    if zero_rows:
        coverage = pd.concat([coverage, pd.DataFrame(zero_rows)], ignore_index=True)
    coverage = coverage.sort_values(["data_rows", "station"], ascending=[False, True])
    return coverage


def write_report(
    path: Path,
    giro: pd.DataFrame,
    normalized: pd.DataFrame,
    analytical: pd.DataFrame,
    coverage: pd.DataFrame,
    excluded: list[str],
    active_stations: list[str],
    freq: str,
    min_cs: float,
    max_interpolate_gap: str,
    station_files: list[Path],
) -> None:
    source_missing = missing_summary(giro, ["foF2", "MUF_D", "hmF2", "TEC", "fmin", "foF2p"])
    normalized_missing = missing_summary(normalized, ["foF2", "MUF_D", "hmF2", "TEC", "fmin", "foF2p"])
    lines = [
        "# Time normalization report",
        "",
        f"- Selected time step: `{freq}`.",
        f"- GIRO confidence filter: `CS >= {min_cs}`, `CS = -1`, or missing CS.",
        "- GIRO normalization: measurements are aggregated inside each time interval; missing GIRO intervals stay empty.",
        "- Index normalization: GFZ/OMNI values are carried forward over their valid time intervals.",
        f"- Active stations used for training grid: {len(active_stations)}.",
        f"- Stations excluded because they have no measurement rows in parsed GIRO: {len(excluded)}.",
        "",
        "## Output files",
        "- `giro_time_grid.csv`: normalized GIRO rows by station and time.",
        "- `analytical_time_grid.csv`: GIRO grid with geophysical and OMNI features joined by time.",
        f"- `stations/*.csv`: {len(station_files)} separate station datasets.",
        "- `station_coverage.csv`: station coverage summary.",
        "",
        "## Shape",
        f"- Parsed GIRO source rows: {len(giro)}.",
        f"- Normalized GIRO rows: {len(normalized)}.",
        f"- Analytical rows: {len(analytical)}.",
        "",
        "## Missing values before normalization",
        format_missing(source_missing),
        "",
        "## Missing values after normalization",
        format_missing(normalized_missing),
        "",
        "## Excluded stations",
        ", ".join(excluded) if excluded else "None in parsed GIRO table.",
        "",
        "## Top stations by rows",
    ]
    for _, row in coverage.head(20).iterrows():
        lines.append(f"- {row['station']}: {int(row['data_rows'])} rows, {row['start_utc']} - {row['end_utc']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def missing_summary(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    if df.empty:
        return {}
    result = {}
    for column in columns:
        if column in df.columns:
            result[column] = round(df[column].isna().mean() * 100, 2)
    return result


def format_missing(values: dict[str, float]) -> str:
    if not values:
        return "No values."
    return "\n".join(f"- {column}: {percent:.2f}%" for column, percent in values.items())


if __name__ == "__main__":
    main()
