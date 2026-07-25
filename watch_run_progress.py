from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a station-by-station ML run.")
    parser.add_argument("--run-dir", default="artifacts/baseline_v0.1")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def latest_files(run_dir: Path) -> pd.DataFrame:
    files = sorted(run_dir.rglob("*"), key=lambda path: path.stat().st_mtime if path.is_file() else 0, reverse=True)
    rows = []
    for path in files:
        if not path.is_file():
            continue
        rows.append(
            {
                "file": str(path.relative_to(run_dir)),
                "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
                "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
            }
        )
        if len(rows) >= 12:
            break
    return pd.DataFrame(rows)


def optuna_summary(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    con = sqlite3.connect(db_path)
    query = """
    select
      s.study_name,
      count(t.trial_id) as trials,
      max(t.number) as max_trial,
      max(t.datetime_complete) as last_complete
    from studies s
    left join trials t on t.study_id = s.study_id
    group by s.study_id, s.study_name
    order by s.study_id desc
    limit 12
    """
    return pd.read_sql_query(query, con)


def station_summaries(run_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((run_dir / "stations").glob("*/run_summary.json")) if (run_dir / "stations").exists() else []:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(data)
    return pd.DataFrame(rows)


def partial_progress(run_dir: Path) -> pd.DataFrame:
    rows = []
    stations_dir = run_dir / "stations"
    if not stations_dir.exists():
        return pd.DataFrame()
    for station_dir in sorted(path for path in stations_dir.iterdir() if path.is_dir()):
        progress_path = station_dir / "partial_progress.json"
        if progress_path.exists():
            rows.append(json.loads(progress_path.read_text(encoding="utf-8")))
            continue
        partial_metrics = station_dir / "partial_metrics.csv"
        if partial_metrics.exists():
            rows.append(
                {
                    "station": station_dir.name,
                    "horizon": None,
                    "split_id": None,
                    "metrics_rows": sum(1 for _ in partial_metrics.open(encoding="utf-8")) - 1,
                    "updated_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(partial_metrics.stat().st_mtime)),
                }
            )
    return pd.DataFrame(rows)


def print_snapshot(run_dir: Path) -> None:
    print("=" * 90)
    print("RUN_DIR:", run_dir.resolve())
    print("\nLatest files")
    files = latest_files(run_dir)
    print(files.to_string(index=False) if not files.empty else "no files yet")

    print("\nOptuna studies")
    optuna = optuna_summary(run_dir / "optuna.db")
    print(optuna.to_string(index=False) if not optuna.empty else "no optuna.db yet")

    print("\nPartial station metrics")
    partial = partial_progress(run_dir)
    print(partial.to_string(index=False) if not partial.empty else "no partial metrics yet")

    print("\nCompleted station summaries")
    summaries = station_summaries(run_dir)
    print(summaries.to_string(index=False) if not summaries.empty else "no completed stations yet")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    while True:
        print_snapshot(run_dir)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
