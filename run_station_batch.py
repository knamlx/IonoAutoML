from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from run_ml_baselines import load_config, resolve_stations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full ML experiment station by station.")
    parser.add_argument("--config", default="configs/experiments/baseline_v0.1.json")
    parser.add_argument("--station", action="append", default=None, help="Station to run. Can be repeated.")
    parser.add_argument("--station-file", default=None, help="Text file with one station code per line.")
    parser.add_argument("--horizon", default=None, help="Optional single horizon override, for example 1h.")
    parser.add_argument("--start-after", default=None, help="Skip stations up to and including this station.")
    parser.add_argument("--max-stations", type=int, default=None, help="Stop after this many stations.")
    parser.add_argument("--force", action="store_true", help="Run even if station run_summary.json already exists.")
    parser.add_argument("--no-automl", action="store_true", help="Pass --no-automl to run_ml_baselines.py.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned station commands without running.")
    return parser.parse_args()


def read_station_file(path: str | None) -> list[str]:
    if not path:
        return []
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def station_done(output_dir: Path, station: str) -> bool:
    return (output_dir / "stations" / station / "run_summary.json").exists()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    output_dir = Path(config["output_dir"])
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    stations = args.station or read_station_file(args.station_file) or resolve_stations(config)
    if args.start_after and args.start_after in stations:
        stations = stations[stations.index(args.start_after) + 1 :]
    if args.max_stations is not None:
        stations = stations[: args.max_stations]

    print(f"config: {config_path}")
    print(f"output_dir: {output_dir}")
    print(f"stations planned: {len(stations)}")
    for station in stations:
        if station_done(output_dir, station) and not args.force:
            print(f"SKIP {station}: run_summary.json exists")
            continue

        command = [
            sys.executable,
            "run_ml_baselines.py",
            "--config",
            str(config_path),
            "--station",
            station,
        ]
        if args.horizon:
            command.extend(["--horizon", args.horizon])
        if args.no_automl:
            command.append("--no-automl")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = log_dir / f"{timestamp}_{station}.log"
        print("RUN", station, "->", log_path)
        print(" ".join(command))
        if args.dry_run:
            continue

        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"started_utc={timestamp}\n")
            log_file.write("command=" + " ".join(command) + "\n\n")
            log_file.flush()
            result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, text=True)
            finished = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            log_file.write(f"\nfinished_utc={finished}\n")
            log_file.write(f"returncode={result.returncode}\n")

        if result.returncode != 0:
            print(f"FAILED {station}: returncode={result.returncode}; see {log_path}")
            sys.exit(result.returncode)
        print(f"DONE {station}")

    print("station batch finished")


if __name__ == "__main__":
    main()
