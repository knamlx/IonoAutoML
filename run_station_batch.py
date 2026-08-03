from __future__ import annotations

import argparse
import subprocess
import sys
import time
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
    parser.add_argument("--parallel-stations", type=int, default=1, help="Number of station processes to run concurrently.")
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
    station_dir = output_dir / "stations" / station
    return (station_dir / "run_summary.json").exists() or any(station_dir.glob("*_run_summary.json"))


def build_command(args: argparse.Namespace, config_path: Path, station: str) -> list[str]:
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
    return command


def log_path_for(log_dir: Path, station: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return log_dir / f"{timestamp}_{station}.log"


def run_command(command: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
        started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_file.write(f"started_utc={started}\n")
        log_file.write("command=" + " ".join(command) + "\n\n")
        log_file.flush()
        result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        finished = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_file.write(f"\nfinished_utc={finished}\n")
        log_file.write(f"returncode={result.returncode}\n")
    return int(result.returncode)


def run_parallel(commands: list[tuple[str, list[str], Path]], max_parallel: int) -> None:
    active: list[tuple[str, subprocess.Popen, object, Path]] = []
    pending = list(commands)
    while pending or active:
        while pending and len(active) < max_parallel:
            station, command, log_path = pending.pop(0)
            log_file = log_path.open("w", encoding="utf-8")
            started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            log_file.write(f"started_utc={started}\n")
            log_file.write("command=" + " ".join(command) + "\n\n")
            log_file.flush()
            process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, text=True)
            active.append((station, process, log_file, log_path))
            print(f"START {station}: pid={process.pid} log={log_path}", flush=True)

        still_active = []
        for station, process, log_file, log_path in active:
            returncode = process.poll()
            if returncode is None:
                still_active.append((station, process, log_file, log_path))
                continue
            finished = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            log_file.write(f"\nfinished_utc={finished}\n")
            log_file.write(f"returncode={returncode}\n")
            log_file.close()
            if returncode != 0:
                print(f"FAILED {station}: returncode={returncode}; see {log_path}", flush=True)
                for other_station, other_process, other_log, other_log_path in still_active:
                    print(f"TERMINATE {other_station}: see {other_log_path}", flush=True)
                    other_process.terminate()
                    other_log.close()
                sys.exit(int(returncode))
            print(f"DONE {station}", flush=True)
        active = still_active
        if pending or active:
            time.sleep(5)


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
    print(f"parallel_stations: {args.parallel_stations}")
    commands: list[tuple[str, list[str], Path]] = []
    for station in stations:
        if station_done(output_dir, station) and not args.force:
            print(f"SKIP {station}: run_summary.json exists")
            continue

        command = build_command(args, config_path, station)
        log_path = log_path_for(log_dir, station)
        print("RUN", station, "->", log_path)
        print(" ".join(command))
        if args.dry_run:
            continue
        commands.append((station, command, log_path))

    if args.dry_run:
        print("station batch dry-run finished")
        return

    if args.parallel_stations <= 1:
        for station, command, log_path in commands:
            returncode = run_command(command, log_path)
            if returncode != 0:
                print(f"FAILED {station}: returncode={returncode}; see {log_path}")
                sys.exit(returncode)
            print(f"DONE {station}")
    else:
        run_parallel(commands, args.parallel_stations)

    print("station batch finished")


if __name__ == "__main__":
    main()
