#!/usr/bin/env python3
"""Collect and preprocess GIRO/NOAA data for HF radio-channel analytics."""

from __future__ import annotations

import argparse
import csv
import json
import re
import os
import subprocess
import sys
import time
import tomllib
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


GIRO_CHAR_IDS = {
    "foF2": 0,
    "foF1": 1,
    "foE": 2,
    "foEs": 3,
    "fbEs": 4,
    "foEa": 5,
    "foP": 6,
    "fxI": 7,
    "MUFD": 8,
    "MD": 9,
    "hF2": 10,
    "hF": 11,
    "hE": 12,
    "hEs": 13,
    "hEa": 14,
    "hP": 15,
    "TypeEs": 16,
    "hmF2": 17,
    "hmF1": 18,
    "hmE": 19,
    "zhalfNm": 20,
    "yF2": 21,
    "yF1": 22,
    "yE": 23,
    "scaleF2": 24,
    "B0": 25,
    "B1": 26,
    "D1": 27,
    "TEC": 28,
    "FF": 29,
    "FE": 30,
    "QF": 31,
    "QE": 32,
    "fmin": 33,
    "fminF": 34,
    "fminE": 35,
    "fminEs": 36,
    "foF2p": 37,
}

TIME_KEYS = (
    "time_tag",
    "time",
    "timestamp",
    "date",
    "datetime",
    "observed_flux_date",
    "forecast_date",
    "issue_datetime",
)

USER_AGENT = "giro-noaa-hf-collector/1.0 (+research data preprocessing)"
GIRO_FORM_URL = "https://giro.uml.edu/didbase/scaled.php?tdsourcetag=s_pctim_aiomsg"
GIRO_DATA_URL = "https://giro.uml.edu/didbase/scaled.php"


@dataclass(frozen=True)
class Paths:
    root: Path
    raw_noaa: Path
    raw_gfz: Path
    raw_omni: Path
    raw_giro: Path
    processed: Path
    metadata: Path
    logs: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.example.toml", help="Path to TOML config.")
    parser.add_argument("--start", help="UTC start, e.g. 2026-05-26T00:00:00Z.")
    parser.add_argument("--end", help="UTC end, e.g. 2026-05-27T00:00:00Z.")
    parser.add_argument("--output-dir", help="Override output directory.")
    parser.add_argument(
        "--sources",
        default="all",
        help="Comma-separated sources to collect: all, giro, gfz, omni, noaa.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        dt = parsedate_to_datetime(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def resolve_interval(args: argparse.Namespace, config: dict[str, Any]) -> tuple[datetime, datetime]:
    run_cfg = config.get("run", {})
    end = parse_dt(args.end) or parse_dt(run_cfg.get("end_utc")) or datetime.now(UTC)
    start = parse_dt(args.start) or parse_dt(run_cfg.get("start_utc"))
    if start is None:
        start = end - timedelta(hours=float(run_cfg.get("lookback_hours", 24)))
    if start >= end:
        raise ValueError("Start time must be earlier than end time.")
    return start, end


def init_paths(config: dict[str, Any], args: argparse.Namespace) -> Paths:
    out_dir = Path(args.output_dir or config.get("run", {}).get("output_dir", "data"))
    paths = Paths(
        root=out_dir,
        raw_noaa=out_dir / "raw" / "noaa",
        raw_gfz=out_dir / "raw" / "gfz",
        raw_omni=out_dir / "raw" / "omni",
        raw_giro=out_dir / "raw" / "giro",
        processed=out_dir / "processed",
        metadata=out_dir / "metadata",
        logs=out_dir / "logs",
    )
    for path in (paths.raw_noaa, paths.raw_gfz, paths.raw_omni, paths.raw_giro, paths.processed, paths.metadata, paths.logs):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def fetch_url(url: str, *, data: dict[str, Any] | None, timeout: int, attempts: int = 3) -> bytes:
    body = urlencode(data, doseq=True).encode("utf-8") if data else None
    if "giro.uml.edu" in url:
        return fetch_url_with_powershell(url, body=body, timeout=timeout)
    headers = {"User-Agent": USER_AGENT}
    if body:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = Request(url, data=body, headers=headers)
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except (TimeoutError, URLError, OSError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(2 * attempt)
    if "giro.uml.edu" in url:
        return fetch_url_with_powershell(url, body=body, timeout=timeout)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def fetch_url_with_powershell(url: str, *, body: bytes | None, timeout: int) -> bytes:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        out_path = tmp.name
    env = os.environ.copy()
    env["FETCH_URL"] = url
    env["FETCH_OUT"] = out_path
    env["FETCH_TIMEOUT"] = str(max(timeout, 120))
    env["FETCH_BODY"] = body.decode("utf-8") if body else ""
    script = r"""
$headers = @{ 'User-Agent' = 'Mozilla/5.0' }
$timeout = [int]$env:FETCH_TIMEOUT
if ([string]::IsNullOrEmpty($env:FETCH_BODY)) {
  Invoke-WebRequest -Uri $env:FETCH_URL -UseBasicParsing -Headers $headers -TimeoutSec $timeout -OutFile $env:FETCH_OUT
} else {
  Invoke-WebRequest -Uri $env:FETCH_URL -Method Post -Body $env:FETCH_BODY -ContentType 'application/x-www-form-urlencoded' -UseBasicParsing -Headers $headers -TimeoutSec $timeout -OutFile $env:FETCH_OUT
}
"""
    try:
        completed = subprocess.run(
            [r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe", "-NoProfile", "-Command", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=max(timeout + 30, 150),
            check=False,
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "PowerShell request failed").strip()
            raise RuntimeError(message)
        with open(out_path, "rb") as fh:
            return fh.read()
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def make_records_payload(dataset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "dataset": dataset,
        "record_count": len(rows),
        "records": rows,
    }


def write_records_json(path: Path, dataset: str, rows: list[dict[str, Any]]) -> None:
    write_json(path, make_records_payload(dataset, rows))


def read_json_bytes(raw: bytes) -> Any:
    text = raw.decode("utf-8-sig", errors="replace").strip("\x00\r\n\t ")
    return json.loads(text)


def normalize_time(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), UTC).isoformat().replace("+00:00", "Z")
    text = str(value).strip()
    if not text:
        return ""
    parsed = parse_dt(text)
    return parsed.isoformat().replace("+00:00", "Z") if parsed else text


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "---", "None", "null", "NaN"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def collect_noaa(config: dict[str, Any], paths: Paths, timeout: int) -> list[dict[str, Any]]:
    noaa_cfg = config.get("noaa", {})
    if not noaa_cfg.get("enabled", True):
        return []
    records: list[dict[str, Any]] = []
    continue_on_error = bool(noaa_cfg.get("continue_on_error", True))
    log_path = paths.logs / "collection_events.jsonl"
    for endpoint in noaa_cfg.get("endpoints", []):
        name = safe_name(endpoint["name"])
        raw_path = paths.raw_noaa / f"{name}.json"
        try:
            raw = fetch_url(endpoint["url"], data=None, timeout=timeout)
            raw_path.write_bytes(raw)
            payload = read_json_bytes(raw)
        except Exception as exc:
            append_jsonl(
                log_path,
                {
                    "source": "NOAA_SWPC",
                    "product": name,
                    "url": endpoint["url"],
                    "status": "failed",
                    "error": str(exc),
                    "time_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                },
            )
            if not continue_on_error:
                raise
            continue
        rows = payload if isinstance(payload, list) else payload.get("data", payload)
        if not isinstance(rows, list):
            rows = [rows]
        table_header = rows[0] if rows and isinstance(rows[0], list) and all(isinstance(col, str) for col in rows[0]) else None
        iterable = rows[1:] if table_header else rows
        for row in iterable:
            if isinstance(row, list):
                if table_header and len(row) == len(table_header):
                    row = dict(zip(table_header, row, strict=True))
                else:
                    row = {f"col_{idx}": value for idx, value in enumerate(row)}
            if not isinstance(row, dict):
                continue
            normalized = {"source": "NOAA_SWPC", "product": name}
            normalized.update(row)
            normalized["time_utc"] = first_time(row)
            records.append(normalized)
    return records


def collect_gfz_indices(
    config: dict[str, Any],
    paths: Paths,
    start: datetime,
    end: datetime,
    timeout: int,
) -> list[dict[str, Any]]:
    gfz_cfg = config.get("gfz", {})
    if not gfz_cfg.get("enabled", True):
        return []
    base_url = gfz_cfg.get("api_url", "https://kp.gfz.de/app/json/")
    indices = gfz_cfg.get("indices", ["Kp", "ap", "Ap", "Fobs", "Fadj"])
    status_filter = str(gfz_cfg.get("status", "")).strip()
    records: list[dict[str, Any]] = []
    for index in indices:
        query = {
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
            "index": index,
        }
        if status_filter:
            query["status"] = status_filter
        url = f"{base_url}?{urlencode(query)}"
        raw = fetch_url(url, data=None, timeout=timeout)
        raw_path = paths.raw_gfz / f"{safe_name(str(index))}_{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}.json"
        raw_path.write_bytes(raw)
        payload = read_json_bytes(raw)
        records.extend(parse_gfz_index_payload(payload, str(index), raw_path.name))
    return records


def parse_gfz_index_payload(payload: dict[str, Any], index: str, raw_file: str) -> list[dict[str, Any]]:
    times = payload.get("datetime", [])
    values = payload.get(index, [])
    statuses = payload.get("status", [])
    rows: list[dict[str, Any]] = []
    for idx, time_value in enumerate(times):
        rows.append(
            {
                "source": "GFZ",
                "product": "kp_gfz_json_api",
                "index": index,
                "time_utc": normalize_time(time_value),
                "value": values[idx] if idx < len(values) else "",
                "status": statuses[idx] if idx < len(statuses) else "",
                "raw_file": raw_file,
            }
        )
    return rows


OMNI_HOURLY_VARIABLES = {
    "by_gsm_nT": "15",
    "bz_gsm_nT": "16",
    "proton_density_n_cc": "23",
    "flow_speed_km_s": "24",
    "dst_nT": "40",
    "ap_nT": "49",
    "f10_7_sfu": "50",
}


def collect_omni(config: dict[str, Any], paths: Paths, start: datetime, end: datetime, timeout: int) -> list[dict[str, Any]]:
    omni_cfg = config.get("omni", {})
    if not omni_cfg.get("enabled", True):
        return []
    resolution = str(omni_cfg.get("resolution", "hour"))
    if resolution != "hour":
        raise ValueError("Only hourly OMNI is supported by this parser mode for now.")
    variables = omni_cfg.get("variables", ["bz_gsm_nT", "dst_nT", "flow_speed_km_s", "proton_density_n_cc"])
    var_ids = [OMNI_HOURLY_VARIABLES[name] for name in variables]
    body = {
        "activity": "retrieve",
        "res": "hour",
        "spacecraft": "omni2",
        "start_date": start.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"),
        "vars": var_ids,
        "scale": "Linear",
        "ymin": "",
        "ymax": "",
        "view": "0",
        "charsize": "",
        "xstyle": "0",
        "ystyle": "0",
        "symbol": "0",
        "symsize": "",
        "linestyle": "solid",
        "table": "0",
        "imagex": "640",
        "imagey": "480",
        "color": "",
        "back": "",
    }
    url = omni_cfg.get("api_url", "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi")
    raw = fetch_url(url, data=body, timeout=timeout)
    raw_path = paths.raw_omni / f"omni_hour_{start:%Y%m%d}_{end:%Y%m%d}.html"
    raw_path.write_bytes(raw)
    text = raw.decode("utf-8", errors="replace")
    return parse_omni_hourly_listing(text, variables, raw_path.name)


def parse_omni_hourly_listing(text: str, variables: list[str], raw_file: str) -> list[dict[str, Any]]:
    pre_match = re.search(r"<pre>(.*?)</pre>", text, flags=re.I | re.S)
    listing = pre_match.group(1) if pre_match else text
    rows: list[dict[str, Any]] = []
    for raw_line in listing.splitlines():
        line = raw_line.strip()
        if not re.match(r"^\d{4}\s+\d{1,3}\s+\d{1,2}\s+", line):
            continue
        parts = line.split()
        year, doy, hour = int(parts[0]), int(parts[1]), int(parts[2])
        dt = datetime(year, 1, 1, tzinfo=UTC) + timedelta(days=doy - 1, hours=hour)
        row = {
            "source": "NASA_OMNIWeb",
            "product": "omni2_hourly",
            "time_utc": dt.isoformat().replace("+00:00", "Z"),
            "raw_file": raw_file,
        }
        for name, value in zip(variables, parts[3:], strict=False):
            row[name] = "" if is_omni_missing(value) else maybe_float(value)
        rows.append(row)
    return rows


def is_omni_missing(value: str) -> bool:
    try:
        number = float(value)
    except ValueError:
        return True
    return abs(number) >= 9999 or value.strip("*") in {"999", "9999", "99999", "999999"}


def first_time(row: dict[str, Any]) -> str:
    for key in TIME_KEYS:
        if key in row:
            return normalize_time(row[key])
    return ""


def giro_characteristic_ids(names: list[str] | str) -> list[int]:
    if isinstance(names, str):
        if names.lower() == "all":
            return list(range(38))
        names = [names]
    ids: list[int] = []
    for name in names:
        if str(name).isdigit():
            ids.append(int(name))
        elif name in GIRO_CHAR_IDS:
            ids.append(GIRO_CHAR_IDS[name])
        else:
            raise ValueError(f"Unknown GIRO characteristic: {name}")
    return ids


def parse_giro_station_options(html: str) -> list[dict[str, str]]:
    location_match = re.search(r'<select[^>]+name="location"[^>]*>(.*?)(?:</select>|</td>)', html, flags=re.I | re.S)
    location_html = location_match.group(1) if location_match else html
    options = re.findall(
        r'<option\s+value="([A-Z0-9_]{4,5})"[^>]*>(.*?)(?=<option\s+value=|</select>|$)',
        location_html,
        flags=re.I | re.S,
    )
    seen: set[str] = set()
    stations: list[dict[str, str]] = []
    for code, label in options:
        if code in seen:
            continue
        clean_label = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", label)).strip()
        stations.append({"station": code, "name": clean_label})
        seen.add(code)
    return stations


def parse_giro_station_codes(html: str) -> list[str]:
    return [station["station"] for station in parse_giro_station_options(html)]


def resolve_giro_stations(stations: list[str] | str, timeout: int, paths: Paths | None = None) -> list[str]:
    if isinstance(stations, str):
        if stations.lower() != "all":
            return [stations]
        raw = fetch_url(GIRO_FORM_URL, data=None, timeout=timeout)
        html = raw.decode("utf-8", errors="replace")
        station_options = parse_giro_station_options(html)
        codes = [station["station"] for station in station_options]
        if not codes:
            raise ValueError("Could not discover GIRO station list from FastChar form.")
        if paths:
            write_json(
                paths.metadata / "stations.json",
                {
                    "schema_version": "1.0",
                    "source": "GIRO_DIDBase",
                    "fetched_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "station_count": len(station_options),
                    "stations": station_options,
                },
            )
        return codes
    return stations


def iter_time_windows(start: datetime, end: datetime, chunk_days: float) -> list[tuple[datetime, datetime]]:
    """Split DIDBase requests so no request crosses a calendar-year boundary."""
    max_days = min(chunk_days, 366) if chunk_days > 0 else 366
    windows: list[tuple[datetime, datetime]] = []
    current = start
    while current < end:
        next_year = datetime(current.year + 1, 1, 1, tzinfo=UTC)
        window_end = min(current + timedelta(days=max_days), next_year, end)
        windows.append((current, window_end))
        current = window_end
    return windows


def collect_giro(
    config: dict[str, Any],
    paths: Paths,
    start: datetime,
    end: datetime,
    timeout: int,
) -> list[dict[str, Any]]:
    if not config.get("giro", {}).get("enabled", True):
        return []
    giro_cfg = config.get("giro", {})
    char_ids = giro_characteristic_ids(giro_cfg.get("characteristics", ["foF2", "MUFD", "hmF2"]))
    stations = resolve_giro_stations(giro_cfg.get("stations", []), timeout, paths)
    windows = iter_time_windows(start, end, float(giro_cfg.get("chunk_days", 31)))
    continue_on_error = bool(giro_cfg.get("continue_on_error", True))
    log_path = paths.logs / "collection_events.jsonl"
    records: list[dict[str, Any]] = []
    for station in stations:
        for window_start, window_end in windows:
            params = {
                "query_submit": "Search",
                "date_start": window_start.strftime("%Y-%m-%d %H:%M"),
                "date_end": window_end.strftime("%Y-%m-%d %H:%M"),
                "location": station,
                "DMUF": str(giro_cfg.get("muf_distance_km", 3000)),
                "chosenchars[]": char_ids,
            }
            raw_path = (
                paths.raw_giro
                / f"station={safe_name(station)}"
                / f"year={window_start:%Y}"
                / f"{safe_name(station)}_{window_start:%Y%m%dT%H%M}_{window_end:%Y%m%dT%H%M}.txt"
            )
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            event = {
                "source": "GIRO_DIDBase",
                "station": station,
                "window_start_utc": window_start.isoformat().replace("+00:00", "Z"),
                "window_end_utc": window_end.isoformat().replace("+00:00", "Z"),
                "raw_file": str(raw_path),
                "status": "started",
                "time_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            }
            append_jsonl(log_path, event)
            try:
                raw = fetch_url(GIRO_DATA_URL, data=params, timeout=timeout)
                raw_path.write_bytes(raw)
                text = raw.decode("latin-1", errors="replace")
                parsed_rows = parse_giro_table(text, station, str(raw_path.relative_to(paths.root)))
                records.extend(parsed_rows)
                event.update({"status": "completed", "bytes": len(raw), "records": len(parsed_rows)})
                append_jsonl(log_path, event)
            except Exception as exc:
                event.update({"status": "failed", "error": str(exc)})
                append_jsonl(log_path, event)
                if not continue_on_error:
                    raise
                records.append(
                    {
                        "source": "GIRO_DIDBase",
                        "station": station,
                        "raw_file": str(raw_path.relative_to(paths.root)),
                        "error": str(exc),
                        "window_start_utc": event["window_start_utc"],
                        "window_end_utc": event["window_end_utc"],
                    }
                )
    return records


def parse_giro_table(text: str, station: str, raw_file: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    headers: list[str] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            possible_header = line.lstrip("#").strip()
            if "Time" in possible_header and ("foF" in possible_header or "hmF" in possible_header or "MUFD" in possible_header):
                headers = normalize_giro_headers(split_table_line(possible_header))
            continue
        if line.upper().startswith("ERROR:"):
            rows.append({"source": "GIRO_DIDBase", "station": station, "raw_file": raw_file, "error": line})
            continue
        parts = split_table_line(line)
        if headers and len(parts) == len(headers):
            row = dict(zip(headers, parts, strict=True))
        elif len(parts) >= 2:
            row = {"time_utc": normalize_time(" ".join(parts[:2]))}
            for idx, value in enumerate(parts[2:], start=1):
                row[f"value_{idx}"] = value
        else:
            continue
        row.setdefault("time_utc", infer_giro_time(row))
        row.update({"source": "GIRO_DIDBase", "station": station, "raw_file": raw_file})
        rows.append(row)
    return rows


def split_table_line(line: str) -> list[str]:
    if "\t" in line:
        return [part.strip() for part in line.split("\t") if part.strip()]
    if "," in line:
        return [part.strip() for part in next(csv.reader([line]))]
    return re.split(r"\s{2,}|\s+", line.strip())


def normalize_giro_headers(headers: list[str]) -> list[str]:
    normalized: list[str] = []
    previous_measurement = ""
    for header in headers:
        if header == "QD" and previous_measurement:
            normalized.append(f"{previous_measurement}_QD")
            continue
        normalized.append(header)
        if header not in {"Time", "CS", "QD"}:
            previous_measurement = header
    return normalized


def infer_giro_time(row: dict[str, Any]) -> str:
    for key in ("Time", "Datetime", "UT", "Timestamp", "time"):
        if key in row:
            return normalize_time(row[key])
    date_keys = [key for key in row if key.lower() in {"date", "yyyy-mm-dd"}]
    time_keys = [key for key in row if key.lower() in {"time", "hh:mm:ss", "hh:mm"}]
    if date_keys and time_keys:
        return normalize_time(f"{row[date_keys[0]]} {row[time_keys[0]]}")
    return ""


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    seen: set[str] = set()
    preferred = ["source", "product", "station", "time_utc", "error"]
    for key in preferred:
        for row in rows:
            if key in row and key not in seen:
                columns.append(key)
                seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                columns.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def filter_noaa_interval(rows: list[dict[str, Any]], start: datetime, end: datetime) -> list[dict[str, Any]]:
    filtered = []
    for row in rows:
        dt = parse_dt(row.get("time_utc"))
        if dt is None or start <= dt < end:
            filtered.append(row)
    return filtered


def filter_time_interval(rows: list[dict[str, Any]], start: datetime, end: datetime) -> list[dict[str, Any]]:
    filtered = []
    for row in rows:
        dt = parse_dt(row.get("time_utc"))
        if dt is not None and start <= dt < end:
            filtered.append(row)
    return filtered


def clean_giro(rows: list[dict[str, Any]], limits: dict[str, Any]) -> list[dict[str, Any]]:
    clean_rows = []
    for row in rows:
        cleaned = dict(row)
        for key in list(cleaned):
            norm = key.strip()
            if norm in {"foF2", "MUFD", "hmF2", "TEC", "fmin"}:
                value = maybe_float(cleaned[key])
                cleaned[key] = "" if value is None else value
        bounded(cleaned, "foF2", limits.get("fof2_min_mhz", 0.5), limits.get("fof2_max_mhz", 25.0))
        bounded(cleaned, "hmF2", limits.get("hmf2_min_km", 100.0), limits.get("hmf2_max_km", 700.0))
        bounded(cleaned, "TEC", limits.get("tec_min_tecu", 0.0), limits.get("tec_max_tecu", 300.0))
        clean_rows.append(cleaned)
    return clean_rows


def bounded(row: dict[str, Any], key: str, low: float, high: float) -> None:
    value = maybe_float(row.get(key))
    if value is not None and not (float(low) <= value <= float(high)):
        row[key] = ""


def build_analytical_dataset(
    giro_rows: list[dict[str, Any]],
    noaa_rows: list[dict[str, Any]],
    gfz_rows: list[dict[str, Any]],
    tolerance_minutes: float,
) -> list[dict[str, Any]]:
    noaa_points = [(parse_dt(row.get("time_utc")), row) for row in noaa_rows if parse_dt(row.get("time_utc"))]
    gfz_points = [(parse_dt(row.get("time_utc")), row) for row in gfz_rows if parse_dt(row.get("time_utc"))]
    dataset: list[dict[str, Any]] = []
    valid_giro_rows = [row for row in giro_rows if not row.get("error")]
    base_rows = valid_giro_rows or [
        {"source": "NOAA_SWPC", "station": "", "time_utc": row.get("time_utc", "")}
        for row in noaa_rows
    ]
    for row in base_rows:
        record = {
            "time_utc": row.get("time_utc", ""),
            "station": row.get("station", ""),
            "foF2_MHz": row.get("foF2", ""),
            "MUFD_3000_MHz": row.get("MUF(D)", row.get("MUFD", "")),
            "hmF2_km": row.get("hmF2", ""),
            "TEC_TECU": row.get("TEC", ""),
            "fmin_MHz": row.get("fmin", ""),
        }
        dt = parse_dt(record["time_utc"])
        if dt:
            nearest = nearest_noaa(dt, noaa_points, tolerance_minutes)
            record.update(extract_noaa_features(nearest))
            record.update(extract_gfz_features(nearest_geophysical(dt, gfz_points, tolerance_minutes)))
        record.update(derive_hf_features(record))
        dataset.append(record)
    return dataset


def nearest_noaa(
    dt: datetime,
    points: list[tuple[datetime | None, dict[str, Any]]],
    tolerance_minutes: float,
) -> list[dict[str, Any]]:
    tolerance = timedelta(minutes=tolerance_minutes)
    return [row for point_dt, row in points if point_dt and abs(point_dt - dt) <= tolerance]


def nearest_geophysical(
    dt: datetime,
    points: list[tuple[datetime | None, dict[str, Any]]],
    tolerance_minutes: float,
) -> list[dict[str, Any]]:
    tolerance = timedelta(minutes=tolerance_minutes)
    best_by_index: dict[str, tuple[timedelta, dict[str, Any]]] = {}
    for point_dt, row in points:
        if not point_dt:
            continue
        delta = abs(point_dt - dt)
        if delta > tolerance:
            continue
        index = str(row.get("index", ""))
        current = best_by_index.get(index)
        if current is None or delta < current[0]:
            best_by_index[index] = (delta, row)
    return [row for _, row in best_by_index.values()]


def extract_gfz_features(rows: list[dict[str, Any]]) -> dict[str, Any]:
    features: dict[str, Any] = {}
    names = {
        "Kp": "Kp",
        "ap": "ap",
        "Ap": "Ap",
        "SN": "sunspot_number",
        "Fobs": "F10_7_observed",
        "Fadj": "F10_7_adjusted",
        "Hp30": "Hp30",
        "Hp60": "Hp60",
        "ap30": "ap30",
        "ap60": "ap60",
    }
    for row in rows:
        index = str(row.get("index", ""))
        feature_name = names.get(index, index)
        if feature_name:
            features[feature_name] = row.get("value", "")
            features[f"{feature_name}_time_utc"] = row.get("time_utc", "")
            features[f"{feature_name}_status"] = row.get("status", "")
    return features


def extract_noaa_features(rows: list[dict[str, Any]]) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for row in rows:
        product = row.get("product", "")
        if product == "planetary_k_index_1m":
            set_latest(features, "Kp", row.get("estimated_kp") or row.get("kp_index"), row.get("time_utc"))
        elif product == "f107_cm_flux":
            set_latest(features, "F10_7", row.get("flux") or row.get("observed_flux"), row.get("time_utc"))
        elif "xrays" in product:
            set_latest(features, "xray_flux", row.get("flux"), row.get("time_utc"))
        elif product == "solar_wind_mag_2_hour":
            set_latest(features, "bt_nT", row.get("bt"), row.get("time_utc"))
            set_latest(features, "bz_gsm_nT", row.get("bz_gsm"), row.get("time_utc"))
        elif product == "solar_wind_plasma_2_hour":
            set_latest(features, "sw_speed_km_s", row.get("speed"), row.get("time_utc"))
            set_latest(features, "sw_density_p_cm3", row.get("density"), row.get("time_utc"))
    return {key: value for key, value in features.items() if not key.endswith("_time")}


def set_latest(features: dict[str, Any], key: str, value: Any, time_utc: Any) -> None:
    if value in (None, ""):
        return
    current_time = parse_dt(features.get(f"{key}_time"))
    new_time = parse_dt(str(time_utc))
    if current_time is None or (new_time and new_time >= current_time):
        features[key] = value
        features[f"{key}_time"] = time_utc


def derive_hf_features(row: dict[str, Any]) -> dict[str, Any]:
    fof2 = maybe_float(row.get("foF2_MHz"))
    mufd = maybe_float(row.get("MUFD_3000_MHz"))
    kp = maybe_float(row.get("Kp"))
    xray = maybe_float(row.get("xray_flux"))
    derived: dict[str, Any] = {}
    if fof2 is not None:
        derived["single_hop_vertical_critical_MHz"] = round(fof2, 3)
        derived["muf_3000_proxy_MHz"] = round(mufd if mufd is not None else fof2 * 3.0, 3)
    score = 0
    if kp is not None:
        score += 2 if kp >= 5 else 1 if kp >= 4 else 0
    if xray is not None:
        score += 2 if xray >= 1e-4 else 1 if xray >= 1e-5 else 0
    if fof2 is not None and fof2 < 3:
        score += 1
    derived["hf_disturbance_score"] = score
    derived["hf_disturbance_level"] = "high" if score >= 3 else "moderate" if score >= 1 else "quiet"
    return derived


def selected_sources(value: str) -> set[str]:
    sources = {item.strip().lower() for item in value.split(",") if item.strip()}
    return {"giro", "gfz", "omni", "noaa"} if not sources or "all" in sources else sources


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    start, end = resolve_interval(args, config)
    paths = init_paths(config, args)
    timeout = int(config.get("run", {}).get("timeout_seconds", 45))
    sources = selected_sources(args.sources)

    noaa_rows = filter_noaa_interval(collect_noaa(config, paths, timeout), start, end) if "noaa" in sources else []
    gfz_rows = filter_time_interval(collect_gfz_indices(config, paths, start, end, timeout), start, end) if "gfz" in sources else []
    omni_rows = filter_time_interval(collect_omni(config, paths, start, end, timeout), start, end) if "omni" in sources else []
    giro_rows = clean_giro(collect_giro(config, paths, start, end, timeout), config.get("preprocess", {})) if "giro" in sources else []
    analytical = build_analytical_dataset(
        giro_rows,
        noaa_rows,
        gfz_rows,
        float(config.get("preprocess", {}).get("join_tolerance_minutes", 90)),
    )

    noaa_csv = paths.processed / "noaa_observations.csv"
    gfz_csv = paths.processed / "geophysical_indices.csv"
    omni_csv = paths.processed / "omni_solar_wind.csv"
    giro_csv = paths.processed / "giro_scaled.csv"
    analytical_csv = paths.processed / "analytical_hf_dataset.csv"
    noaa_json = paths.processed / "noaa_observations.json"
    gfz_json = paths.processed / "geophysical_indices.json"
    omni_json = paths.processed / "omni_solar_wind.json"
    giro_json = paths.processed / "giro_scaled.json"
    analytical_json = paths.processed / "analytical_hf_dataset.json"
    write_csv(noaa_csv, noaa_rows)
    write_csv(gfz_csv, gfz_rows)
    write_csv(omni_csv, omni_rows)
    write_csv(giro_csv, giro_rows)
    write_csv(analytical_csv, analytical)
    write_records_json(noaa_json, "noaa_observations", noaa_rows)
    write_records_json(gfz_json, "geophysical_indices", gfz_rows)
    write_records_json(omni_json, "omni_solar_wind", omni_rows)
    write_records_json(giro_json, "giro_scaled", giro_rows)
    write_records_json(analytical_json, "analytical_hf_dataset", analytical)

    manifest = {
        "started_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "interval": {"start_utc": start.isoformat().replace("+00:00", "Z"), "end_utc": end.isoformat().replace("+00:00", "Z")},
        "sources": sorted(sources),
        "records": {"noaa": len(noaa_rows), "gfz": len(gfz_rows), "omni": len(omni_rows), "giro": len(giro_rows), "analytical": len(analytical)},
        "outputs": {
            "csv": [str(noaa_csv), str(gfz_csv), str(omni_csv), str(giro_csv), str(analytical_csv)],
            "json": [str(noaa_json), str(gfz_json), str(omni_json), str(giro_json), str(analytical_json)],
            "metadata": [str(paths.metadata / "stations.json")],
            "logs": [str(paths.logs / "collection_events.jsonl")],
        },
        "source_references": {
            "giro_rules": "https://giro.uml.edu/didbase/RulesOfTheRoad.html",
            "noaa_swpc_json_index": "https://services.swpc.noaa.gov/json/",
            "gfz_kp_api": "https://kp.gfz.de/app/json/",
            "nasa_omniweb": "https://omniweb.gsfc.nasa.gov/form/dx1.html",
        },
    }
    write_json(paths.root / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
