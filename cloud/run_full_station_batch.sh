#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

mkdir -p artifacts/baseline_v0.1/logs
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_path="artifacts/baseline_v0.1/logs/cloud_full_${timestamp}.log"

echo "Writing log to ${log_path}"
python run_station_batch.py \
  --config configs/experiments/baseline_v0.1_cloud_16cpu.json \
  --parallel-stations 5 \
  2>&1 | tee "${log_path}"
