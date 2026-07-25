#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git tmux rsync htop

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt

python -m py_compile run_ml_baselines.py run_station_batch.py watch_run_progress.py
python -m pytest

echo "VM setup complete"
