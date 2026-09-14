#!/usr/bin/env bash
# Reproduces the headline results in this repo in well under 15 minutes
# (in practice: a few seconds, since the dataset is a 93-row sample -- see
# README.md "Data note" for why).
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/3: inspecting the data =="
python3 src/load_data.py data/twcs_sample.csv

echo
echo "== 2/3: running one example through the live agent =="
python3 src/agent.py "@AppleSupport my battery is draining so fast since the update, please help"

echo
echo "== 3/3: running the full evaluation harness against the golden set =="
python3 eval/run_eval.py
python3 eval/judge_agreement.py
