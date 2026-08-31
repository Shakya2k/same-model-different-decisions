#!/usr/bin/env bash
# Public release reproduction smoke (no private data).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python -m pytest tests -q

cd experiments/threshold_capacity
python experiment_01.py
if [[ -f outputs/experiment_02_full_multiseed.csv ]]; then
  python compute_paired_deltas.py
else
  echo "NOTE: full Exp02 multiseed CSV not present; using committed publication-facing summaries."
  echo "      Regenerate with: python experiment_02_robustness.py  (heavier)"
  test -f outputs/experiment_02_paired_deltas.csv
  test -f outputs/experiment_02_policy_means.csv
fi

cd ../decision_misspecification
python experiment_03.py

cd "$ROOT"
echo "REPRO_SMOKE_OK"
