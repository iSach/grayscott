#!/usr/bin/env bash
# Reproduce: data, one training run per variant, evaluation, figures.
# Default: the three dataset-agnostic variants, in parallel, one thread each
# (about 35 minutes on 2 CPU cores). The Gray-Scott-specific reference runs:
#   VARIANTS="fixed-abs fixed fixed-pd fixed-linear samplewise" ./run_all.sh
# Add `--device cuda` to the train line to use a GPU (not tested here).
set -euo pipefail
cd "$(dirname "$0")"
STEPS="${STEPS:-6000}"

python -m grayscott.data
mkdir -p logs
for v in ${VARIANTS:-global window window-aware}; do
  python -m grayscott.train --variant "$v" --steps "$STEPS" --batch 8 --threads 1 > "logs/$v.log" 2>&1 &
done
wait
python -m grayscott.evaluate
