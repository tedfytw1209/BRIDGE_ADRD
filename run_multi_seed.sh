#!/bin/bash
# Run the main 5-seed experiment sequentially.
# Seeds (1, 2, 3, 4, 42) are managed inside run_multi_seed.py.
#
# Usage:
#   bash run_multi_seed.sh                          # run all 5 seeds, all steps
#   bash run_multi_seed.sh --seeds 1 42             # run specific seeds
#   bash run_multi_seed.sh --force                  # rerun even if done markers present
#   bash run_multi_seed.sh --skip-aggregate         # skip CI aggregation at the end
#   bash run_multi_seed.sh --steps train decoder_mod  # run selected steps only
#
# Any arguments are passed straight through to run_multi_seed.py.

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
EXTRA_ARGS="$*"

mkdir -p "${WORK_DIR}/results_multi_seed"

cd "$WORK_DIR"
python run_multi_seed.py $EXTRA_ARGS
