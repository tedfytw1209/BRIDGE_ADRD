#!/bin/bash
# Submit a single SLURM job that runs the main 5-seed experiment sequentially.
# Seeds (1, 2, 3, 4, 42) are managed inside run_multi_seed.py, which executes
# the full pipeline (train → decoder_mod → imputation → error_analysis →
# classification → eval_demographics) for each seed and then aggregates 95% CI
# results.  This job produces Figure 3 and all main-table confidence intervals.
#
# Usage:
#   bash slurm_multi_seed.sh                          # run all 5 seeds, all steps
#   bash slurm_multi_seed.sh --seeds 1 42             # run specific seeds
#   bash slurm_multi_seed.sh --force                  # rerun even if done markers present
#   bash slurm_multi_seed.sh --skip-aggregate         # skip CI aggregation at the end
#   bash slurm_multi_seed.sh --steps train decoder_mod  # run selected steps only
#
# Any arguments are passed straight through to run_multi_seed.py.

# ── cluster settings (edit these) ─────────────────────────────────────────────
CPUS=8
MEM="64G"
TIME="72:00:00"
GPUS=1                  # GPUs per node; set to "" if no GPU needed
CONDA_ENV="psn"         # conda environment name
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"   # directory containing this script
EXTRA_ARGS="$*"
# ──────────────────────────────────────────────────────────────────────────────

# Build optional GPU flag
GPU_FLAG=""
if [ -n "$GPUS" ]; then
    GPU_FLAG="--gpus-per-node=$GPUS"
fi

mkdir -p "${WORK_DIR}/results_multi_seed"

job_name="psn_multi_seed"
log_file="${WORK_DIR}/results_multi_seed/slurm_%j.log"

sbatch \
    --job-name="$job_name" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    $GPU_FLAG \
    --output="$log_file" \
    --error="$log_file" \
    --wrap="
        module load conda
        conda activate $CONDA_ENV
        cd '$WORK_DIR'
        python run_multi_seed.py $EXTRA_ARGS
    "

echo "  Submitted: $job_name  (log -> results_multi_seed/slurm_<jobid>.log)"
echo "Done. Monitor with: squeue -u \$USER"
