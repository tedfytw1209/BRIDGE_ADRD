#!/bin/bash
# Submit one SLURM job per ablation stage so all 6 stages run in parallel.
# Each job runs all 5 seeds for its stage sequentially.
#
# Usage:
#   bash slurm_ablation.sh              # submit all 6 stages
#   bash slurm_ablation.sh S_trim_topk10  # submit one stage by name
#   bash slurm_ablation.sh 3            # submit one stage by 1-based index

# ── cluster settings (edit these) ─────────────────────────────────────────────
CPUS=8
MEM="64G"
TIME="24:00:00"
GPUS=1                  # GPUs per node; set to "" if no GPU needed
CONDA_ENV="psn"         # conda environment name (or comment out the activation)
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"   # directory containing this script
# ──────────────────────────────────────────────────────────────────────────────

STAGES=(
    # modality_a=1 (absent-modality patients masked from GCN neighbors)
    "S_trim_topk1_rr1_ma1"
    "S_trim_topk3_rr1_ma1"
    "S_trim_topk5_rr1_ma1"
    "S_trim_topk10_rr1_ma1"
    "S_trim_topk20_rr1_ma1"
    "S_trim_nofilter_rr1_ma1"
)

# Build the list of stages to submit
if [ -n "$1" ]; then
    # Single stage: accept name or 1-based index
    arg="$1"
    if [[ "$arg" =~ ^[0-9]+$ ]]; then
        idx=$(( arg - 1 ))
        if [ "$idx" -lt 0 ] || [ "$idx" -ge "${#STAGES[@]}" ]; then
            echo "Error: index $arg out of range (1–${#STAGES[@]})"
            exit 1
        fi
        SUBMIT_STAGES=("${STAGES[$idx]}")
    else
        found=0
        for s in "${STAGES[@]}"; do
            if [ "$s" = "$arg" ]; then
                SUBMIT_STAGES=("$s")
                found=1
                break
            fi
        done
        if [ "$found" -eq 0 ]; then
            echo "Error: unknown stage '$arg'"
            echo "Valid stages: ${STAGES[*]}"
            exit 1
        fi
    fi
else
    SUBMIT_STAGES=("${STAGES[@]}")
fi

# Build optional GPU flag
GPU_FLAG=""
if [ -n "$GPUS" ]; then
    GPU_FLAG="--gpus-per-node=$GPUS"
fi

echo "Submitting ${#SUBMIT_STAGES[@]} stage(s)..."

for stage in "${SUBMIT_STAGES[@]}"; do
    job_name="ablation_${stage}"
    log_file="${WORK_DIR}/results_ablation/${stage}/slurm_%j.log"

    # Ensure log directory exists
    mkdir -p "${WORK_DIR}/results_ablation/${stage}"

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
            python run_ablation_multi_seed.py --stage '$stage'
        "

    echo "  Submitted: $job_name  (log -> results_ablation/${stage}/slurm_<jobid>.log)"
done

echo "Done. Monitor with: squeue -u \$USER"
