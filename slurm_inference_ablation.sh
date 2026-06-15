#!/bin/bash
# Submit one SLURM job per inference-ablation stage so all 7 stages run in parallel.
# Each job re-uses the trained models from run_multi_seed.py (results_multi_seed/)
# and runs only imputation + classification with different graph/fusion options.
# No GPU-intensive training — only forward passes, so wall time is much shorter.
#
# Usage:
#   bash slurm_inference_ablation.sh              # submit all 7 stages
#   bash slurm_inference_ablation.sh S3_gated_fusion   # submit one stage by name
#   bash slurm_inference_ablation.sh 3                 # submit one stage by 1-based index
#
# Optional env vars (override defaults):
#   MODEL_SOURCE=results_multi_seed  # where trained models live

# ── cluster settings (edit these) ─────────────────────────────────────────────
PARTITION="hpg-turin"
CPUS=8
MEM="32G"
TIME="6:00:00"
GRES="gpu:1"            # set to "" if no GPU needed
CONDA_ENV="psn"
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_SOURCE="${MODEL_SOURCE:-results_multi_seed}"
# ──────────────────────────────────────────────────────────────────────────────

STAGES=(
    "S1_encoder_decoder"
    "S2_average_fusion"
    "S3_gated_fusion"
    "S4_topk_filter"
    "S5_no_reliable_ratio"
    "S6_no_topk"
    "S7_no_gated_fusion"
)

# Build the list of stages to submit
if [ -n "$1" ]; then
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

GRES_FLAG=""
if [ -n "$GRES" ]; then
    GRES_FLAG="--gres=$GRES"
fi

echo "Submitting ${#SUBMIT_STAGES[@]} stage(s)  (model source: $MODEL_SOURCE)..."

for stage in "${SUBMIT_STAGES[@]}"; do
    job_name="infer_abl_${stage}"
    log_file="${WORK_DIR}/results_inference_ablation/${stage}/slurm_%j.log"

    mkdir -p "${WORK_DIR}/results_inference_ablation/${stage}"

    sbatch \
        --job-name="$job_name" \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --time="$TIME" \
        $GRES_FLAG \
        --partition="$PARTITION" \
        --account=ruogu.fang \
        --qos=ruogu.fang \
        --output="$log_file" \
        --error="$log_file" \
        --wrap="
            module load conda
            conda activate $CONDA_ENV
            cd '$WORK_DIR'
            python run_inference_ablation.py \
                --stage '$stage' \
                --model-source-dir '$MODEL_SOURCE'
        "

    echo "  Submitted: $job_name  (log -> results_inference_ablation/${stage}/slurm_<jobid>.log)"
done

echo "Done. Monitor with: squeue -u \$USER"
