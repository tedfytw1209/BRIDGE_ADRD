#!/bin/bash
# Run one background process per inference-ablation stage (all 7 in parallel).
# Re-uses trained models from run_multi_seed.py (results_multi_seed/).
#
# Usage:
#   bash run_inference_ablation.sh              # run all 7 stages
#   bash run_inference_ablation.sh S3_gated_fusion   # run one stage by name
#   bash run_inference_ablation.sh 3                 # run one stage by 1-based index
#
# Optional env vars (override defaults):
#   MODEL_SOURCE=results_multi_seed  # where trained models live

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_SOURCE="${MODEL_SOURCE:-results_multi_seed}"

STAGES=(
    "S1_encoder_decoder"
    "S2_average_fusion"
    "S3_gated_fusion"
    "S4_topk_filter"
    "S5_no_reliable_ratio"
    "S6_no_topk"
    "S7_no_gated_fusion"
)

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

echo "Starting ${#SUBMIT_STAGES[@]} stage(s)  (model source: $MODEL_SOURCE)..."

PIDS=()
for stage in "${SUBMIT_STAGES[@]}"; do
    log_dir="${WORK_DIR}/results_inference_ablation/${stage}"
    log_file="${log_dir}/run.log"
    mkdir -p "$log_dir"

    echo "  Starting: $stage  (log -> results_inference_ablation/${stage}/run.log)"
    (
        cd "$WORK_DIR"
        python run_inference_ablation.py \
            --stage "$stage" \
            --model-source-dir "$MODEL_SOURCE"
    ) > "$log_file" 2>&1 &
    PIDS+=($!)
done

echo "Waiting for all stages to complete..."
for pid in "${PIDS[@]}"; do
    wait "$pid"
done
echo "Done."
