#!/bin/bash
# Run one background process per ablation stage (all 6 in parallel).
# Each process runs all 5 seeds for its stage sequentially.
#
# Usage:
#   bash run_ablation.sh              # run all 6 stages
#   bash run_ablation.sh S_trim_topk10_rr1_ma1  # run one stage by name
#   bash run_ablation.sh 3            # run one stage by 1-based index

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"

STAGES=(
    "S_trim_topk1_rr1_ma1"
    "S_trim_topk3_rr1_ma1"
    "S_trim_topk5_rr1_ma1"
    "S_trim_topk10_rr1_ma1"
    "S_trim_topk20_rr1_ma1"
    "S_trim_nofilter_rr1_ma1"
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

echo "Starting ${#SUBMIT_STAGES[@]} stage(s)..."

PIDS=()
for stage in "${SUBMIT_STAGES[@]}"; do
    log_dir="${WORK_DIR}/results_ablation/${stage}"
    log_file="${log_dir}/run.log"
    mkdir -p "$log_dir"

    echo "  Starting: $stage  (log -> results_ablation/${stage}/run.log)"
    (
        cd "$WORK_DIR"
        python run_ablation_multi_seed.py --stage "$stage"
    ) > "$log_file" 2>&1 &
    PIDS+=($!)
done

echo "Waiting for all stages to complete..."
for pid in "${PIDS[@]}"; do
    wait "$pid"
done
echo "Done."
