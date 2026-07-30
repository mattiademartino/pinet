#!/usr/bin/env bash
# Projection-gradient ablation: true (IFT) vs straight-through gradient.
# Two arms x N datasets x five seeds, three runs at a time.

set -u
cd "$(dirname "$0")/../../.." || exit 1

JOBS=${JOBS:-3}
CONFIG=${CONFIG:-benchmark_small_autotune}
OUT=${OUT:-src/benchmarks/QP/results/grad_ablation}
IDS=${IDS:-"dc3_simple_1 dc3_nonconvex_1"}
# 0 keeps the batch size of the configuration. Datasets smaller than that batch
# size need it lowered, otherwise an epoch amounts to a single update.
BATCH_SIZE=${BATCH_SIZE:-0}
LOGS="$OUT/logs"
mkdir -p "$LOGS"

for id in $IDS; do
    for arm in "ift" "st"; do
        for seed in 0 1 2 3 4; do
            echo "$id $arm $seed"
        done
    done
done | xargs -P "$JOBS" -L 1 bash -c '
    id=$0; arm=$1; seed=$2
    echo "start $id $arm $seed"
    python -m src.benchmarks.QP.run_grad_ablation \
        --id "$id" --arm "$arm" --seed "$seed" \
        --config "'"$CONFIG"'" --out "'"$OUT"'" \
        --batch_size "'"$BATCH_SIZE"'" \
        > "'"$LOGS"'/${id}_${arm}_seed${seed}.log" 2>&1
    echo "done  $id $arm $seed (exit $?)"
'
echo "ALL RUNS FINISHED"
