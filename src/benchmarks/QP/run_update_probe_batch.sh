#!/usr/bin/env bash
# One-step counterfactual probe, along both trajectories, on both small datasets.

set -u
cd "$(dirname "$0")/../../.." || exit 1

JOBS=${JOBS:-3}
SEEDS=${SEEDS:-"0 1 2"}
CONFIG=${CONFIG:-benchmark_small_autotune}
OUT=${OUT:-src/benchmarks/QP/results/update_probe}
LOGS="$OUT/logs"
mkdir -p "$LOGS"

for id in "dc3_simple_1" "dc3_nonconvex_1"; do
    for traj in "ift" "st"; do
        for seed in $SEEDS; do
            echo "$id $traj $seed"
        done
    done
done | xargs -P "$JOBS" -L 1 bash -c '
    id=$0; traj=$1; seed=$2
    python -m src.benchmarks.QP.run_update_probe \
        --id "$id" --trajectory "$traj" --seed "$seed" \
        --config "'"$CONFIG"'" --out "'"$OUT"'" \
        > "'"$LOGS"'/${id}_${traj}_seed${seed}.log" 2>&1
    echo "done $id $traj $seed (exit $?)"
'
echo "ALL PROBE RUNS FINISHED"
