#!/usr/bin/env bash
# RL baselines on the QP contextual bandit: five arms x datasets x seeds.
# The projection configuration is the one of the pinet baseline, so that the
# environment is identical across all arms of the comparison.
#
# The seed is the outermost loop on purpose: one complete pass over every arm
# and every dataset finishes before the second seed starts, so a full (if
# single-seed) picture is available as early as possible, and each further seed
# only adds error bars. Parsing between passes is safe and shows what is done.

set -u
cd "$(dirname "$0")/../../.." || exit 1

JOBS=${JOBS:-3}
SEEDS=${SEEDS:-"0 1 2 3 4"}
IDS=${IDS:-"dc3_simple_1 dc3_nonconvex_1"}
ALGOS=${ALGOS:-"pinet pinet_stochastic reinforce ppo sac sac_proj"}
CONFIG=${CONFIG:-benchmark_small_autotune}
RL_CONFIG=${RL_CONFIG:-rl_default}
OUT=${OUT:-src/benchmarks/QP/results/rl}
# Extra flags forwarded to every run, e.g. EXTRA="--total_env_samples 2000000".
EXTRA=${EXTRA:-}
# With RESUME=1 (the default) a run whose .npz already exists is skipped, so an
# interrupted batch can be relaunched without redoing finished work. A run that
# was killed mid-training left no .npz and therefore restarts from scratch:
# run_rl.py writes its results only once, at the end.
RESUME=${RESUME:-1}
LOGS="$OUT/logs"
mkdir -p "$LOGS"

for seed in $SEEDS; do
    for id in $IDS; do
        for algo in $ALGOS; do
            if [ "$RESUME" = "1" ] && [ -f "$OUT/${id}_${algo}_seed${seed}.npz" ]; then
                echo "skip  $id $algo $seed (already done)" >&2
                continue
            fi
            echo "$id $algo $seed"
        done
    done
done | xargs -P "$JOBS" -L 1 bash -c '
    id=$0; algo=$1; seed=$2
    echo "start $id $algo $seed"
    python -m src.benchmarks.QP.run_rl \
        --id "$id" --algo "$algo" --seed "$seed" \
        --config "'"$CONFIG"'" --rl_config "'"$RL_CONFIG"'" --out "'"$OUT"'" \
        '"$EXTRA"' \
        > "'"$LOGS"'/${id}_${algo}_seed${seed}.log" 2>&1
    echo "done  $id $algo $seed (exit $?)"
'
echo "ALL RL RUNS FINISHED"
