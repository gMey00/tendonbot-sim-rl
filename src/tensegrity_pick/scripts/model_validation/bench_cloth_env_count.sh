#!/bin/bash
# Stage-0 de-risk: cloth env-count benchmark sweep driver.
#
# Runs bench_cloth_env_count.py once per env count (Isaac Sim cannot rebuild
# the stage in-process), collects the RESULT lines into a CSV and greps each
# log for PhysX buffer-overflow warnings (silently degraded cloth physics).
#
# Usage:
#     cd src/tensegrity_pick
#     conda activate env_isaaclab            # activation hooks set PYTHONPATH
#     bash scripts/model_validation/bench_cloth_env_count.sh [N1 N2 ...]
#
# Default sweep: 32 64 128 256.  Output:
#     doc/reports/data/cloth_env_benchmark.csv   (repo-root doc/)
#     /tmp/bench_cloth_N<k>.log                  (full per-run logs)
#
# Expect ~5-15 min per N (construction dominates; scales with N because
# cloth authoring + pre-settle are per-env).  Cold shader caches add ~10 min
# to the FIRST run only.

set -u
NS=(${@:-32 64 128 256})

PY="${CONDA_PREFIX:-/home/robot/miniconda3/envs/env_isaaclab}/bin/python"
OUT_DIR="$(git rev-parse --show-toplevel)/doc/reports/data"
CSV="$OUT_DIR/cloth_env_benchmark.csv"
mkdir -p "$OUT_DIR"

if [ ! -f "$CSV" ]; then
    echo "n_envs,construct_s,steps_per_s,env_steps_per_s,cuda_alloc_gb,cuda_reserved_gb,driver_used_gb,reset_s,sane,physx_overflow_warnings" > "$CSV"
fi

for N in "${NS[@]}"; do
    LOG="/tmp/bench_cloth_N${N}.log"
    echo "=== N=$N (log: $LOG) ==="
    PYTHONUNBUFFERED=1 "$PY" scripts/model_validation/bench_cloth_env_count.py \
        --headless --num_envs "$N" > "$LOG" 2>&1
    RES=$(grep -m1 "^RESULT," "$LOG" | cut -d, -f2-)
    OVF=$(grep -ciE "buffer overflow|collisionStackSize|contact.*dropped" "$LOG" || true)
    if [ -n "$RES" ]; then
        echo "$RES,$OVF" >> "$CSV"
        echo "  -> $RES (overflow warnings: $OVF)"
    else
        echo "$N,FAILED,,,,,,,," >> "$CSV"
        echo "  -> FAILED (see $LOG)"
        tail -5 "$LOG"
    fi
done

echo "CSV: $CSV"
column -t -s, "$CSV"
