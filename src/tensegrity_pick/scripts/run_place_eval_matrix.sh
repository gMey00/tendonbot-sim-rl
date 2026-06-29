#!/bin/bash
# Run the full cube-place evaluation matrix:
#   3 variants x {checkpoint (5 seeds), zero (5 seeds), random (5 seeds)}
# Each cell: 50 parallel envs x --num_episodes episodes.
# Intended to run inside an Isaac conda env on a GPU node (see eval_place_alex).
#
# Usage:  bash run_place_eval_matrix.sh [NUM_EPISODES]
set -uo pipefail

NUM_EPISODES="${1:-10}"
PY="python3 scripts/skrl/evaluate_place.py"
LOGROOT="logs/skrl/cube_place"

# variant tag -> Play task id
declare -A PLAY_TASK=(
  [tensegrity]="Template-Tensegrity-Cube-Place-Play-v0"
  [tensegrity_tendon]="Template-Tensegrity-Cube-Place-Tendon-Play-v0"
  [tensegrity_physical_tendon]="Template-Tensegrity-Cube-Place-Physical-Tendon-Play-v0"
)

run_one () {
  local agent="$1" variant="$2" seed="$3" extra="$4"
  local task="${PLAY_TASK[$variant]}"
  local out="${LOGROOT}/eval/${agent}_${variant}_seed${seed}.json"
  if [[ -f "$out" ]]; then echo "[skip] $out exists"; return 0; fi
  echo "=== eval agent=$agent variant=$variant seed=$seed ==="
  $PY --task "$task" --agent "$agent" --seed "$seed" \
      --num_episodes "$NUM_EPISODES" --headless $extra \
      || echo "[FAIL] agent=$agent variant=$variant seed=$seed"
}

for variant in tensegrity tensegrity_tendon tensegrity_physical_tendon; do
  for seed in 0 1 2 3 4; do
    # locate the trained run dir for this variant/seed
    run_dir=$(ls -d ${LOGROOT}/${variant}/2026-06-22_*_seed${seed} 2>/dev/null | tail -1)
    if [[ -n "$run_dir" ]]; then
      run_one checkpoint "$variant" "$seed" "--checkpoint $run_dir"
    else
      echo "[WARN] no run dir for $variant seed$seed"
    fi
  done
  # baselines: same 5 env-seeds for mean+/-std lower bounds
  for seed in 0 1 2 3 4; do
    run_one zero   "$variant" "$seed" ""
    run_one random "$variant" "$seed" ""
  done
done
echo "[done] eval matrix complete"
