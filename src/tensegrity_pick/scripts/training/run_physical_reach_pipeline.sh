#!/usr/bin/env bash
# run_physical_reach_pipeline.sh — 5-seed training + evaluation + plots for the
# two physical tendon reach variants (2026-07 rework).
#
# Resumable: completed trainings (final checkpoint exists) and completed eval
# JSONs are skipped, so the script can be re-launched after an interruption.
#
# USAGE (run detached — a full pass is many hours):
#   cd src/tensegrity_pick
#   nohup ./scripts/training/run_physical_reach_pipeline.sh \
#       > logs/skrl/reach/_physical_pipeline.log 2>&1 &
#
# PHASES
#   1. Train  : {tensegrity_physical_tendon, tensegrity_physical_hier} × seeds 0–4
#               (48 000 timesteps each; sanity gate aborts after the FIRST run
#               if the reward did not improve or went NaN)
#   2. Eval   : zero / random / checkpoint × seeds 0–4 × both Play variants
#   3. Plots  : per-variant training figures, 07 seed aggregate,
#               08–12 eval aggregate figures

set -uo pipefail

CONDA_ENV="env_isaaclab"
SEEDS=(0 1 2 3 4)
TIMESTEPS_FINAL=48000

# Activate the env once — `conda run` does not reliably apply the env's
# activate.d hooks (Isaac Sim PYTHONPATH), the activate pattern does.
# Isaac's setup_conda_env.sh reads unset vars (ZSH_VERSION), so relax -u
# around the activation.
set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u
PYBIN="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"

declare -A TASK_IDS=(
    [tensegrity_physical_tendon]="Template-Reach-Tensegrity-Physical-Tendon-v0"
    [tensegrity_physical_hier]="Template-Reach-Tensegrity-Physical-Hierarchical-v0"
)
declare -A PLAY_IDS=(
    [tensegrity_physical_tendon]="Template-Reach-Tensegrity-Physical-Tendon-Play-v0"
    [tensegrity_physical_hier]="Template-Reach-Tensegrity-Physical-Hierarchical-Play-v0"
)

# Optional variant filter as positional args (enables running the two variants
# in parallel pipeline instances).  With a filter, the plots phase is skipped —
# run the script once without arguments at the end (train/eval are skipped as
# done) or invoke the plot commands manually.
if [[ $# -gt 0 ]]; then
    VARIANTS=("$@")
    DO_PLOTS=false
else
    VARIANTS=(tensegrity_physical_tendon tensegrity_physical_hier)
    DO_PLOTS=true
fi
for v in "${VARIANTS[@]}"; do
    if [[ -z "${TASK_IDS[$v]+x}" ]]; then
        echo "ERROR: unknown variant '$v'"; exit 1
    fi
done

PY() {
    "$PYBIN" -u "$@"
}

log() {
    echo "[$(date '+%F %T')] $*"
}

run_dir_for_seed() {
    # newest run dir for (variant, seed)
    ls -d "logs/skrl/reach/$1"/*_ppo_torch_seed"$2" 2>/dev/null | sort | tail -1
}

train_done() {
    local d
    d=$(run_dir_for_seed "$1" "$2")
    [[ -n "$d" && -f "$d/checkpoints/agent_${TIMESTEPS_FINAL}.pt" ]]
}

# ---------------------------------------------------------------------------
# Phase 1 — training
# ---------------------------------------------------------------------------
FIRST_RUN_GATED=false

for VARIANT in "${VARIANTS[@]}"; do
    TASK="${TASK_IDS[$VARIANT]}"
    for SEED in "${SEEDS[@]}"; do
        if train_done "$VARIANT" "$SEED"; then
            log "TRAIN skip  $VARIANT seed=$SEED (final checkpoint exists)"
            continue
        fi
        # remove partial run dirs for this seed (crashed/interrupted attempts)
        for d in logs/skrl/reach/"$VARIANT"/*_ppo_torch_seed"$SEED"; do
            [[ -d "$d" ]] || continue
            log "TRAIN clean $VARIANT seed=$SEED — removing partial run $d"
            rm -rf "$d"
        done

        log "TRAIN start $VARIANT seed=$SEED ($TASK)"
        PY scripts/skrl/train.py --task "$TASK" --headless --seed "$SEED"
        rc=$?
        if [[ $rc -ne 0 ]] || ! train_done "$VARIANT" "$SEED"; then
            log "TRAIN FAILED $VARIANT seed=$SEED (exit $rc) — aborting pipeline"
            exit 1
        fi
        log "TRAIN done  $VARIANT seed=$SEED"

        # Sanity gate after the very first run of the campaign
        if [[ "$FIRST_RUN_GATED" == false ]]; then
            FIRST_RUN_GATED=true
            RUN_DIR=$(run_dir_for_seed "$VARIANT" "$SEED")
            log "GATE checking learning signal in $RUN_DIR"
            PY - "$RUN_DIR" <<'PYEOF'
import sys
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

acc = EventAccumulator(sys.argv[1]); acc.Reload()
tag = "Reward / Total reward (mean)"
ev = acc.Scalars(tag)
vals = np.array([e.value for e in ev])
first, last = vals[0], np.mean(vals[-max(1, len(vals)//10):])
print(f"[gate] {tag}: first={first:.3f} last10%={last:.3f} n={len(vals)}")
assert np.isfinite(vals).all(), "NaN/Inf in total reward"
assert last > first + 0.2, f"no learning signal: {first:.3f} -> {last:.3f}"
# Degenerate-run guard: a "good" reward with collapsed episodes means the
# policy found an early-termination exploit, not a solution.
ep = np.array([e.value for e in acc.Scalars("Episode / Total timesteps (mean)")])
ep_last = np.mean(ep[-max(1, len(ep)//10):])
print(f"[gate] episode length last10%={ep_last:.1f} (healthy = 180)")
assert ep_last > 150, f"collapsed episodes: mean length {ep_last:.1f} < 150"
print("[gate] PASS")
PYEOF
            if [[ $? -ne 0 ]]; then
                log "GATE FAILED — aborting pipeline (inspect $RUN_DIR)"
                exit 2
            fi
        fi
    done
done

# ---------------------------------------------------------------------------
# Phase 2 — evaluation matrix
# ---------------------------------------------------------------------------
EVAL_DIR="logs/skrl/reach/eval"
mkdir -p "$EVAL_DIR"

for VARIANT in "${VARIANTS[@]}"; do
    PLAY="${PLAY_IDS[$VARIANT]}"
    for SEED in "${SEEDS[@]}"; do
        for AGENT in zero random; do
            OUT="$EVAL_DIR/${AGENT}_${VARIANT}_seed${SEED}.json"
            if [[ -f "$OUT" ]]; then
                log "EVAL skip  $AGENT $VARIANT seed=$SEED"
                continue
            fi
            log "EVAL start $AGENT $VARIANT seed=$SEED"
            PY scripts/skrl/evaluate_reach.py --agent "$AGENT" --task "$PLAY" \
                --seed "$SEED" --num_episodes 10 --headless \
                || { log "EVAL FAILED $AGENT $VARIANT seed=$SEED"; exit 3; }
        done

        if [[ "$VARIANT" == "tensegrity_physical_hier" ]]; then
            OUT="$EVAL_DIR/heuristic_${VARIANT}_seed${SEED}.json"
            if [[ -f "$OUT" ]]; then
                log "EVAL skip  heuristic $VARIANT seed=$SEED"
            else
                log "EVAL start heuristic $VARIANT seed=$SEED"
                PY scripts/skrl/evaluate_reach.py --agent heuristic --task "$PLAY" \
                    --seed "$SEED" --num_episodes 10 --headless \
                    || { log "EVAL FAILED heuristic $VARIANT seed=$SEED"; exit 3; }
            fi
        fi

        OUT="$EVAL_DIR/checkpoint_${VARIANT}_seed${SEED}.json"
        if [[ -f "$OUT" ]]; then
            log "EVAL skip  checkpoint $VARIANT seed=$SEED"
            continue
        fi
        RUN_DIR=$(run_dir_for_seed "$VARIANT" "$SEED")
        if [[ -z "$RUN_DIR" ]]; then
            log "EVAL FAILED checkpoint $VARIANT seed=$SEED — no run dir"
            exit 3
        fi
        log "EVAL start checkpoint $VARIANT seed=$SEED ($RUN_DIR)"
        PY scripts/skrl/evaluate_reach.py --agent checkpoint --task "$PLAY" \
            --seed "$SEED" --num_episodes 10 --headless --checkpoint "$RUN_DIR" \
            || { log "EVAL FAILED checkpoint $VARIANT seed=$SEED"; exit 3; }
    done
done

# ---------------------------------------------------------------------------
# Phase 3 — plots
# ---------------------------------------------------------------------------
if [[ "$DO_PLOTS" != true ]]; then
    log "PLOT skipped (variant-filtered instance)"
    log "PIPELINE COMPLETE (variants: ${VARIANTS[*]})"
    exit 0
fi
log "PLOT per-variant training figures"
for VARIANT in "${VARIANTS[@]}"; do
    PY scripts/plotting/plot_reach_training_results.py --variant "$VARIANT" \
        || log "PLOT WARN: per-variant plots failed for $VARIANT"
done

log "PLOT 07 seed aggregate (PD + tendon May runs, physical July runs)"
PY scripts/plotting/plot_reach_training_results.py --seeds \
    logs/skrl/reach/tensegrity/2026-05-22_*_ppo_torch \
    logs/skrl/reach/tensegrity_tendon/2026-05-22_*_ppo_torch \
    logs/skrl/reach/tensegrity_physical_tendon/*_ppo_torch_seed[0-4] \
    logs/skrl/reach/tensegrity_physical_hier/*_ppo_torch_seed[0-4] \
    || log "PLOT WARN: seed aggregate failed"

log "PLOT 08-12 eval aggregate figures"
PY scripts/plotting/plot_reach_eval_results.py \
    || log "PLOT WARN: eval figures failed"

log "PIPELINE COMPLETE"
