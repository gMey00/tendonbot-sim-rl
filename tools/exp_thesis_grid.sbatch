#!/bin/bash -l
#SBATCH --job-name=thesis_grid
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/home/woody/iwfa/iwfa131h/slurm_logs/thesis_grid_%A_%a.out
#SBATCH --export=NONE
#SBATCH --array=0-119%4

unset SLURM_EXPORT_ENV
source "$HOME/studentische-arbeiten/.config/env_vars.sh"
module load gcc/15.2.0 2>/dev/null || true
if module load python 2>/dev/null && command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    export CONDA_PKGS_DIRS="${INSTALL_PATH}/.cache/conda_pkgs"
else
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
fi
conda activate "${ISAACLAB_ENV_NAME}"
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HTTP_PROXY=http://proxy.nhr.fau.de:80
export HTTPS_PROXY=http://proxy.nhr.fau.de:80
export no_proxy=localhost,127.0.0.1,.nhr.fau.de,.fau.de
export NO_PROXY=localhost,127.0.0.1,.nhr.fau.de,.fau.de
cd "${PROJECT_PATH}/src/tensegrity_pick"
export PYTHONUNBUFFERED=1

# ── THESIS GRID (final, consistent): 24 variants × 5 seeds ──────────────────
# Locked config: FK targets (fixed sampler: self-collision filter + floor
# clearance + Kinova sector), EMA α=0.2 on ALL action spaces (joint has it
# built in; task-space gets REACH_TS_EMA — iteration-21 adopted knob).
# Kinova sector set below from the iteration-22 probe verdict.
export REACH_FK_TARGETS=1
export REACH_TS_EMA=0.2
# REACH_FK_HALF_RANGE is set (or left default 1.5) by the submit-time sed;
# see tracking log iteration 22.
export REACH_FK_HALF_RANGE=1.0

ARMS=(Kinova-F140 Kinova-Frankenstein UR10-F140 UR10-Frankenstein UR5e-F140 UR5e-Frankenstein)
SUFFIX=("" "-IK-Rel" "-IK-Abs" "-OSC")
SEEDS=(0 1 2 42 123)

VAR=$(( SLURM_ARRAY_TASK_ID / 5 ))
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % 5 ))]}
ARM=${ARMS[$(( VAR / 4 ))]}
SP=${SUFFIX[$(( VAR % 4 ))]}
TASK="Template-Reach-${ARM}${SP}-v0"

echo "== thesis_grid[$SLURM_ARRAY_TASK_ID] Task=$TASK SEED=$SEED ema=0.2 hr=$REACH_FK_HALF_RANGE $(date) =="
srun --ntasks=1 python -u scripts/skrl/train.py --task="$TASK" --algorithm=PPO --seed=$SEED --headless
echo "[done] $TASK SEED=$SEED $(date)"
