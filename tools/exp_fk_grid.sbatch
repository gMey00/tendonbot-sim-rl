#!/bin/bash -l
#SBATCH --job-name=fk_grid
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/home/woody/iwfa/iwfa131h/slurm_logs/fk_grid_%A_%a.out
#SBATCH --export=NONE
#SBATCH --array=0-119

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

# Controller-hardening baseline: full 24-variant grid on PER-ROBOT FK targets
# (each robot samples its own reachable poses). 24 variants x 5 seeds = 120.
export REACH_FK_TARGETS=1

ARMS=(Kinova-F140 Kinova-Frankenstein UR10-F140 UR10-Frankenstein UR5e-F140 UR5e-Frankenstein)
SUFFIX=("" "-IK-Rel" "-IK-Abs" "-OSC")
SEEDS=(0 1 2 42 123)

VAR=$(( SLURM_ARRAY_TASK_ID / 5 ))      # 0..23 -> variant
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % 5 ))]}
ARM=${ARMS[$(( VAR / 4 ))]}
SP=${SUFFIX[$(( VAR % 4 ))]}
TASK="Template-Reach-${ARM}${SP}-v0"

echo "== fk_grid[$SLURM_ARRAY_TASK_ID] Task=$TASK SEED=$SEED FK=1 $(date) =="
srun --ntasks=1 python -u scripts/skrl/train.py --task="$TASK" --algorithm=PPO --seed=$SEED --headless
echo "[done] $TASK SEED=$SEED $(date)"
