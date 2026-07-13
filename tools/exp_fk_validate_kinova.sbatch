#!/bin/bash -l
#SBATCH --job-name=fk_valK
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/home/woody/iwfa/iwfa131h/slurm_logs/fk_valK_%A_%a.out
#SBATCH --export=NONE
#SBATCH --array=0-5

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

# Spot-validate the FK sampler on Kinova (continuous joints + iteration-16 spawn
# fix; 7-DOF F140, 9-DOF Frankenstein, and OSC). If these converge with sensible
# pos + pose%, the FK path is safe for the full grid.
export REACH_FK_TARGETS=1

SEEDS=(0 42)
TASKS=(Template-Reach-Kinova-F140-v0 Template-Reach-Kinova-Frankenstein-v0 Template-Reach-Kinova-F140-OSC-v0)
T=$(( SLURM_ARRAY_TASK_ID / 2 ))
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % 2 ))]}
TASK=${TASKS[$T]}

echo "== fk_valK[$SLURM_ARRAY_TASK_ID] Task=$TASK SEED=$SEED FK=1 $(date) =="
srun --ntasks=1 python -u scripts/skrl/train.py --task="$TASK" --algorithm=PPO --seed=$SEED --headless
echo "[done] $TASK SEED=$SEED $(date)"
