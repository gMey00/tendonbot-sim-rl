#!/bin/bash -l
#SBATCH --job-name=orient_refine
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/home/woody/iwfa/iwfa131h/slurm_logs/orient_refine_%A_%a.out
#SBATCH --export=NONE
#SBATCH --array=0-3

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

# ── Iteration-16 experiment: position-gated orientation refinement ──
# Redundant-arm variants only; weight 0.5, gate/std defaults (pos 0.1 m, ori 0.3 rad).
export REACH_ORIENT_REFINE_WEIGHT=0.5

TASKS=(
  Template-Reach-UR5e-Frankenstein-IK-Abs-v0
  Template-Reach-Kinova-F140-OSC-v0
  Template-Reach-Kinova-Frankenstein-IK-Abs-v0
  Template-Reach-UR5e-Frankenstein-OSC-v0
)
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"

echo "============================================================"
echo " orient_refine[$SLURM_ARRAY_TASK_ID]  Task=$TASK  weight=$REACH_ORIENT_REFINE_WEIGHT  $(date)"
echo " Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-?}"
echo "============================================================"

srun --ntasks=1 python -u scripts/skrl/train.py --task="$TASK" --algorithm=PPO --seed=42 --headless

echo "[done] $(date)"
