#!/bin/bash -l
#SBATCH --job-name=wrist_ab
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/home/woody/iwfa/iwfa131h/slurm_logs/wrist_ab_%A_%a.out
#SBATCH --export=NONE
#SBATCH --array=0-11

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

# Tensegrity-wrist matched-pair A/B on UR5e-Frankenstein (joint action, FK targets).
# 4 configs x 3 seeds. All write to the same variant dir, so stagger config start
# to keep the timestamp-to-the-second run-dir names distinct (avoid the collision
# that bit the OSC sweep).
export REACH_FK_TARGETS=1

SEEDS=(0 1 42)
CFG=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % 3 ))]}

case $CFG in
  0) NAME=base_active;  export REACH_FK_SAMPLE_BASE_ONLY=1 ;;
  1) NAME=base_locked;  export REACH_FK_SAMPLE_BASE_ONLY=1; export REACH_LOCK_WRIST=1 ;;
  2) NAME=frank_active ;;
  3) NAME=frank_locked; export REACH_LOCK_WRIST=1 ;;
esac

# Per-config start stagger so run-dir timestamps differ across configs.
sleep $(( CFG * 25 ))

echo "== wrist_ab[$SLURM_ARRAY_TASK_ID] CFG=$NAME SEED=$SEED (base_only=${REACH_FK_SAMPLE_BASE_ONLY:-0} lock=${REACH_LOCK_WRIST:-0}) $(date) =="
srun --ntasks=1 python -u scripts/skrl/train.py --task=Template-Reach-UR5e-Frankenstein-v0 --algorithm=PPO --seed=$SEED --headless
echo "[done] CFG=$NAME SEED=$SEED $(date)"
