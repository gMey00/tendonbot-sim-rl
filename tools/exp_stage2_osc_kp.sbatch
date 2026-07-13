#!/bin/bash -l
#SBATCH --job-name=stage2_kp
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/home/woody/iwfa/iwfa131h/slurm_logs/stage2_kp_%A_%a.out
#SBATCH --export=NONE
#SBATCH --array=0-5%2

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

# Stage 2 (research report Rank 3i): OSC stiffness sweep on ur10_f140_osc, FK
# targets. Baseline: iter-20 grid v2 kp100 clamp(50,200) -> 9.7cm/51%/19%pose.
# Threshold: if no clamp beats ~7 cm / 30% pose, stop tuning OSC in free space.
export REACH_FK_TARGETS=1

SEEDS=(0 42 123)
CFG=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % 3 ))]}

case $CFG in
  0) NAME=kp200_c100_400
     export REACH_OSC_STIFFNESS=200 REACH_OSC_STIFFNESS_MIN=100 REACH_OSC_STIFFNESS_MAX=400 ;;
  1) NAME=kp300_c200_500
     export REACH_OSC_STIFFNESS=300 REACH_OSC_STIFFNESS_MIN=200 REACH_OSC_STIFFNESS_MAX=500
     sleep 25 ;;  # stagger vs cfg 0 (same variant+seed set)
esac

echo "== stage2[$SLURM_ARRAY_TASK_ID] CFG=$NAME SEED=$SEED $(date) =="
srun --ntasks=1 python -u scripts/skrl/train.py --task=Template-Reach-UR10-F140-OSC-v0 --algorithm=PPO --seed=$SEED --headless
echo "[done] CFG=$NAME SEED=$SEED $(date)"
