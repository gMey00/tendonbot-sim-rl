#!/bin/bash -l
#SBATCH --job-name=kin_sector
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/home/woody/iwfa/iwfa131h/slurm_logs/kin_sector_%A_%a.out
#SBATCH --export=NONE
#SBATCH --array=0-3%4

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

# Kinova FK sampling-sector probe: does a narrower sector close the pose% gap
# (22% vs URs' 86%)? kinova_f140 joint, half-range {1.0, 1.2} x seeds {7, 77}
# (disjoint from grid seeds -> no dir collisions). Baseline hr=1.5: 5.9/79/22.
export REACH_FK_TARGETS=1

SEEDS=(7 77)
HRS=(1.0 1.2)
HR=${HRS[$(( SLURM_ARRAY_TASK_ID / 2 ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % 2 ))]}
export REACH_FK_HALF_RANGE=$HR

echo "== kin_sector[$SLURM_ARRAY_TASK_ID] HR=$HR SEED=$SEED $(date) =="
srun --ntasks=1 python -u scripts/skrl/train.py --task=Template-Reach-Kinova-F140-v0 --algorithm=PPO --seed=$SEED --headless
echo "[done] HR=$HR SEED=$SEED $(date)"
