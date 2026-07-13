#!/bin/bash -l
#SBATCH --job-name=stage0b_ema
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/home/woody/iwfa/iwfa131h/slurm_logs/stage0b_ema_%A_%a.out
#SBATCH --export=NONE
#SBATCH --array=0-7%4

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

# Stage 0b: validate the ONE positive iteration-21 finding — EMA (α=0.2) on the
# stock task-space actions — on full/grid seeds, FK targets. Throttled %4.
#  idx 0-4  ur5e_f140_ikabs EMA-only, seeds {0,1,2,42,123}  (baseline 7.8/56/22)
#  idx 5-7  ur5e_f140_ik    EMA,      seeds {0,42,123}      (baseline 8.3/51/21)
export REACH_FK_TARGETS=1 REACH_TS_EMA=0.2

GRID_SEEDS=(0 1 2 42 123)
IK_SEEDS=(0 42 123)
I=$SLURM_ARRAY_TASK_ID
if [ $I -lt 5 ]; then
    NAME=ema_ikabs; TASK=Template-Reach-UR5e-F140-IK-Abs-v0; SEED=${GRID_SEEDS[$I]}
else
    NAME=ema_ikrel; TASK=Template-Reach-UR5e-F140-IK-Rel-v0; SEED=${IK_SEEDS[$((I-5))]}
fi

echo "== stage0b[$I] CFG=$NAME Task=$TASK SEED=$SEED ema=0.2 $(date) =="
srun --ntasks=1 python -u scripts/skrl/train.py --task="$TASK" --algorithm=PPO --seed=$SEED --headless
echo "[done] CFG=$NAME SEED=$SEED $(date)"
