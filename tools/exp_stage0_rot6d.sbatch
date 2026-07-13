#!/bin/bash -l
#SBATCH --job-name=stage0_r6d
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/home/woody/iwfa/iwfa131h/slurm_logs/stage0_r6d_%A_%a.out
#SBATCH --export=NONE
#SBATCH --array=0-13%3

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

# Stage 0 (research report Rank 1): fix IK-Abs rotation action + task-space EMA.
# FK targets. Baselines: iter-20 grid v2 (ur5e_f140_ikabs 7.8cm/56%/22%pose,
# kinova_f140_ikabs 9.4cm/46%/12%pose). Throttled %3 (shared cluster).
#  idx 0-4  cfg A rot6d+EMA  ur5e_f140_ikabs   seeds {0,1,2,42,123}
#  idx 5-9  cfg A rot6d+EMA  kinova_f140_ikabs seeds {0,1,2,42,123}
#  idx 10-11 cfg B rot6d only ur5e_f140_ikabs  seeds {7,77}   (attribution)
#  idx 12-13 cfg C EMA only   ur5e_f140_ikabs  seeds {7,77}   (attribution)
export REACH_FK_TARGETS=1

GRID_SEEDS=(0 1 2 42 123)
AB_SEEDS=(7 77)
I=$SLURM_ARRAY_TASK_ID
if [ $I -lt 5 ]; then
    NAME=rot6d_ema_ur5e; TASK=Template-Reach-UR5e-F140-IK-Abs-v0; SEED=${GRID_SEEDS[$I]}
    export REACH_IKABS_ROT6D=1 REACH_TS_EMA=0.2
elif [ $I -lt 10 ]; then
    NAME=rot6d_ema_kinova; TASK=Template-Reach-Kinova-F140-IK-Abs-v0; SEED=${GRID_SEEDS[$((I-5))]}
    export REACH_IKABS_ROT6D=1 REACH_TS_EMA=0.2
elif [ $I -lt 12 ]; then
    NAME=rot6d_only; TASK=Template-Reach-UR5e-F140-IK-Abs-v0; SEED=${AB_SEEDS[$((I-10))]}
    export REACH_IKABS_ROT6D=1
    sleep 20   # stagger vs cfg C (same variant+seed set -> avoid same-second run dirs)
else
    NAME=ema_only; TASK=Template-Reach-UR5e-F140-IK-Abs-v0; SEED=${AB_SEEDS[$((I-12))]}
    export REACH_TS_EMA=0.2
    sleep 45
fi

echo "== stage0[$I] CFG=$NAME Task=$TASK SEED=$SEED rot6d=${REACH_IKABS_ROT6D:-0} ema=${REACH_TS_EMA:-off} $(date) =="
srun --ntasks=1 python -u scripts/skrl/train.py --task="$TASK" --algorithm=PPO --seed=$SEED --headless
echo "[done] CFG=$NAME SEED=$SEED $(date)"
