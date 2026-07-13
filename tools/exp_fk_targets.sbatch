#!/bin/bash -l
#SBATCH --job-name=fk_targets
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/home/woody/iwfa/iwfa131h/slurm_logs/fk_targets_%A_%a.out
#SBATCH --export=NONE
#SBATCH --array=0-9

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

# Iteration-18 diagnostic: do FK-sampled (reachable-by-construction) targets fix
# (a) OSC-on-UR instability and (b) the 0%-orientation-on-UR limitation?
export REACH_FK_TARGETS=1

SEEDS=(0 1 2 42 123)
# idx 0-4: ur5e_f140_osc (OSC stability under reachable targets)
# idx 5-9: ur5e_f140 joint (orientation trackability, isolated from OSC)
if [ "$SLURM_ARRAY_TASK_ID" -lt 5 ]; then
    TASK=Template-Reach-UR5e-F140-OSC-v0
    SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
else
    TASK=Template-Reach-UR5e-F140-v0
    SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID - 5 ))]}
fi

echo "============================================================"
echo " fk_targets[$SLURM_ARRAY_TASK_ID]  Task=$TASK  SEED=$SEED  REACH_FK_TARGETS=1  $(date)"
echo " Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-?}"
echo "============================================================"

srun --ntasks=1 python -u scripts/skrl/train.py --task="$TASK" --algorithm=PPO --seed=$SEED --headless

echo "[done] Task=$TASK SEED=$SEED $(date)"
