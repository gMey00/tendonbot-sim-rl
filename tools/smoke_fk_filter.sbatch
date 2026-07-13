#!/bin/bash -l
#SBATCH --job-name=smoke_fkf
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:35:00
#SBATCH --output=/home/woody/iwfa/iwfa131h/slurm_logs/smoke_fkf_%j.out
#SBATCH --export=NONE

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

# Smoke 1: fixed FK sampler on Kinova (filter calibration + rejection stats +
# half-range) — zero agent, 512 envs, bounded by timeout (prints happen at reset).
echo "########## SMOKE 1: kinova_f140 FK + filter (zero agent) ##########"
REACH_FK_TARGETS=1 timeout 420 srun --ntasks=1 python -u scripts/agents/zero_agent.py \
  --task Template-Reach-Kinova-F140-v0 --num_envs 512 --headless || true

echo "########## SMOKE 2: ur5e_frankenstein OSC + wrist included (zero agent) ##########"
REACH_FK_TARGETS=1 REACH_OSC_INCLUDE_WRIST=1 timeout 420 srun --ntasks=1 python -u scripts/agents/zero_agent.py \
  --task Template-Reach-UR5e-Frankenstein-OSC-v0 --num_envs 512 --headless || true

echo "[done] $(date)"
