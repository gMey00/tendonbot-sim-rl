#!/bin/bash -l
#SBATCH --job-name=smoke_r6d
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=/home/woody/iwfa/iwfa131h/slurm_logs/smoke_r6d_%j.out
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

echo "##### SMOKE A: IK-Abs rot6d + EMA (expect action shape 9) #####"
REACH_FK_TARGETS=1 REACH_IKABS_ROT6D=1 REACH_TS_EMA=0.2 timeout 360 srun --ntasks=1 \
  python -u scripts/agents/zero_agent.py --task Template-Reach-UR5e-F140-IK-Abs-v0 --num_envs 256 --headless || true

echo "##### SMOKE B: OSC + EMA (expect action shape 13, EMAOSC class) #####"
REACH_FK_TARGETS=1 REACH_TS_EMA=0.2 timeout 360 srun --ntasks=1 \
  python -u scripts/agents/zero_agent.py --task Template-Reach-UR10-F140-OSC-v0 --num_envs 256 --headless || true

echo "[done] $(date)"
