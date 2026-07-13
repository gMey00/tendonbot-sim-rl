#!/bin/bash -l
#SBATCH --job-name=verify_kinova
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=/home/woody/iwfa/iwfa131h/slurm_logs/verify_kinova_%j.out
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
VERIFY=scripts/agents/verify_kinova_limits.py

echo "############## 1. PRE-FIX reproduction (clamp spawner disabled) ##############"
srun --ntasks=1 python -u "$VERIFY" --task Template-Reach-Kinova-F140-v0 --num_envs 4 --headless --disable_fix

echo "############## 2. POST-FIX kinova_f140 (joint) ##############"
srun --ntasks=1 python -u "$VERIFY" --task Template-Reach-Kinova-F140-v0 --num_envs 4 --headless

echo "############## 3. POST-FIX kinova_frankenstein (joint) ##############"
srun --ntasks=1 python -u "$VERIFY" --task Template-Reach-Kinova-Frankenstein-v0 --num_envs 4 --headless

echo "############## 4. POST-FIX kinova_f140_osc ##############"
srun --ntasks=1 python -u "$VERIFY" --task Template-Reach-Kinova-F140-OSC-v0 --num_envs 4 --headless

echo "############## 5. POST-FIX kinova_f140_ikabs ##############"
srun --ntasks=1 python -u "$VERIFY" --task Template-Reach-Kinova-F140-IK-Abs-v0 --num_envs 4 --headless

echo "############## 6. POST-FIX kinova_frankenstein_osc ##############"
srun --ntasks=1 python -u "$VERIFY" --task Template-Reach-Kinova-Frankenstein-OSC-v0 --num_envs 4 --headless

echo "[done] $(date)"
