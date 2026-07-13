#!/bin/bash -l
#SBATCH --job-name=osc_stab
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/home/woody/iwfa/iwfa131h/slurm_logs/osc_stab_%A_%a.out
#SBATCH --export=NONE
#SBATCH --array=0-14

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

# Iteration-17 step 1b: stabilise OSC on the non-redundant UR arm (ur5e_f140_osc,
# the cleanest bimodal case). 3 candidate configs x 5 seeds. Baseline control
# (variable_kp, kp=100) already exists from the iteration-17 multi-seed grid.
SEEDS=(0 1 2 42 123)
CFG=$(( SLURM_ARRAY_TASK_ID / 5 ))
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % 5 ))]}

case $CFG in
  0)  # fixed impedance: remove the policy's control over task stiffness
      export REACH_OSC_IMPEDANCE=fixed
      NAME=fixed ;;
  1)  # lower stiffness ceiling, still variable_kp (same action dim as baseline)
      export REACH_OSC_STIFFNESS=50
      export REACH_OSC_STIFFNESS_MIN=20
      export REACH_OSC_STIFFNESS_MAX=100
      NAME=lowstiff ;;
  2)  # fixed impedance + lower stiffness
      export REACH_OSC_IMPEDANCE=fixed
      export REACH_OSC_STIFFNESS=50
      NAME=fixed_lowstiff ;;
esac

echo "============================================================"
echo " osc_stab[$SLURM_ARRAY_TASK_ID]  CFG=$NAME  SEED=$SEED  $(date)"
echo " REACH_OSC_IMPEDANCE=${REACH_OSC_IMPEDANCE:-variable_kp}  STIFFNESS=${REACH_OSC_STIFFNESS:-100} [${REACH_OSC_STIFFNESS_MIN:-50},${REACH_OSC_STIFFNESS_MAX:-200}]  ZETA=${REACH_OSC_DAMPING_RATIO:-1.0}"
echo " Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-?}"
echo "============================================================"

srun --ntasks=1 python -u scripts/skrl/train.py --task=Template-Reach-UR5e-F140-OSC-v0 --algorithm=PPO --seed=$SEED --headless

echo "[done] CFG=$NAME SEED=$SEED $(date)"
