#!/bin/bash -l
#SBATCH --job-name=probes20
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/home/woody/iwfa/iwfa131h/slurm_logs/probes20_%A_%a.out
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

# Iteration-20 probes (all on FK targets with the fixed sampler). Probe seeds
# {7, 77} are disjoint from the grid's {0,1,2,42,123}, so run-dir names can
# never collide with grid runs of the same variant.
export REACH_FK_TARGETS=1

SEEDS=(7 77)
CFG=$(( SLURM_ARRAY_TASK_ID / 2 ))
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % 2 ))]}

case $CFG in
  0)  # wrist-under-OSC: damped-compliant wrist in the OSC Jacobian (UR 8-DOF)
      NAME=osc_wrist_ur5e
      TASK=Template-Reach-UR5e-Frankenstein-OSC-v0
      export REACH_OSC_INCLUDE_WRIST=1 ;;
  1)  # wrist-under-OSC on the 9-DOF Kinova-Frankenstein
      NAME=osc_wrist_kinova
      TASK=Template-Reach-Kinova-Frankenstein-OSC-v0
      export REACH_OSC_INCLUDE_WRIST=1 ;;
  2)  # task-space probe: IK-Abs with 12 s episodes (time for large reorientation)
      NAME=ikabs_12s
      TASK=Template-Reach-UR5e-F140-IK-Abs-v0
      export REACH_EPISODE_S=12 ;;
  3)  # task-space probe: IK-Rel with 12 s episodes
      NAME=ikrel_12s
      TASK=Template-Reach-UR5e-F140-IK-Rel-v0
      export REACH_EPISODE_S=12 ;;
  4)  # task-space probe: IK-Rel with doubled per-step delta scale
      NAME=ikrel_scale1
      TASK=Template-Reach-UR5e-F140-IK-Rel-v0
      export REACH_IK_REL_SCALE=1.0 ;;
esac

echo "== probes20[$SLURM_ARRAY_TASK_ID] CFG=$NAME Task=$TASK SEED=$SEED $(date) =="
srun --ntasks=1 python -u scripts/skrl/train.py --task="$TASK" --algorithm=PPO --seed=$SEED --headless
echo "[done] CFG=$NAME SEED=$SEED $(date)"
