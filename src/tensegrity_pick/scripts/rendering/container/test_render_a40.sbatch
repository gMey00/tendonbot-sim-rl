#!/bin/bash -l
#SBATCH --job-name=test_render_a40
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:25:00
#SBATCH --output=/home/woody/iwfa/iwfa131h/slurm_logs/test_render_a40_%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
source /home/hpc/iwfa/iwfa131h/studentische-arbeiten/.config/env_vars.sh

SIF="${INSTALL_PATH}/containers/isaac_render.sif"
CTR_DIR="${PROJECT_PATH}/src/tensegrity_pick/scripts/rendering/container"
ICD="${CTR_DIR}/nvidia_icd_relative.json"
ENV_PREFIX="${INSTALL_PATH}/conda_envs/${ISAACLAB_ENV_NAME}"

echo "=== node: $(hostname)  $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true

apptainer exec --nv \
  --bind "${ICD}:/etc/vulkan/icd.d/nvidia_icd_relative.json" \
  --bind "${HPCVAULT}" \
  --bind "${HOME}" \
  --bind "${TMPDIR}" \
  --env VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd_relative.json \
  --env ISAAC_ENV_PREFIX="${ENV_PREFIX}" \
  --env PROJECT_PATH="${PROJECT_PATH}" \
  "${SIF}" bash "${CTR_DIR}/render_in_container.sh" \
    --task Template-Reach-Kinova-F140-Play-v0 \
    --headless --video --video_length 300 \
    --num_envs 4 --env_spacing 3.0 \
    --cam_eye 6.5 -4.5 3.5 --cam_lookat 0.75 1.0 1.0 --cam_resolution 1280 720

echo "=== apptainer exit: $? ==="
echo "=== video? ==="
find "${PROJECT_PATH}/src/tensegrity_pick/logs/skrl/reach/kinova_f140" -path '*/videos/play/*.mp4' -newermt '-30 minutes' -ls 2>/dev/null
echo "=== done $(date) ==="
