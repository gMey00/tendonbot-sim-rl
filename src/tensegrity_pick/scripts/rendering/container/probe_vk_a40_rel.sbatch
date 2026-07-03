#!/bin/bash -l
#SBATCH --job-name=vk_a40_rel
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/woody/iwfa/iwfa131h/slurm_logs/vk_a40_rel_%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
SIF=/home/vault/iwfa/iwfa131h/Isaac/containers/isaac_render.sif
ICD=/home/hpc/iwfa/iwfa131h/studentische-arbeiten/src/tensegrity_pick/scripts/rendering/container/nvidia_icd_relative.json
echo "=== node: $(hostname) ==="
apptainer exec --nv \
  --bind "${ICD}:/etc/vulkan/icd.d/nvidia_icd_relative.json" \
  --env VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd_relative.json \
  "$SIF" bash -c '
    echo "--- all injected nvidia libs ---"; ls /.singularity.d/libs 2>/dev/null | grep -iE "nvidia|GLX|EGL|optix" | sort
    echo "--- vulkaninfo --summary ---"; vulkaninfo --summary 2>&1 | grep -iE "deviceName|driverName|apiVersion|deviceType|GPU id|ERROR|Cannot" | head -30
  '
echo "=== exit: $? ==="
