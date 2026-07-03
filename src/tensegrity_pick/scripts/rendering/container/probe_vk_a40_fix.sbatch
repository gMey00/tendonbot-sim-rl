#!/bin/bash -l
#SBATCH --job-name=vk_a40_fix
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/woody/iwfa/iwfa131h/slurm_logs/vk_a40_fix_%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
SIF=/home/vault/iwfa/iwfa131h/Isaac/containers/isaac_render.sif
echo "=== node: $(hostname) ==="
echo "--- host ICD json content ---"; cat /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
echo; echo "--- in-container --nv + bound ICD + correct VK_ICD_FILENAMES ---"
apptainer exec --nv \
  --bind /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d \
  --env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json \
  "$SIF" bash -c '
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    echo "--- nvidia libs apptainer --nv injected ---"; ls -la /.singularity.d/libs 2>/dev/null | grep -iE "GLX_nvidia|EGL_nvidia|glvkspirv|libcuda" || echo "none in /.singularity.d/libs"
    echo "--- vulkaninfo --summary ---"; vulkaninfo --summary 2>&1 | head -40
  '
echo "=== exit: $? ==="
