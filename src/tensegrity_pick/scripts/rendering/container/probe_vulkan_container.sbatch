#!/bin/bash -l
#SBATCH --job-name=probe_vk_ctr
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:05:00
#SBATCH --output=/home/woody/iwfa/iwfa131h/slurm_logs/probe_vk_ctr_%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
SIF=/home/vault/iwfa/iwfa131h/Isaac/containers/isaac_render.sif
echo "=== node: $(hostname) ==="
echo "=== apptainer --nv exec: what did --nv bind + does Vulkan see the GPU? ==="
apptainer exec --nv "$SIF" bash -c '
  echo "--- glibc in container ---"; ldd --version | head -1
  echo "--- ICD dir (should be bound by --nv if host had it) ---"; ls -la /usr/share/vulkan/icd.d/ 2>&1
  echo "--- nvidia GLX/vulkan libs bound by --nv? ---"; ldconfig -p 2>/dev/null | grep -iE "GLX_nvidia|libvulkan|EGL_nvidia|glvkspirv" || echo "NONE"
  echo "--- vulkaninfo --summary ---"; vulkaninfo --summary 2>&1 | head -35
'
echo "=== exit: $? ==="
