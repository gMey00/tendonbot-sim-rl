#!/bin/bash -l
#SBATCH --job-name=probe_vk_a40
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/woody/iwfa/iwfa131h/slurm_logs/probe_vk_a40_%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
echo "=== node: $(hostname) ==="; ldd --version | head -1
echo "--- host Vulkan ICD ---"; ls -la /usr/share/vulkan/icd.d/ 2>&1
echo "--- host nvidia GLX/Vulkan libs ---"; ldconfig -p 2>/dev/null | grep -iE "GLX_nvidia|libvulkan|EGL_nvidia|glvkspirv" || echo "NONE in ldconfig"
find /usr/lib64 /usr/lib -name 'libGLX_nvidia.so*' 2>/dev/null || true
echo "--- in-container --nv vulkaninfo ---"
SIF=/home/vault/iwfa/iwfa131h/Isaac/containers/isaac_render.sif
apptainer exec --nv "$SIF" bash -c 'vulkaninfo --summary 2>&1 | head -30' 2>&1
echo "=== done ==="
