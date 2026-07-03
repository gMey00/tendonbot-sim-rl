#!/bin/bash -l
#SBATCH --job-name=probe_vulkan
#SBATCH --partition=rtxpro6k
#SBATCH --gres=gpu:rtxpro6k:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:05:00
#SBATCH --output=/home/woody/iwfa/iwfa131h/slurm_logs/probe_vulkan_%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

echo "=== node: $(hostname)  date: $(date) ==="
echo "=== glibc on compute node ==="; ldd --version | head -1
echo; echo "=== NVIDIA Vulkan ICD present? ==="
ls -la /usr/share/vulkan/icd.d/ 2>&1 || echo "NO /usr/share/vulkan/icd.d"
echo "--- nvidia_icd.json anywhere ---"; find /usr /etc -name 'nvidia_icd.json' 2>/dev/null || true
echo; echo "=== NVIDIA Vulkan/GLX userspace libs present? ==="
for lib in libGLX_nvidia.so libEGL_nvidia.so libnvidia-glvkspirv.so libvulkan.so libnvidia-glcore.so; do
  echo "--- $lib ---"; find /usr/lib64 /usr/lib /lib64 -name "${lib}*" 2>/dev/null || true
  ldconfig -p 2>/dev/null | grep -i "$lib" || true
done
echo; echo "=== ldconfig vulkan/glx summary ==="; ldconfig -p 2>/dev/null | grep -iE 'vulkan|GLX_nvidia|EGL_nvidia' || echo "none in ldconfig"
echo; echo "=== nvidia-smi ==="; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
echo "=== done ==="
