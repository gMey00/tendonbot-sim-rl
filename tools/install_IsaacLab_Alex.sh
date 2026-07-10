#!/bin/bash -l
# ============================================================================
#  install_IsaacLab_Alex.sh  —  Isaac Sim 5.1.0 + Isaac Lab on the Alex cluster
# ============================================================================
#  Run ONCE on an Alex *login node* (alex.nhr.fau.de). Login nodes have no GPU
#  but full internet + the shared filesystems, which is exactly what we need
#  for downloading and pip-installing. Do NOT run inside a compute job.
#
#      cd tools && bash ./install_IsaacLab_Alex.sh [learning_framework]
#      (env_vars.sh is sourced automatically from ../.config below)
#
#  learning_framework: rl_games | rsl_rl | sb3 | skrl | robomimic | none
#  default: skrl
#
#  Design goals for Alex:
#    * Everything heavy (Isaac Sim ~15 GB, Isaac Lab, conda) goes on $VAULT,
#      because $HOME quota is small and snapshot-doubled. Code stays on $HOME.
#    * No sudo (not available on HPC) and no system package installs.
#    * Minimal on-disk overhead: download archives are removed after unzip,
#      pip caches are redirected to $VAULT and the download zip is streamed.
#    * Headless only: the login node has no display, so we never launch the GUI.
#    * Python 3.11 (required by Isaac Sim 5.x) via a dedicated conda env.
# ============================================================================

set -euo pipefail

# ZSH_VERSION is unset in bash; pre-define it so that Isaac Sim helper scripts
# sourced during conda activation don't crash under 'set -u'.
export ZSH_VERSION="${ZSH_VERSION:-}"

# --------------------------------------------------------------- locate repo -
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.config/env_vars.sh"

LEARNING_FRAMEWORK="${1:-skrl}"
valid_frameworks=("rl_games" "rsl_rl" "sb3" "skrl" "robomimic" "none")
if [[ ! " ${valid_frameworks[*]} " =~ " ${LEARNING_FRAMEWORK} " ]]; then
    echo "Error: invalid learning framework '${LEARNING_FRAMEWORK}'"
    echo "Valid choices: ${valid_frameworks[*]}"
    exit 1
fi

echo "============================================================"
echo " Installing onto \$VAULT : ${INSTALL_PATH}"
echo " Repo (\$HPC)            : ${PROJECT_PATH}"
echo " Conda env              : ${ISAACLAB_ENV_NAME} (Python 3.11)"
echo " Learning framework     : ${LEARNING_FRAMEWORK}"
echo "============================================================"

# Keep pip/conda caches on $VAULT so they don't blow the small $HOME quota.
export PIP_CACHE_DIR="${INSTALL_PATH}/.cache/pip"
export CONDA_PKGS_DIRS="${INSTALL_PATH}/.cache/conda_pkgs"
mkdir -p "${INSTALL_PATH}" "${PIP_CACHE_DIR}" "${CONDA_PKGS_DIRS}"

# Alex login nodes run AlmaLinux (currently mid-migration 8 -> 9). The Isaac Sim
# binary build is GLIBC-compatible with both. Nothing to configure here, but if
# you hit a GLIBC error after the OS upgrade, re-run this installer to refresh.

# ======== Conda ============================================================ #
# Prefer the cluster-provided conda (module load python) to avoid a second
# Miniconda copy; fall back to a private Miniconda on $VAULT only if absent.
if module load python 2>/dev/null && command -v conda >/dev/null 2>&1; then
    echo "[conda] Using cluster 'python' module conda: $(command -v conda)"
    eval "$(conda shell.bash hook)"
    # Cluster conda is read-only; redirect envs and pkgs cache to $VAULT.
    # 'module load python' overwrites CONDA_PKGS_DIRS to the cluster read-only
    # path, so we re-export our writable path here and also persist it via condarc.
    conda config --add envs_dirs "${INSTALL_PATH}/conda_envs" 2>/dev/null || true
    conda config --prepend pkgs_dirs "${INSTALL_PATH}/.cache/conda_pkgs" 2>/dev/null || true
    export CONDA_ENVS_PATH="${INSTALL_PATH}/conda_envs"
    export CONDA_PKGS_DIRS="${INSTALL_PATH}/.cache/conda_pkgs"
    mkdir -p "${CONDA_ENVS_PATH}" "${CONDA_PKGS_DIRS}"
else
    echo "[conda] Cluster module unavailable; installing private Miniconda on \$VAULT"
    if [ ! -d "${CONDA_PATH}" ]; then
        mkdir -p "${CONDA_PATH}"
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
             -O "${CONDA_PATH}/miniconda.sh"
        bash "${CONDA_PATH}/miniconda.sh" -b -u -p "${CONDA_PATH}"
        rm -f "${CONDA_PATH}/miniconda.sh"
    fi
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    2>/dev/null || true
fi

# Isaac Sim 5.x native libraries require GLIBCXX_3.4.30+; load GCC 15 to provide it.
# This is needed both for post_install.sh and at training runtime.
module load gcc/15.2.0 2>/dev/null || echo "[WARN] gcc/15.2.0 module not found; GLIBC errors may occur"

# ======== Isaac Sim 5.1.0 ================================================== #
if [ ! -d "${ISAACSIM_PATH}" ]; then
    echo "[IsaacSim] Downloading + extracting to ${ISAACSIM_PATH}"
    mkdir -p "${ISAACSIM_PATH}"
    cd "${INSTALL_PATH}"
    ZIP="isaac-sim-standalone-5.1.0-linux-x86_64.zip"
    wget -q "https://download.isaacsim.omniverse.nvidia.com/${ZIP}"
    unzip -q "${ZIP}" -d "${ISAACSIM_PATH}"
    rm -f "${ZIP}"                       # remove archive immediately (saves ~15 GB)
    # post_install.sh sets up desktop icons — non-critical for headless HPC use.
    # It requires GLIBCXX_3.4.30 (provided by gcc/15.2.0 loaded above).
    "${ISAACSIM_PATH}/post_install.sh" || echo "[WARN] post_install.sh failed (non-fatal; conda path unaffected)"
    # Verify embedded python works (optional smoke-test for standalone mode).
    "${ISAACSIM_PYTHON_EXE}" -c "print('Isaac Sim 5.1.0 standalone ready.')" || \
        echo "[WARN] python.sh smoke-test failed; standalone mode may not work but conda path is fine"
else
    echo "[IsaacSim] Already present at ${ISAACSIM_PATH}"
fi

# ======== Isaac Lab ======================================================== #
if [ ! -d "${ISAACLAB_PATH}" ]; then
    echo "[IsaacLab] Cloning to ${ISAACLAB_PATH}"
    git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_PATH}"
fi

cd "${ISAACLAB_PATH}"

# Symlink Isaac Sim into Isaac Lab (how Isaac Lab finds the binaries).
if [ ! -L "_isaac_sim" ]; then
    ln -s "${ISAACSIM_PATH}" "_isaac_sim"
fi

# Create / validate the conda env (Python 3.11 is mandatory for Isaac Sim 5.x).
REQ_PY="3.11"
if conda env list | awk '{print $1}' | grep -qx "${ISAACLAB_ENV_NAME}"; then
    PY_VER="$(conda run -n "${ISAACLAB_ENV_NAME}" python -c \
        'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
    if [ "${PY_VER}" != "${REQ_PY}" ]; then
        echo "[IsaacLab] ${ISAACLAB_ENV_NAME} has Python ${PY_VER}, expected ${REQ_PY}. Recreating."
        conda env remove -n "${ISAACLAB_ENV_NAME}" -y
    fi
fi

if ! conda env list | awk '{print $1}' | grep -qx "${ISAACLAB_ENV_NAME}"; then
    echo "[IsaacLab] Creating conda env '${ISAACLAB_ENV_NAME}'"
    # Temporarily relax strict mode: isaaclab.sh sources third-party scripts
    # (e.g. setup_conda_env.sh) that reference unbound variables like $ZSH_VERSION.
    set +eu
    ./isaaclab.sh --conda "${ISAACLAB_ENV_NAME}"
    set -eu
fi

# conda activate runs env activation scripts (some reference $ZSH_VERSION etc.)
# Use set +eu to let those scripts run without strict-mode interference.
set +eu
conda activate "${ISAACLAB_ENV_NAME}"
set -eu

# Install Isaac Lab + the chosen learning framework.
# cmake is required by several framework dependencies; load it so isaaclab.sh
# can detect it and skip the sudo apt-get fallback (unavailable on HPC).
module load cmake 2>/dev/null || echo "[WARN] could not load cmake module"

if [ "${LEARNING_FRAMEWORK}" != "none" ]; then
    echo "[IsaacLab] Installing framework: ${LEARNING_FRAMEWORK}"
    set +eu
    ./isaaclab.sh --install "${LEARNING_FRAMEWORK}"
    set -eu
else
    echo "[IsaacLab] Installing core only (no learning framework)"
    set +eu
    ./isaaclab.sh --install none
    set -eu
fi

# ======== HPC offline config ============================================== #
# Compute nodes have no direct internet. Patch user.config.json so Isaac Sim
# uses local asset paths and skips extension-registry network checks at startup.
USER_CFG="${ISAACSIM_PATH}/kit/data/Kit/Isaac-Sim/5.1/user.config.json"
if [ -f "${USER_CFG}" ]; then
    echo "[hpc-cfg] Patching ${USER_CFG} for offline compute nodes"
    python3 - << PYEOF
import json
cfg_path = "${USER_CFG}"
proj_res = "${PROJECT_PATH}/res"
with open(cfg_path) as f:
    cfg = json.load(f)
# Point asset_root to local res/ so omniclient never creates an S3 HTTP provider.
cfg["persistent"]["isaac"]["asset_root"]["default"] = proj_res
cfg["persistent"]["isaac"]["asset_root"]["cloud"]   = proj_res
cfg["persistent"]["isaac"]["asset_root"]["nvidia"]  = proj_res
# Disable extension-registry network checks (Azure CDN / CloudFront).
cfg.setdefault("exts", {}).setdefault("omni.kit.registry.nucleus", {})["registries"] = []
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=4)
print("  asset_root -> local res/   |  extension registry -> disabled")
PYEOF
else
    echo "[WARN] user.config.json not found at ${USER_CFG}; skipping HPC patch"
fi

# ======== Project extension + extras ====================================== #
echo "[project] Installing tensegrity_pick extension (editable)"
cd "${PROJECT_PATH}/src/tensegrity_pick"
python -m pip install -e source/tensegrity_pick

# Notebook / monitoring tooling used by the helper scripts.
# psutil powers the monitor's Performance tab (CPU/RAM/process metrics); the GPU
# panel shells out to nvidia-smi, which is already present on the compute nodes.
python -m pip install plotly ipywidgets anywidget tbparse rich plotext psutil 2>/dev/null || {
    echo "[WARN] some extra packages failed to install; install manually if needed:"
    echo "       pip install plotly ipywidgets anywidget tbparse rich plotext psutil"
}
# btop is a system monitor (not a pip package); install via conda-forge.
conda install -n "${ISAACLAB_ENV_NAME}" -c conda-forge btop -y 2>/dev/null || \
    echo "[WARN] btop not installed; watch_alex.sh will fall back to 'top'"

# ======== Smoke test ======================================================= #
echo "[verify] Listing registered environments (headless)"
python scripts/list_envs.py || echo "[WARN] list_envs failed; check the install log above."

echo "============================================================"
echo " Done. Activate later with:"
echo "     source ${PROJECT_PATH}/.config/env_vars.sh"
echo "     conda activate ${ISAACLAB_ENV_NAME}"
echo "============================================================"