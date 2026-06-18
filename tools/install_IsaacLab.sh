#!/bin/bash -l

# This script installs the NVIDIA Isaac Sim and Isaac Lab on a Linux system.
# Usage: ./install_IsaacLab.sh [learning_framework]
# Where learning_framework is one of: rl_games, rsl_rl, sb3, skrl, robomimic, none
# Default: rl_games
#
# Example: ./install_IsaacLab.sh sb3

# Exit on any error
set -e

source ../.config/env_vars.sh
LEARNING_FRAMEWORK=${1:-"skrl"}  # default learning environment to install 
                                    # - possible choices: rl_games, rsl_rl, sb3, skrl, robomimic

# Validate learning framework choice
valid_frameworks=("rl_games" "rsl_rl" "sb3" "skrl" "robomimic" "none")
if [[ ! " ${valid_frameworks[*]} " =~ " ${LEARNING_FRAMEWORK} " ]]; then
    echo "Error: Invalid learning framework '${LEARNING_FRAMEWORK}'"
    echo "Valid choices: ${valid_frameworks[*]}"
    exit 1
fi

# ======== Install MiniConda =========== #
if [ ! -d "${CONDA_PATH}" ]; then
    mkdir -p ${CONDA_PATH}
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ${CONDA_PATH}/miniconda.sh
    bash ${CONDA_PATH}/miniconda.sh -b -u -p ${CONDA_PATH}
    rm ${CONDA_PATH}/miniconda.sh
fi

# Initialize conda if not already done
if [ -f "${CONDA_PATH}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
    echo "Conda initialization loaded successfully"

    # Register conda in the user's shell rc files (bash + zsh) so that
    # 'conda activate' works in every new terminal without manual setup.
    "${CONDA_PATH}/bin/conda" init bash 2>/dev/null || true
    "${CONDA_PATH}/bin/conda" init zsh  2>/dev/null || true

    # Accept Anaconda Terms of Service for default channels (required for non-interactive installs)
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

    # Fix conda entry point errors by installing pydantic-core in base environment
    conda install -n base pydantic-core -y 2>/dev/null || true
else
    echo "Error: Conda initialization script not found at ${CONDA_PATH}/etc/profile.d/conda.sh"
    echo "Please install conda or update CONDA_PATH in env_vars.sh"
    exit 1
fi

# ======== Install Isaac Sim =========== #

if [ ! -d "${INSTALL_PATH}" ]; then
    mkdir -p ${INSTALL_PATH}
fi
cd ${INSTALL_PATH}
if [ ! -d "${ISAACSIM_PATH}" ]; then
    mkdir -p ${ISAACSIM_PATH}
    # Isaac Sim 5.1.0 standalone Linux build
    wget "https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip"
    unzip "isaac-sim-standalone-5.1.0-linux-x86_64.zip" -d ${ISAACSIM_PATH}
    ${ISAACSIM_PATH}/post_install.sh
    # check if installation is working
    ${ISAACSIM_PATH}/isaac-sim.sh --help
    # checks that python path is set correctly
    ${ISAACSIM_PYTHON_EXE} -c "print('Isaac Sim configuration is now complete.')"
    rm "isaac-sim-standalone-5.1.0-linux-x86_64.zip"
else
    echo "Isaac Sim is already installed at ${ISAACSIM_PATH}."
fi

# ======== Install Isaac Lab =========== #
if [ ! -d "${ISAACLAB_PATH}" ]; then
    git clone https://github.com/isaac-sim/IsaacLab.git
    ${ISAACLAB_PATH}/isaaclab.sh --help

    # create isaac sim symbolic link     
    cd ${ISAACLAB_PATH}
    if [ ! -L "_isaac_sim" ]; then
        ln -s "${ISAACSIM_PATH}" "_isaac_sim"
    fi

    # Install Isaac Lab in a conda environment:
    # - uses python 3.11 (required for Isaac Sim 5.x)
    # - points to Isaac Sim binaries through the above symlink
    # - Default name for conda environment is 'env_isaaclab'

    ENV_NAME="env_isaaclab"
    REQ_PY="3.11"

    # If env exists, verify python minor version (must match Isaac Sim 5.X)
    if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        PY_VER=$(conda run -n "${ENV_NAME}" python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')")
        if [ "${PY_VER}" != "${REQ_PY}" ]; then
            echo "[WARN] ${ENV_NAME} has Python ${PY_VER}, expected ${REQ_PY}. Recreating..."
            conda env remove -n "${ENV_NAME}" -y
        fi
    fi

    ./isaaclab.sh --conda  # or "./isaaclab.sh -c"
    
    # Source conda again after environment creation
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
    conda activate env_isaaclab

    # Ensure pip is available in the newly created environment
    conda install -n "${ENV_NAME}" pip -y
    
    # Install dependencies for Learning Frameworks:
    # - needed by robomimic which is not available on Windows 
    # - (Only install if robomimic is used)
    if [ "${LEARNING_FRAMEWORK}" == "robomimic" ]; then
        sudo apt install cmake build-essential
    fi

    # Install Learning Frameworks:
    # - possible choices: rl_games, rsl_rl, sb3, skrl, robomimic, none
    # - call for specific install: "./isaaclab.sh --install rl_games"
    # - for 50 series GPUs: ./isaaclab.sh -p -m pip install --upgrade --pre torch torchvision \
    #                              --index-url https://download.pytorch.org/whl/nightly/cu128
    #    (RTX A6000 is not)
    if [ "${LEARNING_FRAMEWORK}" != "none" ]; then
        echo "Installing learning framework: ${LEARNING_FRAMEWORK}"
        ./isaaclab.sh --install "${LEARNING_FRAMEWORK}" # installs specific learning framework
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install learning framework ${LEARNING_FRAMEWORK}"
            echo "You may need to install it manually later"
        fi
    else
        echo "Skipping learning framework installation (none selected)"
    fi

    cd "${INSTALL_PATH}"
else
    echo "Isaac Lab is already installed at ${ISAACLAB_PATH}."
fi

# activate conda env (if not already done)
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate env_isaaclab

# ======== Additional Python packages =========== #
# Packages needed for the workspace analysis notebook and tooling.
pip install plotly ipywidgets anywidget 2>/dev/null || {
    echo "[WARN] Failed to install extra notebook packages (plotly, ipywidgets, anywidget)."
    echo "Install them manually: pip install plotly ipywidgets anywidget"
}