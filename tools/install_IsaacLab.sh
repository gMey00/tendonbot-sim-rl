# ======== Install MiniConda =========== #
# if [ ! -d "${CONDA_PATH}" ]; then
#     mkdir -p ${CONDA_PATH}
#     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ${CONDA_PATH}/miniconda.sh
#     bash ${CONDA_PATH}/miniconda.sh -b -u -p ${CONDA_PATH}
#     rm ${CONDA_PATH}/miniconda.sh
# fi
# ======== Install Isaac Sim =========== #

if [ ! -d "${INSTALL_PATH}" ]; then
    mkdir -p ${INSTALL_PATH}
fi
if [ ! -d "${ISAACSIM_PATH}" ]; then
    mkdir -p ${ISAACSIM_PATH}
    cd ${INSTALL_PATH}
    wget "https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone%404.5.0-rc.36%2Brelease.19112.f59b3005.gl.linux-x86_64.release.zip"
    unzip "isaac-sim-standalone@4.5.0-rc.36+release.19112.f59b3005.gl.linux-x86_64.release.zip" -d ${ISAACSIM_PATH}
    ${ISAACSIM_PATH}/post_install.sh
    # check if installation is working
    ${ISAACSIM_PATH}/isaac-sim.sh --help
    # checks that python path is set correctly
    ${ISAACSIM_PYTHON_EXE} -c "print('Isaac Sim configuration is now complete.')"
else
    echo "Isaac Sim is already installed at ${ISAACSIM_PATH}."
fi


# ======== Install Isaac Lab =========== #
if [ ! -d "${ISAACSIM_PATH}" ]; then
#if [ ! -d "${ISAACLAB_PATH}" ]; then
    git clone https://github.com/isaac-sim/IsaacLab.git
    ${ISAACLAB_PATH}/isaaclab.sh --help

    # create isaac sim symbolic link     
    cd ${ISAACLAB_PATH}
    ln -s ${ISAACSIM_PATH} _isaac_sim
    # set up the conda environment (optional): 
    # - Default name for conda environment is 'env_isaaclab'
    ./isaaclab.sh --conda  # or "./isaaclab.sh -c"
    conda activate env_isaaclab 
    # Install dependencies for Learning Frameworks:
    # - needed by robomimic which is not available on Windows 
    # - (Only insatll if robomimic is used)
    sudo apt install cmake build-essential
    # Install Learn Frameworks:
    # - possible choices: rl_games, rsl_rl, sb3, skrl, robomimic, none
    # - call for specific install: "./isaaclab.sh --install rl_games"
    # - for 50 series GPUs: ./isaaclab.sh -p -m pip install --upgrade --pre torch torchvision \
    #                              --index-url https://download.pytorch.org/whl/nightly/cu128
    #    (RTX A6000 is not)
    ./isaaclab.sh --install # installs all possible learning environments
    cd ${INSTALL_DIR}
else
    echo "Isaac Lab is already installed at ${ISAACLAB_PATH}."
fi

# activate conda env (if not already done)
conda activate env_isaaclab
