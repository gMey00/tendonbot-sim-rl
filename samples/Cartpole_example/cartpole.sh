#!/bin/bash
# This script sets up the environment for running the Cartpole sample in Isaac Lab.
# Source the environment variables
source ../../.config/env_vars.sh
# Activate the conda environment
conda activate env_isaaclab

# create new template project
${ISAACLAB_PATH}/isaaclab.sh --new

# Task type:
# *External* to create the project outside the Isaac Lab repo

# Project Path: set to/
# ${PROJECT_PATH\}/Samples/Cartpole

# Project name:
# Cartpole

# Isaac Lab workflow:
# Manager-based

# RL library:
# skrl

# RL algorithms for skrl:
# PPO (Proximal Policy Optimization)

# install external project
cd ${PROJECT_PATH}/samples/Cartpole_example/Cartpole
python -m pip install -e source/Cartpole
# Check installation
python scripts/list_envs.py

# Train the agent
python scripts/skrl/train.py --task=Template-Cartpole-v0 # --num_envs=1000 # Default num_envs=4096

# Run the trained agent/policy
python scripts/skrl/play.py --task=Template-Cartpole-v0 --num_envs=10

