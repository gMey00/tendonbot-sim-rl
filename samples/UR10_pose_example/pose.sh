#!/bin/bash
# This script sets up the environment for running a endefector pose reaching sample in Isaac Lab.
# Source the environment variables
source ../../.config/env_vars.sh
# Activate the conda environment
conda activate env_isaaclab

## 1. Set up simualtion environment (USD file) in IsaacSim

# create new sim project
${ISAACLAB_PATH}/isaaclab.sh --sim

## 2. Set up RL project in IsaacLab

# create new template project
${ISAACLAB_PATH}/isaaclab.sh --new

# ------------------------------------------- #
# -- CLI Configuration for the new project -- #
# ------------------------------------------- #
# Task type:
# External (create project outside Isaac Lab repo)

# Project Path: set to/
# ${PROJECT_PATH\}/samples/UR10_pose_example

# Project name:
# Reach

# Isaac Lab workflow:
# Manager-based

# RL library:
# skrl

# RL algorithms for skrl:
# PPO (Proximal Policy Optimization)
# ----------------------------------------- #

# install external project
cd ${PROJECT_PATH}/samples/UR10_pose_example/Reach
python -m pip install -e source/Reach
# Check installation
python scripts/list_envs.py

# --------------------------- #
# Files to modify:
# --------------------------- #
# All files are located in:
# ${PROJECT_PATH}/samples/UR10_pose_example/Reach/source/Reach/Reach/tasks/manager_based/reach/
# 1. Create a new Robot Config file for UR10 with gripper:
#    ./ur_gripper.py
# 2. Modify the environment and MDP managers configuration
#   (Default environment is Cartpole; create a new one for UR10 reaching):
#    ./reach_env_cfg.py
# 3. Register new custom environment configurations in __init__.py
#    ./__init__.py
# 4. Define custom reward functions used in reach_env_cfg.py in
#   ./mdp/rewards.py
# 5. (Optional) Configure/Modify Hyperparameters for training in
#   ./agents/skrl_ppo_cfg.yaml
# --------------------------- #


## 3. Test the environment

# Run a zero agent (no actions, just to test the environment)
# Does Robot Scene load? Does the environment step without errors?
python scripts/zero_agent.py --task Template-Reach-v0 --num_envs=10

# Run a random agent (random actions)
# Do all joints move randomly without errors?
python scripts/random_agent.py --task Template-Reach-v0 --num_envs=10

## 4, Train and run the agent

# Train the agent
python scripts/skrl/train.py --task Template-Reach-v0 --headless # --num_envs=100 # Default num_envs=4096

# Run the trained agent/policy