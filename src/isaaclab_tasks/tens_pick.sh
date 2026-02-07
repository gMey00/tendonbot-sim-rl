#!/bin/bash
# This script sets up the environment for running the tensegrity pick task in Isaac Lab.

# Source the environment variables
source ../../.config/env_vars.sh

# Activate the conda environment
conda activate env_isaaclab

## 1. Set up simulation environment (USD file) in IsaacSim

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

# Project Path: set to
# ${PROJECT_PATH}/src/isaaclab_tasks
# Project name:
# tensegrity_pick

# Isaac Lab workflow:
# Manager-based

# RL library:
# skrl

# RL algorithms for skrl:
# PPO (Proximal Policy Optimization)
# ----------------------------------------- #

# install external project
cd ${PROJECT_PATH}/src/isaaclab_tasks/tensegrity_pick
python -m pip install -e source/tensegrity_pick

# Check installation
python scripts/list_envs.py

# --------------------------- #
# Files to modify:
# --------------------------- #
# All files are located in:
# ${PROJECT_PATH}/src/isaaclab_tasks/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/tensegrity_pick/
#
# 1. Configure the robot and environment in:
#    ./tensegrity_pick_env_cfg.py
#
# 2. Register environment configurations in:
#    ./__init__.py
#
# 3. Define reward functions in:
#    ./mdp/cube_sorting_rewards.py
#
# 4. Define MDP managers (observations, actions, rewards, terminations) in:
#    ./mdp/cube_sorting_mdp.py (or similar MDP configuration files)
#
# 5. (Optional) Configure/Modify Hyperparameters for training in:
#    ./agents/skrl_ppo_cfg.yaml
# --------------------------- #


## 3. Test the environment

# Run a zero agent (no actions, just to test the environment)
# Does Robot Scene load? Does the environment step without errors?
${ISAACLAB_PATH}/isaaclab.sh -p scripts/zero_agent.py --task=Template-Tensegrity-Pick-v0 --num_envs=10

# Run a random agent (random actions)
# Do all joints move randomly without errors?
${ISAACLAB_PATH}/isaaclab.sh -p scripts/random_agent.py --task=Template-Tensegrity-Pick-v0 --num_envs=10

## 4. Train and run the agent

# Train the agent
${ISAACLAB_PATH}/isaaclab.sh -p scripts/skrl/train.py --task=Template-Tensegrity-Pick-v0 --headless # --num_envs=100 # Default num_envs=4096

# Run the trained agent/policy
${ISAACLAB_PATH}/isaaclab.sh -p scripts/skrl/play.py --task=Template-Tensegrity-Pick-v0 --num_envs=10
