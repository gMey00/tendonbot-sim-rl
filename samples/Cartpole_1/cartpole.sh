#!/bin/bash
# This script sets up the environment for running the Cartpole sample in Isaac Lab.
# Source the environment variables
source ../.config/env_vars.sh
# Activate the conda environment
conda activate env_isaaclab

# create new template project
cd $ISAACLAB_PATH
./isaaclab.sh --new