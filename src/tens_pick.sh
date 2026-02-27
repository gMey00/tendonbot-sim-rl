#!/bin/bash
# =============================================================================
# tens_pick.sh — Helper commands for tensegrity RL tasks
# =============================================================================
# Usage:  source ../../.config/env_vars.sh && conda activate env_isaaclab
#         Then copy-paste the commands you need from the sections below.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
source ../../.config/env_vars.sh
conda activate env_isaaclab
cd "${PROJECT_PATH}/src/tensegrity_pick"

# ---------------------------------------------------------------------------
# Installation (run once or after code changes)
# ---------------------------------------------------------------------------
python -m pip install -e source/tensegrity_pick

# ---------------------------------------------------------------------------
# List all registered environments
# ---------------------------------------------------------------------------
python scripts/list_envs.py

# ═══════════════════════════════════════════════════════════════════════════════
# TASKS — each section follows the pattern:
#   1. Zero agent   (verify scene loads)
#   2. Random agent  (verify joints move)
#   3. Train         (headless PPO training)
#   4. Play          (evaluate trained policy)
#
# To add a new task:  duplicate any section below and replace the --task ID.
# ═══════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Cube Sorting (Pick)  —  Template-Tensegrity-Pick-v0
# ---------------------------------------------------------------------------
# 8 green + 8 red cubes on a moving conveyor.
# Robot must pick green cubes and place them into the target drum.

python scripts/zero_agent.py   --task=Template-Tensegrity-Pick-v0  --num_envs=10
python scripts/random_agent.py --task=Template-Tensegrity-Pick-v0  --num_envs=10

python scripts/skrl/train.py   --task=Template-Tensegrity-Pick-v0  --headless
python scripts/skrl/train.py   --task=Template-Tensegrity-Pick-v0  --headless --num_envs=2048

python scripts/skrl/play.py    --task=Template-Tensegrity-Pick-v0  --num_envs=10

# ---------------------------------------------------------------------------
# Reach  —  Template-Tensegrity-Reach-v0 / -Play-v0
# ---------------------------------------------------------------------------
# End-effector must reach a random target position and orientation.
# No gripper action; 5-DOF arm only.

python scripts/zero_agent.py   --task=Template-Tensegrity-Reach-v0      --num_envs=10
python scripts/random_agent.py --task=Template-Tensegrity-Reach-v0      --num_envs=10

python scripts/skrl/train.py   --task=Template-Tensegrity-Reach-v0      --headless
python scripts/skrl/train.py   --task=Template-Tensegrity-Reach-v0      --headless --num_envs=2048

python scripts/skrl/play.py    --task=Template-Tensegrity-Reach-Play-v0 --num_envs=10

# ---------------------------------------------------------------------------
# Place  —  Template-Tensegrity-Place-v0 / -Play-v0
# ---------------------------------------------------------------------------
# 1 green + 1 red cube below robot, conveyor inactive.
# Place the green cube into the drum.
# Curriculum: green-only first, then green + red (colour discrimination).

python scripts/zero_agent.py   --task=Template-Tensegrity-Place-v0      --num_envs=10
python scripts/random_agent.py --task=Template-Tensegrity-Place-v0      --num_envs=10

python scripts/skrl/train.py   --task=Template-Tensegrity-Place-v0      --headless
python scripts/skrl/train.py   --task=Template-Tensegrity-Place-v0      --headless --num_envs=2048

python scripts/skrl/play.py    --task=Template-Tensegrity-Place-Play-v0 --num_envs=10

# ---------------------------------------------------------------------------
# Place (Tendon-driven)  —  Template-Tensegrity-Place-Tendon-v0 / -Play-v0
# ---------------------------------------------------------------------------
# Same task as Place but the 3-DOF arm is driven by 5 tendon tensions
# (2 antagonistic for elbow + 3 at 120° for 2-DOF wrist) instead of
# joint-position deltas.  Arm action dim increases from 3 to 5.
# Uses an explicit effort-passthrough actuator (IdealPDActuator k=0, d=0).

python scripts/zero_agent.py   --task=Template-Tensegrity-Place-Tendon-v0      --num_envs=10
python scripts/random_agent.py --task=Template-Tensegrity-Place-Tendon-v0      --num_envs=10

python scripts/skrl/train.py   --task=Template-Tensegrity-Place-Tendon-v0      --headless
python scripts/skrl/train.py   --task=Template-Tensegrity-Place-Tendon-v0      --headless --num_envs=2048

python scripts/skrl/play.py    --task=Template-Tensegrity-Place-Tendon-Play-v0 --num_envs=10

# ---------------------------------------------------------------------------
# Reach (Tendon-driven)  —  Template-Tensegrity-Reach-Tendon-v0 / -Play-v0
# ---------------------------------------------------------------------------
# Same task as Reach but the 3-DOF arm is driven by 5 tendon tensions.
# Base is still PD position-controlled, no gripper needed.
# Direct performance comparison with Template-Tensegrity-Reach-v0.

python scripts/zero_agent.py   --task=Template-Tensegrity-Reach-Tendon-v0      --num_envs=10
python scripts/random_agent.py --task=Template-Tensegrity-Reach-Tendon-v0      --num_envs=10

python scripts/skrl/train.py   --task=Template-Tensegrity-Reach-Tendon-v0      --headless
python scripts/skrl/train.py   --task=Template-Tensegrity-Reach-Tendon-v0      --headless --num_envs=2048

python scripts/skrl/play.py    --task=Template-Tensegrity-Reach-Tendon-Play-v0 --num_envs=10

# ---------------------------------------------------------------------------
# Step Response Test (Klein 2023 methodology)
# ---------------------------------------------------------------------------
# Runs PID-controlled step responses on each joint (wrist X/Y, elbow)
# with thesis-identical gains and tension limits.
# Outputs: plots, metrics table, NRMSE comparison vs Gazebo baseline.

cd /home/robot/Isaac/IsaacLab
./isaaclab.sh -p /home/robot/studentische-arbeiten/src/tensegrity_pick/scripts/step_response_test.py \
    --headless --num-envs 1 --output-dir /home/robot/studentische-arbeiten/src/tensegrity_pick/scripts/step_response_results

# ---------------------------------------------------------------------------
# <New Task>  —  Template-Tensegrity-<NewTask>-v0
# ---------------------------------------------------------------------------
# <Brief description of the task.>
#
# python scripts/zero_agent.py   --task=Template-Tensegrity-<NewTask>-v0      --num_envs=10
# python scripts/random_agent.py --task=Template-Tensegrity-<NewTask>-v0      --num_envs=10
# python scripts/skrl/train.py   --task=Template-Tensegrity-<NewTask>-v0      --headless
# python scripts/skrl/play.py    --task=Template-Tensegrity-<NewTask>-Play-v0 --num_envs=10

