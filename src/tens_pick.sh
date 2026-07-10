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
# Cube Sort  —  Template-Tensegrity-Cube-Sort-v0
# ---------------------------------------------------------------------------
# 8 green + 8 red cubes on a moving conveyor.
# Robot must pick green cubes and place them into the target drum.

python scripts/zero_agent.py   --task=Template-Tensegrity-Cube-Sort-v0  --num_envs=10
python scripts/random_agent.py --task=Template-Tensegrity-Cube-Sort-v0  --num_envs=10

python scripts/skrl/train.py   --task=Template-Tensegrity-Cube-Sort-v0  --headless
python scripts/skrl/train.py   --task=Template-Tensegrity-Cube-Sort-v0  --headless --num_envs=2048

python scripts/skrl/play.py    --task=Template-Tensegrity-Cube-Sort-v0  --num_envs=10

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

# --- Reach: multi-seed training sweep — 3 variants × 5 seeds ---------------
# NOTE: the single-run lines above use the OLD task IDs `Template-Tensegrity-Reach-*`.
#       The CURRENTLY REGISTERED IDs are `Template-Reach-Tensegrity-*`
#       (see scripts/list_envs.py). The sweep below uses the registered IDs.
# Each run lands in its own logs/skrl/reach/<variant>/<timestamp>_ppo_torch/ dir.
# Seeds hold everything-else-fixed; reports training-seed variance (Henderson et al. 2018).
REACH_VARIANTS=(
  "Template-Reach-Tensegrity-v0"                        # PD
  "Template-Reach-Tensegrity-Tendon-v0"                 # simulated tendon
  "Template-Reach-Tensegrity-Physical-Tendon-v0"        # physical tendon (body-force, direct tension control)
  "Template-Reach-Tensegrity-Physical-Hierarchical-v0"  # physical tendon (inner PID->tension loop)
)
for task in "${REACH_VARIANTS[@]}"; do
  for s in 0 1 2 3 4; do
    echo "[seed-sweep] training ${task} with seed=${s}"
    python scripts/skrl/train.py --task "${task}" --headless --seed "${s}"
  done
done

# --- Reach: baseline / checkpoint evaluation (see scripts/skrl/evaluate_reach.py) -
REACH_PLAY_VARIANTS=(
  "Template-Reach-Tensegrity-Play-v0"                        # PD
  "Template-Reach-Tensegrity-Tendon-Play-v0"                 # tendon
  "Template-Reach-Tensegrity-Physical-Tendon-Play-v0"        # physical tendon (direct)
  "Template-Reach-Tensegrity-Physical-Hierarchical-Play-v0"  # physical tendon (hierarchical)
)
# zero + random: every variant × every seed
for task in "${REACH_PLAY_VARIANTS[@]}"; do
  for s in 0 1 2 3 4; do
    python scripts/skrl/evaluate_reach.py --agent zero   --task "$task" --seed "$s" --num_episodes 10 --headless
    python scripts/skrl/evaluate_reach.py --agent random --task "$task" --seed "$s" --num_episodes 10 --headless
  done
done
# heuristic: PD variant only × 5 seeds (single task-level reference)
for s in 0 1 2 3 4; do
  python scripts/skrl/evaluate_reach.py --agent heuristic \
      --task Template-Reach-Tensegrity-Play-v0 --seed "$s" --num_episodes 10 --headless
done
# PPO checkpoints: per (variant, seed) — point --checkpoint at the run dir
# python scripts/skrl/evaluate_reach.py --agent checkpoint --task <PLAY_ID> --seed <s> \
#     --checkpoint logs/skrl/reach/<variant>/<run_dir> --num_episodes 10 --headless

# ---------------------------------------------------------------------------
# Cube Place  —  Template-Tensegrity-Cube-Place-v0 / -Play-v0
# ---------------------------------------------------------------------------
# 1 green + 1 red cube below robot, conveyor inactive.
# Place the green cube into the drum.
# Curriculum: green-only first, then green + red (colour discrimination).

python scripts/zero_agent.py   --task=Template-Tensegrity-Cube-Place-v0      --num_envs=10
python scripts/random_agent.py --task=Template-Tensegrity-Cube-Place-v0      --num_envs=10

python scripts/skrl/train.py   --task=Template-Tensegrity-Cube-Place-v0      --headless
python scripts/skrl/train.py   --task=Template-Tensegrity-Cube-Place-v0      --headless --num_envs=2048

python scripts/skrl/play.py    --task=Template-Tensegrity-Cube-Place-Play-v0 --num_envs=10

# ---------------------------------------------------------------------------
# Cube Place (Tendon-driven)  —  Template-Tensegrity-Cube-Place-Tendon-v0 / -Play-v0
# ---------------------------------------------------------------------------
# Same task as Cube Place but the 3-DOF arm is driven by 5 tendon tensions
# (2 antagonistic for elbow + 3 at 120° for 2-DOF wrist) instead of
# joint-position deltas.  Arm action dim increases from 3 to 5.
# Uses an explicit effort-passthrough actuator (IdealPDActuator k=0, d=0).

python scripts/zero_agent.py   --task=Template-Tensegrity-Cube-Place-Tendon-v0      --num_envs=10
python scripts/random_agent.py --task=Template-Tensegrity-Cube-Place-Tendon-v0      --num_envs=10

python scripts/skrl/train.py   --task=Template-Tensegrity-Cube-Place-Tendon-v0      --headless
python scripts/skrl/train.py   --task=Template-Tensegrity-Cube-Place-Tendon-v0      --headless --num_envs=2048

python scripts/skrl/play.py    --task=Template-Tensegrity-Cube-Place-Tendon-Play-v0 --num_envs=10

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
# Shirt Place (PD)  —  Template-Tensegrity-Shirt-Place-v0 / -Play-v0
# ---------------------------------------------------------------------------
# Place a T-shirt (rigid proxy cube) into a drum.
# 6-phase reward pipeline: reach → grasp → lift → transport → release → success.
# Currently uses rigid proxy; cloth simulation integration pending.

python scripts/zero_agent.py   --task=Template-Tensegrity-Shirt-Place-v0      --num_envs=10
python scripts/random_agent.py --task=Template-Tensegrity-Shirt-Place-v0      --num_envs=10

python scripts/skrl/train.py   --task=Template-Tensegrity-Shirt-Place-v0      --headless
python scripts/skrl/train.py   --task=Template-Tensegrity-Shirt-Place-v0      --headless --num_envs=2048

python scripts/skrl/play.py    --task=Template-Tensegrity-Shirt-Place-Play-v0 --num_envs=10

# ---------------------------------------------------------------------------
# Shirt Place (Physical Tendon)  —  Template-Tensegrity-Shirt-Place-Physical-Tendon-v0 / -Play-v0
# ---------------------------------------------------------------------------
# Same task as Shirt Place but the 3-DOF arm is driven by 5 tendon tensions
# (body-force elbow tendons + Jacobian-transpose wrist tendons).
# Action dim: 5 tendons + 1 gripper + 2 base = 8.

python scripts/zero_agent.py   --task=Template-Tensegrity-Shirt-Place-Physical-Tendon-v0      --num_envs=10
python scripts/random_agent.py --task=Template-Tensegrity-Shirt-Place-Physical-Tendon-v0      --num_envs=10

python scripts/skrl/train.py   --task=Template-Tensegrity-Shirt-Place-Physical-Tendon-v0      --headless
python scripts/skrl/train.py   --task=Template-Tensegrity-Shirt-Place-Physical-Tendon-v0      --headless --num_envs=2048

python scripts/skrl/play.py    --task=Template-Tensegrity-Shirt-Place-Physical-Tendon-Play-v0 --num_envs=10

# ---------------------------------------------------------------------------
# Shirt Sort (template)  —  Template-Tensegrity-Shirt-Sort-v0
# ---------------------------------------------------------------------------
# Sort T-shirts on a moving conveyor (cloth simulation).
# TODO: Not yet implemented — cloth deformable objects pending.

# python scripts/zero_agent.py   --task=Template-Tensegrity-Shirt-Sort-v0  --num_envs=10
# python scripts/skrl/train.py   --task=Template-Tensegrity-Shirt-Sort-v0  --headless

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

