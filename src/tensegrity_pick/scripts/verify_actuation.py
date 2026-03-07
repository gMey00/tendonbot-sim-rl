"""Verify robot actuation: joint exploration + hardcoded pick-and-place.

Part 1 – Joint Exploration
    Moves each controlled joint individually to its limits while keeping
    the others at default.  All movements are **ramped** to avoid
    physics instability from sudden step changes.

Part 2 – Pick-and-Place Sequence
    Hardcoded sequence proving the robot can physically perform the task:
    settle → close → lift → translate-Y → translate-elbow → open → return

Cube spawn is fixed directly under the grasp-centre rest position so
the gripper fingers wrap symmetrically around the cube.

Usage (with rendering)::

    cd src/tensegrity_pick
    conda run -n env_isaaclab python3 scripts/verify_actuation.py \
        --task Template-Tensegrity-Place-v0 --num_envs 1

Usage (headless)::

    conda run -n env_isaaclab python3 scripts/verify_actuation.py \
        --task Template-Tensegrity-Cube-Place-v0 --num_envs 1 --headless
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hardcoded pick-and-place verification.")
parser.add_argument("--task", type=str, default="Template-Tensegrity-Cube-Place-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import tensegrity_pick.tasks  # noqa: F401

from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import quat_apply


# ---------------------------------------------------------------------------
# Scene geometry constants
# ---------------------------------------------------------------------------

CONVEYOR_SURFACE_HEIGHT_M = 0.80
GRASP_CENTER_LOCAL_Z = 0.1925
DRUM_CENTER_Y = 0.850
DRUM_RIM_HEIGHT_M = 0.880

# Fingertips hang ~23 mm below the grasp centre (gc).
# At default (base_z=-0.25): gc.z ≈ 0.840, fingertips ≈ 0.817, belt = 0.800.
# gc.z varies 1:1 with base_z (prismatic joint), so the minimum base_z
# that keeps fingertips above belt + 1 cm safety margin:
#   0.840 + (min_bz − (−0.25)) − 0.023 > 0.800 + 0.01  →  min_bz > −0.263
MIN_SAFE_BASE_Z = -0.26

# Action indices: [base_y, base_z, elbow, wrist_y, wrist_x, gripper]
IDX_BASE_Y = 0
IDX_BASE_Z = 1
IDX_ELBOW = 2
IDX_WRIST_Y = 3
IDX_WRIST_X = 4
IDX_GRIPPER = 5
ACTION_DIM = 6

# JointPositionActionCfg: target = default + scale * action
ACTION_SCALE = 0.50

# Default joint positions (from place_scene_cfg initial state)
DEFAULT_BASE_Y = 0.0
DEFAULT_BASE_Z = -0.25
DEFAULT_ELBOW = 0.0
DEFAULT_WRIST_Y = 0.0
DEFAULT_WRIST_X = 0.0

OPEN = 1.0
CLOSE = -1.0

# Grasp-centre rest position (measured empirically at default pose).
# The arm hangs ~40 mm in +X from the mount due to URDF geometry.
GC_REST_X = 0.19

# Default action vector (all joints at default, gripper open)
DEFAULT_ACTIONS: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, OPEN)


# ---------------------------------------------------------------------------
# Action helper
# ---------------------------------------------------------------------------

def target_to_action(desired_target: float, default: float) -> float:
    """Convert a desired joint position target to a raw action value.

    JointPositionActionCfg:  target = default + scale * action
    → action = (desired_target − default) / scale
    """
    return (desired_target - default) / ACTION_SCALE


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_state(
    robot: Articulation,
    green: RigidObject,
    env_origin: torch.Tensor,
    step: int,
    label: str,
) -> None:
    """Print key positions for verification."""
    joint_names = robot.data.joint_names
    joint_pos = robot.data.joint_pos[0]

    ee_idx = robot.body_names.index("tool_link_0")
    ee_pos = robot.data.body_pos_w[0, ee_idx]
    ee_quat = robot.data.body_quat_w[0, ee_idx]
    ee_local = ee_pos - env_origin

    gc_offset = ee_pos.new_tensor([0.0, 0.0, GRASP_CENTER_LOCAL_Z])
    gc_world = ee_pos + quat_apply(ee_quat.unsqueeze(0), gc_offset.unsqueeze(0)).squeeze(0)
    gc_local = gc_world - env_origin

    cube_pos = green.data.root_pos_w[0]
    cube_local = cube_pos - env_origin

    gc_to_cube = torch.norm(gc_world - cube_pos).item()

    controlled = [
        "base_y_joint", "base_z_joint", "elbow_joint",
        "wrist_y_joint", "wrist_x_joint", "finger_joint",
    ]
    joint_str = "  ".join(
        f"{n.replace('_joint', '')}={joint_pos[joint_names.index(n)].item():+.4f}"
        for n in controlled
    )

    print(
        f"[{step:5d}] {label:24s} | "
        f"ee=({ee_local[0]:.3f},{ee_local[1]:.3f},{ee_local[2]:.3f}) "
        f"gc=({gc_local[0]:.3f},{gc_local[1]:.3f},{gc_local[2]:.3f}) | "
        f"cube=({cube_local[0]:.3f},{cube_local[1]:.3f},{cube_local[2]:.3f}) | "
        f"gc→cube={gc_to_cube:.3f}m | {joint_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Phase dataclass and runner
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Phase:
    """One phase of a movement sequence.

    ``actions`` holds raw action values (NOT accumulated deltas).
    ``ramp_steps`` controls linear interpolation from previous action
    to this phase's target over the first N steps.
    """

    name: str
    duration_steps: int
    actions: tuple[float, ...]
    ramp_steps: int = 0


def run_phase(
    env: gym.Env,
    robot: Articulation,
    green: RigidObject,
    env_origin: torch.Tensor,
    phase: Phase,
    global_step: int,
    prev_actions: tuple[float, ...] = DEFAULT_ACTIONS,
) -> tuple[int, tuple[float, ...]]:
    """Execute one phase with optional linear ramping.

    Returns (updated global_step, phase.actions for chaining).
    """
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    ramp_label = f", ramp={phase.ramp_steps}" if phase.ramp_steps > 0 else ""
    print(f"\n--- {phase.name} ({phase.duration_steps} steps{ramp_label}) ---", flush=True)

    action = torch.zeros(num_envs, ACTION_DIM, device=device)

    for local_step in range(phase.duration_steps):
        if phase.ramp_steps > 0 and local_step < phase.ramp_steps:
            t = (local_step + 1) / phase.ramp_steps
        else:
            t = 1.0

        for idx in range(ACTION_DIM):
            action[:, idx] = prev_actions[idx] + t * (phase.actions[idx] - prev_actions[idx])

        env.step(action)
        global_step += 1

        is_key_step = (
            local_step == 0
            or local_step == phase.duration_steps - 1
            or local_step % 25 == 0
        )
        if is_key_step:
            print_state(robot, green, env_origin, global_step, phase.name)

    return global_step, phase.actions


def run_phases(
    env: gym.Env,
    robot: Articulation,
    green: RigidObject,
    env_origin: torch.Tensor,
    phases: list[Phase],
    start_step: int = 0,
) -> int:
    """Run a sequence of phases, chaining prev_actions between them."""
    step = start_step
    prev = DEFAULT_ACTIONS
    for phase in phases:
        step, prev = run_phase(env, robot, green, env_origin, phase, step, prev)
    return step


# ---------------------------------------------------------------------------
# Part 1: joint exploration (ramped)
# ---------------------------------------------------------------------------

def run_joint_exploration(
    env: gym.Env,
    robot: Articulation,
    green: RigidObject,
    env_origin: torch.Tensor,
) -> int:
    """Move each joint to its limits individually using ramped movements.

    Returns the total number of sim steps consumed.
    """
    print("=" * 110)
    print("PART 1: JOINT EXPLORATION (ramped)")
    print("  Moving each joint to MIN then MAX while keeping others at default.")
    print("  All movements ramp over 40 steps to avoid physics instability.")
    print("=" * 110)

    ramp = 40
    hold = 60

    # (name, action_index, default, (min_target, max_target))
    joint_tests = [
        ("base_y",  IDX_BASE_Y,  DEFAULT_BASE_Y,  (-0.4, 0.4)),
        ("base_z",  IDX_BASE_Z,  DEFAULT_BASE_Z,  (MIN_SAFE_BASE_Z, -0.05)),
        ("elbow",   IDX_ELBOW,   DEFAULT_ELBOW,   (-1.40, 1.40)),
        ("wrist_y", IDX_WRIST_Y, DEFAULT_WRIST_Y, (-0.70, 0.70)),
        ("wrist_x", IDX_WRIST_X, DEFAULT_WRIST_X, (-0.70, 0.70)),
    ]

    phases: list[Phase] = [Phase("settle", 60, DEFAULT_ACTIONS)]

    for name, idx, default, (lo_target, hi_target) in joint_tests:
        for label, target in [
            (f"{name} → {lo_target:+.2f}", lo_target),
            (f"{name} → {hi_target:+.2f}", hi_target),
        ]:
            target_actions = list(DEFAULT_ACTIONS)
            target_actions[idx] = target_to_action(target, default)
            phases.append(Phase(label, ramp + hold, tuple(target_actions), ramp_steps=ramp))
            phases.append(Phase(f"{name} → default", ramp + hold // 2, DEFAULT_ACTIONS, ramp_steps=ramp))

    # Gripper close / open
    close_actions = list(DEFAULT_ACTIONS)
    close_actions[IDX_GRIPPER] = CLOSE
    phases.append(Phase("gripper CLOSE", 80, tuple(close_actions)))
    phases.append(Phase("gripper OPEN", 80, DEFAULT_ACTIONS))

    step = run_phases(env, robot, green, env_origin, phases)
    print(f"\n  Joint exploration used {step} steps.\n", flush=True)
    return step


# ---------------------------------------------------------------------------
# Part 2: pick-and-place sequence
# ---------------------------------------------------------------------------

def build_pick_and_place_phases() -> list[Phase]:
    """Build the scripted pick-and-place phase list.

    Key geometry (from scene cfg + empirical measurement):
    - At default (base_z=-0.25): gc.z ≈ 0.840, fingertips ≈ 0.817
    - Cube centre at z ≈ 0.825, extent [0.800, 0.850]
    - Fingertips are within the cube at default → grasp from default height.
    - Drum at Y=0.85, rim at Z=0.88.
    - Negative elbow → +Y displacement (toward drum).
    """
    # Slight lowering so gc.z ≈ 0.830 (5 mm above cube centre).
    # Fingertips then at ≈ 0.808 — centre of the cube.
    lower_bz = target_to_action(-0.26, DEFAULT_BASE_Z)

    # Lift: base_z → −0.05 → gc.z ≈ 1.06 (26 cm above belt)
    lift_bz = target_to_action(-0.05, DEFAULT_BASE_Z)

    # Translate Y: base_y → +0.40 (max toward drum side)
    max_by = target_to_action(0.40, DEFAULT_BASE_Y)

    # Elbow: NEGATIVE for +Y reach.
    # Estimated lower-arm length L ≈ 0.72 m.
    # Need Y gap = 0.85 − 0.50 − 0.007 ≈ 0.34 m → sin(θ)=0.34/0.72=0.47
    # θ ≈ 0.50 rad → elbow target = −0.50
    elbow_reach = target_to_action(-0.50, DEFAULT_ELBOW)

    return [
        # Settle at default pose, cube between open fingers
        Phase("1_settle",           60,  DEFAULT_ACTIONS),

        # Slight lowering for better finger contact
        Phase("2_lower",            80,  (0.0,    lower_bz,    0.0,          0.0, 0.0, OPEN),  ramp_steps=40),

        # Close gripper (no ramp — binary action)
        Phase("3_close",           120,  (0.0,    lower_bz,    0.0,          0.0, 0.0, CLOSE)),

        # Let grip forces stabilise
        Phase("4_grip_settle",      80,  (0.0,    lower_bz,    0.0,          0.0, 0.0, CLOSE)),

        # Lift cube well above belt
        Phase("5_lift",            120,  (0.0,    lift_bz,     0.0,          0.0, 0.0, CLOSE), ramp_steps=60),

        # Stabilise at lifted height
        Phase("6_stabilise",        60,  (0.0,    lift_bz,     0.0,          0.0, 0.0, CLOSE)),

        # Translate step 1: slide base toward drum (Y direction)
        Phase("7_translate_y",     150,  (max_by, lift_bz,     0.0,          0.0, 0.0, CLOSE), ramp_steps=80),

        # Translate step 2: swing elbow to bridge remaining Y gap
        Phase("8_translate_elbow", 150,  (max_by, lift_bz,     elbow_reach,  0.0, 0.0, CLOSE), ramp_steps=80),

        # Hold above drum
        Phase("9_over_drum",        80,  (max_by, lift_bz,     elbow_reach,  0.0, 0.0, CLOSE)),

        # Release cube
        Phase("10_open",            80,  (max_by, lift_bz,     elbow_reach,  0.0, 0.0, OPEN)),

        # Watch cube drop into drum
        Phase("11_watch_drop",     150,  (max_by, lift_bz,     elbow_reach,  0.0, 0.0, OPEN)),

        # Return home
        Phase("12_return",         200,  DEFAULT_ACTIONS, ramp_steps=100),

        # Final settle
        Phase("13_final",           80,  DEFAULT_ACTIONS),
    ]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_diagnostics(env: gym.Env, robot: Articulation) -> None:
    """Print action manager structure and joint properties for debugging."""
    unwrapped = env.unwrapped
    action_mgr = unwrapped.action_manager

    print("=" * 110)
    print("DIAGNOSTICS: Action Manager + Joint Properties")
    print("=" * 110)

    print(f"  action_space shape: {env.action_space.shape}", flush=True)

    idx = 0
    for term_name, term in action_mgr._terms.items():
        dim = term.action_dim
        print(f"\n  Action group: '{term_name}'  dim={dim}  indices=[{idx}..{idx + dim - 1}]", flush=True)
        if hasattr(term, "_joint_ids"):
            jids = term._joint_ids
            jnames = [robot.data.joint_names[j] for j in jids] if hasattr(jids, '__iter__') else [str(jids)]
            print(f"    joint_ids  : {list(jids) if hasattr(jids, '__iter__') else jids}", flush=True)
            print(f"    joint_names: {jnames}", flush=True)
        if hasattr(term, "_offset"):
            offset = term._offset[0].tolist() if term._offset.dim() > 1 else term._offset.tolist()
            print(f"    offsets    : {[f'{o:.4f}' for o in offset]}", flush=True)
        idx += dim

    print(f"\n  Robot joint count: {robot.num_joints}", flush=True)
    print(f"  Robot joint names: {robot.data.joint_names}", flush=True)
    joint_pos = robot.data.joint_pos[0]
    joint_limits = robot.data.joint_limits[0] if hasattr(robot.data, 'joint_limits') else None
    for i, name in enumerate(robot.data.joint_names):
        pos_str = f"pos={joint_pos[i].item():.4f}"
        lim_str = ""
        if joint_limits is not None:
            lo = joint_limits[i, 0].item()
            hi = joint_limits[i, 1].item()
            lim_str = f"  limits=[{lo:.4f}, {hi:.4f}]"
        print(f"    [{i:2d}] {name:35s} {pos_str}{lim_str}", flush=True)

    max_ep = getattr(unwrapped, 'max_episode_length', 'N/A')
    print(f"\n  max_episode_length: {max_ep}", flush=True)
    print("=" * 110 + "\n", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # --- Override for deterministic verification ---
    env_cfg.episode_length_s = 120.0

    # Disable arm randomisation
    env_cfg.events.reset_arm.params["position_range"] = (0.0, 0.0)

    # Disable terminations that prevent exploration
    if hasattr(env_cfg.terminations, "joint_effort_saturated"):
        env_cfg.terminations.joint_effort_saturated.params["threshold_ratio"] = 1e6
    env_cfg.terminations.belt_collision.params["max_penetration"] = 100.0
    env_cfg.terminations.joint_vel_diverged.params["max_velocity"] = 1e6

    # Fix cube spawn directly under the grasp-centre rest position.
    # At default pose the gc hangs at x ≈ 0.19 (40 mm offset from mount
    # at x=0.15 due to URDF geometry).  Spawning the cube at x=0.19
    # centres it between the open fingers for a symmetric grasp.
    from tensegrity_pick.tasks.manager_based.cube_place.mdp.rewards import SpawnBox
    fixed_spawn = SpawnBox(
        x_range=(GC_REST_X, GC_REST_X),
        y_range=(0.00, 0.00),
        z_range=(0.83, 0.83),
    )
    env_cfg.events.reset_cubes.params["spawn_box"] = fixed_spawn

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    scene = env.unwrapped.scene
    robot: Articulation = scene["robot"]
    green: RigidObject = scene["green_cube"]
    env_origin = scene.env_origins[0]

    # ==================================================================
    #  DIAGNOSTICS
    # ==================================================================
    print_diagnostics(env, robot)

    # ==================================================================
    #  PART 1:  JOINT EXPLORATION
    # ==================================================================
    run_joint_exploration(env, robot, green, env_origin)

    # ==================================================================
    #  PART 2:  PICK-AND-PLACE SEQUENCE
    # ==================================================================
    print("=" * 110)
    print("PART 2: PICK-AND-PLACE SEQUENCE")
    print("=" * 110)

    # Fresh reset so the cube is back in its spawn position
    env.reset()

    phases = build_pick_and_place_phases()
    step = run_phases(env, robot, green, env_origin, phases)

    # --- Result check ---
    # NOTE: If green_placed fires (cube enters drum), the env resets and
    # the cube is respawned.  In that case the final position here is the
    # respawned cube, not the dropped one.  Check phase 9/10 logs for the
    # actual placement position.
    cube_pos = green.data.root_pos_w[0]
    cube_local = cube_pos - env_origin
    cube_y = cube_local[1].item()
    cube_z = cube_local[2].item()
    dist_to_drum_y = abs(cube_y - DRUM_CENTER_Y)
    on_belt = abs(cube_z - CONVEYOR_SURFACE_HEIGHT_M) < 0.10
    below_rim = cube_z < DRUM_RIM_HEIGHT_M

    print("\n" + "=" * 110, flush=True)
    print("RESULT", flush=True)
    print(f"  Final cube: ({cube_local[0]:.3f}, {cube_local[1]:.3f}, {cube_local[2]:.3f})", flush=True)
    print(f"  Drum centre Y: {DRUM_CENTER_Y:.3f},  rim height: {DRUM_RIM_HEIGHT_M:.3f}", flush=True)
    print(f"  Cube-to-drum Y distance: {dist_to_drum_y:.3f}m", flush=True)
    print(f"  Cube below drum rim: {below_rim}", flush=True)

    if dist_to_drum_y < 0.30 and below_rim:
        print("  ✓ CUBE IN DRUM (Y near centre, Z below rim)", flush=True)
    elif dist_to_drum_y < 0.30 and not below_rim:
        print("  ~ NEAR DRUM Y but cube above rim — may still be falling", flush=True)
    elif on_belt:
        print("  ✗ CUBE STILL ON BELT — grasp failed", flush=True)
    else:
        print(f"  ✗ CUBE NOT NEAR DRUM (Y dist={dist_to_drum_y:.3f}m)", flush=True)

    print("=" * 110, flush=True)
    print(f"Verification complete ({step} steps).  Close window to exit.", flush=True)
    print("=" * 110, flush=True)

    # Keep sim running for visual inspection
    while simulation_app.is_running():
        with torch.inference_mode():
            action = torch.zeros(env.unwrapped.num_envs, ACTION_DIM, device=env.unwrapped.device)
            action[:, IDX_GRIPPER] = OPEN
            env.step(action)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
