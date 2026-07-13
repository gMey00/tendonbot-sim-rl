# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared reach-task setup for the floor-standing F140 comparison arms.

The six comparison variants (UR10 / UR5e / Kinova, each in plain-F140 and
"Frankenstein" + tensegrity-wrist form) all share the same reach-task wiring:

* mounted upright at the workspace-analysis pose ``(0.75, 1.0, 0.75)``;
* EE reference body = the Robotiq ``robotiq_base_link``;
* FK-sampled reachable pose targets over the controlled joints;
* EMA joint-position-to-limits action;
* gravity + self-collision disabled and gripper joints stiffened for stable
  reach training (no grasping needed).

This mirrors the per-robot ``ur10e`` / ``kinova`` reach configs but factors out
the boilerplate so the six variants stay consistent.
"""

from __future__ import annotations

import math
import os

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from tensegrity_pick.tasks.manager_based.reach import mdp
from tensegrity_pick.tasks.manager_based.reach.reach_env_cfg import ReachEnvCfg

# All six variants are mounted upright at the workspace-analysis position.
F140_MOUNT_POSITION = (0.75, 1.0, 0.75)
F140_MOUNT_ROTATION = (1.0, 0.0, 0.0, 0.0)  # identity -> normal "upward" mount
# All variants share the same Robotiq 2F-140 gripper base as the EE reference.
F140_EE_BODY = "robotiq_base_link"


def _stabilize_gripper(robot: ArticulationCfg) -> None:
    """Stiffen all Robotiq 2F-140 joints so they hold still during reach.

    The default gripper joints have near-zero damping; left free they drift and
    can diverge the PhysX solver (NaNs across all joints).  Reach needs no
    grasping, so we clamp them rigidly.
    """
    robot.actuators["gripper_drive"] = ImplicitActuatorCfg(
        joint_names_expr=["finger_joint"],
        effort_limit_sim=1000.0, velocity_limit_sim=1.0,
        stiffness=100.0, damping=100.0, armature=10.0,
    )
    robot.actuators["gripper_finger"] = ImplicitActuatorCfg(
        joint_names_expr=["left_inner_finger_joint", "right_inner_finger_joint"],
        effort_limit_sim=1000.0, velocity_limit_sim=1.0,
        stiffness=100.0, damping=100.0, armature=10.0,
    )
    robot.actuators["gripper_passive"] = ImplicitActuatorCfg(
        joint_names_expr=[
            "left_inner_finger_pad_joint", "right_inner_finger_pad_joint",
            "left_outer_finger_joint", "right_outer_finger_joint",
            "right_outer_knuckle_joint",
        ],
        effort_limit_sim=1000.0, velocity_limit_sim=1.0,
        stiffness=100.0, damping=100.0, armature=10.0,
    )


def configure_f140_reach(
    env: ReachEnvCfg,
    robot_cfg: ArticulationCfg,
    controlled_joints: list[str],
    *,
    joint_range_margin: float,
    clamp_fallback_range: float,
    clamp_max_range: float | None,
) -> None:
    """Apply the shared F140 reach configuration to *env* in-place.

    Parameters
    ----------
    robot_cfg : the variant's ``ArticulationCfg``.
    controlled_joints : arm (+ tensegrity wrist) joints to control and to
        randomise for FK target sampling.
    joint_range_margin : fraction trimmed from each joint range when sampling
        FK targets (larger -> smaller, easier workspace).
    clamp_fallback_range : finite range substituted for infinite joint limits.
    clamp_max_range : if set, ALSO clamp any finite joint whose range exceeds
        this (used for UR's ±2π arm joints; leave ``None`` for Kinova).
    """
    # ── Robot asset: mount upright at the workspace-analysis pose ──────────
    env.scene.robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    env.scene.robot.init_state.pos = F140_MOUNT_POSITION
    env.scene.robot.init_state.rot = F140_MOUNT_ROTATION
    env.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    # Disable gravity: reach is a kinematic task; gravity only adds solver work.
    env.scene.robot.spawn.rigid_props.disable_gravity = True
    _stabilize_gripper(env.scene.robot)

    # ── Command: NVIDIA-style uniform box of reachable EE poses ───────────
    # Diagnostic switch away from FK-sampled full-SO(3) targets to a fixed box in
    # the robot ROOT frame (Isaac Lab's stock UniformPoseCommand logic; here as a
    # mode of the project command so all success / reach-time metrics are kept).
    #
    # The box is sized to fit inside EVERY comparison arm's reach envelope so all
    # six can reach every target: UR5e (0.85 m) is the tightest, Kinova Gen3
    # (0.90 m) next, UR10 (1.30 m) has slack. The farthest box corner is
    # sqrt(0.50² + 0.20² + 0.50²) ≈ 0.735 m ≈ 86 % of UR5e / 82 % of Kinova reach,
    # so no target sits at a full-stretch singularity; nearest corner ≈ 0.39 m is
    # clear of the shoulder deadzone.
    #
    # Orientation is fixed "gripper pointing straight down" with free yaw, exactly
    # like the Franka reference (pitch=π). The Robotiq robotiq_base_link local +z
    # is the approach axis (see cube_place/gripper_cfg: "local Z projects downward
    # toward the cube"); quat_from_euler_xyz(roll=0, pitch=π, yaw=·) maps that +z
    # onto the base-frame −z ⇒ tool down, spun freely about the vertical.
    env.commands.ee_pose.body_name = F140_EE_BODY
    # joint_names is unused by the box sampler but kept: downstream IK/OSC configs
    # read it (self.commands.ee_pose.joint_names) to size their action terms.
    env.commands.ee_pose.joint_names = controlled_joints
    env.commands.ee_pose.joint_range_margin = joint_range_margin
    env.commands.ee_pose.uniform_ranges = mdp.FKSampledPoseCommandCfg.Ranges(
        pos_x=(0.30, 0.50),
        pos_y=(-0.20, 0.20),
        pos_z=(0.25, 0.50),
        roll=(0.0, 0.0),
        pitch=(math.pi, math.pi),
        yaw=(-math.pi, math.pi),
    )
    # ── OPT-IN (iteration-18 diagnostic): FK-sampled reachable full-pose targets ──
    # Off by default (env var unset -> the fixed-orientation box baseline stays
    # bit-exact). When REACH_FK_TARGETS is set, drop the box and sample targets by
    # forward-kinematics of random joint configs (the pre-iteration-14 mode): the
    # target orientation is then *reachable by construction*. Tests whether that
    # removes the OSC-on-UR instability and unlocks orientation tracking on the
    # 6-DOF arms (both plausibly caused by the box's geometrically-unreachable
    # gripper-down orientation, P2). NOTE: per-arm FK targets are NOT a shared
    # cross-arm distribution — position errors are no longer comparable across arms.
    if os.environ.get("REACH_FK_TARGETS", "") not in ("", "0"):
        env.commands.ee_pose.uniform_ranges = None
        # Reject FK samples that are self-colliding or fold through the floor:
        # both produce physically-unreachable targets that poison training
        # (iteration 19: Kinova plateaued at 24-35 cm on the unfiltered
        # full-envelope distribution).  Floor clearances are in the WORLD frame
        # (ground plane at z=0): links keep 5 cm; the EE keeps 30 cm so the
        # ~24 cm Robotiq 2F-140 hanging below robotiq_base_link stays clear.
        env.commands.ee_pose.self_collision_filter = True
        env.commands.ee_pose.fk_min_link_height_w = 0.05
        env.commands.ee_pose.fk_min_ee_height_w = 0.30

    # ── OPT-IN probe knob: episode length (task-space hardening, iteration 20) ──
    # Joint control solves FK full-pose targets within the 6 s episode, but the
    # task-space controllers (greedy DLS / OSC paths) may need more time for the
    # large reorientations.  Unset -> 6 s (baseline unchanged).
    _episode_s = os.environ.get("REACH_EPISODE_S", "")
    if _episode_s not in ("", "0"):
        env.episode_length_s = float(_episode_s)

    # Disable debug-vis marker point-instancers during (headless) training:
    # at 4096 envs they trigger a FabricManager prototype mismatch and a
    # carb.tasking mutex-recursion assertion crash.  They are GUI-only anyway;
    # the PLAY configs re-enable them for visualisation.
    env.commands.ee_pose.debug_vis = False

    # ── Action: EMA joint-position-to-limits over the controlled joints ───
    env.actions.arm_action = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=controlled_joints,
        alpha=0.2,
    )

    # ── OPT-IN: tensegrity-wrist matched-pair A/B knobs (joint action space) ──
    # For the "does the wrist help" study on a Frankenstein arm (iteration-18).
    # Both off by default -> no change. NOTE: only wired for the joint action
    # space (the IK/OSC configs override arm_action after this and read
    # commands.ee_pose.joint_names to size themselves; do not combine these knobs
    # with those spaces without extending the wiring).
    #   REACH_FK_SAMPLE_BASE_ONLY : FK-sample only the base (non-wrist) joints, so
    #     targets are reachable with the wrist at neutral -> base-reachable set.
    #   REACH_LOCK_WRIST          : drop the wrist from the joint action (it then
    #     holds at its PD neutral) -> a base-equivalent controller on the same arm.
    # The 4 cells (base/frank targets × wrist active/locked) bracket the wrist's
    # redundancy benefit (base targets) vs capability gain (frank targets).
    _wrist_joints = ("wrist_x_joint", "wrist_y_joint")
    _base_only_joints = [j for j in controlled_joints if j not in _wrist_joints]
    if os.environ.get("REACH_FK_SAMPLE_BASE_ONLY", "") not in ("", "0"):
        env.commands.ee_pose.joint_names = _base_only_joints
    if os.environ.get("REACH_LOCK_WRIST", "") not in ("", "0"):
        env.actions.arm_action.joint_names = _base_only_joints

    # ── Rewards: track the gripper-base EE body ───────────────────────────
    for term in (
        env.rewards.end_effector_position_tracking,
        env.rewards.end_effector_position_tracking_fine_grained,
        env.rewards.end_effector_orientation_tracking,
        env.rewards.position_reached,
        env.rewards.orientation_reached,
        env.rewards.pose_reached,
    ):
        term.params["asset_cfg"].body_names = [F140_EE_BODY]
    env.rewards.joint_vel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=controlled_joints)

    # ── Reward balance: keep the Isaac Lab reference values ────────────────
    # Left exactly at the stock reach reward (position -0.2, position fine-grained
    # tanh 0.1 @ std 0.1, orientation -0.1, no fine-grained orientation term).
    #
    # An earlier rebalance (orientation weight -0.5, an added fine-grained
    # orientation tanh, and a widened position std of 0.5) was introduced to force
    # orientation learning against the OLD FK-sampled full-SO(3) targets.  With the
    # uniform target box's easy fixed "gripper-down" orientation that instead
    # over-weighted orientation: the 6-DOF UR arms sacrificed position to satisfy
    # the (now-easy) orientation and their position error roughly doubled/tripled,
    # while the 7-DOF Kinova (redundant) improved.  The reference balance — which
    # Isaac Lab validated on the Franka reach, itself a uniform-box task — keeps
    # position the dominant objective, so it is restored here unchanged.

    # ── OPT-IN: position-gated orientation refinement (redundant arms only) ──
    # Iteration 16 experiment.  Disabled by default (env var unset -> weight 0)
    # so the 24-variant position-only baseline stays bit-exact and citable.  When
    # REACH_ORIENT_REFINE_WEIGHT > 0, ADD a reward that tracks orientation only
    # once position is (nearly) achieved (smooth position gate).  It cannot cause
    # the P1 position regression (moving off-target lowers both the gate and the
    # position reward); a redundant arm can instead spend its spare DOF (the
    # Frankenstein wrist / Kinova 7th joint) on orientation at fixed position.
    # Only added for redundant arms (>6 controlled joints): the rigid 6-DOF
    # UR5e-/UR10-F140 arms have no spare DOF and are left untouched (per P2/§3.3).
    _refine_w = float(os.environ.get("REACH_ORIENT_REFINE_WEIGHT", "0") or "0")
    if _refine_w > 0.0 and len(controlled_joints) > 6:
        env.rewards.orientation_refine = RewTerm(
            func=mdp.orientation_refine_gated,
            weight=_refine_w,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[F140_EE_BODY]),
                "command_name": "ee_pose",
                "orientation_std": float(os.environ.get("REACH_ORIENT_REFINE_ORISTD", "0.3")),
                "position_std": float(os.environ.get("REACH_ORIENT_REFINE_POSSTD", "0.1")),
            },
        )

    # ── Observations / terminations: controlled joints only ───────────────
    # Encode joint positions as (sin, cos) to remove the ±π wrap discontinuity
    # that the Kinova's continuous joints otherwise present to the policy (harmless
    # for the bounded UR joints, so applied uniformly for a fair comparison).
    env.observations.policy.joint_pos = ObsTerm(
        func=mdp.joint_pos_sin_cos,
        noise=Unoise(n_min=-0.01, n_max=0.01),
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=controlled_joints)},
    )
    env.observations.policy.joint_vel.params = {
        "asset_cfg": SceneEntityCfg("robot", joint_names=controlled_joints)
    }
    env.terminations.joint_vel_diverged.params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=controlled_joints
    )

    # ── Events: offset reset + clamp wide/infinite joint limits ───────────
    env.events.reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.125, 0.125),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=controlled_joints),
        },
    )
    clamp_params: dict = {
        "asset_cfg": SceneEntityCfg("robot"),
        "fallback_range": clamp_fallback_range,
    }
    if clamp_max_range is not None:
        clamp_params["max_range"] = clamp_max_range
    env.events.clamp_joint_limits = EventTerm(
        func=mdp.clamp_infinite_joint_limits,
        mode="reset",
        params=clamp_params,
    )
    # ── Kinova: clamp the continuous-joint USD limits *before* the PhysX bake ──
    # The Kinova Gen3 continuous joints (joint_1/3/5/7) are authored with limits
    # outside PhysX's supported [-2π, 2π] range; PhysX throws `setLimitParams()`
    # 8-9× when it bakes the articulation at sim.reset().  The reset-mode clamp
    # above only runs *after* the bake, so it silences the limits for stepping
    # but not that init-time error, and a mode="prestartup" event is rejected by
    # Isaac Lab while replicate_physics=True (which reach needs).  So we wrap the
    # spawner: it clamps env_0's joint-limit USD attributes before the clone /
    # bake, using the same default±range/2 rule as the reset term (baked limits,
    # and training, unchanged).  Only Kinova authors out-of-range limits; the UR
    # arms stay on the reset-mode clamp alone.
    usd_path = getattr(env.scene.robot.spawn, "usd_path", "") or ""
    if "Kinova" in usd_path:
        # Module-level spawner + registry (Hydra-safe: the config round-trip
        # serialises spawn.func to "module:qualname", which a closure cannot satisfy).
        mdp.register_joint_limit_clamp(
            usd_path=usd_path,
            joint_defaults=dict(env.scene.robot.init_state.joint_pos or {}),
            fallback_range=clamp_fallback_range,
            max_range=clamp_max_range,
        )
        env.scene.robot.spawn.func = mdp.spawn_usd_with_clamped_joint_limits
