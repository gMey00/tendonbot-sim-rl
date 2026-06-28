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

    # ── Command: FK-sampled reachable EE poses ────────────────────────────
    env.commands.ee_pose.body_name = F140_EE_BODY
    env.commands.ee_pose.joint_names = controlled_joints
    env.commands.ee_pose.joint_range_margin = joint_range_margin
    # Reject FK-sampled configurations that are self-colliding so every target
    # is reachable by a *collision-free* posture (the gripper finger cluster is
    # excluded from the check — see FKSampledPoseCommandCfg).  This keeps each
    # robot's full own-workspace minus only genuinely-infeasible targets.
    env.commands.ee_pose.self_collision_filter = True
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
    # Widen tanh gradient: these arms start ~0.4-0.5 m from target.
    env.rewards.end_effector_position_tracking_fine_grained.params["std"] = 0.5
    env.rewards.joint_vel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=controlled_joints)

    # ── Orientation reward rebalance ──────────────────────────────────────
    # The stock reach reward under-weights orientation (-0.1) versus position
    # (-0.2 plus a +0.1 fine-grained tanh "homing" bonus orientation lacks), so a
    # redundant arm facing arbitrary FK-sampled SO(3) targets rationally ignores
    # orientation and plateaus at ~30° error.  Raise the geodesic-orientation
    # weight and add a matching fine-grained tanh term so orientation gets a dense
    # gradient near the goal too.  Applies to every variant equally (it does not
    # remove any robot capability), so the per-robot action-space comparison stays
    # fair while full 6-DOF tracking becomes learnable.
    env.rewards.end_effector_orientation_tracking.weight = -0.5
    env.rewards.end_effector_orientation_tracking_fine_grained = RewTerm(
        func=mdp.orientation_command_error_tanh,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[F140_EE_BODY]),
            "std": 0.5,
            "command_name": "ee_pose",
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
