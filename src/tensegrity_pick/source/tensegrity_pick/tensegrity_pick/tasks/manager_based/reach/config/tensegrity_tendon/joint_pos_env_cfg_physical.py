# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reach task environment configs for the *physical* tensegrity tendon model.

The physical model replaces the single ``elbow_joint`` with a four-bar
antiparallelogram linkage driven by body-force tendons.  The elbow action is
applied as cable forces at the physical attachment points rather than via the
constant Jacobian-transpose mapping used by the elbow_approx variant.

NOTE: The 5-DOF physical USD must be assembled in Robot Assembler before
these configs can be instantiated.
"""

import math

import torch
from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

import isaaclab.sim as sim_utils
from tensegrity_pick.robots import HierarchicalPhysicalTendonActionCfg, PhysicalTendonEffortActionCfg
from tensegrity_pick.robots.tendon_robot_cfg import TENS_5DOF_GRIPPER_PHYSICAL_TENDON_CFG
from tensegrity_pick.robots.tendon_robot_cfg import TARGET_LINK_NAME_5DOF_PHYSICAL as TARGET_LINK_NAME
from tensegrity_pick.robots.tendon_robot_cfg import CONTROLLED_JOINT_NAMES_5DOF_PHYSICAL as CONTROLLED_JOINT_NAMES
from tensegrity_pick.robots.tensegrity_robot_cfg import TENS_5DOF_GRIPPER_CFG as PD_ROBOT_CFG
from tensegrity_pick.robots.tensegrity_robot_cfg import TARGET_LINK_NAME_5DOF as PD_TARGET_LINK_NAME
from tensegrity_pick.robots.tensegrity_robot_cfg import CONTROLLED_JOINT_NAMES_5DOF as PD_CONTROLLED_JOINT_NAMES
from tensegrity_pick.tasks.manager_based.shared.proj_base_scene_cfg import TENSEGRITY_MOUNT_HEIGHT_M
from tensegrity_pick.tasks.manager_based.reach import mdp
from tensegrity_pick.tasks.manager_based.reach.reach_env_cfg import ReachEnvCfg

# ── Antiparallelogram 4-bar linkage constants (Klein 2023, §3.2) ──────────
_L_E = 0.150   # rod length [m]
_K_E = 0.060   # frame / coupler pivot spacing [m]
_THETA_0 = math.asin(_K_E / _L_E)  # equilibrium rod angle ≈ 23.6°


def antiparallelogram_joint_coupling(positions: torch.Tensor) -> torch.Tensor:
    """Enforce antiparallelogram closure on FK-sampled joint positions.

    Joint order (matching ``CONTROLLED_JOINT_NAMES_5DOF_PHYSICAL``):
        [0] base_y, [1] base_z, [2] rod_left, [3] rod_right,
        [4] coupler_left, [5] wrist_y, [6] wrist_x

    The 4-bar has 1 DOF — ``rod_left`` is the free parameter.
    From ``build_tensegrity_arm_usd.py`` (§ Joint Limits):

        coupler_left = rod_right   (antiparallelogram symmetry)
        coupler_right = rod_left   (enforced by PhysX loop-closure)

    The closure equation (USD coords, zero = equilibrium):

        θ = rod_left_usd + θ₀
        φ = θ + 2·atan2(−k_e·cos θ, l_e − k_e·sin θ)
        rod_right_usd = coupler_left_usd = φ + θ₀
    """
    rod_usd = positions[:, 2]

    theta = rod_usd + _THETA_0
    phi = theta + 2.0 * torch.atan2(
        -_K_E * torch.cos(theta),
        _L_E - _K_E * torch.sin(theta),
    )
    coupled_usd = phi + _THETA_0
    positions[:, 3] = coupled_usd  # rod_right  from closure
    positions[:, 4] = coupled_usd  # coupler_left = rod_right

    return positions


# Joints whose raw angles are physically meaningful policy observations.
# The three linkage joints (rod_left, rod_right, coupler_left) are excluded:
# the lower-arm rotation is measured from the forearm body instead (see
# mdp.lower_arm_angle) — it is the sum of two linkage angles, not any single
# joint, and stays valid even under loop-closure solver drift.
_OBS_JOINT_NAMES = ["base_y_joint", "base_z_joint", "wrist_y_joint", "wrist_x_joint"]


@configclass
class PhysicalTendonReachActionsCfg:
    base_action = mdp.JointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=["base_y_joint", "base_z_joint"],
    )

    arm_tendon = PhysicalTendonEffortActionCfg(
        asset_name="robot",
        joint_names=["wrist_y_joint", "wrist_x_joint"],
    )


@configclass
class HierarchicalTendonReachActionsCfg:
    """Hierarchical variant: base position targets + joint-space set-points
    tracked by the validated inner PID→tension loop."""

    base_action = mdp.JointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=["base_y_joint", "base_z_joint"],
    )

    arm_tendon = HierarchicalPhysicalTendonActionCfg(
        asset_name="robot",
        joint_names=["wrist_y_joint", "wrist_x_joint"],
    )


@configclass
class TensegrityReachPhysicalTendonEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # The elbow tendons act as large external body forces on the linkage.
        # Re-applying them every solver iteration markedly improves constraint
        # accuracy (loop closure, joint limits) under such forces.
        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.min_velocity_iteration_count = 2
        # 120 Hz physics: the loop-closure joint (maximal-coordinate constraint)
        # drifts by centimetres under a sustained 480 N cable tension at 60 Hz;
        # halving dt keeps the four-bar rigid (matches the validated
        # step-response setup, which also ran at 120 Hz).  The policy rate
        # stays at 30 Hz via decimation.
        self.sim.dt = 1.0 / 120.0
        self.decimation = 4
        self.sim.render_interval = self.decimation

        # Physical four-bar linkage arm ("awake" USD bake: closure joint
        # enabled, per-body sleep/stabilization thresholds zeroed — a sleeping
        # articulation silently ignores the cable body forces).
        self.scene.robot = TENS_5DOF_GRIPPER_PHYSICAL_TENDON_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.15, 0.0, TENSEGRITY_MOUNT_HEIGHT_M)
            ),
        )

        # ── FK reference robot (PD-approximated variant) ────────────────
        # Hidden articulation used *only* for FK target sampling.
        # Placed at the same position as the physical robot so that the
        # FK-derived EE poses are in the correct frame.
        # Collisions are disabled so it doesn't interfere with the scene.
        self.scene.fk_reference_robot = PD_ROBOT_CFG.replace(
            prim_path="{ENV_REGEX_NS}/FKRefRobot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.15, 0.0, TENSEGRITY_MOUNT_HEIGHT_M)
            ),
            spawn=PD_ROBOT_CFG.spawn.replace(
                visible=False,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            ),
        )
        # Stabilize gripper on FK reference robot to prevent PhysX NaN corruption
        _gripper_stabilize = ImplicitActuatorCfg(
            joint_names_expr=[".*"],  # placeholder, overridden per-group below
            effort_limit_sim=1000.0,
            velocity_limit_sim=1.0,
            stiffness=100.0,
            damping=100.0,
            armature=10.0,
        )
        self.scene.fk_reference_robot.actuators["gripper_drive"] = _gripper_stabilize.replace(
            joint_names_expr=["finger_joint"],
        )
        self.scene.fk_reference_robot.actuators["gripper_finger"] = _gripper_stabilize.replace(
            joint_names_expr=["left_inner_finger_joint", "right_inner_finger_joint"],
        )
        self.scene.fk_reference_robot.actuators["gripper_passive"] = _gripper_stabilize.replace(
            joint_names_expr=[
                "left_inner_finger_pad_joint",
                "right_inner_finger_pad_joint",
                "left_outer_finger_joint",
                "right_outer_finger_joint",
                "right_outer_knuckle_joint",
            ],
        )
        # Stabilize ALL gripper joints for reach task (no grasping needed)
        self.scene.robot.actuators["gripper_drive"] = ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=1.0,
            stiffness=100.0,
            damping=100.0,
            armature=10.0,
        )
        self.scene.robot.actuators["gripper_finger"] = ImplicitActuatorCfg(
            joint_names_expr=["left_inner_finger_joint", "right_inner_finger_joint"],
            effort_limit_sim=1000.0,
            velocity_limit_sim=1.0,
            stiffness=100.0,
            damping=100.0,
            armature=10.0,
        )
        self.scene.robot.actuators["gripper_passive"] = ImplicitActuatorCfg(
            joint_names_expr=[
                "left_inner_finger_pad_joint",
                "right_inner_finger_pad_joint",
                "left_outer_finger_joint",
                "right_outer_finger_joint",
                "right_outer_knuckle_joint",
            ],
            effort_limit_sim=1000.0,
            velocity_limit_sim=1.0,
            stiffness=100.0,
            damping=100.0,
            armature=10.0,
        )
        # override commands — use PD reference robot for FK target sampling
        self.commands.ee_pose.body_name = TARGET_LINK_NAME
        self.commands.ee_pose.joint_names = CONTROLLED_JOINT_NAMES
        # Sample targets from the *achievable* workspace (margin 0.10 → elbow
        # ≈ ±56°, wrist ≈ ±40°), not the full geometric range: boundary poses
        # cannot be *held* by the cable drives (wrist under-actuated near ±50°,
        # elbow stop-grinding at ±70°), and unreachable targets put the inner
        # controller / policy into permanent-saturation limit cycles.
        self.commands.ee_pose.joint_range_margin = 0.10
        # FK targets are sampled from the PD-approximated robot (stable, no 4-bar)
        self.commands.ee_pose.fk_reference_asset_name = "fk_reference_robot"
        self.commands.ee_pose.fk_reference_body_name = PD_TARGET_LINK_NAME
        self.commands.ee_pose.fk_reference_joint_names = PD_CONTROLLED_JOINT_NAMES
        # No coupling fn — the PD robot uses a single elbow_joint
        self.commands.ee_pose.joint_coupling_fn = None
        # re-declare actions for the physical tendon arm
        self.actions = PhysicalTendonReachActionsCfg()
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.position_reached.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.orientation_reached.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        self.rewards.pose_reached.params["asset_cfg"].body_names = [TARGET_LINK_NAME]
        # Restrict joint velocity penalty to controlled joints
        self.rewards.joint_vel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)
        # ── Observations ─────────────────────────────────────────────────
        # Raw joint angles only where they are physically meaningful (base +
        # wrist).  The lower-arm rotation is measured from the forearm body
        # (correct measurement point — not a single linkage joint).
        self.observations.policy.joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=_OBS_JOINT_NAMES)}
        self.observations.policy.joint_vel.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=_OBS_JOINT_NAMES)}
        self.observations.policy.lower_arm_angle = ObsTerm(
            func=mdp.lower_arm_angle,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={
                "root_cfg": SceneEntityCfg("robot", body_names=["root_link"]),
                "forearm_cfg": SceneEntityCfg("robot", body_names=["forearm_link"]),
            },
        )
        self.observations.policy.lower_arm_ang_vel = ObsTerm(
            func=mdp.lower_arm_ang_vel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={
                "root_cfg": SceneEntityCfg("robot", body_names=["root_link"]),
                "forearm_cfg": SceneEntityCfg("robot", body_names=["forearm_link"]),
            },
        )
        # Actuator (cable) state — evaluation report §9 open issue #1
        self.observations.policy.cable_lengths = ObsTerm(
            func=mdp.tendon_cable_lengths,
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        self.observations.policy.cable_length_rates = ObsTerm(
            func=mdp.tendon_cable_length_rates,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # Applied (lag-filtered, stop-adjusted) elbow tensions — actuator state
        # the policy cannot infer from its own last action (eval report §9 #1).
        self.observations.policy.applied_tensions = ObsTerm(
            func=mdp.tendon_applied_tensions,
            scale=1.0 / 480.0,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # Filter termination velocity check to controlled joints only.
        # Threshold 500 (not 100): the near-massless wrist_intermediate dummy
        # link produces harmless solver velocity spikes >100 rad/s during
        # reset/base-snap transients; at 100 the env re-terminates in a silent
        # reset loop (joint_vel_diverged is a truncation) and never trains.
        self.terminations.joint_vel_diverged.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)
        self.terminations.joint_vel_diverged.params["max_velocity"] = 500.0
        # (With physxJoint:maxJointVelocity = 100 authored in the USD the term
        # above is effectively inert — kept purely as a divergence safety net.)
        #
        # Penalise (heavily, per step) a torn-open or branch-flipped four-bar —
        # closes the "break the linkage, steer with the base" reward hack.
        # Deliberately a REWARD PENALTY, not a termination: the reach reward is
        # net-negative per step, so a failure termination is a suicide exit the
        # policy immediately learns to trigger (measured 2026-07-09: episode
        # length collapsed 180 → 5 steps within 2 k timesteps as a DoneTerm).
        self.rewards.linkage_broken = RewTerm(
            func=mdp.linkage_integrity_penalty,
            weight=-1.0,
            params={
                "gap_threshold": 0.03,
                "branch_angle_tolerance": 0.3,
                "rod_cfg": SceneEntityCfg("robot", body_names=["rod_right_link"]),
                "forearm_cfg": SceneEntityCfg("robot", body_names=["forearm_link"]),
                "root_cfg": SceneEntityCfg("robot", body_names=["root_link"]),
                "rod_joints_cfg": SceneEntityCfg("robot", joint_names=["rod_left_joint", "rod_right_joint"]),
            },
        )
        # override events
        self.events.reset_robot_joints.params["asset_cfg"].joint_names = CONTROLLED_JOINT_NAMES
        self.events.reset_robot_joints.params["position_range"] = (-0.25, 0.25)


@configclass
class TensegrityReachPhysicalTendonEnvCfg_PLAY(TensegrityReachPhysicalTendonEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        self.observations.policy.enable_corruption = False


@configclass
class TensegrityReachPhysicalHierarchicalEnvCfg(TensegrityReachPhysicalTendonEnvCfg):
    """Hierarchical control variant of the physical tendon reach task.

    Identical model, scene, observations, rewards, and terminations as the
    direct force-control variant — only the action interface differs: the
    policy commands joint-space set-points (elbow + wrist) that are tracked
    by the validated inner PID→tension loop
    (:class:`~tensegrity_pick.robots.HierarchicalPhysicalTendonAction`).
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions = HierarchicalTendonReachActionsCfg()


@configclass
class TensegrityReachPhysicalHierarchicalEnvCfg_PLAY(TensegrityReachPhysicalHierarchicalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        self.observations.policy.enable_corruption = False
