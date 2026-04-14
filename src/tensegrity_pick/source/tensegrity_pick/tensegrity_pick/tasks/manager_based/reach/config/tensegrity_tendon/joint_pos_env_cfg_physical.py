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
from isaaclab.managers import SceneEntityCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

import isaaclab.sim as sim_utils
from tensegrity_pick.robots import PhysicalTendonEffortActionCfg
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
class TensegrityReachPhysicalTendonEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Switch robot to the physical linkage 5-DOF arm
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
        self.commands.ee_pose.joint_range_margin = 0.01
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
        # Filter observations to controlled joints only
        self.observations.policy.joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)}
        self.observations.policy.joint_vel.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)}
        # Filter termination velocity check to controlled joints only
        self.terminations.joint_vel_diverged.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)
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
