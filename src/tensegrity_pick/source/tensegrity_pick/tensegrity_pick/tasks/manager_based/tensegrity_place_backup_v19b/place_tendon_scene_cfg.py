"""Scene configuration for the tendon-driven tensegrity place task.

Identical to :class:`PlaceSceneCfg` except the 3-DOF arm actuator is
replaced by an explicit effort-passthrough actuator.  Joint torques are
computed externally by :class:`~tensegrity_pick.robots.tendon_actuator.TendonEffortAction`.
"""

from __future__ import annotations

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from tensegrity_pick.robots import TENS_5DOF_GRIPPER_TENDON_CFG

from .place_scene_cfg import PlaceSceneCfg, PLACE_MOUNT_HEIGHT_M


@configclass
class PlaceTendonSceneCfg(PlaceSceneCfg):
    """Place scene with a tendon-driven arm actuator.

    Only the ``"arm"`` actuator group is changed — everything else
    (base, gripper, cubes, drum, lights, ground) is inherited.
    """

    robot = TENS_5DOF_GRIPPER_TENDON_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.15, 0.0, PLACE_MOUNT_HEIGHT_M),
            joint_pos={
                "base_y_joint": 0.0,
                "base_z_joint": -0.35,
                "elbow_joint": 0.0,
                "wrist_y_joint": 0.0,
                "wrist_x_joint": 0.0,
                "finger_joint": 0.0,
            },
        ),
        actuators={
            # Base: unchanged (implicit PD drives)
            "base": ImplicitActuatorCfg(
                joint_names_expr=["base_y_joint", "base_z_joint"],
                effort_limit_sim=800.0,
                velocity_limit_sim=5.0,
                stiffness=8000.0,
                damping=800.0,
            ),
            # Arm: inherits IdealPDActuator(stiffness=0, damping=0) from tendon cfg
            **{k: v for k, v in TENS_5DOF_GRIPPER_TENDON_CFG.actuators.items() if k == "arm"},
            # Gripper actuators: place-task-specific tuning
            "gripper_drive": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                effort_limit_sim=10.0,
                velocity_limit_sim=1.0,
                stiffness=11.25,
                damping=0.1,
                friction=0.0,
                armature=0.0,
            ),
            "gripper_finger": ImplicitActuatorCfg(
                joint_names_expr=[
                    "left_inner_finger_joint",
                    "right_inner_finger_joint",
                ],
                effort_limit_sim=1.0,
                velocity_limit_sim=1.0,
                stiffness=0.2,
                damping=0.001,
                friction=0.0,
                armature=0.0,
            ),
            "gripper_passive": ImplicitActuatorCfg(
                joint_names_expr=[
                    "left_inner_finger_pad_joint",
                    "right_inner_finger_pad_joint",
                    "left_outer_finger_joint",
                    "right_outer_finger_joint",
                    "right_outer_knuckle_joint",
                ],
                effort_limit_sim=1.0,
                velocity_limit_sim=1.0,
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
            ),
        },
    )
