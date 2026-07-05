# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR5e + Robotiq 2F-140 shirt-distribute environment configuration."""

from isaaclab.utils import configclass

from tensegrity_pick.robots.ur5e_robot_cfg import (
    CONTROLLED_JOINT_NAMES,
    TARGET_LINK_NAME,
    UR5E_GRIPPER_CFG,
)
from tensegrity_pick.tasks.manager_based.shared.cloth_sorting_scene_cfg import (
    SECOND_ROBOT_MOUNT_POS,
    SECOND_ROBOT_MOUNT_ROT_UR,
)
from tensegrity_pick.tasks.manager_based.shirt_distribute import mdp
from tensegrity_pick.tasks.manager_based.shirt_distribute.shirt_distribute_env_cfg import (
    ShirtDistributeEnvCfg,
)


@configclass
class UR5eF140ShirtDistributeEnvCfg(ShirtDistributeEnvCfg):
    """UR5e (6-DOF) shirt-distribute task."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot = UR5E_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UR5E_GRIPPER_CFG.init_state.replace(
                pos=SECOND_ROBOT_MOUNT_POS,
                rot=SECOND_ROBOT_MOUNT_ROT_UR,
            ),
        )

        # Relative joint positions (target = current + scale·action): a zero
        # action HOLDS the sampled far-from-default start pose — the
        # offset-from-default term would yank the arm home on step 1 (see the
        # action-space note in shirt_distribute_env_cfg.py).
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=list(CONTROLLED_JOINT_NAMES),
            scale=0.05,
        )

        self._set_robot_params(
            ee_body=TARGET_LINK_NAME,
            controlled_joints=list(CONTROLLED_JOINT_NAMES) + ["finger_joint"],
            arm_joints=list(CONTROLLED_JOINT_NAMES),
        )


@configclass
class UR5eF140ShirtDistributeEnvCfg_PLAY(UR5eF140ShirtDistributeEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0


@configclass
class UR5eF140ShirtDistributeEMAEnvCfg(UR5eF140ShirtDistributeEnvCfg):
    """EMA joint-position-to-limits action variant (iteration 5).

    Absolute smoothed joint targets (alpha = 0.2, the reach-grid winner and
    the shirt_present determinism fix): the policy MEAN encodes a target
    POSE, so exploration noise does not integrate into a position random
    walk — the failure mode measured on the relative-action variant
    (stochastic 0.8 vs deterministic 0.2–0.45).  Requires the
    ``reset_holding_pose`` EVENT (runs before ``action_manager.reset()``) so
    the EMA buffer snapshots the sampled holding pose instead of the stale
    pre-reset pose.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.arm_action = mdp.EMAJointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=list(CONTROLLED_JOINT_NAMES),
            alpha=0.2,
        )
        # action_l2 was the RELATIVE-action anti-saturation fix; in
        # to-limits space it arbitrarily biases toward mid-range postures.
        self.rewards.action_l2.weight = 0.0
        self.curriculum.action_l2 = None


@configclass
class UR5eF140ShirtDistributeEMAEnvCfg_PLAY(UR5eF140ShirtDistributeEMAEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
