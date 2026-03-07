# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for the tensegrity shirt place task.

Adapts the cube place task structure to cloth-simulated T-shirts.
Uses the same 6-phase reward pipeline (reach → grasp → lift → transport
→ release → success) as the cube place task.

TODO: Implement full env config once cloth simulation is integrated.
"""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .shirt_place_scene_cfg import ShirtPlaceSceneCfg


@configclass
class ActionsCfg:
    """Joint-position delta actions (base + arm + gripper)."""

    base_delta = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["base_y_joint", "base_z_joint"],
        scale=0.50,
        use_default_offset=True,
        clip={"base_y_joint": (-0.5, 0.5), "base_z_joint": (-0.50, 0.0)},
    )
    arm_delta = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["elbow_joint", "wrist_y_joint", "wrist_x_joint"],
        scale=1.0,
        use_default_offset=True,
        clip={
            "elbow_joint": (-1.5, 1.5),
            "wrist_y_joint": (-0.8, 0.8),
            "wrist_x_joint": (-0.8, 0.8),
        },
    )
    gripper = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": 0.7854},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """TODO: Define observation terms for shirt place task."""

        enable_corruption = True
        concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """TODO: Define reward terms for shirt place task."""

    pass


@configclass
class TerminationsCfg:
    """TODO: Define termination terms for shirt place task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class TensegrityShirtPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Shirt place task — template configuration.

    TODO: Complete once cloth simulation is integrated.
    """

    scene: ShirtPlaceSceneCfg = ShirtPlaceSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
    )
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    sim = ManagerBasedRLEnvCfg.sim
    sim.dt = 0.01
    sim.render_interval = 2

    episode_length_s = 8.0
