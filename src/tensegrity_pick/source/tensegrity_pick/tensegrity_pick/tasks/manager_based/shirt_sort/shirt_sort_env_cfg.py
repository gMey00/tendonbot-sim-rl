# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for the tensegrity shirt sort task.

Adapts the cube sort task structure to cloth-simulated T-shirts on a
moving conveyor.  Uses the same reward pipeline philosophy as the cube
tasks.

TODO: Implement full env config once cloth simulation is integrated.
"""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .shirt_sort_scene_cfg import ShirtSortingSceneCfg


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
            "elbow_joint": (-1.2217, 1.2217),
            "wrist_y_joint": (-0.8727, 0.8727),
            "wrist_x_joint": (-0.8727, 0.8727),
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
        """TODO: Define observation terms for shirt sort task."""

        enable_corruption = True
        concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """TODO: Define reward terms for shirt sort task."""

    pass


@configclass
class TerminationsCfg:
    """TODO: Define termination terms for shirt sort task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class TensegrityShirtSortEnvCfg(ManagerBasedRLEnvCfg):
    """Shirt sort task — template configuration.

    TODO: Complete once cloth simulation is integrated.
    """

    scene: ShirtSortingSceneCfg = ShirtSortingSceneCfg(
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
