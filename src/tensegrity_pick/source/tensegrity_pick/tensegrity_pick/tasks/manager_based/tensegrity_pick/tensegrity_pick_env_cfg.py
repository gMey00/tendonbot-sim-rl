# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp  # keep cartpole-style local mdp module
from .cube_sorting_scene_cfg import (
    CubeSortingSceneCfg, 
    CONVEYOR_START_X, 
    CONVEYOR_END_X,
    BELT_HEIGHT_M, 
    BELT_WIDTH_M,
    CONVEYOR_CENTER_Y
)
from .tensegrity_robot_cfg import TENS_5DOF_GRIPPER_CFG
from .mdp import cube_sorting_mdp as task_mdp
from .mdp import rewards_cube_sorting as task_rew

##
# Scene definition
##


@configclass
class CubeSortSceneWrapperCfg(CubeSortingSceneCfg):
    """Wrap of the CubeSortingSceneCfg."""

    #robot = TENS_5DOF_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")



##
# MDP settings
##


@configclass
class ActionsCfg:
    """Joint position delta actions (base + arm + gripper)."""

    base_delta = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["base_y_joint", "base_z_joint"],
        scale=0.02,
        use_default_offset=True,
    )
    arm_delta = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["elbow_joint", "wrist_y_joint", "wrist_x_joint"],
        scale=0.10,
        use_default_offset=True,
    )
    gripper_delta = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        scale=0.20,
        use_default_offset=True,
    )




@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # proprioception
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})

        # EE kinematics (EE body name is tool_link)
        ee_pos_w = ObsTerm(func=task_mdp.ee_pos_w, params={"asset_cfg": SceneEntityCfg("robot", body_names=["tool_link_0"])})
        ee_vel_w = ObsTerm(func=task_mdp.ee_lin_vel_w, params={"asset_cfg": SceneEntityCfg("robot", body_names=["tool_link_0"])})

        # Relative positions to nearest cubes
        nearest_green_rel = ObsTerm(
            func=task_mdp.nearest_obj_rel_pos,
            params={"ee_cfg": SceneEntityCfg("robot", body_names=["tool_link_0"]), "obj_collection_name": "green_cubes"},
        )
        nearest_red_rel = ObsTerm(
            func=task_mdp.nearest_obj_rel_pos,
            params={"ee_cfg": SceneEntityCfg("robot", body_names=["tool_link_0"]), "obj_collection_name": "red_cubes"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    # Reset robot joints around default pose
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "base_y_joint",
                    "base_z_joint",
                    "elbow_joint",
                    "wrist_y_joint",
                    "wrist_x_joint",
                    "finger_joint",
                ],
            ),
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.1, 0.1),
        },
    )

    # Reset cubes: Pattern 1 (activate k and park rest)
    reset_cubes = EventTerm(
        func=task_mdp.reset_cube_pool_pattern1,
        mode="reset",
        params={
            "spawn_box": task_mdp.SpawnBox(
                x_range=(CONVEYOR_START_X + 0.10, CONVEYOR_START_X + 1.80),
                y_range=(-0.20, 0.20),
                z_range=(BELT_HEIGHT_M + 0.03, BELT_HEIGHT_M + 0.05),
            ),
            "num_green_active": 8,
            "num_red_active": 8,
            "green_name": "green_cubes",
            "red_name": "red_cubes",
            "parking_pose": (100.0, 100.0, 1.0),
        },
    )

    # Randomize belt speed per env at reset (stored in env.extras["belt_speed"])
    sample_belt_speed = EventTerm(
        func=task_mdp.sample_and_store_belt_speed,
        mode="reset",
        params={"high": 0.8, "low": 0.2, "key": "belt_speed"},
    )

    # Apply belt effect every control step (interval)
    apply_conveyor = EventTerm(
        func=task_mdp.apply_conveyor_velocity_to_cubes,
        mode="interval",
        interval_range_s=(0.001, 0.001),  # Apply approximately every step (1ms)
        params={
            "cube_collection_names": ("green_cubes", "red_cubes"),
            "belt_speed": None,              # read env.extras["belt_speed"]
            "belt_speed_key": "belt_speed",
            "belt_axis": "x",
            "belt_height": BELT_HEIGHT_M,
            "belt_center_y": CONVEYOR_CENTER_Y,
            "belt_half_width": BELT_WIDTH_M * 0.5,
            "on_belt_height_tol": 0.10,
            "lift_disable_height": 0.15,
        },
    )



@configclass
class RewardsCfg:
    green_in_target = RewTerm(
        func=task_rew.green_in_target,
        weight=10.0,
        params={
            "green_name": "green_cubes",
            "target_bin_name": "drum_target",
            "bin_geom": task_rew.BinCylinder(radius=0.547 * 0.5, height=0.880),
        },
    )
    red_in_target = RewTerm(
        func=task_rew.red_in_target,
        weight=-12.0,
        params={
            "red_name": "red_cubes",
            "target_bin_name": "drum_target",
            "bin_geom": task_rew.BinCylinder(radius=0.547 * 0.5, height=0.880),
        },
    )
    green_missed = RewTerm(
        func=task_rew.green_missed,
        weight=-6.0,
        params={
            "green_name": "green_cubes",
            "x_passed_threshold": CONVEYOR_END_X + 0.35,
            "target_bin_name": "drum_target",
            "bin_geom": task_rew.BinCylinder(radius=0.547 * 0.5, height=0.880),
        },
    )

    # Dense shaping rewards for learning pickup
    reach_green = RewTerm(
        func=task_rew.ee_to_nearest_green_l2,
        weight=-0.5,  # Reduced from -1.0 to not dominate
        params={"ee_cfg": SceneEntityCfg("robot", body_names=["tool_link_0"]), "green_name": "green_cubes"},
    )
    
    cube_lifted = RewTerm(
        func=task_rew.cube_lifted_bonus,
        weight=3.0,  # Bonus for lifting cubes
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names=["tool_link_0"]),
            "green_name": "green_cubes",
            "belt_height": BELT_HEIGHT_M,
            "lift_threshold": 0.08,
            "proximity_threshold": 0.15,
        },
    )
    
    approach_target = RewTerm(
        func=task_rew.cube_approaching_target,
        weight=-0.3,  # Encourage moving lifted cube toward target
        params={
            "green_name": "green_cubes",
            "target_bin_name": "drum_target",
            "belt_height": BELT_HEIGHT_M,
            "lift_threshold": 0.08,
        },
    )
    
    action_smooth = RewTerm(func=task_rew.action_rate_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    # time limit
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # End episode once all *active* cubes passed the conveyor end (Reset option A)
    all_cubes_passed = DoneTerm(
        func=task_mdp.all_active_passed_x,
        params={
            "obj_collection_names": ("green_cubes", "red_cubes"),
            "x_threshold": CONVEYOR_END_X + 0.35,
            "parking_x_threshold": 50.0,
        },
    )


##
# Environment configuration
##      

@configclass
class TensegrityCubeSortEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: CubeSortSceneWrapperCfg = CubeSortSceneWrapperCfg(num_envs=1024, env_spacing=5.0)
    # MDP settings
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        # general settings
        self.decimation = 2
        self.episode_length_s = 8.0
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        # physics settings for stability
        self.sim.physx.solver_type = 1  # TGS solver
        self.sim.physx.enable_stabilization = True