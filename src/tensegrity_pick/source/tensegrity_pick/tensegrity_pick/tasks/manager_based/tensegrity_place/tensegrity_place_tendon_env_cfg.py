"""Environment configuration for the tendon-driven tensegrity place task.

Inherits almost everything from :class:`TensegrityPlaceEnvCfg` — the only
differences are:

1. **Scene** → :class:`PlaceTendonSceneCfg` (explicit effort-passthrough arm
   actuator instead of implicit PD drives).
2. **Arm action** → :class:`TendonEffortActionCfg` (5 tendon tensions instead
   of 3 joint-position deltas).  The RL agent's action space for the arm
   increases from 3 to 5 dimensions.
3. **Observations** → tendon tensions are appended to the policy observation.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from tensegrity_pick.robots import TendonEffortActionCfg
from tensegrity_pick.robots.tendon_actuator import DEFAULT_JACOBIAN_TRANSPOSE

from . import mdp
from .mdp import rewards as task_rew
from .place_tendon_scene_cfg import PlaceTendonSceneCfg
from .tensegrity_place_env_cfg import (
    CONTROLLED_JOINT_NAMES,
    EE_LINK,
    GRASP_BODIES,
    ActionsCfg as BaseActionsCfg,
    CurriculumCfg,
    EventsCfg,
    ObservationsCfg as BaseObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
    TensegrityPlaceEnvCfg,
)


# ── Actions ───────────────────────────────────────────────────────────────

@configclass
class TendonActionsCfg:
    """Mixed action space: implicit PD for base, tendon efforts for arm,
    binary position for gripper."""

    base_delta = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["base_y_joint", "base_z_joint"],
        scale=0.50,
        use_default_offset=True,
        clip={"base_y_joint": (-0.5, 0.5), "base_z_joint": (-0.50, 0.0)},
    )

    arm_tendon = TendonEffortActionCfg(
        asset_name="robot",
        joint_names=["elbow_joint", "wrist_y_joint", "wrist_x_joint"],
        num_tendons=5,
        max_tension=500.0,
        jacobian_transpose=DEFAULT_JACOBIAN_TRANSPOSE,
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": 0.7854},
    )


# ── Observations ──────────────────────────────────────────────────────────

@configclass
class TendonObservationsCfg:
    """Same as base observations but with tendon tensions appended."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Proprioception (controlled joints only)
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
        )
        ee_pos_w = ObsTerm(
            func=task_rew.ee_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES)},
        )
        ee_vel_w = ObsTerm(
            func=task_rew.ee_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES)},
        )
        green_rel = ObsTerm(
            func=task_rew.cube_rel_pos,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
                "cube_name": "green_cube",
            },
        )
        red_rel = ObsTerm(
            func=task_rew.cube_rel_pos,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
                "cube_name": "red_cube",
            },
        )
        drum_rel = ObsTerm(
            func=task_rew.drum_rel_pos,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=GRASP_BODIES),
                "drum_name": "drum_target",
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ── Environment config ────────────────────────────────────────────────────

@configclass
class TensegrityPlaceTendonEnvCfg(TensegrityPlaceEnvCfg):
    """Tensegrity place task driven by tendon tensions instead of joint
    position deltas.

    The arm's 3 revolute joints (elbow, wrist_y, wrist_x) are actuated by
    5 tendons: 2 antagonistic for the elbow and 3 at 120° for the 2-DOF
    wrist.  Everything else (base, gripper, rewards, terminations,
    curriculum) is identical to the base task.
    """

    scene: PlaceTendonSceneCfg = PlaceTendonSceneCfg(num_envs=4096, env_spacing=5.0)
    actions: TendonActionsCfg = TendonActionsCfg()
    observations: TendonObservationsCfg = TendonObservationsCfg()


@configclass
class TensegrityPlaceTendonEnvCfg_PLAY(TensegrityPlaceTendonEnvCfg):
    """Smaller evaluation / play configuration."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
