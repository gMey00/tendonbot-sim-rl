"""Tendon-driven variant of the tensegrity reach environment.

Inherits the full MDP (rewards, terminations, curriculum, observations, commands)
from :class:`TensegrityReachEnvCfg`.  Only two things change:

1. **Scene** → The robot uses the tendon-driven arm actuator
   (:class:`~tensegrity_pick.robots.tendon_robot_cfg.TENS_5DOF_GRIPPER_TENDON_CFG`).
2. **Actions** → 5-DOF base position + 5 tendon tensions instead of 5-DOF
   joint position deltas.

This allows direct performance comparison between PD-driven and tendon-driven
control on an *identical* reach task.
"""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from tensegrity_pick.robots import TendonEffortActionCfg
from tensegrity_pick.robots.tendon_actuator import DEFAULT_JACOBIAN_TRANSPOSE
from tensegrity_pick.robots.tendon_robot_cfg import TENS_5DOF_GRIPPER_TENDON_CFG

from . import mdp
from ..tensegrity_pick.proj_base_scene_cfg import ProjBaseSceneCfg, ROBOT_MOUNT_HEIGHT_M
from .tensegrity_reach_env_cfg import (
    CONTROLLED_JOINT_NAMES,
    TARGET_LINK_NAME,
    CommandsCfg,
    CurriculumCfg,
    EventsCfg,
    RewardsCfg,
    TerminationsCfg,
    TensegrityReachEnvCfg,
)


# ── Scene ─────────────────────────────────────────────────────────────────

@configclass
class ReachTendonSceneCfg(ProjBaseSceneCfg):
    """Base scene with the tendon-driven 5-DOF robot for reach."""

    robot = TENS_5DOF_GRIPPER_TENDON_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.15, 0.0, ROBOT_MOUNT_HEIGHT_M),
        ),
    )


# ── Actions ───────────────────────────────────────────────────────────────

@configclass
class TendonReachActionsCfg:
    """Base position + tendon arm.  No gripper needed for reach."""

    base_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["base_y_joint", "base_z_joint"],
        scale=0.25,
        use_default_offset=True,
    )

    arm_tendon = TendonEffortActionCfg(
        asset_name="robot",
        joint_names=["elbow_joint", "wrist_y_joint", "wrist_x_joint"],
        num_tendons=5,
        max_tension=500.0,
        jacobian_transpose=DEFAULT_JACOBIAN_TRANSPOSE,
    )


# ── Observations ──────────────────────────────────────────────────────────
# Same structure as the base reach obs; we re-declare to ensure
# consistency with the (potentially different) action dimensions.

@configclass
class TendonReachObservationsCfg:
    """Observations for the tendon-driven reach task."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINT_NAMES)},
        )
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ── Environment config ────────────────────────────────────────────────────

@configclass
class TensegrityReachTendonEnvCfg(TensegrityReachEnvCfg):
    """Reach task with tendon-driven arm — direct drop-in replacement.

    Rewards, terminations, curriculum, commands, and events are *identical*
    to the PD-driven version, enabling fair performance comparison.
    """

    scene: ReachTendonSceneCfg = ReachTendonSceneCfg(num_envs=2000, env_spacing=5.0)
    actions: TendonReachActionsCfg = TendonReachActionsCfg()
    observations: TendonReachObservationsCfg = TendonReachObservationsCfg()


@configclass
class TensegrityReachTendonEnvCfg_PLAY(TensegrityReachTendonEnvCfg):
    """Smaller evaluation / play configuration."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 5.0
        self.observations.policy.enable_corruption = False
