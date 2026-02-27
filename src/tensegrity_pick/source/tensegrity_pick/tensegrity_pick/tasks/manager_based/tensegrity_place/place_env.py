# place_env.py
#
# Custom environment with physics-based gripper for the place task.
#
# Grasping relies on the physically simulated Robotiq 2F-140 gripper whose
# joints are driven via BinaryJointPositionActionCfg.  No teleportation or
# sticky-gripper logic is used — the cube must be physically held between
# the finger pads.
#
# Modelled after IsaacLab's lift task: the env only adds minimal custom
# state tracking (was_grasped latch) and keeps the reward logic in the
# RewardManager.  No custom step() override or injected rewards.

from __future__ import annotations

from typing import Sequence

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils.math import quat_apply

from .place_scene_cfg import CONVEYOR_SURFACE_HEIGHT_M
from .mdp.rewards import GRASP_CENTER_LOCAL_Z


EE_BODY_NAME = "tool_link_0"
GREEN_CUBE_KEY = "green_cube"

# Physics-based grasp detection thresholds
GRASP_PROXIMITY_THRESHOLD = 0.25
GRASP_LIFT_THRESHOLD = 0.03
BELT_HEIGHT = CONVEYOR_SURFACE_HEIGHT_M


class PlaceEnvWithStickyGripper(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with a physically simulated Robotiq gripper.

    Despite the legacy class name (kept for registration compatibility),
    this version does **not** use a sticky gripper.  Grasping is purely
    physics-based: the Robotiq 2F-140 finger joints are actuated via
    ``BinaryJointPositionActionCfg`` and the cube is held by contact
    forces between the finger pads and the cube surface.

    ``grasp_active`` is detected by proximity + lift: the cube must be
    close to the end-effector **and** lifted above the belt surface.
    """

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        robot: Articulation = self.scene["robot"]
        self._ee_body_idx: int = robot.body_names.index(EE_BODY_NAME)

        # Latched flag: True once the cube has been grasped at least once
        # this episode.  Prevents reward-hacking (pushing cube into drum
        # without grasping).
        self._was_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------
    # Grasp centre helper
    # ------------------------------------------------------------------

    def _grasp_center_pos(self) -> torch.Tensor:
        """Grasp centre: EE body position projected along local -z by the pad offset."""
        robot: Articulation = self.scene["robot"]
        ee_pos = robot.data.body_pos_w[:, self._ee_body_idx, :]
        ee_quat = robot.data.body_quat_w[:, self._ee_body_idx, :]
        local_offset = ee_pos.new_tensor([0.0, 0.0, GRASP_CENTER_LOCAL_Z])
        world_offset = quat_apply(ee_quat, local_offset.expand_as(ee_pos))
        return ee_pos + world_offset

    # ------------------------------------------------------------------
    # Physics-based grasp detection
    # ------------------------------------------------------------------

    @property
    def grasp_active(self) -> torch.Tensor:
        """Per-env bool: True when the cube is physically held by the gripper."""
        green: RigidObject = self.scene[GREEN_CUBE_KEY]

        gc_pos = self._grasp_center_pos()
        cube_pos = green.data.root_pos_w
        distance = torch.norm(cube_pos - gc_pos, dim=-1)

        local_z = cube_pos[:, 2] - self.scene.env_origins[:, 2]
        is_lifted = local_z > (BELT_HEIGHT + GRASP_LIFT_THRESHOLD)
        is_close = distance < GRASP_PROXIMITY_THRESHOLD

        return is_close & is_lifted

    @property
    def was_grasped(self) -> torch.Tensor:
        """Per-env bool: True once the cube has been grasped this episode."""
        return self._was_grasped

    # ------------------------------------------------------------------
    # Override _step_impl to update latched grasp flag
    # ------------------------------------------------------------------

    def step(self, action: torch.Tensor):
        """Standard step with grasp-latch update after physics."""
        obs, reward, terminated, time_outs, extras = super().step(action)
        # Update latched grasp flag after physics
        self._was_grasped |= self.grasp_active
        return obs, reward, terminated, time_outs, extras

    # ------------------------------------------------------------------
    # Reset custom state for specific envs
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int]):
        result = super()._reset_idx(env_ids)
        env_ids_t = (
            torch.tensor(env_ids, device=self.device, dtype=torch.long)
            if not isinstance(env_ids, torch.Tensor)
            else env_ids
        )
        self._was_grasped[env_ids_t] = False
        return result
