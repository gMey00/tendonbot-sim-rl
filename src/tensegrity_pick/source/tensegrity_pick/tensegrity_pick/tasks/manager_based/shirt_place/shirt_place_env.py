# shirt_place_env.py
#
# Custom environment for the shirt placement task.
# Adapted from cube_place/place_env.py with single-object shirt_proxy
# instead of green/red cubes.
#
# Adds latched ``was_grasped`` and ``was_placed`` flags for:
#  - Gating transport rewards (only after genuine lift)
#  - Tracking episode-level grasp and place success metrics

from __future__ import annotations

from typing import Sequence

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils.math import quat_apply

from .shirt_place_scene_cfg import CONVEYOR_SURFACE_HEIGHT_M
from .mdp.rewards import GRASP_CENTER_LOCAL_Z, BinCylinder, _in_upright_cylinder, _get_world_pos


EE_BODY_CANDIDATES = ("tool_link_0", "robotiq_base_link", "end_effector_link")
SHIRT_PROXY_KEY = "shirt_proxy"
DRUM_KEY = "drum_target"
FINGER_JOINT_NAME = "finger_joint"

# Physics-based grasp detection thresholds
GRASP_PROXIMITY_THRESHOLD = 0.10
GRASP_LIFT_THRESHOLD = 0.06
GRASP_MIN_CLOSURE = 0.20
BELT_HEIGHT = CONVEYOR_SURFACE_HEIGHT_M

# Drum geometry — must match _BIN_GEOM in shirt_place_env_cfg.py
_DRUM_GEOM = BinCylinder(radius=0.547 * 0.5, height=0.30)


class TensegrityShirtPlaceEnv(ManagerBasedRLEnv):
    """Manager-based RL env for the tensegrity shirt-placement task.

    Extends :class:`ManagerBasedRLEnv` with latched grasp/place flags.
    Uses a rigid-body proxy as stand-in for the cloth centroid.
    """

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        robot: Articulation = self.scene["robot"]
        ee_body_name = next(n for n in EE_BODY_CANDIDATES if n in robot.body_names)
        self._ee_body_idx: int = robot.body_names.index(ee_body_name)
        self._finger_joint_idx: int = robot.joint_names.index(FINGER_JOINT_NAME)

        self._was_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._was_placed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------
    # Grasp centre helper
    # ------------------------------------------------------------------

    def _grasp_center_pos(self) -> torch.Tensor:
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
        """Per-env bool: True when the shirt proxy is physically held."""
        shirt: RigidObject = self.scene[SHIRT_PROXY_KEY]
        robot: Articulation = self.scene["robot"]

        gc_pos = self._grasp_center_pos()
        shirt_pos = shirt.data.root_pos_w
        distance = torch.norm(shirt_pos - gc_pos, dim=-1)

        local_z = shirt_pos[:, 2] - self.scene.env_origins[:, 2]
        is_lifted = local_z > (BELT_HEIGHT + GRASP_LIFT_THRESHOLD)
        is_close = distance < GRASP_PROXIMITY_THRESHOLD

        finger_pos = robot.data.joint_pos[:, self._finger_joint_idx]
        is_closing = finger_pos > GRASP_MIN_CLOSURE

        return is_close & is_lifted & is_closing

    @property
    def was_grasped(self) -> torch.Tensor:
        return self._was_grasped

    @property
    def was_placed(self) -> torch.Tensor:
        return self._was_placed

    def _shirt_in_drum(self) -> torch.Tensor:
        shirt = self.scene[SHIRT_PROXY_KEY]
        drum_pos = _get_world_pos(self.scene[DRUM_KEY], env=self)
        return _in_upright_cylinder(shirt.data.root_pos_w, drum_pos, _DRUM_GEOM)

    # ------------------------------------------------------------------
    # Step: update latched flags
    # ------------------------------------------------------------------

    def step(self, action: torch.Tensor):
        obs, reward, terminated, time_outs, extras = super().step(action)
        still_running = ~(terminated | time_outs)
        self._was_grasped[still_running] |= self.grasp_active[still_running]
        self._was_placed[still_running] |= (
            self._shirt_in_drum() & self._was_grasped
        )[still_running]
        return obs, reward, terminated, time_outs, extras

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int]):
        env_ids_t = (
            torch.tensor(env_ids, device=self.device, dtype=torch.long)
            if not isinstance(env_ids, torch.Tensor)
            else env_ids
        )

        grasp_rate = torch.tensor(0.0, device=self.device)
        place_rate = torch.tensor(0.0, device=self.device)
        mean_ep_len = torch.tensor(0.0, device=self.device)
        if len(env_ids_t) > 0:
            grasp_rate = self._was_grasped[env_ids_t].float().mean()
            place_rate = self._was_placed[env_ids_t].float().mean()
            mean_ep_len = self.episode_length_buf[env_ids_t].float().mean()

        result = super()._reset_idx(env_ids)
        self._was_grasped[env_ids_t] = False
        self._was_placed[env_ids_t] = False

        self.extras["log"]["Metrics/grasp_rate"] = grasp_rate
        self.extras["log"]["Metrics/place_success_rate"] = place_rate
        self.extras["log"]["Metrics/mean_episode_length"] = mean_ep_len
        return result
