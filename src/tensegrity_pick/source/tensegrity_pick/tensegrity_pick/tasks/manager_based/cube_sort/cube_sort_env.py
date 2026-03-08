# cube_sort_env.py
#
# Custom environment for the cube sorting task with event-based rewards.
#
# Per research report: event-based rewards for placement/miss/red-grab
# are computed in step() and NOT dt-scaled. Continuous rewards remain
# in the RewardManager (dt-scaled via Isaac Lab).

from __future__ import annotations

from typing import Sequence

import torch

from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils.math import quat_apply

from ..shared.gripper_cfg import (
    BinCylinder,
    GRASP_CENTER_LOCAL_Z,
    FINGER_JOINT_CLOSE_POS,
    get_world_pos,
    in_upright_cylinder,
)
from .cube_sorting_scene_cfg import (
    BELT_HEIGHT_M,
    CUBE_LABELS,
    TARGET_LABEL,
    DISTRACTOR_LABEL,
    CONVEYOR_START_X,
    CONVEYOR_END_X,
)


CUBES_KEY = "cubes"
EE_BODY_NAME = "tool_link_0"
FINGER_JOINT_NAME = "finger_joint"

GRASP_PROXIMITY_THRESHOLD = 0.10
GRASP_LIFT_THRESHOLD = 0.06
GRASP_MIN_CLOSURE = 0.20

# Event reward magnitudes (NOT dt-scaled, per research report Section 3)
PLACEMENT_BASE_REWARD = 15.0
PLACEMENT_SCALING = 0.5
ALL_COMPLETE_BONUS = 50.0
CUBE_MISSED_PENALTY = -8.0
RED_GRABBED_PENALTY = -5.0

# Drum geometry for placement detection
DRUM_RADIUS = 0.547 * 0.5
DRUM_HEIGHT = 0.30

# Conveyor end threshold for miss detection
MISS_X_THRESHOLD = CONVEYOR_END_X + 0.35


class TensegrityCubeSortEnv(ManagerBasedRLEnv):
    """Manager-based RL env for the tensegrity cube-sorting task.

    Extends :class:`ManagerBasedRLEnv` with:
      - ``cube_labels``: ``(M,)`` int32 tensor mapping each cube index to a label
      - ``was_grasped``: latched per-env bool used to gate transport/release rewards
      - Event-based rewards: placement bonus, miss penalty, red-grab penalty
      - Per-cube tracking: ``_prev_in_drum``, ``_prev_missed`` for edge detection
    """

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        robot: Articulation = self.scene["robot"]
        self._ee_body_idx: int = robot.body_names.index(EE_BODY_NAME)
        self._finger_joint_idx: int = robot.joint_names.index(FINGER_JOINT_NAME)

        num_cubes = len(CUBE_LABELS)
        self._was_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._targets_placed = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._targets_missed = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._prev_in_drum = torch.zeros(self.num_envs, num_cubes, dtype=torch.bool, device=self.device)
        self._prev_missed = torch.zeros(self.num_envs, num_cubes, dtype=torch.bool, device=self.device)
        self._prev_red_grabbed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._bin_geom = BinCylinder(radius=DRUM_RADIUS, height=DRUM_HEIGHT)
        self._target_mask = (self.cube_labels == TARGET_LABEL)[None, :]  # (1, M)
        self._distractor_mask = (self.cube_labels == DISTRACTOR_LABEL)[None, :]

    @property
    def cube_labels(self) -> torch.Tensor:
        """(M,) int32 label for each cube in the unified collection."""
        if not hasattr(self, "_cube_labels"):
            self._cube_labels = torch.tensor(CUBE_LABELS, dtype=torch.int32, device=self.device)
        return self._cube_labels

    def _grasp_center_pos(self) -> torch.Tensor:
        """Grasp centre: EE body position projected along local +z."""
        robot: Articulation = self.scene["robot"]
        ee_pos = robot.data.body_pos_w[:, self._ee_body_idx, :]
        ee_quat = robot.data.body_quat_w[:, self._ee_body_idx, :]
        local_offset = ee_pos.new_tensor([0.0, 0.0, GRASP_CENTER_LOCAL_Z])
        world_offset = quat_apply(ee_quat, local_offset.expand_as(ee_pos))
        return ee_pos + world_offset

    @property
    def grasp_active(self) -> torch.Tensor:
        """Per-env bool: True when ANY target cube is physically held."""
        cubes: RigidObjectCollection = self.scene[CUBES_KEY]
        robot: Articulation = self.scene["robot"]

        gc_pos = self._grasp_center_pos()
        cube_pos = cubes.data.object_pos_w
        distance = torch.norm(cube_pos - gc_pos[:, None, :], dim=-1)
        is_close = distance < GRASP_PROXIMITY_THRESHOLD

        local_z = cube_pos[:, :, 2] - self.scene.env_origins[:, None, 2]
        is_lifted = local_z > (BELT_HEIGHT_M + GRASP_LIFT_THRESHOLD)

        finger_pos = robot.data.joint_pos[:, self._finger_joint_idx]
        is_closing = finger_pos > GRASP_MIN_CLOSURE

        per_cube = is_close & is_lifted & is_closing[:, None] & self._target_mask
        return torch.any(per_cube, dim=1)

    @property
    def was_grasped(self) -> torch.Tensor:
        """Per-env bool: True once any target cube has been grasped this episode."""
        return self._was_grasped

    def _detect_red_grabbed(self) -> torch.Tensor:
        """Per-env bool: True when gripper is closing on a red/distractor cube."""
        cubes: RigidObjectCollection = self.scene[CUBES_KEY]
        robot: Articulation = self.scene["robot"]

        gc_pos = self._grasp_center_pos()
        cube_pos = cubes.data.object_pos_w
        distance = torch.norm(cube_pos - gc_pos[:, None, :], dim=-1)
        is_close = distance < GRASP_PROXIMITY_THRESHOLD

        finger_pos = robot.data.joint_pos[:, self._finger_joint_idx]
        is_closing = finger_pos > GRASP_MIN_CLOSURE

        per_cube = is_close & is_closing[:, None] & self._distractor_mask
        return torch.any(per_cube, dim=1)

    def _compute_event_rewards(self) -> torch.Tensor:
        """Compute one-time event rewards (NOT dt-scaled).

        Events:
          - Target cube newly placed in drum: +15.0 + 0.5 * n_already_placed
          - All targets placed: +50.0
          - Target cube missed (passed belt end): -8.0
          - Red cube grabbed (new grab): -5.0
        """
        cubes: RigidObjectCollection = self.scene[CUBES_KEY]
        cube_pos = cubes.data.object_pos_w
        drum_pos = get_world_pos(self.scene["drum_target"], env=self)

        # Detect cubes currently in drum
        current_in_drum = in_upright_cylinder(cube_pos, drum_pos, self._bin_geom)

        # Detect cubes that passed the belt end (exclude parked cubes at x=100)
        local_x = cube_pos[..., 0] - self.scene.env_origins[:, None, 0]
        not_parked = local_x < 50.0
        current_missed = (local_x > MISS_X_THRESHOLD) & not_parked

        # --- Placement events (target cubes only) ---
        new_placements = current_in_drum & ~self._prev_in_drum & self._target_mask
        num_new_placed = new_placements.to(torch.float32).sum(dim=1)

        placement_reward = num_new_placed * (
            PLACEMENT_BASE_REWARD + PLACEMENT_SCALING * self._targets_placed.float()
        )
        self._targets_placed += num_new_placed.to(torch.int32)

        # Completion bonus (count only non-parked targets)
        active_target_count = (self._target_mask & not_parked).sum(dim=1).float()
        all_complete = (self._targets_placed >= active_target_count).float()
        # Only fire once: check if we JUST completed
        was_complete_before = ((self._targets_placed - num_new_placed.to(torch.int32)) >= active_target_count).float()
        completion_bonus = (all_complete - was_complete_before).clamp(min=0.0) * ALL_COMPLETE_BONUS

        # --- Miss events (target cubes only) ---
        new_misses = current_missed & ~self._prev_missed & self._target_mask & ~current_in_drum
        num_new_missed = new_misses.to(torch.float32).sum(dim=1)
        miss_penalty = num_new_missed * CUBE_MISSED_PENALTY
        self._targets_missed += num_new_missed.to(torch.int32)

        # --- Red cube grabbed (edge-triggered) ---
        red_grabbed_now = self._detect_red_grabbed()
        new_red_grab = red_grabbed_now & ~self._prev_red_grabbed
        red_penalty = new_red_grab.float() * RED_GRABBED_PENALTY

        # Update tracking state
        self._prev_in_drum = current_in_drum
        self._prev_missed = current_missed
        self._prev_red_grabbed = red_grabbed_now

        return placement_reward + completion_bonus + miss_penalty + red_penalty

    def step(self, action: torch.Tensor):
        obs, reward, terminated, time_outs, extras = super().step(action)

        still_running = ~(terminated | time_outs)
        self._was_grasped[still_running] |= self.grasp_active[still_running]

        # Add event-based rewards (not dt-scaled)
        event_reward = self._compute_event_rewards()
        reward = reward + event_reward

        return obs, reward, terminated, time_outs, extras

    def _reset_idx(self, env_ids: Sequence[int]):
        env_ids_t = (
            torch.tensor(env_ids, device=self.device, dtype=torch.long)
            if not isinstance(env_ids, torch.Tensor)
            else env_ids
        )

        # Log metrics before reset
        grasp_rate = torch.tensor(0.0, device=self.device)
        mean_ep_len = torch.tensor(0.0, device=self.device)
        mean_placed = torch.tensor(0.0, device=self.device)
        mean_missed = torch.tensor(0.0, device=self.device)
        if len(env_ids_t) > 0:
            grasp_rate = self._was_grasped[env_ids_t].float().mean()
            mean_ep_len = self.episode_length_buf[env_ids_t].float().mean()
            mean_placed = self._targets_placed[env_ids_t].float().mean()
            mean_missed = self._targets_missed[env_ids_t].float().mean()

        result = super()._reset_idx(env_ids)

        # Reset per-env tracking
        self._was_grasped[env_ids_t] = False
        self._targets_placed[env_ids_t] = 0
        self._targets_missed[env_ids_t] = 0
        self._prev_in_drum[env_ids_t] = False
        self._prev_missed[env_ids_t] = False
        self._prev_red_grabbed[env_ids_t] = False

        # Log metrics
        self.extras["log"]["Metrics/grasp_rate"] = grasp_rate
        self.extras["log"]["Metrics/mean_episode_length"] = mean_ep_len
        self.extras["log"]["Metrics/mean_targets_placed"] = mean_placed
        self.extras["log"]["Metrics/mean_targets_missed"] = mean_missed

        return result
