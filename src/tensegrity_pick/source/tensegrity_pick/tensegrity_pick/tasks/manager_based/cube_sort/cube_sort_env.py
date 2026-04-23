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
PLACEMENT_BASE_REWARD = 50.0
PLACEMENT_SCALING = 10.0
ALL_COMPLETE_BONUS = 100.0
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
      - Event-based rewards: placement bonus, miss penalty, red-grab penalty
      - Per-cube tracking: ``_prev_in_drum``, ``_prev_missed`` for edge detection

    The ``was_grasped`` env-level latch was REMOVED in R4 of the cube_sort
    rework. The per-cube reward machine in ``mdp/per_cube_state.py`` now
    handles all grasp-conditioned rewards via Markovian per-cube state.
    """

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        robot: Articulation = self.scene["robot"]
        self._ee_body_idx: int = robot.body_names.index(EE_BODY_NAME)
        self._finger_joint_idx: int = robot.joint_names.index(FINGER_JOINT_NAME)

        num_cubes = len(CUBE_LABELS)
        # _was_grasped removed in R4 (per-cube reward machine handles grasp gating).
        self._targets_placed = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._targets_missed = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._prev_in_drum = torch.zeros(self.num_envs, num_cubes, dtype=torch.bool, device=self.device)
        self._prev_missed = torch.zeros(self.num_envs, num_cubes, dtype=torch.bool, device=self.device)
        self._prev_red_grabbed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._is_placed = torch.zeros(self.num_envs, num_cubes, dtype=torch.bool, device=self.device)

        # Per-cube grasp tracking: (N, M) bool — latched per unique cube
        self._cube_was_grasped = torch.zeros(self.num_envs, num_cubes, dtype=torch.bool, device=self.device)
        # Red cube grab event counter (per-env)
        self._red_grabbed_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        num_labels = max(TARGET_LABEL, DISTRACTOR_LABEL) + 1
        self._tracked_idx = torch.full(
            (self.num_envs, num_labels), -1, dtype=torch.long, device=self.device,
        )

        self._bin_geom = BinCylinder(radius=DRUM_RADIUS, height=DRUM_HEIGHT)
        self._target_mask = (self.cube_labels == TARGET_LABEL)[None, :]  # (1, M)
        self._target_label = int(TARGET_LABEL)  # exposed for set-encoder obs (R6)
        self._distractor_mask = (self.cube_labels == DISTRACTOR_LABEL)[None, :]

        # Count active cubes per label for normalised metrics
        # Extract from reset_cubes event config
        reset_params = cfg.events.reset_cubes.params
        active_per_label = reset_params.get("active_per_label", {})
        self._num_active_green = int(active_per_label.get(str(TARGET_LABEL), 0))
        self._num_active_red = int(active_per_label.get(str(DISTRACTOR_LABEL), 0))

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
    def grasp_active_per_cube(self) -> torch.Tensor:
        """(N, M) bool: per-cube grasp detection for ALL cubes (targets only)."""
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

        return is_close & is_lifted & is_closing[:, None] & self._target_mask

    @property
    def was_grasped(self) -> torch.Tensor:
        """DEPRECATED (R4): kept as a thin shim for backward-compat with code
        that hasn't migrated to ``env._mdp_state['ever_held']`` yet.

        Returns the per-env ANY-cube ``ever_held`` boolean from the new
        per-cube state if available, else falls back to the legacy
        proximity-based ``grasp_active`` accumulator.
        """
        state = getattr(self, "_mdp_state", None)
        if state is not None:
            return (state["ever_held"] & self._target_mask).any(dim=1)
        # Fallback (R4 reward term not yet active): always False.
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _detect_red_grabbed(self) -> torch.Tensor:
        """Per-env bool: True when gripper is lifting a red/distractor cube."""
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

        per_cube = is_close & is_lifted & is_closing[:, None] & self._distractor_mask
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
        self._red_grabbed_count += new_red_grab.to(torch.int32)

        # Update tracking state
        self._prev_in_drum = current_in_drum
        self._prev_missed = current_missed
        self._prev_red_grabbed = red_grabbed_now
        self._is_placed = self._is_placed | new_placements

        return placement_reward + completion_bonus + miss_penalty + red_penalty

    def step(self, action: torch.Tensor):
        obs, reward, terminated, time_outs, extras = super().step(action)

        # Track per-cube grasps for episode-level metric logging
        # (uses the legacy ``grasp_active_per_cube`` proximity heuristic; will
        # be replaced by ``env._mdp_state['ever_held']`` once R4 logging is fully
        # rewired).
        still_running = ~(terminated | time_outs)
        per_cube_grasp = self.grasp_active_per_cube  # (N, M)
        self._cube_was_grasped[still_running] |= per_cube_grasp[still_running]

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

        # ── Compute metrics before reset ──────────────────────────────
        zero = torch.tensor(0.0, device=self.device)
        mean_ep_len = zero
        # Raw counts
        green_placed_count = zero
        green_missed_count = zero
        green_grasped_count = zero
        red_grabbed_count = zero
        red_in_drum_count = zero
        # Normalised rates
        green_placement_rate = zero
        green_miss_rate = zero
        green_grasp_rate = zero
        # Legacy (binary per-env)
        any_grasp_rate = zero

        if len(env_ids_t) > 0:
            mean_ep_len = self.episode_length_buf[env_ids_t].float().mean()

            # Green cube counts
            placed = self._targets_placed[env_ids_t].float()
            missed = self._targets_missed[env_ids_t].float()
            green_placed_count = placed.mean()
            green_missed_count = missed.mean()

            # Per-cube green grasps (how many unique green cubes were grasped)
            per_cube_grasped = self._cube_was_grasped[env_ids_t] & self._target_mask
            green_grasped_count = per_cube_grasped.float().sum(dim=1).mean()

            # Red cube metrics
            red_grabbed_count = self._red_grabbed_count[env_ids_t].float().mean()

            # Red cubes currently in drum at end of episode (sorting errors)
            cubes: RigidObjectCollection = self.scene[CUBES_KEY]
            cube_pos = cubes.data.object_pos_w[env_ids_t]
            drum_pos = get_world_pos(self.scene["drum_target"], env=self)
            if drum_pos.shape[0] != env_ids_t.shape[0]:
                drum_pos = drum_pos[env_ids_t]
            red_in_drum = in_upright_cylinder(cube_pos, drum_pos, self._bin_geom)
            red_in_drum = red_in_drum & self._distractor_mask
            red_in_drum_count = red_in_drum.float().sum(dim=1).mean()

            # Normalised rates (per active cube)
            num_green = max(self._num_active_green, 1)
            num_red = max(self._num_active_red, 1)
            green_placement_rate = green_placed_count / num_green
            green_miss_rate = green_missed_count / num_green
            green_grasp_rate = green_grasped_count / num_green

            # Legacy binary grasp rate (now derived from per-cube latch)
            any_grasp_rate = self._cube_was_grasped[env_ids_t].any(dim=1).float().mean()

        result = super()._reset_idx(env_ids)

        # Reset per-env tracking
        # _was_grasped reset removed in R4 (attribute deleted).
        self._targets_placed[env_ids_t] = 0
        self._targets_missed[env_ids_t] = 0
        self._prev_in_drum[env_ids_t] = False
        self._prev_missed[env_ids_t] = False
        self._prev_red_grabbed[env_ids_t] = False
        self._is_placed[env_ids_t] = False
        self._cube_was_grasped[env_ids_t] = False
        self._red_grabbed_count[env_ids_t] = 0
        self._tracked_idx[env_ids_t] = -1

        # ── Log metrics ───────────────────────────────────────────────
        log = self.extras["log"]
        log["Metrics/mean_episode_length"] = mean_ep_len

        # Per-cube normalised rates (the PRIMARY metrics to watch)
        log["Metrics/green_grasp_rate"] = green_grasp_rate
        log["Metrics/green_placement_rate"] = green_placement_rate
        log["Metrics/green_miss_rate"] = green_miss_rate

        # Raw counts per episode
        log["Metrics/green_grasped_count"] = green_grasped_count
        log["Metrics/green_placed_count"] = green_placed_count
        log["Metrics/green_missed_count"] = green_missed_count

        # Red cube metrics
        log["Metrics/red_grabbed_count"] = red_grabbed_count
        log["Metrics/red_in_drum_count"] = red_in_drum_count

        # Legacy (kept for backward compat — this is binary, NOT per-cube)
        log["Metrics/grasp_rate"] = any_grasp_rate
        log["Metrics/mean_targets_placed"] = green_placed_count
        log["Metrics/mean_targets_missed"] = green_missed_count

        return result
