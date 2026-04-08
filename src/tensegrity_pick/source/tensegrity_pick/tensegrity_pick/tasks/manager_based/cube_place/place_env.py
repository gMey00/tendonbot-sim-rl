# place_env.py
#
# Custom environment with physics-based Robotiq 2F-140 gripper.
# Grasping is purely physics-based via BinaryJointPositionActionCfg.
# Adds a latched ``was_grasped`` flag used by the approach reward to
# gate lateral transport — only activated after a genuine lift.

from __future__ import annotations

from typing import Sequence

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils.math import quat_apply

from .place_scene_cfg import CONVEYOR_SURFACE_HEIGHT_M
from .mdp.rewards import GRASP_CENTER_LOCAL_Z, BinCylinder, ConveyorBounds, _in_upright_cylinder, _get_world_pos


EE_BODY_CANDIDATES = ("tool_link_0", "robotiq_base_link", "end_effector_link")
GREEN_CUBE_KEY = "green_cube"
DRUM_KEY = "drum_target"
FINGER_JOINT_NAME = "finger_joint"

# Physics-based grasp detection thresholds
# Aligned with lift reward gating (belt + 0.06) to avoid false-positive
# "grasp" at reset (spawn z is belt + [0.03, 0.05]).
GRASP_PROXIMITY_THRESHOLD = 0.10
GRASP_LIFT_THRESHOLD = 0.06
GRASP_MIN_CLOSURE = 0.20  # ~25% of 0.7854; reject open-gripper bumps
BELT_HEIGHT = CONVEYOR_SURFACE_HEIGHT_M

# Conveyor bounds for red-on-belt metric
_CONVEYOR_BOUNDS = ConveyorBounds(y_min=-0.4, y_max=0.4, z_min=0.70)

# Drum geometry — must match _BIN_GEOM in place_env_cfg.py
_DRUM_GEOM = BinCylinder(radius=0.547 * 0.5, height=0.30)


class TensegrityPlaceEnv(ManagerBasedRLEnv):
    """Manager-based RL env for the tensegrity cube-placement task.

    Extends :class:`ManagerBasedRLEnv` with a latched ``was_grasped``
    flag.  ``grasp_active`` is detected by proximity + lift: the cube
    must be close to the end-effector **and** lifted above the belt.
    """

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        robot: Articulation = self.scene["robot"]
        ee_body_name = next(n for n in EE_BODY_CANDIDATES if n in robot.body_names)
        self._ee_body_idx: int = robot.body_names.index(ee_body_name)
        self._finger_joint_idx: int = robot.joint_names.index(FINGER_JOINT_NAME)

        # Latched flag: True once the cube has been grasped at least once
        # this episode.  Prevents reward-hacking (pushing cube into drum
        # without grasping).
        self._was_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Latched flag: True once the green cube entered the drum this episode
        # (after a valid grasp).  Used as a per-episode binary placement metric.
        self._was_placed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Latched flag: True once the task is complete — either the cube has
        # been placed in the drum, or was grasped and then dropped far from
        # the EE (unreachable).  Used to gate reaching rewards off and
        # activate the return-to-neutral reward.
        self._task_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Latched flag: True once the red cube left the conveyor belt
        # this episode.  Used as a per-episode metric for red cube handling.
        self._red_left_belt = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------
    # Grasp centre helper
    # ------------------------------------------------------------------

    def _grasp_center_pos(self) -> torch.Tensor:
        """Grasp centre: EE body position projected along local +z (downward in world)."""
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
        robot: Articulation = self.scene["robot"]

        gc_pos = self._grasp_center_pos()
        cube_pos = green.data.root_pos_w
        distance = torch.norm(cube_pos - gc_pos, dim=-1)

        local_z = cube_pos[:, 2] - self.scene.env_origins[:, 2]
        is_lifted = local_z > (BELT_HEIGHT + GRASP_LIFT_THRESHOLD)
        is_close = distance < GRASP_PROXIMITY_THRESHOLD

        finger_pos = robot.data.joint_pos[:, self._finger_joint_idx]
        is_closing = finger_pos > GRASP_MIN_CLOSURE

        return is_close & is_lifted & is_closing

    @property
    def was_grasped(self) -> torch.Tensor:
        """Per-env bool: True once the cube has been grasped this episode."""
        return self._was_grasped

    @property
    def was_placed(self) -> torch.Tensor:
        """Per-env bool: True once the green cube entered the drum this episode."""
        return self._was_placed

    @property
    def task_done(self) -> torch.Tensor:
        """Per-env bool: True once the cube is placed or dropped out of reach."""
        return self._task_done

    def _green_in_drum(self) -> torch.Tensor:
        """Per-env bool: True when the green cube is currently inside the drum."""
        green = self.scene[GREEN_CUBE_KEY]
        drum_pos = _get_world_pos(self.scene[DRUM_KEY], env=self)
        return _in_upright_cylinder(green.data.root_pos_w, drum_pos, _DRUM_GEOM)

    # ------------------------------------------------------------------
    # Override _step_impl to update latched flags
    # ------------------------------------------------------------------

    def step(self, action: torch.Tensor):
        """Standard step with grasp- and place-latch update after physics."""
        obs, reward, terminated, time_outs, extras = super().step(action)
        # Update latched flags only for envs that are still running.
        # Terminated/timed-out envs have already been reset inside
        # super().step(), so their post-reset state must not re-latch.
        still_running = ~(terminated | time_outs)
        self._was_grasped[still_running] |= self.grasp_active[still_running]
        self._was_placed[still_running] |= (
            self._green_in_drum() & self._was_grasped
        )[still_running]

        # Latch task_done: cube placed OR (had cube, released, now far)
        green = self.scene[GREEN_CUBE_KEY]
        gc_pos = self._grasp_center_pos()
        dist = torch.norm(green.data.root_pos_w - gc_pos, dim=-1)
        cube_far = dist > 0.20
        done_now = self._was_placed | (
            self._was_grasped & (~self.grasp_active) & cube_far
        )
        self._task_done[still_running] |= done_now[still_running]

        # Latch red_left_belt: red cube went off conveyor
        red = self.scene["red_cube"]
        red_local = red.data.root_pos_w - self.scene.env_origins
        red_active = (red_local[:, 0] < 50.0)  # not parked
        red_off = red_active & (
            (red_local[:, 1] < _CONVEYOR_BOUNDS.y_min)
            | (red_local[:, 1] > _CONVEYOR_BOUNDS.y_max)
            | (red_local[:, 2] < _CONVEYOR_BOUNDS.z_min)
        )
        self._red_left_belt[still_running] |= red_off[still_running]

        return obs, reward, terminated, time_outs, extras

    # ------------------------------------------------------------------
    # Reset custom state for specific envs
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int]):
        env_ids_t = (
            torch.tensor(env_ids, device=self.device, dtype=torch.long)
            if not isinstance(env_ids, torch.Tensor)
            else env_ids
        )

        # ── Collect episode-level metrics BEFORE reset clears state ───
        grasp_rate = torch.tensor(0.0, device=self.device)
        mean_ep_len = torch.tensor(0.0, device=self.device)
        red_on_belt_rate = torch.tensor(1.0, device=self.device)
        if len(env_ids_t) > 0:
            grasp_rate = self._was_grasped[env_ids_t].float().mean()
            place_rate = self._was_placed[env_ids_t].float().mean()
            mean_ep_len = self.episode_length_buf[env_ids_t].float().mean()
            red_on_belt_rate = (~self._red_left_belt[env_ids_t]).float().mean()

        result = super()._reset_idx(env_ids)
        self._was_grasped[env_ids_t] = False
        self._was_placed[env_ids_t] = False
        self._task_done[env_ids_t] = False
        self._red_left_belt[env_ids_t] = False

        # ── Inject custom scalars AFTER super (which creates extras["log"]) ──
        self.extras["log"]["Metrics/grasp_rate"] = grasp_rate
        self.extras["log"]["Metrics/place_success_rate"] = place_rate
        self.extras["log"]["Metrics/mean_episode_length"] = mean_ep_len
        self.extras["log"]["Metrics/red_on_conveyor_rate"] = red_on_belt_rate
        return result
