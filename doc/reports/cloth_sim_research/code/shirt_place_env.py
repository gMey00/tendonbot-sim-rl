# shirt_place_env.py
#
# Custom environment for the shirt placement task.
#
# Integrates PBD particle cloth (via ``ClothObject``) with a kinematic
# rigid-body proxy (``shirt_proxy``) that tracks the cloth centroid.
# All reward / observation functions read from the proxy, so they work
# without modification.
#
# Lifecycle:
#   1. Startup event spawns cloth mesh + applies PBD schemas (before sim.reset)
#   2. super().__init__() → scene creation, cloning, sim.reset()
#   3. ClothObject() acquires ParticleClothView (GPU tensor access)
#   4. Each step: cloth.update() → sync proxy to centroid → rewards
#   5. Each reset: cloth.reset(env_ids)
#
# Also adds latched ``was_grasped`` and ``was_placed`` flags.

from __future__ import annotations

from typing import Sequence

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils.math import quat_apply

from .shirt_place_scene_cfg import (
    CONVEYOR_SURFACE_HEIGHT_M,
    SHIRT_BELT_CLEARANCE_M,
    SHIRT_CLOTH_CFG,
    SHIRT_FOLD_PROB,
    SHIRT_SPAWN_BOX,
    UPSTREAM_SETTLE_POS,
)
from .mdp.rewards import GRASP_CENTER_LOCAL_Z, BinCylinder, _in_upright_cylinder, _get_world_pos
from ..shared.cloth_object import ClothObject


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

    Extends :class:`ManagerBasedRLEnv` with:
      - PBD particle cloth via ``ClothObject``
      - Kinematic proxy synced to cloth centroid
      - Latched ``was_grasped`` / ``was_placed`` flags
    """

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        # ── Robot body / joint indices ────────────────────────────────
        robot: Articulation = self.scene["robot"]
        ee_body_name = next(n for n in EE_BODY_CANDIDATES if n in robot.body_names)
        self._ee_body_idx: int = robot.body_names.index(ee_body_name)
        self._finger_joint_idx: int = robot.joint_names.index(FINGER_JOINT_NAME)

        # ── Cloth object (GPU tensor access) ──────────────────────────
        self._cloth = ClothObject(
            cfg=SHIRT_CLOTH_CFG,
            num_envs=self.num_envs,
            device=self.device,
        )

        # ── Latched flags + cloth-aware success metric ────────────────
        self._was_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._was_placed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Peak fraction of cloth particles inside the drum during the episode —
        # a cloth-aware success signal (centroid-in-drum can be true while most
        # of the shirt hangs outside).
        self._max_drum_fraction = torch.zeros(self.num_envs, device=self.device)

        # One-off pre-settle: collapse the raw closed garment mesh into a
        # relaxed flat sheet and adopt it as the canonical rest shape, so
        # per-reset placement no longer triggers a self-collision pop.
        self._presettle_cloth()

    # ------------------------------------------------------------------
    # One-off pre-settle of the cloth rest shape
    # ------------------------------------------------------------------

    def _presettle_cloth(self, steps: int = 320) -> None:
        """Settle one shirt flat on the upstream (free) belt and reuse the shape.

        Settling away from the robot — on the empty upstream conveyor — gives a
        clean relaxed sheet without the gripper interfering, which is then the
        canonical rest shape teleported to the work area at each reset.  The
        garment is released from a height so it pancakes flat as it lands instead
        of freezing in the source mesh's puffy 3D shape.
        """
        try:
            ids = torch.arange(self.num_envs, device=self.device)
            origins = self.scene.env_origins
            drop = self._cloth.cfg.pbd_params.presettle_drop_height_m
            z_local = (
                CONVEYOR_SURFACE_HEIGHT_M + drop - self._cloth.flat_rest_min_z
            )
            centroids = origins.clone()
            centroids[:, 0] = origins[:, 0] + UPSTREAM_SETTLE_POS[0]
            centroids[:, 1] = origins[:, 1] + UPSTREAM_SETTLE_POS[1]
            centroids[:, 2] = origins[:, 2] + z_local
            yaws = torch.zeros(self.num_envs, device=self.device)
            self._cloth.reset_randomized(ids, centroids, yaws)

            dt = self.cfg.sim.dt
            for _ in range(steps):
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt)
            self._cloth.recompute_flat_rest_from_current()
        except Exception as exc:  # pragma: no cover - safety net
            import logging
            logging.getLogger(__name__).warning(
                "Cloth pre-settle skipped (%s); using geometric rest shape.", exc
            )

    # ------------------------------------------------------------------
    # Cloth access (for external use)
    # ------------------------------------------------------------------

    @property
    def cloth(self) -> ClothObject:
        return self._cloth

    # ------------------------------------------------------------------
    # Randomized cloth placement
    # ------------------------------------------------------------------

    def _reset_cloth(self, env_ids: torch.Tensor) -> None:
        """Lay the shirt flat on the belt at a randomized pose (+ optional fold)."""
        n = env_ids.numel()
        if n == 0:
            return
        dev = self.device
        box = SHIRT_SPAWN_BOX
        x = torch.empty(n, device=dev).uniform_(*box.x_range)
        y = torch.empty(n, device=dev).uniform_(*box.y_range)
        yaw = torch.empty(n, device=dev).uniform_(*box.yaw_range)

        origins = self.scene.env_origins[env_ids]
        # World centroid: random XY on the belt; Z so the shirt's lowest point
        # rests just above the belt surface.
        z_local = CONVEYOR_SURFACE_HEIGHT_M + SHIRT_BELT_CLEARANCE_M - self._cloth.flat_rest_min_z
        centroids = origins.clone()
        centroids[:, 0] = origins[:, 0] + x
        centroids[:, 1] = origins[:, 1] + y
        centroids[:, 2] = origins[:, 2] + z_local

        # Fold a fraction of resets for initial-state diversity (hinge at 30–60%).
        fold = torch.where(
            torch.rand(n, device=dev) < SHIRT_FOLD_PROB,
            torch.empty(n, device=dev).uniform_(0.3, 0.6),
            torch.full((n,), -1.0, device=dev),
        )
        self._cloth.reset_randomized(env_ids, centroids, yaw, fold)

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

    def _shirt_particle_fraction_in_drum(self) -> torch.Tensor:
        """Fraction of cloth particles inside the drum cylinder, per env (N,)."""
        drum_pos = _get_world_pos(self.scene[DRUM_KEY], env=self)  # [N, 3]
        pts = self._cloth.nodal_pos_w                              # [N, P, 3]
        inside = _in_upright_cylinder(pts, drum_pos, _DRUM_GEOM)   # [N, P] bool
        return inside.float().mean(dim=1)

    # ------------------------------------------------------------------
    # Step: update cloth → sync proxy → update latched flags
    # ------------------------------------------------------------------

    def _sync_proxy_to_cloth(self) -> None:
        """Teleport the kinematic proxy to the cloth centroid."""
        proxy: RigidObject = self.scene[SHIRT_PROXY_KEY]
        centroid = self._cloth.centroid_pos_w  # [N, 3]
        # Build root state: [pos(3), quat(4), lin_vel(3), ang_vel(3)] = 13
        root_state = torch.zeros(
            self.num_envs, 13, device=self.device, dtype=torch.float32,
        )
        root_state[:, :3] = centroid
        root_state[:, 3] = 1.0  # quat w=1 (identity)
        # lin_vel from cloth centroid velocity
        root_state[:, 7:10] = self._cloth.centroid_vel_w
        proxy.write_root_state_to_sim(root_state)

    def step(self, action: torch.Tensor):
        obs, reward, terminated, time_outs, extras = super().step(action)

        # Update cloth state from GPU buffers
        self._cloth.update()
        # Sync the kinematic proxy to the cloth centroid
        self._sync_proxy_to_cloth()

        still_running = ~(terminated | time_outs)
        self._was_grasped[still_running] |= self.grasp_active[still_running]
        self._was_placed[still_running] |= (
            self._shirt_in_drum() & self._was_grasped
        )[still_running]
        # Track the peak cloth-in-drum fraction (cloth-aware success metric).
        frac = self._shirt_particle_fraction_in_drum()
        self._max_drum_fraction[still_running] = torch.maximum(
            self._max_drum_fraction[still_running], frac[still_running]
        )
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
        drum_fraction = torch.tensor(0.0, device=self.device)
        if len(env_ids_t) > 0:
            grasp_rate = self._was_grasped[env_ids_t].float().mean()
            place_rate = self._was_placed[env_ids_t].float().mean()
            mean_ep_len = self.episode_length_buf[env_ids_t].float().mean()
            drum_fraction = self._max_drum_fraction[env_ids_t].mean()

        result = super()._reset_idx(env_ids)
        self._was_grasped[env_ids_t] = False
        self._was_placed[env_ids_t] = False
        self._max_drum_fraction[env_ids_t] = 0.0

        # Lay the shirt flat at a randomized pose, then sync the proxy/cloth
        # buffers so observations on the reset step are consistent.
        self._reset_cloth(env_ids_t)
        self._cloth.update()
        self._sync_proxy_to_cloth()

        self.extras["log"]["Metrics/grasp_rate"] = grasp_rate
        self.extras["log"]["Metrics/place_success_rate"] = place_rate
        self.extras["log"]["Metrics/mean_episode_length"] = mean_ep_len
        self.extras["log"]["Metrics/shirt_in_drum_fraction"] = drum_fraction
        return result
