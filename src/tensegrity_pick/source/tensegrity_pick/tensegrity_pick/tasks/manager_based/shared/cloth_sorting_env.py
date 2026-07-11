# cloth_sorting_env.py
#
# Robot-agnostic base environment for the cloth-sorting pipeline tasks
# (shirt_pick, shirt_present, shirt_distribute).
#
# Extracts the *generic* cloth machinery proven in the shirt_place task
# (``shirt_place/shirt_place_env.py``) so all three pipeline tasks share it:
#
#   * PBD particle cloth via ``ClothObject`` (GPU tensor access)
#   * one-off cloth pre-settle on the free upstream belt
#   * kinematic ``shirt_proxy`` synced to the cloth centroid each step
#   * kinematic Robotiq 2F-140 four-bar drive (command-coherent close/open)
#   * deterministic highest-point attachment grasp (ANCHOR/WELD)
#   * latched ``was_grasped`` flag + grasp-rate logging
#
# Task-specific logic (rewards, success metrics, state-bank resets, second
# attachment for the bimanual stretch) lives in the task env subclasses.

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, replace
from typing import Sequence

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils.math import quat_apply

from .cloth_metrics import depth_layer_count_in_ball, load_present_markers
from .cloth_object import ClothObject, ClothObjectCfg
from .gripper_cfg import (
    FINGER_JOINT_CLOSE_POS,
    FINGER_TIP_CLOSED_DIST,
    FINGER_TIP_OPEN_DIST,
    tip_frame_sign,
)
from .cloth_sorting_scene_cfg import (
    CLOTH_SORTING_SHIRT_CFG,
    CONVEYOR_SURFACE_HEIGHT_M,
    SHIRT_BELT_CLEARANCE_M,
    SHIRT_REST_XY,
    UPSTREAM_SETTLE_POS,
)

# EE body per robot family (first match wins): tensegrity, UR/F140, Kinova.
EE_BODY_CANDIDATES = ("tool_link_0", "robotiq_base_link", "end_effector_link")
SHIRT_PROXY_KEY = "shirt_proxy"
FINGER_JOINT_NAME = "finger_joint"

# ── Deterministic attachment-grasp thresholds ─────────────────────────
# Values validated in shirt_place (see shirt_place_env.py for the tuning
# rationale: tight trigger + pad-sized weld radius grip only the pinched
# cluster at the finger tip).
ATTACH_TRIGGER_DIST = 0.10
ATTACH_WELD_RADIUS = 0.07
FINGER_CMD_CLOSE = 0.40
FINGER_CMD_OPEN = 0.40
# Capture radius for hanging-bank anchors (same pad-sized patch as the
# validated weld radius — "one random point" = one pad-sized particle patch).
HANG_ANCHOR_RADIUS = 0.07

# Robotiq 2F-140 four-bar mimic ratios relative to ``finger_joint`` (same
# table as shirt_place_env.py / run_shirt_validation.py — all arms carry the
# identical gripper).  The passive joints have no drive stiffness, so the env
# kinematically forces the linkage to a coherent pose each step.
GRIPPER_MIMIC = {
    "right_outer_knuckle_joint": 1.0,
    "left_outer_finger_joint": 0.0,
    "right_outer_finger_joint": 0.0,
    "left_inner_finger_joint": -1.0,
    "right_inner_finger_joint": -1.0,
    "left_inner_finger_pad_joint": 1.0,
    "right_inner_finger_pad_joint": 1.0,
}
GRIPPER_KIN_RATE = 4.0  # rad/s — smooth, visible close in ≈0.2 s


@dataclass
class GraspFidelityCfg:
    """Opt-in grasp-fidelity gates for the deterministic attachment grasp.

    Stage-2 S5 (research report §2-Q5): training and the stage-1 policies
    keep the idealized deterministic attachment (SoftGym lineage — "decouple
    grasping from planning"); these EVALUATION-MODE gates make the grasp
    physically honest so checkpoint rankings can be validated against a
    realistic grasp model.  **Both gates default OFF**, and with both off the
    grasp path is bit-identical to the ungated code.

    Pre-condition gate (deterministic, re-checked every attach-eligible step):
      (a) approach alignment — finger approach axis within
          ``max_approach_angle_deg`` of the local cloth surface normal
          (PCA over the particles inside the weld ball);
      (b) finger clearance — the finger tip must stay
          ``min_tip_clearance_m`` above the belt surface (no rigid jam
          pinching against the conveyor);
      (c) layer estimate — at most ``max_layers`` separated cloth layers in
          the weld ball along the approach axis (multi-layer pinch is the
          documented parallel-jaw failure: UniFolding "cannot grasp a single
          layer reliably"; DRAPER multi-layer rates).

    Stochastic misgrasp/slip gate (one draw per grasp ATTEMPT — a failed
    attempt latches until the gripper reopens, so holding the close command
    does not re-roll the dice every step):
      * on attach, failure probability keyed to grasp location and layers,
        calibrated to DRAPER / DeepCloth-ROB (arXiv 2409.15159): real Franka
        parallel-jaw misgrasp 16–22 % — border/edge grasps at the low end
        (edges/corners are the easy case: the tweezer baseline exceeds 90 %
        there), mid-panel at the high end, multi-layer pinches worse;
      * during hold, the anchor slips when a hang-weight load proxy
        (garment mass × fraction hanging below the grasp × (g + upward tip
        acceleration)) exceeds a per-attempt capacity draw
        ``N(slip_capacity_mean_n, slip_capacity_std_n)`` — the printed
        2F-140 grip force is 10–125 N, but a fabric pinch slips far below
        that; the default 4 N mean makes a fully-hanging 0.30 kg shirt
        (≈ 2.9 N static) hold on most draws and slip under dynamic pulls.

    Every number is a field so the M4 ranking harness can sweep them; draws
    use a dedicated generator seeded with ``seed`` (deterministic evals).
    """

    precondition_gate: bool = False
    stochastic_gate: bool = False
    # (a) approach alignment
    max_approach_angle_deg: float = 60.0
    min_normal_particles: int = 4     # fewer in the ball → defer to capture
    # (b) finger clearance above the belt surface
    min_tip_clearance_m: float = 0.005
    # (c) layer estimate at the grasp point
    layer_radius_m: float = ATTACH_WELD_RADIUS
    layer_gap_m: float = 0.015
    max_layers: int = 2
    # stochastic misgrasp (per attempt)
    p_misgrasp_border: float = 0.16
    p_misgrasp_mid_panel: float = 0.22
    p_misgrasp_multilayer: float = 0.50
    border_dist_threshold_m: float = 0.03
    # slip during hold (per-attempt capacity draw, Newtons)
    slip_capacity_mean_n: float = 4.0
    slip_capacity_std_n: float = 1.5
    slip_capacity_min_n: float = 0.5
    gravity_m_s2: float = 9.81
    seed: int = 0


class ClothSortingEnvBase(ManagerBasedRLEnv):
    """Shared manager-based RL env for the cloth-sorting pipeline tasks.

    Subclasses may override:
      * ``shirt_rest_xy``   — flat-lay position of the shirt at reset
      * ``enable_hand_grasp`` — set False to disable the learning robot's
        deterministic attachment grasp (e.g. while the cloth is anchored to a
        passive holder in shirt_present's stub)
      * ``_reset_cloth``    — e.g. sample from a cached state bank instead of
        the flat lay (see doc/TODO.md, crumpled-state bank)
    """

    cloth_cfg: ClothObjectCfg = CLOTH_SORTING_SHIRT_CFG
    shirt_rest_xy: tuple[float, float] = SHIRT_REST_XY
    enable_hand_grasp: bool = True

    # ── Crumpled-state bank (Task-1 initial-state distribution) ───────
    # Path to a bank generated by ``scripts/generate_crumpled_bank.py``.
    # None (or a missing file) → the flat lay is used.  ``bank_fraction``
    # mixes bank states with flat lays per reset (curriculum knob).
    # Cheap augmentation is applied at load time: random yaw + random
    # mirror (a reflection preserves all PBD constraint distances).
    crumpled_bank_path: str | None = None
    bank_fraction: float = 1.0
    bank_xy_jitter: float = 0.03

    # ── Grasp-fidelity gates (Stage-2 S5, evaluation-mode, DEFAULT OFF) ──
    # Class-level default; each instance gets its own copy at init.  Eval
    # scripts opt in by mutating the instance, e.g.
    # ``env.grasp_fidelity.stochastic_gate = True`` (see GraspFidelityCfg).
    grasp_fidelity: GraspFidelityCfg = GraspFidelityCfg()

    # ── Hanging-state bank (Task-2/3 initial-state distribution) ──────
    # Path to a bank generated by ``scripts/generate_hanging_bank.py``:
    # relaxed hang shapes anchored at ONE random particle patch, stored
    # anchor-centred (anchor point at the local origin) so a task can
    # restore them at any world anchor — see
    # ``_reset_cloth_hanging_from_bank``.  None → helper returns False and
    # the task falls back to its own reset.
    hanging_bank_path: str | None = None

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        # ── Robot body / joint indices ────────────────────────────────
        robot: Articulation = self.scene["robot"]
        ee_body_name = next(n for n in EE_BODY_CANDIDATES if n in robot.body_names)
        self._ee_body_idx: int = robot.body_names.index(ee_body_name)
        # Local-Z direction toward the finger tips differs per EE frame
        # (tool_link_0: −Z, robotiq_base_link: +Z — measured, see gripper_cfg).
        self._tip_sign: float = tip_frame_sign(ee_body_name)
        self._finger_joint_idx: int = robot.joint_names.index(FINGER_JOINT_NAME)
        self._mimic_ids: list[int] = [robot.joint_names.index(j) for j in GRIPPER_MIMIC]
        self._mimic_ratios: torch.Tensor = torch.tensor(
            [GRIPPER_MIMIC[j] for j in GRIPPER_MIMIC],
            device=self.device, dtype=torch.float32,
        )

        # ── Cloth object (GPU tensor access) ──────────────────────────
        self._cloth = ClothObject(
            cfg=self.cloth_cfg,
            num_envs=self.num_envs,
            device=self.device,
        )

        # ── Latched flags ─────────────────────────────────────────────
        self._was_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ── Grasp-fidelity gate state (inert while both gates are off) ─
        self.grasp_fidelity = replace(type(self).grasp_fidelity)  # own copy
        self._gf_rng = torch.Generator(device=self.device)
        self._gf_rng.manual_seed(int(self.grasp_fidelity.seed))
        self._misgrasp_latch = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._slip_capacity = torch.full((self.num_envs,), float("inf"), device=self.device)
        self._gf_prev_tip = torch.zeros(self.num_envs, 3, device=self.device)
        self._gf_prev_vel = torch.zeros(self.num_envs, 3, device=self.device)
        # Two-stage validity: tip finite-difference velocity is valid one call
        # after a reset, acceleration one call later (avoids a spurious
        # vel/dt acceleration spike on the second post-reset step).
        self._gf_prev_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._gf_vel_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._gf_border_dist: torch.Tensor | None = None
        # Cumulative event counters for the M3/M4 rigs (python ints).
        self.gf_event_counts = {
            "attempts": 0, "blocked_precondition": 0, "misgrasps": 0, "slips": 0,
        }

        # One-off pre-settle: collapse the raw garment mesh into a relaxed
        # flat sheet and adopt it as the canonical rest shape.
        self._presettle_cloth()

        # ── Cloth state banks (optional) ──────────────────────────────
        self._bank_pos: torch.Tensor | None = None
        self._bank_vel: torch.Tensor | None = None
        data = self._load_state_bank(self.crumpled_bank_path, "Crumpled",
                                     "scripts/generate_crumpled_bank.py")
        if data is not None:
            self._bank_pos = data["pos"].to(self.device)
            self._bank_vel = data["vel"].to(self.device)

        self._hang_pos: torch.Tensor | None = None
        self._hang_vel: torch.Tensor | None = None
        self._hang_drape: torch.Tensor | None = None
        data = self._load_state_bank(self.hanging_bank_path, "Hanging",
                                     "scripts/generate_hanging_bank.py")
        if data is not None:
            self._hang_pos = data["pos"].to(self.device)
            self._hang_vel = data["vel"].to(self.device)
            if "drape" in data:
                self._hang_drape = data["drape"].to(self.device)

    def _load_state_bank(self, path: str | None, label: str, generator: str) -> dict | None:
        """Load a particle state bank ``.pt``; None + warning on any mismatch."""
        log = logging.getLogger(__name__)
        if not path:
            return None
        if not os.path.isfile(path):
            log.warning("%s bank not found at %s — falling back "
                        "(generate it with %s).", label, path, generator)
            return None
        data = torch.load(path, map_location=self.device)
        if data["pos"].shape[1] != self._cloth.num_particles:
            log.warning(
                "%s bank %s has %d particles/state but the cloth has %d — bank ignored.",
                label, path, data["pos"].shape[1], self._cloth.num_particles,
            )
            return None
        log.info("%s bank loaded: %d states (%s)", label, data["pos"].shape[0], path)
        return data

    # ------------------------------------------------------------------
    # Cloth access
    # ------------------------------------------------------------------

    @property
    def cloth(self) -> ClothObject:
        return self._cloth

    @property
    def grasp_active(self) -> torch.Tensor:
        """Per-env bool: True while the cloth is held by the attachment grasp."""
        return self._cloth.is_attached

    @property
    def was_grasped(self) -> torch.Tensor:
        return self._was_grasped

    @property
    def shirt_grasp_point_w(self) -> torch.Tensor:
        """Deterministic grasp target ``[N, 3]`` — default: the highest region.

        ``_update_grasp`` triggers the attach when the finger tip closes near
        THIS point; subclasses retarget it by overriding (e.g. shirt_present:
        the LOWEST hanging point, the literature-standard second grasp).
        """
        return self._cloth.highest_point_w

    @property
    def shirt_min_z_w(self) -> torch.Tensor:
        """World Z of the lowest cloth particle per env ``[N]``."""
        return self._cloth.nodal_pos_w[:, :, 2].min(dim=1).values

    @property
    def shirt_lowest_point_w(self) -> torch.Tensor:
        """World position of the lowest cloth particle ``[N, 3]``.

        The literature-standard second-grasp target on a hanging garment
        (Maitin-Shepard 2010; Doumanoglou 2014) — used by shirt_present.
        """
        pts = self._cloth.nodal_pos_w                      # [N, P, 3]
        idx = pts[:, :, 2].argmin(dim=1)                   # [N]
        return pts[torch.arange(pts.shape[0], device=pts.device), idx]

    # ------------------------------------------------------------------
    # One-off pre-settle of the cloth rest shape (validated in shirt_place)
    # ------------------------------------------------------------------

    def _presettle_cloth(self, steps: int = 320) -> None:
        """Settle one shirt flat on the free upstream belt, adopt as rest shape."""
        try:
            ids = torch.arange(self.num_envs, device=self.device)
            origins = self.scene.env_origins
            drop = self._cloth.cfg.pbd_params.presettle_drop_height_m
            z_local = CONVEYOR_SURFACE_HEIGHT_M + drop - self._cloth.flat_rest_min_z
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
    # Cloth reset (subclasses override, e.g. cached state banks)
    # ------------------------------------------------------------------

    def _reset_cloth(self, env_ids: torch.Tensor) -> None:
        """Reset the shirt on the belt at ``shirt_rest_xy``.

        With a loaded crumpled-state bank, a fraction ``bank_fraction`` of the
        resets restore a cached crumpled state (with yaw + mirror augmentation
        and a small XY jitter); the rest use the flat lay.  Without a bank,
        all resets are flat lays.
        """
        n = env_ids.numel()
        if n == 0:
            return
        use_bank = torch.zeros(n, dtype=torch.bool, device=self.device)
        if self._bank_pos is not None and self.bank_fraction > 0.0:
            use_bank = torch.rand(n, device=self.device) < self.bank_fraction

        bank_ids = env_ids[use_bank]
        flat_ids = env_ids[~use_bank]

        if flat_ids.numel() > 0:
            origins = self.scene.env_origins[flat_ids]
            z_local = (
                CONVEYOR_SURFACE_HEIGHT_M + SHIRT_BELT_CLEARANCE_M
                - self._cloth.flat_rest_min_z
            )
            centroids = origins.clone()
            centroids[:, 0] = origins[:, 0] + self.shirt_rest_xy[0]
            centroids[:, 1] = origins[:, 1] + self.shirt_rest_xy[1]
            centroids[:, 2] = origins[:, 2] + z_local
            yaw = torch.zeros(flat_ids.numel(), device=self.device)
            self._cloth.reset_randomized(flat_ids, centroids, yaw)

        if bank_ids.numel() > 0:
            self._reset_cloth_from_bank(bank_ids)

    def _reset_cloth_from_bank(self, env_ids: torch.Tensor) -> None:
        """Restore sampled crumpled bank states (yaw + mirror augmented).

        Bank states are env-local and xy-centred on their centroid (z stays
        belt-relative), so they can be placed at any belt position/yaw.  Both
        particle positions AND velocities are written — the deterministic
        cache-restore path validated by
        ``scripts/model_validation/test_cache_restore_reset.py``.
        """
        k = env_ids.numel()
        dev = self.device
        idx = torch.randint(0, self._bank_pos.shape[0], (k,), device=dev)
        pos = self._bank_pos[idx].clone()                       # [k, P, 3]
        vel = self._bank_vel[idx].clone()

        # Mirror augmentation (x → −x reflection preserves PBD distances).
        mirror = (torch.rand(k, device=dev) < 0.5).view(k, 1)
        pos[..., 0] = torch.where(mirror, -pos[..., 0], pos[..., 0])
        vel[..., 0] = torch.where(mirror, -vel[..., 0], vel[..., 0])

        # Yaw augmentation about the vertical axis.
        yaw = torch.rand(k, device=dev) * 2.0 * torch.pi
        cy, sy = torch.cos(yaw).view(k, 1), torch.sin(yaw).view(k, 1)
        for t in (pos, vel):
            x, y = t[..., 0].clone(), t[..., 1].clone()
            t[..., 0] = cy * x - sy * y
            t[..., 1] = sy * x + cy * y

        # Place at the task rest position (+ small XY jitter).
        origins = self.scene.env_origins[env_ids]
        jitter = (torch.rand(k, 2, device=dev) - 0.5) * 2.0 * self.bank_xy_jitter
        pos[..., 0] += (origins[:, 0] + self.shirt_rest_xy[0] + jitter[:, 0]).view(k, 1)
        pos[..., 1] += (origins[:, 1] + self.shirt_rest_xy[1] + jitter[:, 1]).view(k, 1)
        pos[..., 2] += origins[:, 2].view(k, 1)

        self._cloth.write_nodal_state_to_sim(
            pos.reshape(k, -1), vel.reshape(k, -1), env_ids,
        )

    def _reset_cloth_hanging_from_bank(
        self,
        env_ids: torch.Tensor,
        anchor_pos_w: torch.Tensor,
        slot: int = 1,
        radius: float = HANG_ANCHOR_RADIUS,
        max_drape: float | None = None,
    ) -> bool:
        """Restore relaxed hanging states pinned at per-env world anchor points.

        Bank states are anchor-centred (the pinned particle patch sits at the
        local origin), so any world anchor works: a static holder anchor
        (shirt_present, slot 1) or the learning robot's own finger tip
        (shirt_distribute, slot 0 — remember to close the gripper with
        ``_force_gripper_closed`` so ``_update_grasp`` does not immediately
        release).  Applies yaw + mirror augmentation about the anchor's
        vertical axis, writes pos AND vel (deterministic cache-restore path),
        then re-attaches the given slot by ``radius`` capture at the anchor.

        Args:
            env_ids:      ``[k]`` env indices to reset.
            anchor_pos_w: ``[num_envs, 3]`` full-size world anchors (indexed
                          internally, same convention as ``ClothObject.attach``).
            max_drape:    only sample states whose hang length below the
                          anchor is ≤ this (m) — keeps long hangs clear of
                          scene geometry below the anchor (e.g. the drum tops
                          under the presentation pose).

        Returns False when no hanging bank is loaded (caller must fall back).
        """
        if self._hang_pos is None:
            return False
        k = env_ids.numel()
        if k == 0:
            return True
        dev = self.device
        pool = torch.arange(self._hang_pos.shape[0], device=dev)
        if max_drape is not None and self._hang_drape is not None:
            pool = pool[self._hang_drape <= max_drape]
            if pool.numel() == 0:
                logging.getLogger(__name__).warning(
                    "Hanging bank: no state with drape <= %.2f m — using the "
                    "full bank.", max_drape,
                )
                pool = torch.arange(self._hang_pos.shape[0], device=dev)
        idx = pool[torch.randint(0, pool.numel(), (k,), device=dev)]
        pos = self._hang_pos[idx].clone()                        # [k, P, 3]
        vel = self._hang_vel[idx].clone()

        # Mirror (x → −x through the anchor) + yaw about the anchor vertical
        # axis — both preserve PBD constraint distances (anchor at origin).
        mirror = (torch.rand(k, device=dev) < 0.5).view(k, 1)
        pos[..., 0] = torch.where(mirror, -pos[..., 0], pos[..., 0])
        vel[..., 0] = torch.where(mirror, -vel[..., 0], vel[..., 0])
        yaw = torch.rand(k, device=dev) * 2.0 * torch.pi
        cy, sy = torch.cos(yaw).view(k, 1), torch.sin(yaw).view(k, 1)
        for t in (pos, vel):
            x, y = t[..., 0].clone(), t[..., 1].clone()
            t[..., 0] = cy * x - sy * y
            t[..., 1] = sy * x + cy * y

        pos += anchor_pos_w[env_ids].unsqueeze(1)
        self._cloth.write_nodal_state_to_sim(
            pos.reshape(k, -1), vel.reshape(k, -1), env_ids,
        )
        self._cloth.update()
        self._cloth.attach(env_ids, anchor_pos_w, radius, slot=slot)
        return True

    # ------------------------------------------------------------------
    # Robotiq gripper helpers (identical hardware across all arms)
    # ------------------------------------------------------------------

    def _finger_tip_pos(self) -> torch.Tensor:
        """World position of the dynamic finger tip ``[N, 3]``.

        Tip distance calibrated on the tensegrity's ``tool_link_0``, projected
        along the frame-correct local-Z direction (``self._tip_sign``) so the
        same code serves the UR/Kinova ``robotiq_base_link`` frame — validated
        by scripts/model_validation/check_f140_finger_tip.py (the frames are
        co-located with antiparallel Z on every F140 arm).
        """
        robot: Articulation = self.scene["robot"]
        finger_pos = robot.data.joint_pos[:, self._finger_joint_idx]
        closure = torch.clamp(finger_pos / FINGER_JOINT_CLOSE_POS, 0.0, 1.0)
        tip_z = self._tip_sign * (
            FINGER_TIP_OPEN_DIST
            + closure * (FINGER_TIP_CLOSED_DIST - FINGER_TIP_OPEN_DIST)
        )
        ee_pos = robot.data.body_pos_w[:, self._ee_body_idx, :]
        ee_quat = robot.data.body_quat_w[:, self._ee_body_idx, :]
        offset = torch.stack(
            [torch.zeros_like(tip_z), torch.zeros_like(tip_z), tip_z], dim=-1
        )
        return ee_pos + quat_apply(ee_quat, offset)

    def _drive_gripper(self, robot: Articulation) -> None:
        """Kinematically drive the whole Robotiq linkage to its commanded pose.

        Same mechanism as shirt_place: rate-limit the finger toward the binary
        action's command, write the coherent four-bar pose, and align the PD
        targets so the actuators do not fight the kinematic write.
        """
        cmd = robot.data.joint_pos_target[:, self._finger_joint_idx]
        cmd = torch.clamp(cmd, 0.0, FINGER_JOINT_CLOSE_POS)
        cur = robot.data.joint_pos[:, self._finger_joint_idx]
        max_step = GRIPPER_KIN_RATE * self.cfg.sim.dt * self.cfg.decimation
        finger = cur + torch.clamp(cmd - cur, -max_step, max_step)

        ids = [self._finger_joint_idx] + self._mimic_ids
        ratios = torch.cat([torch.ones(1, device=self.device), self._mimic_ratios])
        pose = finger.unsqueeze(1) * ratios.unsqueeze(0)
        robot.write_joint_position_to_sim(pose, joint_ids=ids)
        robot.write_joint_velocity_to_sim(torch.zeros_like(pose), joint_ids=ids)
        robot.set_joint_position_target(pose, joint_ids=ids)

    def _force_gripper_closed(self, env_ids: torch.Tensor) -> None:
        """Write the CLOSED four-bar pose + PD targets for ``env_ids``.

        For tasks that reset with the shirt already in the gripper
        (shirt_distribute): without this, the default open pose (and open
        joint target) makes ``_update_grasp`` release the restored attachment
        on the first step.  Note the policy's gripper action overwrites the
        target from the next action step on — holding the grasp beyond reset
        is the policy's job.
        """
        robot: Articulation = self.scene["robot"]
        ids = [self._finger_joint_idx] + self._mimic_ids
        ratios = torch.cat([torch.ones(1, device=self.device), self._mimic_ratios])
        pose = FINGER_JOINT_CLOSE_POS * ratios.unsqueeze(0).repeat(env_ids.numel(), 1)
        robot.write_joint_position_to_sim(pose, joint_ids=ids, env_ids=env_ids)
        robot.write_joint_velocity_to_sim(torch.zeros_like(pose), joint_ids=ids, env_ids=env_ids)
        robot.set_joint_position_target(pose, joint_ids=ids, env_ids=env_ids)

    def _update_grasp(self) -> None:
        """Attach / hold / detach the deterministic grasp at ``shirt_grasp_point_w``.

        With ``grasp_fidelity`` gates enabled (evaluation-mode, default OFF —
        see :class:`GraspFidelityCfg`) the attach additionally passes the
        physical pre-conditions and/or a per-attempt misgrasp draw, and held
        anchors can slip under load.  Both flags off ⇒ bit-identical to the
        ungated path (no RNG consumed, no extra cloth reads).
        """
        robot: Articulation = self.scene["robot"]
        self._drive_gripper(robot)
        if not self.enable_hand_grasp:
            return

        if hasattr(robot.data, "joint_pos_target"):
            finger_cmd = robot.data.joint_pos_target[:, self._finger_joint_idx]
        else:
            finger_cmd = robot.data.joint_pos[:, self._finger_joint_idx]

        tip = self._finger_tip_pos()
        target = self.shirt_grasp_point_w
        near = torch.norm(tip - target, dim=-1) < ATTACH_TRIGGER_DIST
        closing = finger_cmd > FINGER_CMD_CLOSE
        opening = finger_cmd < FINGER_CMD_OPEN
        attached = self._cloth.is_attached

        gf = self.grasp_fidelity
        gates_on = gf.precondition_gate or gf.stochastic_gate

        to_attach = closing & near & (~attached)
        if gates_on:
            if gf.stochastic_gate:
                # A new attempt requires reopening after a failed one.
                self._misgrasp_latch &= ~opening
                to_attach &= ~self._misgrasp_latch
            if to_attach.any():
                to_attach = self._apply_grasp_gates(to_attach, tip, robot)
        if to_attach.any():
            self._cloth.attach(
                torch.nonzero(to_attach, as_tuple=False).squeeze(-1),
                tip, ATTACH_WELD_RADIUS,
            )
            if gf.stochastic_gate:
                # Per-attempt pinch capacity for the slip check (Newtons).
                fresh = to_attach & self._cloth.is_attached
                cap = gf.slip_capacity_mean_n + gf.slip_capacity_std_n * torch.randn(
                    self.num_envs, device=self.device, generator=self._gf_rng
                )
                self._slip_capacity = torch.where(
                    fresh, cap.clamp(min=gf.slip_capacity_min_n), self._slip_capacity
                )
        to_detach = attached & opening
        if to_detach.any():
            self._cloth.detach(torch.nonzero(to_detach, as_tuple=False).squeeze(-1))
        if gf.stochastic_gate:
            self._apply_slip_gate(tip)
        self._cloth.hold(tip)

    # ------------------------------------------------------------------
    # Grasp-fidelity gates (Stage-2 S5; see GraspFidelityCfg)
    # ------------------------------------------------------------------

    def _grasp_approach_axis(self, robot: Articulation) -> torch.Tensor:
        """Unit finger approach direction (EE local Z toward the tips) ``[N, 3]``."""
        ee_quat = robot.data.body_quat_w[:, self._ee_body_idx, :]
        local = torch.tensor(
            [0.0, 0.0, self._tip_sign], device=self.device
        ).expand(self.num_envs, 3)
        return quat_apply(ee_quat, local)

    def _grasp_border_dist(self, nearest_idx: torch.Tensor) -> torch.Tensor:
        """Border nearness (m) of grasp particles; +inf if markers unusable."""
        if self._gf_border_dist is None:
            try:
                bd = load_present_markers()["border_dist"]
                if bd.shape[0] != self._cloth.num_particles:
                    raise ValueError(
                        f"marker particle count {bd.shape[0]} != cloth "
                        f"{self._cloth.num_particles}"
                    )
                self._gf_border_dist = bd.to(self.device)
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Grasp gate: present_markers unusable (%s) — treating every "
                    "grasp as mid-panel.", exc,
                )
                self._gf_border_dist = torch.full(
                    (self._cloth.num_particles,), float("inf"), device=self.device
                )
        return self._gf_border_dist[nearest_idx]

    def _apply_grasp_gates(
        self,
        to_attach: torch.Tensor,
        tip: torch.Tensor,
        robot: Articulation | None,
        approach: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Filter attach candidates through the pre-condition / misgrasp gates.

        ``approach`` overrides the robot-derived finger approach axis — used
        by the robot-free validation rig (check_grasp_gates.py) to script
        top-down pinches through the exact gate code path.
        """
        gf = self.grasp_fidelity
        allow = to_attach.clone()
        self.gf_event_counts["attempts"] += int(to_attach.sum())

        pos = self._cloth.nodal_pos_w                          # [N, P, 3]
        if approach is None:
            approach = self._grasp_approach_axis(robot)        # [N, 3]
        layers = depth_layer_count_in_ball(
            pos, tip, gf.layer_radius_m, axis=approach, layer_gap=gf.layer_gap_m
        )

        if gf.precondition_gate:
            # (b) finger clearance above the belt surface (no rigid jam).
            surface_z = self.scene.env_origins[:, 2] + CONVEYOR_SURFACE_HEIGHT_M
            clearance_ok = tip[:, 2] - surface_z >= gf.min_tip_clearance_m
            # (a) approach alignment vs the local cloth surface normal (PCA
            # over the weld-ball particles; too few particles → defer to the
            # attach's own capture logic).
            rel = pos - tip.unsqueeze(1)
            inball = (rel.norm(dim=-1) <= gf.layer_radius_m)   # [N, P]
            w = inball.float()
            cnt = w.sum(dim=1)
            mean = (pos * w.unsqueeze(-1)).sum(dim=1) / cnt.clamp(min=1.0).unsqueeze(-1)
            d = (pos - mean.unsqueeze(1)) * w.unsqueeze(-1)
            cov = d.transpose(1, 2) @ d / cnt.clamp(min=1.0).view(-1, 1, 1)
            # Regularize so eigh is well-posed on degenerate/empty balls.
            cov = cov + 1e-12 * torch.eye(3, device=self.device).unsqueeze(0)
            normal = torch.linalg.eigh(cov).eigenvectors[..., 0]  # smallest eval
            cos = (approach * normal).sum(dim=-1).abs() / approach.norm(dim=-1).clamp(min=1e-9)
            align_ok = (cos >= math.cos(math.radians(gf.max_approach_angle_deg))) | (
                cnt < gf.min_normal_particles
            )
            # (c) single- vs multi-layer pinch.
            layer_ok = layers <= gf.max_layers
            passed = clearance_ok & align_ok & layer_ok
            self.gf_event_counts["blocked_precondition"] += int(
                (allow & ~passed).sum()
            )
            allow &= passed

        if gf.stochastic_gate and allow.any():
            nearest = (pos - tip.unsqueeze(1)).norm(dim=-1).argmin(dim=1)  # [N]
            border = self._grasp_border_dist(nearest) <= gf.border_dist_threshold_m
            p = torch.where(
                border,
                torch.full_like(tip[:, 0], gf.p_misgrasp_border),
                torch.full_like(tip[:, 0], gf.p_misgrasp_mid_panel),
            )
            p = torch.where(
                layers > gf.max_layers, p.clamp(min=gf.p_misgrasp_multilayer), p
            )
            draw = torch.rand(self.num_envs, device=self.device, generator=self._gf_rng)
            fail = allow & (draw < p)
            self._misgrasp_latch |= fail
            self.gf_event_counts["misgrasps"] += int(fail.sum())
            allow &= ~fail
        return allow

    def _apply_slip_gate(self, tip: torch.Tensor) -> None:
        """Detach held anchors whose hang-weight load exceeds their capacity."""
        gf = self.grasp_fidelity
        dt = self.cfg.sim.dt * self.cfg.decimation
        vel = (tip - self._gf_prev_tip) / dt
        acc = torch.where(
            self._gf_vel_valid.unsqueeze(-1),
            (vel - self._gf_prev_vel) / dt,
            torch.zeros_like(vel),
        )
        self._gf_vel_valid = self._gf_prev_valid.clone()
        self._gf_prev_vel = vel
        self._gf_prev_tip = tip.clone()
        self._gf_prev_valid[:] = True

        attached = self._cloth.is_attached
        if not attached.any():
            return
        pos = self._cloth.nodal_pos_w
        frac_below = (pos[:, :, 2] < tip[:, 2].unsqueeze(1)).float().mean(dim=1)
        mass = float(self._cloth.cfg.pbd_params.mass_kg)
        load = mass * frac_below * (gf.gravity_m_s2 + acc[:, 2]).clamp(min=0.0)
        slip = attached & (load > self._slip_capacity)
        if slip.any():
            self._cloth.detach(torch.nonzero(slip, as_tuple=False).squeeze(-1))
            self._misgrasp_latch |= slip       # regrasp needs a reopen cycle
            self.gf_event_counts["slips"] += int(slip.sum())

    # ------------------------------------------------------------------
    # Proxy sync
    # ------------------------------------------------------------------

    def _sync_proxy_to_cloth(self) -> None:
        """Teleport the kinematic proxy to the cloth centroid."""
        proxy: RigidObject = self.scene[SHIRT_PROXY_KEY]
        root_state = torch.zeros(self.num_envs, 13, device=self.device, dtype=torch.float32)
        root_state[:, :3] = self._cloth.centroid_pos_w
        root_state[:, 3] = 1.0
        root_state[:, 7:10] = self._cloth.centroid_vel_w
        proxy.write_root_state_to_sim(root_state)

    # ------------------------------------------------------------------
    # Step / reset
    # ------------------------------------------------------------------

    def step(self, action: torch.Tensor):
        obs, reward, terminated, time_outs, extras = super().step(action)

        self._cloth.update()
        self._update_grasp()
        self._sync_proxy_to_cloth()

        still_running = ~(terminated | time_outs)
        self._was_grasped[still_running] |= self.grasp_active[still_running]
        return obs, reward, terminated, time_outs, extras

    def _reset_idx(self, env_ids: Sequence[int]):
        env_ids_t = (
            torch.tensor(env_ids, device=self.device, dtype=torch.long)
            if not isinstance(env_ids, torch.Tensor)
            else env_ids
        )

        grasp_rate = torch.tensor(0.0, device=self.device)
        if len(env_ids_t) > 0:
            grasp_rate = self._was_grasped[env_ids_t].float().mean()

        result = super()._reset_idx(env_ids)
        self._was_grasped[env_ids_t] = False

        # Grasp-fidelity gate state does not leak across episodes.
        self._misgrasp_latch[env_ids_t] = False
        self._slip_capacity[env_ids_t] = float("inf")
        self._gf_prev_valid[env_ids_t] = False
        self._gf_vel_valid[env_ids_t] = False

        # Release attachments, reset the cloth, sync buffers so reset-step
        # observations are consistent.
        self._cloth.reset_attachment(env_ids_t)
        self._reset_cloth(env_ids_t)
        self._cloth.update()
        self._sync_proxy_to_cloth()

        self.extras["log"]["Metrics/grasp_rate"] = grasp_rate
        return result
