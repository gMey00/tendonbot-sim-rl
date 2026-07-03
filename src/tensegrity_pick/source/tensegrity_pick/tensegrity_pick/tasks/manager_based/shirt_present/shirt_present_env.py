# shirt_present_env.py
#
# Task 2 of the cloth-sorting pipeline: the second robot grasps a second
# holding point on the hanging shirt and stretches it for front/back camera
# assessment.
#
# Initial states: the shirt hangs from a static solver anchor at the
# presentation pose (slot 1), pinned at ONE RANDOM particle patch — restored
# from the cached hanging-state bank (``scripts/generate_hanging_bank.py``),
# standing in for the retrieving robot's grip.  Slot 0 is the learning arm's
# own deterministic attachment grasp (two-attachment stretch, Stage-0
# de-risked through tautness ratio 1.15).
#
# Task MDP (naive two-grasp presentation heuristic):
#   * the deterministic grasp targets the LOWEST hanging point
#     (``shirt_grasp_point_w`` override — the literature-standard second
#     grasp: Maitin-Shepard 2010, Doumanoglou 2014)
#   * per-step task state: inter-grasp tautness ratio (patch-centroid
#     separation / flat-rest separation, the Stage-0-validated definition)
#     and projected-silhouette coverage in the inspection-camera plane
#     (``shared/cloth_metrics.py``)
#   * windowed presented latch (>= PRESENT_WINDOW_FRAC of the last
#     PRESENT_WINDOW steps — consecutive-step latches are too brittle) with
#     the speed gate on the CLOTH CENTROID, not the EE (shirt_pick Phase-3
#     lesson: residual PD sway at raised postures is 0.24-0.27 m/s while the
#     hanging garment low-pass filters to 0.06-0.20)
#   * drop_event one-shot + was_dropped metric (shirt_pick pattern)
#   * ``snapshot_terminal_states`` hook for the Task-2 -> Task-3 bank
# Open task work (see doc/TODO.md):
#   * later: initialize from the Task-1 terminal-state bank instead of the
#     idealized random-point hang (skill-chaining distribution shift)

from __future__ import annotations

from typing import Sequence

import torch

from ..shared.cloth_metrics import flat_silhouette_area, silhouette_coverage
from ..shared.cloth_sorting_env import ClothSortingEnvBase
from ..shared.cloth_sorting_scene_cfg import HANGING_BANK_PATH, PRESENTATION_POS
from ..shared.proj_base_scene_cfg import DRUM_HEIGHT_M

# Particles within this radius of the anchor are pinned (pad-sized, matches
# the validated ATTACH_WELD_RADIUS).
HOLDER_ANCHOR_RADIUS = 0.07
# Only restore hang states short enough to clear the drum tops: the reusable
# drum at (0.15, 1.0) sits directly under the presentation pose (0.15, 0.90),
# so a full-length drape (up to ~0.86 m) would dip into it.
MAX_HANG_DRAPE = PRESENTATION_POS[2] - DRUM_HEIGHT_M - 0.04

# ── Success predicate ────────────────────────────────────────────────
# Tautness band: taut but not overstretched.  Lower edge / coverage threshold
# are CALIBRATED from measured distributions (raw hang vs scripted stretch,
# scripts/model_validation/baseline_shirt_present.py) — see the tracking
# report before changing.  Upper edge 1.10 keeps a safety margin below the
# Stage-0-validated stretch limit (stable through 1.15, untested beyond).
STRETCH_BAND = (0.90, 1.10)
# Fraction of the flat one-sided area the camera-plane silhouette must
# recover.  Placeholder until the baseline calibration (raw hang expected
# ~0.3-0.5; scripted stretches measure the achievable ceiling).
COVERAGE_THRESHOLD = 0.55
# Cloth-centroid speed gate (NOT the EE — see the class docstring note).
PRESENT_VEL_THRESHOLD = 0.20   # m/s
PRESENT_WINDOW = 60            # steps (1 s @ 60 Hz)
PRESENT_WINDOW_FRAC = 0.8      # fraction of the window that must qualify


class ShirtPresentEnv(ClothSortingEnvBase):
    """Regrasp the lowest point and stretch the hanging shirt for inspection."""

    # Slot 0 = the learning arm's deterministic attachment grasp (enabled —
    # the base env's attach/hold/detach machinery, retargeted to the lowest
    # point via ``shirt_grasp_point_w``).  Slot 1 = the static holder anchor.
    enable_hand_grasp = True

    # Random-particle hang states (missing file → centre-hang fallback below).
    hanging_bank_path = HANGING_BANK_PATH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n, dev = self.num_envs, self.device
        # Static holder anchor at the presentation pose (world coords per env).
        self._anchor_pos = self.scene.env_origins + torch.tensor(
            PRESENTATION_POS, device=dev, dtype=torch.float32
        ).unsqueeze(0)

        # Reference area for silhouette coverage (flat one-sided rest shape).
        self._ref_area = flat_silhouette_area(self._cloth.flat_rest_pos)

        # Per-step task-state buffers (recomputed after every physics step —
        # rewards/observations consume the values from the END of the
        # previous step, the same one-step delay as every cloth quantity).
        self._stretch_buf = torch.zeros(n, device=dev)
        self._coverage_buf = torch.zeros(n, device=dev)

        # Windowed presented latch + drop bookkeeping (shirt_pick pattern).
        self._present_ring = torch.zeros(n, PRESENT_WINDOW, dtype=torch.bool, device=dev)
        self._ring_idx = 0
        self._was_presented = torch.zeros(n, dtype=torch.bool, device=dev)
        self._drop_event = torch.zeros(n, device=dev)
        self._was_dropped = torch.zeros(n, dtype=torch.bool, device=dev)
        self._prev_attached = torch.zeros(n, dtype=torch.bool, device=dev)

    # ------------------------------------------------------------------
    # Grasp target: the LOWEST hanging point (naive second-grab heuristic)
    # ------------------------------------------------------------------

    @property
    def shirt_grasp_point_w(self) -> torch.Tensor:
        """The lowest hanging point — the deterministic attach target."""
        return self.shirt_lowest_point_w

    # ------------------------------------------------------------------
    # Task-state accessors (consumed by mdp/rewards.py and the baseline)
    # ------------------------------------------------------------------

    @property
    def holder_attached(self) -> torch.Tensor:
        """Per-env bool: the holder anchor (slot 1) still pins the cloth."""
        return self._cloth.is_attached_slot(1)

    @property
    def stretch_ratio(self) -> torch.Tensor:
        """Inter-grasp tautness ``[N]`` — 0.0 until both attachments hold."""
        return self._stretch_buf

    @property
    def coverage(self) -> torch.Tensor:
        """Camera-plane silhouette coverage of the flat area ``[N]``."""
        return self._coverage_buf

    @property
    def drop_event(self) -> torch.Tensor:
        """One-shot ``[N]``: 1.0 the step an established hand grasp was lost."""
        return self._drop_event

    @property
    def was_presented(self) -> torch.Tensor:
        return self._was_presented

    @property
    def presented_now(self) -> torch.Tensor:
        """Instantaneous success predicate ``[N]`` (pre-latch).

        both attachments ∧ taut-but-not-overstretched ∧ silhouette coverage
        above threshold ∧ cloth centroid still.
        """
        cloth_speed = self._cloth.centroid_vel_w.norm(dim=-1)
        stretch_ok = (
            (self._stretch_buf >= STRETCH_BAND[0])
            & (self._stretch_buf <= STRETCH_BAND[1])
        )
        return (
            self.grasp_active
            & self.holder_attached
            & stretch_ok
            & (self._coverage_buf >= COVERAGE_THRESHOLD)
            & (cloth_speed < PRESENT_VEL_THRESHOLD)
        )

    # ------------------------------------------------------------------
    # Task-state computation
    # ------------------------------------------------------------------

    def _slot_patch_centroids(self, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Current & flat-rest centroids of one slot's attached patch.

        Returns ``(cur_w [N,3], rest [N,3])``; envs whose slot holds nothing
        get the all-particle means (callers must gate on attachment).
        """
        mask = self._cloth._attach_mask[slot].float()            # [N, P]
        cnt = mask.sum(dim=1, keepdim=True).clamp(min=1.0)       # [N, 1]
        cur = torch.einsum("np,npc->nc", mask, self._cloth.nodal_pos_w) / cnt
        rest = (mask @ self._cloth.flat_rest_pos) / cnt          # [N, 3]
        return cur, rest

    def _update_task_state(self) -> None:
        """Recompute the stretch/coverage buffers from the current cloth state.

        Stretch ratio: separation of the two attached patch centroids over
        their separation in ``flat_rest_pos`` — exactly the definition the
        Stage-0 two-attachment stretch test validated stable through 1.15.
        """
        self._coverage_buf = silhouette_coverage(
            self._cloth.nodal_pos_w, self._ref_area, view_axis=1,
        )
        both = self.grasp_active & self.holder_attached
        if both.any():
            cur0, rest0 = self._slot_patch_centroids(0)
            cur1, rest1 = self._slot_patch_centroids(1)
            dist = torch.norm(cur0 - cur1, dim=-1)
            rest = torch.norm(rest0 - rest1, dim=-1).clamp(min=1e-6)
            self._stretch_buf = torch.where(
                both, dist / rest, torch.zeros_like(dist),
            )
        else:
            self._stretch_buf = torch.zeros_like(self._stretch_buf)

    # ------------------------------------------------------------------
    # Cloth reset: hang from the holder anchor
    # ------------------------------------------------------------------

    def _reset_cloth(self, env_ids: torch.Tensor) -> None:
        """Hang the shirt from the holder anchor, pinned at one random point.

        Primary path: restore a relaxed random-particle hang from the cached
        hanging-state bank (slot 1 anchored at ``PRESENTATION_POS``).
        Fallback without a bank: teleport the flat sheet to the pose and pin
        its centre patch (it drapes into an idealized centre-hang — NOT
        settled, so early-episode swing is larger than with the bank).
        TODO(pipeline): eventually sample from the Task-1 terminal-state bank.
        """
        n = env_ids.numel()
        if n == 0:
            return
        if self._reset_cloth_hanging_from_bank(
            env_ids, self._anchor_pos, slot=1, radius=HOLDER_ANCHOR_RADIUS,
            max_drape=MAX_HANG_DRAPE,
        ):
            return
        origins = self.scene.env_origins[env_ids]
        centroids = origins.clone()
        centroids[:, 0] = origins[:, 0] + PRESENTATION_POS[0]
        centroids[:, 1] = origins[:, 1] + PRESENTATION_POS[1]
        centroids[:, 2] = origins[:, 2] + PRESENTATION_POS[2]
        yaw = torch.zeros(n, device=self.device)
        self._cloth.reset_randomized(env_ids, centroids, yaw)
        self._cloth.update()
        self._cloth.attach(env_ids, self._anchor_pos, HOLDER_ANCHOR_RADIUS, slot=1)

    # ------------------------------------------------------------------
    # Step hook: hand grasp (base machinery) + keep the holder pinned
    # ------------------------------------------------------------------

    def _update_grasp(self) -> None:
        # Base: drive the gripper and attach/hold/detach the hand grasp
        # (slot 0) at ``shirt_grasp_point_w`` = the lowest hanging point.
        super()._update_grasp()
        # Hold the pinned patch at the static presentation anchor (slot 1).
        self._cloth.hold(self._anchor_pos, slot=1)

    # ------------------------------------------------------------------
    # Step / reset
    # ------------------------------------------------------------------

    def step(self, action: torch.Tensor):
        obs, reward, terminated, time_outs, extras = super().step(action)
        still_running = ~(terminated | time_outs)

        self._update_task_state()

        # Drop detection: the hand grasp existed after the previous step but
        # is gone now (the task never legitimately releases).
        released = self._prev_attached & ~self.grasp_active
        self._drop_event = released.float()
        self._was_dropped |= released & still_running
        self._prev_attached = self.grasp_active.clone()

        # Windowed presented latch.
        self._present_ring[:, self._ring_idx] = self.presented_now
        self._ring_idx = (self._ring_idx + 1) % PRESENT_WINDOW
        window_frac = self._present_ring.float().mean(dim=1)
        self._was_presented |= (
            (window_frac >= PRESENT_WINDOW_FRAC) & still_running
        )
        return obs, reward, terminated, time_outs, extras

    def _reset_idx(self, env_ids: Sequence[int]):
        env_ids_t = (
            torch.tensor(env_ids, device=self.device, dtype=torch.long)
            if not isinstance(env_ids, torch.Tensor)
            else env_ids
        )
        present_rate = torch.tensor(0.0, device=self.device)
        drop_rate = torch.tensor(0.0, device=self.device)
        final_cov = torch.tensor(0.0, device=self.device)
        final_stretch = torch.tensor(0.0, device=self.device)
        if len(env_ids_t) > 0:
            present_rate = self._was_presented[env_ids_t].float().mean()
            drop_rate = self._was_dropped[env_ids_t].float().mean()
            final_cov = self._coverage_buf[env_ids_t].mean()
            final_stretch = self._stretch_buf[env_ids_t].mean()

        result = super()._reset_idx(env_ids)

        self._was_presented[env_ids_t] = False
        self._present_ring[env_ids_t] = False
        self._drop_event[env_ids_t] = 0.0
        self._was_dropped[env_ids_t] = False
        self._prev_attached[env_ids_t] = False
        # Refresh the task-state buffers so reset-step observations are
        # consistent with the restored hang (coverage of the raw hang,
        # stretch 0 while ungrasped).
        self._update_task_state()

        self.extras["log"]["Metrics/present_rate"] = present_rate
        self.extras["log"]["Metrics/drop_rate"] = drop_rate
        self.extras["log"]["Metrics/final_coverage"] = final_cov
        self.extras["log"]["Metrics/final_stretch_ratio"] = final_stretch
        return result

    # ------------------------------------------------------------------
    # Task-2 → Task-3 terminal-state bank hook
    # ------------------------------------------------------------------

    def snapshot_terminal_states(self) -> dict:
        """Capture the current per-env state for the shirt_distribute bank.

        Mirrors ``ShirtPickEnv.snapshot_terminal_states`` but includes BOTH
        grasp states (hand slot 0 + holder slot 1) so Task 3 can restore the
        bimanual configuration.  Filtering to ``was_presented`` envs is the
        caller's choice.
        """
        self._cloth.update()
        origins = self.scene.env_origins
        return {
            "pos": (self._cloth.nodal_pos_w - origins.unsqueeze(1)).cpu(),
            "vel": self._cloth.nodal_vel_w.cpu(),
            "attach_mask_hand": self._cloth._attach_mask[0].cpu(),
            "attach_mask_holder": self._cloth._attach_mask[1].cpu(),
            "grasp_point_local": (self.shirt_grasp_point_w - origins).cpu(),
            "anchor_local": (self._anchor_pos - origins).cpu(),
            "stretch_ratio": self._stretch_buf.cpu(),
            "coverage": self._coverage_buf.cpu(),
            "presented": self._was_presented.cpu(),
            "joint_pos": self.scene["robot"].data.joint_pos.cpu(),
        }
