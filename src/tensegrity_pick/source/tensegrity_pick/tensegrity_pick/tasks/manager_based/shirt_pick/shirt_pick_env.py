# shirt_pick_env.py
#
# Task 1 of the cloth-sorting pipeline: the retrieving robot picks a shirt
# from the conveyor at its highest point and holds it in front of the
# inspection camera (presentation pose).
#
# The generic cloth machinery (deterministic highest-point grasp, proxy sync,
# bank/flat reset) is inherited from ``ClothSortingEnvBase``.  This class adds
# the task's success/failure bookkeeping:
#   * ``drop_event``      — one-shot per-env flag the step a grasp is lost
#                           (the task never releases; consumed by the reward)
#   * stable-hold latch   — ``present_rate`` counts only holds that stay
#                           within PRESENT_DIST of the pose, slower than
#                           PRESENT_VEL, for HOLD_STEPS consecutive steps
#   * terminal-state snapshotting hook (Task-1 → Task-2 bank) — see
#     ``snapshot_terminal_states``.

from __future__ import annotations

from typing import Sequence

import torch

from ..shared.cloth_sorting_env import ClothSortingEnvBase
from ..shared.cloth_sorting_scene_cfg import CRUMPLED_BANK_PATH, PRESENTATION_POS

PRESENT_DIST_THRESHOLD = 0.15  # m — grasp point to presentation pose
# Stable-hold latch: WINDOWED (≥ WINDOW_FRAC of the last WINDOW steps), and
# the speed gate is on the CLOTH CENTROID, not the EE (both measured
# decisions, see diag_shirt_pick_policy.py):
#   * consecutive-steps latches are too brittle — the PD arm always has
#     residual EE sway with jitter spikes;
#   * at the raised presentation posture (elbow-extended lever) the EE sway
#     is 0.24–0.27 m/s — above any sane EE gate — while the hanging garment
#     low-pass filters it to 0.06–0.20 m/s.  The camera inspects the cloth,
#     so the cloth's stillness is the honest criterion.
PRESENT_VEL_THRESHOLD = 0.20   # m/s — cloth centroid speed gate
PRESENT_WINDOW = 60            # steps (1 s @ 60 Hz)
PRESENT_WINDOW_FRAC = 0.8      # fraction of the window that must qualify


class ShirtPickEnv(ClothSortingEnvBase):
    """Pick the shirt from the belt and present it to the inspection camera."""

    # Initial states come from the cached crumpled bank when it exists
    # (generate with scripts/generate_crumpled_bank.py); missing file →
    # ClothSortingEnvBase falls back to the flat lay with a warning.
    crumpled_bank_path = CRUMPLED_BANK_PATH
    bank_fraction = 1.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n, dev = self.num_envs, self.device
        self._was_presented = torch.zeros(n, dtype=torch.bool, device=dev)
        # Ring buffer over the last PRESENT_WINDOW steps of the per-step
        # presented predicate (windowed stable-hold latch).
        self._present_ring = torch.zeros(n, PRESENT_WINDOW, dtype=torch.bool, device=dev)
        self._ring_idx = 0
        self._drop_event = torch.zeros(n, device=dev)
        self._was_dropped = torch.zeros(n, dtype=torch.bool, device=dev)
        self._prev_attached = torch.zeros(n, dtype=torch.bool, device=dev)
        self._present_target = self.scene.env_origins + torch.tensor(
            PRESENTATION_POS, device=dev, dtype=torch.float32
        ).unsqueeze(0)
        # ── S2 grasp-point head state (inert while the head is not configured:
        # zero offset ⇒ shirt_grasp_point_w falls through to the base highest
        # point, so stage-1 tasks/checkpoints behave identically) ────────────
        self._grasp_offset_xy = torch.zeros(n, 2, device=dev)
        self._present_latch_event = torch.zeros(n, device=dev)
        self._hold_region = torch.full((n,), -1, dtype=torch.long, device=dev)
        self._pred_coverage = torch.zeros(n, device=dev)
        self._head_enabled = getattr(self.cfg.actions, "grasp_offset", None) is not None

    # ------------------------------------------------------------------
    # Reward accessors
    # ------------------------------------------------------------------

    @property
    def drop_event(self) -> torch.Tensor:
        """One-shot ``[N]``: 1.0 the step an established grasp was lost.

        Set after the physics/grasp update in :meth:`step`; like shirt_place's
        ``release_event`` it is consumed by the reward manager on the NEXT
        step (one-step delay, dt-scaled one-shot weight applies).
        """
        return self._drop_event

    @property
    def was_presented(self) -> torch.Tensor:
        return self._was_presented

    @property
    def present_latch_event(self) -> torch.Tensor:
        """One-shot ``[N]``: 1.0 the step the stable-present latch fires.

        Like ``drop_event`` it is consumed by the reward manager on the NEXT
        step (dt-scaled one-shot weight applies) — used by the S2 head's
        ``coverage_terminal_bonus``.
        """
        return self._present_latch_event

    # ------------------------------------------------------------------
    # S2 grasp-point head: surface-snapped, policy-offset grasp target
    # ------------------------------------------------------------------

    @property
    def shirt_grasp_point_w(self) -> torch.Tensor:
        """Grasp target ``[N, 3]`` — the stage-1 highest point, optionally
        shifted by the S2 head's xy offset and snapped to the local cloth top.

        The snap (mean of the top-``SNAP_TOPK`` particles within
        ``SNAP_RADIUS`` of the shifted xy; radius grows to the nearest
        particle when the offset points off the pile) keeps the target on the
        VISIBLE UPPER SURFACE — the depth-camera contract.  Once the grasp is
        latched (or while the offset is zero) this falls through to the base
        highest point, so the deterministic attach trigger mechanics and all
        post-grasp reward semantics are untouched.
        """
        base = self._cloth.highest_point_w
        use_head = (~self.grasp_active) & (self._grasp_offset_xy.abs().sum(dim=-1) > 1e-9)
        if not use_head.any():
            return base
        from .mdp.grasp_head import SNAP_RADIUS, SNAP_TOPK

        pts = self._cloth.nodal_pos_w                                  # [N, P, 3]
        tgt_xy = base[:, :2] + self._grasp_offset_xy                   # [N, 2]
        d_xy = (pts[:, :, :2] - tgt_xy.unsqueeze(1)).norm(dim=-1)      # [N, P]
        radius = torch.clamp(
            d_xy.min(dim=1, keepdim=True).values + 0.01, min=SNAP_RADIUS
        )
        z = pts[:, :, 2].masked_fill(d_xy > radius, float("-inf"))
        top_z, top_i = z.topk(min(SNAP_TOPK, z.shape[1]), dim=1)       # [N, k]
        valid = torch.isfinite(top_z)                                  # [N, k]
        sel = torch.gather(pts, 1, top_i.unsqueeze(-1).expand(-1, -1, 3))
        cnt = valid.sum(dim=1, keepdim=True).clamp(min=1)
        snapped = (sel * valid.unsqueeze(-1)).sum(dim=1) / cnt
        return torch.where(use_head.unsqueeze(-1), snapped, base)

    # ------------------------------------------------------------------
    # Step / reset
    # ------------------------------------------------------------------

    def step(self, action: torch.Tensor):
        obs, reward, terminated, time_outs, extras = super().step(action)
        still_running = ~(terminated | time_outs)

        # Drop detection: the grasp existed after the previous step but is
        # gone now (the pick task never legitimately releases).
        released = self._prev_attached & ~self.grasp_active
        self._drop_event = released.float()
        self._was_dropped |= released & still_running
        self._prev_attached = self.grasp_active.clone()

        # Stable-hold presented latch (windowed, cloth-centroid speed gate).
        d = torch.norm(self.shirt_grasp_point_w - self._present_target, dim=-1)
        cloth_speed = self._cloth.centroid_vel_w.norm(dim=-1)
        p_now = (
            self.grasp_active
            & (d < PRESENT_DIST_THRESHOLD)
            & (cloth_speed < PRESENT_VEL_THRESHOLD)
        )
        self._present_ring[:, self._ring_idx] = p_now
        self._ring_idx = (self._ring_idx + 1) % PRESENT_WINDOW
        window_frac = self._present_ring.float().mean(dim=1)
        newly_latched = (
            (window_frac >= PRESENT_WINDOW_FRAC) & still_running & ~self._was_presented
        )
        self._present_latch_event = newly_latched.float()
        self._was_presented |= newly_latched

        # S2 head bookkeeping: freeze the hold region + predicted coverage at
        # the latch (consumed by coverage_terminal_bonus / Metrics logging).
        if self._head_enabled and newly_latched.any():
            from .mdp.grasp_head import _Markers, hold_region_sym8

            mk = _Markers.get(self.device)
            region = hold_region_sym8(self)
            ids = newly_latched
            self._hold_region[ids] = region[ids]
            self._pred_coverage[ids] = torch.where(
                region[ids] >= 0,
                mk.cov_by_region[region[ids].clamp(min=0)],
                torch.zeros_like(self._pred_coverage[ids]),
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
        if len(env_ids_t) > 0:
            present_rate = self._was_presented[env_ids_t].float().mean()
            drop_rate = self._was_dropped[env_ids_t].float().mean()
        result = super()._reset_idx(env_ids)
        self._was_presented[env_ids_t] = False
        self._present_ring[env_ids_t] = False
        self._drop_event[env_ids_t] = 0.0
        self._was_dropped[env_ids_t] = False
        self._prev_attached[env_ids_t] = False
        self._present_latch_event[env_ids_t] = 0.0
        self.extras["log"]["Metrics/present_rate"] = present_rate
        self.extras["log"]["Metrics/drop_rate"] = drop_rate
        if self._head_enabled:
            from .mdp.grasp_head import _Markers

            mk = _Markers.get(self.device)
            region = self._hold_region[env_ids_t]
            latched = region >= 0
            high_value = torch.zeros((), device=self.device)
            pred_cov = torch.zeros((), device=self.device)
            if len(env_ids_t) > 0:
                high_value = (
                    mk.high_value_mask[region.clamp(min=0)] & latched
                ).float().mean()
                pred_cov = torch.where(
                    latched, self._pred_coverage[env_ids_t], torch.zeros_like(region, dtype=torch.float)
                ).sum() / latched.sum().clamp(min=1)
            self.extras["log"]["Metrics/high_value_hold_rate"] = high_value
            self.extras["log"]["Metrics/pred_coverage"] = pred_cov
            self._hold_region[env_ids_t] = -1
            self._pred_coverage[env_ids_t] = 0.0
            self._grasp_offset_xy[env_ids_t] = 0.0
        return result

    # ------------------------------------------------------------------
    # Task-1 → Task-2 terminal-state bank hook
    # ------------------------------------------------------------------

    def snapshot_terminal_states(self) -> dict:
        """Capture the current per-env state for the shirt_present bank.

        Returns env-local particle positions/velocities, the grasp-point
        world offset from the presentation pose, the attachment mask and the
        presented flag — everything shirt_present needs to reset from real
        Task-1 terminal states (see doc/TODO.md, skill-chaining banks).
        Filtering to ``was_presented`` envs is the caller's choice.
        """
        self._cloth.update()
        origins = self.scene.env_origins
        return {
            "pos": (self._cloth.nodal_pos_w - origins.unsqueeze(1)).cpu(),
            "vel": self._cloth.nodal_vel_w.cpu(),
            "attach_mask": self._cloth._attach_mask[0].cpu(),
            "grasp_point_local": (self.shirt_grasp_point_w - origins).cpu(),
            "presented": self._was_presented.cpu(),
            "joint_pos": self.scene["robot"].data.joint_pos.cpu(),
        }
