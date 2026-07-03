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
# Metric gates for the stable-hold latch.  The speed gate is 0.25 m/s and the
# criterion is WINDOWED (≥ WINDOW_FRAC of the last WINDOW steps), not
# consecutive: the PD arm holding the cloth has ~0.18 m/s residual EE sway
# with jitter spikes, so a consecutive-steps latch at 0.20 m/s reported 0 %
# for a policy that measurably parks the shirt 5–8 cm from the pose for 60 %
# of the episode (see diag_shirt_pick_policy.py, run 2026-07-03).
PRESENT_VEL_THRESHOLD = 0.25   # m/s — EE speed gate (metric)
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

        # Stable-hold presented latch (windowed).
        robot = self.scene["robot"]
        d = torch.norm(self.shirt_grasp_point_w - self._present_target, dim=-1)
        ee_speed = robot.data.body_lin_vel_w[:, self._ee_body_idx, :].norm(dim=-1)
        p_now = (
            self.grasp_active
            & (d < PRESENT_DIST_THRESHOLD)
            & (ee_speed < PRESENT_VEL_THRESHOLD)
        )
        self._present_ring[:, self._ring_idx] = p_now
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
        if len(env_ids_t) > 0:
            present_rate = self._was_presented[env_ids_t].float().mean()
            drop_rate = self._was_dropped[env_ids_t].float().mean()
        result = super()._reset_idx(env_ids)
        self._was_presented[env_ids_t] = False
        self._present_ring[env_ids_t] = False
        self._drop_event[env_ids_t] = 0.0
        self._was_dropped[env_ids_t] = False
        self._prev_attached[env_ids_t] = False
        self.extras["log"]["Metrics/present_rate"] = present_rate
        self.extras["log"]["Metrics/drop_rate"] = drop_rate
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
