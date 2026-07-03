# shirt_present_env.py
#
# Task 2 of the cloth-sorting pipeline: the second robot grasps a second
# holding point on the hanging shirt and stretches it for front/back camera
# assessment.
#
# Initial states: the shirt hangs from a static solver anchor at the
# presentation pose (slot 1), pinned at ONE RANDOM particle patch — restored
# from the cached hanging-state bank (``scripts/generate_hanging_bank.py``),
# standing in for the retrieving robot's grip.  Slot 0 stays reserved for the
# learning arm's own grasp (two-attachment stretch, Stage-0 de-risked).
# Open task work (see doc/TODO.md):
#   * enable the learning arm's grasp at the LOWEST hanging point (naive
#     second-grab heuristic) + stretch MDP
#   * tautness proxy (inter-grasp distance / rest distance), projected
#     silhouette-coverage success metric (shared/cloth_metrics.py) from the
#     inspection-camera viewpoints
#   * later: initialize from the Task-1 terminal-state bank instead of the
#     idealized random-point hang (skill-chaining distribution shift)

from __future__ import annotations

from typing import Sequence

import torch

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


class ShirtPresentEnv(ClothSortingEnvBase):
    """Stretch the hanging shirt for inspection (bimanual, second arm learns)."""

    # The holder anchor uses slot 1; the learning robot's deterministic grasp
    # (slot 0) stays off until the task's regrasp MDP is implemented.
    enable_hand_grasp = False

    # Random-particle hang states (missing file → centre-hang fallback below).
    hanging_bank_path = HANGING_BANK_PATH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Static holder anchor at the presentation pose (world coords per env).
        self._anchor_pos = self.scene.env_origins + torch.tensor(
            PRESENTATION_POS, device=self.device, dtype=torch.float32
        ).unsqueeze(0)

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
        # Slot 1 = holder/partner anchor; slot 0 stays reserved for the
        # learning arm's own grasp (multi-slot attachments, Stage-0 de-risk).
        self._cloth.attach(env_ids, self._anchor_pos, HOLDER_ANCHOR_RADIUS, slot=1)

    # ------------------------------------------------------------------
    # Step hook: keep the holder anchor pinned
    # ------------------------------------------------------------------

    def _update_grasp(self) -> None:
        robot = self.scene["robot"]
        self._drive_gripper(robot)
        # Hold the pinned patch at the static presentation anchor (slot 1).
        self._cloth.hold(self._anchor_pos, slot=1)
