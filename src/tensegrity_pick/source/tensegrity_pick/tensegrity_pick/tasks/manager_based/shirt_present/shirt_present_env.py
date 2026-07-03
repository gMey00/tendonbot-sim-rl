# shirt_present_env.py
#
# Task 2 of the cloth-sorting pipeline: the second robot grasps a second
# holding point on the hanging shirt and stretches it for front/back camera
# assessment.
#
# STUB status — the shirt hangs from a static solver anchor at the
# presentation pose, standing in for the retrieving robot's grip (the passive
# ``holder_robot`` in the scene is visual only for now).  Open pipeline work
# (see doc/TODO.md):
#   * initialize from the Task-1 terminal-state bank instead of the idealized
#     centre-hang (skill-chaining distribution shift is the central risk)
#   * SECOND attachment for the learning robot — ``ClothObject`` currently
#     manages one attachment set per env, and the anchor slot is used by the
#     holder; supporting robot-grasp + holder-anchor simultaneously is the
#     core Task-2 implementation work (PBD two-attachment stability!)
#   * lowest-point regrasp target, tautness proxy (inter-grasp distance /
#     rest distance), projected-coverage reward from the camera viewpoints

from __future__ import annotations

from typing import Sequence

import torch

from ..shared.cloth_sorting_env import ClothSortingEnvBase
from ..shared.cloth_sorting_scene_cfg import PRESENTATION_POS

# Particles within this radius of the anchor are pinned (pad-sized, matches
# the validated ATTACH_WELD_RADIUS).
HOLDER_ANCHOR_RADIUS = 0.07


class ShirtPresentEnv(ClothSortingEnvBase):
    """Stretch the hanging shirt for inspection (bimanual, second arm learns)."""

    # The single ClothObject attachment slot is used by the holder anchor, so
    # the learning robot's deterministic grasp stays off in the stub.
    enable_hand_grasp = False

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
        """Teleport the flat shirt to the presentation pose and pin its centre.

        The sheet drapes under gravity into a centre-hung garment — an
        idealized stand-in for the Task-1 terminal state ("held at the former
        highest point").  TODO(pipeline): sample from the cached Task-1
        terminal-state bank instead.
        """
        n = env_ids.numel()
        if n == 0:
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
