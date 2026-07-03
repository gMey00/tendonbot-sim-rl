# shirt_distribute_env.py
#
# Task 3 of the cloth-sorting pipeline: the second robot throws/places the
# classified shirt into the commanded bin.  Goal-conditioned (TossingBot-
# style): the condition label only selects WHICH bin position is observed —
# the per-episode target bin is resampled uniformly so "nearest bin" and
# "correct bin" diverge and the policy cannot ignore the goal.
#
# STUB status — the shirt starts flat on the belt edge near the second robot;
# the deterministic highest-point grasp is inherited functional from
# ``ClothSortingEnvBase``.  Open pipeline work (see doc/TODO.md):
#   * initialize grasped, from the Task-2 terminal-state bank
#   * landing-in-correct-bin sparse reward + release shaping (port the
#     anti-hover release-event design from shirt_place)
#   * optional bin-layout randomization beyond the 3 fixed drums

from __future__ import annotations

from typing import Sequence

import torch

from ..shared.cloth_sorting_env import ClothSortingEnvBase
from ..shared.cloth_sorting_scene_cfg import DRUM_NAMES, DRUM_POSITIONS
from ..shared.gripper_cfg import BinCylinder, get_world_pos, in_upright_cylinder

# Same interior-depth cylinder as shirt_place (draped cloth rarely reaches the
# drum bottom, so the success volume spans the full interior).
_DRUM_GEOM = BinCylinder(radius=0.547 * 0.5, height=0.85)
# Fraction of (released) cloth particles inside the target drum that counts as
# a correct distribution (same threshold as shirt_place placements).
DISTRIBUTE_FRACTION_THRESHOLD = 0.15


class ShirtDistributeEnv(ClothSortingEnvBase):
    """Throw/place the shirt into the commanded condition bin."""

    # Belt edge closest to the second robot — within its reach envelope.
    shirt_rest_xy = (0.75, 0.30)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_bin = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._was_distributed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------
    # Goal conditioning
    # ------------------------------------------------------------------

    @property
    def target_bin_pos_w(self) -> torch.Tensor:
        """World position of each env's commanded bin ``[N, 3]``."""
        drums = torch.stack(
            [get_world_pos(self.scene[name], env=self) for name in DRUM_NAMES], dim=1
        )  # [N, 3(bins), 3]
        idx = self._target_bin.view(-1, 1, 1).expand(-1, 1, 3)
        return drums.gather(1, idx).squeeze(1)

    def _target_fraction_in_bin(self) -> torch.Tensor:
        """Fraction of cloth particles inside the commanded drum ``[N]``."""
        center = self.target_bin_pos_w
        pts = self._cloth.nodal_pos_w
        inside = in_upright_cylinder(pts, center, _DRUM_GEOM)
        return inside.float().mean(dim=1)

    # ------------------------------------------------------------------
    # Step / reset
    # ------------------------------------------------------------------

    def step(self, action: torch.Tensor):
        obs, reward, terminated, time_outs, extras = super().step(action)

        # Correct distribution: shirt released with enough cloth in the
        # commanded bin (mirrors shirt_place's release-required placement).
        released = ~self.grasp_active
        frac = self._target_fraction_in_bin() * released.float()
        distributed = (frac > DISTRIBUTE_FRACTION_THRESHOLD) & self._was_grasped & released
        still_running = ~(terminated | time_outs)
        self._was_distributed[still_running] |= distributed[still_running]
        return obs, reward, terminated, time_outs, extras

    def _reset_idx(self, env_ids: Sequence[int]):
        env_ids_t = (
            torch.tensor(env_ids, device=self.device, dtype=torch.long)
            if not isinstance(env_ids, torch.Tensor)
            else env_ids
        )
        distribute_rate = torch.tensor(0.0, device=self.device)
        if len(env_ids_t) > 0:
            distribute_rate = self._was_distributed[env_ids_t].float().mean()
            # Resample the commanded bin BEFORE the parent reset so reset-step
            # observations already see the new goal.
            self._target_bin[env_ids_t] = torch.randint(
                len(DRUM_POSITIONS), (len(env_ids_t),), device=self.device
            )
        result = super()._reset_idx(env_ids)
        self._was_distributed[env_ids_t] = False
        self.extras["log"]["Metrics/distribute_success_rate"] = distribute_rate
        return result
