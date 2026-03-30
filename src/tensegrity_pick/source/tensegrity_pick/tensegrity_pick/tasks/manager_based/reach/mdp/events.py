"""Custom event functions for the reach task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def clamp_infinite_joint_limits(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    fallback_range: float = 6.2832,  # 2π
    max_range: float | None = None,
):
    """Replace infinite or excessively wide joint position limits with finite defaults.

    Some USDs (e.g. Kinova Gen3) define continuous joints with [-inf, inf]
    limits.  Actions that map to joint limits (e.g.
    ``JointPositionToLimitsAction``) produce NaN when the limits are
    infinite.  This event replaces infinite limits with
    ``default_pos ± fallback_range/2``.

    When *max_range* is provided, **any** joint whose current range exceeds
    *max_range* is clamped — not just infinite ones.  This is useful for
    robots like the UR10e whose finite limits (±2π) are too wide for
    efficient RL exploration.

    Should be added as a ``mode="reset"`` event so that limits are written
    before the first action processing.  Subsequent calls are idempotent.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    limits = asset.data.joint_pos_limits.clone()
    defaults = asset.data.default_joint_pos

    # Identify joints with infinite or excessively large limits
    lower = limits[..., 0]
    upper = limits[..., 1]
    threshold = max_range if max_range is not None else 4 * torch.pi
    bad_mask = ~torch.isfinite(lower) | ~torch.isfinite(upper) | ((upper - lower) > threshold)

    if not bad_mask.any():
        return

    half = fallback_range / 2.0
    # Use the first environment's defaults (broadcasted)
    new_lower = defaults - half
    new_upper = defaults + half

    limits[..., 0] = torch.where(bad_mask, new_lower, lower)
    limits[..., 1] = torch.where(bad_mask, new_upper, upper)

    asset.write_joint_position_limit_to_sim(limits)
