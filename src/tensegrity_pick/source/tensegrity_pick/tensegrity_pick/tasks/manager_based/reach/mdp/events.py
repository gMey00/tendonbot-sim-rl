"""Custom event functions for the reach task."""

from __future__ import annotations

import math
import re
import torch
from typing import TYPE_CHECKING

from pxr import Usd, UsdPhysics

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils.prims import clone

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


def _clamp_revolute_joint_limits_on_prim(
    root_prim: "Usd.Prim",
    joint_defaults: dict,
    fallback_range: float,
    max_range: float | None,
) -> int:
    """Clamp over-wide / infinite revolute-joint USD limits under *root_prim*.

    Applies the same ``default_pos ± fallback_range/2`` rule as
    :func:`clamp_infinite_joint_limits`, but directly on the USD attributes.
    USD revolute limits are authored in **degrees**; joint defaults are in
    radians, so the rule is applied in radians and converted.  Returns the
    number of joints clamped.
    """
    threshold_rad = max_range if max_range is not None else 4.0 * math.pi
    threshold_deg = math.degrees(threshold_rad)
    half_rad = fallback_range / 2.0

    def _default_rad(joint_name: str) -> float:
        if joint_name in joint_defaults:
            return joint_defaults[joint_name]
        for pattern, value in joint_defaults.items():
            if re.fullmatch(pattern, joint_name):
                return value
        return 0.0

    n_fixed = 0
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        lo = joint.GetLowerLimitAttr().Get()
        hi = joint.GetUpperLimitAttr().Get()
        unlimited = (
            lo is None or hi is None or not math.isfinite(lo) or not math.isfinite(hi)
        )
        too_wide = (not unlimited) and (abs(hi - lo) > threshold_deg)
        if not (unlimited or too_wide):
            continue
        d = _default_rad(prim.GetName())
        joint.CreateLowerLimitAttr().Set(math.degrees(d - half_rad))
        joint.CreateUpperLimitAttr().Set(math.degrees(d + half_rad))
        n_fixed += 1
    return n_fixed


# Process-global registry: usd_path -> (joint_defaults, fallback_range, max_range).
# Populated by :func:`register_joint_limit_clamp` at config-build time and read by
# the module-level spawner below.  A module-level function + registry (rather than
# a closure) is required so the spawner survives Isaac Lab's Hydra config
# round-trip, which serialises ``spawn.func`` to ``module:qualname`` and rebuilds
# it via ``getattr`` — a closure has no resolvable module attribute.
_JOINT_LIMIT_CLAMP_REGISTRY: dict = {}


def register_joint_limit_clamp(
    usd_path: str,
    joint_defaults: dict,
    fallback_range: float = 6.2832,  # 2π
    max_range: float | None = None,
) -> None:
    """Register clamp parameters for a USD asset, keyed by its ``usd_path``.

    See :func:`spawn_usd_with_clamped_joint_limits` for how they are applied.
    """
    _JOINT_LIMIT_CLAMP_REGISTRY[usd_path] = (dict(joint_defaults or {}), fallback_range, max_range)


@clone
def spawn_usd_with_clamped_joint_limits(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """USD spawner that clamps wide/infinite joint limits **before the PhysX bake**.

    Drop-in replacement for :func:`isaaclab...spawn_from_usd`: set it as a robot
    cfg's ``spawn.func`` and register the clamp parameters for its ``usd_path``
    via :func:`register_joint_limit_clamp`.

    Motivation
    ----------
    The Kinova Gen3 continuous joints (``joint_1/3/5/7``) are authored in the
    referenced NVIDIA asset with limits outside PhysX's supported ``[-2π, 2π]``
    range.  When PhysX bakes the articulation at ``sim.reset()`` it calls
    ``PxArticulationJointReducedCoordinate::setLimitParams()`` on those joints and
    throws (8–9× at load, one surfacing as a GUI popup):

        ``setLimitParams() only supports limit angles in range [-2Pi, 2Pi]``

    The reset-mode :func:`clamp_infinite_joint_limits` runs only *after* the bake,
    so it silences the limits for stepping but cannot prevent the init-time error.
    A ``mode="prestartup"`` event is rejected by Isaac Lab whenever
    ``replicate_physics=True`` (which reach needs for 4096-env training), so the
    only remaining before-bake hook is the spawner itself.

    How it works
    ------------
    ``@clone`` spawns the single **source** prim (``env_0``) and only *then*
    clones it to the other envs, so clamping the source's joint limits here
    happens before both the clone and the PhysX bake.  With
    ``replicate_physics=True`` PhysX parses only ``env_0`` and replicates it, so a
    valid ``env_0`` is exactly what removes the error at its source; the clones
    inherit the clamped limits too.  Uses the identical ``default_pos ±
    fallback_range/2`` rule as the reset-mode term, so the baked limits — and thus
    training dynamics — are unchanged; the reset-mode term stays as a no-op net.
    """
    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    params = _JOINT_LIMIT_CLAMP_REGISTRY.get(cfg.usd_path)
    if params is not None:
        n_fixed = _clamp_revolute_joint_limits_on_prim(prim, *params)
        if n_fixed:
            print(
                f"[reach] joint-limit spawner: clamped {n_fixed} revolute joint "
                f"limit(s) on '{prim_path}' before PhysX bake"
            )
    return prim
