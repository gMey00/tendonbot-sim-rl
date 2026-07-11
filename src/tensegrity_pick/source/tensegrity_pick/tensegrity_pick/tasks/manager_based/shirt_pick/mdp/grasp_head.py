"""Learned grasp-point refinement head (Stage-2 roadmap, S2/T1).

The stage-1 policy picks the crumpled shirt at its HIGHEST point
(depth-camera-trivial, literature-standard).  The heuristics study proved the
presentation ceiling is set by the grasp PAIR: an arbitrary first grasp caps
Task-2 coverage at 0.713 (oracle second grasp) while hem-corner/side first
grasps enable ~0.82.  This module makes the first-grasp TARGET
policy-selectable around that bootstrap:

* **Action** (``GraspOffsetAction``, 2 dims): a continuous xy offset around
  the highest point, clamped to ``GRASP_OFFSET_RADIUS`` and SNAPPED to the
  local top of the cloth surface at the shifted location (mean of the
  ``SNAP_TOPK`` highest particles within ``SNAP_RADIUS``).  The snap keeps the
  target on the visible upper surface — exactly what a depth camera provides
  (local height-map maximum at a chosen pixel), so the camera contract holds.
  Zero action ≡ the stage-1 highest point (bootstrap preserved); the target
  freezes to base behaviour once the grasp is latched (post-grasp reward
  semantics identical to stage 1).  Design choice (a) over the discrete
  K-candidate head (b): the pipeline's Gaussian PPO handles a small continuous
  offset natively, while a discrete choice must be shoehorned through argmax
  on continuous logits; (b) remains the documented comparison config for the
  Alex sweep (encode as 13 scores → argmax) if (a) misses the gate.

* **Observations**: the study's 12 region-landmark keypoint positions
  relative to the finger tip (privileged read of exactly those particles —
  INF's shared camera-contract terms will replace this with estimated
  keypoints + visibility flags; rebase when merged) and the border distance
  of the current grasp target (border-near grasps present better, study §5).

* **Reward** (``coverage_terminal_bonus``): one-shot at the present latch,
  ∝ the PREDICTED downstream coverage of the achieved hold region —
  ``REGION_COVERAGE_LUT`` (min-max normalised), derived offline from the
  study's stratified pair map by
  ``scripts/model_validation/build_pick_region_coverage_lut.py`` (best
  second-grasp partner per first-grasp region, median cov_plane, min 25
  trials/cell).  dt-scaling: one-shot ⇒ effective bonus = weight/60.

Data dependency: ``doc/reports/data/present_markers.pt`` (study artifact,
read-only) — per-particle region ids + border distances + keypoint indices on
the registered flat rest shape.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ── Head geometry ─────────────────────────────────────────────────────────
GRASP_OFFSET_RADIUS = 0.15  # m — max xy offset around the highest point
SNAP_RADIUS = 0.05          # m — xy neighbourhood snapped to the local top
SNAP_TOPK = 20              # particles averaged for the local-top point

# ── Region → expected Task-2 coverage (offline lookup) ────────────────────
# Best-partner median cov_plane per FIRST-grasp region, stratified pair map
# (present_h3s_stratified.csv, 8192 valid trials, min 25/cell).  Regenerate /
# validate with scripts/model_validation/build_pick_region_coverage_lut.py.
REGION_COVERAGE_LUT = {
    "collar": 0.7096,
    "shoulder": 0.6918,
    "sleeve": 0.7253,
    "chest": 0.6836,
    "side": 0.8226,
    "belly": 0.6952,
    "hem_corner": 0.8219,
    "hem_c": 0.687,
}
# The study's symmetrized class order (present_markers.pt ``sym_classes``).
SYM_CLASSES = ["collar", "shoulder", "sleeve", "chest", "side", "belly", "hem_corner", "hem_c"]
HIGH_VALUE_REGIONS = ("side", "hem_corner")


def _find_markers_file() -> str:
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(12):
        cand = os.path.join(d, "doc", "reports", "data", "present_markers.pt")
        if os.path.isfile(cand):
            return cand
        d = os.path.dirname(d)
    raise FileNotFoundError(
        "present_markers.pt not found walking up from " + __file__
    )


class _Markers:
    """Lazy per-device cache of the study marker tensors."""

    _cache: dict[str, "_Markers"] = {}

    def __init__(self, device: str):
        data = torch.load(_find_markers_file(), map_location=device, weights_only=False)
        assert list(data["sym_classes"]) == SYM_CLASSES, "study class order changed"
        self.keypoint_idx: torch.Tensor = data["keypoint_idx"].to(device)      # [12]
        self.border_dist: torch.Tensor = data["border_dist"].to(device)        # [P]
        self.particle_region: torch.Tensor = data["particle_region"].to(device)  # [P] sym8
        cov = torch.tensor([REGION_COVERAGE_LUT[c] for c in SYM_CLASSES], device=device)
        self.cov_by_region = cov                                               # [8] raw
        self.cov_norm_by_region = (cov - cov.min()) / (cov.max() - cov.min())  # [8] ∈ [0,1]
        self.high_value_mask = torch.tensor(
            [c in HIGH_VALUE_REGIONS for c in SYM_CLASSES], device=device
        )

    @classmethod
    def get(cls, device) -> "_Markers":
        key = str(device)
        if key not in cls._cache:
            cls._cache[key] = cls(key)
        return cls._cache[key]


def _ready(env: "ManagerBasedRLEnv") -> bool:
    return getattr(env, "_cloth", None) is not None


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

def region_keypoints_rel_tip(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """The 12 region-landmark keypoint particle positions − finger tip (N, 36).

    Privileged stand-in for the camera contract: reads ONLY the 12 study
    keypoint particles.  INF's shared observation terms (estimated keypoints +
    visibility flags) replace this after their merge.
    """
    if not _ready(env):
        return torch.zeros(env.num_envs, 36, device=env.device)
    mk = _Markers.get(env.device)
    kp = env._cloth.nodal_pos_w[:, mk.keypoint_idx, :]          # [N, 12, 3]
    rel = kp - env._finger_tip_pos().unsqueeze(1)               # [N, 12, 3]
    return rel.reshape(env.num_envs, -1)


def grasp_target_border_dist(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Border distance (flat-rest, m) of the particle nearest the current
    grasp target (N, 1) — border-near holds present better (study §5)."""
    if not _ready(env):
        return torch.zeros(env.num_envs, 1, device=env.device)
    mk = _Markers.get(env.device)
    d = (env._cloth.nodal_pos_w - env.shirt_grasp_point_w.unsqueeze(1)).norm(dim=-1)
    nearest = d.argmin(dim=1)                                   # [N]
    return mk.border_dist[nearest].unsqueeze(-1)


# ---------------------------------------------------------------------------
# Action term
# ---------------------------------------------------------------------------

class GraspOffsetAction(ActionTerm):
    """2-dim xy offset of the grasp target around the highest point.

    Writes ``env._grasp_offset_xy`` (consumed by
    ``ShirtPickEnv.shirt_grasp_point_w``'s surface-snapped override); applies
    nothing to any articulation.  Raw actions are clamped to [-1, 1] and
    scaled by ``cfg.max_radius``.
    """

    cfg: "GraspOffsetActionCfg"

    def __init__(self, cfg: "GraspOffsetActionCfg", env):
        super().__init__(cfg, env)
        self._raw = torch.zeros(env.num_envs, 2, device=env.device)
        self._processed = torch.zeros(env.num_envs, 2, device=env.device)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        self._processed[:] = torch.clamp(actions, -1.0, 1.0) * self.cfg.max_radius
        self._env._grasp_offset_xy[:] = self._processed

    def apply_actions(self):
        pass

    def reset(self, env_ids=None):
        if env_ids is None:
            self._raw.zero_()
            self._processed.zero_()
            self._env._grasp_offset_xy.zero_()
        else:
            self._raw[env_ids] = 0.0
            self._processed[env_ids] = 0.0
            self._env._grasp_offset_xy[env_ids] = 0.0
        return {}


@configclass
class GraspOffsetActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = GraspOffsetAction
    asset_name: str = "robot"  # required by ActionTerm; no asset is driven
    max_radius: float = GRASP_OFFSET_RADIUS


# ---------------------------------------------------------------------------
# Reward + latch-time helpers
# ---------------------------------------------------------------------------

def hold_region_sym8(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Per-env sym8 region id of the CURRENT hold (N,), -1 where no grasp.

    Representative particle = welded (slot-0) particle nearest the finger tip
    — the same convention as the terminal-bank generator.
    """
    mk = _Markers.get(env.device)
    mask = env._cloth._attach_mask[0]                            # [N, P]
    tip = env._finger_tip_pos()                                  # [N, 3]
    d = (env._cloth.nodal_pos_w - tip.unsqueeze(1)).norm(dim=-1)  # [N, P]
    d = d.masked_fill(~mask, float("inf"))
    rep = d.argmin(dim=1)                                        # [N]
    region = mk.particle_region[rep]
    return torch.where(mask.any(dim=1), region, torch.full_like(region, -1))


def coverage_terminal_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """One-shot at the present latch: normalised predicted Task-2 coverage of
    the achieved hold region (0 = chest-grade hold, 1 = side/hem-corner).

    Fires on ``present_latch_event`` (one-step delay like ``drop_event``), so
    it can only be earned by episodes that reach the stage-1 success latch —
    the head cannot trade presenting away for region shopping.  Size the
    weight ~60× a comparable per-step term (dt-scaled one-shot).
    """
    if not _ready(env):
        return torch.zeros(env.num_envs, device=env.device)
    latch = getattr(env, "present_latch_event", None)
    if latch is None:
        return torch.zeros(env.num_envs, device=env.device)
    mk = _Markers.get(env.device)
    region = env._hold_region                                    # set at latch
    norm_cov = torch.where(
        region >= 0,
        mk.cov_norm_by_region[region.clamp(min=0)],
        torch.zeros_like(latch),
    )
    return latch * norm_cov


# ---------------------------------------------------------------------------
# Config helper (applied by robot-variant head cfgs)
# ---------------------------------------------------------------------------

def apply_grasp_head(cfg) -> None:
    """Attach the head's action/observation/reward terms to a task cfg."""
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import RewardTermCfg as RewTerm

    cfg.actions.grasp_offset = GraspOffsetActionCfg()
    cfg.observations.policy.region_keypoints = ObsTerm(func=region_keypoints_rel_tip)
    cfg.observations.policy.grasp_border_dist = ObsTerm(func=grasp_target_border_dist)
    # One-shot ⇒ dt-scaled effective bonus = 1800/60 = 30 × norm-coverage —
    # comparable to ~1 s of the per-step `presented` term (weight 30): worth
    # rerouting the approach, not worth skipping the present latch.
    cfg.rewards.coverage_bonus = RewTerm(func=coverage_terminal_bonus, weight=1800.0)
