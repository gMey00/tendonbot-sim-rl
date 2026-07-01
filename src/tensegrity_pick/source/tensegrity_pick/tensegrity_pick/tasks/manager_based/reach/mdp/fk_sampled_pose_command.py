# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command generator that samples reachable poses via forward kinematics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_unique, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class FKSampledPoseCommand(CommandTerm):
    """Generate reachable end-effector pose targets by sampling random joint angles
    and computing forward kinematics through the physics engine.

    When ``fk_reference_asset_name`` is set, a **separate** articulation is used
    exclusively for FK sampling (the "reference robot").  This is critical for
    robots whose kinematic chain can break during training (e.g. physical 4-bar
    linkage): the reference robot is never actuated, so its kinematics stay
    clean.  The main ``asset_name`` robot is still used for tracking metrics.
    """

    cfg: FKSampledPoseCommandCfg

    def __init__(self, cfg: FKSampledPoseCommandCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx: int = self.robot.find_bodies(cfg.body_name)[0][0]

        # FK reference robot — used only for sampling reachable targets.
        if cfg.fk_reference_asset_name is not None:
            self.fk_robot: Articulation = env.scene[cfg.fk_reference_asset_name]
            fk_body = cfg.fk_reference_body_name or cfg.body_name
            self.fk_body_idx: int = self.fk_robot.find_bodies(fk_body)[0][0]
            fk_jnames = cfg.fk_reference_joint_names
            if fk_jnames is not None:
                self.fk_joint_ids: list[int] = self.fk_robot.find_joints(fk_jnames)[0]
            else:
                self.fk_joint_ids = list(range(self.fk_robot.num_joints))
        else:
            self.fk_robot = None

        if cfg.joint_names is not None:
            self.joint_ids: list[int] = self.robot.find_joints(cfg.joint_names)[0]
        else:
            self.joint_ids = list(range(self.robot.num_joints))

        # Resolve which articulation / indices the FK sampling actually drives:
        # the dedicated reference robot if one is configured, else the main robot.
        if self.fk_robot is not None:
            self._sampling_art: Articulation = self.fk_robot
            self._sampling_joint_ids: list[int] = self.fk_joint_ids
            self._sampling_body_idx: int = self.fk_body_idx
        else:
            self._sampling_art = self.robot
            self._sampling_joint_ids = self.joint_ids
            self._sampling_body_idx = self.body_idx

        # Body indices used for the (optional) geometric self-collision filter.
        # We exclude the gripper's internal finger cluster (many links naturally
        # <5 cm apart) so the pairwise-origin check only flags genuine arm/base
        # self-intersection — keeping the gripper *base* link as its representative.
        if cfg.self_collision_filter:
            self._collision_body_ids: list[int] | None = self._resolve_collision_bodies(self._sampling_art)
        else:
            self._collision_body_ids = None
        # Lazily calibrated on the first resample (see _calibrate_collision_pairs).
        self._collision_pairs: list[tuple[int, int]] | None = None

        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)

        self._step_counter = torch.zeros(self.num_envs, device=self.device)
        self._first_reach_step = torch.full((self.num_envs,), float("inf"), device=self.device)
        self._has_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reach_time_mean"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reach_time_variance"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "FKSampledPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tRandomised joint count: {len(self.joint_ids)}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_b

    def _update_metrics(self) -> None:
        from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

        self._step_counter += 1.0
        just_reached = (~self._has_reached) & (self.metrics["position_error"] < self.cfg.success_threshold)
        self._first_reach_step[just_reached] = self._step_counter[just_reached]
        self._has_reached |= just_reached

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)

        step_dt = self._env.step_dt
        reached_mask = self._has_reached[env_ids]

        if reached_mask.any():
            reach_times = self._first_reach_step[env_ids][reached_mask] * step_dt
            reach_time_mean = reach_times.mean().item()
            reach_time_variance = reach_times.var().item() if reach_times.numel() > 1 else 0.0
        else:
            reach_time_mean = float("nan")
            reach_time_variance = float("nan")

        # Write metrics so super().reset() picks them up into the extras dict
        # (this is what populates the TensorBoard Episode_Reward/* tags).
        success_values = reached_mask.float()
        self.metrics["success_rate"][env_ids] = success_values
        self.metrics["reach_time_mean"][env_ids] = reach_time_mean
        self.metrics["reach_time_variance"][env_ids] = reach_time_variance

        self._step_counter[env_ids] = 0.0
        self._first_reach_step[env_ids] = float("inf")
        self._has_reached[env_ids] = False

        # super().reset() reads metrics into extras THEN zeroes them.
        extras = super().reset(env_ids)

        # Restore the metric tensors so external consumers (eval harness,
        # downstream observers) can still read the just-finished episode's
        # success / reach-time per env after reset returns.
        self.metrics["success_rate"][env_ids] = success_values
        self.metrics["reach_time_mean"][env_ids] = reach_time_mean
        self.metrics["reach_time_variance"][env_ids] = reach_time_variance

        return extras

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        # Two target-generation modes share this command's metric / reset / vis
        # machinery so FK-sampled and uniform-box variants log identically and
        # can be compared apples-to-apples:
        #   * uniform_ranges set -> sample a fixed box in the robot ROOT frame
        #     (Isaac Lab's stock UniformPoseCommand logic, ported here);
        #   * otherwise          -> the original FK-through-physics sampling.
        if self.cfg.uniform_ranges is not None:
            self._resample_uniform(env_ids)
            return
        # The reference robot (if any) uses simple revolute joints and needs no
        # coupling closure; the main-robot path keeps the optional coupling fn.
        use_coupling = self.fk_robot is None
        self._resample_fk(env_ids, use_coupling=use_coupling)

    def _resample_uniform(self, env_ids: Sequence[int]) -> None:
        """Sample EE-pose targets uniformly from a fixed box in the robot ROOT
        frame, matching Isaac Lab's stock ``UniformPoseCommand``.

        Position is drawn from ``cfg.uniform_ranges.pos_{x,y,z}`` and orientation
        from uniform Euler ``roll``/``pitch``/``yaw`` (X-Y-Z). Because the target
        is expressed in each robot's own base frame, one box definition is
        automatically per-robot-relative and stays inside every arm's envelope.
        """
        rg = self.cfg.uniform_ranges
        n = len(env_ids)
        pos = torch.empty(n, 3, device=self.device)
        pos[:, 0].uniform_(*rg.pos_x)
        pos[:, 1].uniform_(*rg.pos_y)
        pos[:, 2].uniform_(*rg.pos_z)

        euler = torch.empty(n, 3, device=self.device)
        euler[:, 0].uniform_(*rg.roll)
        euler[:, 1].uniform_(*rg.pitch)
        euler[:, 2].uniform_(*rg.yaw)
        quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])

        self.pose_command_b[env_ids, :3] = pos
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    # ------------------------------------------------------------------
    # Geometric self-collision filter (origin-distance, kinematic, GPU)
    # ------------------------------------------------------------------
    def _resolve_collision_bodies(self, art: Articulation) -> list[int]:
        """Body indices to include in the self-collision check.

        Excludes any body whose name matches a gripper-internal pattern (finger /
        knuckle / pad / inner / outer): the Robotiq 2F-140 packs ~9 such links
        within a few cm, which would swamp a pairwise-origin check with false
        positives.  The gripper *base* link is kept as the gripper's stand-in.
        """
        names = art.body_names
        patterns = tuple(self.cfg.self_collision_exclude_patterns)
        return [k for k, nm in enumerate(names) if not any(p in nm for p in patterns)]

    @staticmethod
    def _segment_segment_distance(
        p1: torch.Tensor, q1: torch.Tensor, p2: torch.Tensor, q2: torch.Tensor
    ) -> torch.Tensor:
        """Batched shortest distance between segments ``p1-q1`` and ``p2-q2``.

        Standard clamped closest-point-on-two-segments solver (Ericson, *Real-Time
        Collision Detection*).  All inputs ``(n, 3)`` → returns ``(n,)``.
        """
        d1 = q1 - p1
        d2 = q2 - p2
        r = p1 - p2
        a = (d1 * d1).sum(-1)
        e = (d2 * d2).sum(-1)
        f = (d2 * r).sum(-1)
        c = (d1 * r).sum(-1)
        b = (d1 * d2).sum(-1)
        denom = a * e - b * b
        s = torch.where(denom > 1e-9, (b * f - c * e) / denom.clamp_min(1e-9), torch.zeros_like(denom))
        s = s.clamp(0.0, 1.0)
        t = ((b * s + f) / e.clamp_min(1e-9)).clamp(0.0, 1.0)
        s = ((b * t - c) / a.clamp_min(1e-9)).clamp(0.0, 1.0)
        c1 = p1 + s.unsqueeze(-1) * d1
        c2 = p2 + t.unsqueeze(-1) * d2
        return torch.norm(c1 - c2, dim=-1)

    def _collision_free(self, link_pos_w: torch.Tensor) -> torch.Tensor:
        """``True`` per env where no non-adjacent link *capsules* interpenetrate.

        ``link_pos_w`` is ``(n_envs, n_check_bodies, 3)`` of checked-body origins in
        chain order.  Each link is modelled as a capsule of radius
        ``self_collision_radius`` whose axis runs from its parent origin
        (``checked[i-1]``) to its own origin (``checked[i]``); the base link (i=0)
        is a sphere at its origin.  Two links collide when their capsule axes pass
        within ``2 * radius``.

        A pure body-*origin* distance check (the obvious simpler version) does not
        work here: it is blind to real link-surface collisions — a forearm lying
        alongside the upper arm keeps the joint origins far apart — while
        false-positiving on the rigidly-stacked wrist/EE/gripper-base frames that
        sit a few cm apart in every configuration.  Capsules + an adjacency skip of
        ≥2 fix both (verified empirically across the workspace).
        """
        n_bodies = link_pos_w.shape[1]
        two_r = 2.0 * self.cfg.self_collision_radius
        # Capsule axis endpoints: a0 = parent origin (shifted), a1 = own origin.
        a1 = link_pos_w
        a0 = link_pos_w.clone()
        a0[:, 1:] = link_pos_w[:, :-1]  # link 0 stays a point (a0==a1)

        # Lazily calibrate which body pairs to actually test.  A fixed adjacency
        # skip cannot generalise: different arms stack a different number of
        # rigid wrist/tool/EE/gripper frames that sit within ``2*radius`` in
        # *every* configuration and would false-positive constantly.  Instead, on
        # the first full-size batch we measure, per candidate pair, the fraction
        # of configs where the capsules are within ``2*radius``; pairs that are
        # close almost always are *structural* (rigidly stacked) and excluded,
        # leaving only pairs that are sometimes-far / sometimes-close — i.e. the
        # genuine self-collision candidates (gripper-into-base, elbow-into-body…).
        if self._collision_pairs is None:
            self._calibrate_collision_pairs(a0, a1, n_bodies, two_r)

        free = torch.ones(link_pos_w.shape[0], dtype=torch.bool, device=link_pos_w.device)
        for i, j in self._collision_pairs:
            dist = self._segment_segment_distance(a0[:, i], a1[:, i], a0[:, j], a1[:, j])
            free &= dist > two_r
        return free

    def _calibrate_collision_pairs(
        self, a0: torch.Tensor, a1: torch.Tensor, n_bodies: int, two_r: float
    ) -> None:
        """Pick the body pairs to test by excluding rigidly-stacked (structural)
        pairs — see ``_collision_free``.  Runs once, on the first resample (which
        is the full startup reset over all envs, so the fraction estimate is
        well-sampled), and caches the kept-pair list for the rest of training.
        """
        skip = self.cfg.self_collision_adjacency_skip
        thr = self.cfg.self_collision_structural_threshold
        candidates = [
            (i, j) for i in range(n_bodies) for j in range(i + 1 + skip, n_bodies)
        ]
        kept = []
        for i, j in candidates:
            dist = self._segment_segment_distance(a0[:, i], a1[:, i], a0[:, j], a1[:, j])
            frac_close = (dist < two_r).float().mean().item()
            if frac_close < thr:
                kept.append((i, j))
        self._collision_pairs = kept
        print(
            f"[FKSampledPoseCommand] self-collision filter calibrated on {a0.shape[0]} envs: "
            f"testing {len(kept)}/{len(candidates)} body pairs "
            f"(excluded {len(candidates) - len(kept)} structural/rigid pairs)."
        )

    # ------------------------------------------------------------------
    # Unified FK sampling (optionally with self-collision rejection)
    # ------------------------------------------------------------------
    def _resample_fk(self, env_ids: Sequence[int], *, use_coupling: bool) -> None:
        """Sample reachable EE-pose targets via FK, optionally rejecting
        self-colliding configurations.

        A single articulation (the dedicated reference robot if configured, else
        the main robot) is teleported to random joint configurations; FK gives
        the EE pose.  When ``cfg.self_collision_filter`` is set, each env keeps
        the first sampled configuration that is geometrically self-collision-free,
        re-sampling only the still-colliding envs for up to
        ``cfg.self_collision_max_iters`` rounds.  Envs that never find a clean
        sample fall back to their very first sample (rare).  With the filter off
        this collapses to a single sampling round — identical to the original.
        """
        import isaaclab.utils.math as math_utils_il

        art = self._sampling_art
        jids = self._sampling_joint_ids
        bidx = self._sampling_body_idx

        joint_limits = art.data.soft_joint_pos_limits[0, jids]
        lower = joint_limits[:, 0].clone()
        upper = joint_limits[:, 1].clone()

        # Some imported USDs expose invalid/infinite soft limits; sampling from
        # those produces NaNs.  Substitute a finite fallback range about default.
        default_joint_pos = art.data.default_joint_pos[0, jids]
        fallback_half_range = torch.full_like(default_joint_pos, 0.75)
        invalid_limit_mask = (
            ~torch.isfinite(lower)
            | ~torch.isfinite(upper)
            | ((upper - lower) <= 1.0e-5)
            | ((upper - lower) > (8.0 * torch.pi))
        )
        lower = torch.where(invalid_limit_mask, default_joint_pos - fallback_half_range, lower)
        upper = torch.where(invalid_limit_mask, default_joint_pos + fallback_half_range, upper)

        # Sample from the inner portion of each joint range (cfg.joint_range_margin).
        margin = (upper - lower) * self.cfg.joint_range_margin
        lower = lower + margin
        upper = upper - margin

        n = len(env_ids)
        saved_joint_positions = art.data.joint_pos[env_ids].clone()
        saved_joint_velocities = art.data.joint_vel[env_ids].clone()

        root_pos_w = self.robot.data.root_pos_w[env_ids]
        root_quat_w = self.robot.data.root_quat_w[env_ids]

        accepted = torch.zeros(n, dtype=torch.bool, device=self.device)
        out_pos_b = torch.zeros(n, 3, device=self.device)
        out_quat_b = torch.zeros(n, 4, device=self.device)
        out_quat_b[:, 0] = 1.0

        max_iters = self.cfg.self_collision_max_iters if self.cfg.self_collision_filter else 1
        for it in range(max_iters):
            random_joint_positions = lower + (upper - lower) * torch.rand(
                n, len(jids), device=self.device
            )
            if use_coupling and self.cfg.joint_coupling_fn is not None:
                random_joint_positions = self.cfg.joint_coupling_fn(random_joint_positions)

            art.write_joint_position_to_sim(random_joint_positions, joint_ids=jids, env_ids=env_ids)
            # Force PhysX to flush the write and recompute kinematics, then read
            # link transforms directly (bypassing the TimestampedBuffer cache).
            art.data._physics_sim_view.update_articulations_kinematic()
            raw_link_transforms = art.data._root_physx_view.get_link_transforms().clone()
            raw_link_transforms[..., 3:7] = math_utils_il.convert_quat(raw_link_transforms[..., 3:7], to="wxyz")
            link_subset = raw_link_transforms[env_ids]
            body_pose = link_subset[:, bidx]

            # Express target in the *main* (controlled) robot's root frame.
            pos_b, quat_b = subtract_frame_transforms(
                root_pos_w, root_quat_w, body_pose[:, :3], body_pose[:, 3:7]
            )

            if it == 0:
                # Fallback for envs that never find a collision-free sample.
                out_pos_b[:] = pos_b
                out_quat_b[:] = quat_b

            if self.cfg.self_collision_filter:
                free = self._collision_free(link_subset[:, self._collision_body_ids, :3])
            else:
                free = torch.ones(n, dtype=torch.bool, device=self.device)

            newly = free & ~accepted
            out_pos_b[newly] = pos_b[newly]
            out_quat_b[newly] = quat_b[newly]
            accepted |= free
            if bool(accepted.all()):
                break

        self.pose_command_b[env_ids, :3] = out_pos_b
        self.pose_command_b[env_ids, 3:] = quat_unique(out_quat_b) if self.cfg.make_quat_unique else out_quat_b

        # Restore the original configuration and flush solver state.
        art.write_joint_position_to_sim(saved_joint_positions, env_ids=env_ids)
        art.write_joint_velocity_to_sim(saved_joint_velocities, env_ids=env_ids)
        art.data._physics_sim_view.update_articulations_kinematic()
        art.data._body_link_pose_w.timestamp = -1

    def _update_command(self) -> None:
        pass

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        from isaaclab.markers import VisualizationMarkers

        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        if not self.robot.is_initialized:
            return
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        body_link_pose_w = self.robot.data.body_link_pose_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_link_pose_w[:, :3], body_link_pose_w[:, 3:7])


@configclass
class FKSampledPoseCommandCfg(CommandTermCfg):
    """Configuration for FK-based reachable pose command sampling."""

    class_type: type = FKSampledPoseCommand

    asset_name: str = MISSING
    body_name: str = MISSING
    joint_names: list[str] | None = None
    make_quat_unique: bool = False
    success_threshold: float = 0.02
    joint_range_margin: float = 0.10

    @configclass
    class Ranges:
        """Uniform sampling ranges (min, max) for the fixed-box target mode.

        Position ranges are in the robot ROOT frame [m]; orientation ranges are
        uniform Euler angles [rad] applied X-Y-Z. Mirrors the field layout of
        Isaac Lab's ``UniformPoseCommandCfg.Ranges`` for familiarity."""

        pos_x: tuple[float, float] = MISSING
        pos_y: tuple[float, float] = MISSING
        pos_z: tuple[float, float] = MISSING
        roll: tuple[float, float] = MISSING
        pitch: tuple[float, float] = MISSING
        yaw: tuple[float, float] = MISSING

    uniform_ranges: Ranges | None = None
    """If set, targets are drawn uniformly from this fixed box in the robot ROOT
    frame (Isaac Lab's stock ``UniformPoseCommand`` logic) instead of via FK
    sampling. All the command's metrics / reset / debug-vis behaviour is shared
    between the two modes, so FK-sampled and uniform-box variants log
    identically and stay directly comparable."""

    # ── Self-collision rejection filter ───────────────────────────────────
    self_collision_filter: bool = False
    """If set, reject FK-sampled configurations that are geometrically
    self-colliding, so every target corresponds to a *collision-free* reaching
    configuration (and is therefore actually reachable by the physical arm).

    The check is purely kinematic (pairwise body-origin distances on the FK
    config), runs on the GPU, and never touches the physics solver."""

    self_collision_radius: float = 0.05
    """Capsule radius (m) used to model every checked link for the self-collision
    test (two links collide when their axes pass within ``2 * radius``).  Kept
    modest so the filter removes only genuine self-intersection and does not
    erode the reachable workspace (verified: ~5% rejection at full range)."""

    self_collision_adjacency_skip: int = 1
    """Links within this many chain-index steps are always treated as connected
    and never flagged.  Kept at 1 because the auto-calibration
    (``self_collision_structural_threshold``) additionally removes *any* pair —
    however far apart in the chain — that is rigidly close in every configuration
    (the stacked wrist/EE/tool/gripper frames), which a fixed skip cannot do
    generally."""

    self_collision_structural_threshold: float = 0.9
    """A candidate body pair that is within ``2*radius`` in at least this fraction
    of sampled configurations is deemed *structural* (rigidly stacked) and is
    excluded from the self-collision test.  Genuine collision pairs (gripper into
    base, elbow into body) are close only occasionally and are kept."""

    self_collision_max_iters: int = 8
    """Max rejection-sampling rounds; only still-colliding envs are re-sampled."""

    self_collision_exclude_patterns: list[str] = (
        "inner", "outer", "finger", "knuckle", "pad",
    )
    """Body-name substrings excluded from the check (the Robotiq finger cluster),
    so the gripper's tightly-packed internal links don't swamp the pairwise
    origin check with false positives.  The gripper *base* link is kept."""
    """Fraction of each joint's range to exclude at both ends when sampling.

    A margin of 0.10 (default) samples from the inner 80% of each joint range,
    avoiding extreme configurations.  Larger values narrow the workspace, which
    can help robots with large joint ranges (e.g. UR10e ±2π) converge faster."""

    joint_coupling_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
    """Optional function to enforce kinematic coupling after independent sampling.

    When set, called with the (N, J) sampled joint positions tensor and must
    return a modified tensor of the same shape.  Useful for robots with closed
    kinematic chains (e.g. antiparallelogram 4-bar linkage) where some joints
    are not independent."""

    fk_reference_asset_name: str | None = None
    """Scene entity name for a *separate* articulation used exclusively for FK
    sampling.  When set, the reference robot is teleported to random joint
    configurations to compute reachable EE poses.  The main ``asset_name``
    robot is never touched during sampling, so it cannot "break"."""

    fk_reference_body_name: str | None = None
    """End-effector body on the FK reference robot.  Defaults to ``body_name``."""

    fk_reference_joint_names: list[str] | None = None
    """Joints to randomise on the FK reference robot.  Defaults to all joints."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/goal_pose"
    )
    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )

    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)