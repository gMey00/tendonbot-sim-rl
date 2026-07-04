# shirt_distribute_env.py
#
# Task 3 of the cloth-sorting pipeline: the second robot places the classified
# shirt into the commanded drum.  Goal-conditioned (TossingBot-style
# formulation, research report §4): the condition label only selects WHICH bin
# position is observed — the per-episode target bin is resampled uniformly so
# "nearest bin" and "correct bin" diverge and the policy cannot ignore the goal.
#
# Initial state (implemented 2026-07-04, replaces the flat-on-belt stub): the
# episode starts with the shirt ALREADY HANGING from the learning robot's own
# closed gripper, welded at one random particle patch (slot 0), with the arm in
# a pose sampled from plausible end-of-Task-2 configurations:
#
#   * At init, a one-off HOLDING-POSE SWEEP samples arm joint configurations
#     (guided rejection sampling, see ``_build_holding_pose_bank``), keeps the
#     ones whose closed-gripper finger tip lands inside ``TIP_BOX_*`` — an
#     env-local box spanning the presentation area ``PRESENTATION_POS``
#     (0.15, 0.90, 1.60) and the region between it and the robot's home
#     posture — with the fingers pointing downward.  FK comes from stepping
#     the actual sim (2 steps/round), so the recorded tip positions are the
#     true PD equilibria, not idealized kinematics.
#   * Each reset samples a bank pose, writes it (positions + PD targets),
#     forces the gripper CLOSED (``_force_gripper_closed`` — without it the
#     open default pose makes ``_update_grasp`` release the restored grasp on
#     the first step), and restores a relaxed random-particle hang from the
#     cached hanging bank at the recorded fingertip (slot 0), filtered by
#     per-env ``max_drape`` so the hang clears the drum tops (0.88 m).
#
# PIPELINE SEAM: when the real Task-2 terminal-state bank exists, replace
# ``_build_holding_pose_bank`` + ``_reset_cloth`` with "sample (joint_pos,
# cloth_state) from the Task-2 bank" — the rest of the MDP is agnostic to how
# the holding state was produced.
#
# The policy's gripper action overwrites the finger target from the first
# action step; an "open" command (any action ≥ 0 on the binary term) releases
# immediately.  That is legitimate — release IS the task's final act — but a
# bad release (not over the commanded drum) fires the one-shot
# ``bad_release_event`` penalty, so episodes that start with an instant drop
# are punished, not farmed.

from __future__ import annotations

import logging
from typing import Sequence

import torch

from isaaclab.assets import Articulation

from ..shared.cloth_sorting_env import ClothSortingEnvBase
from ..shared.cloth_sorting_scene_cfg import (
    DRUM_NAMES,
    DRUM_POSITIONS,
    HANGING_BANK_PATH,
)
from ..shared.gripper_cfg import BinCylinder, get_world_pos, in_upright_cylinder
from ..shared.proj_base_scene_cfg import DRUM_HEIGHT_M

# Same interior-depth cylinder as shirt_place (draped cloth rarely reaches the
# drum bottom, so the success volume spans the full interior).
_DRUM_GEOM = BinCylinder(radius=0.547 * 0.5, height=0.85)
# Fraction of (released) cloth particles inside the target drum that counts as
# a correct distribution (same threshold as shirt_place placements).
DISTRIBUTE_FRACTION_THRESHOLD = 0.15

# ── Release-event grading (ported from shirt_place, retargeted to the
#    commanded bin) ─────────────────────────────────────────────────────────
# A release fires the graded one-shot bonus when the gripper opens with the
# grasp point over the commanded drum's footprint and above its top edge; the
# magnitude grades the drop (centering × whole-shirt-lift), so a centred drop
# with the shirt fully cleared pays ~4× a barely-valid rim-graze.
RELEASE_TOP_CLEARANCE = 0.02      # gp must be above drum top + this at release
CLEAR_RIM_CLEARANCE = 0.04        # "cleared" rim reference = drum top + this
CLEAR_MARGIN = 0.05               # full clearance_fraction this far above rim
CLEAR_DRUM_RADIUS = 0.32          # xy gate for the clearance/anti-hover fade
DRUM_TOP = DRUM_HEIGHT_M          # 0.88 m

# ── Holding-pose sweep (initial-state distribution) ───────────────────────
# Env-local box the closed-gripper finger tip must land in.  Spans the
# presentation area (0.15, 0.90, 1.60) and the region between it and the
# robot's home posture (base at (0.75, 1.0, 0.75)).  Z floor 1.20: the tip
# with fingers down needs the flange ≥ ~0.1–0.24 m higher, and the UR5e's
# flange tops out ≈ 1.83 m — the first sweep with a 1.55 floor found ZERO
# poses in 120 rounds (the acceptance volume sat at the workspace edge).
TIP_BOX_X = (0.10, 1.00)
TIP_BOX_Y = (0.65, 1.30)
TIP_BOX_Z = (1.20, 1.75)
# Fingers must point downward: world-z drop from the EE frame origin to the
# dynamic finger tip ≥ 0.10 m (of the ~0.235 m closed-tip offset ⇒ ≤ ~65°
# from vertical) — end-of-Task-2 the gripper holds the shirt hanging.
TIP_DOWN_MIN_DROP = 0.10
# Guided rejection sampling: each round, envs are split between EXPLORE
# (uniform half-range widths, capped, around the default pose) and EXPLOIT
# (perturb already-accepted poses by ±EXPLOIT_NOISE) as soon as one seed
# exists.  Measured need: blind exploration accepts only ~0.4 % (6/1600
# samples, 2026-07-04 job 3809918) — the old design waited for 8 seeds
# before exploiting, so densification never engaged.
POSE_SWEEP_MAX_ROUNDS = 300
POSE_SWEEP_SETTLE_STEPS = 2
POSE_BANK_TARGET = 256
POSE_BANK_MIN = 32
EXPLORE_WIDTH_CAP = 3.1416        # rad, per-joint half-range cap
EXPLOIT_FRACTION = 0.5            # env fraction perturbing accepted poses
EXPLOIT_NOISE = 0.25              # rad
# ── Pose-dependent drape limit ─────────────────────────────────────────────
# The restored hang must clear whatever is BELOW the sampled tip, not a blunt
# global bound: over a drum footprint (xy within DRUM_AVOID_RADIUS = drum
# radius 0.27 + cloth xy half-extent ~0.25) the bottom must clear the drum
# top (0.88 + margin); near the belt (tip y such that the ~±0.25 m cloth
# reaches the collider's y ≤ 0.45) it must clear the belt surface (0.80);
# elsewhere only the floor is below and the full bank is admissible.  Poses
# whose allowed drape is below the shortest bank level are REJECTED at sweep
# time (the bank's drape spans 0.47–0.95 m, mean 0.74 — a relaxed one-point
# hang of this garment cannot be shorter).
DRAPE_MARGIN = 0.04
DRUM_AVOID_RADIUS = 0.55
BELT_AVOID_Y = 0.72
BELT_TOP = 0.80                   # CONVEYOR_SURFACE_HEIGHT_M
# max_drape quantization levels for the bucketed bank restore (ascending);
# 0.55 admits the ~9 shortest bank states (× yaw/mirror augmentation).
DRAPE_LEVELS = (0.55, 0.62, 0.72, 0.82, 0.95)


class ShirtDistributeEnv(ClothSortingEnvBase):
    """Place the already-held shirt into the commanded condition bin."""

    hanging_bank_path = HANGING_BANK_PATH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_bin = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._was_distributed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._was_released = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # One-shot events, consumed by the reward manager on the next step
        # (shirt_pick/shirt_place bookkeeping pattern).
        self._release_event = torch.zeros(self.num_envs, device=self.device)
        self._bad_release_event = torch.zeros(self.num_envs, device=self.device)
        # Peak released-cloth fraction inside the commanded drum (metric).
        self._max_target_fraction = torch.zeros(self.num_envs, device=self.device)
        # Reset-step anchor correction (see ``_update_grasp`` override): the
        # true fingertip of the freshly written holding pose, used instead of
        # the stale ``robot.data`` FK for exactly one step after each reset.
        self._reset_anchor = torch.zeros(self.num_envs, 3, device=self.device)
        self._reset_anchor_pending = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Arm joint ids in the order of the action term's joint list.
        robot: Articulation = self.scene["robot"]
        arm_names = list(self.cfg.actions.arm_action.joint_names)
        self._arm_joint_ids = [robot.joint_names.index(n) for n in arm_names]

        # One-off holding-pose sweep (needs the settled sim from __init__).
        self._build_holding_pose_bank()

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

    def _fraction_in_bins(self) -> torch.Tensor:
        """Fraction of cloth particles inside each drum ``[N, 3]``."""
        pts = self._cloth.nodal_pos_w
        fracs = []
        for name in DRUM_NAMES:
            center = get_world_pos(self.scene[name], env=self)
            fracs.append(in_upright_cylinder(pts, center, _DRUM_GEOM).float().mean(dim=1))
        return torch.stack(fracs, dim=1)

    def _target_fraction_in_bin(self) -> torch.Tensor:
        """Fraction of cloth particles inside the commanded drum ``[N]``."""
        center = self.target_bin_pos_w
        pts = self._cloth.nodal_pos_w
        inside = in_upright_cylinder(pts, center, _DRUM_GEOM)
        return inside.float().mean(dim=1)

    def _wrong_fraction_in_bin(self) -> torch.Tensor:
        """Max cloth fraction inside any NON-commanded drum ``[N]``."""
        fracs = self._fraction_in_bins()                       # [N, 3]
        fracs.scatter_(1, self._target_bin.view(-1, 1), 0.0)
        return fracs.max(dim=1).values

    # ------------------------------------------------------------------
    # Reward-manager hooks (read by mdp/rewards.py)
    # ------------------------------------------------------------------

    @property
    def was_distributed(self) -> torch.Tensor:
        return self._was_distributed

    @property
    def was_released(self) -> torch.Tensor:
        return self._was_released

    @property
    def release_event(self) -> torch.Tensor:
        """One-shot ``[N]``: graded (0.25–1.0) the step a good release fired."""
        return self._release_event

    @property
    def bad_release_event(self) -> torch.Tensor:
        """One-shot ``[N]``: 1.0 the step the grasp opened NOT over the goal."""
        return self._bad_release_event

    @property
    def clearance_fraction(self) -> torch.Tensor:
        """Smooth [0,1] "shirt fully cleared over the commanded drum" ``[N]``.

        1.0 when the grasp point is over the commanded drum's footprint AND
        the whole shirt (lowest particle) is ``CLEAR_MARGIN`` above its rim.
        Fades the per-step positioning shaping (anti-hover, shirt_place
        design): hovering cleared over the right drum earns ~nothing — only
        releasing pays.
        """
        target = self.target_bin_pos_w
        gp = self.shirt_grasp_point_w
        d_xy = torch.norm(gp[:, :2] - target[:, :2], dim=-1)
        in_xy = d_xy < CLEAR_DRUM_RADIUS
        rim = DRUM_TOP + CLEAR_RIM_CLEARANCE
        bottom_z = self.shirt_min_z_w - self.scene.env_origins[:, 2]
        clr = torch.clamp((bottom_z - rim) / CLEAR_MARGIN, 0.0, 1.0)
        gate = in_xy & self._was_grasped & self.grasp_active
        return torch.where(gate, clr, torch.zeros_like(clr))

    # ------------------------------------------------------------------
    # Holding-pose bank (initial-state distribution over arm configs)
    # ------------------------------------------------------------------

    def _allowed_drape(self, tip_local: torch.Tensor) -> torch.Tensor:
        """Max admissible hang length below each env-local tip ``[K]``.

        Pose-dependent: constrained by the drum tops only when the tip is
        over/near a drum footprint, by the belt surface only near the belt,
        otherwise unconstrained (floor is far below all sampled tips).
        """
        dev = tip_local.device
        allowed = torch.full((tip_local.shape[0],), 10.0, device=dev)
        near_belt = tip_local[:, 1] < BELT_AVOID_Y
        allowed = torch.where(
            near_belt, tip_local[:, 2] - BELT_TOP - DRAPE_MARGIN, allowed
        )
        for pos in DRUM_POSITIONS:
            center = torch.tensor(pos[:2], device=dev, dtype=torch.float32)
            near = torch.norm(tip_local[:, :2] - center, dim=-1) < DRUM_AVOID_RADIUS
            allowed = torch.where(
                near,
                torch.minimum(allowed, tip_local[:, 2] - DRUM_TOP - DRAPE_MARGIN),
                allowed,
            )
        return allowed

    def _build_holding_pose_bank(self) -> None:
        """Sample end-of-Task-2-like holding poses by guided rejection FK.

        Rounds of: write sampled arm joint positions (+ closed gripper),
        step the sim ``POSE_SWEEP_SETTLE_STEPS`` steps, read the dynamic
        finger tip, accept poses whose tip lies in ``TIP_BOX_*`` with the
        fingers pointing down.  Early rounds explore widely around the
        default pose; once seeds exist, later rounds perturb accepted poses
        (±``EXPLOIT_NOISE``) to densify the bank.  Stores
        ``self._pose_bank_q [K, A]`` and env-local ``self._pose_bank_tip
        [K, 3]`` — the recorded tips are the PD equilibria the reset
        restores, so the hang anchor and the first-step fingertip agree.
        """
        log = logging.getLogger(__name__)
        robot: Articulation = self.scene["robot"]
        dev = self.device
        n = self.num_envs
        dt = self.cfg.sim.dt
        arm_ids = self._arm_joint_ids
        a = len(arm_ids)
        all_ids = torch.arange(n, device=dev)

        default_q = robot.data.default_joint_pos[:, arm_ids]          # [N, A]
        limits = robot.data.soft_joint_pos_limits[:, arm_ids, :]      # [N, A, 2]
        lo, hi = limits[..., 0], limits[..., 1]
        width = torch.clamp(0.5 * (hi - lo), max=EXPLORE_WIDTH_CAP)

        self._force_gripper_closed(all_ids)

        bank_q: list[torch.Tensor] = []
        bank_tip: list[torch.Tensor] = []
        bank_drape: list[torch.Tensor] = []
        n_accepted = 0
        zeros_vel = torch.zeros(n, a, device=dev)
        # Failure diagnostics: where DID the sampled tips land?
        seen_min = torch.full((3,), float("inf"), device=dev)
        seen_max = torch.full((3,), float("-inf"), device=dev)

        for rnd in range(POSE_SWEEP_MAX_ROUNDS):
            if n_accepted >= POSE_BANK_TARGET:
                break
            # Explore everywhere; as soon as one seed exists, an
            # EXPLOIT_FRACTION of the envs perturbs accepted poses instead.
            q = default_q + (torch.rand(n, a, device=dev) - 0.5) * 2.0 * width
            if n_accepted > 0:
                pool = torch.cat(bank_q, dim=0)
                base = pool[torch.randint(0, pool.shape[0], (n,), device=dev)]
                q_exploit = base + (torch.rand(n, a, device=dev) - 0.5) * 2.0 * EXPLOIT_NOISE
                exploit = torch.rand(n, device=dev) < EXPLOIT_FRACTION
                q = torch.where(exploit.unsqueeze(1), q_exploit, q)
            q = torch.clamp(q, lo, hi)

            robot.write_joint_state_to_sim(q, zeros_vel, joint_ids=arm_ids)
            robot.set_joint_position_target(q, joint_ids=arm_ids)
            for _ in range(POSE_SWEEP_SETTLE_STEPS):
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt)

            tip = self._finger_tip_pos()                               # [N, 3]
            tip_local = tip - self.scene.env_origins
            seen_min = torch.minimum(seen_min, tip_local.min(dim=0).values)
            seen_max = torch.maximum(seen_max, tip_local.max(dim=0).values)
            ee_z = robot.data.body_pos_w[:, self._ee_body_idx, 2]
            in_box = (
                (tip_local[:, 0] > TIP_BOX_X[0]) & (tip_local[:, 0] < TIP_BOX_X[1])
                & (tip_local[:, 1] > TIP_BOX_Y[0]) & (tip_local[:, 1] < TIP_BOX_Y[1])
                & (tip_local[:, 2] > TIP_BOX_Z[0]) & (tip_local[:, 2] < TIP_BOX_Z[1])
            )
            down = (ee_z - tip[:, 2]) > TIP_DOWN_MIN_DROP
            # A pose is only useful if at least the shortest bank states fit
            # below its tip (pose-dependent drape limit).
            allowed = self._allowed_drape(tip_local)
            fits = allowed >= DRAPE_LEVELS[0]
            # Read back actual joint positions (PD equilibrium ≈ q, but store
            # the truth so reset writes reproduce the recorded tip).
            ok = in_box & down & fits
            if ok.any():
                q_now = robot.data.joint_pos[:, arm_ids]
                bank_q.append(q_now[ok].clone())
                bank_tip.append(tip_local[ok].clone())
                bank_drape.append(allowed[ok].clone())
                n_accepted += int(ok.sum())

        if n_accepted == 0:
            raise RuntimeError(
                "Holding-pose sweep found NO pose with the finger tip inside "
                f"TIP_BOX x{TIP_BOX_X} y{TIP_BOX_Y} z{TIP_BOX_Z} after "
                f"{POSE_SWEEP_MAX_ROUNDS} rounds — sampled tips spanned "
                f"x[{seen_min[0]:.2f},{seen_max[0]:.2f}] "
                f"y[{seen_min[1]:.2f},{seen_max[1]:.2f}] "
                f"z[{seen_min[2]:.2f},{seen_max[2]:.2f}]; "
                "check the robot mount/box."
            )
        self._pose_bank_q = torch.cat(bank_q, dim=0)[:POSE_BANK_TARGET]
        self._pose_bank_tip = torch.cat(bank_tip, dim=0)[:POSE_BANK_TARGET]
        self._pose_bank_drape = torch.cat(bank_drape, dim=0)[:POSE_BANK_TARGET]
        if n_accepted < POSE_BANK_MIN:
            log.warning(
                "Holding-pose bank has only %d poses (< %d) — initial-state "
                "diversity is thin; widen TIP_BOX or raise POSE_SWEEP_MAX_ROUNDS.",
                n_accepted, POSE_BANK_MIN,
            )
        tips = self._pose_bank_tip
        log.info(
            "Holding-pose bank: %d poses; tip x [%.2f, %.2f] y [%.2f, %.2f] "
            "z [%.2f, %.2f]",
            self._pose_bank_q.shape[0],
            tips[:, 0].min(), tips[:, 0].max(),
            tips[:, 1].min(), tips[:, 1].max(),
            tips[:, 2].min(), tips[:, 2].max(),
        )

    # ------------------------------------------------------------------
    # Cloth reset: restore the hang at the sampled holding pose's fingertip
    # ------------------------------------------------------------------

    def _reset_cloth(self, env_ids: torch.Tensor) -> None:
        """Grasped-hang reset (§ recipe): pose → closed gripper → bank hang."""
        k = env_ids.numel()
        if k == 0:
            return
        dev = self.device
        robot: Articulation = self.scene["robot"]

        idx = torch.randint(0, self._pose_bank_q.shape[0], (k,), device=dev)
        q = self._pose_bank_q[idx]
        tip_local = self._pose_bank_tip[idx]

        robot.write_joint_state_to_sim(
            q, torch.zeros_like(q), joint_ids=self._arm_joint_ids, env_ids=env_ids,
        )
        robot.set_joint_position_target(
            q, joint_ids=self._arm_joint_ids, env_ids=env_ids,
        )
        self._force_gripper_closed(env_ids)

        anchors = torch.zeros(self.num_envs, 3, device=dev)
        anchors[env_ids] = self.scene.env_origins[env_ids] + tip_local

        # Bucket by the pose's admissible drape (pose-dependent: drum tops /
        # belt / free space — computed at sweep time) quantized DOWN to the
        # nearest DRAPE_LEVEL (the restore helper takes one max_drape/call);
        # sweep acceptance guarantees allowed ≥ DRAPE_LEVELS[0].
        allowed = self._pose_bank_drape[idx]
        restored = torch.zeros(k, dtype=torch.bool, device=dev)
        levels = torch.tensor(DRAPE_LEVELS, device=dev)
        assigned = (torch.bucketize(allowed, levels, right=True) - 1).clamp(min=0)
        for li in range(len(DRAPE_LEVELS)):
            sel = assigned == li
            if not sel.any():
                continue
            ok = self._reset_cloth_hanging_from_bank(
                env_ids[sel], anchors, slot=0, max_drape=float(DRAPE_LEVELS[li]),
            )
            restored |= sel & ok
        leftover = ~restored
        if leftover.any():
            # No hanging bank loaded: idealized centre-pin fallback (NOT
            # settled — only for smoke runs without the bank asset).
            ids = env_ids[leftover]
            origins = self.scene.env_origins[ids]
            centroids = anchors[ids].clone()
            yaw = torch.zeros(ids.numel(), device=dev)
            self._cloth.reset_randomized(ids, centroids, yaw)
            self._cloth.update()
            self._cloth.attach(ids, anchors, 0.07, slot=0)

        # The restored attachment IS a grasp: latch was_grasped so the
        # release/success gates are armed from step 0.
        self._was_grasped[env_ids] = True
        # Arm the reset-step anchor correction: robot.data body poses are
        # STALE until the next physics step (write_joint_state_to_sim does not
        # refresh the lazily cached body-state buffers), so the base
        # ``_update_grasp``'s ``hold(tip)`` would yank the restored cloth to
        # the PRE-reset fingertip for one step.  ``_update_grasp`` (overridden
        # below) substitutes this anchor for exactly one step.
        self._reset_anchor[env_ids] = anchors[env_ids]
        self._reset_anchor_pending[env_ids] = True

    # ------------------------------------------------------------------
    # Grasp update with reset-step anchor correction
    # ------------------------------------------------------------------

    def _update_grasp(self) -> None:
        """Base attach/hold/detach, with the stale-FK reset-step fix.

        Identical to ``ClothSortingEnvBase._update_grasp`` except that for
        envs reset THIS step the dynamic fingertip (computed from the stale
        ``robot.data`` body poses) is replaced by the recorded reset anchor —
        the pose bank's true PD-equilibrium tip — so the restored hang is not
        dragged to the pre-reset gripper position for one step.
        """
        from ..shared.cloth_sorting_env import (
            ATTACH_TRIGGER_DIST,
            ATTACH_WELD_RADIUS,
            FINGER_CMD_CLOSE,
            FINGER_CMD_OPEN,
        )

        robot: Articulation = self.scene["robot"]
        self._drive_gripper(robot)
        if not self.enable_hand_grasp:
            return

        if hasattr(robot.data, "joint_pos_target"):
            finger_cmd = robot.data.joint_pos_target[:, self._finger_joint_idx]
        else:
            finger_cmd = robot.data.joint_pos[:, self._finger_joint_idx]

        tip = self._finger_tip_pos()
        if self._reset_anchor_pending.any():
            tip = torch.where(
                self._reset_anchor_pending.unsqueeze(1), self._reset_anchor, tip
            )
            self._reset_anchor_pending[:] = False
        target = self.shirt_grasp_point_w
        near = torch.norm(tip - target, dim=-1) < ATTACH_TRIGGER_DIST
        closing = finger_cmd > FINGER_CMD_CLOSE
        opening = finger_cmd < FINGER_CMD_OPEN
        attached = self._cloth.is_attached

        to_attach = closing & near & (~attached)
        if to_attach.any():
            self._cloth.attach(
                torch.nonzero(to_attach, as_tuple=False).squeeze(-1),
                tip, ATTACH_WELD_RADIUS,
            )
        to_detach = attached & opening
        if to_detach.any():
            self._cloth.detach(torch.nonzero(to_detach, as_tuple=False).squeeze(-1))
        self._cloth.hold(tip)

    # ------------------------------------------------------------------
    # Step / reset
    # ------------------------------------------------------------------

    def step(self, action: torch.Tensor):
        prev_attached = self.grasp_active.clone()
        obs, reward, terminated, time_outs, extras = super().step(action)
        # NOTE: super().step() ran _update_grasp() AFTER the physics step, so
        # grasp_active now reflects this step's attach/detach transitions.

        still_running = ~(terminated | time_outs)

        # ── One-shot graded release event over the COMMANDED bin ─────────
        released = prev_attached & (~self.grasp_active)
        target = self.target_bin_pos_w
        gp = self.shirt_grasp_point_w
        d_xy = torch.norm(gp[:, :2] - target[:, :2], dim=-1)
        local_z = gp[:, 2] - self.scene.env_origins[:, 2]
        good_release = (
            released
            & (d_xy < _DRUM_GEOM.radius)
            & (local_z > DRUM_TOP + RELEASE_TOP_CLEARANCE)
        )
        rim = DRUM_TOP + CLEAR_RIM_CLEARANCE
        bottom_z = self.shirt_min_z_w - self.scene.env_origins[:, 2]
        centering = torch.clamp(1.0 - d_xy / _DRUM_GEOM.radius, 0.0, 1.0)
        height_q = torch.clamp((bottom_z - (rim - 0.10)) / 0.10, 0.0, 1.0)
        quality = centering * height_q
        self._release_event = good_release.float() * (0.25 + 0.75 * quality)
        self._bad_release_event = (released & ~good_release).float()
        self._was_released[still_running] |= released[still_running]

        # ── Correct distribution: released cloth settled in the commanded
        #    drum (release-required, mirrors shirt_place) ──────────────────
        free = ~self.grasp_active
        frac = self._target_fraction_in_bin() * free.float()
        self._max_target_fraction[still_running] = torch.maximum(
            self._max_target_fraction[still_running], frac[still_running]
        )
        distributed = (frac > DISTRIBUTE_FRACTION_THRESHOLD) & self._was_grasped & free
        self._was_distributed[still_running] |= distributed[still_running]
        return obs, reward, terminated, time_outs, extras

    def _reset_idx(self, env_ids: Sequence[int]):
        env_ids_t = (
            torch.tensor(env_ids, device=self.device, dtype=torch.long)
            if not isinstance(env_ids, torch.Tensor)
            else env_ids
        )
        fin_dist = fin_rel = fin_bin = fin_frac = None
        if len(env_ids_t) > 0:
            fin_dist = self._was_distributed[env_ids_t].float()
            fin_rel = self._was_released[env_ids_t].float()
            fin_bin = self._target_bin[env_ids_t]
            fin_frac = self._max_target_fraction[env_ids_t].mean()
            # Exact per-episode outcomes of the batch that just finished —
            # consumed by scripts/skrl/evaluate_shirt_distribute.py for the
            # per-bin breakdown (extras log only carries batch means).
            self.last_finished_batch = {
                "bin": fin_bin.clone(),
                "success": fin_dist.clone(),
                "released": fin_rel.clone(),
            }
            # Resample the commanded bin BEFORE the parent reset so reset-step
            # observations already see the new goal.
            self._target_bin[env_ids_t] = torch.randint(
                len(DRUM_POSITIONS), (len(env_ids_t),), device=self.device
            )
        result = super()._reset_idx(env_ids)
        # Metric writes must come AFTER super()._reset_idx(): the manager
        # rebuilds extras["log"] in there — writes made before are silently
        # wiped (found in run 1: distribute_success_rate never reached TB).
        if fin_dist is not None:
            self.extras["log"]["Metrics/distribute_success_rate"] = fin_dist.mean()
            self.extras["log"]["Metrics/release_rate"] = fin_rel.mean()
            self.extras["log"]["Metrics/max_target_fraction"] = fin_frac
            # Per-bin success (only when the batch contains that bin — the
            # scalar means over exactly the envs that just finished).
            for b in range(len(DRUM_POSITIONS)):
                m = fin_bin == b
                if m.any():
                    self.extras["log"][f"Metrics/distribute_success_bin{b}"] = (
                        fin_dist[m].mean()
                    )
        self._was_distributed[env_ids_t] = False
        self._was_released[env_ids_t] = False
        self._release_event[env_ids_t] = 0.0
        self._bad_release_event[env_ids_t] = 0.0
        self._max_target_fraction[env_ids_t] = 0.0
        return result
