# shirt_place_env.py
#
# Custom environment for the shirt placement task.
#
# Integrates PBD particle cloth (via ``ClothObject``) with a kinematic
# rigid-body proxy (``shirt_proxy``) that tracks the cloth centroid.
# All reward / observation functions read from the proxy, so they work
# without modification.
#
# Lifecycle:
#   1. Startup event spawns cloth mesh + applies PBD schemas (before sim.reset)
#   2. super().__init__() → scene creation, cloning, sim.reset()
#   3. ClothObject() acquires ParticleClothView (GPU tensor access)
#   4. Each step: cloth.update() → sync proxy to centroid → rewards
#   5. Each reset: cloth.reset(env_ids)
#
# Also adds latched ``was_grasped`` and ``was_placed`` flags.

from __future__ import annotations

from typing import Sequence

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils.math import quat_apply

from .shirt_place_scene_cfg import (
    CONVEYOR_SURFACE_HEIGHT_M,
    SHIRT_BELT_CLEARANCE_M,
    SHIRT_CLOTH_CFG,
    SHIRT_REST_XY,
    UPSTREAM_SETTLE_POS,
)
from .mdp.rewards import (
    FINGER_JOINT_CLOSE_POS,
    FINGER_TIP_CLOSED_Z,
    FINGER_TIP_OPEN_Z,
    GRASP_CENTER_LOCAL_Z,
    BinCylinder,
    _in_upright_cylinder,
    _get_world_pos,
)
from ..shared.cloth_object import ClothObject


EE_BODY_CANDIDATES = ("tool_link_0", "robotiq_base_link", "end_effector_link")
SHIRT_PROXY_KEY = "shirt_proxy"
DRUM_KEY = "drum_target"
FINGER_JOINT_NAME = "finger_joint"

# ── Deterministic attachment-grasp thresholds ─────────────────────────
# The gripper grasps the shirt at its highest point by welding the nearby cloth
# particles to the *dynamic finger tip* (the gripper's lowest point, which moves
# with closure) — a top-down grasp, validated in run_shirt_validation.py.  Attach
# fires when the gripper is commanded closed AND the tip has reached the cloth's
# highest point; detach fires when it is commanded open (robust to the physical
# gripper jamming).
# Small, gripper-pad-sized capture region (mirrors the GarmentLab / DexGarmentLab
# AttachmentBlock).  The old 0.18 m radius welded a ~36 cm blob (hundreds of
# vertices), dragged neighbours along ("force field"), and — with the tip
# attaching from up to 0.12 m away — floated the patch far below the tip.  A tight
# trigger + small radius grip only the actually-pinched cluster, at the tip.
ATTACH_TRIGGER_DIST = 0.10   # finger tip → highest point distance to allow attach
ATTACH_WELD_RADIUS = 0.07    # particles within this of the tip are welded
# ^ Sized to the arm's reach: base_z bottoms out ≈0.05 m above the belt-level flat
# cloth, so the radius must bridge that vertical gap plus a small pad footprint.
# For a flat sheet ~0.05 m below the tip this captures a ~0.05 m-radius disc
# (≈80 vertices — a pinch), vs the old 0.18 m blob (~1000 vertices).  Once welded,
# the recentred offsets hold the cluster *at* the tip, so there is no float gap.
FINGER_CMD_CLOSE = 0.40      # commanded finger target above this ⇒ closing
FINGER_CMD_OPEN = 0.40       # commanded finger target below this ⇒ opening → release

# Robotiq 2F-140 four-bar mimic ratios (relative to ``finger_joint``).  The five
# passive joints have zero drive stiffness, so the env kinematically forces the
# whole linkage to this coherent pose each step — otherwise it drifts into a
# broken/inverted state and the gripper can no longer open.  (Same ratios as
# run_shirt_validation.py.)
GRIPPER_MIMIC = {
    "right_outer_knuckle_joint":    1.0,
    "left_outer_finger_joint":      0.0,
    "right_outer_finger_joint":     0.0,
    "left_inner_finger_joint":     -1.0,
    "right_inner_finger_joint":    -1.0,
    "left_inner_finger_pad_joint":  1.0,
    "right_inner_finger_pad_joint": 1.0,
}
# Rate (rad/s) at which the gripper is kinematically driven toward its commanded
# closure.  The native finger_joint PD drive (stiffness 11.25, velocity_limit
# 1.0 rad/s) and the inner-finger drive (stiffness 10, target 0) fight the
# linkage write, so the visible pads barely move; we instead drive the *whole*
# gripper kinematically.  ~4 rad/s closes the 0.785 rad span in ≈0.2 s — a smooth,
# visible close.
GRIPPER_KIN_RATE = 4.0
PLACE_FRACTION_THRESHOLD = 0.15  # cloth-fraction-in-drum counted as a placement
BELT_HEIGHT = CONVEYOR_SURFACE_HEIGHT_M
# Anti-hover: a release counts as "good" (earns the one-time release bonus) when
# the gripper opens with the grasp point centred over the drum footprint and above
# its rim.  After a placement latches, the episode ends this many control steps
# later so the per-step shaping can no longer be farmed by holding.
RELEASE_RIM_CLEARANCE = 0.10
PLACE_SETTLE_STEPS = 15
# Clearance predicate (must match the ``clearance_over_drum`` reward params) used
# to fade out the per-step "positioned over the drum" shaping as the shirt clears,
# so hovering-cleared is a reward desert and only releasing pays.
CLEAR_RIM_CLEARANCE = 0.08
CLEAR_MARGIN = 0.20
CLEAR_DRUM_RADIUS = 0.32

# Drum geometry — must match _BIN_GEOM in shirt_place_env_cfg.py.  Height spans
# the drum interior depth (not the 0.30 m used for the rigid cube) so draped
# cloth anywhere inside the drum is counted as placed.
_DRUM_GEOM = BinCylinder(radius=0.547 * 0.5, height=0.85)


class TensegrityShirtPlaceEnv(ManagerBasedRLEnv):
    """Manager-based RL env for the tensegrity shirt-placement task.

    Extends :class:`ManagerBasedRLEnv` with:
      - PBD particle cloth via ``ClothObject``
      - Kinematic proxy synced to cloth centroid
      - Latched ``was_grasped`` / ``was_placed`` flags
    """

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        # ── Robot body / joint indices ────────────────────────────────
        robot: Articulation = self.scene["robot"]
        ee_body_name = next(n for n in EE_BODY_CANDIDATES if n in robot.body_names)
        self._ee_body_idx: int = robot.body_names.index(ee_body_name)
        self._finger_joint_idx: int = robot.joint_names.index(FINGER_JOINT_NAME)

        # Robotiq four-bar linkage joints kinematically mimicked each step so the
        # gripper stays coherent and can always re-open (see GRIPPER_MIMIC).
        self._mimic_ids: list[int] = [robot.joint_names.index(j) for j in GRIPPER_MIMIC]
        self._mimic_ratios: torch.Tensor = torch.tensor(
            [GRIPPER_MIMIC[j] for j in GRIPPER_MIMIC],
            device=self.device, dtype=torch.float32,
        )

        # ── Cloth object (GPU tensor access) ──────────────────────────
        self._cloth = ClothObject(
            cfg=SHIRT_CLOTH_CFG,
            num_envs=self.num_envs,
            device=self.device,
        )

        # ── Latched flags + cloth-aware success metric ────────────────
        self._was_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._was_placed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Anti-hover: one-shot flag set for the single step a good release happens
        # (grasp opened while centred over + cleared above the drum), consumed by
        # the ``release_event`` reward.  ``_steps_since_place`` counts control steps
        # since the placement latched, so the episode can terminate shortly after a
        # successful drop (the shaping can no longer be farmed by holding).
        self._release_event = torch.zeros(self.num_envs, device=self.device)
        self._steps_since_place = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device,
        )
        # Peak fraction of cloth particles inside the drum during the episode —
        # a cloth-aware success signal (centroid-in-drum can be true while most
        # of the shirt hangs outside).
        self._max_drum_fraction = torch.zeros(self.num_envs, device=self.device)

        # One-off pre-settle: collapse the raw closed garment mesh into a
        # relaxed flat sheet and adopt it as the canonical rest shape, so
        # per-reset placement no longer triggers a self-collision pop.
        self._presettle_cloth()

    # ------------------------------------------------------------------
    # One-off pre-settle of the cloth rest shape
    # ------------------------------------------------------------------

    def _presettle_cloth(self, steps: int = 320) -> None:
        """Settle one shirt flat on the upstream (free) belt and reuse the shape.

        Settling away from the robot — on the empty upstream conveyor — gives a
        clean relaxed sheet without the gripper interfering, which is then the
        canonical rest shape teleported to the work area at each reset.  The
        garment is released from a height so it pancakes flat as it lands instead
        of freezing in the source mesh's puffy 3D shape.
        """
        try:
            ids = torch.arange(self.num_envs, device=self.device)
            origins = self.scene.env_origins
            drop = self._cloth.cfg.pbd_params.presettle_drop_height_m
            z_local = (
                CONVEYOR_SURFACE_HEIGHT_M + drop - self._cloth.flat_rest_min_z
            )
            centroids = origins.clone()
            centroids[:, 0] = origins[:, 0] + UPSTREAM_SETTLE_POS[0]
            centroids[:, 1] = origins[:, 1] + UPSTREAM_SETTLE_POS[1]
            centroids[:, 2] = origins[:, 2] + z_local
            yaws = torch.zeros(self.num_envs, device=self.device)
            self._cloth.reset_randomized(ids, centroids, yaws)

            dt = self.cfg.sim.dt
            for _ in range(steps):
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt)
            self._cloth.recompute_flat_rest_from_current()
        except Exception as exc:  # pragma: no cover - safety net
            import logging
            logging.getLogger(__name__).warning(
                "Cloth pre-settle skipped (%s); using geometric rest shape.", exc
            )

    # ------------------------------------------------------------------
    # Cloth access (for external use)
    # ------------------------------------------------------------------

    @property
    def cloth(self) -> ClothObject:
        return self._cloth

    # ------------------------------------------------------------------
    # Randomized cloth placement
    # ------------------------------------------------------------------

    def _reset_cloth(self, env_ids: torch.Tensor) -> None:
        """Lay the shirt flat at a *fixed* pose directly under the gripper.

        First iteration: no fold, no crumble, no XY/yaw randomization — the
        garment is laid flat and centred under the gripper's default grasp
        position so the policy can focus on grasp → lift → transport → place.
        """
        n = env_ids.numel()
        if n == 0:
            return
        dev = self.device
        origins = self.scene.env_origins[env_ids]
        # World centroid: fixed XY under the gripper; Z so the shirt's lowest
        # point rests just above the belt surface.
        z_local = CONVEYOR_SURFACE_HEIGHT_M + SHIRT_BELT_CLEARANCE_M - self._cloth.flat_rest_min_z
        centroids = origins.clone()
        centroids[:, 0] = origins[:, 0] + SHIRT_REST_XY[0]
        centroids[:, 1] = origins[:, 1] + SHIRT_REST_XY[1]
        centroids[:, 2] = origins[:, 2] + z_local
        yaw = torch.zeros(n, device=dev)
        self._cloth.reset_randomized(env_ids, centroids, yaw)

    # ------------------------------------------------------------------
    # Grasp centre helper
    # ------------------------------------------------------------------

    def _finger_tip_pos(self) -> torch.Tensor:
        """World position of the *dynamic finger tip* ``[N, 3]``.

        The gripper's lowest point, which descends as the gripper closes
        (``FINGER_TIP_OPEN_Z`` → ``FINGER_TIP_CLOSED_Z``).  All grasp computations
        use the tip rather than the static pad centre, so the robot brings the
        tip straight down onto the cloth (top-down grasp) instead of aligning the
        pad centre and approaching from the side.
        """
        robot: Articulation = self.scene["robot"]
        finger_pos = robot.data.joint_pos[:, self._finger_joint_idx]
        closure = torch.clamp(finger_pos / FINGER_JOINT_CLOSE_POS, 0.0, 1.0)
        tip_z = FINGER_TIP_OPEN_Z + closure * (FINGER_TIP_CLOSED_Z - FINGER_TIP_OPEN_Z)
        ee_pos = robot.data.body_pos_w[:, self._ee_body_idx, :]
        ee_quat = robot.data.body_quat_w[:, self._ee_body_idx, :]
        offset = torch.stack(
            [torch.zeros_like(tip_z), torch.zeros_like(tip_z), tip_z], dim=-1
        )
        return ee_pos + quat_apply(ee_quat, offset)

    def _drive_gripper(self, robot: Articulation) -> None:
        """Kinematically drive the *whole* Robotiq gripper to its commanded pose.

        The visible finger pads close via the inner-finger joints, but their PD
        drive (stiffness 10, target 0) yanks them back each step and fights any
        linkage write, so the gripper looked stuck open even while the weld grasp
        (which keys off the *commanded* target) fired.  Here we instead drive the
        entire linkage kinematically toward the commanded closure:

          * read the binary action's commanded finger target (0 open ↔ 0.785 closed),
          * rate-limit the finger toward it (smooth, visible close in ≈0.2 s),
          * write finger + the seven mimic joints to the coherent four-bar pose, and
          * set their PD targets to the *same* pose so the actuators stop fighting.

        This makes the visual gripper match the command (and the deterministic
        weld), and keeps the four-bar linkage sound so it can always re-open.
        """
        cmd = robot.data.joint_pos_target[:, self._finger_joint_idx]          # [N]
        cmd = torch.clamp(cmd, 0.0, FINGER_JOINT_CLOSE_POS)
        cur = robot.data.joint_pos[:, self._finger_joint_idx]                 # [N]
        max_step = GRIPPER_KIN_RATE * self.cfg.sim.dt * self.cfg.decimation
        finger = cur + torch.clamp(cmd - cur, -max_step, max_step)            # [N]

        # Coherent four-bar pose: [finger_joint, *mimic_joints].
        ids = [self._finger_joint_idx] + self._mimic_ids
        ratios = torch.cat(
            [torch.ones(1, device=self.device), self._mimic_ratios]
        )                                                                     # [1+M]
        pose = finger.unsqueeze(1) * ratios.unsqueeze(0)                      # [N, 1+M]
        robot.write_joint_position_to_sim(pose, joint_ids=ids)
        robot.write_joint_velocity_to_sim(torch.zeros_like(pose), joint_ids=ids)
        # Align PD targets so the implicit actuators do not fight the kinematic
        # pose (otherwise the inner-finger drive pulls the pads back open).
        robot.set_joint_position_target(pose, joint_ids=ids)

    @property
    def shirt_grasp_point_w(self) -> torch.Tensor:
        """Deterministic grasp target — the cloth's highest region ``[N, 3]``.

        On a flat shirt this is the top surface near the centre; once grasped and
        lifted, the welded patch is the highest region, so this tracks the tip.
        """
        return self._cloth.highest_point_w

    @property
    def shirt_min_z_w(self) -> torch.Tensor:
        """World Z of the lowest cloth particle per env ``[N]``.

        Used to check the *entire* hanging shirt clears the drum rim before the
        drop, so it falls into the opening instead of draping over the edge.
        """
        return self._cloth.nodal_pos_w[:, :, 2].min(dim=1).values

    # ------------------------------------------------------------------
    # Deterministic attachment grasp (weld at the highest point)
    # ------------------------------------------------------------------

    def _update_grasp(self) -> None:
        """Attach / hold / detach the welded cloth grasp each control step.

        Attach when the gripper is closing AND its grasp centre has reached the
        shirt's highest point; hold the welded patch on the tip; detach on open.
        """
        robot: Articulation = self.scene["robot"]
        # Drive the whole gripper kinematically to its commanded pose so the
        # visible pads close/open in lockstep with the command (and the weld).
        self._drive_gripper(robot)

        # Use the *commanded* finger target (robust to a physically jammed
        # gripper) to decide grasp/release.
        if hasattr(robot.data, "joint_pos_target"):
            finger_cmd = robot.data.joint_pos_target[:, self._finger_joint_idx]
        else:
            finger_cmd = robot.data.joint_pos[:, self._finger_joint_idx]

        tip = self._finger_tip_pos()                      # [N, 3] dynamic tip
        high = self._cloth.highest_point_w                # [N, 3]
        near = torch.norm(tip - high, dim=-1) < ATTACH_TRIGGER_DIST
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
            self._cloth.detach(
                torch.nonzero(to_detach, as_tuple=False).squeeze(-1)
            )
        # Track all currently-attached patches to the live finger tip.
        self._cloth.hold(tip)

    @property
    def grasp_active(self) -> torch.Tensor:
        """Per-env bool: True while the shirt is held by the welded grasp."""
        return self._cloth.is_attached

    @property
    def clearance_fraction(self) -> torch.Tensor:
        """Smooth [0,1] "shirt fully cleared over the drum" signal ``[N]``.

        1.0 when the grasp point is over the drum footprint AND the whole shirt
        (lowest particle) is ``CLEAR_MARGIN`` above the rim; 0 otherwise.  Used to
        fade the per-step positioning shaping so a cleared-over-drum shirt earns
        almost nothing unless it is released (breaks the hover-over-drum trap).
        """
        drum_pos = _get_world_pos(self.scene[DRUM_KEY], env=self)
        gp = self.shirt_grasp_point_w
        d_xy = torch.norm(gp[:, :2] - drum_pos[:, :2], dim=-1)
        in_xy = d_xy < CLEAR_DRUM_RADIUS
        rim = BELT_HEIGHT + CLEAR_RIM_CLEARANCE
        bottom_z = self.shirt_min_z_w - self.scene.env_origins[:, 2]
        clr = torch.clamp((bottom_z - rim) / CLEAR_MARGIN, 0.0, 1.0)
        gate = in_xy & self._was_grasped
        return torch.where(gate, clr, torch.zeros_like(clr))

    @property
    def was_grasped(self) -> torch.Tensor:
        return self._was_grasped

    @property
    def was_placed(self) -> torch.Tensor:
        return self._was_placed

    @property
    def release_event(self) -> torch.Tensor:
        """Per-env one-shot ``[num_envs]``: 1.0 the step a good release fired."""
        return self._release_event

    @property
    def steps_since_place(self) -> torch.Tensor:
        """Control steps since a placement latched (``-1`` if not yet placed)."""
        return self._steps_since_place

    def _shirt_in_drum(self) -> torch.Tensor:
        shirt = self.scene[SHIRT_PROXY_KEY]
        drum_pos = _get_world_pos(self.scene[DRUM_KEY], env=self)
        return _in_upright_cylinder(shirt.data.root_pos_w, drum_pos, _DRUM_GEOM)

    def _shirt_particle_fraction_in_drum(self) -> torch.Tensor:
        """Fraction of cloth particles inside the drum cylinder, per env (N,)."""
        drum_pos = _get_world_pos(self.scene[DRUM_KEY], env=self)  # [N, 3]
        pts = self._cloth.nodal_pos_w                              # [N, P, 3]
        inside = _in_upright_cylinder(pts, drum_pos, _DRUM_GEOM)   # [N, P] bool
        return inside.float().mean(dim=1)

    # ------------------------------------------------------------------
    # Step: update cloth → sync proxy → update latched flags
    # ------------------------------------------------------------------

    def _sync_proxy_to_cloth(self) -> None:
        """Teleport the kinematic proxy to the cloth centroid."""
        proxy: RigidObject = self.scene[SHIRT_PROXY_KEY]
        centroid = self._cloth.centroid_pos_w  # [N, 3]
        # Build root state: [pos(3), quat(4), lin_vel(3), ang_vel(3)] = 13
        root_state = torch.zeros(
            self.num_envs, 13, device=self.device, dtype=torch.float32,
        )
        root_state[:, :3] = centroid
        root_state[:, 3] = 1.0  # quat w=1 (identity)
        # lin_vel from cloth centroid velocity
        root_state[:, 7:10] = self._cloth.centroid_vel_w
        proxy.write_root_state_to_sim(root_state)

    def step(self, action: torch.Tensor):
        obs, reward, terminated, time_outs, extras = super().step(action)

        # Update cloth state from GPU buffers
        self._cloth.update()
        # Attach / hold / detach the deterministic welded grasp.  Capture the
        # pre-update grasp state so a release (attached → open) can be detected.
        prev_attached = self.grasp_active.clone()
        self._update_grasp()
        # Sync the kinematic proxy to the cloth centroid
        self._sync_proxy_to_cloth()

        # ── One-time release bonus ────────────────────────────────────────
        # Fire for the single step the gripper opens while the grasp point is
        # centred over the drum footprint and above its rim (a sensible drop),
        # consumed by the ``release_event`` reward next step.
        released = prev_attached & (~self.grasp_active)
        drum_pos = _get_world_pos(self.scene[DRUM_KEY], env=self)
        gp = self.shirt_grasp_point_w
        d_xy = torch.norm(gp[:, :2] - drum_pos[:, :2], dim=-1)
        local_z = gp[:, 2] - self.scene.env_origins[:, 2]
        good_release = (
            released
            & (d_xy < _DRUM_GEOM.radius)
            & (local_z > BELT_HEIGHT + RELEASE_RIM_CLEARANCE)
        )
        self._release_event = good_release.float()

        still_running = ~(terminated | time_outs)
        self._was_grasped[still_running] |= self.grasp_active[still_running]
        # A placement counts only when the shirt is a) in the drum AND b) *released*
        # (not held).  Requiring the release closes the reward-hack where the policy
        # simply lowers the still-gripped shirt into the drum so particles register
        # inside without ever dropping it — the task wants the shirt held above the
        # drum, then let go so it falls in.  The fraction metric likewise counts
        # only released cloth, so it honestly reflects "dropped in".
        released = ~self.grasp_active
        frac = self._shirt_particle_fraction_in_drum() * released.float()
        self._max_drum_fraction[still_running] = torch.maximum(
            self._max_drum_fraction[still_running], frac[still_running]
        )
        placed_now = (frac > PLACE_FRACTION_THRESHOLD) & self._was_grasped & released
        self._was_placed[still_running] |= placed_now[still_running]
        # Count control steps since a placement latched (for terminate-after-place).
        newly_placed = placed_now & (self._steps_since_place < 0) & still_running
        self._steps_since_place[newly_placed] = 0
        counting = (self._steps_since_place >= 0) & still_running
        self._steps_since_place[counting] += 1
        return obs, reward, terminated, time_outs, extras

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int]):
        env_ids_t = (
            torch.tensor(env_ids, device=self.device, dtype=torch.long)
            if not isinstance(env_ids, torch.Tensor)
            else env_ids
        )

        grasp_rate = torch.tensor(0.0, device=self.device)
        place_rate = torch.tensor(0.0, device=self.device)
        mean_ep_len = torch.tensor(0.0, device=self.device)
        drum_fraction = torch.tensor(0.0, device=self.device)
        if len(env_ids_t) > 0:
            grasp_rate = self._was_grasped[env_ids_t].float().mean()
            place_rate = self._was_placed[env_ids_t].float().mean()
            mean_ep_len = self.episode_length_buf[env_ids_t].float().mean()
            drum_fraction = self._max_drum_fraction[env_ids_t].mean()

        result = super()._reset_idx(env_ids)
        self._was_grasped[env_ids_t] = False
        self._was_placed[env_ids_t] = False
        self._max_drum_fraction[env_ids_t] = 0.0
        self._release_event[env_ids_t] = 0.0
        self._steps_since_place[env_ids_t] = -1

        # Release any welded grasp, then lay the shirt flat at the fixed pose and
        # sync the proxy/cloth buffers so reset-step observations are consistent.
        self._cloth.reset_attachment(env_ids_t)
        self._reset_cloth(env_ids_t)
        self._cloth.update()
        self._sync_proxy_to_cloth()

        self.extras["log"]["Metrics/grasp_rate"] = grasp_rate
        self.extras["log"]["Metrics/place_success_rate"] = place_rate
        self.extras["log"]["Metrics/mean_episode_length"] = mean_ep_len
        self.extras["log"]["Metrics/shirt_in_drum_fraction"] = drum_fraction
        return result
