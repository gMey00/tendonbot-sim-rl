"""Shared harness for the robot-free shirt-presentation heuristic study.

Import AFTER ``AppLauncher(...).app`` has been created (isaaclab import order).
Provides ``StudyRig``: boots ``Template-Shirt-Pick-Tensegrity-v0`` with manual
stepping (no manager resets), restores hanging states from the cached bank at a
raised upstream anchor, and implements the study's single canonical
grasp → stretch → damped-settle → measure procedure used by every run script.

Conventions
-----------
* All experiments happen at a per-env "study anchor": the upstream settle
  position, ``ANCHOR_HEIGHT_M`` above the belt surface (raised vs the bank's
  1.0 m so a full-drape hang PLUS a downward stretch stays clear of the belt).
* Grasps are ``ClothObject`` attachment slots (pad radius 0.07 m — one grasp
  point).  Slot roles per trial: the *top* slot pins the hang anchor, the
  *moving* slot executes the stretch.
* The stretch is the HORIZONTAL presentation stretch: the second anchor moves
  (straight line, ``STRETCH_SPEED``) to the SAME HEIGHT as the first, offset
  along world X — the camera-plane horizontal — at ``target_ratio ×
  rest_distance`` (tautness guard ≤ 1.15).  The taut chord ends up horizontal
  in the image plane and GRAVITY provides the vertical extension of the
  garment below it.  Pull direction ±x defaults to the moving point's current
  side (no dragging the cloth across itself) and is override-able (the
  regrasp heuristic stretches opposite to the previous stage).
* Every trial row records the bank state index, both grasp particle indices
  (+ their flat-rest coordinates), yaw/mirror augmentation, target and
  measured stretch ratio, coverage (plane-aligned + yaw-marginalized),
  extents, drape and settle steps → any figure is reproducible from the CSV.

Settle protocol (copied from scripts/generate_hanging_bank.py): a damped
phase (per-step velocity scale 0.97 — synthetic air drag, PBD cloth pendulums
for tens of seconds otherwise) until ROBUST criteria fire (p95 particle speed
+ centroid speed — the max never converges), then a short free phase.
"""

from __future__ import annotations

import csv
import os
import time

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import tensegrity_pick.tasks  # noqa: F401
from tensegrity_pick.tasks.manager_based.shared.cloth_metrics import (
    flat_silhouette_area,
    plane_extent,
    rest_distance,
    silhouette_coverage,
    stretch_ratio,
    two_sided_visible_fraction,
)
from tensegrity_pick.tasks.manager_based.shared.cloth_sorting_env import (
    HANG_ANCHOR_RADIUS,
)
from tensegrity_pick.tasks.manager_based.shared.cloth_sorting_scene_cfg import (
    CONVEYOR_SURFACE_HEIGHT_M,
    PROJ_ASSETS_PATH,
    UPSTREAM_SETTLE_POS,
)

# ── Study protocol constants ───────────────────────────────────────────────
ANCHOR_HEIGHT_M = 1.3      # study anchor above the belt: the garment drapes at
                           # most ~0.95 m (bank max) below the horizontal chord
STRETCH_SPEED = 0.45       # m/s moving-anchor speed during the stretch
RAISE_SPEED = 0.6          # m/s re-hang raise speed (H2 regrasp), = bank lift
TAUT_LIMIT = 1.15          # validated two-attachment stability limit (Stage-0)

# Damped-settle protocol (generate_hanging_bank.py values).
SETTLE_DRAG = 0.97
SETTLE_P95_VEL = 0.10
SETTLE_CENTROID_VEL = 0.08
SETTLE_MAX_STEPS = 420
FREE_SETTLE_STEPS = 30

# Yaw-marginalized coverage: silhouette from ±Y is identical → period π.
N_YAW_SAMPLES = 12

BANK_PATH = os.path.join(PROJ_ASSETS_PATH, "Props", "Cloth", "banks",
                         "tshirt_hanging_bank.pt")

CSV_FIELDS = [
    "method", "trial", "seed", "stage",
    "bank_idx", "yaw", "mirror",
    "idx_top", "idx_move",
    "rest_x_top", "rest_y_top", "rest_x_move", "rest_y_move",
    "rest_dist", "target_ratio", "meas_ratio",
    "cov_plane", "cov_yaw_mean", "cov_yaw_max", "vis_frac",
    "extent_u", "extent_v", "drape_below_top",
    "settle_steps", "stretch_steps", "p95_speed",
    "x_sign", "attached_ok", "valid",
    "kp_a", "kp_b",
]


class StudyRig:
    """Booted study environment + the canonical trial procedures."""

    def __init__(self, args_cli, seed: int):
        torch.manual_seed(seed)
        self.seed = seed
        cfg = parse_env_cfg(args_cli.task, device=args_cli.device,
                            num_envs=args_cli.num_envs)
        cfg.episode_length_s = 1.0e6  # no manager resets — manual stepping
        self.env = gym.make(args_cli.task, cfg=cfg)
        self.env.reset()
        u = self.env.unwrapped
        self.u = u
        self.cloth = u.cloth
        self.dev = u.device
        self.n = u.num_envs
        self.p = self.cloth.num_particles
        self.dt = u.cfg.sim.dt
        self.all_ids = torch.arange(self.n, device=self.dev)
        self._all_i32 = self.all_ids.to(torch.int32)

        # Study anchor: upstream settle position, raised.
        origins = u.scene.env_origins
        self.anchor = origins.clone()
        self.anchor[:, 0] += UPSTREAM_SETTLE_POS[0]
        self.anchor[:, 1] += UPSTREAM_SETTLE_POS[1]
        self.anchor[:, 2] += CONVEYOR_SURFACE_HEIGHT_M + ANCHOR_HEIGHT_M

        # Hanging bank (anchor-centred states).
        bank = torch.load(BANK_PATH, weights_only=False)
        self.bank_pos = bank["pos"].to(self.dev)
        self.bank_vel = bank["vel"].to(self.dev)
        self.bank_anchor_idx = bank["anchor_idx"].to(self.dev)
        self.bank_size = self.bank_pos.shape[0]

        self.flat_rest = self.cloth.flat_rest_pos          # [P, 3]
        self.ref_area = flat_silhouette_area(self.flat_rest)
        # Current hold-target centre per slot (None = slot inactive).
        self.hold_pos: dict[int, torch.Tensor | None] = {0: None, 1: None}

        print(f"[rig] {self.n} envs, {self.p} particles, dt={self.dt:.4f}, "
              f"bank={self.bank_size} states, ref_area={self.ref_area:.4f} m² "
              f"anchor z={self.anchor[0, 2]:.2f}")

    # ── Stepping ──────────────────────────────────────────────────────

    def sim_steps(self, k: int, drag: float | None = None) -> None:
        """Step the sim ``k`` times, re-pinning every active hold slot."""
        u, cloth = self.u, self.cloth
        for _ in range(k):
            u.scene.write_data_to_sim()
            u.sim.step(render=False)
            u.scene.update(self.dt)
            cloth.update()
            for slot, tgt in self.hold_pos.items():
                if tgt is not None:
                    cloth.hold(tgt, slot=slot)
            if drag is not None:
                cloth._vel_flat[:] = (cloth.nodal_vel_w * drag).reshape(self.n, -1)
                cloth._set_velocities(cloth._vel_flat, self._all_i32)

    def settle(self, max_steps: int = SETTLE_MAX_STEPS,
               free_steps: int = FREE_SETTLE_STEPS) -> int:
        """Damped settle until the robust criteria fire, then a free phase."""
        steps = max_steps
        for i in range(max_steps):
            self.sim_steps(1, drag=SETTLE_DRAG)
            speeds = self.cloth.nodal_vel_w.norm(dim=-1)
            p95 = speeds.quantile(0.95, dim=1).max().item()
            cvel = self.cloth.centroid_vel_w.norm(dim=-1).max().item()
            if p95 < SETTLE_P95_VEL and cvel < SETTLE_CENTROID_VEL:
                steps = i + 1
                break
        self.sim_steps(free_steps)
        return steps

    # ── State restore + grasps ────────────────────────────────────────

    def restore_hang(self, bank_idx: torch.Tensor, yaw: torch.Tensor,
                     mirror: torch.Tensor) -> torch.Tensor:
        """Restore bank states pinned at the study anchor (slot 0).

        Own copy of ``ClothSortingEnvBase._reset_cloth_hanging_from_bank``
        with EXPLICIT augmentation parameters so the trial is fully logged /
        replayable and ``anchor_idx`` stays known.  Returns the top-grasp
        particle index per env ``[N]``.
        """
        pos = self.bank_pos[bank_idx].clone()                    # [N, P, 3]
        vel = self.bank_vel[bank_idx].clone()
        m = mirror.view(self.n, 1)
        pos[..., 0] = torch.where(m, -pos[..., 0], pos[..., 0])
        vel[..., 0] = torch.where(m, -vel[..., 0], vel[..., 0])
        cy, sy = torch.cos(yaw).view(self.n, 1), torch.sin(yaw).view(self.n, 1)
        for t in (pos, vel):
            x, y = t[..., 0].clone(), t[..., 1].clone()
            t[..., 0] = cy * x - sy * y
            t[..., 1] = sy * x + cy * y
        pos += self.anchor.unsqueeze(1)
        self.detach_all()
        self.cloth.write_nodal_state_to_sim(
            pos.reshape(self.n, -1), vel.reshape(self.n, -1), self.all_ids)
        self.cloth.update()
        self.cloth.attach(self.all_ids, self.anchor, HANG_ANCHOR_RADIUS, slot=0)
        self.hold_pos[0] = self.anchor.clone()
        return self.bank_anchor_idx[bank_idx]

    def detach_all(self) -> None:
        for slot in (0, 1):
            self.cloth.detach(self.all_ids, slot=slot)
            self.hold_pos[slot] = None

    def lowest_particle(self) -> torch.Tensor:
        """Index of the lowest hanging particle per env ``[N]``."""
        return self.cloth.nodal_pos_w[:, :, 2].argmin(dim=1)

    def attach_at_particles(self, idx: torch.Tensor, slot: int) -> torch.Tensor:
        """Attach ``slot`` at each env's particle ``idx`` current position.

        Returns the per-env success mask (False when the capture radius held
        no free particles — e.g. the target sits inside the other grasp).
        """
        centers = self.cloth.nodal_pos_w[self.all_ids, idx]      # [N, 3]
        self.cloth.attach(self.all_ids, centers, HANG_ANCHOR_RADIUS, slot=slot)
        self.hold_pos[slot] = centers.clone()
        return self.cloth.is_attached_slot(slot).clone()

    def move_hold(self, slot: int, target: torch.Tensor,
                  speed: float = RAISE_SPEED) -> int:
        """Move one slot's hold centre to ``target`` at constant speed."""
        b = self.hold_pos[slot].clone()
        step_len = speed * self.dt
        max_steps = int((target - b).norm(dim=-1).max().item() / step_len) + 30
        steps = 0
        for _ in range(max_steps):
            delta = target - b
            dist = delta.norm(dim=-1, keepdim=True)
            if (dist < 1e-4).all():
                break
            b += delta / dist.clamp(min=1e-9) * dist.clamp(max=step_len)
            self.hold_pos[slot] = b
            self.sim_steps(1)
            steps += 1
        return steps

    # ── The canonical stretch procedure ───────────────────────────────

    def stretch(self, top_slot: int, move_slot: int,
                idx_top: torch.Tensor, idx_move: torch.Tensor,
                target_ratio: torch.Tensor,
                x_sign: torch.Tensor | None = None,
                settle: bool = True) -> dict:
        """Horizontal presentation stretch (the study's canonical procedure).

        Moves ``move_slot`` on a straight line to the SAME HEIGHT (and Y) as
        ``top_slot``, offset ``target_ratio × rest_dist`` along world X — the
        inspection camera's horizontal image axis.  The taut chord between
        the grasps ends up horizontal in the camera plane; gravity extends
        the garment below it.  Ratio capped at ``TAUT_LIMIT``.

        Args:
            x_sign: optional ``[N]`` ±1 pull direction.  Default: the side of
                the moving point's current horizontal offset from the holder
                (never drags the cloth across itself).  The regrasp heuristic
                passes the NEGATED previous sign to stretch the other way.

        Returns the sign used (``"x_sign"``) for direction chaining.
        """
        a = self.hold_pos[top_slot]
        b = self.hold_pos[move_slot].clone()
        rest = rest_distance(self.flat_rest, idx_top, idx_move)  # [N]
        ratio = target_ratio.clamp(max=TAUT_LIMIT)
        if x_sign is None:
            x_sign = torch.where(b[:, 0] >= a[:, 0], 1.0, -1.0)
        length = ratio * rest                                    # [N]
        target = a.clone()
        target[:, 0] += x_sign * length
        step_len = STRETCH_SPEED * self.dt
        max_steps = int((target - b).norm(dim=-1).max().item() / step_len) + 60
        steps = 0
        for _ in range(max_steps):
            delta = target - b
            dist = delta.norm(dim=-1, keepdim=True)
            if (dist < 1e-4).all():
                break
            b += delta / dist.clamp(min=1e-9) * dist.clamp(max=step_len)
            self.hold_pos[move_slot] = b
            self.sim_steps(1)
            steps += 1
        settle_steps = self.settle() if settle else 0
        return {"stretch_steps": steps, "settle_steps": settle_steps,
                "x_sign": x_sign, "final_len": length}

    # ── Measurement ───────────────────────────────────────────────────

    def measure(self, idx_top: torch.Tensor, idx_move: torch.Tensor,
                top_z: torch.Tensor) -> dict:
        """All per-trial metrics from the current settled state (tensors [N])."""
        pos = self.cloth.nodal_pos_w
        rest = rest_distance(self.flat_rest, idx_top, idx_move)
        cov_plane = silhouette_coverage(pos, self.ref_area)
        covs = []
        for k in range(N_YAW_SAMPLES):
            ang = torch.pi * k / N_YAW_SAMPLES
            c, s = torch.cos(torch.tensor(ang)), torch.sin(torch.tensor(ang))
            q = pos.clone()
            q[..., 0] = c * pos[..., 0] - s * pos[..., 1]
            q[..., 1] = s * pos[..., 0] + c * pos[..., 1]
            covs.append(silhouette_coverage(q, self.ref_area))
        covs = torch.stack(covs)                                 # [K, N]
        ext = plane_extent(pos)
        pa = pos[self.all_ids, idx_top]
        pb = pos[self.all_ids, idx_move]
        return {
            "rest_dist": rest,
            "meas_ratio": stretch_ratio(pa, pb, rest),
            "cov_plane": cov_plane,
            "cov_yaw_mean": covs.mean(dim=0),
            "cov_yaw_max": covs.max(dim=0).values,
            "vis_frac": two_sided_visible_fraction(pos),
            "extent_u": ext[:, 0], "extent_v": ext[:, 1],
            "drape_below_top": top_z - pos[:, :, 2].min(dim=1).values,
            "p95_speed": self.cloth.nodal_vel_w.norm(dim=-1).quantile(0.95, dim=1),
        }

    def rest_xy(self, idx: torch.Tensor) -> torch.Tensor:
        """Flat-rest-shape (x, y) coordinates of particle indices ``[N, 2]``."""
        return self.flat_rest[idx][:, :2]

    def close(self) -> None:
        self.env.close()


# ── CSV / state-dump helpers ───────────────────────────────────────────────

class TrialLog:
    """Accumulates trial rows; flushes the CSV after every round (crash-safe)."""

    def __init__(self, csv_path: str, states_dir: str | None = None):
        self.csv_path = csv_path
        self.states_dir = states_dir
        self.rows: list[dict] = []
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if states_dir:
            os.makedirs(states_dir, exist_ok=True)
        self.t0 = time.perf_counter()

    def add_round(self, method: str, seed: int, stage: int, trial0: int,
                  scalars: dict, meas: dict, rig: StudyRig,
                  save_states: bool = True) -> None:
        """Append one batched round (env i → trial ``trial0 + i``).

        ``scalars``: per-env tensors/values keyed by CSV field (bank_idx, yaw,
        mirror, idx_top, idx_move, target_ratio, settle/stretch steps, flags).
        ``meas``: output of ``StudyRig.measure``.
        """
        n = rig.n
        rest_top = rig.rest_xy(scalars["idx_top"])
        rest_move = rig.rest_xy(scalars["idx_move"])
        merged: dict = {**scalars, **meas,
                        "rest_x_top": rest_top[:, 0], "rest_y_top": rest_top[:, 1],
                        "rest_x_move": rest_move[:, 0], "rest_y_move": rest_move[:, 1]}
        for i in range(n):
            row = {"method": method, "trial": trial0 + i, "seed": seed,
                   "stage": stage}
            for f in CSV_FIELDS:
                if f in row:
                    continue
                v = merged.get(f, "")
                if isinstance(v, torch.Tensor):
                    v = v[i].item() if v.dim() else v.item()
                if isinstance(v, bool):
                    v = int(v)
                if isinstance(v, float):
                    v = f"{v:.6g}"
                row[f] = v
            self.rows.append(row)
        if save_states and self.states_dir:
            pos = (rig.cloth.nodal_pos_w - rig.anchor.unsqueeze(1)).half().cpu()
            for i in range(n):
                torch.save(
                    {"pos": pos[i].clone(),
                     "cov_plane": float(meas["cov_plane"][i]),
                     "method": method, "stage": stage, "trial": trial0 + i},
                    os.path.join(self.states_dir,
                                 f"{method}_s{stage}_t{trial0 + i:05d}.pt"))
        self.flush()

    def flush(self) -> None:
        with open(self.csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(self.rows)

    def status(self, label: str) -> None:
        dt = time.perf_counter() - self.t0
        print(f"[{label}] {len(self.rows)} rows, {dt:.0f}s elapsed "
              f"({len(self.rows) / max(dt, 1e-9) * 60:.1f} trials/min)",
              flush=True)


def dump_flat_rest(rig: StudyRig, out_path: str) -> None:
    """Save the flat rest shape + ref area (region assignment happens offline)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"flat_rest_pos": rig.flat_rest.cpu(),
                "ref_area": rig.ref_area,
                "bank_anchor_idx": rig.bank_anchor_idx.cpu()}, out_path)
    print(f"[rig] flat rest shape saved to {out_path}")
