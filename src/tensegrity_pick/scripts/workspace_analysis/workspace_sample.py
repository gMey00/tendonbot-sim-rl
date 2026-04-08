"""Data generation for 5-DOF tensegrity robot workspace analysis.

Performs Monte Carlo forward-kinematics sampling by:

1. Uniformly sampling random joint configurations within the joint limits
   (Monte Carlo approach, analogous to MathWorks ``generateRobotWorkspace`` [4]).
2. Computing forward kinematics (FK) via the IsaacSim 5.1.0 physics engine.
3. Computing the Yoshikawa manipulability index and inverse condition number
   from the geometric Jacobian [1, 2].
4. Saving raw data (positions, manipulability, condition number) as NumPy
   files for downstream visualisation.

During analysis, collision on scene objects (conveyors, drums) is disabled so
the robot arm can sweep freely through the entire joint space without
interference (the robot itself retains self-collision settings).

A translucent box visualises the desired workspace derived from the Klein 2023
master-thesis task geometry [3].

Quality Measures
----------------
**Yoshikawa Manipulability Index** [1]:
    For a robot with *n* DOFs and a 3-DOF positional task space the linear
    Jacobian is J in R^(3xn).  The Yoshikawa index is

        w(q) = sqrt( det( J(q) @ J(q)^T ) )

    Higher values indicate the end effector can be moved equally well in all
    Cartesian directions (far from singularities).  The index is zero at
    singular configurations.

**Inverse Condition Number** (isotropy index) [2]:
    kappa_inv = sigma_min / sigma_max  of  J

    kappa_inv in [0, 1] where 1 means perfectly isotropic motion capability.
    Singular configurations yield kappa_inv = 0.

**Reachability Density**:
    The number of random FK samples landing in each voxel, proportional to
    the joint-space volume that maps to that Cartesian region.  Higher
    density = more configurations can reach that area = better reachability.

References
----------
[1] Yoshikawa, T. (1985). "Manipulability of Robotic Mechanisms."
    The International Journal of Robotics Research, 4(2), 3-9.
[2] Salisbury, J. K., & Craig, J. J. (1982). "Articulated Hands: Force
    Control and Kinematic Issues." IJRR, 1(1), 4-17.
[3] Klein, R. (2023). Master's Thesis -- Tensegrity Robot Pick-and-Place.
[4] MathWorks (2024). "Workspace Analysis for Manipulators."
    https://de.mathworks.com/help/robotics/ug/workspace-analysis-for-manipulators.html
[5] Togai, M. (1986). "An Application of the Singular Value Decomposition
    to Manipulability and Sensitivity of Industrial Robots."

Usage
-----
    cd src/tensegrity_pick
    conda run -n env_isaaclab python3 scripts/workspace_analysis/workspace_sample.py --headless
    conda run -n env_isaaclab python3 scripts/workspace_analysis/workspace_sample.py  # with GUI
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

from workspace_config import (
    COLLISION_ADJACENCY_SKIP,
    COLLISION_MIN_DISTANCE,
    DEFAULT_NUM_ENVS,
    DEFAULT_NUM_SAMPLES,
    DESIRED_WS_MAX,
    DESIRED_WS_MIN,
    GRIPPER_TIP_OFFSET,
    ROBOTS,
)

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Monte Carlo FK sampling for robot workspace analysis",
)
parser.add_argument(
    "--robot", choices=list(ROBOTS), default="tensegrity",
    help="Robot to analyse (default: tensegrity)",
)
parser.add_argument(
    "--num_samples", type=int, default=DEFAULT_NUM_SAMPLES,
    help=f"Total FK samples (default: {DEFAULT_NUM_SAMPLES:,})",
)
parser.add_argument(
    "--num_envs", type=int, default=DEFAULT_NUM_ENVS,
    help=f"Parallel environments for batched sampling (default: {DEFAULT_NUM_ENVS:,})",
)
parser.add_argument(
    "--mount_height", type=float, default=None,
    help="Override robot mount height in metres (default: scene default)",
)
parser.add_argument(
    "--mount_direction", choices=["up", "down"], default=None,
    help="Mount orientation: 'down' = ceiling-mounted, 'up' = floor-mounted (default: per-robot)",
)
parser.add_argument(
    "--output_dir", type=str, default=None,
    help="Directory for output files (default: outputs/workspace_analysis[_ur10e])",
)
parser.add_argument(
    "--append", action="store_true",
    help="Append new samples to existing data in output_dir instead of overwriting",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────
import numpy as np
import torch

from pxr import Usd  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

from isaaclab.assets.articulation import ArticulationCfg  # noqa: E402

from tensegrity_pick.tasks.manager_based.shared.proj_base_scene_cfg import (  # noqa: E402
    CONVEYOR_SURFACE_HEIGHT_M,
    CONVEYOR_WIDTH_M,
    DRUM_CENTER_TO_CONVEYOR_EDGE_M,
    DRUM_HEIGHT_M,
    ProjBaseSceneCfg,
)


from workspace_analysis_helper import compute_desired_workspace_coverage  # noqa: E402

GRIPPER_TIP_LOCAL_OFFSET = torch.tensor(GRIPPER_TIP_OFFSET)


# ── Logging ───────────────────────────────────────────────────────────────

def _log(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


# ── Collision control ─────────────────────────────────────────────────────

def disable_scene_object_collisions(stage: Usd.Stage) -> int:
    """Disable collision on every non-robot, non-ground prim in *stage*.

    Returns the number of collision attributes toggled off.
    """
    toggled = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if any(tag in path for tag in ("/Robot", "GroundPlane", "Light", "DesiredWorkspace")):
            continue
        attr = prim.GetAttribute("physics:collisionEnabled")
        if attr and attr.HasValue():
            attr.Set(False)
            toggled += 1
    return toggled


# ── Desired-workspace box (translucent visual in USD) ─────────────────────

def spawn_desired_workspace_box(
    stage: Usd.Stage,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
) -> None:
    """Spawn a translucent blue cuboid representing the desired workspace [Ref. 3].

    Uses Isaac Lab's CuboidCfg with GlassMdlCfg (OmniGlass material) so that
    translucency renders correctly under the RTX renderer used by IsaacSim.
    """
    prim_path = "/World/DesiredWorkspaceBox"
    cfg = sim_utils.CuboidCfg(
        size=size,
        visual_material=sim_utils.GlassMdlCfg(
            glass_color=(0.2, 0.5, 1.0),
            frosting_roughness=0.3,
            thin_walled=True,
        ),
    )
    cfg.func(prim_path, cfg, translation=center)


# ── Jacobian / manipulability computation ─────────────────────────────────

def extract_ee_linear_jacobian(
    jacobian_raw: torch.Tensor,
    ee_body_index: int,
    joint_ids: list[int],
    num_robot_bodies: int,
) -> torch.Tensor | None:
    """Extract the 3 x len(joint_ids) linear Jacobian for the EE body.

    The PhysX Jacobian may be returned in two layouts:

    * **4-D** ``(N, num_links, 6, num_dofs)``
    * **3-D** ``(N, num_links * 6, num_dofs)``

    Returns ``None`` if the Jacobian shape cannot be resolved.
    """
    ndim = jacobian_raw.ndim

    if ndim == 4:
        num_link_slots = jacobian_raw.shape[1]
        body_offset = _resolve_body_offset(ee_body_index, num_link_slots, num_robot_bodies)
        if body_offset is None:
            return None
        J_lin = jacobian_raw[:, body_offset, 3:6, :]
        return J_lin[:, :, joint_ids]

    if ndim == 3:
        total_rows = jacobian_raw.shape[1]
        if total_rows % 6 != 0:
            _log(f"  WARNING: Jacobian row count {total_rows} not divisible by 6")
            return None
        num_link_slots = total_rows // 6
        body_offset = _resolve_body_offset(ee_body_index, num_link_slots, num_robot_bodies)
        if body_offset is None:
            return None
        linear_row_start = body_offset * 6 + 3
        linear_row_end = linear_row_start + 3
        return jacobian_raw[:, linear_row_start:linear_row_end, :][:, :, joint_ids]

    _log(f"  WARNING: Unexpected Jacobian ndim={ndim}, shape={jacobian_raw.shape}")
    return None


def _resolve_body_offset(
    ee_body_index: int,
    num_link_slots: int,
    num_robot_bodies: int,
) -> int | None:
    """Determine which slot in the Jacobian corresponds to the EE body."""
    if num_link_slots == num_robot_bodies:
        body_offset = ee_body_index
    elif num_link_slots == num_robot_bodies - 1:
        body_offset = ee_body_index - 1
    else:
        _log(
            f"  WARNING: Jacobian has {num_link_slots} link slots but robot has "
            f"{num_robot_bodies} bodies -- cannot determine EE offset"
        )
        return None
    if body_offset < 0 or body_offset >= num_link_slots:
        _log(f"  WARNING: EE body offset {body_offset} out of range [0, {num_link_slots})")
        return None
    return body_offset


def compute_yoshikawa_index(jacobian_linear: torch.Tensor) -> torch.Tensor:
    """Yoshikawa manipulability index: w = sqrt( det(J @ J^T) ).

    Ref. [1]: Yoshikawa, T. (1985). "Manipulability of Robotic Mechanisms."
    """
    JJT = torch.bmm(jacobian_linear, jacobian_linear.transpose(1, 2))
    return torch.sqrt(torch.clamp(torch.det(JJT), min=0.0))


def compute_inverse_condition_number(jacobian_linear: torch.Tensor) -> torch.Tensor:
    """Inverse condition number (isotropy index): sigma_min / sigma_max.

    Ref. [2]: Salisbury & Craig (1982).  Ref. [5]: Togai (1986).
    """
    singular_values = torch.linalg.svdvals(jacobian_linear)
    sigma_min = singular_values[:, -1]
    sigma_max = singular_values[:, 0]
    return torch.where(
        sigma_max > 1e-10,
        sigma_min / sigma_max,
        torch.zeros_like(sigma_max),
    )


# ── Geometric self-collision check ────────────────────────────────────────

def compute_self_collision_mask(
    body_positions: torch.Tensor,
    adjacency_skip: int = 1,
    min_distance: float = 0.05,
) -> torch.Tensor:
    """Geometric self-collision check via pairwise body-body distances.

    Unlike PhysX ``enabled_self_collisions`` (which feeds self-contact
    constraints back into the solver and can destabilise articulations),
    this check runs **post-FK** on the body positions and never modifies
    the physics.

    Parameters
    ----------
    body_positions : ``(num_envs, num_bodies, 3)``
        World-frame positions of all articulation bodies.
    adjacency_skip : int
        Body pairs with ``|i - j| <= adjacency_skip`` are considered
        kinematically adjacent and always allowed to be close.  Default 1
        means only directly connected links are skipped.
    min_distance : float
        Minimum distance (metres) between non-adjacent body origins.
        Environments where any pair is closer are flagged as colliding.

    Returns
    -------
    collision_free : ``(num_envs,)`` bool tensor — ``True`` = no collision.
    """
    num_envs, num_bodies, _ = body_positions.shape
    collision_free = torch.ones(num_envs, dtype=torch.bool, device=body_positions.device)

    for i in range(num_bodies):
        for j in range(i + 1, num_bodies):
            if abs(i - j) <= adjacency_skip:
                continue
            dist = torch.norm(body_positions[:, i] - body_positions[:, j], dim=-1)
            collision_free &= dist > min_distance

    return collision_free


# ── Monte Carlo FK sampling ──────────────────────────────────────────────

def sample_workspace(
    robot: object,
    ee_body_index: int,
    joint_ids: list[int],
    env_origins: torch.Tensor,
    num_samples: int,
    num_envs: int,
    sim: SimulationContext,
    scene: InteractiveScene,
    device: str,
    mount_height: float = 2.30,
    filter_self_collisions: bool = True,
    collision_min_distance: float = 0.05,
    collision_adjacency_skip: int = 1,
    linked_joint_indices: dict[int, int] | None = None,
    elbow_filter_body_index: int | None = None,
    elbow_filter_max_rad: float = 0.0,
    antiparallelogram_cols: tuple[int, int, int] | None = None,
    num_settle_steps: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect EE positions and manipulability via randomised FK.

    Parameters
    ----------
    mount_height : float
        Height of the robot mount in env-local Z (metres).  The arm always
        hangs below the mount, so any sample with Z >= mount_height is a
        physics-solver artefact and is rejected.
    filter_self_collisions : bool
        When True, post-FK geometric body-distance checks discard
        samples where non-adjacent links are closer than
        *collision_min_distance*.  This avoids PhysX
        ``enabled_self_collisions`` which can destabilise the solver.
    collision_min_distance : float
        Minimum body-origin distance (metres) for the geometric check.
    collision_adjacency_skip : int
        Body pairs with ``|i - j| <= adjacency_skip`` are kinematically
        adjacent and excluded from the distance check.
    linked_joint_indices : dict[int, int] | None
        Mapping {target_col: source_col} within the sampled joint
        positions tensor.  After random sampling, target columns are
        copied from source columns (e.g. rod_right = rod_left).
    elbow_filter_body_index : int | None
        When set, extract the X-axis rotation of this body after each
        physics step and reject samples where |angle| > *elbow_filter_max_rad*.
    elbow_filter_max_rad : float
        Maximum absolute elbow angle (rad) for the post-physics filter.
    antiparallelogram_cols : tuple[int, int, int] | None
        Column indices (rod_left, rod_right, coupler_left) in the
        sampled joint positions tensor.  When set, rod_right and
        coupler_left are computed from rod_left via the closure condition
        instead of random sampling.
    num_settle_steps : int
        Number of physics steps per batch.  For robots with closed-loop
        kinematic constraints (e.g. antiparallelogram linkage), extra
        steps let the PhysX constraint solver converge before reading
        body transforms.  Default 1 is sufficient for open-chain robots.

    Returns
    -------
    positions  : ``(N, 3)`` env-local EE positions.
    yoshikawa  : ``(N,)`` Yoshikawa manipulability values.
    condition  : ``(N,)`` inverse condition numbers.
    """
    joint_limits = robot.data.soft_joint_pos_limits[0, joint_ids]
    lower = joint_limits[:, 0]
    upper = joint_limits[:, 1]

    # Clamp infinite limits (continuous-rotation joints) to ±π so that
    # uniform random sampling stays finite.
    lower = torch.clamp(lower, min=-math.pi)
    upper = torch.clamp(upper, max=math.pi)

    dt = sim.get_physics_dt()
    num_batches = math.ceil(num_samples / num_envs)
    num_robot_bodies = len(robot.body_names)

    all_positions: list[np.ndarray] = []
    all_yoshikawa: list[np.ndarray] = []
    all_condition: list[np.ndarray] = []
    jacobian_available = True
    total_collision_filtered = 0

    zero_joint_positions = torch.zeros(num_envs, len(joint_ids), device=device)

    for batch_index in range(num_batches):
        random_joint_positions = lower + (upper - lower) * torch.rand(
            num_envs, len(joint_ids), device=device,
        )

        # Enforce kinematic constraints
        if linked_joint_indices:
            for tgt_col, src_col in linked_joint_indices.items():
                random_joint_positions[:, tgt_col] = random_joint_positions[:, src_col]

        # Antiparallelogram closure: compute rod_right and coupler_left
        # from rod_left via the McCarthy & Soh (2010) closure equation.
        #   θ = θ₀ + δ_left
        #   φ = θ + 2·arctan(−k_e·cosθ / (l_e − k_e·sinθ))
        #   rod_right = φ + θ₀
        #   coupler_left = rod_right  (antiparallelogram symmetry)
        if antiparallelogram_cols is not None:
            _rl, _rr, _cl = antiparallelogram_cols
            _le = 0.150   # rod length [m]
            _ke = 0.060   # joint spacing [m]
            _t0 = math.asin(_ke / _le)  # equilibrium angle
            theta = _t0 + random_joint_positions[:, _rl]
            t = -_ke * torch.cos(theta) / (_le - _ke * torch.sin(theta))
            phi = theta + 2.0 * torch.atan(t)
            rod_right_val = phi + _t0
            random_joint_positions[:, _rr] = rod_right_val
            random_joint_positions[:, _cl] = rod_right_val

        # Reset to initial state before each batch so the PhysX loop-
        # closure constraint (coupler_right_joint, excludeFromArticulation)
        # starts from a known-good state.  Without this reset the
        # constraint solver warm-start data accumulates errors across
        # batches and eventually the constraint breaks (rod hangs down).
        if num_settle_steps > 1:
            robot.write_joint_position_to_sim(zero_joint_positions, joint_ids=joint_ids)
            robot.write_joint_velocity_to_sim(zero_joint_positions, joint_ids=joint_ids)
            sim.step()

        robot.write_joint_position_to_sim(random_joint_positions, joint_ids=joint_ids)
        robot.write_joint_velocity_to_sim(zero_joint_positions, joint_ids=joint_ids)

        for _settle in range(num_settle_steps):
            sim.step()
            if _settle < num_settle_steps - 1:
                # Re-zero velocities between settle steps so constraint-solver
                # impulses don't accumulate into dynamic drift.
                robot.write_joint_velocity_to_sim(zero_joint_positions, joint_ids=joint_ids)
        scene.update(dt)

        # ── Self-collision mask (geometric body-distance check) ─────
        # NOTE: disabled — the geometric body-distance check does not work
        # reliably for the reworked tensegrity model; PhysX articulation
        # self-collision is handled via the USD articulation settings instead.
        # if filter_self_collisions:
        #     arm_body_pos = robot.data.body_pos_w[:, :ee_body_index + 1, :]
        #     collision_free = compute_self_collision_mask(
        #         arm_body_pos,
        #         adjacency_skip=collision_adjacency_skip,
        #         min_distance=collision_min_distance,
        #     )
        #     collision_free_np = collision_free.cpu().numpy()
        #     total_collision_filtered += int((~collision_free).sum().item())
        # else:
        #     collision_free_np = np.ones(num_envs, dtype=bool)
        collision_free_np = np.ones(num_envs, dtype=bool)

        # Validate loop-closure constraint integrity.  If the PhysX
        # constraint broke (e.g. coupler_right disconnected), the
        # kinematic chain is invalid and body positions are garbage.
        # Check that no adjacent body pair is farther apart than the
        # maximum link length (0.5 m covers all links generously;
        # the longest single segment is the forearm at ~0.26 m).
        if num_settle_steps > 1:
            body_pos_local = robot.data.body_pos_w[:, :, :3] - env_origins.unsqueeze(1)
            for bi in range(1, min(ee_body_index + 1, body_pos_local.shape[1])):
                pair_dist = torch.norm(
                    body_pos_local[:, bi] - body_pos_local[:, bi - 1], dim=-1,
                )
                collision_free_np &= (pair_dist < 0.5).cpu().numpy()

        ee_world = robot.data.body_pos_w[:, ee_body_index, :3]
        ee_quat = robot.data.body_quat_w[:, ee_body_index, :]
        tip_offset_world = quat_apply(
            ee_quat, GRIPPER_TIP_LOCAL_OFFSET.to(device).expand(num_envs, -1),
        )
        tip_world = ee_world + tip_offset_world
        batch_positions = (tip_world - env_origins).cpu().numpy()

        # Reject physics-divergent samples using physically motivated bounds.
        #
        # Three independent constraints, each derived from the kinematic chain:
        #
        #  1. dist_from_mount < 1.95 m
        #     Triangle-inequality bound: arm_root_travel(≤0.707) + arm_length(1.197)
        #     = 1.904 m max reach from mount; 1.95 m provides ~2.5% margin.
        #
        #  2. Z < mount_height − 0.15
        #     Arm always hangs BELOW the ceiling mount.  Even at its shortest
        #     configuration (elbow at 0°) the EE is ~0.2 m below the mount.
        #
        #  3. |Y| < base_y_max + arm_length = 0.5 + 1.197 = 1.70 m
        #     The base Y axis is the only source of lateral motion; the arm can
        #     add at most arm_length further in Y.  This catches artefacts that
        #     sit close to mount height in Z (making dist_from_mount plausible)
        #     but at impossible Y displacements (e.g. Y=2 m with arm only 1.2 m).
        #
        # mount_pos in env-local frame = (0.15, 0.0, mount_height)
        mount_local = np.array([0.15, 0.0, mount_height], dtype=np.float32)
        dist_from_mount = np.linalg.norm(batch_positions - mount_local, axis=1)
        physics_valid = (
            (dist_from_mount < 1.95)
            & (batch_positions[:, 2] < mount_height - 0.15)
            & (np.abs(batch_positions[:, 1]) < 1.7)
        )
        collision_free_np &= physics_valid

        # ── Elbow-angle post-filter (physical model) ──────────────
        if elbow_filter_body_index is not None:
            quat = robot.data.body_quat_w[:, elbow_filter_body_index, :]  # (N,4) wxyz
            # Extract rotation around the local X axis (elbow flexion)
            elbow_angle = 2.0 * torch.atan2(quat[:, 1], quat[:, 0])  # 2*atan2(qx, qw)
            elbow_ok = (elbow_angle.abs() <= elbow_filter_max_rad).cpu().numpy()
            collision_free_np &= elbow_ok

        batch_yoshikawa: np.ndarray
        batch_condition: np.ndarray

        if jacobian_available:
            try:
                jacobian_raw = robot.root_physx_view.get_jacobians()
                if batch_index == 0:
                    _log(f"  Jacobian shape: {jacobian_raw.shape}  (ndim={jacobian_raw.ndim})")
                jacobian_linear = extract_ee_linear_jacobian(
                    jacobian_raw, ee_body_index, joint_ids, num_robot_bodies,
                )
                if jacobian_linear is not None:
                    batch_yoshikawa = compute_yoshikawa_index(jacobian_linear).cpu().numpy()
                    batch_condition = compute_inverse_condition_number(jacobian_linear).cpu().numpy()
                else:
                    jacobian_available = False
            except Exception as exc:
                if batch_index == 0:
                    _log(f"  WARNING: Jacobian unavailable ({exc}); skipping manipulability")
                jacobian_available = False

        if not jacobian_available:
            batch_yoshikawa = np.full(num_envs, np.nan, dtype=np.float32)
            batch_condition = np.full(num_envs, np.nan, dtype=np.float32)

        # Apply self-collision filter
        all_positions.append(batch_positions[collision_free_np])
        all_yoshikawa.append(batch_yoshikawa[collision_free_np])
        all_condition.append(batch_condition[collision_free_np])

        if (batch_index + 1) % 25 == 0 or batch_index == num_batches - 1:
            collected = sum(len(a) for a in all_positions)
            _log(f"  Batch {batch_index + 1}/{num_batches}  (collected: {collected:,})")

    positions = np.concatenate(all_positions, axis=0)[:num_samples]
    yoshikawa = np.concatenate(all_yoshikawa, axis=0)[:num_samples]
    condition = np.concatenate(all_condition, axis=0)[:num_samples]

    if total_collision_filtered > 0:
        total_tested = num_batches * num_envs
        _log(
            f"  Self-collision filter: removed {total_collision_filtered:,} / "
            f"{total_tested:,} samples ({total_collision_filtered / total_tested * 100:.1f}%)"
        )

    if not jacobian_available:
        _log("  NOTE: Manipulability values are NaN (Jacobian was unavailable)")

    return positions, yoshikawa, condition


# ── Statistics ────────────────────────────────────────────────────────────

def _print_metric_block(
    label: str,
    values_global: np.ndarray,
    values_inside: np.ndarray,
) -> None:
    """Print a statistics block for a manipulability metric."""
    valid_global = values_global[np.isfinite(values_global)]
    valid_inside = values_inside[np.isfinite(values_inside)]

    if len(valid_global) == 0:
        _log(f"\n  {label}: No valid samples")
        return

    _log(f"\n  {label} — Global (all reachable):")
    _log(f"    Mean  : {valid_global.mean():.6f}")
    _log(f"    Median: {np.median(valid_global):.6f}")
    _log(f"    Std   : {valid_global.std():.6f}")
    _log(f"    Range : [{valid_global.min():.6f}, {valid_global.max():.6f}]")

    if len(valid_inside) == 0:
        _log(f"  {label} — Inside desired WS: N/A (no samples)")
        return

    _log(f"  {label} — Inside desired WS ({len(valid_inside):,} samples):")
    _log(f"    Mean  : {valid_inside.mean():.6f}")
    _log(f"    Median: {np.median(valid_inside):.6f}")
    _log(f"    Std   : {valid_inside.std():.6f}")
    _log(f"    P5    : {np.percentile(valid_inside, 5):.6f}")
    _log(f"    P25   : {np.percentile(valid_inside, 25):.6f}")
    _log(f"    P75   : {np.percentile(valid_inside, 75):.6f}")
    _log(f"    P95   : {np.percentile(valid_inside, 95):.6f}")
    _log(f"    Range : [{valid_inside.min():.6f}, {valid_inside.max():.6f}]")


def print_statistics(
    positions: np.ndarray,
    yoshikawa: np.ndarray,
    condition: np.ndarray,
    mount_height: float,
) -> None:
    """Print workspace and manipulability statistics to stderr."""
    _log(f"\n{'=' * 70}")
    _log("WORKSPACE ANALYSIS RESULTS")
    _log(f"{'=' * 70}")
    _log(f"  Total FK samples : {len(positions):,}")
    if len(positions) == 0:
        _log("  WARNING: No samples survived filtering. Cannot compute statistics.")
        return
    _log(f"  X range          : [{positions[:, 0].min():.4f}, {positions[:, 0].max():.4f}] m")
    _log(f"  Y range          : [{positions[:, 1].min():.4f}, {positions[:, 1].max():.4f}] m")
    _log(f"  Z range          : [{positions[:, 2].min():.4f}, {positions[:, 2].max():.4f}] m")

    # ── Sample-level coverage ─────────────────────────────────────────
    inside = np.all(
        (positions >= DESIRED_WS_MIN) & (positions <= DESIRED_WS_MAX), axis=1,
    )
    inside_fraction = inside.sum() / len(positions) * 100
    _log(f"\n  Samples inside desired WS : {inside.sum():,} / {len(positions):,} ({inside_fraction:.1f}%)")

    # ── Volumetric coverage of desired workspace ──────────────────────
    coverage, reached, total = compute_desired_workspace_coverage(positions)
    _log(f"\n  --- DESIRED WORKSPACE COVERAGE (voxel-based, 2 cm) ---")
    _log(f"    Voxels reached  : {reached:,} / {total:,}")
    _log(f"    Volume coverage : {coverage * 100:.1f}%")

    desired_volume = float(np.prod(DESIRED_WS_MAX - DESIRED_WS_MIN))
    _log(f"    Desired WS vol  : {desired_volume:.4f} m³")

    # ── Manipulability inside desired workspace ───────────────────────
    inside_yoshikawa = yoshikawa[inside]
    inside_condition = condition[inside]

    _print_metric_block(
        "Yoshikawa Manipulability Index [Ref. 1]",
        yoshikawa, inside_yoshikawa,
    )
    _print_metric_block(
        "Inverse Condition Number (Isotropy) [Ref. 2]",
        condition, inside_condition,
    )

    _log(f"\n  Desired workspace box (env-local) [Ref. 3]:")
    _log(f"    X : [{DESIRED_WS_MIN[0]:.2f}, {DESIRED_WS_MAX[0]:.2f}] m")
    _log(f"    Y : [{DESIRED_WS_MIN[1]:.2f}, {DESIRED_WS_MAX[1]:.2f}] m")
    _log(f"    Z : [{DESIRED_WS_MIN[2]:.2f}, {DESIRED_WS_MAX[2]:.2f}] m")

    _log(f"\n  Scene reference points (env-local):")
    _log(f"    Robot mount     : (0.15, 0.00, {mount_height:.2f})")
    _log(f"    Belt surface    : z = {CONVEYOR_SURFACE_HEIGHT_M:.2f}")
    drum_y = CONVEYOR_WIDTH_M * 0.5 + DRUM_CENTER_TO_CONVEYOR_EDGE_M
    _log(f"    Drum centre     : (0.15, {drum_y:.2f}, 0.00)")
    _log(f"    Drum rim height : z = {DRUM_HEIGHT_M:.2f}")
    _log(f"{'=' * 70}\n")


def save_statistics_json(
    positions: np.ndarray,
    yoshikawa: np.ndarray,
    condition: np.ndarray,
    mount_height: float,
    mount_direction: str,
    robot_name: str,
    output_dir: Path,
) -> None:
    """Save machine-readable statistics as JSON for documentation tooling."""
    inside = np.all(
        (positions >= DESIRED_WS_MIN) & (positions <= DESIRED_WS_MAX), axis=1,
    )
    coverage_frac, reached_voxels, total_voxels = compute_desired_workspace_coverage(positions)

    def _metric_dict(values: np.ndarray) -> dict:
        valid = values[np.isfinite(values)]
        if len(valid) == 0:
            return {"n": 0}
        return {
            "n": int(len(valid)),
            "mean": round(float(valid.mean()), 6),
            "median": round(float(np.median(valid)), 6),
            "std": round(float(valid.std()), 6),
            "min": round(float(valid.min()), 6),
            "max": round(float(valid.max()), 6),
            "p5": round(float(np.percentile(valid, 5)), 6),
            "p25": round(float(np.percentile(valid, 25)), 6),
            "p75": round(float(np.percentile(valid, 75)), 6),
            "p95": round(float(np.percentile(valid, 95)), 6),
        }

    stats = {
        "robot": robot_name,
        "mount_height_m": round(mount_height, 3),
        "mount_direction": mount_direction,
        "total_samples": len(positions),
        "samples_inside_desired_ws": int(inside.sum()),
        "desired_ws_coverage_fraction": round(coverage_frac, 4),
        "desired_ws_voxels_reached": reached_voxels,
        "desired_ws_voxels_total": total_voxels,
        "yoshikawa_global": _metric_dict(yoshikawa),
        "yoshikawa_inside_ws": _metric_dict(yoshikawa[inside]),
        "condition_global": _metric_dict(condition),
        "condition_inside_ws": _metric_dict(condition[inside]),
    }

    output_path = output_dir / "statistics.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    _log(f"  Statistics saved to {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    num_envs = args_cli.num_envs
    num_samples = args_cli.num_samples
    robot_choice = args_cli.robot
    device = "cuda:0"

    robot_cfg = ROBOTS[robot_choice]
    output_dir = Path(args_cli.output_dir or robot_cfg.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    EE_BODY_NAME = robot_cfg.ee_body_name
    CONTROLLED_JOINT_NAMES = list(robot_cfg.controlled_joints)

    sim = SimulationContext(SimulationCfg(dt=1.0 / 60.0, device=device))

    mount_height = args_cli.mount_height or robot_cfg.mount_height
    mount_direction = args_cli.mount_direction or robot_cfg.default_mount_direction
    mount_rotation = robot_cfg.mount_rotations[mount_direction]

    scene_cfg = ProjBaseSceneCfg(num_envs=num_envs, env_spacing=5.0)

    init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.15, 0.0, mount_height),
        rot=mount_rotation,
    )
    if robot_choice == "tensegrity":
        from tensegrity_pick.robots import TENS_5DOF_GRIPPER_CFG

        scene_cfg.robot = TENS_5DOF_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=init_state,
        )
    elif robot_choice == "tensegrity_physical":
        from tensegrity_pick.robots.tendon_robot_cfg import TENS_5DOF_GRIPPER_PHYSICAL_TENDON_CFG

        # FK sampling teleports joint positions directly via write_joint_position_to_sim,
        # bypassing the tendon actuator entirely.  The physical USD has the correct linkage
        # geometry, so the workspace captured here reflects the real antiparallelogram
        # elbow kinematics rather than the single-DOF elbow_approx approximation.
        scene_cfg.robot = TENS_5DOF_GRIPPER_PHYSICAL_TENDON_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=init_state,
        )
    elif robot_choice == "ur10e":
        from tensegrity_pick.robots import UR10E_GRIPPER_CFG

        scene_cfg.robot = UR10E_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=init_state,
        )
    elif robot_choice == "kinova":
        from tensegrity_pick.robots import KINOVA_GEN3_GRIPPER_CFG

        scene_cfg.robot = KINOVA_GEN3_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=init_state,
        )
    else:
        raise ValueError(f"Unknown robot: {robot_choice!r}")

    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()

    stage = sim.stage
    robot = scene["robot"]
    env_origins = scene.env_origins

    ee_body_index = robot.find_bodies(EE_BODY_NAME)[0][0]
    joint_ids = robot.find_joints(CONTROLLED_JOINT_NAMES)[0]

    _log(f"\nRobot bodies : {robot.body_names}")
    _log(f"Num bodies   : {len(robot.body_names)}")
    _log(f"Num joints   : {robot.num_joints}")
    _log(f"Joint names  : {robot.joint_names}")
    _log(f"EE body      : {EE_BODY_NAME} (index {ee_body_index})")
    _log(f"Controlled   : {CONTROLLED_JOINT_NAMES}")
    _log(f"Joint IDs    : {joint_ids}")

    limits = robot.data.soft_joint_pos_limits[0, joint_ids]
    for joint_index, name in enumerate(CONTROLLED_JOINT_NAMES):
        _log(f"  {name:20s}: [{limits[joint_index, 0].item():+.4f}, {limits[joint_index, 1].item():+.4f}]")

    _log("\nDisabling collisions on scene objects...")
    toggled = disable_scene_object_collisions(stage)
    _log(f"  Toggled {toggled} collision attributes")

    origin_0 = env_origins[0].cpu().numpy()
    ws_center = ((DESIRED_WS_MIN + DESIRED_WS_MAX) / 2 + origin_0).tolist()
    ws_size = (DESIRED_WS_MAX - DESIRED_WS_MIN).tolist()
    spawn_desired_workspace_box(stage, tuple(ws_center), tuple(ws_size))
    _log("  Spawned translucent desired-workspace box")

    _log(f"\nSampling {num_samples:,} configs ({num_envs} parallel envs)...\n")

    # Build linked-joint index mapping (legacy, for non-closure robots)
    linked_joint_indices: dict[int, int] | None = None
    if robot_cfg.linked_joints:
        linked_joint_indices = {}
        for tgt_name, src_name in robot_cfg.linked_joints.items():
            linked_joint_indices[CONTROLLED_JOINT_NAMES.index(tgt_name)] = (
                CONTROLLED_JOINT_NAMES.index(src_name)
            )
        _log(f"  Linked joints: {robot_cfg.linked_joints}")

    # Resolve elbow-filter body index (optional post-physics filter)
    elbow_filter_body_index: int | None = None
    if robot_cfg.elbow_filter_body:
        elbow_filter_body_index = robot.find_bodies(robot_cfg.elbow_filter_body)[0][0]
        _log(f"  Elbow filter: body={robot_cfg.elbow_filter_body} "
             f"(idx={elbow_filter_body_index}), max={math.degrees(robot_cfg.elbow_filter_max_rad):.1f}°")

    # Antiparallelogram closure condition (physical model)
    antiparallelogram_cols: tuple[int, int, int] | None = None
    if robot_cfg.antiparallelogram_closure:
        _rl = CONTROLLED_JOINT_NAMES.index("rod_left_joint")
        _rr = CONTROLLED_JOINT_NAMES.index("rod_right_joint")
        _cl = CONTROLLED_JOINT_NAMES.index("coupler_left_joint")
        antiparallelogram_cols = (_rl, _rr, _cl)
        _log(f"  Antiparallelogram closure: cols=({_rl}, {_rr}, {_cl})")

    positions, yoshikawa, condition = sample_workspace(
        robot, ee_body_index, joint_ids, env_origins,
        num_samples, num_envs, sim, scene, device,
        mount_height=mount_height,
        collision_min_distance=COLLISION_MIN_DISTANCE,
        collision_adjacency_skip=COLLISION_ADJACENCY_SKIP,
        linked_joint_indices=linked_joint_indices,
        elbow_filter_body_index=elbow_filter_body_index,
        elbow_filter_max_rad=robot_cfg.elbow_filter_max_rad,
        antiparallelogram_cols=antiparallelogram_cols,
        num_settle_steps=robot_cfg.num_settle_steps,
    )

    # ── Append mode: merge with existing data ─────────────────────────
    if args_cli.append:
        existing_files = (
            output_dir / "ee_positions.npy",
            output_dir / "yoshikawa.npy",
            output_dir / "condition_number.npy",
        )
        if all(f.exists() for f in existing_files):
            _log("Append mode: merging with existing data...")
            positions = np.concatenate([np.load(existing_files[0]), positions], axis=0)
            yoshikawa = np.concatenate([np.load(existing_files[1]), yoshikawa], axis=0)
            condition = np.concatenate([np.load(existing_files[2]), condition], axis=0)
            _log(f"  Combined sample count: {len(positions):,}")
        else:
            _log("Append mode: no existing data found, saving as new.")

    np.save(output_dir / "ee_positions.npy", positions)
    np.save(output_dir / "yoshikawa.npy", yoshikawa)
    np.save(output_dir / "condition_number.npy", condition)
    _log(f"Raw data saved to {output_dir}/")

    print_statistics(positions, yoshikawa, condition, mount_height)
    save_statistics_json(
        positions, yoshikawa, condition, mount_height, mount_direction,
        robot_choice, output_dir,
    )

    _log("Done.")
    simulation_app.close()


if __name__ == "__main__":
    main()
