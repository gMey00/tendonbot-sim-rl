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
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Monte Carlo FK sampling for 5-DOF tensegrity robot workspace analysis",
)
parser.add_argument(
    "--num_samples", type=int, default=200_000,
    help="Total FK samples (default: 200 000)",
)
parser.add_argument(
    "--num_envs", type=int, default=4096,
    help="Parallel environments for batched sampling (default: 4096)",
)
parser.add_argument(
    "--output_dir", type=str, default=None,
    help="Directory for output files (default: outputs/workspace_analysis)",
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

from tensegrity_pick.tasks.manager_based.tensegrity_pick.proj_base_scene_cfg import (  # noqa: E402
    CONVEYOR_SURFACE_HEIGHT_M,
    CONVEYOR_WIDTH_M,
    DRUM_CENTER_TO_CONVEYOR_EDGE_M,
    DRUM_HEIGHT_M,
    ROBOT_MOUNT_HEIGHT_M,
    ProjBaseSceneCfg,
)


# ── Constants ─────────────────────────────────────────────────────────────

EE_BODY_NAME = "tool_link_0"

# The gripper tip extends 22.5 cm along the EE body's local Z axis.
GRIPPER_TIP_LOCAL_OFFSET = torch.tensor([0.0, 0.0, 0.225])

CONTROLLED_JOINT_NAMES = [
    "base_y_joint",
    "base_z_joint",
    "elbow_joint",
    "wrist_y_joint",
    "wrist_x_joint",
]

# Desired workspace box in env-local coordinates [Klein 2023, Ref. 3].
DESIRED_WS_MIN = np.array([-0.05, -0.40, 0.80])
DESIRED_WS_MAX = np.array([0.35, 0.95, 1.30])


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect EE positions and manipulability via randomised FK.

    Returns
    -------
    positions  : ``(N, 3)`` env-local EE positions.
    yoshikawa  : ``(N,)`` Yoshikawa manipulability values.
    condition  : ``(N,)`` inverse condition numbers.
    """
    joint_limits = robot.data.soft_joint_pos_limits[0, joint_ids]
    lower = joint_limits[:, 0]
    upper = joint_limits[:, 1]

    dt = sim.get_physics_dt()
    num_batches = math.ceil(num_samples / num_envs)
    num_robot_bodies = len(robot.body_names)

    all_positions: list[np.ndarray] = []
    all_yoshikawa: list[np.ndarray] = []
    all_condition: list[np.ndarray] = []
    jacobian_available = True

    for batch_index in range(num_batches):
        random_joint_positions = lower + (upper - lower) * torch.rand(
            num_envs, len(joint_ids), device=device,
        )

        robot.write_joint_position_to_sim(random_joint_positions, joint_ids=joint_ids)
        robot.write_joint_velocity_to_sim(
            torch.zeros(num_envs, len(joint_ids), device=device), joint_ids=joint_ids,
        )

        sim.step()
        scene.update(dt)

        ee_world = robot.data.body_pos_w[:, ee_body_index, :3]
        ee_quat = robot.data.body_quat_w[:, ee_body_index, :]
        tip_offset_world = quat_apply(
            ee_quat, GRIPPER_TIP_LOCAL_OFFSET.to(device).expand(num_envs, -1),
        )
        tip_world = ee_world + tip_offset_world
        all_positions.append((tip_world - env_origins).cpu().numpy())

        if jacobian_available:
            try:
                jacobian_raw = robot.root_physx_view.get_jacobians()
                if batch_index == 0:
                    _log(f"  Jacobian shape: {jacobian_raw.shape}  (ndim={jacobian_raw.ndim})")
                jacobian_linear = extract_ee_linear_jacobian(
                    jacobian_raw, ee_body_index, joint_ids, num_robot_bodies,
                )
                if jacobian_linear is not None:
                    all_yoshikawa.append(compute_yoshikawa_index(jacobian_linear).cpu().numpy())
                    all_condition.append(compute_inverse_condition_number(jacobian_linear).cpu().numpy())
                else:
                    jacobian_available = False
            except Exception as exc:
                if batch_index == 0:
                    _log(f"  WARNING: Jacobian unavailable ({exc}); skipping manipulability")
                jacobian_available = False

        if not jacobian_available:
            all_yoshikawa.append(np.full(num_envs, np.nan, dtype=np.float32))
            all_condition.append(np.full(num_envs, np.nan, dtype=np.float32))

        if (batch_index + 1) % 25 == 0 or batch_index == num_batches - 1:
            _log(f"  Batch {batch_index + 1}/{num_batches}")

    positions = np.concatenate(all_positions, axis=0)[:num_samples]
    yoshikawa = np.concatenate(all_yoshikawa, axis=0)[:num_samples]
    condition = np.concatenate(all_condition, axis=0)[:num_samples]

    if not jacobian_available:
        _log("  NOTE: Manipulability values are NaN (Jacobian was unavailable)")

    return positions, yoshikawa, condition


# ── Statistics ────────────────────────────────────────────────────────────

def print_statistics(
    positions: np.ndarray,
    yoshikawa: np.ndarray,
    condition: np.ndarray,
) -> None:
    """Print workspace and manipulability statistics to stderr."""
    _log(f"\n{'=' * 60}")
    _log("WORKSPACE ANALYSIS RESULTS")
    _log(f"{'=' * 60}")
    _log(f"  Total samples : {len(positions):,}")
    _log(f"  X range       : [{positions[:, 0].min():.4f}, {positions[:, 0].max():.4f}] m")
    _log(f"  Y range       : [{positions[:, 1].min():.4f}, {positions[:, 1].max():.4f}] m")
    _log(f"  Z range       : [{positions[:, 2].min():.4f}, {positions[:, 2].max():.4f}] m")

    inside = np.all(
        (positions >= DESIRED_WS_MIN) & (positions <= DESIRED_WS_MAX), axis=1,
    )
    coverage_percentage = inside.sum() / len(positions) * 100
    _log(f"  Points inside desired WS : {inside.sum():,} ({coverage_percentage:.1f}%)")

    valid_yoshikawa = yoshikawa[np.isfinite(yoshikawa)]
    if len(valid_yoshikawa) > 0:
        _log(f"\n  Yoshikawa Manipulability Index [Ref. 1]:")
        _log(f"    Min   : {valid_yoshikawa.min():.6f}")
        _log(f"    Max   : {valid_yoshikawa.max():.6f}")
        _log(f"    Mean  : {valid_yoshikawa.mean():.6f}")
        _log(f"    Median: {np.median(valid_yoshikawa):.6f}")
        _log(f"    Std   : {valid_yoshikawa.std():.6f}")

        inside_yoshikawa = yoshikawa[inside & np.isfinite(yoshikawa)]
        if len(inside_yoshikawa) > 0:
            _log(f"    Mean (inside desired WS): {inside_yoshikawa.mean():.6f}")
        else:
            _log(f"    Mean (inside desired WS): N/A (no samples)")

    valid_condition = condition[np.isfinite(condition)]
    if len(valid_condition) > 0:
        _log(f"\n  Inverse Condition Number (Isotropy) [Ref. 2]:")
        _log(f"    Min   : {valid_condition.min():.6f}")
        _log(f"    Max   : {valid_condition.max():.6f}")
        _log(f"    Mean  : {valid_condition.mean():.6f}")
        _log(f"    Median: {np.median(valid_condition):.6f}")

    _log(f"\n  Desired workspace box (env-local) [Ref. 3]:")
    _log(f"    X : [{DESIRED_WS_MIN[0]:.2f}, {DESIRED_WS_MAX[0]:.2f}] m")
    _log(f"    Y : [{DESIRED_WS_MIN[1]:.2f}, {DESIRED_WS_MAX[1]:.2f}] m")
    _log(f"    Z : [{DESIRED_WS_MIN[2]:.2f}, {DESIRED_WS_MAX[2]:.2f}] m")

    _log(f"\n  Scene reference points (env-local):")
    _log(f"    Robot mount     : (0.15, 0.00, {ROBOT_MOUNT_HEIGHT_M:.2f})")
    _log(f"    Belt surface    : z = {CONVEYOR_SURFACE_HEIGHT_M:.2f}")
    drum_y = CONVEYOR_WIDTH_M * 0.5 + DRUM_CENTER_TO_CONVEYOR_EDGE_M
    _log(f"    Drum centre     : (0.15, {drum_y:.2f}, 0.00)")
    _log(f"    Drum rim height : z = {DRUM_HEIGHT_M:.2f}")
    _log(f"{'=' * 60}\n")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    num_envs = args_cli.num_envs
    num_samples = args_cli.num_samples
    device = "cuda:0"

    output_dir = Path(args_cli.output_dir or "outputs/workspace_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    sim = SimulationContext(SimulationCfg(dt=1.0 / 60.0, device=device))

    scene_cfg = ProjBaseSceneCfg(num_envs=num_envs, env_spacing=5.0)
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
    positions, yoshikawa, condition = sample_workspace(
        robot, ee_body_index, joint_ids, env_origins,
        num_samples, num_envs, sim, scene, device,
    )

    np.save(output_dir / "ee_positions.npy", positions)
    np.save(output_dir / "yoshikawa.npy", yoshikawa)
    np.save(output_dir / "condition_number.npy", condition)
    _log(f"Raw data saved to {output_dir}/")

    print_statistics(positions, yoshikawa, condition)

    _log("Done.")
    simulation_app.close()


if __name__ == "__main__":
    main()
