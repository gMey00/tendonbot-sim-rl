"""Shirt PBD cloth validation — drop, drape, and robot grasp tests.

Uses the full shirt_place task setup: ShirtPlaceSceneCfg, ClothObject,
``apply_cloth_startup_event``.

Phases
------
Phase 1 — Drop on cube (3 trials × 2.5 s)
    T-shirt released above a rigid 0.20 m cube from three X-offsets
    (centred, edge, miss).  Reports final cloth centroid.

Phase 2 — Drape over sphere (3 s)
    T-shirt released above a rigid sphere (r = 0.08 m).  Reports cloth
    spread (bounding-box Z extent) and centroid height.

Phase 3 — Robot grasp & lift (scripted open-loop, ~13 s)
    T-shirt dropped on conveyor, centroid aligned with EE.  Robot lowers
    with open gripper, closes gradually via hybrid control (PD for driven
    joints, kinematic for passive Robotiq linkage joints), then lifts.
    Reports cloth-on-belt height, gripper closure angle, and cloth rise.

Usage
-----
.. code-block:: bash

    cd src/tensegrity_pick

    # Headless (console output only)
    conda run -n env_isaaclab python3 scripts/model_validation/run_shirt_validation.py \\
        --headless

    # GUI (watch the phases live)
    conda run -n env_isaaclab python3 scripts/model_validation/run_shirt_validation.py
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

# ── AppLauncher (BEFORE any Isaac Lab / Omniverse imports) ────────────

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Shirt cloth validation: drop, drape, and robot grasp tests.",
)
parser.add_argument(
    "--num_envs", type=int, default=1,
    help="Number of parallel environments (default: 1)",
)
parser.add_argument("--record", default=None, help="Directory to write an MP4 + key PNGs.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
if args_cli.record:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac Lab / Omniverse imports (AFTER launcher) ────────────────────

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from tensegrity_pick.tasks.manager_based.shirt_place.shirt_place_scene_cfg import (  # noqa: E402
    CONVEYOR_SURFACE_HEIGHT_M,
    SHIRT_CLOTH_CFG,
    ShirtPlaceSceneCfg,
    UPSTREAM_SETTLE_POS,
)
from tensegrity_pick.tasks.manager_based.shared.cloth_object import (  # noqa: E402
    ClothObject,
    apply_cloth_startup_event,
    disable_complex_colliders_event,
)
from tensegrity_pick.tasks.manager_based.shared.gripper_cfg import (  # noqa: E402
    GRASP_CENTER_LOCAL_Z,
)

# ── Constants ─────────────────────────────────────────────────────────

DT = 0.01  # Physics time-step (s) — matches shirt_place task
RENDER_INTERVAL = 2

# Optional frame recorder (set in main when --record is given).
RECORDER = None


def _maybe_record() -> None:
    if RECORDER is not None:
        RECORDER.maybe_capture()


# Optional position trace (set TRACE_FILE env var to enable) — bypasses Kit's
# stdout redirection so we can verify object vs cloth coordinates.
import os  # noqa: E402
_TRACE_PATH = os.environ.get("SHIRT_TRACE_FILE")
_TRACE_FH = open(_TRACE_PATH, "w") if _TRACE_PATH else None


def _trace(msg: str) -> None:
    if _TRACE_FH is not None:
        _TRACE_FH.write(msg + "\n")
        _TRACE_FH.flush()


# Camera eye offsets (relative to the look-at target).  The grasp/transport
# phase uses a lower, more side-on view to follow the gripper; the drop/drape
# phase uses a higher, more top-down view so the shirt is clearly seen
# descending onto and draping over the *top* of the object.
_GRASP_EYE_OFFSET = (0.60, -0.65, 0.42)
_DROP_EYE_OFFSET = (0.35, -0.55, 0.80)


def _aim_camera_at(point, device) -> None:
    """Point the follow-camera at a *fixed* world location (e.g. the drop target).

    During the drop/drape phases we want the camera locked onto the cube/sphere so
    the viewer watches the shirt descend from above and drape onto it — following
    the cloth centroid instead frames the settled (low) cloth and pushes the small
    object out of frame.  Also raise the eye for a more top-down view of the drape.
    Restore with ``_aim_camera_at_cloth``.
    """
    if RECORDER is None:
        return
    pt = torch.as_tensor(point, device=device, dtype=torch.float32)
    RECORDER._tracker = lambda: pt
    RECORDER._eye_offset = _DROP_EYE_OFFSET


def _aim_camera_at_cloth(cloth) -> None:
    if RECORDER is not None:
        RECORDER._tracker = lambda: cloth.centroid_pos_w[0]
        RECORDER._eye_offset = _GRASP_EYE_OFFSET

# Test cube: rigid, kinematic, for drop tests.  Sized comparably to the ~0.5 m
# shirt so the garment actually drapes OVER it (with edges hanging off) instead
# of sliding off a tiny block — the old 0.2 m cube was sized for a much smaller
# cloth and the correctly-sized shirt just slid off it.
CUBE_SIZE = (0.40, 0.40, 0.40)  # m
CUBE_POS = (0.60, 0.0, CONVEYOR_SURFACE_HEIGHT_M + CUBE_SIZE[2] / 2)
CUBE_TOP_Z = CUBE_POS[2] + CUBE_SIZE[2] / 2  # top surface

# Drape sphere: rigid, kinematic, for drape test (also scaled up to the shirt).
SPHERE_RADIUS = 0.20  # m
SPHERE_POS = (-0.30, 0.0, CONVEYOR_SURFACE_HEIGHT_M + SPHERE_RADIUS)

# Drop height: cloth centroid placed here before release.  Set well above the
# (now taller) objects so the shirt is clearly seen descending onto them.
DROP_Z = CONVEYOR_SURFACE_HEIGHT_M + 0.85

# Phase 1: X-offsets from cube centre for each trial (scaled to the 0.40 m cube:
# centred / over the edge / clean miss onto the belt).
DROP_OFFSETS_X = [0.00, 0.22, 0.50]
DROP_LABELS = ["centred", "edge", "miss"]

# The shirt consistently drifts ~+0.25 m in +Y while it falls and pancakes
# (asymmetric t-shirt panel/sleeve mass), landing off the far side of the
# target; pre-bias the drop target by that much in −Y so it settles *centred*
# on the cube/sphere.  (The X drift is small and run-to-run noisy, so we don't
# compensate it — the 0.40 m cube is wide enough to catch the shirt either way.)
DROP_DRIFT_Y = 0.15
DROP_DRIFT_X = 0.0

# Phase durations (seconds)
DROP_SETTLE_S = 2.5
DRAPE_SETTLE_S = 3.0

# Phase 3: robot joint waypoints
#
# The Robotiq 2F-140 four-bar linkage is NOT constrained in PhysX.
# Driven joints (finger, inner fingers) use PD control to generate
# real contact torques.  Passive joints (K=0) are kinematically
# constrained each step to keep the linkage stable.
#
# Mimic ratios from Isaac Lab's set_finger_joint_pos_robotiq_2f140:
GRIPPER_MIMIC = {
    "right_outer_knuckle_joint":  1.0,
    "left_outer_finger_joint":    0.0,
    "right_outer_finger_joint":   0.0,
    "left_inner_finger_joint":   -1.0,
    "right_inner_finger_joint":  -1.0,
    "left_inner_finger_pad_joint": 1.0,
    "right_inner_finger_pad_joint": 1.0,
}
EE_BODY_CANDIDATES = ("tool_link_0", "robotiq_base_link", "end_effector_link")

# Arm-only waypoints (finger=0 → open; finger=0.7854 → closed)
_HOME = {"base_y_joint": 0.0, "base_z_joint": -0.25,
         "elbow_joint": 0.0, "wrist_y_joint": 0.0,
         "wrist_x_joint": 0.0, "finger_joint": 0.0}

# ── Extended scene config ─────────────────────────────────────────────


@configclass
class ShirtValidationSceneCfg(ShirtPlaceSceneCfg):
    """Shirt-place scene + test objects for cloth validation.

    Inherits robot, conveyor, drum, shirt_proxy, and
    ``replicate_physics = False``.
    """

    test_cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TestCube",
        spawn=sim_utils.CuboidCfg(
            size=CUBE_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.25, 0.35, 0.85),
                metallic=0.05,
                roughness=0.70,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.8,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_POS),
    )

    drape_sphere: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DrapeSphere",
        spawn=sim_utils.SphereCfg(
            radius=SPHERE_RADIUS,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.85, 0.30, 0.25),
                metallic=0.10,
                roughness=0.60,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.8,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=SPHERE_POS),
    )


# ── Helpers ───────────────────────────────────────────────────────────


def _fmt(v: torch.Tensor) -> str:
    """Format a 3-vector ``(x, y, z)``."""
    return f"({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})"


def _teleport_cloth(
    cloth: ClothObject,
    target_centroid: torch.Tensor,
    device: str,
    yaw: float = 0.0,
) -> None:
    """Place the shirt laid *flat* (on its side) with its centroid at *target_centroid*.

    Uses the flat-lay reset so a shirt dropped above an object actually drapes
    over it, instead of the raw upright mesh which topples.
    """
    ids = torch.tensor([0], device=device, dtype=torch.long)
    centroids = target_centroid.reshape(1, 3).to(device)
    yaws = torch.tensor([yaw], device=device)
    cloth.reset_randomized(ids, centroids, yaws)
    cloth.update()


def _step_n(
    sim: SimulationContext,
    scene: InteractiveScene,
    cloth: ClothObject,
    n: int,
    *,
    robot: Articulation | None = None,
    target: torch.Tensor | None = None,
    joint_ids: list[int] | None = None,
    grip_pos: torch.Tensor | None = None,
    grip_ids: list[int] | None = None,
) -> None:
    """Step physics *n* times, optionally holding a robot target.

    If *grip_pos* / *grip_ids* are given, those joints are kinematically
    forced to the given positions every step (bypassing PD drives).
    This is needed for the Robotiq 2F-140 whose passive four-bar joints
    are not constrained in PhysX.
    """
    grip_zero_vel = None
    if grip_pos is not None:
        grip_zero_vel = torch.zeros(1, len(grip_ids), device=grip_pos.device)
    for _ in range(n):
        if robot is not None and target is not None:
            robot.set_joint_position_target(target, joint_ids=joint_ids)
        if grip_pos is not None:
            robot.write_joint_position_to_sim(grip_pos, joint_ids=grip_ids)
            robot.write_joint_velocity_to_sim(grip_zero_vel, joint_ids=grip_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(DT)
        # Refresh cloth state every step so the follow-camera (which tracks the
        # cloth centroid) actually tracks the *falling/draping* cloth.  Updating
        # only at the end left the tracker pinned to the stale drop-height
        # position, framing the cube/sphere drape at the bottom edge / off-screen.
        cloth.update()
        _maybe_record()


def _cloth_bbox_z(cloth: ClothObject) -> tuple[float, float]:
    """``(min_z, max_z)`` of all cloth particles for env 0."""
    zs = cloth.nodal_pos_w[0, :, 2]
    return float(zs.min()), float(zs.max())


def _presettle_cloth(
    sim: SimulationContext,
    scene: InteractiveScene,
    cloth: ClothObject,
    device: str,
    steps: int = 320,
) -> None:
    """Settle one shirt flat on the UPSTREAM (free) belt and adopt its shape.

    Mirrors the RL task's pre-settle: drop the garment from a height onto the
    empty upstream conveyor — away from the robot so the gripper never
    interferes — so it pancakes flat as it lands, then make that relaxed sheet
    the canonical rest shape for all subsequent placements.  This also ensures
    the shirt does NOT start at the robot base (its spawn ``init_pos``), which
    previously made it clip the arm.
    """
    ids = torch.tensor([0], device=device, dtype=torch.long)
    drop = cloth.cfg.pbd_params.presettle_drop_height_m
    z = CONVEYOR_SURFACE_HEIGHT_M + drop - cloth.flat_rest_min_z
    centroid = torch.tensor(
        [[UPSTREAM_SETTLE_POS[0], UPSTREAM_SETTLE_POS[1], z]],
        device=device, dtype=torch.float32,
    )
    cloth.reset_randomized(ids, centroid, torch.zeros(1, device=device))
    for _ in range(steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(DT)
    cloth.recompute_flat_rest_from_current()
    cloth.update()
    zmin, zmax = _cloth_bbox_z(cloth)
    print(f"  [presettle] shirt relaxed on upstream belt: "
          f"thickness={zmax - zmin:.3f} m  (belt={CONVEYOR_SURFACE_HEIGHT_M:.3f})")


# ── Phase 1 — Drop tests ─────────────────────────────────────────────


def run_drop_tests(
    sim: SimulationContext,
    scene: InteractiveScene,
    cloth: ClothObject,
    device: str,
    *,
    robot: Articulation | None = None,
    home_target: torch.Tensor | None = None,
    joint_ids: list[int] | None = None,
    grip_pos: torch.Tensor | None = None,
    grip_ids: list[int] | None = None,
) -> None:
    """Drop cloth above cube from 3 offsets and report results."""
    n_steps = int(DROP_SETTLE_S / DT)

    print(f"\nPhase 1: Drop on Cube  ({len(DROP_OFFSETS_X)} trials × {DROP_SETTLE_S:.1f} s)")
    print("─" * 56)
    print(f"  Cube  {CUBE_SIZE[0]:.2f}×{CUBE_SIZE[1]:.2f}×{CUBE_SIZE[2]:.2f} m"
          f"  pos=({CUBE_POS[0]:.2f}, {CUBE_POS[1]:.2f}, {CUBE_POS[2]:.2f})"
          f"  top Z={CUBE_TOP_Z:.3f}")
    print()

    for i, (dx, label) in enumerate(zip(DROP_OFFSETS_X, DROP_LABELS)):
        target = torch.tensor(
            [CUBE_POS[0] + dx + DROP_DRIFT_X, CUBE_POS[1] - DROP_DRIFT_Y, DROP_Z],
            device=device, dtype=torch.float32,
        )
        _teleport_cloth(cloth, target, device)
        # Lock the camera onto this trial's cube target so the shirt is seen
        # descending from above and draping onto the cube.
        _aim_camera_at((CUBE_POS[0] + dx, CUBE_POS[1], CUBE_TOP_Z + 0.05), device)
        c_after = cloth.centroid_pos_w[0]
        cube_w = scene["test_cube"].data.root_pos_w[0]
        _trace(f"[CUBE t{i+1}] cube_w={[round(v,3) for v in cube_w.tolist()]} "
               f"target={[round(v,3) for v in target.tolist()]} "
               f"centroid_after_teleport={[round(v,3) for v in c_after.tolist()]}")
        _step_n(sim, scene, cloth, n_steps,
                robot=robot, target=home_target, joint_ids=joint_ids,
                grip_pos=grip_pos, grip_ids=grip_ids)

        centroid = cloth.centroid_pos_w[0]
        zmin, zmax = _cloth_bbox_z(cloth)
        _trace(f"[CUBE t{i+1}] centroid_after_settle={[round(v,3) for v in centroid.tolist()]} "
               f"bboxZ=[{zmin:.3f},{zmax:.3f}]")
        # A flat cloth draped over the cube spans from the cube top down toward
        # the belt, so its centroid sits *below* the cube top — checking the
        # centroid height (old logic) wrongly reported draping as "FELL".  Detect
        # draping by the cloth straddling the cube top *and* sitting over the cube
        # footprint (so a "miss" that lands on the belt is not counted).
        near_cube = (
            abs(float(centroid[0]) - CUBE_POS[0]) < CUBE_SIZE[0] / 2 + 0.10
            and abs(float(centroid[1]) - CUBE_POS[1]) < CUBE_SIZE[1] / 2 + 0.10
        )
        straddles = (zmax > CUBE_TOP_Z - 0.05) and (zmin < CUBE_TOP_Z - 0.05)
        verdict = "DRAPED ON CUBE" if (near_cube and straddles) else "fell to belt"

        print(f"  Trial {i+1}/{len(DROP_OFFSETS_X)} — {label} (X offset {dx:+.2f} m)")
        print(f"    Drop from:  {_fmt(target)}")
        print(f"    Final:      {_fmt(centroid)}   bbox Z [{zmin:.3f} – {zmax:.3f}]")
        print(f"    Result:     {verdict}")
        print()


# ── Phase 2 — Drape test ─────────────────────────────────────────────


def run_drape_test(
    sim: SimulationContext,
    scene: InteractiveScene,
    cloth: ClothObject,
    device: str,
    *,
    robot: Articulation | None = None,
    home_target: torch.Tensor | None = None,
    joint_ids: list[int] | None = None,
    grip_pos: torch.Tensor | None = None,
    grip_ids: list[int] | None = None,
) -> bool:
    """Drop cloth above sphere, report spread and draping."""
    n_steps = int(DRAPE_SETTLE_S / DT)

    print(f"Phase 2: Drape over Sphere  ({DRAPE_SETTLE_S:.1f} s)")
    print("─" * 56)
    print(f"  Sphere  r={SPHERE_RADIUS:.2f} m"
          f"  pos=({SPHERE_POS[0]:.2f}, {SPHERE_POS[1]:.2f}, {SPHERE_POS[2]:.2f})"
          f"  top Z={SPHERE_POS[2]+SPHERE_RADIUS:.3f}")
    print()

    target = torch.tensor(
        [SPHERE_POS[0] + DROP_DRIFT_X, SPHERE_POS[1] - DROP_DRIFT_Y, DROP_Z],
        device=device, dtype=torch.float32,
    )
    _teleport_cloth(cloth, target, device)
    # Lock the camera onto the sphere so the shirt is seen descending onto it.
    _aim_camera_at((SPHERE_POS[0], SPHERE_POS[1], SPHERE_POS[2] + SPHERE_RADIUS + 0.05), device)
    c_after = cloth.centroid_pos_w[0]
    sph_w = scene["drape_sphere"].data.root_pos_w[0]
    _trace(f"[SPHERE] sphere_w={[round(v,3) for v in sph_w.tolist()]} "
           f"target={[round(v,3) for v in target.tolist()]} "
           f"centroid_after_teleport={[round(v,3) for v in c_after.tolist()]}")
    _step_n(sim, scene, cloth, n_steps,
            robot=robot, target=home_target, joint_ids=joint_ids,
            grip_pos=grip_pos, grip_ids=grip_ids)

    centroid = cloth.centroid_pos_w[0]
    zmin, zmax = _cloth_bbox_z(cloth)
    spread_z = zmax - zmin
    _trace(f"[SPHERE] centroid_after_settle={[round(v,3) for v in centroid.tolist()]} "
           f"bboxZ=[{zmin:.3f},{zmax:.3f}]")

    draped = spread_z > SPHERE_RADIUS * 1.5
    verdict = "DRAPED" if draped else "FLAT"
    mark = "PASS" if draped else "FAIL"

    print(f"  Drop from: {_fmt(target)}")
    print(f"  Final:     {_fmt(centroid)}")
    print(f"  Spread Z:  {spread_z:.3f} m   (particles [{zmin:.3f} – {zmax:.3f}])")
    print(f"  Result:    {verdict}  [{mark}]")
    print()
    return draped


# ── Phase 3 — Robot grasp & lift ─────────────────────────────────────

# Driven gripper joints (non-zero stiffness — produce contact torques)
_DRIVEN_GRIP = {
    "finger_joint":             1.0,   # K=11.25
    "left_inner_finger_joint": -1.0,   # K=10
    "right_inner_finger_joint":-1.0,   # K=10
}

# Passive gripper joints (K=0 — must be kinematically constrained)
_PASSIVE_GRIP = {
    "right_outer_knuckle_joint":    1.0,
    "left_outer_finger_joint":      0.0,
    "right_outer_finger_joint":     0.0,
    "left_inner_finger_pad_joint":  1.0,
    "right_inner_finger_pad_joint": 1.0,
}


def run_grasp_test(
    sim: SimulationContext,
    scene: InteractiveScene,
    cloth: ClothObject,
    robot: Articulation,
    ee_body_idx: int,
    device: str,
) -> bool:
    """Scripted grasp: drop cloth on conveyor → lower → close → lift.

    Hybrid gripper control — driven joints (finger, inner fingers) use
    PD position control so they generate real contact torques via
    PhysX, while passive joints (K=0) are kinematically constrained
    to keep the four-bar linkage stable.
    """
    print("Phase 3: Robot Grasp & Lift")
    print("─" * 56)

    # Restore the cloth-follow camera for the grasp/transport phase.
    _aim_camera_at_cloth(cloth)

    # ── Joint groups ──────────────────────────────────────────────────
    arm_names = ["base_y_joint", "base_z_joint", "elbow_joint",
                 "wrist_y_joint", "wrist_x_joint"]
    arm_ids = [robot.joint_names.index(j) for j in arm_names]

    driven_names = list(_DRIVEN_GRIP.keys())
    driven_ids = [robot.joint_names.index(j) for j in driven_names]
    driven_mimic = torch.tensor(
        [[_DRIVEN_GRIP[j] for j in driven_names]],
        device=device, dtype=torch.float32,
    )

    passive_names = list(_PASSIVE_GRIP.keys())
    passive_ids = [robot.joint_names.index(j) for j in passive_names]
    passive_mimic = torch.tensor(
        [[_PASSIVE_GRIP[j] for j in passive_names]],
        device=device, dtype=torch.float32,
    )
    passive_zero_vel = torch.zeros(1, len(passive_ids), device=device)

    def _arm_t(d: dict) -> torch.Tensor:
        return torch.tensor(
            [[d.get(j, 0.0) for j in arm_names]],
            device=device, dtype=torch.float32,
        )

    def _step_hybrid(
        n: int,
        arm_target: torch.Tensor,
        finger_angle: float,
    ) -> None:
        """Step with PD for arm + driven grip, kinematic for passive grip."""
        driven_target = finger_angle * driven_mimic
        passive_pos = finger_angle * passive_mimic
        for _ in range(n):
            robot.set_joint_position_target(arm_target, joint_ids=arm_ids)
            robot.set_joint_position_target(driven_target, joint_ids=driven_ids)
            robot.write_joint_position_to_sim(passive_pos, joint_ids=passive_ids)
            robot.write_joint_velocity_to_sim(passive_zero_vel, joint_ids=passive_ids)
            scene.write_data_to_sim()
            sim.step()
            scene.update(DT)
            cloth.update()  # keep the follow-camera tracking the live cloth
            _maybe_record()

    def _print_state(label: str, dur: float | None = None) -> None:
        ee = robot.data.body_pos_w[0, ee_body_idx]
        c = cloth.centroid_pos_w[0]
        d = float(torch.norm(ee - c))
        finger = float(robot.data.joint_pos[0, driven_ids[0]])
        t = f" [{dur:.1f}s]" if dur is not None else ""
        print(f"  {label:8s}{t}  EE={_fmt(ee)}  shirt={_fmt(c)}"
              f"  dist={d:.3f} m  finger={finger:.3f}")

    # ── Hard-reset ALL robot joints to home ───────────────────────────
    n_all = robot.num_joints
    all_ids = list(range(n_all))
    init_pos = torch.zeros(1, n_all, device=device, dtype=torch.float32)
    for j_name, j_val in _HOME.items():
        init_pos[0, robot.joint_names.index(j_name)] = j_val
    robot.write_joint_position_to_sim(init_pos, joint_ids=all_ids)
    robot.write_joint_velocity_to_sim(
        torch.zeros(1, n_all, device=device), joint_ids=all_ids,
    )

    # ── Step 0a: Settle the shirt flat on the UPSTREAM (free) conveyor ──
    # Away from the robot so the gripper does not disturb the settle.
    upstream_drop = torch.tensor(
        [-2.0, 0.0, CONVEYOR_SURFACE_HEIGHT_M + 0.15],
        device=device, dtype=torch.float32,
    )
    _teleport_cloth(cloth, upstream_drop, device)
    _step_hybrid(int(2.0 / DT), _arm_t(_HOME), 0.0)
    _print_state("upstream", 2.0)

    # ── Step 0b: Teleport the settled shirt (shape preserved) to the EE ──
    ee_home = robot.data.body_pos_w[0, ee_body_idx]
    c0 = cloth.centroid_pos_w[0]
    xy_off = torch.zeros(3, device=device)
    xy_off[0] = ee_home[0] - c0[0]
    xy_off[1] = ee_home[1] - c0[1]
    shifted_pos = cloth.nodal_pos_w[0] + xy_off.unsqueeze(0)
    env_id_long = torch.tensor([0], device=device, dtype=torch.long)
    env_id_i32 = torch.tensor([0], device=device, dtype=torch.int32)
    cloth.write_nodal_pos_to_sim(shifted_pos.reshape(1, -1), env_id_long)
    cloth._vel_flat[0] = 0.0
    cloth._cloth_view.set_velocities(cloth._vel_flat, env_id_i32)
    # Brief settle at the work area
    _step_hybrid(int(0.5 / DT), _arm_t(_HOME), 0.0)
    _print_state("settled", 2.5)
    cloth_on_belt_z = float(cloth.centroid_pos_w[0][2])

    # ── Step 1: Lower arm with open gripper so pads straddle cloth ────
    # Position EE so that GRASP_CENTER_LOCAL_Z aligns with cloth height.
    # From empirical calibration: base_z=-0.25 → EE_z≈1.07,
    # base_z=-0.50 → EE_z≈0.95.  Slope ≈ 0.48.
    # Target EE_z = cloth_z - GRASP_CENTER_LOCAL_Z (since local Z is
    # negative = downward).
    target_ee_z = cloth_on_belt_z - GRASP_CENTER_LOCAL_Z  # e.g. 0.81+0.19=1.00
    # Approximate base_z needed:
    approach_bz = -0.25 + (target_ee_z - 1.07) / 0.48
    approach_bz = max(-0.55, min(-0.20, approach_bz))  # clamp
    _APPROACH = {**_HOME, "base_z_joint": approach_bz}

    _step_hybrid(int(2.0 / DT), _arm_t(_APPROACH), 0.0)
    _print_state("lower", 2.0)

    # ── Step 2: Gradually close gripper over 2 s (PD-driven) ─────────
    CLOSE_S = 2.0
    close_steps = int(CLOSE_S / DT)
    finger_target = 0.7854
    arm_approach = _arm_t(_APPROACH)
    for i in range(close_steps):
        frac = (i + 1) / close_steps
        angle = frac * finger_target
        driven_t = angle * driven_mimic
        passive_p = angle * passive_mimic
        robot.set_joint_position_target(arm_approach, joint_ids=arm_ids)
        robot.set_joint_position_target(driven_t, joint_ids=driven_ids)
        robot.write_joint_position_to_sim(passive_p, joint_ids=passive_ids)
        robot.write_joint_velocity_to_sim(passive_zero_vel, joint_ids=passive_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(DT)
        cloth.update()  # keep the follow-camera tracking the live cloth
        _maybe_record()
    _print_state("close", CLOSE_S)

    # ── Step 3: Hold closed (let contacts + arm PD settle) ───────────
    _step_hybrid(int(1.5 / DT), arm_approach, finger_target)
    _print_state("hold", 1.5)

    z_before_lift = float(cloth.centroid_pos_w[0][2])

    # ── Step 4: Gradually lift over 3 s ───────────────────────────────
    LIFT_S = 3.0
    lift_steps = int(LIFT_S / DT)
    lift_bz_start = approach_bz
    lift_bz_end = -0.10
    for i in range(lift_steps):
        frac = (i + 1) / lift_steps
        bz = lift_bz_start + frac * (lift_bz_end - lift_bz_start)
        arm_t = _arm_t({**_APPROACH, "base_z_joint": bz})
        driven_t = finger_target * driven_mimic
        passive_p = finger_target * passive_mimic
        robot.set_joint_position_target(arm_t, joint_ids=arm_ids)
        robot.set_joint_position_target(driven_t, joint_ids=driven_ids)
        robot.write_joint_position_to_sim(passive_p, joint_ids=passive_ids)
        robot.write_joint_velocity_to_sim(passive_zero_vel, joint_ids=passive_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(DT)
        cloth.update()  # keep the follow-camera tracking the live cloth
        _maybe_record()
    _print_state("lift", LIFT_S)

    # ── Step 5: Hold lifted 1.5 s ────────────────────────────────────
    _LIFTED = {**_APPROACH, "base_z_joint": lift_bz_end}
    _step_hybrid(int(1.5 / DT), _arm_t(_LIFTED), finger_target)
    _print_state("hold-up", 1.5)

    z_after_lift = float(cloth.centroid_pos_w[0][2])

    # ── Result ────────────────────────────────────────────────────────
    rise = z_after_lift - z_before_lift
    finger_final = float(robot.data.joint_pos[0, driven_ids[0]])

    # Sub-checks
    grip_ok = finger_final > 0.40
    belt_ok = abs(cloth_on_belt_z - CONVEYOR_SURFACE_HEIGHT_M) < 0.10
    # 0.02 m threshold: with PBD adhesion=0.2 and a scripted (non-learned)
    # trajectory only a fraction of cloth particles are between the pads.
    # The RL task learns to maximise contact through training.
    lift_ok = rise > 0.02

    all_ok = grip_ok and belt_ok and lift_ok
    mark = "PASS" if all_ok else "FAIL"

    print()
    print(f"  Cloth-on-belt Z:   {cloth_on_belt_z:.3f} m  (conveyor={CONVEYOR_SURFACE_HEIGHT_M:.3f})"
          f"  {'OK' if belt_ok else 'FAIL'}")
    print(f"  Gripper closure:   {finger_final:.3f} rad  (target={finger_target:.3f})"
          f"  {'OK' if grip_ok else 'FAIL'}")
    print(f"  Cloth rise (lift): {rise:+.3f} m"
          f"  {'OK' if lift_ok else 'FAIL'}")
    print(f"  Result: [{mark}]")
    print()
    return all_ok


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    num_envs = args_cli.num_envs

    print("=" * 60)
    print(" Shirt Cloth Validation — PBD Particle Cloth")
    print("=" * 60)

    # ── Simulation context with cloth-compatible PhysX settings ───────
    sim_cfg = SimulationCfg(dt=DT)
    sim_cfg.render_interval = RENDER_INTERVAL
    sim_cfg.physx.gpu_collision_stack_size = 2**31 - 1  # int32 max (2**31 overflows negative)
    sim_cfg.physx.solver_type = 1
    sim_cfg.physx.bounce_threshold_velocity = 0.2
    sim_cfg.physx.enable_stabilization = True
    sim_cfg.physx.gpu_max_rigid_contact_count = 2**21
    sim_cfg.physx.gpu_max_rigid_patch_count = 2**19

    sim = SimulationContext(sim_cfg)
    device = sim.device

    # ── Scene: full shirt-place scene + test cube & sphere ────────────
    scene_cfg = ShirtValidationSceneCfg(num_envs=num_envs, env_spacing=5.0)
    if args_cli.record:
        from _recorder import add_record_camera
        add_record_camera(scene_cfg, prim_path="/World/envs/env_0/RecordCam")
        scene_cfg.dome_light.spawn.intensity = 1500.0  # avoid blown-out renders
    scene = InteractiveScene(scene_cfg)

    # ── Apply cloth schemas BEFORE sim.reset() ────────────────────────
    mock_env = SimpleNamespace(sim=sim, scene=scene)
    # Replace the conveyor's complex colliders with the box (as the task does).
    disable_complex_colliders_event(mock_env, None, ("Conveyor", "ConveyorUpstream"))
    apply_cloth_startup_event(mock_env, None, SHIRT_CLOTH_CFG)

    # ── Initialise physics ────────────────────────────────────────────
    sim.reset()
    scene.update(DT)

    # ── ClothObject (GPU tensor access) ───────────────────────────────
    cloth = ClothObject(SHIRT_CLOTH_CFG, num_envs=num_envs, device=device)

    # ── One-off pre-settle on the upstream belt (mirrors the RL task) ──
    # Relax the flattened shirt into a clean draped sheet away from the robot,
    # then use that as the rest shape for every later placement.  Without this
    # the shirt would start at its spawn point on the robot base and clip the
    # arm, and the drop/drape "results" would just reflect the unrelaxed shell.
    _presettle_cloth(sim, scene, cloth, device)

    # ── Frame recorder: follow the cloth so the (small, flat) shirt stays
    # framed across all phases — drop on cube, drape on sphere, upstream
    # settle, grasp. ──
    global RECORDER
    if args_cli.record:
        from _recorder import FrameRecorder
        cam = scene["record_cam"]
        RECORDER = FrameRecorder(
            cam, every=4,
            tracker=lambda: cloth.centroid_pos_w[0],
            eye_offset=_GRASP_EYE_OFFSET,
        )

    # ── Robot info ────────────────────────────────────────────────────
    robot: Articulation = scene["robot"]
    ee_name = next(n for n in EE_BODY_CANDIDATES if n in robot.body_names)
    ee_idx = robot.body_names.index(ee_name)

    # ── Precompute arm + gripper targets for holding during cloth phases ──
    arm_names = ["base_y_joint", "base_z_joint", "elbow_joint",
                 "wrist_y_joint", "wrist_x_joint"]
    arm_ids = [robot.joint_names.index(j) for j in arm_names]
    gripper_names = ["finger_joint"] + list(GRIPPER_MIMIC.keys())
    grip_ids = [robot.joint_names.index(j) for j in gripper_names]

    home_arm = torch.tensor(
        [[_HOME[j] for j in arm_names]],
        device=device, dtype=torch.float32,
    )
    grip_open = torch.tensor(
        [[0.0] + [0.0 for _ in GRIPPER_MIMIC]],
        device=device, dtype=torch.float32,
    )

    print(f"\n  Envs:        {num_envs}")
    print(f"  Cloth verts: {cloth.num_particles}")
    print(f"  EE body:     {ee_name}")
    print(f"  Conveyor Z:  {CONVEYOR_SURFACE_HEIGHT_M:.3f} m")
    print(f"  Cube top Z:  {CUBE_TOP_Z:.3f} m")
    print(f"  Sphere top:  {SPHERE_POS[2]+SPHERE_RADIUS:.3f} m")

    # ── Run validation phases ─────────────────────────────────────────
    run_drop_tests(sim, scene, cloth, device,
                   robot=robot, home_target=home_arm, joint_ids=arm_ids,
                   grip_pos=grip_open, grip_ids=grip_ids)
    drape_ok = run_drape_test(sim, scene, cloth, device,
                              robot=robot, home_target=home_arm, joint_ids=arm_ids,
                              grip_pos=grip_open, grip_ids=grip_ids)
    grasp_ok = run_grasp_test(sim, scene, cloth, robot, ee_idx, device)

    # ── Summary ───────────────────────────────────────────────────────
    checks = [drape_ok, grasp_ok]
    passed = sum(checks)
    total = len(checks)
    print("=" * 60)
    status = "ALL PASSED" if passed == total else f"{passed}/{total} passed"
    print(f" Summary: {status}  (drape={'OK' if drape_ok else 'FAIL'}"
          f"  grasp={'OK' if grasp_ok else 'FAIL'})")
    print(f" Drop tests are report-only — inspect centroid visually.")
    if not grasp_ok:
        print(f" Grasp FAIL indicates PBD cloth-gripper interaction needs")
        print(f" tuning (adhesion, collision shapes, or material params).")
    print("=" * 60)

    if RECORDER is not None:
        RECORDER.save(args_cli.record, "shirt_drape_grasp", fps=20, n_keys=6)

    simulation_app.close()


if __name__ == "__main__":
    main()
