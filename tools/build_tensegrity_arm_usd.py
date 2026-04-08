#!/usr/bin/env python3
"""
build_tensegrity_arm_usd.py
============================
Builds a physics-ready Isaac Lab articulation USD for the tensegrity arm
from a Blender-exported mesh file (tensegrity_arm_cad.usdc).

Kinematic chain — elbow_approx variants (Z-down from ceiling mount):

  /tensegrity_arm  (ArticulationRootAPI, defaultPrim)
      │─ root_joint            fixed  (body0=WORLD, body1=root_link)
  root_link   [upper_arm meshes]                  (0, 0,  0.000)
      │─ elbow_joint          revolute X    @  z = −0.430
  forearm_link [forearm + elbow_cyl]              (0, 0, −0.430)
      │─ wrist_x_joint        revolute X    @  z = −0.836
  wrist_intermediate_link  [KINEMATIC FRAME]      (0, 0, −0.836)
      │─ wrist_y_joint        revolute Y    @  z = −0.836
  wrist_link  [lower_wrist_imu meshes]            (0, 0, −0.836)
      │─ tool_joint           fixed         @  z = −0.972
  tool_link   [EE FRAME — gripper attaches here]  (0, 0, −0.972)

Kinematic chain — physical variants (antiparallelogram four-bar linkage):

  /tensegrity_arm  (ArticulationRootAPI, defaultPrim)
      │─ root_joint                fixed  (body0=WORLD)
  root_link   [upper_arm meshes]                  (0,     0,     0.000)
      ├─ rod_left_joint           revolute X  @ (0, −0.03, −0.36)
      │  rod_left_link  [left rods]               (0, −0.03, −0.360)
      │      └─ coupler_left_joint revolute X @ (0, +0.03, −0.4975)
      │         forearm_link [forearm + rods]      (0,     0, −0.4975)
      │             └─ wrist chain (same as above)
      └─ rod_right_joint          revolute X  @ (0, +0.03, −0.36)
         rod_right_link [right rods]              (0, +0.03, −0.360)
             └─ coupler_right_joint ◄ LOOP CLOSURE → forearm_link

  The two crossing rods form an antiparallelogram (crossed four-bar
  linkage).  coupler_right_joint closes the kinematic loop; PhysX
  enforces the constraint via position projection.
  See doc/antiparallelogram_kinematics.ipynb for the derivation.

Fixed-base articulation
-----------------------
The ``root_joint`` (FixedJoint) anchors root_link to the world frame.
Per the PhysX / Isaac Sim articulation rules:

  - For a fixed-base articulation, the ArticulationRoot schema must be on
    an ancestor of the fixed joint in the USD hierarchy.

Since ``root_joint`` lives under ``/tensegrity_arm``, placing the
ArticulationRootAPI on the ``/tensegrity_arm`` Xform satisfies this rule.
PhysX parses this as a fixed-base articulation with ``root_link`` as the
articulation base link.

When using Robot Assembler to attach a ceiling-mount base or a gripper:
  - Base attachment:  disable or remove ``root_joint``, then assemble
    the ceiling-mount prim at ``root_link``.
  - Gripper attachment:  assemble the Robotiq 2F-140 at ``tool_link``
    (the Robot Assembler creates a new fixed joint automatically).

Isaac Lab ``fix_root_link`` flag
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Because the USD already contains a proper fixed joint, set
``fix_root_link=None`` (or omit it) in your ArticulationRootPropertiesCfg.
Setting it to ``True`` would create a redundant second fixed joint.

Joint limits (elbow_approx):
  elbow_joint        : ±1.2217 rad (±70°, Klein 2023 §4.2 p.79 practical workspace)
  wrist_x_joint      : ±0.8727 rad (±50°, Klein 2023 §4.2 p.79 practical workspace)
  wrist_y_joint      : ±0.8727 rad (±50°, Klein 2023 §4.2 p.79 practical workspace)

Joint limits (physical):
  rod_left_joint     : [−0.6886, +0.5332] rad — from closure condition at ±70° elbow
  rod_right_joint    : [−0.5332, +0.6886] rad — closure-derived mirror of rod_left
  coupler_left_joint : [−0.6886, +0.6886] rad — union of coupler range
  coupler_right_joint: [−0.6886, +0.6886] rad — loop closure joint
  wrist_x_joint      : ±0.8727 rad (±50°, Klein 2023 §4.2 p.79 practical workspace)
  wrist_y_joint      : ±0.8727 rad (±50°, Klein 2023 §4.2 p.79 practical workspace)

Link masses:
- Elbow-approximation variants:
  root_link              : 2.00 kg  (structural upper arm)
  forearm_link           : 2.092 kg (0.192 cylinder approx + 1.90 forearm)
  wrist_intermediate_link: 0.001 kg (kinematic frame — see note below)
  wrist_link             : 0.40 kg  (from Klein 2023 wrist spec)
  tool_link              : 0.001 kg (EE frame — see note below)

- Physical (antiparallelogram) variants:
  root_link              : 2.00 kg  (structural upper arm)
  rod_left_link          : 0.096 kg (one rod pair)
  rod_right_link         : 0.096 kg (one rod pair)
  forearm_link           : 1.90 kg  (forearm only; rod mass moved to rod links)
  wrist_intermediate_link: 0.001 kg (kinematic frame)
  wrist_link             : 0.40 kg
  tool_link              : 0.001 kg (EE frame)

Dummy links — why 0.001 kg + explicit inertia, not omitted mass
---------------------------------------------------------------
wrist_intermediate_link and tool_link are pure kinematic frames: they
carry no physical structure and have no collision geometry.

Mass cannot be omitted for geometry-free links: PhysX computes mass as
volume x density. With no geometry, volume = 0, giving a zero-mass
rigid body, which causes a division by zero in the inertia computation
and either crashes the solver or produces NaN dynamics.

The inertia tensor must also be set explicitly: without collision
geometry PhysX cannot auto-compute it from shape + mass, so it would
also be zero. A zero-inertia body is mathematically ill-posed.

We therefore set both to small but non-zero values:
  mass            = 0.001 kg
  diagonalInertia = (1e-6, 1e-6, 1e-6) kg·m²

USD drive properties are ALL zeroed (stiffness=0, damping=0, maxForce=0).
Isaac Lab ImplicitActuatorCfg overrides them at runtime.

Supported variants
------------------
  elbow_approx      - cylinder elbow, full-resolution meshes  [default]
  elbow_approx_lo   - cylinder elbow, low-resolution meshes
  physical          - antiparallelogram four-bar linkage, full-resolution
  physical_lo       - antiparallelogram four-bar linkage, low-resolution

The physical variants replace the single elbow revolute joint with two
crossing rods (rod_left_link, rod_right_link) and a PhysX loop-closure
joint.  The forearm link origin shifts from z=−0.430 to z=−0.4975
(the coupler-pivot level of the four-bar linkage).  See the
ANTIPARALLELOGRAM LINKAGE CONSTANTS section and
doc/antiparallelogram_kinematics.ipynb for the geometry derivation.

Usage
-----
The script is self-bootstrapping: it finds Isaac Sim via the ``ISAAC_PATH``
environment variable (set automatically by ``python.sh``) and patches
``sys.path`` / registers USD schema plugins at startup — no manual environment
setup is needed beyond choosing an invocation option below.

Option A — Isaac Sim's bundled Python (no conda active):
    conda deactivate
    ~/Isaac/IsaacSim/python.sh \\
      tools/build_tensegrity_arm_usd.py [--variant elbow_approx_lo]

Option B — your own Python 3.11 conda environment:
    # ONE-TIME setup (run once per conda env, re-run after IsaacSim updates):
    source ~/Isaac/IsaacSim/setup_conda_env.sh
    # Then every time:
    python tools/build_tensegrity_arm_usd.py [--variant elbow_approx_lo]

Do NOT run ``python.sh`` while a conda env is active — it will fail with a
clear error.  Either deactivate conda first (Option A) or use Option B.

Output:
  res/Tensegrity/threedof_arm/tensegrity_threedof_arm_<variant>.usd

After building:
  1. Open the output in Isaac Sim — Play to verify joints and collision.
     The robot should remain fixed in place (root_joint anchors to world).
     Toggle Show > Guides in the viewport to inspect collision geometry.
  2. Robot Assembler: attach Robotiq 2F-140 at tool_link.
     For ceiling-mount base attachment, disable root_joint and assemble at root_link.
     Save assembled result as:
             res/Tensegrity/fivedof_manipulator/fivedof_linear_base_elbow_approx.usd
             res/Tensegrity/fivedof_manipulator/fivedof_linear_base_elbow_approx_robotiq2f140.usd
  3. In your Isaac Lab ArticulationCfg, set fix_root_link=None
     (the USD already has the fixed joint — True would create a duplicate).
  4. Update USD paths in tensegrity_robot_cfg.py.
  5. Run scripts/measure_positions.py and update GEOMETRY.md if EE offset changed.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import importlib
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _bootstrap_pxr() -> None:
    """Make pxr importable when running via Isaac Sim's python.sh.

    python.sh sets ISAAC_PATH and LD_LIBRARY_PATH for the kit runtime, but it
    does NOT add the extscache directories to PYTHONPATH.  This function:

      1. Locates the pxr extscache dirs via ISAAC_PATH (or ~/Isaac/IsaacSim).
      2. Inserts them into sys.path so ``from pxr import ...`` works.
      3. Pre-loads the USD shared libraries with RTLD_GLOBAL so their C++
         symbols are globally visible (required by the pxr extension modules).
      4. Registers the PhysxSchema USD plugins via Plug.Registry BEFORE
         PhysxSchema is imported — without this step the schema registry
         raises 'Failed to find plugin for schema type PhysxSchema*' on the
         first Stage.Open / Stage.CreateNew call.
    """
    isaac = os.environ.get("ISAAC_PATH") or os.path.expanduser("~/Isaac/IsaacSim")
    extscache = os.path.join(isaac, "extscache")
    if not os.path.isdir(extscache):
        return

    def _latest(pattern: str) -> str | None:
        matches = sorted(glob.glob(os.path.join(extscache, pattern)))
        return matches[-1] if matches else None

    usd_libs = _latest("omni.usd.libs-*")
    physx    = _latest("omni.usd.schema.physx-*")
    if not usd_libs or not physx:
        return

    # 1. Add to sys.path (Python import resolution)
    for p in [physx, usd_libs]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # 2. Pre-load USD .so files with RTLD_GLOBAL
    for so in sorted(glob.glob(os.path.join(usd_libs, "bin", "libusd_*.so"))):
        try:
            ctypes.CDLL(so, ctypes.RTLD_GLOBAL)
        except OSError:
            pass
    for so in sorted(glob.glob(os.path.join(physx, "bin", "*.so"))):
        try:
            ctypes.CDLL(so, ctypes.RTLD_GLOBAL)
        except OSError:
            pass

    # 3. Register PhysxSchema plugins so USD schema registry knows where they are.
    #    Must happen BEFORE `from pxr import PhysxSchema`.
    from pxr import Plug  # noqa: PLC0415  (intentional late import)
    reg = Plug.Registry()
    for sub in ("PhysxSchema", "PhysxSchemaAddition", "OmniUsdPhysicsDeformableSchema"):
        plugin_resources = os.path.join(physx, "plugins", sub, "resources")
        if os.path.isdir(plugin_resources):
            reg.RegisterPlugins(plugin_resources)


def _bootstrap_robot_schema_python() -> None:
    """Make Isaac Robot Schema Python package importable in standalone scripts.

    In Isaac Sim 5.x the package ``usd.schema.isaac`` is provided by the
    ``isaacsim.robot.schema`` extension. Depending on launch mode, extension
    paths are not always present in ``sys.path`` for standalone scripts.
    """
    isaac = os.environ.get("ISAAC_PATH") or os.path.expanduser("~/Isaac/IsaacSim")
    candidates = [
        os.path.join(isaac, "exts", "isaacsim.robot.schema"),
        os.path.join(isaac, "extsDeprecated", "omni.usd.schema.isaac"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)


_bootstrap_pxr()
_bootstrap_robot_schema_python()

try:
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env and conda_env != "base":
        print(
            f"[ERROR] pxr not found — conda environment '{conda_env}' is active.\n"
            "\n"
            "python.sh uses its own Python interpreter, but a conda env intercepts it.\n"
            "Deactivate conda first, then retry:\n"
            "\n"
            "  conda deactivate\n"
            "  ~/Isaac/IsaacSim/python.sh tools/build_tensegrity_arm_usd.py\n"
            "\n"
            "Or install pxr into your conda env once (Option B):\n"
            "  source ~/Isaac/IsaacSim/setup_conda_env.sh\n"
            "  python tools/build_tensegrity_arm_usd.py",
            file=sys.stderr,
        )
    else:
        print(
            "[ERROR] pxr not found.  Isaac Sim's USD libraries are required.\n"
            "\n"
            "Option A — Isaac Sim's bundled Python (no conda active):\n"
            "  conda deactivate\n"
            "  ~/Isaac/IsaacSim/python.sh tools/build_tensegrity_arm_usd.py\n"
            "\n"
            "Option B — your own Python 3.11 conda env:\n"
            "  source ~/Isaac/IsaacSim/setup_conda_env.sh   # one-time setup\n"
            "  python tools/build_tensegrity_arm_usd.py",
            file=sys.stderr,
        )
    sys.exit(1)

def _import_robot_schema_module():
    """Import Isaac Robot Schema helpers across Isaac Sim version variants."""
    module_names = [
        "usd.schema.isaac.robot_schema",
        "usd.schema.isaac",
        "omni.usd.schema.isaac.robot_schema",
        "omni.usd.schema.isaac",
    ]
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "ApplyRobotAPI"):
                return module
            if hasattr(module, "robot_schema") and hasattr(module.robot_schema, "ApplyRobotAPI"):
                return module.robot_schema
        except Exception:
            continue
    return None


isaac_robot_schema = _import_robot_schema_module()


# ═════════════════════════════════════════════════════════════════════════════
# PATHS
# ═════════════════════════════════════════════════════════════════════════════

_SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
SOURCE_USDC  = PROJECT_ROOT / "res" / "Tensegrity" / "meshes" / "tensegrity_arm_cad.usdc"
OUTPUT_DIR   = PROJECT_ROOT / "res" / "Tensegrity" / "threedof_arm"


# ═════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LinkSpec:
    name: str
    world_z: float    # rest-pose Z relative to robot root
    mass_kg: float
    world_x: float = 0.0
    world_y: float = 0.0
    is_dummy: bool = False
    # Dummy links have no collision geometry.
    # They require both mass and diagonalInertia set explicitly so PhysX
    # never encounters a zero-mass or zero-inertia rigid body.
    # See module docstring for full rationale.


@dataclass(frozen=True)
class JointSpec:
    name: str
    parent: str
    child: str
    joint_type: str   # "revolute" | "fixed"
    axis: str = "X"
    lower_rad: float = 0.0
    upper_rad: float = 0.0
    joint_world_z: float = 0.0
    joint_world_x: float = 0.0
    joint_world_y: float = 0.0


@dataclass(frozen=True)
class MeshSpec:
    source_name: str    # top-level prim name in source USDC
    link: str
    is_collision: bool
    collision_approx: str = "convexHull"
    world_z: float = 0.0
    world_x: float = 0.0
    world_y: float = 0.0


@dataclass(frozen=True)
class ImuSpec:
    """Describes one physical IMU mounting location on the robot.

    Creates a named Xform child prim under parent_link.  The local position is
    computed at build time as ``world_pos − parent_link.world_pos`` so that the
    same ImuSpec works across variants with different link origins.

    The prim is purely documentary (purpose='guide', no physics APIs).

    How to use in Isaac Lab
    -----------------------
    Point ImuCfg at the PARENT LINK (a rigid body), not at this Xform child.
    The local position printed during the build can be copied into
    ImuCfg.OffsetCfg:

        from isaaclab.sensors import ImuCfg

        imu_wrist = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/wrist_link",
            offset=ImuCfg.OffsetCfg(
                pos=( 0.000,  0.012, -0.034),  # ← computed local_pos
                rot=( 1.000,  0.000,  0.000,  0.000),  # ← ImuSpec.local_rot (w,x,y,z)
            ),
        )

    The Xform prim name (e.g. "imu_wrist_frame") is only for visual inspection
    in Isaac Sim — ImuCfg does not reference it.
    """
    name: str                                    # prim name under parent_link
    parent_link: str                             # must match a name in LINKS
    world_pos: tuple[float, float, float]        # metres, in robot root frame
    local_rot: tuple[float, float, float, float] # quaternion (w, x, y, z)


# Inertia tensor applied to every dummy link.
# 1e-6 kg·m² is small enough to be physically negligible yet large enough
# to prevent division-by-zero inside the PhysX inertia solver.
_DUMMY_INERTIA = Gf.Vec3f(1e-6, 1e-6, 1e-6)


@dataclass(frozen=True)
class VariantConfig:
    """Complete specification for one robot variant."""
    links: list[LinkSpec]
    joints: list[JointSpec]
    meshes: list[MeshSpec]
    closure_joints: list[JointSpec] = field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════════════
# ANTIPARALLELOGRAM LINKAGE CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
# Klein (2023), §3.2.1 — see doc/antiparallelogram_kinematics.ipynb
#
# The elbow mechanism is a crossed four-bar linkage (antiparallelogram).
# Frame pivots A, B sit on root_link (spacing k_e).
# Coupler pivots D, C sit on forearm_link (rod length l_e).
# Rods cross: AD and BC.
#
# Symmetric equilibrium angle: θ₀ = arcsin(k_e / l_e).
# The single revolute "elbow_joint" in the elbow_approx variant approximates
# this mechanism; the physical variant replaces it with the actual linkage.

_LINKAGE_ROD_LENGTH = 0.150       # l_e [m]
_LINKAGE_JOINT_SPACING = 0.060    # k_e [m]
_LINKAGE_FRAME_PIVOT_Z = -0.360   # world z of A, B
_LINKAGE_COUPLER_PIVOT_Z = -0.4975  # world z of D, C at equilibrium
_LINKAGE_PIVOT_Y_OFFSET = 0.030   # |y| of each pivot = k_e / 2

# ── Rod and coupler limits derived from ±70° elbow constraint ──────────────
# Computed via delta_for_elbow(±70°) using the antiparallelogram closure
# condition (see doc/antiparallelogram_kinematics.ipynb, Cell 2).
#
# Key relationship:  elbow_angle = rod_left_joint + rod_right_joint
#                    coupler_left = rod_right,  coupler_right = rod_left
#
#   +70° elbow → rod_left = +0.5332 rad,  rod_right = +0.6886 rad
#   −70° elbow → rod_left = −0.6886 rad,  rod_right = −0.5332 rad
_ROD_LEFT_LOWER  = -0.6886   # δ at −70° elbow
_ROD_LEFT_UPPER  = +0.5332   # δ at +70° elbow
_ROD_RIGHT_LOWER = -0.5332   # φ + θ₀ at −70° elbow
_ROD_RIGHT_UPPER = +0.6886   # φ + θ₀ at +70° elbow
_COUPLER_LIMIT   =  0.6886   # max(|coupler_left|, |coupler_right|) ≈ ±39.5°


# ═════════════════════════════════════════════════════════════════════════════
# SHARED LINK / JOINT FRAGMENTS
# ═════════════════════════════════════════════════════════════════════════════

_WRIST_LINKS: list[LinkSpec] = [
    # Pure kinematic frame: splits the 2-DOF wrist into two sequential
    # revolute joints (X then Y).  No geometry, no real mass.
    LinkSpec("wrist_intermediate_link", world_z=-0.836, mass_kg=0.001, is_dummy=True),
    LinkSpec("wrist_link",              world_z=-0.836, mass_kg=0.40),
    # EE coordinate frame only.  The Robotiq 2F-140 root link (with real mass)
    # is attached here as a child via Robot Assembler.
    LinkSpec("tool_link",               world_z=-0.972, mass_kg=0.001, is_dummy=True),
]

_WRIST_JOINTS: list[JointSpec] = [
    JointSpec(
        name="wrist_x_joint",
        parent="forearm_link", child="wrist_intermediate_link",
        joint_type="revolute", axis="X",
        lower_rad=-0.8727, upper_rad=0.8727,  # ±50°, Klein (2023) §4.2 p.79
        joint_world_z=-0.836,
    ),
    JointSpec(
        name="wrist_y_joint",
        parent="wrist_intermediate_link", child="wrist_link",
        joint_type="revolute", axis="Y",
        lower_rad=-0.8727, upper_rad=0.8727,  # ±50°, Klein (2023) §4.2 p.79
        joint_world_z=-0.836,
    ),
    JointSpec(
        name="tool_joint",
        parent="wrist_link", child="tool_link",
        joint_type="fixed",
        joint_world_z=-0.972,
    ),
]


# ═════════════════════════════════════════════════════════════════════════════
# ELBOW-APPROXIMATION VARIANT
# ═════════════════════════════════════════════════════════════════════════════

_LINKS_ELBOW_APPROX: list[LinkSpec] = [
    LinkSpec("root_link",    world_z=0.000,  mass_kg=2.00),
    LinkSpec("forearm_link", world_z=-0.430, mass_kg=2.092),
    *_WRIST_LINKS,
]

_JOINTS_ELBOW_APPROX: list[JointSpec] = [
    JointSpec(
        name="elbow_joint",
        parent="root_link", child="forearm_link",
        joint_type="revolute", axis="X",
        lower_rad=-1.2217, upper_rad=1.2217,  # ±70°, Klein (2023) §4.2 p.79
        joint_world_z=-0.430,
    ),
    *_WRIST_JOINTS,
]


# ═════════════════════════════════════════════════════════════════════════════
# PHYSICAL (ANTIPARALLELOGRAM) VARIANT
# ═════════════════════════════════════════════════════════════════════════════
# Two crossing rod links replace the single elbow_joint.  A PhysX loop-
# closure joint on the second rod completes the four-bar linkage.
# Rod masses: 0.096 kg each (total 0.192 kg, matching elbow cylinder mass).
# Forearm origin moves to the coupler pivot level (z = −0.4975).

_LINKS_PHYSICAL: list[LinkSpec] = [
    LinkSpec("root_link",      world_z=0.000,                    mass_kg=2.00),
    LinkSpec("rod_left_link",  world_y=-_LINKAGE_PIVOT_Y_OFFSET, world_z=_LINKAGE_FRAME_PIVOT_Z,  mass_kg=0.096),
    LinkSpec("rod_right_link", world_y=+_LINKAGE_PIVOT_Y_OFFSET, world_z=_LINKAGE_FRAME_PIVOT_Z,  mass_kg=0.096),
    LinkSpec("forearm_link",   world_z=_LINKAGE_COUPLER_PIVOT_Z, mass_kg=1.90),
    *_WRIST_LINKS,
]

# Articulation tree joints (serial chain).
# Tree path:  root → rod_left → forearm → wrist chain
#             root → rod_right  (branch, closed by coupler_right_joint)
_JOINTS_PHYSICAL: list[JointSpec] = [
    JointSpec(
        name="rod_left_joint",
        parent="root_link", child="rod_left_link",
        joint_type="revolute", axis="X",
        lower_rad=_ROD_LEFT_LOWER, upper_rad=_ROD_LEFT_UPPER,
        joint_world_y=-_LINKAGE_PIVOT_Y_OFFSET,
        joint_world_z=_LINKAGE_FRAME_PIVOT_Z,
    ),
    JointSpec(
        name="coupler_left_joint",
        parent="rod_left_link", child="forearm_link",
        joint_type="revolute", axis="X",
        lower_rad=-_COUPLER_LIMIT, upper_rad=_COUPLER_LIMIT,
        joint_world_y=+_LINKAGE_PIVOT_Y_OFFSET,
        joint_world_z=_LINKAGE_COUPLER_PIVOT_Z,
    ),
    JointSpec(
        name="rod_right_joint",
        parent="root_link", child="rod_right_link",
        joint_type="revolute", axis="X",
        lower_rad=_ROD_RIGHT_LOWER, upper_rad=_ROD_RIGHT_UPPER,
        joint_world_y=+_LINKAGE_PIVOT_Y_OFFSET,
        joint_world_z=_LINKAGE_FRAME_PIVOT_Z,
    ),
    *_WRIST_JOINTS,
]

# Loop-closure joint: connects rod_right_link back to forearm_link,
# forming the four-bar cycle.  PhysX detects the kinematic loop and
# enforces the closure constraint via position projection.
_CLOSURE_JOINTS_PHYSICAL: list[JointSpec] = [
    JointSpec(
        name="coupler_right_joint",
        parent="rod_right_link", child="forearm_link",
        joint_type="revolute", axis="X",
        lower_rad=-_COUPLER_LIMIT, upper_rad=_COUPLER_LIMIT,
        joint_world_y=-_LINKAGE_PIVOT_Y_OFFSET,
        joint_world_z=_LINKAGE_COUPLER_PIVOT_Z,
    ),
]


# ═════════════════════════════════════════════════════════════════════════════
# VARIANT CONFIGURATIONS
# ═════════════════════════════════════════════════════════════════════════════

VARIANTS: dict[str, VariantConfig] = {
    "elbow_approx": VariantConfig(
        links=_LINKS_ELBOW_APPROX,
        joints=_JOINTS_ELBOW_APPROX,
        meshes=[
            MeshSpec("upper_arm_visual",          link="root_link",    is_collision=False, world_z=  0.000),
            MeshSpec("upper_arm_collision",       link="root_link",    is_collision=True,  world_z=  0.000),
            MeshSpec("elbow_approx_visual",       link="forearm_link", is_collision=False, world_z= -0.430),
            MeshSpec("elbow_approx_collision",    link="forearm_link", is_collision=True,  world_z= -0.430),
            MeshSpec("forearm_visual",            link="forearm_link", is_collision=False, world_z= -0.497477),
            MeshSpec("forearm_collision",         link="forearm_link", is_collision=True,  world_z= -0.497477),
            MeshSpec("lower_wrist_imu_visual",    link="wrist_link",   is_collision=False, world_z= -0.836),
            MeshSpec("lower_wrist_imu_collision", link="wrist_link",   is_collision=True,  world_z= -0.836),
        ],
    ),
    "elbow_approx_lo": VariantConfig(
        links=_LINKS_ELBOW_APPROX,
        joints=_JOINTS_ELBOW_APPROX,
        meshes=[
            MeshSpec("upper_arm_visual_low_res",       link="root_link",    is_collision=False, world_z=  0.000),
            MeshSpec("upper_arm_collision",            link="root_link",    is_collision=True,  world_z=  0.000),
            MeshSpec("elbow_approx_visual",            link="forearm_link", is_collision=False, world_z= -0.430),
            MeshSpec("elbow_approx_collision",         link="forearm_link", is_collision=True,  world_z= -0.430),
            MeshSpec("forearm_visual_low_res",         link="forearm_link", is_collision=False, world_z= -0.497477),
            MeshSpec("forearm_collision",              link="forearm_link", is_collision=True,  world_z= -0.497477),
            MeshSpec("lower_wrist_imu_visual_low_res", link="wrist_link",   is_collision=False, world_z= -0.836),
            MeshSpec("lower_wrist_imu_collision",      link="wrist_link",   is_collision=True,  world_z= -0.836),
        ],
    ),
    "physical": VariantConfig(
        links=_LINKS_PHYSICAL,
        joints=_JOINTS_PHYSICAL,
        meshes=[
            MeshSpec("upper_arm_visual",           link="root_link",      is_collision=False, world_z= 0.000),
            MeshSpec("upper_arm_collision",        link="root_link",      is_collision=True,  world_z= 0.000),
            MeshSpec("joint_rods_left_visual",     link="rod_left_link",  is_collision=False, world_y=-0.03, world_z=-0.36),
            MeshSpec("joint_rods_left_collision",  link="rod_left_link",  is_collision=True,  world_y=-0.03, world_z=-0.36),
            MeshSpec("joint_rods_right_visual",    link="rod_right_link", is_collision=False, world_y=+0.03, world_z=-0.36),
            MeshSpec("joint_rods_right_collision", link="rod_right_link", is_collision=True,  world_y=+0.03, world_z=-0.36),
            MeshSpec("forearm_visual",             link="forearm_link",   is_collision=False, world_z=-0.497477),
            MeshSpec("forearm_collision",          link="forearm_link",   is_collision=True,  world_z=-0.497477),
            MeshSpec("lower_wrist_imu_visual",     link="wrist_link",     is_collision=False, world_z=-0.836),
            MeshSpec("lower_wrist_imu_collision",  link="wrist_link",     is_collision=True,  world_z=-0.836),
        ],
        closure_joints=_CLOSURE_JOINTS_PHYSICAL,
    ),
    "physical_lo": VariantConfig(
        links=_LINKS_PHYSICAL,
        joints=_JOINTS_PHYSICAL,
        meshes=[
            MeshSpec("upper_arm_visual_low_res",       link="root_link",      is_collision=False, world_z= 0.000),
            MeshSpec("upper_arm_collision",            link="root_link",      is_collision=True,  world_z= 0.000),
            MeshSpec("joint_rods_left_visual",         link="rod_left_link",  is_collision=False, world_y=-0.03, world_z=-0.36),
            MeshSpec("joint_rods_left_collision",      link="rod_left_link",  is_collision=True,  world_y=-0.03, world_z=-0.36),
            MeshSpec("joint_rods_right_visual",        link="rod_right_link", is_collision=False, world_y=+0.03, world_z=-0.36),
            MeshSpec("joint_rods_right_collision",     link="rod_right_link", is_collision=True,  world_y=+0.03, world_z=-0.36),
            MeshSpec("forearm_visual_low_res",         link="forearm_link",   is_collision=False, world_z=-0.497477),
            MeshSpec("forearm_collision",              link="forearm_link",   is_collision=True,  world_z=-0.497477),
            MeshSpec("lower_wrist_imu_visual_low_res", link="wrist_link",     is_collision=False, world_z=-0.836),
            MeshSpec("lower_wrist_imu_collision",      link="wrist_link",     is_collision=True,  world_z=-0.836),
        ],
        closure_joints=_CLOSURE_JOINTS_PHYSICAL,
    ),
}

_VARIANT_FILENAMES: dict[str, str] = {
    "elbow_approx":    "tensegrity_threedof_arm_elbow_approx.usd",
    "elbow_approx_lo": "tensegrity_threedof_arm_elbow_approx_lo.usd",
    "physical":        "tensegrity_threedof_arm_physical.usd",
    "physical_lo":     "tensegrity_threedof_arm_physical_lo.usd",
}


# ═════════════════════════════════════════════════════════════════════════════
# IMU FRAME POSITIONS
# ═════════════════════════════════════════════════════════════════════════════
# IMU mounting locations are determined by the physical hardware, not by
# visual mesh resolution, so these are shared across all VARIANTS.
#
# world_pos is the IMU chip centre in the robot root frame (metres).
# local_pos is computed at build time as world_pos − parent_link.world_pos.
#
# local_rot is a quaternion (w, x, y, z) describing how the IMU's sensor
#   frame is oriented relative to the link frame.
#   Identity (1,0,0,0) means the IMU axes are aligned with the link axes.
#   Update from your CAD reference if the chip is rotated on its mount.

IMU_FRAMES: list[ImuSpec] = [
    # Forearm IMU — mounted on the forearm link, near the wrist joint.
    # The IMU chip axes are rotated 90° around the local Y axis.
    ImuSpec(
        name="imu_forearm_frame",
        parent_link="forearm_link",
        world_pos=(0.03735, 0.000, -0.7422),
        local_rot=(0.707, 0.000, 0.707, 0.000),
    ),
    # Wrist IMU — mounted on the lower wrist / IMU housing link.
    ImuSpec(
        name="imu_wrist_frame",
        parent_link="wrist_link",
        world_pos=(0.000, 0.000, -0.93),
        local_rot=(1.000, 0.000, 0.000, 0.000),
    ),
]


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _find_prim_by_name(stage: Usd.Stage, name: str) -> Optional[Usd.Prim]:
    """Traverse stage and return the first prim whose GetName() == name.

    Robust against Blender wrapping objects under a 'Scene'/'Collection' Scope.
    """
    for prim in stage.Traverse():
        if prim.GetName() == name:
            return prim
    return None


def _set_translate(prim: Usd.Prim, xyz: tuple[float, float, float]) -> None:
    """Update xformOp:translate, preserving any existing rotate/scale ops.

    Blender's Y→Z-up conversion may have baked a rotation op. We only
    overwrite the translate value, leaving the op order intact.
    """
    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(*xyz))
            return
    xformable.AddTranslateOp().Set(Gf.Vec3d(*xyz))


def _apply_collision_to_subtree(prim: Usd.Prim, approximation: str) -> int:
    """Recursively apply CollisionAPI + MeshCollisionAPI to every Mesh in subtree."""
    count = 0
    if prim.IsA(UsdGeom.Mesh):
        UsdPhysics.CollisionAPI.Apply(prim)
        mesh_coll = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_coll.CreateApproximationAttr().Set(approximation)
        count += 1
    for child in prim.GetChildren():
        count += _apply_collision_to_subtree(child, approximation)
    return count


def _set_purpose_subtree(prim: Usd.Prim, purpose: str) -> None:
    """Set UsdGeom.Imageable purpose on *prim* and every descendant.

    For collision meshes we use ``purpose='guide'`` so they are:
      - Invisible in the default render viewport (no double-rendering).
      - Still visible when Show > Guides is toggled in Isaac Sim.
      - Still fully active for physics (CollisionAPI is orthogonal to purpose).
    """
    imageable = UsdGeom.Imageable(prim)
    if imageable:
        imageable.CreatePurposeAttr(purpose)
    for child in prim.GetChildren():
        _set_purpose_subtree(child, purpose)


def _copy_materials_from_source(
    src_stage: Usd.Stage,
    dst_stage: Usd.Stage,
    dst_robot_path: str,
) -> int:
    """Copy all UsdShade.Material prims from source USDC into output stage.

    Blender can author materials at varying depths (for example
    ``/Materials/*`` or ``/World/Looks/*``). We therefore traverse the full
    source stage and copy each material prim to:

      dst = <robot_path> + <source_material_path>

    This keeps material names stable while avoiding copying unrelated scene
    geometry from the source file.

    Returns the number of material prims copied.
    """
    copied = 0
    copied_paths: set[str] = set()

    for prim in src_stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue

        src_path = prim.GetPath()
        dst_path = Sdf.Path(f"{dst_robot_path}{src_path}")

        # Ensure the parent scope exists before copying the material spec.
        parent = dst_path.GetParentPath()
        if parent and str(parent):
            UsdGeom.Scope.Define(dst_stage, str(parent))

        ok = Sdf.CopySpec(
            src_stage.GetRootLayer(), src_path,
            dst_stage.GetRootLayer(), dst_path,
        )
        if ok:
            copied += 1
            copied_paths.add(str(dst_path))
            print(f"  Mats    : copied {src_path} → {dst_path}")

    return len(copied_paths)


def _remap_material_bindings(
    dst_stage: Usd.Stage,
    robot_path: str,
) -> int:
    """Fix material:binding paths so they point into the robot subtree.

    When we copy meshes from the source USDC, their ``material:binding``
    relationships still point at absolute source paths like
    ``/Materials/Material_001``.  After we copy the material scope into
    ``/tensegrity_arm/Materials/Material_001`` we need to remap the
    bindings.

    Returns the number of bindings remapped.
    """
    remapped = 0
    robot_prefix = robot_path.rstrip("/")

    for prim in dst_stage.Traverse():
        for rel in prim.GetRelationships():
            rel_name = rel.GetName()
            if not rel_name.startswith("material:binding"):
                continue

            targets = rel.GetTargets()
            if not targets:
                continue

            new_targets = []
            changed = False
            for target in targets:
                target_str = str(target)

                # Already remapped into robot subtree.
                if target_str.startswith(robot_prefix + "/"):
                    new_targets.append(target)
                    continue

                # Preserve absolute source path suffix under robot root:
                # /World/Looks/Blue -> /tensegrity_arm/World/Looks/Blue
                candidate = Sdf.Path(f"{robot_prefix}{target_str}")
                if dst_stage.GetPrimAtPath(candidate).IsValid():
                    new_targets.append(candidate)
                    changed = True
                else:
                    new_targets.append(target)

            if changed:
                rel.SetTargets(new_targets)
                remapped += 1

    return remapped


def _create_fallback_material(
    stage: Usd.Stage,
    robot_path: str,
) -> str:
    """Create a neutral grey UsdPreviewSurface material as a fallback.

    Returns the Sdf.Path string of the created material.
    Used when the source USDC contains no materials at all.
    """
    mat_scope_path = f"{robot_path}/Materials"
    mat_path = f"{mat_scope_path}/default_grey"

    UsdGeom.Scope.Define(stage, mat_scope_path)
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.6, 0.6, 0.6)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat_path


def _bind_fallback_material_to_visuals(
    stage: Usd.Stage,
    robot_path: str,
    fallback_mat_path: str,
) -> int:
    """Bind the fallback material to every visual mesh that has no binding.

    Only meshes whose parent subtree is NOT purpose='guide' (i.e. not
    collision meshes) are considered.

    Returns the number of meshes bound.
    """
    bound = 0
    mat_prim = stage.GetPrimAtPath(fallback_mat_path)
    if not mat_prim or not mat_prim.IsValid():
        return 0
    mat = UsdShade.Material(mat_prim)

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        # Skip collision meshes (purpose=guide)
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            purpose = imageable.ComputePurpose()
            if purpose == UsdGeom.Tokens.guide:
                continue
        # Skip if already has a material binding
        binding_api = UsdShade.MaterialBindingAPI(prim)
        (existing_mat, _) = binding_api.ComputeBoundMaterial()
        if existing_mat:
            continue
        # Bind fallback
        UsdShade.MaterialBindingAPI.Apply(prim)
        binding_api = UsdShade.MaterialBindingAPI(prim)
        binding_api.Bind(mat)
        bound += 1

    return bound


def _apply_robot_schema_metadata(
    robot_prim: Usd.Prim,
    link_prims: list[Usd.Prim],
    joint_prims: list[Usd.Prim],
) -> tuple[bool, int, int]:
    """Apply Robot/Link/Joint schemas and populate robotLinks/robotJoints.

    Robot Assembler and Gain Tuner rely on these robot-schema relationships,
    not only on raw physics articulation schemas.
    """
    try:
        # Robot root API.
        robot_prim.AddAppliedSchema("IsaacRobotAPI")

        # Per-link and per-joint marker APIs.
        for link_prim in link_prims:
            link_prim.AddAppliedSchema("IsaacLinkAPI")
        for joint_prim in joint_prims:
            joint_prim.AddAppliedSchema("IsaacJointAPI")

        # Relationship lists expected by Robot Schema tooling.
        links_rel = robot_prim.GetRelationship("isaac:physics:robotLinks")
        if not links_rel:
            links_rel = robot_prim.CreateRelationship("isaac:physics:robotLinks", custom=True)

        joints_rel = robot_prim.GetRelationship("isaac:physics:robotJoints")
        if not joints_rel:
            joints_rel = robot_prim.CreateRelationship("isaac:physics:robotJoints", custom=True)

        links_rel.SetTargets([prim.GetPath() for prim in link_prims])
        joints_rel.SetTargets([prim.GetPath() for prim in joint_prims])

        return True, len(link_prims), len(joint_prims)
    except Exception:
        return False, 0, 0


def _copy_mesh_into_link(
    src_stage: Usd.Stage,
    dst_stage: Usd.Stage,
    ms: MeshSpec,
    robot_path: str,
    link_map: dict[str, LinkSpec],
) -> Usd.Prim:
    """Copy a mesh prim from source USDC into the correct link in the output stage.

    Sets the translate to link-local space (full 3-D offset):
      local = mesh.world − parent_link.world
    This corrects for Blender placing all mesh origins in robot-root (world) space.

    For collision meshes (``ms.is_collision=True``):
      1. Applies ``UsdPhysics.CollisionAPI`` + ``MeshCollisionAPI`` to every
         ``UsdGeom.Mesh`` descendant.
      2. Sets ``purpose='guide'`` on the entire subtree so the collision
         geometry is hidden from the default render viewport.
    """
    src_prim = _find_prim_by_name(src_stage, ms.source_name)
    if src_prim is None:
        raise RuntimeError(
            f"Source prim '{ms.source_name}' not found in source USDC.\n"
            "  Verify Blender object names match MeshSpec.source_name values."
        )

    dst_path_str = f"{robot_path}/{ms.link}/{ms.source_name}"
    dst_path     = Sdf.Path(dst_path_str)

    ok = Sdf.CopySpec(
        src_stage.GetRootLayer(), src_prim.GetPath(),
        dst_stage.GetRootLayer(), dst_path,
    )
    if not ok:
        raise RuntimeError(f"Sdf.CopySpec failed: {src_prim.GetPath()} → {dst_path}")

    lk = link_map[ms.link]
    local_offset = (ms.world_x - lk.world_x, ms.world_y - lk.world_y, ms.world_z - lk.world_z)

    dst_prim = dst_stage.GetPrimAtPath(dst_path_str)
    if not dst_prim.IsValid():
        raise RuntimeError(f"Copied prim not reachable at '{dst_path_str}'")

    _set_translate(dst_prim, local_offset)

    if ms.is_collision:
        n = _apply_collision_to_subtree(dst_prim, ms.collision_approx)
        if n == 0:
            print(
                f"  [WARN] No UsdGeom.Mesh inside '{ms.source_name}'"
                " — CollisionAPI not applied."
            )
        # Hide collision meshes from the render viewport.
        _set_purpose_subtree(dst_prim, UsdGeom.Tokens.guide)
    return dst_prim


def _create_imu_frames(
    stage: Usd.Stage,
    imu_frames: list[ImuSpec],
    robot_path: str,
    link_map: dict[str, LinkSpec],
) -> None:
    """Create named Xform child prims for every IMU mounting location.

    local_pos is computed at build time as ``world_pos − parent_link.world_pos``
    so that the same ImuSpec works across variants with different link origins.

    Purpose is set to 'guide' so the frames do not appear in renders but
    are visible in the Stage panel for inspection.
    """
    for imu in imu_frames:
        if imu.parent_link not in link_map:
            raise ValueError(
                f"ImuSpec '{imu.name}' references unknown link '{imu.parent_link}'.\n"
                f"  Valid link names: {list(link_map.keys())}"
            )
        lk = link_map[imu.parent_link]
        local_pos = (
            imu.world_pos[0] - lk.world_x,
            imu.world_pos[1] - lk.world_y,
            imu.world_pos[2] - lk.world_z,
        )

        frame_path  = f"{robot_path}/{imu.parent_link}/{imu.name}"
        frame_xform = UsdGeom.Xform.Define(stage, frame_path)

        frame_xform.AddTranslateOp().Set(Gf.Vec3d(*local_pos))
        frame_xform.AddOrientOp().Set(
            Gf.Quatf(imu.local_rot[0], imu.local_rot[1],
                     imu.local_rot[2], imu.local_rot[3])
        )

        frame_xform.CreatePurposeAttr(UsdGeom.Tokens.guide)

        print(
            f"  IMU     : {imu.name:30s}  → {imu.parent_link}"
            f"  pos=({local_pos[0]:+.4f},{local_pos[1]:+.4f},{local_pos[2]:+.4f})"
        )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN BUILD
# ═════════════════════════════════════════════════════════════════════════════

def build_robot_usd(
    variant_key: str,
    src_usd_path: Path,
    output_path: Path,
) -> None:
    variant = VARIANTS[variant_key]
    link_map = {lk.name: lk for lk in variant.links}

    print(f"\n  Source  : {src_usd_path}")
    if not src_usd_path.exists():
        raise FileNotFoundError(f"Source USDC not found: {src_usd_path}")
    src_stage = Usd.Stage.Open(str(src_usd_path))
    if not src_stage:
        raise RuntimeError(f"Failed to open source stage: {src_usd_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Output  : {output_path}")

    stage = Usd.Stage.CreateNew(str(output_path))
    stage.SetMetadata("upAxis", "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # ── Articulation root ──────────────────────────────────────────────────
    robot_path  = "/tensegrity_arm"
    robot_xform = UsdGeom.Xform.Define(stage, robot_path)
    robot_prim  = robot_xform.GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(robot_prim)

    physx_art = PhysxSchema.PhysxArticulationAPI.Apply(robot_prim)
    physx_art.CreateEnabledSelfCollisionsAttr(False)
    physx_art.CreateSolverPositionIterationCountAttr(16)
    physx_art.CreateSolverVelocityIterationCountAttr(4)
    physx_art.CreateSleepThresholdAttr(0.00005)
    physx_art.CreateStabilizationThresholdAttr(0.00001)
    stage.SetDefaultPrim(robot_prim)

    # ── Links ──────────────────────────────────────────────────────────────
    link_prim_map: dict[str, Usd.Prim] = {}

    for lk in variant.links:
        lk_path  = f"{robot_path}/{lk.name}"
        lk_xform = UsdGeom.Xform.Define(stage, lk_path)
        lk_prim  = lk_xform.GetPrim()
        lk_xform.AddTranslateOp().Set(Gf.Vec3d(lk.world_x, lk.world_y, lk.world_z))

        UsdPhysics.RigidBodyAPI.Apply(lk_prim)
        mass_api = UsdPhysics.MassAPI.Apply(lk_prim)
        mass_api.CreateMassAttr().Set(float(lk.mass_kg))

        if lk.is_dummy:
            # No collision geometry → PhysX cannot auto-compute inertia.
            # Set a tiny but non-zero diagonal tensor explicitly to prevent
            # division-by-zero in the inertia solver.
            mass_api.CreateDiagonalInertiaAttr().Set(_DUMMY_INERTIA)
        # For real links: diagonalInertia left unset → PhysX computes from
        # collision mesh geometry + mass automatically.

        link_prim_map[lk.name] = lk_prim
        tag = " [dummy]" if lk.is_dummy else ""
        print(f"  Link    : {lk.name:30s}  z={lk.world_z:+.4f}  m={lk.mass_kg} kg{tag}")

    # ── Meshes ─────────────────────────────────────────────────────────────
    for ms in variant.meshes:
        role = "collision" if ms.is_collision else "visual  "
        lk   = link_map[ms.link]
        local_offset = (ms.world_x - lk.world_x, ms.world_y - lk.world_y, ms.world_z - lk.world_z)
        print(
            f"  Mesh    : {ms.source_name:35s}  [{role}]"
            f"  → {ms.link}  local=({local_offset[0]:+.4f},{local_offset[1]:+.4f},{local_offset[2]:+.4f})"
        )
        _copy_mesh_into_link(src_stage, stage, ms, robot_path, link_map)

    # ── Materials ──────────────────────────────────────────────────────────
    # Copy materials from Blender source USDC so that material:binding
    # relationships on the visual meshes resolve correctly.
    n_mats = _copy_materials_from_source(src_stage, stage, robot_path)
    if n_mats > 0:
        n_remap = _remap_material_bindings(stage, robot_path)
        print(f"  Mats    : {n_mats} material(s) copied, {n_remap} binding(s) remapped")
    else:
        print("  Mats    : no materials found in source — creating fallback")
        fb_path = _create_fallback_material(stage, robot_path)
        n_bound = _bind_fallback_material_to_visuals(stage, robot_path, fb_path)
        print(f"  Mats    : fallback material bound to {n_bound} visual mesh(es)")

    # ── IMU frames ────────────────────────────────────────────────────────
    _create_imu_frames(stage, IMU_FRAMES, robot_path, link_map)

    # ── Root joint (fixed to world) ───────────────────────────────────────
    # This FixedJoint anchors root_link to the world frame, making the
    # articulation a fixed-base system.  PhysX requires ArticulationRoot
    # to be on an ancestor of this fixed joint — the /tensegrity_arm Xform
    # satisfies this.
    #
    # body0 (parent) is left EMPTY → PhysX interprets this as "world".
    # body1 (child)  is root_link.
    root_joint_path = f"{robot_path}/root_joint"
    root_joint = UsdPhysics.FixedJoint.Define(stage, root_joint_path)
    # body0 intentionally has NO target → world frame
    root_joint.CreateBody0Rel()  # empty = world
    root_joint.CreateBody1Rel().SetTargets([link_prim_map["root_link"].GetPath()])
    root_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    root_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    root_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    root_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    print(f"  Joint   : {'root_joint':25s}  fixed (world → root_link)")

    joint_prims_for_schema: list[Usd.Prim] = [root_joint.GetPrim()]

    # ── Joints (articulation tree) ──────────────────────────────────────
    for js in variant.joints:
        jpath       = f"{robot_path}/{js.name}"
        parent_prim = link_prim_map[js.parent]
        child_prim  = link_prim_map[js.child]

        parent_lk = link_map[js.parent]
        child_lk  = link_map[js.child]
        local_pos_parent = Gf.Vec3f(
            float(js.joint_world_x - parent_lk.world_x),
            float(js.joint_world_y - parent_lk.world_y),
            float(js.joint_world_z - parent_lk.world_z),
        )
        local_pos_child = Gf.Vec3f(
            float(js.joint_world_x - child_lk.world_x),
            float(js.joint_world_y - child_lk.world_y),
            float(js.joint_world_z - child_lk.world_z),
        )
        identity_rot = Gf.Quatf(1.0, 0.0, 0.0, 0.0)

        if js.joint_type == "revolute":
            joint = UsdPhysics.RevoluteJoint.Define(stage, jpath)
            joint.CreateAxisAttr(js.axis)
            joint.CreateLowerLimitAttr(math.degrees(js.lower_rad))
            joint.CreateUpperLimitAttr(math.degrees(js.upper_rad))

            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(0.0)
            drive.CreateMaxForceAttr(0.0)

            physx_jt = PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
            physx_jt.CreateJointFrictionAttr(0.0)
            physx_jt.CreateMaxJointVelocityAttr(100.0)

            print(
                f"  Joint   : {js.name:25s}  revolute-{js.axis}"
                f"  [{math.degrees(js.lower_rad):.0f}°, {math.degrees(js.upper_rad):.0f}°]"
                f"  parent_local=({local_pos_parent[0]:+.4f},{local_pos_parent[1]:+.4f},{local_pos_parent[2]:+.4f})"
            )
        elif js.joint_type == "fixed":
            joint = UsdPhysics.FixedJoint.Define(stage, jpath)
            print(
                f"  Joint   : {js.name:25s}  fixed"
                f"  parent_local=({local_pos_parent[0]:+.4f},{local_pos_parent[1]:+.4f},{local_pos_parent[2]:+.4f})"
            )
        else:
            raise ValueError(f"Unknown joint_type: '{js.joint_type}'")

        joint.CreateBody0Rel().SetTargets([parent_prim.GetPath()])
        joint.CreateBody1Rel().SetTargets([child_prim.GetPath()])
        joint.CreateLocalPos0Attr().Set(local_pos_parent)
        joint.CreateLocalRot0Attr().Set(identity_rot)
        joint.CreateLocalPos1Attr().Set(local_pos_child)
        joint.CreateLocalRot1Attr().Set(identity_rot)

        joint_prims_for_schema.append(joint.GetPrim())

    # ── Closure joints (loop constraints) ─────────────────────────────────
    for js in variant.closure_joints:
        jpath       = f"{robot_path}/{js.name}"
        parent_prim = link_prim_map[js.parent]
        child_prim  = link_prim_map[js.child]

        parent_lk = link_map[js.parent]
        child_lk  = link_map[js.child]
        local_pos_parent = Gf.Vec3f(
            float(js.joint_world_x - parent_lk.world_x),
            float(js.joint_world_y - parent_lk.world_y),
            float(js.joint_world_z - parent_lk.world_z),
        )
        local_pos_child = Gf.Vec3f(
            float(js.joint_world_x - child_lk.world_x),
            float(js.joint_world_y - child_lk.world_y),
            float(js.joint_world_z - child_lk.world_z),
        )
        identity_rot = Gf.Quatf(1.0, 0.0, 0.0, 0.0)

        joint = UsdPhysics.RevoluteJoint.Define(stage, jpath)
        joint.CreateAxisAttr(js.axis)
        joint.CreateLowerLimitAttr(math.degrees(js.lower_rad))
        joint.CreateUpperLimitAttr(math.degrees(js.upper_rad))

        joint.CreateBody0Rel().SetTargets([parent_prim.GetPath()])
        joint.CreateBody1Rel().SetTargets([child_prim.GetPath()])
        joint.CreateLocalPos0Attr().Set(local_pos_parent)
        joint.CreateLocalRot0Attr().Set(identity_rot)
        joint.CreateLocalPos1Attr().Set(local_pos_child)
        joint.CreateLocalRot1Attr().Set(identity_rot)

        # Exclude from reduced-coordinate articulation so PhysX treats
        # this as a regular constraint enforced by the position/velocity
        # solver, closing the kinematic loop externally.
        joint.GetPrim().CreateAttribute(
            "physics:excludeFromArticulation",
            Sdf.ValueTypeNames.Bool,
            custom=False,
        ).Set(True)

        joint_prims_for_schema.append(joint.GetPrim())
        print(
            f"  Closure : {js.name:25s}  revolute-{js.axis}"
            f"  [{math.degrees(js.lower_rad):.0f}°, {math.degrees(js.upper_rad):.0f}°]"
            f"  excludeFromArticulation"
            f"  (loop: {js.parent} → {js.child})"
        )

    # ── Isaac Robot Schema (required by Robot Assembler) ─────────────────
    ordered_link_prims = [link_prim_map[lk.name] for lk in variant.links]
    ok_schema, n_links_schema, n_joints_schema = _apply_robot_schema_metadata(
        robot_prim,
        ordered_link_prims,
        joint_prims_for_schema,
    )
    if ok_schema:
        module_name = getattr(isaac_robot_schema, "__name__", "internal")
        print(
            "  Schema  : IsaacRobotAPI applied "
            f"(module={module_name}, links={n_links_schema}, joints={n_joints_schema})"
        )
    else:
        module_name = getattr(isaac_robot_schema, "__name__", "not-imported")
        print(
            "  [WARN] Failed to apply Robot Schema metadata; "
            "Robot Assembler / Gain Tuner may not list links and joints. "
            f"(module={module_name})"
        )

    # ── Save ───────────────────────────────────────────────────────────────
    stage.GetRootLayer().Save()
    print(f"\n  ✓ Saved : {output_path}")
    print(
        "\n  Next steps:\n"
        f"    1. Open {output_path.name} in Isaac Sim; Play to verify joints.\n"
        "       The robot should stay fixed in place (root_joint → world).\n"
        "       Toggle Show > Guides to inspect collision geometry.\n"
        "    2. Robot Assembler: attach Robotiq 2F-140 at tool_link.\n"
        "       Save as fivedof_manipulator/fivedof_linear_base_[VARIANT].usd\n"
        "                  fivedof_manipulator/fivedof_linear_base_[VARIANT]_robotiq2f140.usd\n"
        "    3. In your Isaac Lab ArticulationCfg, set fix_root_link=None\n"
        "       (the USD already has the fixed joint).\n"
        "    4. Run scripts/measure_positions.py and update GEOMETRY.md.\n"
    )


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a physics-ready Isaac Lab articulation USD for the tensegrity arm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--variant", "-v", default="elbow_approx",
                   choices=list(VARIANTS.keys()),
                   help="Robot variant to build (default: elbow_approx)")
    p.add_argument("--src", default=str(SOURCE_USDC), metavar="PATH",
                   help="Source mesh USDC from Blender")
    p.add_argument("--out-dir", default=str(OUTPUT_DIR), metavar="DIR",
                   help="Output directory")
    return p.parse_args()


def main() -> None:
    args       = _parse_args()
    filename   = _VARIANT_FILENAMES.get(args.variant, f"tensegrity_threedof_arm_{args.variant}.usd")
    output_usd = Path(args.out_dir) / filename

    print(f"\n{'─' * 70}")
    print(f"  build_tensegrity_arm_usd.py  —  variant: {args.variant}")
    print(f"{'─' * 70}")

    build_robot_usd(args.variant, Path(args.src), output_usd)

    print(f"{'─' * 70}\n")


if __name__ == "__main__":
    main()