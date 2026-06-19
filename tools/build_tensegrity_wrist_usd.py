#!/usr/bin/env python3
"""
build_tensegrity_wrist_usd.py
=============================
Builds a physics-ready Isaac Lab articulation USD for the standalone
two-DOF tensegrity wrist from a Blender-exported mesh file
(tensegrity_wrist_cad.usdc).

The wrist is the isolated 2-DOF lower section of the full tensegrity arm
(see tools/build_tensegrity_arm_usd.py).  It shares the same kinematic
philosophy as the arm's wrist sub-chain but uses slightly different
dimensions taken directly from the wrist CAD export.  Unlike the arm,
there is only ONE variant — the source USDC has a single mesh resolution
and no physical / approximation variants.

Kinematic chain (Z-down from fixture mount):

  /tensegrity_wrist  (ArticulationRootAPI, defaultPrim)
      │─ root_joint            fixed  (body0=WORLD, body1=root_link)
  root_link   [upper wrist meshes]                (0, 0,  0.00000)
      │─ wrist_x_joint        revolute X    @  z = −0.04273
  wrist_intermediate_link  [KINEMATIC FRAME]      (0, 0, −0.04273)
      │─ wrist_y_joint        revolute Y    @  z = −0.04273
  wrist_lower_link  [lower wrist meshes]          (0, 0, −0.04273)
      │─ tool_joint           fixed         @  z = −0.1083
  tool_link   [EE FRAME — gripper attaches here]  (0, 0, −0.1083)

The X-then-Y revolute pair is co-located at the gimbal centre
(z = −0.04273, the lower-wrist link origin authored in the CAD export),
forming the universal-style 2-DOF wrist.

Fixed-base articulation
-----------------------
``root_joint`` (FixedJoint) anchors root_link to the world frame.  PhysX
requires the ArticulationRoot schema on an ancestor of the fixed joint;
placing it on ``/tensegrity_wrist`` satisfies this.

  - To attach a gripper:  assemble the Robotiq 2F-140 at ``tool_link``.
  - To mount on a fixture/base:  disable ``root_joint`` and assemble at
    ``root_link``.

Because the USD already contains a proper fixed joint, set
``fix_root_link=None`` (or omit it) in your ArticulationRootPropertiesCfg.

Joint limits:
  wrist_x_joint : ±0.8727 rad (±50°, same as arm wrist — Klein 2023 §4.2 p.79)
  wrist_y_joint : ±0.8727 rad (±50°, same as arm wrist — Klein 2023 §4.2 p.79)

Link masses:
  root_link              : 0.590 kg  (upper wrist)
  wrist_intermediate_link: 0.001 kg  (kinematic frame — see note below)
  wrist_lower_link       : 0.132 kg  (lower wrist)
  tool_link              : 0.001 kg  (EE frame — see note below)

Dummy links — why 0.001 kg + explicit inertia, not omitted mass
---------------------------------------------------------------
wrist_intermediate_link and tool_link are pure kinematic frames: they
carry no physical structure and have no collision geometry.

Mass cannot be omitted for geometry-free links: PhysX computes mass as
volume x density.  With no geometry, volume = 0, giving a zero-mass rigid
body, which causes a division by zero in the inertia computation and
either crashes the solver or produces NaN dynamics.

The inertia tensor must also be set explicitly: without collision
geometry PhysX cannot auto-compute it from shape + mass, so it would also
be zero.  A zero-inertia body is mathematically ill-posed.

We therefore set both to small but non-zero values:
  mass            = 0.001 kg
  diagonalInertia = (1e-6, 1e-6, 1e-6) kg·m²

USD drive properties are ALL zeroed (stiffness=0, damping=0, maxForce=0).
Isaac Lab ImplicitActuatorCfg overrides them at runtime.

Usage
-----
The script is self-bootstrapping: it finds Isaac Sim via the ``ISAAC_PATH``
environment variable (set automatically by ``python.sh``) and patches
``sys.path`` / registers USD schema plugins at startup.

Option A — Isaac Sim's bundled Python (no conda active):
    conda deactivate
    ~/Isaac/IsaacSim/python.sh tools/build_tensegrity_wrist_usd.py

Option B — your own Python 3.11 conda environment:
    # ONE-TIME setup (run once per conda env, re-run after IsaacSim updates):
    source ~/Isaac/IsaacSim/setup_conda_env.sh
    # Then every time:
    python tools/build_tensegrity_wrist_usd.py

Do NOT run ``python.sh`` while a conda env is active — it will fail with a
clear error.  Either deactivate conda first (Option A) or use Option B.

Output:
  res/Tensegrity/twodof_wrist/tensegrity_twodof_wrist.usd

After building:
  1. Open the output in Isaac Sim — Play to verify joints and collision.
     The wrist should remain fixed in place (root_joint anchors to world).
     Toggle Show > Guides in the viewport to inspect collision geometry.
  2. Robot Assembler: attach Robotiq 2F-140 at tool_link.
  3. In your Isaac Lab ArticulationCfg, set fix_root_link=None
     (the USD already has the fixed joint — True would create a duplicate).
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import importlib
import math
import os
import sys
from dataclasses import dataclass
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
            "  ~/Isaac/IsaacSim/python.sh tools/build_tensegrity_wrist_usd.py\n"
            "\n"
            "Or install pxr into your conda env once (Option B):\n"
            "  source ~/Isaac/IsaacSim/setup_conda_env.sh\n"
            "  python tools/build_tensegrity_wrist_usd.py",
            file=sys.stderr,
        )
    else:
        print(
            "[ERROR] pxr not found.  Isaac Sim's USD libraries are required.\n"
            "\n"
            "Option A — Isaac Sim's bundled Python (no conda active):\n"
            "  conda deactivate\n"
            "  ~/Isaac/IsaacSim/python.sh tools/build_tensegrity_wrist_usd.py\n"
            "\n"
            "Option B — your own Python 3.11 conda env:\n"
            "  source ~/Isaac/IsaacSim/setup_conda_env.sh   # one-time setup\n"
            "  python tools/build_tensegrity_wrist_usd.py",
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
            if module is not None:
                return module
        except Exception:
            continue
    return None


isaac_robot_schema = _import_robot_schema_module()


# ═════════════════════════════════════════════════════════════════════════════
# PATHS
# ═════════════════════════════════════════════════════════════════════════════

_SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
SOURCE_USDC  = PROJECT_ROOT / "res" / "Tensegrity" / "meshes" / "tensegrity_wrist_cad.usdc"
OUTPUT_DIR   = PROJECT_ROOT / "res" / "Tensegrity" / "twodof_wrist"
OUTPUT_FILENAME = "tensegrity_twodof_wrist.usd"

ROBOT_PATH = "/tensegrity_wrist"


# ═════════════════════════════════════════════════════════════════════════════
# GEOMETRY CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
# Z positions taken directly from the wrist CAD export
# (tensegrity_wrist_cad.usdc).  The upper-wrist meshes sit at the world
# origin; the lower-wrist meshes are authored at z = −0.04273014888167381,
# which is also the gimbal centre where both revolute axes intersect.

_LOWER_LINK_Z = -0.04273014888167381   # lower-wrist link origin / wrist gimbal
_TOOL_Z       = -0.1083                 # tool / EE frame (gripper attach point) [m]

_WRIST_LIMIT  = 0.8727                  # ±0.8727 rad (±50°) — same as arm wrist (Klein 2023 §4.2)


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


# Inertia tensor applied to every dummy link.
# 1e-6 kg·m² is small enough to be physically negligible yet large enough
# to prevent division-by-zero inside the PhysX inertia solver.
_DUMMY_INERTIA = Gf.Vec3f(1e-6, 1e-6, 1e-6)


# ═════════════════════════════════════════════════════════════════════════════
# WRIST SPECIFICATION (single variant)
# ═════════════════════════════════════════════════════════════════════════════

LINKS: list[LinkSpec] = [
    LinkSpec("root_link",               world_z=0.000,         mass_kg=0.590),
    LinkSpec("wrist_intermediate_link", world_z=_LOWER_LINK_Z, mass_kg=0.001, is_dummy=True),
    LinkSpec("wrist_lower_link",        world_z=_LOWER_LINK_Z, mass_kg=0.132),
    LinkSpec("tool_link",               world_z=_TOOL_Z,       mass_kg=0.001, is_dummy=True),
]

JOINTS: list[JointSpec] = [
    JointSpec(
        name="wrist_x_joint",
        parent="root_link", child="wrist_intermediate_link",
        joint_type="revolute", axis="X",
        lower_rad=-_WRIST_LIMIT, upper_rad=_WRIST_LIMIT,
        joint_world_z=_LOWER_LINK_Z,
    ),
    JointSpec(
        name="wrist_y_joint",
        parent="wrist_intermediate_link", child="wrist_lower_link",
        joint_type="revolute", axis="Y",
        lower_rad=-_WRIST_LIMIT, upper_rad=_WRIST_LIMIT,
        joint_world_z=_LOWER_LINK_Z,
    ),
    JointSpec(
        name="tool_joint",
        parent="wrist_lower_link", child="tool_link",
        joint_type="fixed",
        joint_world_z=_TOOL_Z,
    ),
]

MESHES: list[MeshSpec] = [
    MeshSpec("upper_visual",    link="root_link",        is_collision=False, world_z=0.000),
    MeshSpec("upper_collision", link="root_link",        is_collision=True,  world_z=0.000),
    MeshSpec("lower_visual",    link="wrist_lower_link", is_collision=False, world_z=_LOWER_LINK_Z),
    MeshSpec("lower_collision", link="wrist_lower_link", is_collision=True,  world_z=_LOWER_LINK_Z),
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
    ``/root/_materials/*``).  We therefore traverse the full source stage and
    copy each material prim to:

      dst = <robot_path> + <source_material_path>

    This keeps material names stable while avoiding copying unrelated scene
    geometry from the source file.

    Returns the number of material prims copied.
    """
    copied_paths: set[str] = set()

    for prim in src_stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue

        src_path = prim.GetPath()
        dst_path = Sdf.Path(f"{dst_robot_path}{src_path}")

        # Ensure the parent scope exists before copying the material spec.
        parent = dst_path.GetParentPath()
        if parent and str(parent):
            UsdGeom.Scope.Define(dst_stage, parent)

        ok = Sdf.CopySpec(
            src_stage.GetRootLayer(), src_path,
            dst_stage.GetRootLayer(), dst_path,
        )
        if ok:
            copied_paths.add(str(dst_path))

    return len(copied_paths)


def _remap_material_bindings(
    dst_stage: Usd.Stage,
    robot_path: str,
) -> int:
    """Fix material:binding paths so they point into the robot subtree.

    When we copy meshes from the source USDC, their ``material:binding``
    relationships still point at absolute source paths like
    ``/root/_materials/Aluminum___Satin``.  After we copy the material scope
    into ``/tensegrity_wrist/root/_materials/Aluminum___Satin`` we need to
    remap the bindings.

    Returns the number of bindings remapped.
    """
    remapped = 0
    robot_prefix = robot_path.rstrip("/")

    for prim in dst_stage.Traverse():
        for rel in prim.GetRelationships():
            if rel.GetName() != "material:binding":
                continue
            new_targets = []
            changed = False
            for target in rel.GetTargets():
                target_str = str(target)
                if not target_str.startswith(robot_prefix):
                    new_targets.append(Sdf.Path(f"{robot_prefix}{target_str}"))
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
            purpose_attr = imageable.GetPurposeAttr()
            if purpose_attr and purpose_attr.Get() == UsdGeom.Tokens.guide:
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
            links_rel = robot_prim.CreateRelationship("isaac:physics:robotLinks", custom=False)

        joints_rel = robot_prim.GetRelationship("isaac:physics:robotJoints")
        if not joints_rel:
            joints_rel = robot_prim.CreateRelationship("isaac:physics:robotJoints", custom=False)

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
    This corrects for Blender placing mesh origins in robot-root (world) space.

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
                f"  [WARN] no Mesh found under '{ms.source_name}' — "
                "collision API applied to 0 prims"
            )
        # Hide collision meshes from the render viewport.
        _set_purpose_subtree(dst_prim, UsdGeom.Tokens.guide)
    return dst_prim


# ═════════════════════════════════════════════════════════════════════════════
# MAIN BUILD
# ═════════════════════════════════════════════════════════════════════════════

def build_wrist_usd(
    src_usd_path: Path,
    output_path: Path,
) -> None:
    link_map = {lk.name: lk for lk in LINKS}

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
    robot_xform = UsdGeom.Xform.Define(stage, ROBOT_PATH)
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

    for lk in LINKS:
        lk_path  = f"{ROBOT_PATH}/{lk.name}"
        lk_xform = UsdGeom.Xform.Define(stage, lk_path)
        lk_prim  = lk_xform.GetPrim()
        lk_xform.AddTranslateOp().Set(Gf.Vec3d(lk.world_x, lk.world_y, lk.world_z))

        UsdPhysics.RigidBodyAPI.Apply(lk_prim)
        mass_api = UsdPhysics.MassAPI.Apply(lk_prim)
        mass_api.CreateMassAttr().Set(float(lk.mass_kg))

        if lk.is_dummy:
            # Geometry-free links: explicit inertia required (see docstring).
            mass_api.CreateDiagonalInertiaAttr().Set(_DUMMY_INERTIA)
        # For real links: diagonalInertia left unset → PhysX computes from
        # collision mesh geometry + mass automatically.

        link_prim_map[lk.name] = lk_prim
        tag = " [dummy]" if lk.is_dummy else ""
        print(f"  Link    : {lk.name:30s}  z={lk.world_z:+.5f}  m={lk.mass_kg} kg{tag}")

    # ── Meshes ─────────────────────────────────────────────────────────────
    for ms in MESHES:
        role = "collision" if ms.is_collision else "visual  "
        lk   = link_map[ms.link]
        local_offset = (ms.world_x - lk.world_x, ms.world_y - lk.world_y, ms.world_z - lk.world_z)
        print(
            f"  Mesh    : {ms.source_name:18s}  [{role}]"
            f"  → {ms.link}  local=({local_offset[0]:+.4f},{local_offset[1]:+.4f},{local_offset[2]:+.4f})"
        )
        _copy_mesh_into_link(src_stage, stage, ms, ROBOT_PATH, link_map)

    # ── Materials ──────────────────────────────────────────────────────────
    # Copy materials from Blender source USDC so that material:binding
    # relationships on the visual meshes resolve correctly.
    n_mats = _copy_materials_from_source(src_stage, stage, ROBOT_PATH)
    if n_mats > 0:
        n_remap = _remap_material_bindings(stage, ROBOT_PATH)
        print(f"  Mats    : {n_mats} material(s) copied, {n_remap} binding(s) remapped")
    else:
        print("  Mats    : no materials found in source — creating fallback")
        fb_path = _create_fallback_material(stage, ROBOT_PATH)
        n_bound = _bind_fallback_material_to_visuals(stage, ROBOT_PATH, fb_path)
        print(f"  Mats    : fallback material bound to {n_bound} visual mesh(es)")

    # ── Root joint (fixed to world) ───────────────────────────────────────
    # This FixedJoint anchors root_link to the world frame, making the
    # articulation a fixed-base system.  PhysX requires ArticulationRoot
    # to be on an ancestor of this fixed joint — the /tensegrity_wrist Xform
    # satisfies this.
    #
    # body0 (parent) is left EMPTY → PhysX interprets this as "world".
    # body1 (child)  is root_link.
    root_joint_path = f"{ROBOT_PATH}/root_joint"
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
    for js in JOINTS:
        jpath       = f"{ROBOT_PATH}/{js.name}"
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

            # Zeroed USD drive — Isaac Lab ImplicitActuatorCfg overrides at runtime.
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

    # ── Isaac Robot Schema (required by Robot Assembler) ─────────────────
    ordered_link_prims = [link_prim_map[lk.name] for lk in LINKS]
    ok_schema, n_links_schema, n_joints_schema = _apply_robot_schema_metadata(
        robot_prim,
        ordered_link_prims,
        joint_prims_for_schema,
    )
    if ok_schema:
        print(
            f"  Schema  : IsaacRobotAPI applied  "
            f"({n_links_schema} links, {n_joints_schema} joints)"
        )
    else:
        print(
            "  [WARN] Isaac Robot Schema not applied — Robot Assembler may not\n"
            "         recognise links/joints.  Build still produces a valid\n"
            "         PhysX articulation."
        )

    # ── Save ───────────────────────────────────────────────────────────────
    stage.GetRootLayer().Save()
    print(f"\n  ✓ Saved : {output_path}")
    print(
        "\n  Next steps:\n"
        f"    1. Open {output_path.name} in Isaac Sim; Play to verify joints.\n"
        "       The wrist should stay fixed in place (root_joint → world).\n"
        "       Toggle Show > Guides to inspect collision geometry.\n"
        "    2. Robot Assembler: attach Robotiq 2F-140 at tool_link.\n"
        "    3. In your Isaac Lab ArticulationCfg, set fix_root_link=None\n"
        "       (the USD already has the fixed joint).\n"
    )


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a physics-ready Isaac Lab articulation USD for the two-DOF tensegrity wrist.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--src", default=str(SOURCE_USDC), metavar="PATH",
                   help="Source mesh USDC from Blender")
    p.add_argument("--out-dir", default=str(OUTPUT_DIR), metavar="DIR",
                   help="Output directory")
    return p.parse_args()


def main() -> None:
    args       = _parse_args()
    output_usd = Path(args.out_dir) / OUTPUT_FILENAME

    print(f"\n{'─' * 70}")
    print("  build_tensegrity_wrist_usd.py  —  two-DOF wrist (single variant)")
    print(f"{'─' * 70}")

    build_wrist_usd(Path(args.src), output_usd)

    print(f"{'─' * 70}\n")


if __name__ == "__main__":
    main()
