"""Create and bind a rigid UsdPhysics PhysicsMaterial (friction / restitution).

Authors a ``UsdShade.Material`` carrying ``UsdPhysics.MaterialAPI`` and binds it
to a prim with the ``physics`` material purpose.  Runs in plain Python (source
``Scripts/_usdenv.sh``) or in the Isaac Sim Script Editor.

When does this matter?
----------------------
A rigid physics material governs friction/restitution of **rigid colliders**
(the ground plane, table, gripper) that the cloth touches — the cloth side of
that contact is governed by the *cloth* material:

  * PBD particle cloth  -> ``PhysxPBDMaterialAPI.friction``  (set in make_pbd_cloth.py)
  * FEM surface cloth   -> ``PhysxDeformableSurfaceMaterialAPI.dynamicFriction``

So bind this to a collider, not to the cloth mesh, unless you specifically want
to author a rigid material on the cloth prim for a downstream rigid use.
"""

from __future__ import annotations

import argparse
import sys

from pxr import Sdf, Usd, UsdPhysics, UsdShade


def add_physics_material(
    stage: Usd.Stage,
    target_path: str,
    material_path: str = "/root/PhysicsMaterials/cloth_mat",
    static_friction: float = 0.4,
    dynamic_friction: float = 0.3,
    restitution: float = 0.0,
) -> None:
    target = stage.GetPrimAtPath(Sdf.Path(target_path))
    if not target or not target.IsValid():
        raise ValueError(f"No prim at target path: {target_path}")

    # Material must be a UsdShade.Material prim; MaterialAPI is applied to it.
    material = UsdShade.Material.Define(stage, Sdf.Path(material_path))
    mat_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    mat_api.CreateStaticFrictionAttr().Set(static_friction)
    mat_api.CreateDynamicFrictionAttr().Set(dynamic_friction)
    mat_api.CreateRestitutionAttr().Set(restitution)

    UsdShade.MaterialBindingAPI.Apply(target).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )
    print(f"Bound physics material {material_path} (staticF={static_friction}, "
          f"dynamicF={dynamic_friction}, restitution={restitution}) to {target_path}")


def _get_script_editor_stage():
    try:
        import omni.usd  # only available inside Isaac Sim
        return omni.usd.get_context().get_stage()
    except Exception:
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Bind a rigid physics material to a prim.")
    parser.add_argument("--input", help="Input USD (omit when running in the Script Editor).")
    parser.add_argument("--output", help="Output USD (defaults to overwriting --input).")
    parser.add_argument("--prim", default="/root/World/mesh/Mesh", help="Prim to bind the material to.")
    parser.add_argument("--material", default="/root/PhysicsMaterials/cloth_mat")
    parser.add_argument("--static-friction", type=float, default=0.4)
    parser.add_argument("--dynamic-friction", type=float, default=0.3)
    parser.add_argument("--restitution", type=float, default=0.0)
    args = parser.parse_args(argv)

    stage = _get_script_editor_stage()
    standalone = stage is None
    if standalone:
        if not args.input:
            print("ERROR: --input is required when not running in the Isaac Sim Script Editor.")
            return 2
        stage = Usd.Stage.Open(args.input)
        if stage is None:
            print(f"ERROR: cannot open {args.input}")
            return 2

    add_physics_material(
        stage, args.prim, args.material,
        args.static_friction, args.dynamic_friction, args.restitution,
    )

    if standalone:
        out = args.output or args.input
        stage.GetRootLayer().Export(out)
        print(f"Saved to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
