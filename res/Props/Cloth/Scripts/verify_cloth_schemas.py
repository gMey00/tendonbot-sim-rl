#!/usr/bin/env python3
"""Verify that a cloth USD has the expected physics schemas + attributes.

Static (no-solver) check: opens the USD and confirms the particle-cloth or
surface-deformable schema stack is applied with sane attribute values and a
correctly configured physics scene.  Complements ``check_cloth_readiness.py``
(which validates the *bare mesh*) by validating the *authored physics*.

Usage::

    source Scripts/_usdenv.sh
    "$KPY" Scripts/verify_cloth_schemas.py tshirt_pbd.usd
    "$KPY" Scripts/verify_cloth_schemas.py tshirt_xpbd.usd

Exit codes: 0 = all expected schemas present and valid, 1 = problems found.
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from pxr import Usd, UsdGeom, UsdPhysics

try:
    from pxr import PhysxSchema
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"PhysxSchema required (source Scripts/_usdenv.sh): {exc}")


def _get(prim: Usd.Prim, attr: str):
    a = prim.GetAttribute(attr)
    return a.Get() if a and a.IsValid() else None


def verify(stage: Usd.Stage) -> List[str]:
    problems: List[str] = []

    # --- Physics scene -----------------------------------------------------
    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if not scenes:
        problems.append("No UsdPhysics.Scene found.")
    for s in scenes:
        if s.HasAPI(PhysxSchema.PhysxSceneAPI):
            if not _get(s, "physxScene:enableGPUDynamics"):
                problems.append(f"{s.GetPath()}: GPU dynamics not enabled (required for cloth).")
        else:
            problems.append(f"{s.GetPath()}: missing PhysxSceneAPI.")

    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    pbd = [m for m in meshes if m.HasAPI(PhysxSchema.PhysxParticleClothAPI)]
    fem = [m for m in meshes if "OmniPhysicsDeformableBodyAPI" in m.GetAppliedSchemas()]

    if not pbd and not fem:
        problems.append("No mesh has PhysxParticleClothAPI or OmniPhysicsDeformableBodyAPI.")

    # --- PBD particle cloth ------------------------------------------------
    for m in pbd:
        print(f"\n[PBD] {m.GetPath()}")
        if not m.HasAPI(PhysxSchema.PhysxAutoParticleClothAPI):
            problems.append(f"{m.GetPath()}: PhysxParticleClothAPI without PhysxAutoParticleClothAPI (no springs).")
        rel = m.GetRelationship("physxParticle:particleSystem")
        targets = rel.GetTargets() if rel else []
        if not targets:
            problems.append(f"{m.GetPath()}: particleSystem relationship has no target.")
        else:
            ps = stage.GetPrimAtPath(targets[0])
            if not ps or not ps.IsA(PhysxSchema.PhysxParticleSystem):
                problems.append(f"{m.GetPath()}: particleSystem target {targets[0]} is not a PhysxParticleSystem.")
            else:
                pco = _get(ps, "particleContactOffset")
                sro = _get(ps, "solidRestOffset")
                if pco is None or sro is None:
                    problems.append(f"{ps.GetPath()}: missing particleContactOffset/solidRestOffset.")
                elif not (sro < pco):
                    problems.append(f"{ps.GetPath()}: solidRestOffset ({sro}) must be < particleContactOffset ({pco}).")
                print(f"      particleContactOffset={pco} solidRestOffset={sro} "
                      f"iters={_get(ps,'solverPositionIterationCount')} "
                      f"selfColl={_get(ps,'globalSelfCollisionEnabled')}")
        for a in ("physxAutoParticleCloth:springStretchStiffness",
                  "physxAutoParticleCloth:springBendStiffness",
                  "physxAutoParticleCloth:springShearStiffness"):
            v = _get(m, a)
            if v is None or v <= 0:
                problems.append(f"{m.GetPath()}: {a}={v} (expected > 0).")
        mass = _get(m, "physics:mass")
        print(f"      stretch={_get(m,'physxAutoParticleCloth:springStretchStiffness')} "
              f"bend={_get(m,'physxAutoParticleCloth:springBendStiffness')} "
              f"shear={_get(m,'physxAutoParticleCloth:springShearStiffness')} mass={mass}")
        if not mass or mass <= 0:
            problems.append(f"{m.GetPath()}: physics:mass={mass} (expected > 0).")
        if m.HasAPI(UsdPhysics.RigidBodyAPI):
            problems.append(f"{m.GetPath()}: has RigidBodyAPI (incompatible with particle cloth).")

    # --- OmniPhysics surface deformable (beta) ----------------------------
    for m in fem:
        print(f"\n[surface deformable] {m.GetPath()}")
        applied = m.GetAppliedSchemas()
        if "OmniPhysicsSurfaceDeformableSimAPI" not in applied:
            problems.append(f"{m.GetPath()}: missing OmniPhysicsSurfaceDeformableSimAPI.")
        if not (m.HasAPI(UsdPhysics.CollisionAPI) or m.HasAPI(PhysxSchema.PhysxCollisionAPI)):
            problems.append(f"{m.GetPath()}: surface deformable needs a collision API.")
        rest = _get(m, "omniphysics:restShapePoints")
        tri = _get(m, "omniphysics:restTriVtxIndices")
        if not rest or not tri:
            problems.append(f"{m.GetPath()}: restShapePoints/restTriVtxIndices not authored.")
        print(f"      restPoints={len(rest) if rest else 0} restTris={len(tri) if tri else 0} "
              f"enabled={_get(m,'omniphysics:deformableBodyEnabled')}")
        mat_rel = m.GetRelationship("material:binding:physics")
        targets = mat_rel.GetTargets() if mat_rel else []
        if not targets:
            problems.append(f"{m.GetPath()}: no physics material bound.")
        else:
            mat = stage.GetPrimAtPath(targets[0])
            if "OmniPhysicsSurfaceDeformableMaterialAPI" not in mat.GetAppliedSchemas():
                problems.append(f"{targets[0]}: not an OmniPhysicsSurfaceDeformableMaterialAPI material.")
            else:
                print(f"      material youngs={_get(mat,'omniphysics:youngsModulus')} "
                      f"poisson={_get(mat,'omniphysics:poissonsRatio')} "
                      f"density={_get(mat,'omniphysics:density')} "
                      f"thickness={_get(mat,'omniphysics:surfaceThickness')}")
        if m.HasAPI(UsdPhysics.RigidBodyAPI):
            problems.append(f"{m.GetPath()}: has RigidBodyAPI (incompatible with deformable).")

    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verify cloth physics schemas in a USD file.")
    parser.add_argument("usd_path")
    args = parser.parse_args(argv)

    stage = Usd.Stage.Open(args.usd_path)
    if stage is None:
        print(f"ERROR: cannot open {args.usd_path}")
        return 1

    print("=" * 72)
    print(f"  Schema verification: {args.usd_path}")
    print("=" * 72)
    problems = verify(stage)

    print("\n" + "-" * 72)
    if problems:
        print("RESULT: FAIL")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("RESULT: PASS — physics schemas applied and consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
