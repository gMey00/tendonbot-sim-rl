# Cloth Simulation Assets

T-shirt meshes and helper scripts for PhysX cloth simulation in Isaac Sim 5.1.

The active garment is a **flat-laid ClothesNet shirt** built from
`TNSC_Tshirt_Ts1_0`. The earlier inflated `tshirt_mod` / short-sleeve shells and
the authoring pipeline built around them turned out not to settle or self-collide
correctly and have been retired to [`Deprecated/`](Deprecated/) — see
[Deprecated assets & scripts](#deprecated-assets--scripts) below.

## Directory Structure

```
Cloth/
├── tshirt_clothesnet.usd                   # ← active: PBD particle cloth (welded, ~11k verts)
├── tshirt_clothesnet_xpbd.usd              # ← active: XPBD/FEM surface deformable (same mesh)
├── TNSC_Tshirt_Ts1_0/                       # source ClothesNet mesh used to build the assets
├── TNSC_Tshirt*/                            # 16 more ClothesNet candidate shirts (source pool)
├── cloth_simulation_isaacsim_research.md    # earlier backend/parameter notes (historical)
├── Scripts/
│   ├── _usdenv.sh                          # Sets up plain-Python OpenUSD + PhysxSchema env
│   ├── check_cloth_readiness.py            # Validate a bare mesh (geometry)
│   ├── verify_cloth_schemas.py             # Validate authored physics schemas (static)
│   └── simulate_cloth_check.py             # Boot Isaac Sim, simulate, check behaviour (dynamic)
├── Deprecated/                              # retired assets + old authoring pipeline
└── README.md                               # This file
```

## Active assets

| Asset | Backend | Built from | Notes |
|-------|---------|-----------|-------|
| `tshirt_clothesnet.usd` | PBD particle cloth | `TNSC_Tshirt_Ts1_0.obj` | Welded to a manifold mesh (~11 k verts, ~21.7 k tris), mean edge ≈ 0.0098 m, flat-laid, metric, thin axis → Z |
| `tshirt_clothesnet_xpbd.usd` | XPBD surface deformable | same mesh | Surface-deformable schema + material pre-authored offline (no physics scene), flat `restShapePoints`; needs `/physics/enableDeformableBeta` |

Both are loaded by the shirt-place task via `ClothObject`, selectable with one
line (`SHIRT_CLOTH_BACKEND` in `shirt_place_scene_cfg.py`). The decisive PBD fix
was matching the **collision offsets to the mesh particle spacing**
(`particle_contact_offset = mean edge`, `solid_rest_offset = ½ edge`); offsets
larger than ½ edge make spring neighbours overlap and the solver injects energy.
The other `TNSC_Tshirt_*` directories are the candidate source pool — only
`Ts1_0` has been turned into a USD so far.

The cloth backend research (backend comparison and parameter research) lives in
[`cloth_simulation_isaacsim_research.md`](cloth_simulation_isaacsim_research.md)
here.

> **Note:** the welded ClothesNet assets were produced by standalone build
> scripts (vertex welding + offset-aware surface-deformable authoring) that are
> not committed in this folder. The `make_*` scripts that previously lived in
> `Scripts/` are the **old** pipeline and have been deprecated. If a fresh shirt
> needs authoring from the source pool, port the working build steps in before
> reusing this folder's tooling.

## How the scripts run

The remaining scripts are **validators** — they use plain OpenUSD + PhysxSchema
and (except the dynamic check) do not need a running Kit/Isaac-Sim app. Source
the environment helper once per shell:

```bash
source Scripts/_usdenv.sh     # exports $KPY and the USD/PhysX library paths
```

Then run with `"$KPY"`. The only script that needs the full Isaac Sim runtime is
`simulate_cloth_check.py` (it boots the PhysX solver) — run it through the Isaac
Lab launcher.

## Validation workflow

### 1. Validate the mesh (geometry)

```bash
source Scripts/_usdenv.sh
"$KPY" Scripts/check_cloth_readiness.py tshirt_clothesnet.usd
```

Checks triangulation, finite coordinates, index ranges, duplicate triangles,
non-manifold edges, vertex spacing, and VRAM estimates. Exit 0 = pass.

### 2. Verify the authored schemas (static, no solver)

```bash
"$KPY" Scripts/verify_cloth_schemas.py tshirt_clothesnet.usd
"$KPY" Scripts/verify_cloth_schemas.py tshirt_clothesnet_xpbd.usd
```

Confirms the cloth/deformable APIs are applied and internally consistent
(particle-system relationship + offset ordering, positive stiffness/mass, bound
material).

### 3. Verify behaviour in the solver (dynamic)

```bash
# Activate the Isaac Lab conda env first (env_isaaclab) and unset VIRTUAL_ENV.
/home/robot/Isaac/IsaacLab/isaaclab.sh -p Scripts/simulate_cloth_check.py \
    --usd tshirt_clothesnet_xpbd.usd --steps 200 --ground-z -1.0 --backend xpbd
```

Boots Isaac Sim headless, drops the cloth onto a ground plane, runs PhysX, and
asserts the cloth stays finite, falls under gravity, and does not explode.
Enables `/physics/enableDeformableBeta` automatically (needed for XPBD).

## Simulation Backends

| Backend | Status in 5.1 | Schemas | Notes |
|---------|---------------|---------|-------|
| **PBD Particle Cloth** | Functional (active default) | `PhysxParticleClothAPI` + `PhysxAutoParticleClothAPI` + `PhysxParticleSystem` + `PhysxPBDMaterialAPI` | Best for garments; spring stretch/bend/shear; grasp adhesion |
| **XPBD Surface Deformable** | Functional (beta) | `OmniPhysicsDeformableBodyAPI` + `OmniPhysicsSurfaceDeformableSimAPI` + `OmniPhysicsSurfaceDeformableMaterialAPI` | FEM; Young's modulus/Poisson/thickness; needs `enableDeformableBeta`; settles flattest |
| **PhysX FEM `PhysxDeformableSurfaceAPI`** | **Removed (crashes)** | — | The old surface-deformable schema is no longer cooked by the 5.1 solver. Do not use. |
| **Newton VBD** | Experimental | — | Not yet in Isaac Lab; best future option |

See [cloth_simulation_isaacsim_research.md](cloth_simulation_isaacsim_research.md)
for the detailed backend comparison and parameter research.

## Scripts Reference

### `check_cloth_readiness.py`
Pure-Python (pxr + numpy) validator for a **bare mesh**. Auto-discovers the first
`UsdGeom.Mesh` or accepts `--prim`. Reports vertex/triangle/edge stats and VRAM
estimates for parallel environments.

### `verify_cloth_schemas.py`
Static validator for the **authored physics** (complements
`check_cloth_readiness.py`). Confirms the scene + cloth/deformable schemas are
applied and internally consistent. No solver required.

### `simulate_cloth_check.py`
Dynamic validator. Boots Isaac Sim headless, adds an up-axis-aware ground plane,
steps PhysX, and checks the cloth falls, stays finite, and stays bounded.

## Deprecated assets & scripts

[`Deprecated/`](Deprecated/) holds the retired first-attempt assets and the
authoring pipeline built around them. None of it is used by the current task.

| Item | Why retired |
|------|-------------|
| `tshirt_mod.usdc` | Inflated/upright short-sleeve shell; would not lie flat or self-collide |
| `TNSC_T_Shirt_Short_Sleeve/` | Original GarmentLab shirt (Y-up, mislabelled cm); sub-tolerance scale, unstable |
| `tshirt_pbd.usd`, `tshirt_xpbd.usd` | Generated from `tshirt_mod`; superseded by the welded ClothesNet assets |
| `Scripts/make_pbd_cloth.py` | Old PBD authoring; offsets not tracked to spacing → divergence |
| `Scripts/make_xpbd_cloth.py` | Old XPBD authoring; runtime authoring on a referenced mesh failed to cook |
| `Scripts/make_cloth_checked.py` | Wrapper that drove the two `make_*` scripts |
| `Scripts/add_physics_material.py` | Old material binding; superseded by the `"physics"`-purpose binding done in `cloth_object.py` |

The top-level `textures` symlink (pointing into the short-sleeve directory) was
removed when that directory moved to `Deprecated/`.
