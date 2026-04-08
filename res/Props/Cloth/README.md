# Cloth Simulation Assets

T-shirt meshes and helper scripts for PhysX cloth simulation in Isaac Sim 5.1.

## Directory Structure

```
Cloth/
├── tshirt_mod.usdc                         # Simplified T-shirt mesh (6,705 vertices)
├── TNSC_T_Shirt_Short_Sleeve/             # GarmentLab T-shirt with textures (6,749 vertices)
│   ├── TNSC_T_Shirt_Short_Sleeve_obj.usd
│   ├── border_obj.usd
│   └── textures/
├── cloth_simulation_isaacsim_research.md   # Backend comparison & parameter research
├── Scripts/
│   ├── check_cloth_readiness.py           # Standalone mesh validator (no Isaac Sim needed)
│   ├── make_pbd_cloth.py                  # Apply PBD particle cloth to USD mesh
│   ├── make_xpbd_cloth.py                 # Apply XPBD surface deformable to USD mesh
│   ├── make_cloth_checked.py              # Script Editor helper: preflight + apply
│   └── add_physics_material.py            # Bind physics material to cloth mesh
└── README.md                              # This file
```

## Meshes

| Mesh | Vertices | Triangles | Avg Edge | VRAM (8192 envs) |
|------|----------|-----------|----------|------------------|
| `tshirt_mod.usdc` | 6,705 | 13,260 | 0.0125 m | ~9.8 GB |
| `TNSC_T_Shirt_Short_Sleeve_obj.usd` | 6,749 | 13,260 | 0.0125 m | ~9.9 GB |

Both meshes pass all cloth-readiness checks (triangulated, no degenerate faces, no nonmanifold edges).

## Workflow

### 1. Validate mesh (no Isaac Sim required)

```bash
conda activate env_isaaclab
python Scripts/check_cloth_readiness.py tshirt_mod.usdc
python Scripts/check_cloth_readiness.py TNSC_T_Shirt_Short_Sleeve/TNSC_T_Shirt_Short_Sleeve_obj.usd
```

Checks: triangulation, finite coordinates, index ranges, duplicate triangles,
nonmanifold edges, vertex spacing, and VRAM estimates.
Exit code 0 = pass, 1 = fail, 2 = usage error.

### 2. Apply PBD particle cloth (requires Isaac Sim)

```bash
~/.local/share/ov/pkg/isaac-sim-*/python.sh Scripts/make_pbd_cloth.py \
    --input  tshirt_mod.usdc \
    --output tshirt_pbd.usd \
    --prim   /root/World/mesh/Mesh
```

Creates a physics scene, particle system, PBD material (cotton T-shirt defaults:
stretch=2000, bend=200, shear=100), and applies cloth API to the mesh.

### 3. Apply XPBD surface deformable (requires Isaac Sim + beta flag)

```bash
~/.local/share/ov/pkg/isaac-sim-*/python.sh Scripts/make_xpbd_cloth.py \
    --input  tshirt_mod.usdc \
    --output tshirt_xpbd.usd \
    --prim   /root/World/mesh/Mesh
```

Requires `physics.enableDeformableBeta = true` in Isaac Sim settings.
Uses Young's modulus / Poisson's ratio instead of spring stiffnesses.

### 4. Verify in Isaac Sim

Open the produced USD, press Play, and confirm:
- Cloth drapes under gravity without flying away
- No interpenetration with ground plane
- Self-collision works (no folding through itself)

## Simulation Backends

| Backend | Status | API | Notes |
|---------|--------|-----|-------|
| **PBD Particle Cloth** | Deprecated (functional) | `PhysxParticleClothAPI` | Best for garments in Isaac Sim 5.1 |
| **XPBD Surface Deformable** | Beta | `OmniPhysicsDeformableBodyAPI` | No aerodynamics yet |
| **Newton VBD** | Experimental | Not yet in Isaac Lab | Best future option |

See [cloth_simulation_isaacsim_research.md](cloth_simulation_isaacsim_research.md) for detailed backend comparison.

## Scripts Reference

### `check_cloth_readiness.py`
Pure Python validator using only `pxr` (OpenUSD) and `numpy`. Auto-discovers
the first `UsdGeom.Mesh` prim or accepts `--prim` path. Reports vertex count,
triangle count, edge statistics, and VRAM estimates for parallel environments.

### `make_pbd_cloth.py`
Applies the full PhysX PBD particle cloth pipeline: physics scene, particle system,
PBD material, particle cloth API, and mass/self-collision/adhesion properties.
Works standalone or in Script Editor.

### `make_xpbd_cloth.py`
Applies XPBD surface deformable pipeline: physics scene with deformable beta,
surface deformable body, and deformable material. Works standalone or in Script Editor.

### `make_cloth_checked.py`
Script Editor convenience script that runs `check_cloth_readiness` preflight checks
and then applies a surface deformable, all in one step.

### `add_physics_material.py`
Binds a `PhysicsMaterial` (friction, restitution) to a cloth mesh prim.
Run in the Isaac Sim Script Editor.
