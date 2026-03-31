# Cloth simulation in Isaac Sim 5.1.0 for garment manipulation

**Isaac Sim 5.1.0 is in a transitional state for cloth simulation.** The original PBD particle cloth API (`PhysxParticleClothAPI`) is officially deprecated but still functional, while its replacement — XPBD FEM surface deformables — ships as a beta feature. Isaac Lab's `DeformableObjectCfg` wraps only FEM volumetric soft bodies, not particle cloth or surface deformables, making true cloth simulation in Isaac Lab an unsupported workflow requiring custom engineering. The emerging Newton physics engine with its VBD solver offers the most physically principled cloth simulation path, but its Isaac Lab integration currently focuses on rigid-body solvers. For a GarmentLab-based t-shirt with 4096 parallel environments on an RTX A6000, the practical approach is to use the deprecated PBD particle cloth via direct Isaac Sim APIs (bypassing Isaac Lab's asset abstraction) with ≤2,000 vertices per garment, or to adopt Newton's standalone VBD solver for higher-fidelity cloth if Isaac Lab's environment management can be replaced.

---

## 1. Three simulation backends and when to use each

Isaac Sim 5.1.0 offers three distinct paths for simulating deformable cloth, each with fundamentally different physics models, APIs, and maturity levels.

### PhysX PBD particle cloth (deprecated, still functional)

PBD particle cloth places particles at mesh vertices and connects them with spring-damper constraints for stretching, bending, and shearing. The solver directly projects particle positions to satisfy constraints, then derives velocities — making it inherently stable with larger timesteps. The `PxParticleClothPreProcessor::partitionSprings()` call partitions springs to maximize GPU parallelism. Triangle connectivity is maintained for aerodynamic force computation (drag and lift). **This is GPU-only** — no CPU fallback exists for any particle simulation.

The key schemas are `PhysxSchema.PhysxParticleClothAPI`, `PhysxSchema.PhysxAutoParticleClothAPI` (auto-generates springs from mesh topology), `PhysxSchema.PhysxParticleSystem` (the simulation container), and `PhysxSchema.PhysxPBDMaterialAPI` (material properties). NVIDIA's migration documentation confirms: "The particle cloth feature is deprecated. A new deformable body feature, including USD schema, for simulating deformable surfaces will be released in an upcoming release."

**For garments, PBD particle cloth remains the most practical option in Isaac Sim 5.1.0.** It was designed specifically for thin surfaces like clothing, supports open (non-manifold) meshes, and provides aerodynamic effects. GarmentLab, the primary reference implementation for garment simulation in Isaac Sim, was built entirely on this API.

### XPBD FEM surface deformables (new beta replacement)

The new surface deformable system uses XPBD FEM corotational linear elasticity on triangle meshes rather than mass-spring systems. It requires enabling `physics.enableDeformableBeta = true` in preferences. The schemas are entirely different: `OmniPhysicsDeformableBodyAPI`, `OmniPhysicsSurfaceDeformableSimAPI`, `PhysxBaseDeformableBodyAPI`, and material APIs using Young's modulus and Poisson's ratio instead of spring stiffnesses. The simulation mesh doubles as the collision mesh and must consist of triangle faces only.

**Critical behavioral difference**: "Particle cloth behavior can only be approximated using surface deformables, as the two systems are based on different underlying physical models." Aerodynamics (lift, drag, wind) and inflatables are not supported in surface deformables. Stretching and shearing stiffness are controlled indirectly through Young's modulus, Poisson's ratio, and surface thickness, with `surfaceStretchStiffness` not yet implemented.

### Newton VBD (future best option)

Newton is an open-source, GPU-accelerated physics engine developed jointly by NVIDIA, Google DeepMind, and Disney Research under the Linux Foundation. Newton 1.0 GA shipped at GTC 2026 (March 2026). Its **Vertex Block Descent solver** — published at SIGGRAPH 2024 by Chen et al. — uses block coordinate descent on the variational form of implicit Euler integration. Unlike PBD, VBD is numerically convergent (actually converges to the implicit Euler solution), unconditionally stable, and resolution-independent in stiffness behavior.

VBD guarantees **penetration-free contact** via vertex-triangle and edge-edge detection, runs ~30 FPS on RTX 4090 for a Franka-cloth manipulation demo, and is **300× faster than GPU-IPC**. Newton plugs into Isaac Lab 3.0 / Isaac Sim 6.0 as a swappable physics backend, with experimental support in Isaac Lab 2.x via the `feature/newton` branch. However, the Isaac Lab Newton integration currently focuses primarily on the MuJoCo-Warp rigid-body solver — the full VBD cloth pipeline through Isaac Lab is not yet available.

### FEM volumetric soft bodies vs. particle cloth for garments

PhysX FEM deformable bodies use tetrahedral meshes with volumetric elements — designed for objects with meaningful thickness (sponges, stuffed animals, food). **FEM soft bodies are a poor fit for garments**, which are thin surfaces without meaningful volume. A t-shirt would need to be artificially thickened into a 3D volume, producing physically unrealistic behavior. Particle cloth (PBD) and surface deformables (XPBD FEM on triangles) are the correct choices for garments. Isaac Lab's `DeformableObjectCfg` only wraps volumetric FEM — it cannot be used for cloth.

| Feature | PBD Particle Cloth | Surface Deformable (XPBD) | Newton VBD | FEM Soft Body |
|---|---|---|---|---|
| **Status in 5.1.0** | Deprecated (functional) | Beta | Experimental | Stable |
| **Isaac Lab support** | None | None | Experimental | `DeformableObjectCfg` |
| **Mesh type** | Surface triangles | Surface triangles | Surface triangles | Tetrahedral volume |
| **Parameters** | Spring stiffness/damping | Young's modulus, Poisson | tri_ke, edge_ke | Young's modulus, Poisson |
| **Aerodynamics** | Yes (drag, lift) | No | No | No |
| **Self-collision** | Particle phase flags | Schema attribute | Penetration-free | Schema attribute |
| **Garment suitability** | **Excellent** | Good (beta) | **Excellent** | Poor |

---

## 2. Preparing a GarmentLab t-shirt mesh for cloth simulation

### Mesh requirements and manifold constraints

Particle cloth does **not** require a closed manifold — it is specifically designed for open surfaces like garments. Each vertex becomes a particle, and springs are generated along edges from the triangle connectivity. Surface deformables similarly support open surface meshes. The mesh **must be triangulated**: `PhysxAutoParticleClothAPI` generates stretch, bend, and shear springs from triangle topology, and the surface deformable system is explicitly "limited to triangle faces." Quad meshes must be converted before import (in Blender: `Ctrl+T` in Edit mode).

Additional mesh quality requirements: remove duplicate vertices and faces, weld disconnected components (Blender: Merge by Distance), ensure reasonably uniform edge lengths (very uneven triangles cause simulation artifacts), and scale to meters (Isaac Sim's default unit). UV coordinates should be preserved if visual materials are needed.

### Vertex count guidelines for 4096 parallel environments

With **4096 parallel environments on 48 GB VRAM**, GPU memory is the primary constraint. Each particle consumes approximately **64 bytes** (position Vec4, velocity Vec4, phase, constraint indices). For a t-shirt mesh:

- **2,000 vertices × 4,096 envs** = ~8.2M particles → ~525 MB particle state + constraint buffers
- **5,000 vertices × 4,096 envs** = ~20.5M particles → ~1.3 GB particle state + constraint buffers
- Add robot articulations (~2–4 GB), solver buffers, broadphase structures, and RL training buffers (PPO rollouts, policy networks: 2–8 GB)

**Recommended: ≤2,000–3,000 vertices per garment for 4096 environments.** ClothesNet's preprocessing pipeline uses quadric edge collapse decimation to downsample meshes with too many vertices. GarmentLab's source garments from ClothesNet typically have 2,000–10,000 vertices after cleaning; aggressive decimation to ~2,000 is advisable at scale.

### GarmentLab asset structure and required modifications

GarmentLab assets originate from ClothesNet (~4,400 garment models across 11 categories). They are OBJ meshes cleaned through a pipeline: duplicate face/vertex removal, quad→triangle conversion, quadric edge collapse decimation for oversized meshes, and weld modifiers for disconnected components. GarmentLab was built on **Isaac Sim 4.0.0** using the deprecated PBD particle cloth API.

For Isaac Sim 5.1.0, the conversion pipeline is:

1. Convert OBJ → USD using `asset_usd_converter.py` or programmatic USD stage manipulation
2. Verify triangulation and vertex count (decimate if >3,000 vertices for parallel RL)
3. Apply particle cloth schemas programmatically (the deprecated APIs still function)
4. Configure particle system offsets based on mesh vertex spacing: `solidRestOffset ≈ 0.5 × average_edge_length`

### Remeshing tools

- **Blender**: Decimate modifier (quadric edge collapse mode) for vertex reduction; Remesh modifier (voxel or smooth) for uniform retopology
- **Instant Meshes**: Open-source quadrilateral remesher — output requires triangulation afterward
- **Isaac Sim built-in**: For surface deformables, `PhysxAutoDeformableMeshSimplificationAPI` with `remeshingEnabled=True` and configurable `targetTriangleCount`; for particle cloth, no built-in remeshing exists but `PhysxAutoParticleClothAPI` auto-generates springs

---

## 3. USD schema configuration and recommended parameters

### Complete PBD particle cloth setup (standalone Isaac Sim script)

```python
from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf
from omni.physx.scripts import physicsUtils, particleUtils
import omni.usd

stage = omni.usd.get_context().get_stage()
default_prim_path = stage.GetDefaultPrim().GetPath()

# 1. Physics Scene with GPU dynamics enabled
scene_path = default_prim_path.AppendChild("physicsScene")
scene = UsdPhysics.Scene.Define(stage, scene_path)
physxAPI = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
physxAPI.CreateSolverTypeAttr("TGS")

# 2. Create Particle System
particle_system_path = default_prim_path.AppendChild("particleSystem")
# For a t-shirt with ~0.01m average vertex spacing:
radius = 0.005  # half vertex spacing
rest_offset = radius
contact_offset = rest_offset * 1.5

particleUtils.add_physx_particle_system(
    stage=stage,
    particle_system_path=particle_system_path,
    contact_offset=contact_offset,
    rest_offset=rest_offset,
    particle_contact_offset=contact_offset,
    solid_rest_offset=rest_offset,
    fluid_rest_offset=0.0,
    solver_position_iterations=16,
    simulation_owner=scene_path,
)

# 3. Create PBD Material
particle_material_path = default_prim_path.AppendChild("particleMaterial")
particleUtils.add_pbd_particle_material(
    stage, particle_material_path,
    drag=0.02,
    lift=0.02,
    friction=0.8,
    damping=0.3,
)
physicsUtils.add_physics_material_to_prim(
    stage, stage.GetPrimAtPath(particle_system_path), particle_material_path
)

# 4. Apply Particle Cloth to mesh (after loading OBJ as USD reference)
cloth_mesh_path = default_prim_path.AppendChild("tshirt")
particleUtils.add_physx_particle_cloth(
    stage=stage,
    path=cloth_mesh_path,
    dynamic_mesh_path=None,
    particle_system_path=particle_system_path,
    spring_stretch_stiffness=2000.0,
    spring_bend_stiffness=200.0,
    spring_shear_stiffness=100.0,
    spring_damping=0.3,
    self_collision=True,
    self_collision_filter=True,
)

# 5. Set mass (typical cotton t-shirt: 0.15–0.25 kg)
mass_api = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(cloth_mesh_path))
mass_api.GetMassAttr().Set(0.2)
```

### Particle system parameters reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `contactOffset` | float | auto | Contact threshold for particle–rigid body collisions |
| `restOffset` | float | 0.99 × particleContactOffset | Equilibrium distance for particle–rigid body |
| `particleContactOffset` | float | 0.05 m | Neighborhood radius for inter-particle interactions |
| `solidRestOffset` | float | 0.99 × particleContactOffset | Particle-particle rest distance |
| `solverPositionIterationCount` | int | 4 | Constraint solver iterations (use **16–32** for cloth) |
| `maxNeighborhood` | int | 96 | Maximum particles per neighborhood lookup |
| `globalSelfCollisionEnabled` | bool | false | Global self-collision toggle |
| `wind` | Vec3f | (0,0,0) | Wind force vector |
| `enableCCD` | bool | false | Continuous collision detection |

### Recommended cotton t-shirt parameters

These values synthesize community-tested configurations from NVIDIA forum threads and the GarmentLab codebase:

| Parameter | Recommended Range | Notes |
|---|---|---|
| **Stretch stiffness** | 1,500–3,000 N/m | Values >10,000 cause instability without very small dt |
| **Bend stiffness** | 100–500 N/m | Cotton bends easily; 200 is a good starting point |
| **Shear stiffness** | 50–200 N/m | Lower than stretch; 100 is typical |
| **Spring damping** | 0.2–0.5 | Higher = more stable, less oscillation |
| **Total mass** | 0.15–0.25 kg | Real cotton t-shirt weight |
| **Friction** | 0.6–1.0 | Cotton is relatively grippy |
| **Drag** | 0.01–0.05 | Aerodynamic drag coefficient |
| **Lift** | 0.01–0.05 | Aerodynamic lift coefficient |
| **Solver iterations** | 16–32 | Higher for stiffer materials |
| **Physics timestep** | 1/120 – 1/200 s | Smaller for stability with high stiffness |
| **Self-collision** | True | Essential for folding tasks |
| **Solid rest offset** | 0.5 × vertex_spacing | Prevents self-interpenetration |

### Mass property computation

PhysX computes per-particle mass using a priority chain: if `mass` is set on the `MassAPI`, then `perParticleMass = mass / numParticles`. If only `density` is set, then `totalMass = totalTriangleArea × clothThickness × density` where `clothThickness = 2 × solidRestOffset`. Priority order: default 1000 kg/m³ < material density (scene) < material density (system) < mass-component density < mass-component mass.

### Collision filtering

Cloth-cloth collision uses `particleGroup` assignments — particles in different groups collide; same-group particles use the `selfCollision` flag. Cloth-rigid body collision is enabled by default via `nonParticleCollisionEnabled` on the particle system. Cloth-ground plane works automatically if the ground has a collider. The `Create > Physics > Attachment` command creates fixed constraints between cloth and a collider prim and auto-filters collision at attachment points.

### New surface deformable setup (beta alternative)

```python
from omni.physx.scripts import deformableUtils

# Single-prim setup
deformableUtils.set_physics_surface_deformable_body(
    stage=stage,
    prim_path=cloth_mesh_path,
)

# Material with cloth-appropriate values
deformableUtils.add_deformable_material(
    stage=stage,
    path=material_path,
    youngs_modulus=5000.0,       # Much lower than default 50 MPa
    poissons_ratio=0.3,
    elasticity_damping=0.01,
    dynamic_friction=0.8,
    density=350.0,              # Cotton ~300–400 kg/m³
)
```

The full parameter mapping from deprecated to new schemas is documented in NVIDIA's migration guide. Key mappings: `springBendStiffness` → `surfaceBendStiffness`, `springStretchStiffness/shearStiffness` → controlled indirectly via `youngsModulus`, `poissonsRatio`, and `surfaceThickness`, `springDamping` → `elasticityDamping`, `friction` → `dynamicFriction`, `damping` → `linearDamping`.

---

## 4. Isaac Lab integration hits a fundamental wall for cloth

### DeformableObjectCfg supports FEM soft bodies only

**Isaac Lab's `DeformableObjectCfg` wraps PhysX FEM volumetric soft bodies — not particle cloth or surface deformables.** This is the single most important architectural constraint. The class is explicitly marked experimental and uses `physx.SoftBodyView` internally, which operates on tetrahedral meshes. Cloth simulation through Isaac Lab's API is tracked as an open feature request in GitHub Issue #2004, with the Isaac Lab team confirming: "cloth and fluid simulation isn't officially supported through the Isaac Lab API yet."

```python
from isaaclab.assets import DeformableObject, DeformableObjectCfg
import isaaclab.sim as sim_utils

# This creates an FEM soft body — NOT cloth
cfg = DeformableObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.MeshCuboidCfg(
        size=(0.2, 0.2, 0.2),
        deformable_props=sim_utils.schemas.DeformableBodyPropertiesCfg(
            simulation_hexahedral_resolution=10,
            self_collision=False,
            solver_position_iteration_count=16,
        ),
        physics_material=sim_utils.DeformableBodyMaterialCfg(
            poissons_ratio=0.4,
            youngs_modulus=1e5,
        ),
    ),
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
)
```

### State access and observation design

The `DeformableObjectData` class provides nodal state tensors shaped for RL:

```python
cloth = scene["object"]  # DeformableObject instance

# Nodal positions: (num_envs, max_sim_vertices_per_body, 3)
nodal_pos = cloth.data.nodal_pos_w

# Nodal velocities: (num_envs, max_sim_vertices_per_body, 3)
nodal_vel = cloth.data.nodal_vel_w

# Root position (mean of all nodes): (num_envs, 3)
root_pos = cloth.data.root_pos_w

# For RL observations, subsample keypoints to reduce dimensionality:
key_indices = [0, 50, 100, 200, 500]
obs_pos = nodal_pos[:, key_indices, :]  # (num_envs, 5, 3)
```

### Resetting deformable objects

Reset is performed by writing the default nodal state back to the simulation:

```python
def reset_cloth(deformable_obj, env_ids):
    default_state = deformable_obj.data.default_nodal_state_w[env_ids].clone()
    nodal_pos = default_state[..., :3]
    nodal_pos = deformable_obj.transform_nodal_pos(
        nodal_pos, pos=new_pos, quat=new_quat
    )
    default_state[..., :3] = nodal_pos
    deformable_obj.write_nodal_state_to_sim(default_state, env_ids)
```

**Known reset issues**: kinematic flags may persist after reset (GitHub Issue #1481), and the `DeformableObject.reset()` method is currently a no-op — users must manually write nodal state.

### Critical limitations for cloth manipulation in Isaac Lab

- **`replicate_physics=True` does not work with deformable bodies** — must be set to `False` in `InteractiveSceneCfg`, increasing startup time
- No particle cloth or surface deformable wrapping exists
- Lifting deformable objects with grippers is reported as problematic (Issue #2304)
- The direct workflow is recommended over manager-based for deformable manipulation due to irregular observation sizes and custom reset requirements
- The `DeformableObject` class uses `physx.SoftBodyView` internally, which has no equivalent for `ParticleClothView`

### Practical path: bypassing Isaac Lab's asset API

For true cloth simulation in parallel Isaac Lab environments, the viable approach is to use Isaac Sim's lower-level PhysX APIs directly within an Isaac Lab environment class. This means creating particle cloth prims via `particleUtils.add_physx_particle_cloth()` during scene construction, then accessing particle state through the PhysX tensor API (`SimulationView.create_particle_cloth_view()`) rather than through `DeformableObjectCfg`. This requires significant custom engineering — there is no reference implementation in Isaac Lab.

---

## 5. Gripper-cloth interaction requires careful tuning

### The penetration problem

Gripper-cloth interaction is one of the hardest aspects of garment simulation in Isaac Sim. Cloth particles are thin — gripper fingers can penetrate through them, trapping particles inside the finger geometry and generating spurious forces. NVIDIA forum threads document this extensively for Franka grippers and can be expected to apply equally to the Robotiq 2F-140.

**NVIDIA's recommended mitigations:**

- Set gripper finger collision approximation to **`convexHull`** instead of SDF (triangle mesh collision falls back to convexHull anyway and may cause GPU cooking failures)
- **Reduce controller gains**: stiffness ~50–100 on joints, ~100 on gripper, damping ~10 — fast movements cause interpenetration
- **Increase simulation frequency**: 200+ Hz (`TimeStepsPerSecond = 240`)
- **Lower contact/rest offsets** on both cloth particle system and gripper rigid bodies

### Two approaches for reliable cloth grasping

**Attach blocks (GarmentLab's original approach)**: Small rigid blocks are attached to gripper fingertips, and particle cloth is attached to these blocks. This creates a reliable constraint but produces unrealistic behavior — even minimal single-point contact can lift a garment. As of January 2024, particle cloth attachment directly to articulation links (like gripper fingers) was not supported and caused crashes. The attach-block workaround bypasses this limitation.

**Adhesion + friction (DexGarmentLab's improved approach)**: Three PBD material parameters enable physics-based grasping without attachment constraints: `adhesion` (particle-to-rigid body adhesive force), `friction` (particle-to-rigid coefficient), and `particleFrictionScale` / `particleAdhesionScale` (inter-particle scaling). This produces more realistic grasping with dexterous hands but requires careful tuning.

### Self-collision for folded garments

Self-collision is controlled via `PxParticlePhaseFlag::eParticlePhaseSelfCollide` and the `selfCollisionFilter` flag. For folded t-shirts, self-collision **must be enabled** to prevent layers from passing through each other. The `selfCollisionFilterDistance` parameter controls how close same-cloth particles can approach before repulsion activates — set this slightly larger than `solidRestOffset` to prevent jitter while maintaining fold integrity.

### Robotiq 2F-140 gripper configuration

The Robotiq 2F-140 is available in Isaac Sim at `/Isaac/Robots/Robotiq/2F-140/` and as a combined asset `ur10e_robotiq2f-140.usd`. In Isaac Lab, use `UR10_ROBOTIQ_CFG` from `isaaclab_assets.robots.universal_robots`. For cloth grasping, configure the finger pads with high-friction physics material (`static_friction=1.5`, `dynamic_friction=1.5`) and ensure the finger collision meshes use convex hull approximation.

---

## 6. Performance at 4096 environments demands aggressive optimization

### Memory budget on RTX A6000 (48 GB)

A rough VRAM allocation for 4096 environments with cloth and a robot arm:

| Component | Estimated VRAM |
|---|---|
| Robot articulations (7-DOF + gripper) × 4096 | 2–4 GB |
| Particle cloth (2,000 particles × 4096 × ~64 B) | 0.5–1.0 GB |
| PhysX solver buffers (broadphase, contacts, constraints) | 2–6 GB |
| RL training (PPO rollouts, policy, optimizer) | 2–8 GB |
| Miscellaneous (scene queries, fabric overhead) | 1–3 GB |
| **Total estimate** | **8–22 GB** |

**2,000 vertices per garment at 4096 environments should fit within 48 GB** with headroom for RL training. At 5,000 vertices, memory becomes tight. At 10,000 vertices, 4096 environments is likely infeasible.

### Simulation stability settings

- **Simulation dt**: Use `1/120 s` minimum; `1/200 s` for cloth folding with high stiffness. This translates to 2–4 substeps per 60 Hz render frame.
- **Solver position iterations**: **16–32** for cloth (default of 4 is far too low). Higher iteration counts improve stiffness fidelity but cost proportionally more compute.
- **`maxNeighborhood`**: Default 96 is usually sufficient. Lowering to 48–64 saves memory at the risk of missing interactions in dense configurations.
- **GPU buffer sizes**: Increase `gpu_max_rigid_patch_count` to `2**24` to avoid "Patch buffer overflow detected" errors. Also increase `gpu_found_lost_pairs_capacity` if overflow warnings appear.
- **Run headless**: Essential for maximum environment count — rendering overhead is significant.

### Critical `SimulationCfg` settings for Isaac Lab

```python
from isaaclab.sim import SimulationCfg, PhysxCfg

sim_cfg = SimulationCfg(
    dt=1.0 / 120.0,
    render_interval=2,  # decimation
    physics_material=sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    physx=PhysxCfg(
        gpu_max_rigid_patch_count=2**24,
        enable_scene_query_support=False,
    ),
)
```

### Known issues at scale

Fabric (physics fabric) can cause issues with particle system rendering and programmatic access when enabled for performance. Particles in different `PhysxParticleSystem` prims cannot collide — all interacting cloth must share one system. The `replicate_physics=False` requirement for deformable bodies increases scene creation time linearly with environment count.

---

## 7. Newton VBD provides a clear future migration path

### VBD solver setup for cloth

```python
import newton
from newton.solvers import SolverVBD, SolverFeatherstone

builder = newton.ModelBuilder()
builder.add_cloth_mesh(
    pos=(0.0, 1.0, 0.0),
    rot=(0.0, 0.0, 0.0, 1.0),
    vel=(0.0, 0.0, 0.0),
    vertices=mesh_vertices,   # From GarmentLab OBJ
    indices=mesh_triangles,   # Triangle indices
    mass=0.2,
    tri_ke=1e4,               # Stretching stiffness
    tri_ka=1e4,               # Area preservation stiffness
    edge_ke=10.0,             # Bending stiffness
)
builder.color()  # Required — generates vertex coloring for parallel solving
model = builder.finalize()

cloth_solver = SolverVBD(
    model,
    iterations=10,
    particle_enable_self_contact=True,
    particle_self_contact_radius=0.1,
    particle_self_contact_margin=0.1,
    integrate_with_external_rigid_solver=True,  # One-way coupling
)

state_0 = model.state()
state_1 = model.state()
control = model.control()
contacts = model.contacts()

for step in range(num_steps):
    model.collide(state_0, contacts)
    cloth_solver.step(state_0, state_1, control, contacts, dt=1.0/60.0/10)
    state_0, state_1 = state_1, state_0
```

### Parameter mapping from PhysX PBD to Newton VBD

There is **no direct equivalence** — the physical models are different. Approximate correspondences:

| Newton VBD | PhysX PBD | Typical Values |
|---|---|---|
| `tri_ke` (stretch stiffness) | `springStretchStiffness` | 1e4 (VBD) vs 2000 N/m (PBD) |
| `tri_ka` (area stiffness) | No equivalent | 1e4 |
| `edge_ke` (bend stiffness) | `springBendStiffness` | 10–1000 (VBD) vs 200 N/m (PBD) |
| `particle_self_contact_radius` | `solidRestOffset` | 0.1 (VBD) vs 0.005 (PBD) |
| `iterations` | `solverPositionIterationCount` | 10 (VBD) vs 16–32 (PBD) |

VBD uses physical units (Pa-like quantities) that are resolution-independent, while PBD spring stiffnesses are resolution-dependent — the same stiffness value on a coarser mesh produces stiffer behavior.

### Migration practicalities

Newton does not use PhysX USD schemas for cloth. It uses the `newton.ModelBuilder` Python API, importing geometry programmatically or via `builder.add_usd()`. The Franka-cloth demo uses multi-solver coupling: `SolverFeatherstone` (or `SolverMuJoCo`) for the robot arm and `SolverVBD` for the cloth, with one-way coupling where rigid body state acts as kinematic input for the cloth solver. Two-way coupling is planned for future releases.

The Isaac Lab Newton integration (via the `isaaclab_newton` extension) currently supports MuJoCo-Warp, XPBD, and Featherstone solvers. VBD cloth through Isaac Lab is anticipated in future updates.

---

## 8. Existing work and reference implementations

### GarmentLab (NeurIPS 2024)

GarmentLab is the primary reference for garment simulation in Isaac Sim. Built on **Isaac Sim 4.0.0**, it uses PBD particle cloth for large garments (t-shirts, dresses, trousers) and FEM for small elastic garments (gloves, socks). Assets come from ClothesNet (~4,400 garment models). The codebase uses attach blocks for gripper-garment interaction with Franka arms — small rigid blocks at fingertips create attachment constraints, bypassing PBD's tendency to allow gripper penetration. The successor **DexGarmentLab** replaced attach blocks with adhesion+friction-based grasping for dexterous hands.

### Simulator landscape for cloth manipulation

**Most cloth manipulation research does not use Isaac Sim.** The dominant simulator is **PyFlex/SoftGym** (based on NVIDIA Flex, predecessor to PhysX 5 particles):

- **ClothFunnels** (ICRA 2023): PyFlex, CUDA particle-based, Blender rendering with domain randomization
- **FlingBot** (CoRL 2021, Best System Paper): PyFlex, dual-UR5 arms, self-supervised fling primitive
- **DextAIRity** (RSS 2022): PyFlex, air-based manipulation with centrifugal pump

GarmentLab and DexGarmentLab are the notable exceptions using Isaac Sim.

### NVIDIA official resources

- Built-in demo: `Window > Simulation > Demo Scenes > Particle Cloth`
- PhysX particle system docs: NVIDIA's PhysX 5 documentation covers the C++ API extensively
- Deformable migration guide: Complete parameter mapping from deprecated to new schemas
- Isaac Lab deformable tutorial: `scripts/tutorials/01_assets/run_deformable_object.py` (FEM only)
- Isaac Lab teddy bear lift: `scripts/environments/state_machine/lift_teddy_bear.py` (FEM soft body with Franka)

### Key NVIDIA forum threads for troubleshooting

The developer forums contain critical practical guidance: cloth particle settings for folding tasks, gripper penetration workarounds for Isaac Sim 4.5, particle cloth attachment failures with articulated robots, and surface gripper incompatibility with particle objects.

---

## Conclusion

The path to cloth simulation for RL-based garment manipulation in Isaac Sim 5.1.0 is technically viable but architecturally unsupported by Isaac Lab's current abstractions. **The recommended short-term approach**: use the deprecated PBD particle cloth API via direct Isaac Sim scripting within an Isaac Lab direct-workflow environment, keeping the mesh under 2,000 vertices, using attach-block or adhesion-based grasping with the Robotiq 2F-140, and running at 120–200 Hz with 16+ solver iterations. The `DeformableObjectCfg` path is a dead end for cloth — it wraps only volumetric FEM.

**The strategic bet is Newton VBD.** It offers physically principled, penetration-free cloth simulation with unconditional stability, resolution-independent stiffness, and 300× speedup over comparable methods. The Franka-cloth demo already exists as a reference. As Isaac Lab's Newton integration matures beyond the current MuJoCo-Warp focus, migrating to VBD will provide both better cloth fidelity and cleaner API support. For a research project starting today, implementing the PBD particle cloth approach for immediate training while prototyping the Newton VBD pipeline for future migration is the highest-value strategy.