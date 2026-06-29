# Reference cloth pipelines (full source in this folder)

Two open-source Isaac Sim garment-manipulation frameworks were cloned into
`repos/` and are the closest known-good references. NOTE they use the higher-level
`isaacsim.core` / `omni.isaac.core` ParticleSystem + ClothPrim wrappers (not the
raw `omni.physx.scripts.particleUtils` calls our code uses), and they load garment
meshes that appear to be (near-)flat patterns which they **scale down** to metres.

## GarmentLab — `Env/Config/GarmentConfig.py` + `Env/Garment/Garment.py`
```
scale  = [0.005, 0.005, 0.005]      # garment USDs are ~200x too big -> metres
ori    = [0, 0, pi/2]
stretch_stiffness = 1e4
bend_stiffness    = 100
shear_stiffness   = 100
spring_damping    = 0.2
particle_contact_offset = 0.02
enable_ccd = True
global_self_collision_enabled = True
non_particle_collision_enabled = True
solver_position_iteration_count = 16
friction = 0.3
solver_type = "TGS"        # set on the physics scene (Env/env/BaseEnv.py)
physics_dt / rendering_dt = 1/60
# mass not set explicitly (uses ClothPrim default)
```

## DexGarmentLab — `Env_Config/Garment/Particle_Garment.py`
```
scale  = [0.0085, 0.0085, 0.0085]
stretch_stiffness = 1e12  (comment notes 1e6)
bend_stiffness    = 100
shear_stiffness   = 100
spring_damping    = 10
contact_offset           = 0.010    # "important parameter"
rest_offset              = 0.0075   # "important parameter"
particle_contact_offset  = 0.010    # "important parameter"
fluid_rest_offset = 0.0075 ; solid_rest_offset = 0.0075
adhesion = 0.1 ; adhesion_offset_scale = 0.0 ; cohesion = 0.0
particle_adhesion_scale = 0.5 ; particle_friction_scale = 0.5
friction = 25.0 ; gravity_scale = 1.0
particle_mass = 1e-2
global_self_collision_enabled = True
non_particle_collision_enabled = False
enable_ccd = True
solver_position_iteration_count = 16
solver_type = "TGS"        # Env_StandAlone/BaseEnv.py
```

## Open questions about the references (for the researcher to resolve)
- Are their garment meshes flat patterns or 3-D shells? (Asset dirs:
  `Assets/Garment/...`.) If flat, that is likely the crux — our asset is a 3-D
  worn shirt with panels 0.25 m apart.
- They run single-garment demos. Do these stiffness/offset values stay stable at
  many parallel envs and with a robot interacting?
- They use `isaacsim.core` ClothPrim/ParticleSystem, not raw particleUtils — does
  that wrapper set anything (mass model, particle spacing, max_velocity, contact
  offset auto-scaling) that we are missing by calling particleUtils directly?
- DexGarmentLab uses stretch 1e12 and it is stable — why does our 1e4 (let alone
  300–2000) diverge? Substeps? `particle_mass` semantics? offset scaling?
