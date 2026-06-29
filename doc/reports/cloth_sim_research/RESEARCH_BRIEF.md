# Research brief: stable, flat, graspable PBD cloth t-shirt in Isaac Sim 5.1 / Isaac Lab

## Your task (for the researching AI)

We need an authoritative, sourced answer to: **what is the correct, robust way to
simulate a t-shirt as cloth in Isaac Sim 5.1 / Isaac Lab so that it (a) lies flat
on a conveyor belt, (b) drapes and folds like real fabric, (c) can be grasped by a
gripper, and (d) runs stably across hundreds of parallel RL environments?**

We have been unable to get this right after many iterations. Our current solution
only "works" by cheating (1.0 kg garment mass, self-collision disabled, a hand-rolled
geometric flatten of the mesh) and still never produces a clean flat lay. Please find
the real, canonical solution — with citations to NVIDIA docs, the PhysX particle/FEM
cloth documentation, Isaac Sim/Lab API docs and forum threads, and the two reference
repos included here. Tell us specifically what we are doing wrong and what the correct
recipe is. Where the answer is "use a different asset / authoring step", say so and
explain how.

Deliverable we want back:
1. Root-cause explanation of each failure mode below.
2. A concrete, parameter-by-parameter recipe (with units and the API calls) for
   stable garment cloth in Isaac Sim 5.1, including how mass, contact offsets, and
   particle spacing must relate to the mesh resolution.
3. Guidance on the asset: should we flatten the inflated 3-D mesh, and how (proper
   method, not naive axis-scaling)? Or load a flat garment pattern? Or use the
   inflated mesh as-is and let physics settle it (and how to keep that stable)?
4. Whether PBD particle cloth or the newer FEM/deformable-surface cloth is the right
   backend in 5.1, and how each is configured, plus parallel-env performance notes.
5. How to make self-collision stable (so a two-layer flat garment is graspable).

## System / stack
- Isaac Sim 5.1, Isaac Lab `ManagerBasedRLEnv` + `InteractiveScene`.
- PBD particle cloth created with raw `omni.physx.scripts.particleUtils`
  (`add_physx_particle_system`, `add_pbd_particle_material`, `add_physx_particle_cloth`),
  read/written with `omni.physics.tensors ... create_particle_cloth_view`
  (`get_positions/set_positions`, `get_velocities/set_velocities`).
- Physics dt 0.01 s (100 Hz), decimation 2, TGS solver (`physx.solver_type=1`),
  `enable_stabilization=True`, `replicate_physics=False`.
- GPU: RTX A6000 48 GB. Target: 512 parallel envs for RL (each env has its own
  cloth — no GPU physics replication).
- A known Isaac 5.1 quirk we hit: PBD particle cloth seems to be processed by the
  FEM/deformable-surface path; the particle system `maxVelocity` attribute does NOT
  clamp particle speed on that path; the relevant overflow is
  `gpu_max_deformable_surface_contacts`, not the rigid contact buffer. Please confirm
  / correct this understanding for 5.1.

## The asset (see data/mesh_probe.txt)
`tshirt_mod.usdc`, mesh `/root/World/mesh/Mesh`: 6705 verts, ~13260 tris, OPEN
surface (154 boundary edges). It is a **worn / inflated 3-D shirt**, NOT a flat
pattern: extent 0.78 (shoulders, X) × 0.28 (body depth, Y) × 0.69 (neck→waist, Z,
authored vertical). Front and back panels are genuinely ~0.25 m apart along Y
(per-cell spread along the thin Y axis ≈ the full Y extent). It is already in metres
(no scaling needed), unlike the reference assets which are scaled by ~0.005–0.0085.

## What we need it to do
1. Spawn lying FLAT on the conveyor belt (belt surface at z = 0.80 m), not standing
   vertically through the belt.
2. Settle into a thin, flat lay (~1–3 cm thick) that looks like a t-shirt on a table,
   and drape/fold naturally.
3. Be grasped by a Robotiq-style gripper and lifted (PBD adhesion or friction grasp).
4. Be reset to randomized flat poses each RL episode, stable across 512 envs.

## The failure modes we keep hitting (please root-cause each)

### F1 — The mesh is an inflated 3-D shell, so "lay flat" is not trivial
Placing the mesh centroid at belt height makes the 0.69 m-tall shirt stand vertical
and clip half-through the belt. Permuting axes so the thin (Y) axis is vertical still
leaves a 0.25–0.28 m-thick puffy double layer, not a flat sheet. PhysX bakes spring
rest-lengths from the mesh **at apply time**, so the puffy shape becomes the rest
shape and the shirt wants to stay puffy. How is a garment like this supposed to be
prepared? (Proper flattening / re-meshing / UV-based flat pattern / authoring in
Blender? Or a physics pre-settle that is then re-baked as rest — is re-baking rest
even possible for particle cloth in 5.1?)

### F2 — Geometric "flatten" hack
To force a flat rest shape we reorient the thin axis to vertical and *scale* it by
`flatten_scale` (e.g. 0.05 → ~1.4 cm). Projecting to a plane instead (scale 0) makes
the seam/side triangles zero-area slivers and the solver explodes; scaling avoids the
slivers but is clearly a hack and interacts badly with self-collision (F4). Is there
a sanctioned way to flatten/relax an inflated garment for particle cloth?

### F3 — Solver divergence / instability (the core blocker)
With "reasonable" cloth params (stretch 300–2000, light mass 0.2–0.5 kg, contact
offsets derived from mean edge length ~0.006–0.01) the cloth **gradually diverges**
while resting/settling (energy pumping; max|pos| grows over ~2 s until blow-up), even
with TGS + stabilization + 16–64 iterations. We could only stabilize it by using
(a) larger reference-style contact offsets (rest 0.0075 / contact 0.010 /
particle_contact 0.020) AND (b) an unrealistically heavy 1.0 kg garment mass.
With light mass it either diverges or slowly **sinks through** the belt collider over
an episode. DexGarmentLab uses stretch 1e12 and stays stable — why does our ≤2000
diverge? What is the correct relationship between particle mass, particle spacing,
contact/rest offset, stiffness, dt and iteration count for stability? Is 100 Hz / no
substeps the problem? Is our per-particle mass (1.0 kg / 6705 ≈ 1.5e-4 kg) wrong by
orders of magnitude vs. what the solver expects?

### F4 — Self-collision explodes
Enabling `global_self_collision` (which the references use) makes our compressed
two-layer mesh explode to ~1e11 immediately (the layers spawn inside each other's
self-collision contact radius). With self-collision OFF the cloth is stable but the
front/back layers interpenetrate, so realistic folding and a reliable two-sided grasp
are impossible. How do you run a flat two-layer garment with stable self-collision?
(self_collision_filter, particle_contact_offset vs layer spacing, rest_offset vs edge
length, etc.)

### F5 — Never a clean flat lay
Even in the stable config the shirt relaxes from a flat 0.05 m sheet into a ~0.1–0.2 m
bunched pile over ~1.5 s (footprint shrinks, thickness grows). We want it to stay flat
and drape, not bunch. Is this a stiffness/mass/offset issue, or a consequence of the
bad rest shape from F1/F2?

### F6 — Realism vs. stability trade-off
Our only stable recipe needs 1.0 kg mass (a real tee is ~0.15 kg) and no
self-collision. We suspect we are compensating for a deeper setup error (asset,
offsets, mass model, or wrong cloth backend). Please identify it.

## What is in this package
- `RESEARCH_BRIEF.md` — this file.
- `code/cloth_object.py` — our full PBD cloth integration (the heart of the problem:
  `PBDClothParams`, `apply_cloth_startup_event` with the reorient+flatten+place
  transform, `ClothObject` tensor wrapper, randomized reset).
- `code/shirt_place_env.py`, `code/shirt_place_scene_cfg.py`,
  `code/shirt_place_env_cfg.py` — the Isaac Lab env, scene (conveyor box collider,
  proxy), and PhysX settings.
- `code/run_shirt_validation.py` — standalone validation (drop/drape/grasp).
- `code/sandbox_stability_test.py` — minimal one-file stability harness that sweeps
  flatten/stretch/bend/mass/offsets/iters and logs per-step divergence onset (good
  for the researcher to propose params we can quickly test).
- `code/mesh_probe.py` — dumps the mesh extents / open-vs-closed / panel separation.
- `data/mesh_probe.txt` — the probe output (asset is an inflated 3-D shirt).
- `data/experiments_and_results.md` — every config tried and its measured outcome.
- `references/` — GarmentLab + DexGarmentLab garment loaders and a parameter summary;
  these are the closest known-good Isaac Sim cloth pipelines (full repos in `repos/`).

## Concrete questions to answer (prioritized)
1. **Asset**: For Isaac Sim particle cloth, must the garment mesh be a (near-)flat
   pattern? How do you correctly turn an inflated 3-D shirt mesh into a sim-ready flat
   garment (tooling + steps)? If keeping the 3-D mesh, how do you settle it flat and
   make THAT the rest configuration (can you re-bake rest lengths in 5.1)?
2. **Stability recipe**: the exact, sourced relationship between particle_contact_offset,
   rest_offset, contact_offset, mesh edge length / particle spacing, particle mass /
   density, stretch/bend/shear stiffness, dt, substeps, and solver iterations that
   yields stable inextensible cloth at 100 Hz. Why does DexGarmentLab's 1e12 stretch
   stay stable while our ≤2000 diverges?
3. **Mass model**: how is per-particle mass set for particle cloth in 5.1 (MassAPI on
   the mesh vs. `particle_mass` on ClothPrim vs. density)? What value is physically
   correct for a ~0.15 kg, 6705-vertex tee, and does the solver need it heavier?
4. **Self-collision**: correct settings to run a stable flat two-layer garment with
   self-collision ON (so it is graspable), including filtering of spring-connected
   neighbours and the offset/spacing constraints.
5. **Backend**: PBD particle cloth vs. FEM/deformable-surface cloth in 5.1 — which is
   recommended for graspable garments in Isaac Lab, how to configure it, and known
   pitfalls (incl. whether `maxVelocity` clamping works, and the
   `gpu_max_deformable_surface_contacts` buffer).
6. **Scale / Isaac Lab integration**: running per-env particle cloth with
   `replicate_physics=False` across 512 envs — performance, memory (contact buffers),
   and any required scene/cloner settings; and the recommended way to reset cloth to a
   randomized flat pose each episode via the tensor API.
7. **API correctness**: are we missing anything by using raw `particleUtils` instead of
   `isaacsim.core` `ParticleSystem`/`ClothPrim` (which the references use)? Does the
   higher-level wrapper configure something important (auto offsets, mass, particle
   spacing) that we are not?

Please return findings with explicit citations (URLs / doc sections / forum posts /
file+line in the reference repos) and a ready-to-test parameter set we can drop into
`PBDClothParams` and `sandbox_stability_test.py`.
