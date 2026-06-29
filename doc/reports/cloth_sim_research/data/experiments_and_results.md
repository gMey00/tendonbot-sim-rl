# Experiment log — what was tried, and the measured outcome

Environment: Isaac Sim 5.1, Isaac Lab (ManagerBasedRLEnv), PBD particle cloth via
`omni.physx.scripts.particleUtils.add_physx_particle_system` /
`add_physx_particle_cloth`, GPU tensor access via
`omni.physics.tensors ... create_particle_cloth_view`. Single GPU RTX A6000 48 GB.
Physics dt = 0.01 s (100 Hz), decimation 2, TGS solver (`physx.solver_type = 1`),
`enable_stabilization = True`, `replicate_physics = False` (required for cloth).
Target deployment: 512 parallel envs for RL training (cloth must be per-env).

Diagnostics write metrics directly to a file because Kit redirects stdout after
AppLauncher launches. Metrics reported as: env-0 cloth "thickness" = (max-min) of
particle Z; minZ relative to belt (belt surface = 0.80 m); horizontal footprint =
(max-min) of particle X,Y; "diverged" = any |position| > 100.

## A. Standalone stability sandbox (code/sandbox_stability_test.py)
Minimal scene = ground + a 0.4 m-thick static box collider with its top at the
belt height (0.80 m) + the shirt. Reorient+flatten applied at spawn (see
cloth_object.apply_cloth_startup_event), then settle and log per-step max|pos|.

| flatten_scale | stretch | bend | shear | spring_damp | mass(kg) | self-coll | rest/contact/pcontact offset | result |
|---|---|---|---|---|---|---|---|---|
| 1.00 (no compress) | 300 | 0.8 | 10 | 0.4 | 0.2 | off | 0.0064/0.0096/0.0096 (from mean edge) | settles ~2 s then DIVERGES at step ~240 (gradual energy gain: max\|pos\| 2→18→145) |
| 0.10 | 300 | 0.8 | 10 | 2.0 | 0.2 | off | computed | still diverging (thickness→10 m, footprint→73 m by step 250) |
| 0.10 | 300 | 0.8 | 10 | 0.4 | **1.0** | off | **0.0075/0.010/0.020** | **STABLE** (max\|pos\| ~1.0), rests on belt (minZ +0.0001), footprint 0.76×0.61, but thickness 0.27 m (crumples/folds, never a clean flat lay) |
| 0.05 | 1000 | 60 | 30 | 0.5 | 0.5 | off | 0.0075/0.010/0.020 | stable but SINKS to minZ −0.31 (penetrates the box collider) and 0.53 m thick |
| any | — | — | — | — | — | **ON** | — | EXPLODES to ~1e11 (the compressed two layers spawn inside each other's self-collision contact radius and violently repel) |

Key findings from the sandbox:
- Divergence with the original params is GRADUAL (energy pumping in a resting
  cloth on a collider), not an instant blow-up — independent of the flatten.
- The two things that stop the divergence are (1) reference-style **larger
  contact offsets** (rest 0.0075 / contact 0.010 / particle_contact 0.020) and
  (2) **heavier mass** (1.0 kg). With light mass (0.2–0.5 kg) the cloth either
  diverges or slowly sinks THROUGH the collider over an episode.
- Self-collision ON is unusable with our geometric compression (instant explosion).

## B. Real Isaac-Lab env (Template-Tensegrity-Shirt-Place-Play-v0)
Builds the actual RL scene (robot + conveyor box collider + drum + cloth), runs
the in-`__init__` pre-settle (320 steps on the upstream belt), then env.reset()
and a 150-step zero-action rollout.

Config: stretch 800, bend 20, shear 20, spring_damp 0.5, mass 1.0,
flatten_scale 0.05, offsets 0.0075/0.010/0.020, iters 20, self-coll off.

| stage | thickness | minZ rel belt | footprint XY | note |
|---|---|---|---|---|
| after reset | 0.046 m | +0.020 | 0.50×0.49 | flat, resting on belt — GOOD |
| after 150 steps | 0.19 m | +0.007 | 0.31×0.40 | stays on belt (no sink) but relaxes/bunches: footprint shrinks 0.50→0.31 and thickness grows 0.05→0.19. NOT a clean flat lay. |

No PhysX "deformable surface contact buffer overflow" warnings with self-coll off.

## C. The drop/drape validation teleport (separate concern, mostly cosmetic)
Trace of the actual validation run proved the teleport coordinates were always
correct (cloth centroid teleports exactly above the cube/sphere). The visible
problem there was a follow-camera bug (tracker read a stale centroid) — fixed.
Not a physics problem; ignore for the cloth-sim research.

## Current "best" config (still unsatisfactory)
stretch 800 / bend 20 / shear 20 / spring_damp 0.5 / material drag 0.02 lift 0.02
friction 0.8 damping 0.3 / adhesion 0.2 adhesion_offset_scale 2.0 /
mass 1.0 kg / flatten_scale 0.05 / rest 0.0075 contact 0.010 particle_contact 0.020 /
solver_position_iterations 20 / max_neighborhood 96 / self_collision OFF / CCD off /
gpu_max_deformable_surface_contacts 2**21.

It is STABLE and rests on the belt, but: (1) requires an unrealistic 1.0 kg mass,
(2) requires self-collision OFF (so folds/layers interpenetrate; grasping a 2-layer
flat cloth is unreliable), (3) never forms a clean flat lay — it relaxes into a
~0.1–0.2 m bunched pile, (4) relies on a hand-rolled geometric "flatten" (scale the
inflated mesh's thin axis) which is a hack.
