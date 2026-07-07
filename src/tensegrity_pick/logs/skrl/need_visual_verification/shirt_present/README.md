# Shirt Present — Visual Verification Guide

The selected `shirt_present` (pipeline Task 2) policy — **watch it regrasp and
stretch the hanging shirt in the simulator** to confirm the trained behaviour
matches the reported numbers before reusing it downstream (its stretched
terminal states seed Task 3, `shirt_distribute`).

Selected checkpoint: **UR5e-F140, run `2026-07-04_15-21-09_ppo_torch_seed43`,
`agent_96000`** — deterministic windowed present latch **0.927** across two
eval seeds (7 & 11, 96 episodes each), grasp 0.99, drop 0.00, final silhouette
coverage 0.68. Scripted baseline: 0.125. Full history:
[shirt_present_optimization_tracking.md](../../../../../../doc/reports/shirt_present_optimization_tracking.md).

## What was kept / removed

A **copy** — the full training run (all iterations) still lives under
`logs/skrl/shirt_present/` on the Alex checkout. Here, only the final run,
pruned to the two converged checkpoints:

```
ur5e_f140/<run>/
├── checkpoints/
│   ├── agent_96000.pt   ← eval-SELECTED policy — play THIS
│   └── best_agent.pt    ← highest TRAINING-reward checkpoint (reference only;
│                           training reward does not predict deterministic
│                           quality on this task — see the tracking report)
├── events.out.tfevents… ← TensorBoard curve (convergence sanity)
├── params/{agent,env}.yaml
└── run_status.json
```

## How to play the checkpoint

⚠️ **Render on a workstation, not on Alex.** Isaac Sim's RTX renderer segfaults
on the cluster GPU driver (`doc/alex_quickstart.md` §7a). Copy the checkpoint
over and play locally.

```bash
cd src/tensegrity_pick

conda run --no-capture-output -n env_isaaclab \
  python3 scripts/skrl/play.py \
  --task Template-Shirt-Present-UR5e-F140-Play-v0 \
  --num_envs 4 \
  --checkpoint logs/skrl/need_visual_verification/shirt_present/ur5e_f140/*/checkpoints/agent_96000.pt
```

`--num_envs 4` keeps the scene readable. The Play task ID and the checkpoint
must both be the UR5e-F140 variant.

## The task, in one picture

The shirt hangs from a **static anchor at the presentation pose** (a stand-in
for the retriever robot's grip — the passive tensegrity holder is scenery),
pinned at one random particle patch, so it drapes into a random hanging shape.
The UR5e must:

1. **reach down to the lowest hanging point** of the drape,
2. **close the gripper** there (a deterministic proximity attach — the cloth
   "sticks" to the fingertip when the tip is within ~10 cm and closing; it is
   NOT contact physics), then
3. **pull the grabbed point out** so the garment spans taut between the two
   grasps and faces the −Y inspection camera, and **hold it still**.

## What to look for in EVERY playback

**PASS looks like:**
- The arm drives its fingertip **down onto the lowest point** of the hanging
  shirt and the shirt snaps onto the fingertip (grasp) — the grabbed point
  then tracks the gripper.
- The arm **pulls the shirt open into a stretched, roughly camera-facing
  sheet** — visibly more spread than the raw hang — and **holds it still** for
  the rest of the episode (no swinging at the end).
- **Both** grasps hold throughout: the top stays pinned at the anchor, the
  hand keeps its point — the cloth spans between them.
- Full-length episodes: no early freeze, no teleport/explosion (NaN), no arm
  flinging to a joint limit.

**Watch for / known limitations (the ~7 % that miss 0.927):**
- **Reach miss:** occasionally the fingertip stalls a few cm short of the
  lowest point and never triggers the attach — that env just holds the raw
  hang. Expect roughly **3–4 of the 4 envs** to reach a clearly-presented pose.
- **Marginal coverage:** the RL success gate is silhouette coverage ≥ **0.50**;
  the FAPS heuristic study recommends **0.65** for a genuinely inspectable
  garment. Some passing episodes are only *moderately* opened — judge by eye
  whether the stretch looks camera-inspectable, and note it if many look
  bunched (candidate re-threshold, tracked in `doc/TODO.md`).
- **Overstretch:** the pull should stop at a taut sheet, not keep yanking the
  cloth thin/rippling. A brief over-pull that relaxes is fine; sustained
  over-tension that distorts the mesh is not.
- **Cosmetic:** the shirt now renders **FAPS-green** (was red) after the
  `cloth_object.py` refactor on `tendonbot-sim-rl` — expected, not a fault.

**The stillness gate is on the CLOTH centroid, not the gripper** — the raised
UR5e posture always has some residual end-effector sway; that is normal and
does not fail the task as long as the hanging garment itself is settled.
