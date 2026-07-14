# Shirt Distribute — Visual Verification Guide

The selected `shirt_distribute` (pipeline Task 3) policy — **watch it carry the
hanging shirt to the commanded drum and release it inside** to confirm the
trained behaviour matches the reported numbers before reusing it downstream.

Selected checkpoint: **UR5e-F140, run
`2026-07-08_17-14-56_ppo_torch_seed1_filmU`, `agent_92000`** — the FiLM per-goal
PPO policy (mode-collapse fix). Deterministic per-bin success **0.910 / 0.899 /
0.832** (bins reusable / recyclable / trash), overall **0.882** over 1080
episodes (4 eval seeds × 270 ep). Scripted DLS-IK baseline: 0.88 overall.
Full history: [shirt_distribute_optimization_tracking.md](../../../../../../doc/reports/tracking/shirt_distribute_optimization_tracking.md)
(Phases 2 / 2b / 2c) and the
[research report](../../../../../../doc/reports/tmp/RESEARCH_REPORT_goal_conditioned_mode_collapse).

## Why this policy needed a visual check

Task 3 is **goal-conditioned**: each episode a target drum is sampled uniformly
and only its position enters the observation, so the policy cannot ignore the
goal. Early training collapsed to a **subset of drums** (per-seed 2-of-3-bin
specialisation); a literature-review-driven fix (per-goal value/advantage
normalization + a per-goal multi-head critic + a goal one-hot, then FiLM
goal-gating + per-goal actor heads) broke the collapse and produced the balanced
policy staged here. The eval numbers are deterministic (mean-action); this
playback confirms the behaviour *looks* right in the simulator.

## What was kept / removed

A **copy** — the full training run (all 48 checkpoints) still lives under
`logs/skrl/shirt_distribute/<run>_seed1_filmU/` on the Alex checkout. Here, only
the final run pruned to two checkpoints:

```
ur5e_f140/<run>/
├── checkpoints/
│   ├── agent_92000.pt   ← eval-SELECTED policy — play THIS
│   └── best_agent.pt    ← highest TRAINING-reward checkpoint (reference only;
│                           training reward does not predict deterministic
│                           quality on this task — the eval-selected 92000
│                           beats 96000/best, see the tracking report)
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
  --task Template-Shirt-Distribute-UR5e-F140-PerGoal-FiLM-Play-v0 \
  --num_envs 6 \
  --checkpoint logs/skrl/need_visual_verification/shirt_distribute/ur5e_f140/*/checkpoints/agent_92000.pt
```

`--num_envs 6` shows all three drums being commanded across envs while staying
readable. The Play task ID **must be the `PerGoal-FiLM` variant** — it selects
the FiLM per-goal policy architecture and the per-goal Runner that `play.py`
loads the checkpoint into (a stock task ID would build the wrong network and
fail to load). Goals are sampled uniformly per episode.

## The task, in one picture

The shirt starts **already hanging from the robot's own closed gripper** at a
sampled end-of-Task-2 holding pose (the cached pose bank; a stand-in for the
inspected garment handed over from Task 2). The commanded drum — **reusable
(left, `(0.15, 1.0)`), recyclable (right, `(1.35, 1.0)`), or trash (behind,
`(0.75, 1.6)`)** — enters the observation as a relative vector plus a one-hot.
The UR5e must:

1. **carry** the hanging shirt over the *commanded* drum's opening,
2. **clear** the whole garment above the rim, then
3. **open the gripper** to release it so the cloth falls **inside that drum**
   (release-required success: ≥ 15 % of cloth particles in the drum interior).

## What to look for in EVERY playback

**PASS looks like:**
- The arm carries the hanging shirt to the **commanded** drum (not the nearest
  one) — check the goal per env: the shirt should travel to a *different* drum
  in different envs.
- It positions the garment **over the drum opening**, lifts the lowest point
  clear of the rim, then **opens the gripper** and the shirt **drops into the
  drum**.
- After release it does **not** dive into the belt or fling to a joint limit —
  the arm settles calm-and-high (post-drop settle reward).
- Full-length episodes: no early freeze, no teleport/explosion (NaN).

**Watch for / known limitations (the ~12 % that miss 0.882):**
- **Trash drum (bin 2) is the weak one — ~0.83.** It is mounted **behind** the
  pedestal-mounted arm, the hardest to carry to and release into; the scripted
  IK baseline itself only reaches 0.88 overall. Expect commands to the *behind*
  drum to miss more often than the left/right drums (which run ~0.90). This is a
  **single-drum reachability ceiling**, not the old mode collapse (all three
  drums are clearly attempted). Note per drum whether the misses are
  reach-limited (arm can't get the garment over the rim) vs release-timing.
- **Wrong-bin / early release:** occasionally the gripper opens short of the
  commanded drum and the shirt lands on the floor or a neighbouring drum — a
  miss, not a hang.
- **Cosmetic:** the shirt renders **FAPS-green** after the `cloth_object.py`
  refactor — expected, not a fault.

**Judge by the commanded drum**, not the nearest: the whole point of the
goal-conditioning is that "nearest ≠ correct", so a policy that always drops in
the closest drum would be *wrong* even if it looks tidy. Confirm the shirt
tracks the commanded drum across envs.

See [findings.md](findings.md) for post-playback observations (to be filled in
after the first workstation run).
