# Shirt Present — Visual Verification Guide (HEM-TO-HEM baseline)

This is the **hem-to-hem** presentation policy — the current `shirt_present`
env (merged into `project/tendonbot-sim-rl` 2026-07-10 as the new hem↔hem
baseline; the earlier naive lowest-point env/policy `agent_96000` is superseded,
its checkpoint kept in `logs/skrl/theses_logs/…2026-07-04…/`).  Play it with
**the current env** — its observation space is hem-to-hem (`pull_target_rel`
added, `lowest_point_rel` removed), so it is NOT interchangeable with the old
`agent_96000` checkpoint.

Geometry: the retriever holds the shirt at an arbitrary upper point so it hangs
hem-down, and the learning arm grasps the **accessible (lowest) hem corner** and
pulls it up to the holder's height, **horizontally** along the camera-plane x —
gravity drapes the body below the taut chord.  This is a FIRST STEP toward a
better hem↔hem policy (present 0.380, below the naive 0.927 — see limitations).

## Result (deterministic, 2 eval seeds × 96 episodes)

Selected checkpoint: **UR5e-F140, run `2026-07-08_16-03-46_ppo_torch_seed43`,
`agent_104000`** (a resume from the 8h-wall-cut run 4):

| metric | value |
|---|---|
| present latch (windowed) @ coverage gate 0.60 | **0.380** |
| grasp rate | 0.73 |
| final silhouette coverage | 0.637 |
| stretch ratio (flat-rest) | 0.83 |
| drop rate | 0.04 |

**This is WORKING but BELOW the production policy.**  The first pass's naive
lowest-point policy (`agent_96000`, on `project/tendonbot-sim-rl`) reaches
present **0.927 @ coverage 0.68** and remains the recommended production policy.
The hem-to-hem redesign does not beat it in-scene: the study's *winning* hem↔hem
grasp needs a high second hem corner RL cannot learn to grasp, and the
*learnable* accessible-corner version's coverage (0.64) is below the naive
stretch's (0.68).  Full write-up:
[tracking report Phase 2](../../../../../../doc/reports/shirt_present_optimization_tracking.md).

## How to play

```bash
# On the project/shirt-present worktree, in the env_isaaclab conda env,
# on a WORKSTATION (Alex cannot render — RTX segfaults on the cluster driver).
cd src/tensegrity_pick
PYTHONPATH="$PWD/source/tensegrity_pick:$PYTHONPATH" \
python scripts/skrl/play.py --task Template-Shirt-Present-UR5e-F140-Play-v0 \
    --num_envs 16 \
    --checkpoint logs/skrl/need_visual_verification/shirt_present/ur5e_f140/2026-07-08_16-03-46_ppo_torch_seed43/checkpoints/agent_104000.pt
```

(The `PYTHONPATH` pin makes imports resolve to this worktree, not any shared
editable install — see the tracking report's isolation note.)

## What to watch — PASS criteria

You should SEE, on most envs:

1. **Open-gripper approach.**  The arm approaches the lowest hem corner with the
   gripper OPEN, then closes and grasps AT the corner (the first pass's "closes
   too early" behaviour is gone).
2. **Grasp of a HEM CORNER** (a seam-marked bottom corner of the shirt), not the
   mid-garment lowest point.
3. **Horizontal pull to the holder's height:** the grasped corner is lifted to
   the holder anchor's height and pulled sideways so the two grips form a
   roughly HORIZONTAL taut chord; the body drapes below it toward the camera.
4. **A held, still presentation** for ~1 s (the windowed latch) on the episodes
   that succeed (~38 %).

## Known limitations (expected — this is the documented shortfall)

- **~62 % of episodes do NOT fully present.**  The garment is often
  under-spread (coverage < 0.60) because a random-point hang bunches more than
  the naive lowest-point stretch; the horizontal pull only partly recovers it.
- **Occasional drops** mid-hold (~4 %).
- **The passive holder arm** is posed above the anchor pointing down so it looks
  like it grips the anchor patch (finding #3 fix); its fingertip alignment is
  cosmetic — nudge `_HOLDER_REST_DROP` in the scene cfg if it looks off.
- **Camera occlusion / cloth-robot brush** may still occur occasionally
  (findings #2/#5): the coverage metric has no camera sensor and the UR5e has
  self-collision disabled, so these are best-effort/cosmetic.

## Findings from the FIRST pass (what motivated this redesign)

See [findings.md](findings.md) — Georg's notes from inspecting the naive
`agent_96000`.  All six were addressed; note in particular that finding #1's
"gripper closes too early" was found to be **cosmetic** (the deterministic
attach only fires within 10 cm of the target), and a reward penalty for it
*broke* grasp learning entirely — so it was removed (tracking report Phase 2).
