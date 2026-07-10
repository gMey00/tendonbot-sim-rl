# Shirt Distribute — Visual Verification Findings

Selected checkpoint (see `README.md`): `ur5e_f140`, run
`2026-07-08_17-14-56_ppo_torch_seed1_filmU`, `agent_92000.pt`.

> **Status: live workstation playback PENDING.** The numbers below are from
> deterministic (mean-action) evaluation on Alex (1080 episodes); the
> observational notes are *expectations* to confirm/refute at the workstation.
> Fill the "Live playback" section after the first render.

## What the metrics say (deterministic eval, 1080 ep, uniform goals)

| bin | drum (position) | success |
|-----|-----------------|---------|
| 0 | reusable — left `(0.15, 1.0)` | **0.910** |
| 1 | recyclable — right `(1.35, 1.0)` | **0.899** |
| 2 | trash — behind `(0.75, 1.6)` | **0.832** |
| — | overall | **0.882** |

- **Mode collapse is solved** — all three drums are learned and balanced (the
  Phase-1 policy hard-zeroed one drum at 0.00–0.09). bins 0 and 1 clear the
  ≥ 0.85 §2 bar robustly across eval seeds; all three clear it on eval seed 8.
- **bin 2 (trash, behind the arm) is the weak drum at ~0.83** — a single-drum
  reachability ceiling, eval-seed-sensitive (0.76–0.89). The scripted IK
  baseline is 0.88 overall, so the policy sits at the task ceiling.
- release_rate ≈ 1.0, mean |action| ≈ 0.28 (clean deterministic mean — not a
  saturated-noise controller; the Phase-1 relative-action trap is avoided).

## Working well (expected — confirm at playback)

- The shirt should travel to the **commanded** drum (different drum per env),
  not the nearest — the goal-conditioning is the whole point.
- Carry → clear-over-rim → open-gripper → drop-inside, then settle calm-and-high
  (no post-drop dive into the belt).

## Known limitations to look for

- **Trash-drum (behind) misses** more than left/right — reach-limited: judge
  whether the arm simply can't get the garment over the rim, vs release timing.
- **Wrong-bin / early release** occasionally: gripper opens short of the drum →
  floor or neighbouring drum.

## Live playback (fill after workstation render)

_TODO: observations from `play.py … PerGoal-FiLM-Play-v0`. Per drum, note
pass/fail feel, whether bin-2 misses are reach-limited, and any cloth/robot
clipping or release-timing artifacts._

## Candidate next steps (task-side, to lift bin 2 over 0.85)

- TossingBot-style release-velocity conditioning for the far/behind drum.
- A bin-2-specific approach/release reward (the behind drum needs a different
  carry trajectory over the base).
- Wider holding-pose coverage for reaching behind the pedestal.
