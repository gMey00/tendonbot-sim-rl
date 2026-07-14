# Reach — Visual Verification Guide

Curated checkpoints from the final reach grid (seed 42, 100k timesteps,
2026-07-02) for **watching each policy reach in the simulator**. Use this to
confirm the trained behaviour matches the reported numbers before running the
evaluation grid or reusing these controllers on downstream tasks.

## What was kept / removed

This directory is a **copy** — the full logs (all iterations, all
checkpoints) still live under `logs/skrl/reach/`. Here, per variant, only the
**final grid run** is kept, pruned to the two converged checkpoints:

```
<variant>/<run>/
├── checkpoints/
│   ├── best_agent.pt      ← highest-reward checkpoint — play THIS
│   └── agent_100000.pt    ← the policy exactly as trained at the end
├── events.out.tfevents…   ← TensorBoard curve (convergence sanity)
├── params/{agent,env}.yaml
└── run_status.json
```

Superseded runs (pre-fix iterations) and the intermediate checkpoints
(`agent_10000 … agent_90000`) were removed — they are not the policies you are
verifying. `ur10_frankenstein_osc` keeps **three** runs (seeds 42, 7, 123); see
Priority 1.

## How to play a checkpoint

⚠️ **Render on a workstation, not on Alex.** Isaac Sim's RTX renderer segfaults
on the cluster GPU driver (see `doc/alex_quickstart.md` §7a). Copy the
checkpoint over and play locally.

```bash
cd src/tensegrity_pick

conda run --no-capture-output -n env_isaaclab \
  python3 scripts/skrl/play.py \
  --task Template-Reach-UR5e-F140-OSC-Play-v0 \
  --num_envs 4 \
  --checkpoint logs/skrl/visual_verification/reach/ur5e_f140_osc/*/checkpoints/best_agent.pt
```

Swap the `--task` (Play ID) and the `--checkpoint` path together — they must
refer to the same variant. `--num_envs 4` keeps the scene readable.

## What to look for in EVERY playback

The target is a **frame marker** (one new pose per 6 s / 180-step episode). A
healthy policy drives the EE body (`robotiq_base_link`) to the marker's
**position** and holds it there until the episode resets.

**PASS looks like:**
- EE settles within a few cm of the marker and stays put; smooth motion.
- Full-length episodes — no early freeze, no teleport/explosion (NaN), no
  arm flinging off to a joint limit.
- On the Frankenstein arms, the 2-DOF tensegrity wrist looks **softly
  compliant**, not buzzing or oscillating.

**Do NOT flag these — they are correct by design:**
- The gripper **not** pointing exactly straight down, or spinning freely in
  yaw. Orientation is **loose by design**: a fixed orientation is geometrically
  unreachable across the target box for the 6-DOF arms, so reach is a
  position-only objective. (The 7-DOF Kinovas happen to track orientation
  better — also expected, not a bug.)
- The arm sitting still when it is already on target, or between episodes.

Position is the metric. "Reaches the marker and holds" = pass.

---

## Priority 1 — the OSC fixes (watch these first)

These six were **divergent or collapsed before the fix** (tensegrity wrist kept
on PD + excluded from OSC, plus partial inertial decoupling). Highest chance of
revealing a subtle problem — verify carefully.

| Variant (dir) | Play task ID | Before → after | Watch for |
|---|---|---|---|
| `kinova_frankenstein_osc` | `…-Kinova-Frankenstein-OSC-Play-v0` | **collapsed** (episodes died in ~3 steps) → 3.0 cm / 90 % | **Full-length episodes.** Before, it froze and terminated almost immediately. Confirm it runs the whole episode and holds on target. |
| `ur5e_frankenstein_osc` | `…-UR5e-Frankenstein-OSC-Play-v0` | 41 cm / 0 % → 3.8 cm / 91 % | Wrist stays **stable & compliant** (before: flailed and diverged). Arm reaches and holds. |
| `ur5e_f140_osc` | `…-UR5e-F140-OSC-Play-v0` | 11.9 cm / 67 % → 2.1 cm / 93 % | Tight, damped approach (this is the partial-decoupling win on a non-redundant arm). No under-damped overshoot/wobble. |
| `ur10_frankenstein_osc` **seed 42** | `…-UR10-Frankenstein-OSC-Play-v0` | 65 cm / 0 % → **20.1 cm / 58 %** | **The one honest weak spot.** It reaches but with a **visible position offset (~20 cm)** — close but biased. This is a seed-unlucky local optimum, not a controller failure. Compare against seed 7 / 123 below. |
| `ur10_frankenstein_osc` **seed 7** | same | → 4.1 cm / 90 % | Should reach **cleanly** — the same config, different seed. |
| `ur10_frankenstein_osc` **seed 123** | same | → 6.5 cm / 76 % | Reaches, slightly looser than seed 7. Together the trio shows the config is sound; seed 42 just found a worse optimum. |

`ur10_frankenstein_osc` seed paths (all share one Play ID):
```
ur10_frankenstein_osc/2026-07-02_10-33-04_ppo_torch_seed42/checkpoints/best_agent.pt
ur10_frankenstein_osc/2026-07-02_11-41-00_ppo_torch_seed7/checkpoints/best_agent.pt
ur10_frankenstein_osc/2026-07-02_11-41-00_ppo_torch_seed123/checkpoints/best_agent.pt
```

## Priority 2 — other tuned controllers (spot-check)

Changed but not previously broken: partial decoupling on the rigid-arm OSC, and
the IK-Abs exploration-std fix. Confirm no artifacts.

| Variant (dir) | Play task ID | Result | Watch for |
|---|---|---|---|
| `kinova_f140_osc` | `…-Kinova-F140-OSC-Play-v0` | 3.7 cm / 87 % | Clean, tight reach (decoupling on 7-DOF redundant arm + null-space posture). No residual jitter. |
| `ur10_f140_osc` | `…-UR10-F140-OSC-Play-v0` | 3.1 cm / 91 % | Clean reach on the heavy arm — decoupling should make it behave like the light UR5e. |
| `ur10_frankenstein_ikabs` | `…-UR10-Frankenstein-IK-Abs-Play-v0` | 8.9 → **5.1 cm / 83 %** | The IK-Abs `initial_log_std=−1` fix. Smooth approach to the absolute target, not jerky/overshooting. |
| `kinova_f140_ikabs` | `…-Kinova-F140-IK-Abs-Play-v0` | 5.8 cm / **67 %** | Weakest success in the grid — watch whether it **hovers/overshoots** the marker rather than settling. Loose but acceptable. |
| `ur10_f140_ikabs` | `…-UR10-F140-IK-Abs-Play-v0` | 4.7 cm / 85 % | Same IK-Abs family; confirm it settles rather than orbiting the target. |

## Priority 3 — untouched controllers (quick sanity, low risk)

The 13 joint (EMA) and IK-Rel variants were **not modified** in this work
(bit-identical to prior runs). Just confirm reach still works — a couple of
spot-checks is enough.

- Best behaviour to eyeball as a reference: **`ur5e_f140`** (`…-UR5e-F140-Play-v0`,
  1.8 cm / 96 %) — the tightest reacher in the grid.
- One Frankenstein joint variant, e.g. **`kinova_frankenstein`**
  (`…-Kinova-Frankenstein-Play-v0`, 2.5 cm / 93 %) — confirms the compliant
  wrist is well-behaved under plain joint control too.
- Remaining joint / IK-Rel variants (`*_ik`, and the bare arm dirs): reach in
  the 2–5 cm band; play only if a Priority 1–2 result surprises you.

Full per-variant numbers: `doc/reports/tracking/reach_optimization_tracking.md`
(iteration 15) and the reach task `README.md` results matrix.
