# Cloth-Sorting Pipeline — Stage 0 Physics De-Risk

**Scope:** the three "Stage 0 — remaining physics de-risk" items from
[doc/TODO.md](../TODO.md) (Master-Thesis Goal Tasks), prerequisite to the
crumpled-state bank and the Task-1 (shirt_pick) implementation.
**Environment:** Isaac Sim 5.1.0 + Isaac Lab, PBD particle cloth
(`tshirt_clothesnet.usd`, 11 048 particles), NVIDIA RTX A6000 (48 GB).
**Date:** 2026-07-03.

Every result below is reproducible with a dedicated script; usage lines are
quoted per section (all run from `src/tensegrity_pick` inside the
`env_isaaclab` conda env, `PYTHONUNBUFFERED=1 python <script> --headless`).

## TL;DR

| De-risk item | Verdict | Key number |
|---|---|---|
| Deterministic cache-restore reset | **PASS** | bit-exact (0.0 m cross-trial deviation); restore 1.2 ms vs 5.5 s settle (≈ 4600×) |
| Two simultaneous attachments under stretch | **PASS** | stable through tautness ratio 1.15; steady-state max particle speed ≤ 0.7 m/s; bbox +5 % |
| Max stable parallel cloth env count | see [§3](#3-parallel-cloth-env-count-benchmark) | — |
| shirt_place regression after the multi-slot refactor | **PASS 5/5** | `test_shirt_fixes.py` unchanged verdicts |

---

## 1. Deterministic cache-restore reset

**Why:** both bank mechanisms (crumpled Task-1 initial states, task-to-task
terminal states) restore cached particle states at reset.  If a restore is
not deterministic and complete, the banks silently inject state noise.

**Script:** [`scripts/model_validation/test_cache_restore_reset.py`](../../src/tensegrity_pick/scripts/model_validation/test_cache_restore_reset.py)

```bash
PYTHONUNBUFFERED=1 python scripts/model_validation/test_cache_restore_reset.py --headless
```

**Method:** settle the shirt on the belt (300 zero-action steps), cache
particle positions+velocities and the robot joint/root state, then restore →
simulate 120 steps → snapshot every 30 steps, three trials; compare snapshots
across trials.

**Results (4 envs):**

| Check | Result |
|---|---|
| Cross-trial max \|Δpos\| at steps 30/60/90/120 | **0.000e+00 m** at every snapshot (bit-exact) |
| Centroid drift of the restored settled state over 120 steps | xy 0.7 mm, z 0.1 mm (tol 50 mm) |
| Restore cost | **1.2 ms** (vs 5.5 s settle-from-scratch → ≈ 4600× cheaper) |

**Implementation note:** determinism requires restoring **both** halves of
the particle state; a new `ClothObject.write_nodal_state_to_sim(pos, vel,
env_ids)` writes them together
([cloth_object.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shared/cloth_object.py)).
Positions-only restores leave stale solver velocities (the historical
"no clean particle reset API" problem, Isaac Lab discussion #1105).

**Observed caveat:** "settled" by the SoftGym criterion (max particle
speed < 0.01 m/s) is never reached — a few particles keep jittering at
~0.17 m/s indefinitely (self-collision contact chatter).  This is harmless
(the completeness check shows sub-mm drift) but means settle loops must be
step-capped, not velocity-gated alone — the bank generator caps at 300 steps.

**Conclusion:** cache-restore resets are exact and effectively free.
The bank architecture is safe to build on.

---

## 2. Two simultaneous attachments under stretch

**Why:** shirt_present needs the retriever to keep holding while the second
arm grasps and stretches.  The research report flags PBD mass-spring
overstretch between two pinned particle groups as the core Task-2 physics
risk (with "handover" as the fallback design).

**Implementation:** `ClothObject` was generalised from one attachment set to
**two independent slots** (`attach/hold/detach(..., slot=)`, default slot 0 =
robot hand so all existing callers are unchanged; slot 1 = second gripper /
holder anchor).  Particles can belong to at most one slot (capture excludes
the other slot's particles), and ANCHOR masses are rebuilt from the union of
slot masks so detaching one grasp never un-pins the other.
shirt_present's static holder anchor now uses slot 1, freeing slot 0 for the
learning arm's future grasp.

**Backward-compat gate:** the shirt_place mechanics regression
(`scripts/model_validation/test_shirt_fixes.py`) still passes **5/5**
(stretch ratio 1.00, release fall +0.255 m, mimic error 0.0 rad, tip-grasp
attach 2/2, weld 196 particles / 0.000 m tip gap).

**Script:** [`scripts/model_validation/test_two_attachments.py`](../../src/tensegrity_pick/scripts/model_validation/test_two_attachments.py)

```bash
PYTHONUNBUFFERED=1 python scripts/model_validation/test_two_attachments.py --headless
```

**Method:** hang the shirt from a static slot-1 anchor at the presentation
pose (the shirt_present configuration), attach slot 0 at the hanging shirt's
lowest point (the literature-standard second grasp), pull it down at
0.05 m/s, and sweep the tautness ratio

> ratio = inter-grasp distance / rest distance
> (rest distance = grasp-cluster centroid distance in the flat rest shape)

from slack to 1.15.  At each checkpoint the stretch is held 5 steps and the
max particle speed averaged (a one-step attach transient is not instability).

**Results (4 envs, slot 0 ⌀300 particles, slot 1 ⌀199, overlap 0):**

| Tautness ratio | max particle speed (5-step avg) | bbox vs hang | NaN | attachments intact |
|---|---|---|---|---|
| 0.70 | 0.70 m/s | ×1.051 | no | yes |
| 0.80 | 0.60 m/s | ×1.052 | no | yes |
| 0.90 | 0.57 m/s | ×1.053 | no | yes |
| 0.95 | 0.54 m/s | ×1.053 | no | yes |
| **1.00** | **0.52 m/s** | **×1.052** | no | yes |
| 1.05 | 0.53 m/s | ×1.053 | no | yes |
| 1.10 | 0.56 m/s | ×1.051 | no | yes |
| 1.15 | 0.59 m/s | ×1.051 | no | yes |

**Conclusions:**

1. **Two simultaneous ANCHOR attachments are stable** under slow stretch —
   no solver blow-up, no NaN, no attachment loss, even 15 % past nominal
   tautness.  The Task-2 "handover" fallback is NOT needed.
2. The system is *more* forgiving than the report anticipated, for a
   structural reason: the straight-line rest distance between cluster
   centroids underestimates the true fabric path, and the 24-iteration
   stretch constraints redistribute strain across the mesh, so ratio 1.0–1.15
   still has real slack.  The Task-2 tautness cap should therefore be
   calibrated against measured strain (bbox/edge growth), not assumed to
   fail at ratio 1.0.
3. The one-step attach transient (~2.8 m/s particle snap when the second
   grasp recentres its cluster) is expected and decays within ~5 steps —
   reward/termination predicates in Task 2 should ignore the first few steps
   after a grasp.

---

## 3. Parallel cloth env-count benchmark

**Why:** the report's Stage-0 threshold is ≥ 256 stable cloth envs; PPO
throughput planning for all three tasks depends on where the practical knee is.

**Scripts:**
[`scripts/model_validation/bench_cloth_env_count.py`](../../src/tensegrity_pick/scripts/model_validation/bench_cloth_env_count.py)
(one N per process) and the sweep driver
[`scripts/model_validation/bench_cloth_env_count.sh`](../../src/tensegrity_pick/scripts/model_validation/bench_cloth_env_count.sh)
(collects `doc/reports/data/cloth_env_benchmark.csv`, greps the logs for
PhysX buffer-overflow warnings = silently degraded contacts).

```bash
bash scripts/model_validation/bench_cloth_env_count.sh 32 64 128 256
```

**Results (shirt_pick env, 11 048 particles/cloth, zero-action stepping,
240 timed steps after warmup; full rows in
[data/cloth_env_benchmark.csv](data/cloth_env_benchmark.csv)):**

| N envs | total particles | construct + pre-settle | steps/s | env·steps/s | GPU used (driver) | PhysX overflows | sane |
|---|---|---|---|---|---|---|---|
| 32 | 0.35 M | 28 s | 13.18 | **422** | 4.4 GB | 0 | ✓ |
| 64 | 0.71 M | 64 s | 6.24 | 399 | 5.5 GB | 0 | ✓ |
| 128 | 1.41 M | 145 s | 2.45 | 313 | 7.8 GB | 0 | ✓ |
| 256 | 2.83 M | 400 s | 0.87 | 223 | 12.1 GB | 0 | ✓ |

**Conclusions:**

1. **Stability threshold met:** 256 parallel cloth envs run without NaNs,
   without PhysX buffer-overflow warnings, and with modest GPU memory
   (12 GB of 48 GB) — the report's ≥ 256 bar is cleared, and memory is not
   the limiting resource.
2. **Throughput, not stability, is the real constraint — and it scales
   SUB-linearly.**  Sample throughput *peaks around N = 32–64*
   (~400–420 env·steps/s) and *falls* to 223 at N = 256: the per-env
   particle systems (`replicate_physics=False`) do not parallelize the way
   rigid-body scenes do, so adding envs beyond ~64 buys batch diversity at a
   net throughput LOSS.  This retroactively explains shirt_place's ~2 it/s
   at 128 envs.
3. **Training env-count recommendation:** 64–128.  128 keeps the proven
   shirt_place PPO batch shape (128 envs × 48 rollouts) at a 27 % throughput
   discount vs 64; going above 128 is strictly worse on this GPU.
4. Construction (cloth authoring + PhysX init + 320-step pre-settle) scales
   ~linearly at ~1.1–1.6 s/env — at 256 envs the ~7 min startup makes
   short debug runs expensive, another reason to iterate at small N.
5. Per-reset cost is negligible at every N (≤ 0.02 s — flat-lay teleport;
   the bank restore measured in §1 keeps it that way).

---

## Stage-0 verdict

All three de-risk items closed.  The pipeline can proceed to the
crumpled-state bank (drop-and-settle generator + bank-restore resets — the
mechanisms validated in §1) and Task-1 training.  Task 2's bimanual stretch
is physically viable with the multi-slot attachments (§2); its remaining
work is MDP design, not physics.
