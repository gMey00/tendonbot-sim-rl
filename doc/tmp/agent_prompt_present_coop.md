# Task: `shirt_present` hem↔hem curriculum + cooperative holder (Stage-2 agent **T2**)

You are a coding agent with full access to this repo, running on the **Alex NHR HPC
cluster** (read `doc/Alex_cluster/alex_quickstart.md` first — Slurm-only GPU access,
`tools/train_alex.sh`, conda env `env_isaaclab`). You continue the `shirt_present`
work (pipeline Task 2) through roadmap stages **S0(T2-part) → S1 → [gate G1] → S3** of
[doc/pipeline_stage2_execution_plan.md](pipeline_stage2_execution_plan.md), based on
[doc/reports/tmp/RESEARCH_REPORT_shirt_sorting_stage2.md](reports/tmp/RESEARCH_REPORT_shirt_sorting_stage2.md)
(§2-Q1, §2-Q2, §3 Stages 0/1/3). Three sibling agents run in parallel (T1 =
shirt_pick, T3 = shirt_distribute, INF = shared infra) — coordination rules in §8
are hard requirements.

## 1. Starting point — don't re-derive

- The current env IS the hem↔hem geometry (merged 2026-07-10): holder = passive
  IK-posed tensegrity gripping at (0.15, 0.90, 1.60); learning arm = UR5e-F140;
  best checkpoint det. present **0.380 @ coverage 0.637, grasp 0.73**
  (`logs/skrl/need_visual_verification/shirt_present/` — read its README **and
  `findings.md`**). Task package:
  `src/tensegrity_pick/source/.../tasks/manager_based/shirt_present/`.
- The two measured limiters (tracking report Phase 2): (1) the HIGH second hem
  corner is unlearnable so far (grasp rate < 0.1); (2) a random first grasp caps
  coverage at ~0.64 (study pair map: hem↔hem cells reach 0.82).
- Geometry/threshold ground truth =
  [present_heuristics_study.md](reports/present_heuristics_study.md): hem↔hem 0.820
  @ tautness 1.05 (realizable oracle bound); success threshold 0.65 (curriculum end
  0.75); regrasp hurts with ANY target (−0.031/−0.051) — the hand grasp is never
  released; tautness pulse (1.10 → relax 1.05) is legitimate and halves settle time;
  two-attachment stretch validated only to tautness ≤ 1.15.
- Retained MDP lessons (do not regress): never penalize "gripper closed while far"
  (grasp rate → 0); tautness normalized by flat-rest distance; one-shot rewards need
  ~60× per-step weights (dt-scaling); checkpoint selection ONLY by deterministic
  eval on unseen seeds (`scripts/skrl/evaluate_shirt_present.py`).

## 2. Stage S0 (your part) + S1 — hem↔hem curriculum

Goal: **hem↔hem grasp rate > 0.6** (from < 0.1) at present-gate coverage **0.65**.

1. **Raise the success/coverage gate 0.50 → 0.65** (study threshold). Log
   `final_coverage` stays in place, so past runs remain re-scoreable.
2. **Tautness-pulse legitimacy**: verify no reward term penalizes an
   overshoot-then-relax trajectory (overstretch onset is 1.10 — a brief pulse AT
   1.10 must not be net-punished; check `overstretch_penalty` accounting vs the
   study's §5.6 finding and adjust the onset/shape if it is).
3. **Hem-biased hanging bank** — the blocking data problem: the 356-state bank has
   only **19 hem-corner-anchored states** (measured 2026-07-12). Extend
   `scripts/asset_generation/generate_hanging_bank.py` (you own it) with
   region-biased anchor sampling (the study's region machinery:
   `scripts/model_validation/present_heuristics/regions.py` + `markers.py`,
   `PARTICLE_REGION`) and generate a curriculum bank with controlled anchor-region
   mix (e.g. 50 % hem_corner / 25 % side / 25 % uniform — document your choice).
   Keep the old bank file untouched; new file alongside it.
4. **Target-conditioned grasp curriculum**: stage episodes from easy to hard —
   (a) target corner starts LOW/near (bank-filter on drape + target-corner height),
   (b) raise the target corner height with a curriculum term as grasp rate improves,
   (c) optional reach-assist resets (arm reset nearer the target early on).
   Curriculum semantics gotcha: `num_steps` counts trainer timesteps (env.step
   calls), NOT per-env samples.
5. **Retrain** (fold 1–4 into ONE training campaign, ≤ 3 seeds × 2 configs before
   the gate review). Evaluate deterministically; stage the best checkpoint for
   visual verification (§7).

**Gate G1 (STOP and hand back):** report grasp rate + coverage. > 0.6 → Georg
authorizes S3. 0.3–0.6 → one documented curriculum iteration, then re-gate.
< 0.3 → abort to the accessible-corner/oracle-target task (banking 0.679–0.713)
per the report's fallback; do NOT self-authorize further hem↔hem attempts.

Also at G1: T1 delivers the REAL Task-1→2 terminal bank. Run the **seam gate**:
retrain your best config on the real bank; if present rate drops > 0.05 vs the
synthetic bank, fall back to synthetic + inject T1's slip distribution (report §4
mitigation), and document the shift (this is the thesis's #1 flagged risk).

## 3. Stage S3 (after G1 authorization) — cooperative holder

Trigger criterion (from the report): grasp rate > 0.6 but the coverage ceiling
stays < 0.75 with the fixed holder → the holder pose is the binding constraint.

1. **Movable scripted holder first**: promote the holder tensegrity from scenery to
   a scripted repositioner — move its holding pose toward study-informed
   configurations (hold height/side chosen from the pair map given the realized
   first-grasp region). Constraints: the tensegrity base is prismatic in y/z ONLY
   (no x translation), weak wrist torques (its drives are artificially stiffened
   as scenery — if it becomes an actor, restore physical gains and re-verify it
   can hold; see `shirt_present_env_cfg.py.__post_init__` and
   [project memory](reports/tracking/shirt_present_optimization_tracking.md)). The holder
   NEVER releases (regrasp hurts with any target).
2. **Single joint policy over both arms** (the report's recommended formulation —
   NOT MAPPO): extend the SAME manager-based env with a second action term on
   `holder_robot` (low-dim: base_y, base_z, optionally wrist) and holder
   proprioception observations; one policy outputs the concatenated action.
   Shared presentation reward (coverage + tautness band + stillness) + light
   per-arm shaping (reach shaping for the grasp arm, holding-pose-stability
   shaping for the holder). Cap the pull/stretch shaping at tautness 1.15.
   Watch for the lazy-agent pathology (holder parks; grasp arm does everything) —
   if the holder stays inert, add a small holder-pose-diversity or
   coverage-improvement-credit term, document what you measured.
3. **Success:** coverage ≥ 0.65 trending to 0.75 at grasp rate ≥ 0.9,
   deterministic unseen-seed eval. **Abort:** if the joint policy does not beat
   your best fixed-holder result, hand back with the comparison — Georg decides
   (single-agent ship vs MAPPO ablation S6; do not start S6 yourself).

## 4. Existing infra (read before writing code)

- `ClothObject` two attachment slots; holder = slot 1, hand = slot 0;
  `shirt_grasp_point_w` override targets the hem corner; `_update_grasp` is
  shared — do not copy it.
- INF will land camera-realistic observation terms (12 keypoints + visibility
  flags, noisy grasp flag) in `shared/cloth_sorting_mdp.py` early in Phase 1 —
  rebase when Georg merges, and integrate them in YOUR cfg (you own
  `shirt_present/**`; INF never edits your files).
- Eval: `evaluate_shirt_present.py` (+ `_sweep` variant); remember
  `torch.manual_seed` for rollout differentiation.
- Study data (read-only ground truth): `doc/reports/data/present_*.csv`,
  `present_oracle_policy.json`, `present_markers.pt`.

## 5. Validation before training (repo rule)

Zero/random agents keep working (`scripts/skrl/play_zero.py`, `random_agent.py`,
4 envs headless ≥ 45 s); a scripted probe of the new curriculum resets (corner
height distribution over the new bank) before any GPU campaign; smoke-train
(8 envs, few hundred steps) before Slurm submission.

## 6. Documentation & reporting (mandatory, repo style)

- Update `doc/reports/tracking/shirt_present_optimization_tracking.md` as you go (dated
  iterations, losers kept for the record, measured numbers not adjectives).
- Update `shirt_present/README.md` MDP tables when the MDP changes.
- Tick your items in `doc/TODO.md` (Pipeline Stage 2 → S0/S1/S3).
- Revalidation scripts with usage headers in `scripts/model_validation/` for
  every claim.

## 7. Visual verification staging (blocking loop with Georg)

For every candidate checkpoint: stage under
`logs/skrl/need_visual_verification/shirt_present/<run_name>/` with a README
containing the exact play command, deterministic eval table, and a **"what to
look for" checklist** specific to your change — for S1: does the arm reach the
HIGH corner honestly (no fling/hook artifacts), is the chord horizontal at holder
height, bunching?; for S3: holder motion smooth, no two-arm oscillation, no
overstretch (visible tearing/stretch > 1.15), camera view unoccluded. Georg
plays it and writes `findings.md` in the same folder — **treat findings.md as
blocking input**: read it before your next training iteration.

## 8. Coordination & constraints (hard rules)

- Branch: `project/shirt-present-coop` (from `project/tendonbot-sim-rl`); commit
  in logical increments, push regularly; Georg merges.
- You own `shirt_present/**`, `scripts/asset_generation/generate_hanging_bank.py`,
  and new hanging-bank files. **Do NOT edit `shared/**`** (INF owns it — if a
  shared change seems unavoidable, implement a local override in
  `shirt_present/` and flag it), nor `shirt_pick/**`, `shirt_distribute/**`.
- One Isaac process per allocated GPU; all training via Slurm.
- STOP at the gates (G1, S3 success/abort) and hand back a summary — never
  self-authorize the next stage or budget overruns (≤ 3 seeds × 2 configs per
  campaign before a gate review).
