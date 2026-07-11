# Task: `shirt_distribute` polish — bin 2, layout randomization, pipeline seams (Stage-2 agent **T3**)

You are a coding agent with full access to this repo, running on the **Alex NHR HPC
cluster** (read `doc/Alex_cluster/alex_quickstart.md` first — Slurm-only GPU,
`tools/train_alex.sh`, conda env `env_isaaclab`). You close the remaining
`shirt_distribute` items of
[doc/pipeline_stage2_execution_plan.md](pipeline_stage2_execution_plan.md) ("Task 3
polish"), based on
[doc/reports/RESEARCH_REPORT_shirt_sorting_stage2.md](reports/RESEARCH_REPORT_shirt_sorting_stage2.md)
(§2-Q1 item 3: Task 3 is "essentially done" — this is targeted polish, not a
redesign). Siblings: T1 (shirt_pick), T2 (shirt_present), INF (shared infra) —
coordination rules in §7.

## 1. Starting point — don't re-derive

- Goal-conditioned drop into the commanded drum; mode collapse SOLVED via
  per-goal value/advantage normalization + multi-head critic + FiLM goal-gating
  (`shirt_distribute/learning/`, `…-PerGoal-FiLM-v0`): best checkpoint
  `seed1_filmU/agent_92000`, 1080-ep eval mean **0.882** — bins
  **0.910 / 0.899 / 0.832**. Full arc:
  `doc/reports/shirt_distribute_optimization_tracking.md` Phases 2/2b/2c; staged
  playback: `logs/skrl/need_visual_verification/shirt_distribute/` (read README
  + any `findings.md`).
- **Bin 2 (trash, behind the pedestal arm) sits at a reachability ceiling ≈ 0.83**
  — persists across FiLM seeds; the scripted baseline itself only reaches 0.88
  there. This is a task-side limit, not a learning lever.
- Init: "shirt already in gripper" from the cached holding-pose bank
  (`ur5e_f140_distribute_pose_bank.pt`) + fingertip hang restore;
  `RelativeJointPositionAction` (zero action holds the pose).

## 2. Work items (ordered)

1. **Lift bin 2 ≥ 0.85** — task-side levers from the TODO (pick one at a time,
   measure): (a) release-velocity conditioning (TossingBot-style: reward shaping
   that credits carry velocity toward the drum at release, letting the shirt
   travel), (b) a bin-2-specific approach/release shaping term, (c) wider
   holding-pose coverage for reaching behind the base (extend the pose bank).
   Justify the chosen lever against the scripted-baseline ceiling (0.88) — if
   the baseline itself can't clear 0.85 with the same action path, document WHY
   (reachability geometry) and propose the minimal scene/mount change to Georg
   instead of burning GPU-days.
2. **Bin-layout randomization** so nearest ≠ correct generalizes: per-episode
   drum permutation/jitter with the commanded bin still resolved via
   `target_bin_rel`. Target: > 85 % correct-bin across labels AND layouts.
   Watch for a re-emergence of per-goal collapse under layout variation — the
   per-goal normalization machinery should absorb it; measure per-bin rates.
3. **Hold-presented-until-commanded gate** (report §2-Q1): a short initial phase
   where the policy must HOLD the inspected/taut presentation state until the
   drop command arrives (models the classifier latency). Implement as an episode
   phase flag in the observation + a hold reward before the command; keep it
   cheap — this is a seam-realism feature, not a new skill.
4. **Task-2→3 real seam (Phase 2, when Georg hands you T2's terminal bank):**
   retrain/fine-tune from the REAL Task-2 terminal states (particle state + both
   grasp masks + holder pose) instead of the synthetic holding-pose bank. Seam
   gate: if success drops > 0.05 vs the synthetic bank, report the shift
   (distribution-shift analysis: which start poses fail) — this is the thesis's
   #1 flagged risk, the negative result is valuable.

## 3. Existing infra

- Rewards: graded one-shot `release_event`, anti-hover clearance fade,
  release-required `in_target_bin`, `bad_release` discovery curriculum,
  wrong-bin penalty, `settle_after_success` — all dt-scaled (one-shots ~60×
  per-step). Don't rebalance without a measured reason.
- Eval: `scripts/skrl/evaluate_shirt_distribute.py`;
  `scripts/plot_shirt_distribute_training_results.py`. Kinematic facts: drum rim
  0.88 m, max lift puts the lowest particle ≈ 0.93 m, carry height ≤ 1.2 m —
  clearance predicates beyond these are dead code (shirt_place lesson).
- Baseline: `scripts/model_validation/baseline_shirt_distribute.py` (0.88) —
  extend it for bin-2 lever validation BEFORE training.

## 4. Validation before training

Zero/random agents keep working; scripted-baseline probe of any new lever
(release velocity, layout randomization reset) before Slurm campaigns;
≤ 3 seeds × 2 configs per campaign before a gate review with Georg.

## 5. Documentation & reporting

- `doc/reports/shirt_distribute_optimization_tracking.md`: dated iterations,
  losers kept, measured numbers only.
- `shirt_distribute/README.md`: MDP tables on change.
- Tick `doc/TODO.md` (Pipeline Stage 2 → Task 3 polish).

## 6. Visual verification staging

Stage candidates under
`logs/skrl/need_visual_verification/shirt_distribute/<run_name>/` with README
(play command, per-bin eval table, checklist). Checklist for your changes:
bin-2 releases land INSIDE the drum (not rim bounces), no throwing that scatters
the shirt outside the scene bounds, layout-randomized episodes go to the
COMMANDED (not nearest) drum, the hold-presented phase looks like holding (no
jitter dance). Georg writes `findings.md` — blocking input for your next
iteration.

## 7. Coordination & constraints (hard rules)

- Branch: `project/shirt-distribute-polish` (from `project/tendonbot-sim-rl`);
  push regularly; Georg merges.
- You own `shirt_distribute/**` and its scripts/banks. **Do NOT edit
  `shared/**`** (INF owns it; local override + flag if unavoidable), nor
  `shirt_pick/**`, `shirt_present/**`.
- All GPU via Slurm; one Isaac process per allocated GPU.
- STOP at gates (bin-2 lever choice if baseline-capped; seam-gate outcome) and
  hand back a summary rather than self-authorizing scene changes or budget
  overruns.
