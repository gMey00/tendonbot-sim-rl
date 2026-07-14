# Task: Pipeline infra — observation contract, asymmetric AC, grasp-fidelity gates (Stage-2 agent **INF**)

You are a coding agent with full access to this repo, running on the **FAPS server**
(RTX A6000; local conda env `env_isaaclab` — activate it, don't unset PYTHONPATH;
**one Isaac process at a time**, check `pgrep -f isaac` first, Georg's interactive
sessions have priority; your GPU needs are smoke-tests only). You execute roadmap
stages **S4 + S5** of
[doc/pipeline_stage2_execution_plan.md](pipeline_stage2_execution_plan.md), based on
[doc/reports/tmp/RESEARCH_REPORT_shirt_sorting_stage2.md](reports/tmp/RESEARCH_REPORT_shirt_sorting_stage2.md)
(§2-Q3, §2-Q5, §3 Stages 4/5). **You are the ONLY agent allowed to edit
`tasks/manager_based/shared/**`** — three task agents (T1/T2/T3) depend on your
deliverables and rebase on them; ship early, keep everything opt-in and
backwards-compatible (the three trained stage-1 policies must keep playing
unchanged).

## 1. Milestone order (deliberately front-loaded)

Your Phase-1 deliverables unblock the others — do them in THIS order and tell
Georg to merge after each milestone:

1. **M1 — skrl asymmetric-AC verdict (≤ 1 day, decides everyone's Phase-2 stack).**
2. **M2 — camera-realistic observation library** in `shared/cloth_sorting_mdp.py`.
3. **M3 — grasp-fidelity gates** in `shared/cloth_sorting_env.py` (opt-in).
4. **M4 — ranking-preservation eval harness + noise-calibration configs (Phase 2).**

## 2. M1 — asymmetric actor–critic verdict (S4)

Verify whether installed **skrl 2.1.0** actually separates policy vs critic
observation groups: build a minimal manager-based env cfg with a `"critic"`
observation group (Isaac Lab maps it to `num_states`/`state_space`), train a few
hundred steps, and instrument whether the value network consumes the critic group
or silently aliases the policy group (inspect the wrapper's `state()` path and
the model input shapes at runtime — don't trust docs). Known context: this was
flagged in-repo (TODO Cross-cutting) and in the research report; skrl historically
aliased STATES to OBSERVATIONS.

- **If separated:** document the working config pattern (a minimal example cfg +
  agent yaml) for T1/T2 to copy.
- **If aliased:** budget and build the **RSL-RL fallback runner** (none exists in
  `scripts/` today): a `scripts/rsl_rl/train.py`+`play.py` pair following the
  Isaac Lab template, validated on the existing `shirt_present` env (its
  stage-1 checkpoint won't transfer — just prove the training loop runs and the
  critic sees privileged obs). Keep skrl as default for symmetric runs.
- Hand Georg the verdict as a one-paragraph summary — it gates Phase 2.

## 3. M2 — camera-realistic observation library (S4)

New observation terms in `shared/cloth_sorting_mdp.py` (pure functions on env
attrs, zeros-fallback guard for the ObservationManager shape-probe — established
pattern there). Per the report's observation contract (§2-Q3):

- **12 region-landmark keypoints + per-keypoint visibility flags**: positions of
  the 12 study keypoints (particle indices from
  `doc/reports/data/present_markers.pt` — load once, cache; the study's
  `markers.py` shows the format) relative to a reference frame; visibility from
  the study's two-sided depth-layer logic
  (`shared/cloth_metrics.py::two_sided_visible_fraction` internals — factor a
  per-particle visibility helper out of it rather than duplicating).
- **Configurable noise models** (off by default): keypoint position Gaussian
  σ ≈ 1–2 cm + per-keypoint dropout at (1 − recall) with recall ≈ 0.74 (Lips
  2024 — cite in the docstring); grasp-active flag with flip probability +
  latency steps (gripper-current proxy); use Isaac Lab's obs corruption/noise
  hooks where they fit.
- **N-point down-sampled surface cloud** (visible particles only, fixed N,
  deterministic given seed) and **coverage-from-mask** (should be ≡ the existing
  silhouette coverage — assert that, it validates the contract).
- **Tautness stays privileged**: document in the module docstring that tautness
  must NOT enter actor groups (not camera-observable); it remains available for
  critic groups and rewards.
- Unit-test offline where possible (extend
  `scripts/model_validation/test_cloth_metrics.py` or a sibling test file —
  keypoint visibility and noise determinism are pure-torch testable).
- **Do NOT edit any task's ObservationsCfg** — T1/T2/T3 integrate your terms
  themselves. Provide a short usage snippet in the tracking report.

## 4. M3 — grasp-fidelity gates (S5, opt-in, default OFF)

Extend `shared/cloth_sorting_env.py::_update_grasp` with two **evaluation-mode
gates**, both behind env-cfg flags defaulting to off (the stage-1 policies and
all training keep the deterministic attachment — SoftGym-lineage rationale in
the report §2-Q5):

1. **Pre-condition gate**: before allowing the anchor to form, require
   (a) approach alignment (finger-tip approach axis vs local cloth surface
   normal within a threshold), (b) finger clearance (no rigid contact jam), and
   (c) a **layer count estimate** at the grasp point (particles within the weld
   radius clustered by the depth-layer logic of
   `two_sided_visible_fraction` — > 2 layers = multi-layer pinch).
2. **Stochastic misgrasp/slip gate**: on attach, draw failure with probability
   keyed to (grasp location: on-border/seam vs mid-panel via `BORDER_DIST`,
   layer count, and pull force); during hold, slip when the effective anchor
   load proxy exceeds a capacity draw. Calibrate defaults to the published
   parallel-jaw stats (DRAPER: 16–22 % misgrasp; cite in the config docstring)
   and make every number a cfg field.
3. Validate on a robot-free rig (reuse the heuristics-study harness pattern —
   `scripts/model_validation/present_heuristics/study_common.py` shows how to
   boot + script attachments): measure the gates' misgrasp rates by region and
   confirm the deterministic path is bit-identical with flags off (regression:
   `test_shirt_fixes.py` must stay 5/5).

## 5. M4 — ranking-preservation harness (S5, Phase 2)

A script (`scripts/model_validation/eval_grasp_fidelity.py`, usage header) that
takes ≥ 2 checkpoints of a task, evaluates each deterministically under
(a) idealized and (b) gated grasp, and reports per-checkpoint success + the
ranking verdict. Decision rule to print: promote gates into training ONLY if
success drops > 0.15 absolute AND the ranking reorders (report §2-Q5 gate).
Run it on the existing shirt_pick and shirt_present checkpoints and hand Georg
the first ranking-preservation table.

## 6. Validation & regression (hard requirements)

- With all new flags OFF: `test_shirt_fixes.py` 5/5, zero/random agents on all
  three tasks unchanged, and one stage-1 checkpoint per task plays with
  unchanged deterministic metrics (`evaluate_shirt_pick.py` etc.) — your
  shared-code changes must be invisible until opted into.
- Offline unit tests for the pure-torch pieces; smoke boots at 4–8 envs locally.

## 7. Documentation & reporting

- NEW tracking report `doc/reports/pipeline_infra_tracking.md` (create): dated
  iterations, the M1 verdict with evidence, calibration sources, usage snippets
  for T1/T2/T3.
- Tick `doc/TODO.md` (Pipeline Stage 2 → S4/S5 + the old Cross-cutting
  asymmetric-AC item).
- No `need_visual_verification` staging needed unless you produce a policy;
  your M3 rig outputs are tables/figures — put them in the tracking report.

## 8. Coordination & constraints (hard rules)

- Branch: `project/pipeline-infra` (from `project/tendonbot-sim-rl`); push after
  EVERY milestone and tell Georg to merge — T1/T2 rebase on your M2/M3.
- You own `shared/**`, `scripts/skrl/train.py` (asym-AC/RSL-RL additions only),
  your new scripts/tests. **Do NOT edit** `shirt_pick/**`, `shirt_present/**`,
  `shirt_distribute/**`, task configs, or the heuristics-study scripts (the
  study data/markers are read-only inputs).
- Everything opt-in/backwards-compatible; breaking the three trained stage-1
  policies is a hard failure.
- FAPS GPU etiquette: one Isaac process, yield to Georg; no training campaigns
  on this box (if M1's RSL-RL validation needs a longer run, ask Georg for an
  Alex allocation).
