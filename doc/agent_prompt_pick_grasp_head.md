# Task: `shirt_pick` terminal bank + learned grasp-point head (Stage-2 agent **T1**)

You are a coding agent with full access to this repo, running on the **FAPS server**
(RTX A6000, local `env_isaaclab` conda env — activate it, don't unset PYTHONPATH;
one Isaac process at a time on this GPU, check `pgrep -f isaac` first — Georg's
interactive sessions have priority). Heavy training moves to the Alex cluster
(`doc/Alex_cluster/alex_quickstart.md`) in Phase 2. You execute roadmap stages
**S0(T1-part) → S2** of
[doc/pipeline_stage2_execution_plan.md](pipeline_stage2_execution_plan.md), based on
[doc/reports/RESEARCH_REPORT_shirt_sorting_stage2.md](reports/RESEARCH_REPORT_shirt_sorting_stage2.md)
(§2-Q1 option c, §3 Stages 0/2). Siblings: T2 (shirt_present), T3
(shirt_distribute), INF (shared infra) — coordination rules in §8.

## 1. Starting point — don't re-derive

- `shirt_pick` is trained to its stage-1 bar: deterministic **present 1.000 /
  grasp 1.000** with `logs/skrl/shirt_pick/2026-07-03_19-33-17_ppo_torch_seed3/
  checkpoints/agent_12000.pt` (low-drop alternative `agent_8000`: present 0.979,
  drop 0.01–0.02); known blemish: **5–7 % post-latch grasp slips**. Task package:
  `src/tensegrity_pick/source/.../tasks/manager_based/shirt_pick/`; tracking:
  `doc/reports/shirt_pick_optimization_tracking.md`.
- The pick heuristic is the **highest point** of the crumpled shirt —
  depth-camera-trivial and literature-standard; semantic keypoints are UNRELIABLE
  on crumpled cloth (research report §2-Q1) — keep it as the bootstrap.
- WHY your work matters: the heuristics study proves the presentation ceiling is
  set by the grasp PAIR — an arbitrary first grasp caps Task-2 coverage at
  **0.713** (oracle second grasp), while a hem-corner first grasp enables
  **0.820**. Task 1 chooses the first grasp.

## 2. Stage S0 (your part) — the REAL Task-1→2 terminal bank

1. Write a rollout script (`scripts/asset_generation/` or `scripts/skrl/`) that
   runs the trained policy (`agent_12000`) deterministically over the crumpled
   bank and snapshots at the present latch via the existing
   `ShirtPickEnv.snapshot_terminal_states()` hook (particle pos/vel + attach mask
   + joint state + presented flag).
2. **Include the post-latch slip episodes** — do NOT filter them; they are the
   real distribution Task 2 must survive (report §3 Stage 0). Record per-state
   metadata: presented flag, slip flag, grasp-particle index, grasp-region label
   (use the study's region assignment:
   `scripts/model_validation/present_heuristics/regions.py` on the flat-rest
   coords — `doc/reports/data/present_heuristics_flat_rest.pt`).
3. Target ≥ **500 presented-terminal states**; save alongside the other banks
   (`res/Props/Cloth/banks/`), document the generation command in the file's
   metadata dict and the tracking report. Also report the empirical
   **first-grasp-region distribution** (this quantifies today's gap to
   hem-corner holds — a thesis figure).
4. Hand the bank path to Georg for the G1 seam gate (T2 consumes it). This is
   your Phase-1 deliverable — light GPU, runs locally.

## 3. Stage S2 — learned grasp-point refinement head

Goal (decision gate): **realized Task-2 presentation coverage from Task-1 holds
> 0.713** (the arbitrary-first-grasp oracle bound) — else revert to plain
highest-point and bank the fallback (report §2-Q1 gate; budget ≈ 100 k trainer
steps × 3 seeds before reverting).

1. **Observations** (camera-contract-compatible): the study's 12 region-landmark
   keypoints (read ONLY those 12 particle positions from privileged state — INF's
   shared observation terms will provide keypoints + visibility flags; rebase
   when merged and use them) + a border-distance feature of the current grasp
   target (from `present_markers.pt` `BORDER_DIST`).
2. **Action**: make the grasp target policy-selectable around the depth-trivial
   bootstrap — design options (pick one, justify with a measured comparison):
   (a) continuous offset on `shirt_grasp_point_w` around the highest point
   (clamped radius), (b) discrete choice among K candidates (highest point + the
   visible/nearest region keypoints). Keep the deterministic attach trigger
   mechanics untouched (`shirt_grasp_point_w` is the intended override hook — do
   not copy `_update_grasp`).
3. **Reward**: keep the existing sequential structure (reach→grasp→lift→
   transport→presented; it reached 1.000 — do not destabilize) and ADD a terminal
   bonus ∝ **predicted downstream coverage** of the achieved hold region. Build
   the lookup table offline first: first-grasp region → expected Task-2 coverage,
   derived from the study's stratified pair map
   (`doc/reports/data/present_h3s_stratified.csv`, best-partner column per first
   region — the study's §4 "best partner per first grasp"). dt-scaling: one-shot
   bonuses need ~60× per-step weights.
4. **Success metric**: report BOTH (a) present rate (must stay ≥ 0.95 — do not
   trade away the stage-1 bar) and (b) the fraction of episodes whose terminal
   hold region is hem_corner/side (the high-value regions), plus the
   lookup-predicted coverage. The TRUE gate (realized Task-2 coverage) is
   measured by chaining: regenerate the terminal bank from your best checkpoint
   and have Georg pass it to T2 for a coverage eval — coordinate through him at
   the gate, don't run Task-2 training yourself.
5. Train on Alex (Slurm) in Phase 2; ≤ 3 seeds × 2 configs before the gate
   review.

## 4. Existing infra

- Crumpled bank (288 trash-toss states) + `_reset_cloth_from_bank`
  (yaw/mirror/jitter augmentation) — unchanged.
- Present gate is on **cloth centroid speed** (< 0.20 windowed), NOT EE speed —
  keep it (EE gates are unharvestable at the raised posture, hard-won lesson).
- Eval: `scripts/skrl/evaluate_shirt_pick.py` (deterministic mean actions;
  `torch.manual_seed` required).
- Study data (read-only): everything under `doc/reports/data/present_*`,
  region/marker code under `scripts/model_validation/present_heuristics/`.

## 5. Validation before training

Zero/random agents keep working; validate the offline lookup table with a
standalone script (`scripts/model_validation/`, usage header) that prints the
region→coverage table and its provenance; smoke-train locally (8 envs) before
any Slurm submission.

## 6. Documentation & reporting

- `doc/reports/shirt_pick_optimization_tracking.md`: dated iterations, losers
  kept, measured numbers only.
- `shirt_pick/README.md`: MDP tables when the MDP changes.
- Tick `doc/TODO.md` (Pipeline Stage 2 → S0/S2).

## 7. Visual verification staging

Stage candidate checkpoints under
`logs/skrl/need_visual_verification/shirt_pick/<run_name>/` with README (exact
play command, eval table, checklist). For S2, the checklist must include: does
the arm still pick RELIABLY (no hesitation shopping between candidates), does
the chosen grasp point look camera-plausible (on the visible upper surface, not
tunneled into the pile), and does the carried hold end at the presentation pose
without new slip behavior. Georg writes `findings.md` — blocking input for your
next iteration.

## 8. Coordination & constraints (hard rules)

- Branch: `project/shirt-pick-grasp-head` (from `project/tendonbot-sim-rl`);
  push regularly; Georg merges.
- You own `shirt_pick/**`, your new scripts, and the Task-1 terminal-bank files.
  **Do NOT edit `shared/**`** (INF owns it; local override + flag if unavoidable),
  nor `shirt_present/**`, `shirt_distribute/**`, the heuristics-study scripts.
- FAPS GPU: one Isaac process at a time; yield to Georg's sessions. Training
  sweeps on Alex via Slurm only.
- STOP at the S2 decision gate with a summary; never self-authorize the revert
  or extra seed budget.
