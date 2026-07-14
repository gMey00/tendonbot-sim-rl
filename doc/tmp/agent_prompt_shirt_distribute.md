# Task: Implement, validate, debug and train `shirt_distribute` (cloth-sorting pipeline, Task 3) — Alex HPC agent

You are a coding agent with full access to this repo, running on the **Alex NHR HPC cluster**
(read `doc/alex_quickstart.md` first — Slurm-only GPU access, `tools/train_alex.sh` wrapper,
Isaac Sim/Lab + conda env `env_isaaclab` installed under `$HPCVAULT` via
`tools/install_IsaacLab_Alex.sh`). Your job is the **implement → validate (scripted baseline) →
train → analyze** loop for the initial `shirt_distribute` task until it satisfies §2, then hand
back a clear summary. Two sibling agents work in parallel on `shirt_present` (also on Alex) and
on a presentation-heuristic study (FAPS server) — **coordination rules in §8 are hard
requirements.**

---

## 1. The task

Goal-conditioned bin placement (TossingBot-style formulation, see
`doc/reports/tmp/RESEARCH_cloth_sorting_pipeline.md` §4): the second robot holds the classified
shirt and must **drop/place it into the commanded drum** (3 drums at `(0.15, 1.0)`,
`(1.35, 1.0)`, `(0.75, 1.6)`; the per-episode target bin is resampled uniformly so "nearest bin"
≠ "correct bin" and the policy cannot ignore the goal — the machinery exists in the stub).

**Initial state (your first implementation job):** the episode starts with the shirt **already
hanging from the learning robot's own closed gripper**, attached at **one random shirt particle**
(slot 0), with the arm in a pose **sampled from plausible end-of-Task-2 configurations** — i.e.
the gripper somewhere in the region where the stretch/presentation ends (around the presentation
area at `PRESENTATION_POS = (0.15, 0.90, 1.60)` and between it and the robot's home posture).
Sample from a random distribution over that region; the exact distribution is your call — make it
wide enough to be robust, document it. (Later the pipeline will replace this with the real
Task-2 terminal-state bank; keep that seam visible.)

Part of your job is to **figure out fitting observations and success metrics** (starting points
in §4/§6) — document reasoning and measured alternatives in the tracking report.

Recommended reset recipe (all pieces exist):
1. Sample arm joint positions (nominal "holding" posture + bounded noise; verify with FK that
   the finger tip lands in your target region — reject/resample outliers).
2. Write them via `robot.write_joint_state_to_sim` + set position targets, then
   `self._force_gripper_closed(env_ids)` (base-env helper — without it the open default pose
   makes `_update_grasp` release the restored grasp on the first step).
3. `self._reset_cloth_hanging_from_bank(env_ids, tip_pos_w, slot=0, max_drape=...)` — restores a
   relaxed random-particle hang at the fingertip and re-attaches slot 0 (bank:
   `res/Props/Cloth/banks/tshirt_hanging_bank.pt`, 356 states; regenerate on Alex with
   `scripts/asset_generation/generate_hanging_bank.py --headless --num_envs 32 --rounds 12` if missing).
   Pick `max_drape` so the hang clears the drum tops (z ≈ 0.88) and belt from your sampled poses.
4. Beware: the policy's gripper action overwrites the finger target from the first action step —
   an "open" command releases immediately. That is legitimate (release IS the task's final act),
   but check your action initialization/reward so episodes don't start with an instant drop
   (e.g. initialize previous-action to "closed", or penalize pre-transport release).

## 2. Success criteria

- A deterministic (mean-action) checkpoint with `Metrics/distribute_success_rate` ≥ **0.85**
  across all three commanded bins (report the per-bin breakdown), across seeds/episodes —
  release required, landing settled in the CORRECT drum.
- No degenerate strategies: no hovering-without-release (anti-hover lesson from shirt_place), no
  single-bin specialization (check the per-bin confusion), no flinging that only works for the
  nearest bin.
- Zero/random agents keep working; a scripted baseline (§6.2) validates the MDP before training.
- Everything documented per §7.

## 3. Robot & action space (decided — deviate only with measured justification)

Same decision and reasoning as the shirt_present agent (keep the two tasks comparable):

- **Robot: UR5e-F140** (`Template-Shirt-Distribute-UR5e-F140-v0`) — best reach-grid variant
  (1.8 cm / 96 %, `doc/reports/tracking/reach_optimization_tracking.md` Iter. 14–15), already mounted at
  the second-robot pedestal `(0.75, 1.0, 0.75)` in the shared scene. Fallback: `Kinova-F140`.
- **Action space: joint-space** (reach winner, 3.1 cm / 91 %). Stub uses
  `JointPositionActionCfg` (delta, scale 0.5) — proven on the cloth tasks; the reach winner was
  `EMAJointPositionToLimitsActionCfg` (α = 0.2, `reach/config/f140_reach_common.py`). Start with
  the stub, switch only on measured need; document. A release-throw variant (TossingBot) is a
  LATER option only if the far drum proves outside comfortable reach.

## 4. What is already built for you (shared infra — read before writing code)

Task package: `src/tensegrity_pick/source/.../tasks/manager_based/shirt_distribute/`
Shared: `.../tasks/manager_based/shared/` — **read-only for you** (§8).

- `ShirtDistributeEnv` stub: per-episode random `_target_bin`, `target_bin_pos_w`,
  `_target_fraction_in_bin` (particle fraction inside the commanded drum cylinder, threshold
  0.15 like shirt_place), `Metrics/distribute_success_rate` (release-required), and the
  `target_bin_rel_shirt` observation in `shared/cloth_sorting_mdp.py`. The stub currently resets
  the shirt flat on the belt edge — replace with the grasped-hang reset (§1).
- `shared/cloth_sorting_env.py::ClothSortingEnvBase`: PBD cloth machinery, deterministic
  attach/hold/detach grasp (`_update_grasp`; `grasp_active` = slot 0), Robotiq four-bar drive,
  `_force_gripper_closed`, `_reset_cloth_hanging_from_bank` (yaw+mirror augmentation +
  `max_drape` filter), `shirt_grasp_point_w` overridable grasp-target property.
- Finger-tip frames calibrated for the UR/Kinova `robotiq_base_link`
  (`gripper_cfg.EE_FRAME_TIP_SIGN`; probe `scripts/model_validation/check_f140_finger_tip.py`) —
  `_finger_tip_pos` and the belt penalty/termination helpers are frame-correct as of 2026-07-04.
- `shared/cloth_metrics.py` (pure torch): silhouette/stretch metrics — probably not needed here,
  but available.
- **Port from `shirt_place`** (the validated drop-into-drum task — your closest relative):
  release-event one-shot with quality grading, anti-hover fade, clearance shaping,
  `placed_and_settled`, belt penalties; `shirt_place/mdp/rewards.py` +
  `doc/reports/tracking/shirt_place_optimization_tracking.md` document why each exists.
- Reference for env bookkeeping style: `shirt_pick/shirt_pick_env.py` (one-shot events consumed
  by the reward manager next step, metrics logged in `_reset_idx`, windowed latches).

## 5. Hard-won lessons (violate these and you will re-discover them expensively)

1. **dt-scaling:** rewards are multiplied by dt (1/60 s) — per-step weights earn
   ≈ weight × episode-seconds, one-shots weight/60 → size one-shots ~60× per-step terms.
2. **Release-required success + anti-hover:** shirt_place farmed shaping by hovering over the
   drum without releasing until the release-event grading + hover fade fixed it — port that
   design, don't re-derive it.
3. **Windowed/settled success evaluation:** after release, let the cloth settle before judging
   (shirt_place's `placed_and_settled`); cloth speed gates on the CENTROID, never the EE.
4. **Validate the MDP with a scripted baseline before training** (grasp-carry-release over each
   drum) — it catches unreachable bins and broken success accounting for free.
5. **`mdp/__init__.py` must import local rewards via `from .rewards import *`** — a plain
   `from . import rewards` silently keeps isaaclab's own `rewards` module bound.
6. **Cloth env count:** throughput peaks at N = 32–64 and FALLS above (A6000 numbers); never
   train above 128 without re-benchmarking (`scripts/model_validation/bench_cloth_env_count.sh`)
   — re-run once on the RTX PRO 6000 and record the numbers.
7. **Deterministic eval:** mean actions; explicit `torch.manual_seed` (env seed alone gives
   identical rollouts); copy `scripts/skrl/evaluate_shirt_pick.py`. skrl `--max_iterations` is
   ×48-rollout units.
8. Ops: `PYTHONUNBUFFERED=1`; carb traps SIGTERM → `kill -9`; `torch.inference_mode()` around
   state writes after inference stepping; boot 1–2 min; ignore the USD `Sublayer ... has cycles`
   warning.

## 6. Suggested work plan

1. **Grasped-hang reset** (§1 recipe) + obs set: `target_bin_rel` (exists), shirt-centroid rel
   EE, `grasp_active`, joint pos/vel, previous action, gripper closure. Verify with
   `scripts/agents/zero_agent.py`/`random_agent.py --num_envs 4 --headless` (PASS = "Gym observation
   space" + ≥45 s stepping; they loop forever — kill) and confirm in logs that the hang restore
   + closed gripper survives ≥ 100 zero-action steps without dropping.
2. **Scripted baseline** (`scripts/model_validation/baseline_shirt_distribute.py`): P-servo
   carry to each drum center → release → settle; measure per-drum success + carry times → sets
   episode length and confirms all three drums are reachable from your init distribution.
3. **Rewards** (shirt_place port, dt-scaled): approach-target-bin tanh (grasp-gated) →
   clearance shaping → graded release event over the correct bin (one-shot, ~60×) →
   `placed_and_settled` bonus → wrong-bin release / drop / belt-contact penalties →
   action-rate/joint-vel with the standard curriculum step.
4. **Train** (`tools/train_alex.sh Template-Shirt-Distribute-UR5e-F140-v0 --headless
   --num_envs 64`, adjust after your env-count re-benchmark). Diagnose stalls with a
   gate-fraction diagnostic (copy `diag_shirt_pick_policy.py`) before touching weights.
5. **Deterministic eval** ≥ 2 seeds × ≥ 48 episodes with per-bin breakdown; select checkpoint;
   keep the Task-2-bank seam documented (what changes when init comes from the real bank).

## 7. Documentation & reporting (repo style — mandatory)

- **`doc/reports/tracking/shirt_distribute_optimization_tracking.md`** (stub exists): dated iteration
  entries — goal, changes, experiments (including losers, kept for the record), results tables,
  root-cause analyses. Update as you go.
- **`shirt_distribute/README.md`** (stub exists): task definition, MDP tables, decisions with
  rationale, final results + figures (port `scripts/plotting/plot_shirt_pick_training_results.py`;
  figures under `shirt_distribute/figures/<variant>/`).
- **Revalidation scripts** with usage-comment headers in `scripts/model_validation/` for every
  claim (baseline, diagnostics, eval).
- Tick/annotate your items in `doc/TODO.md` (Task 3 section).
- Cite measured numbers, never adjectives.

## 8. Coordination & constraints (hard rules)

- **Work on your own branch** `project/shirt-distribute` (from `project/tendonbot-sim-rl`);
  commit in logical increments and push regularly — Georg merges.
- **Do NOT edit `tasks/manager_based/shared/**`, `shirt_pick/**`, `shirt_present/**`,
  `reach/**`, `shirt_place/**` or other agents' files.** If a shared-code change seems
  unavoidable, implement the minimal override inside `shirt_distribute/` instead and flag it in
  your summary + tracking report.
- One Isaac Sim process per allocated GPU; all GPU work through Slurm.
- Training is authorized. Do not modify robot effort limits or cloth solver parameters without a
  measured justification documented in the config comment.
