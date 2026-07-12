# Pipeline Infra (INF) — Tracking Report

Stage-2 agent **INF** (branch `project/pipeline-infra`): observation contract,
asymmetric actor–critic, grasp-fidelity gates. Task prompt:
[agent_prompt_pipeline_infra.md](../agent_prompt_pipeline_infra.md); roadmap
stages S4 + S5 of
[pipeline_stage2_execution_plan.md](../pipeline_stage2_execution_plan.md).

Milestones: **M1** skrl asym-AC verdict → **M2** camera-realistic observation
library → **M3** grasp-fidelity gates (opt-in) → **M4** ranking-preservation
harness. Georg: please merge into `project/tendonbot-sim-rl` after each
milestone section below flips to ✅ (T1/T2 rebase on M2/M3).

---

## M1 — skrl 2.1.0 asymmetric actor–critic verdict ✅ (2026-07-12)

**VERDICT: SEPARATED — skrl 2.1.0 + the installed Isaac Lab wrapper implement
true asymmetric actor–critic. No RSL-RL fallback is needed.** A manager-based
env that defines a `"critic"` observation group gets it routed, end to end,
into the value network only; the policy network never sees it. Phase-2
training configs for T1/T2 can use the pattern below with the stock
`scripts/skrl/train.py` (no train-script changes required).

### Evidence

1. **Static trace** (installed sources, not docs):
   - `skrl/envs/wrappers/torch/isaaclab_envs.py::IsaacLabWrapper` —
     `state_space` ← `single_observation_space["critic"]`, `state()` returns
     the flattened critic tensor refreshed on every `step()`/`reset()`.
   - `skrl/trainers/torch/base.py` + `sequential.py` — the single-agent train
     AND eval loops fetch `env.state()` each step and pass it into
     `agent.act(...)` / `agent.record_transition(...)` (this is the skrl-2.x
     observation/state refactor; the 1.x aliasing the repo TODO flagged is
     gone).
   - `skrl/agents/torch/ppo/ppo.py` — memory holds a separate `states`
     tensor; every model call receives
     `{"observations": ..., "states": ...}`.
   - `skrl/utils/model_instantiators/torch/common.py` — yaml `input: STATES`
     compiles to `inputs["states"]` sized by `num_states`;
     `input: OBSERVATIONS` to `inputs["observations"]`.
2. **Runtime probe** —
   [check_skrl_asymmetric_ac.py](../../src/tensegrity_pick/scripts/model_validation/check_skrl_asymmetric_ac.py)
   (reach task, 8 envs, 64 trainer steps = 4 PPO updates; preprocessors off so
   raw tensors reach the models). Injected a `critic` group = policy terms +
   a 7-wide constant probe (42.0) and spied on both models' `act()`:

   ```
   spaces          : policy 22  critic 29
   model act calls : policy 80  value 84
   value inputs    : state dims seen {29}, probe cols == 42.0 on all calls: True
   policy inputs   : obs dims seen {22}, free of probe value: True
   first Linear in : policy 22 (exp 22), value 29 (exp 29)
   VERDICT: SEPARATED — asymmetric AC works
   ```

   The value net's first Linear materialized at the critic width and its raw
   input carried the probe constant on all 84 calls (rollout + GAE
   bootstrap + minibatch updates); the policy net stayed at the policy width
   and never saw the probe.

### Working config pattern (copy this, T1/T2)

**Env cfg** — add a `critic` group next to `policy` in the task's
`ObservationsCfg` (group name must be exactly `"critic"`; Isaac Lab exposes it
as `num_states`/`state_space`, the skrl wrapper picks it up):

```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):          # camera-realistic actor terms only
        ...                             # (M2 library terms go here)

    @configclass
    class CriticCfg(ObsGroup):          # privileged: policy terms + tautness,
        ...                             # true attachment state, particle field
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
```

**Agent yaml** — separate models; value reads `STATES`:

```yaml
models:
  separate: True            # REQUIRED for asym (shared net can't split inputs)
  policy:
    class: GaussianMixin
    network:
      - name: net
        input: OBSERVATIONS # actor: camera-realistic group
        layers: [256, 128, 64]
        activations: elu
    output: ACTIONS
  value:
    class: DeterministicMixin
    network:
      - name: net
        input: STATES       # critic: privileged group
        layers: [256, 128, 64]
        activations: elu
    output: ONE

agent:
  class: PPO
  observation_preprocessor: RunningStandardScaler   # BOTH keys — see gotcha
  observation_preprocessor_kwargs: null
  state_preprocessor: RunningStandardScaler
  state_preprocessor_kwargs: null
  ...
```

### Gotchas (each verified in the installed source)

- **`state_preprocessor` remap:** `Runner._check_cfg_compatibility` treats a
  yaml that has `state_preprocessor` but NO `observation_preprocessor` as a
  legacy skrl-1.x config and silently **remaps it to
  `observation_preprocessor`, deleting the state one** (all current repo
  yamls rely on this shim — they are legacy-style and unaffected). An
  asymmetric yaml must set **both** keys explicitly, as above.
- `separate: False` cannot express different inputs per role — asymmetric
  configs must use `separate: True`.
- Only the group names `"policy"` and `"critic"` are wired; other group names
  are ignored by the skrl wrapper.
- Play/eval path (`Runner`-based `play.py`, trainer `eval()`) also fetches
  `state()` — checkpoints trained asymmetrically play fine, but the env must
  keep providing the critic group (it's part of the env cfg, so it does).
- rsl_rl is **not installed** in `env_isaaclab`; the fallback would cost an
  install + new runner scripts. Not needed on this verdict — revisit only if
  the pattern breaks in a future skrl upgrade.

---

## M2 — camera-realistic observation library ✅ (2026-07-12)

New observation terms in
[shared/cloth_sorting_mdp.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shared/cloth_sorting_mdp.py)
(pure functions on env attrs, zeros-fallback shape-probe guard — the
established pattern), backed by new pure-torch helpers in
[shared/cloth_metrics.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shared/cloth_metrics.py)
so everything statistical is offline-testable:

| Term | Shape | Real-perception source it models |
|---|---|---|
| `keypoints_with_visibility(noise_std, recall)` | `[N, 48]` (12 × xyz + 12 flags) | keypoint detector on the 12 study region landmarks (`present_markers.pt`); visibility = the study's two-sided depth-layer logic; occluded/undetected → zeroed pos + flag 0 |
| `surface_point_cloud(num_points, visible_only)` | `[N, num_points × 3]` | segmented down-sampled point cloud (UniFolding/VCD input); deterministic given seed, no fake zero-pad points (wrap-around) |
| `coverage_from_mask()` | `[N, 1]` | segmentation-mask area ÷ flat area — literally `silhouette_coverage`, equivalence asserted in tests + live smoke |
| `grasp_active_noisy(flip_prob, latency_steps)` | `[N, 1]` | gripper-current grasp estimator (2F-140 exposes motor current): delayed + occasionally flipped binary |

- **Noise models default OFF** (identical to clean terms until a config opts
  in). Keypoint noise calibrated to Lips et al. 2024 (74 % mAP → recall ≈
  0.74 dropout; ~9 px ≈ 1–2 cm → `noise_std` 0.01–0.02), cited in the
  docstring. Draws use the run seed (deterministic).
- **Isaac Lab noise hooks:** the standard `ObsTerm(noise=...)` corruption
  hook works ON TOP of these terms for plain Gaussian obs noise, but the
  keypoint dropout must zero position AND flag *together* (a detector miss),
  and the grasp-flag latency needs cross-step state — both structurally
  beyond the per-term additive-noise hook, hence term-level parameters.
- **Privileged terms stay privileged:** tautness/stretch ratio is NOT
  camera-observable — module docstring forbids it in actor groups; keep it
  in `critic` groups and rewards (report §2-Q3).
- New pure helpers: `per_particle_visibility` (factored out of
  `two_sided_visible_fraction` — behavior regression-tested),
  `depth_layer_count_in_ball` (also serves the M3 gates),
  `keypoint_detector_noise`, `downsample_masked_points`,
  `load_present_markers` (repo-root-relative, cached).

**Validation**
- Offline: new
  [test_camera_obs.py](../../src/tensegrity_pick/scripts/model_validation/test_camera_obs.py)
  — 26/26 PASS (visibility ground truth incl. 3-layer occlusion, layer
  counting, dropout rate 0.741 vs 0.74 target, seed determinism, wrap-around
  sampling, marker contract, coverage == independent mask-area
  reimplementation). Existing `test_cloth_metrics.py` still ALL PASS.
- Live (4 envs, shirt_pick):
  [check_camera_obs_terms.py](../../src/tensegrity_pick/scripts/model_validation/check_camera_obs_terms.py)
  — 12/12 PASS, including the ObservationManager shape-probe path (terms
  injected into a runtime cfg copy) and `coverage_from_mask ==
  silhouette_coverage` on live cloth (0.1166 == 0.1166).
- Regression: `test_shirt_fixes.py` 5/5 after the refactor.

**Usage snippet for T1/T2/T3** (your task's `ObservationsCfg`; INF does not
edit task cfgs):

```python
from ...shared import cloth_sorting_mdp as shared_mdp

@configclass
class PolicyCfg(ObsGroup):   # camera-realistic actor
    keypoints = ObsTerm(func=shared_mdp.keypoints_with_visibility)
    # opt-in detector noise for DR runs:
    # keypoints = ObsTerm(func=shared_mdp.keypoints_with_visibility,
    #                     params={"noise_std": 0.015, "recall": 0.74})
    cloud = ObsTerm(func=shared_mdp.surface_point_cloud,
                    params={"num_points": 64})
    coverage = ObsTerm(func=shared_mdp.coverage_from_mask)
    grasp = ObsTerm(func=shared_mdp.grasp_active_noisy)
    # NO tautness here — critic group only.
```

## M3 — grasp-fidelity gates ⏳

(in progress)

## M3 — grasp-fidelity gates ✅ (2026-07-12)

Opt-in evaluation-mode gates on the deterministic attachment grasp in
[shared/cloth_sorting_env.py](../../src/tensegrity_pick/source/tensegrity_pick/tensegrity_pick/tasks/manager_based/shared/cloth_sorting_env.py)
(`GraspFidelityCfg` + `_update_grasp` extensions). **Both flags default OFF**;
with flags off the grasp path is bit-identical (no RNG consumed, no extra
cloth reads — stochastic draws use a dedicated seeded generator, so even
enabling them never perturbs env determinism).

- **Pre-condition gate** (deterministic, per attach-eligible step):
  (a) approach axis within `max_approach_angle_deg` (60°) of the local cloth
  surface normal (PCA over the weld-ball particles); (b) finger tip
  ≥ `min_tip_clearance_m` above the belt (no rigid pinch-jam); (c) ≤
  `max_layers` (2) separated depth layers in the weld ball along the approach
  axis (`cloth_metrics.depth_layer_count_in_ball`).
- **Stochastic misgrasp/slip gate** (one draw per grasp *attempt*; a failed
  attempt latches until the gripper reopens, so holding "close" doesn't
  re-roll every step): attach failure keyed to grasp location via the study's
  `BORDER_DIST` (border ≤ 3 cm → `p_misgrasp_border` 0.16, else mid-panel
  0.22; > `max_layers` → ≥ 0.50) — calibrated to DRAPER / DeepCloth-ROB
  (arXiv 2409.15159: real Franka parallel-jaw misgrasp 16–22 %, edges/corners
  the easy case). During hold, slip when the hang-weight load proxy
  (garment mass × fraction below the grasp × (g + upward tip acceleration))
  exceeds a per-attempt capacity draw N(4.0, 1.5) N clamped ≥ 0.5 N.
- Every number is a cfg field; opt-in from an eval script is two lines:

  ```python
  env.unwrapped.grasp_fidelity.precondition_gate = True
  env.unwrapped.grasp_fidelity.stochastic_gate = True   # seeds itself (cfg.seed)
  ```

**Robot-free rig validation**
([check_grasp_gates.py](../../src/tensegrity_pick/scripts/model_validation/check_grasp_gates.py),
study-harness reuse: hanging-bank restores + scripted pinches through the real
`_apply_grasp_gates`/`_apply_slip_gate` code; 96 attempts per region per
sweep) — ALL PASS:

- *Flags-off inertness:* zero gate events while stepping; `test_shirt_fixes.py`
  stays 5/5 (below).
- *Stochastic gate only* (top-down pinch, per-region misgrasp rate vs the
  border/mid mixture expected from the region's measured border fraction —
  max deviation 0.082 ≈ 2σ at n = 96):

  | region | border frac | misgrasp | | region | border frac | misgrasp |
  |---|---|---|---|---|---|---|
  | collar | 0.95 | 0.14 | | side | 0.01 | 0.28 |
  | shoulder | 0.09 | 0.14 | | belly | 0.00 | 0.30 |
  | sleeve | 0.36 | 0.18 | | hem_corner | 0.53 | 0.16 |
  | chest | 0.00 | 0.24 | | hem_c | 0.67 | 0.18 |

- *Pre-condition gate:* a straight-DOWN pinch on a hanging garment is blocked
  0.89 on average (panels hang vertically — the alignment gate correctly
  rejects a physically hopeless approach), while a surface-ALIGNED pinch at
  the same points passes 1.00 (block 0.00). Multi-layer detections are ≈ 0 on
  this relaxed single-hang bank (layers in contact merge — the documented
  lower-bound caveat; crumpled/folded states are where (c) bites).
- *Slip gate:* static hang slip rate 0.208 vs the analytic Φ((mg − μ)/σ) =
  0.241 (0.30 kg shirt → 2.94 N static load); an accelerating upward pull
  produced 118 additional slips (dynamic-load path works). A two-stage
  validity buffer avoids a spurious acceleration spike on the first steps
  after reset.

**Caveats for task agents**
- `ShirtDistributeEnv` overrides `_update_grasp` wholesale (reset-anchor
  fix) and therefore BYPASSES the gates — T3's episodes start already
  grasped, so attach gating is moot there; if T3 wants slip-under-load, call
  `self._apply_slip_gate(tip)` from its override (one line).
- The layer estimate shares `per_particle_visibility`'s contact-merge
  limitation: it is a lower bound; expect it to matter on crumpled piles
  (shirt_pick bank) rather than relaxed hangs.

## Resolved false alarm: "shirt_pick playback regression" was a stale checkpoint

First M4 runs used pick run `2026-07-03_02-47-24` (agent_16000/20000, the
tracking report's Phase-2 eval table: present 0.729/0.844) and got
present_rate = 0.000. A worktree bisect showed branch point `7f89063` and the
current branch give **bit-identical** results (present 0.000, grasp 0.812,
mean|act| 1.028) — so (a) the INF/T1 Stage-2 changes are provably invisible
in playback, and (b) nothing regressed tonight. The actual cause:
`PRESENTATION_POS` moved in Phase 3 ((0.15, 0.50, 1.10) → (0.15, 0.90, 1.60))
and the Phase-2 checkpoints carry the shirt to the OLD pose — the windowed
latch correctly scores them 0. The production Phase-3 run
`2026-07-03_19-33-17_ppo_torch_seed3` (agent_12000: present 1.000/grasp
1.000) is the valid stage-1 baseline; the M4 study below uses its
checkpoints. Lesson recorded: always rank checkpoints of the CURRENT task
definition; the Phase-2 table in shirt_pick_optimization_tracking.md is
historical.

## M4 — ranking-preservation harness ✅ (2026-07-12)

[eval_grasp_fidelity.py](../../src/tensegrity_pick/scripts/model_validation/eval_grasp_fidelity.py):
one boot per task, ≥ 2 checkpoints, each evaluated deterministically
(mean actions, seed 7, 96 episodes, identical reset draws) under
(a) idealized and (b) gated grasp; prints per-checkpoint success, gate-event
counts, and the pre-registered verdict (promote gates into training ONLY if
success drops > 0.15 absolute AND the ranking reorders — report §2-Q5).
Implementation note: ALL env interaction (reset included) must run inside
`torch.inference_mode()` — a second reset outside it trips "Inplace update to
inference tensor" (first multi-run harness in the repo to hit this).

### shirt_pick (production stage-1 run `2026-07-03_19-33-17_seed3`)

| checkpoint | ideal present | gated present | drop | misgrasps | slips | gated grasp_rate |
|---|---|---|---|---|---|---|
| agent_8000 | 0.990 | 0.000 | 0.990 | 72 | 204 | 0.750 |
| agent_12000 | 0.990 | 0.000 | 0.990 | 57 | 208 | 0.750 |

- The idealized column reproduces the stage-1 selection numbers (0.979–1.000
  in the pick tracking report) — **the M1–M3 shared-code changes are invisible
  in checkpoint playback** (also confirmed bit-identically vs the branch
  point, see the false-alarm section above).
- **Ranking preserved, verdict: KEEP gates evaluation-only** (drop 0.99 >
  0.15, but no reorder — the rule requires both).
- Reading the gated 0.000 correctly: the policies re-grasp resiliently under
  gates (444 attempts/96 eps, grasp_rate still 0.75) but ~2 slips/episode
  break the windowed stable-hold latch. A whole-garment carry loads the pinch
  with the full 0.30 kg (2.94 N static + swing accelerations) against a
  N(4.0, 1.5) N capacity draw → ≥ 1 slip/episode is near-certain. The
  DRAPER-calibrated *misgrasp* side is well-grounded; the *slip capacity* is
  our estimate — before reading gated numbers as absolutes, calibrate
  `slip_capacity_mean_n/std_n` against a real 2F-140 cloth-pinch pull test
  (the 2F-140 commands 10–125 N grip force; fabric pinch capacity is the
  unknown). Every knob is a cfg field.

### shirt_present (UR5e run `2026-07-08_16-03-46_seed43`) — non-informative, env drift

| checkpoint | ideal present | gated present | ideal grasp | gated grasp | misgr | slips |
|---|---|---|---|---|---|---|
| best_agent | 0.000 | 0.000 | 0.094 | 0.042 | 2 | 1 |
| agent_104000 | 0.000 | 0.000 | 0.073 | 0.031 | 4 | 5 |

These checkpoints predate the 2026-07-10/11 holder-grip rework (holder anchor
→ 1.60, IK closed-fingertip grip, arm stiffening — commit `1dd5a3b`): on
today's env their idealized grasp_rate is 0.07–0.09 vs the 0.73 measured on
their training-era env (need_visual_verification README). With no
current-env-trained present checkpoint pair available yet, the table only
demonstrates the harness runs end-to-end on the task; **rerun once T2 lands
Phase-1 checkpoints** (one command, usage in the script header). Consistent
detail: the precondition gate blocks 68/77 attempts — misaligned approaches
on an env the policy wasn't trained for.

### Standing recommendation

Gates stay **evaluation-only** (G2 decision input). Two follow-ups for
Phase 2: (1) calibrate slip capacity physically, (2) re-run both studies on
T1/T2's Phase-2 checkpoints — the harness + this section are the template.
