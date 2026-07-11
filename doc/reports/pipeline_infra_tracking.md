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

## M3 — grasp-fidelity gates ⏳

(pending)

## M4 — ranking-preservation harness ⏳

(pending)
