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

## M2 — camera-realistic observation library ⏳

(in progress)

## M3 — grasp-fidelity gates ⏳

(pending)

## M4 — ranking-preservation harness ⏳

(pending)
