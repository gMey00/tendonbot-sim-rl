# THESIS dataset — F140 reach grid (iteration 22, 2026-07-13)

Final consistent dataset of the action-space study: **24 variants × 5 seeds
{0,1,2,42,123} = 120 runs** (`grid/<variant>/<run>/`), plus the tensegrity-wrist
matched-pair A/B (**4 configs × 3 seeds = 12 runs**, `../wrist_ab/<config>/`).
Each run: tfevents, `params/`, `checkpoints/{best_agent,agent_100000}.pt`.
`MANIFEST.txt` maps every run to its task/config/seed.

**Locked config (identical for all runs):**
- Targets: per-robot FK-sampled full poses — `REACH_FK_TARGETS=1` (fixed
  sampler: kinematic self-collision filter + floor clearance links≥5cm /
  EE≥30cm), Kinova sampling sector `REACH_FK_HALF_RANGE=1.0`.
- Actions: EMA α=0.2 on every action space (`REACH_TS_EMA=0.2` for
  IK-Rel/IK-Abs/OSC; built into the joint action).
- PPO/skrl, 100k timesteps, 4096 envs; reward = IsaacLab reference reach.

**To play a checkpoint with the same target distribution/action smoothing**
(the env vars must match training or the behaviour/targets differ):

```bash
REACH_FK_TARGETS=1 REACH_TS_EMA=0.2 REACH_FK_HALF_RANGE=1.0 \
  python scripts/skrl/play.py --task <Task-Play-v0> --num_envs 4 \
  --checkpoint <run>/checkpoints/best_agent.pt
```
(`REACH_FK_HALF_RANGE` only affects the Kinova variants; harmless elsewhere.
`REACH_TS_EMA` only affects task-space variants.)

Results tables & analysis: `doc/reports/tracking/reach_optimization_tracking.md`
iterations 17–22 (iteration 22 = this dataset); summary in the reach README
and `action_spaces.md`.
