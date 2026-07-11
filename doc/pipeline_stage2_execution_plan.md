# Pipeline Stage 2 — Execution Plan (parallel agents, gates, GPU schedule)

Operationalizes the roadmap of
[RESEARCH_REPORT_shirt_sorting_stage2.md](reports/RESEARCH_REPORT_shirt_sorting_stage2.md)
into four parallel coding-agent workstreams, with explicit file ownership,
decision gates, and instructions for Georg (when to invoke which agent, what to
merge, what to visually verify). TODO items live in
[TODO.md § Pipeline Stage 2](TODO.md).

## 0. Feasibility verdicts (checked against the repo, 2026-07-12)

| Report item | Verdict | Evidence / adjustment |
|---|---|---|
| S0 terminal bank | ✅ feasible, cheap | `ShirtPickEnv.snapshot_terminal_states` exists; only a rollout script is missing |
| S1 hem curriculum | ✅ feasible **with bank regen** | current hanging bank has only **19/356 hem-corner-anchored states** (measured) — regenerate region-biased first |
| S2 learned grasp head | ✅ feasible, most design-open | grasp target is already the overridable `shirt_grasp_point_w`; needs a policy-selectable candidate mechanism + offline coverage-lookup table from the study CSVs |
| S3 movable holder / joint policy | ✅ feasible in manager-based | two articulations + concatenated action terms work in one manager-based env — **no DirectMARLEnv rewrite needed** for the recommended path |
| S4 asymmetric AC | ⚠ feasible, unknown cost | skrl separation unverified; **no RSL-RL runner exists in `scripts/`** — fallback costs a new train script |
| S5 grasp gates | ✅ feasible, low cost | touches `shared/cloth_sorting_env.py::_update_grasp` → must be owned by the single shared-owner agent, opt-in flags |
| S6 MAPPO ablation | ⚠ stretch | skrl IPPO/MAPPO plumbed in `train.py` but never run; gymnasium ≥ 1.0 breakage documented |
| S7 distillation, S8 contact pinch | ✅ stretch, bounded | S8 can reuse the robot-free heuristics-study rig |

## 1. Agents, branches, file ownership (hard rules)

All agents branch from `project/tendonbot-sim-rl`; Georg merges. **Nobody edits
another agent's files.** `shared/**` is writable by INF ONLY; task agents that
need a shared change implement a local override in their task package and flag
it (established rule from the Task-2/3 agent prompts).

| Agent | Prompt | Branch | Owns (rw) | GPU |
|---|---|---|---|---|
| **T1** pick grasp head | [agent_prompt_pick_grasp_head.md](agent_prompt_pick_grasp_head.md) | `project/shirt-pick-grasp-head` | `shirt_pick/**`, new rollout/eval scripts, Task-1 bank files | FAPS (bank rollout) → Alex (S2 training) |
| **T2** present coop | [agent_prompt_present_coop.md](agent_prompt_present_coop.md) | `project/shirt-present-coop` | `shirt_present/**`, `generate_hanging_bank.py`, hanging-bank files | Alex (all training) |
| **T3** distribute polish | [agent_prompt_distribute_polish.md](agent_prompt_distribute_polish.md) | `project/shirt-distribute-polish` | `shirt_distribute/**` | Alex |
| **INF** pipeline infra | [agent_prompt_pipeline_infra.md](agent_prompt_pipeline_infra.md) | `project/pipeline-infra` | `shared/**`, `scripts/skrl/train.py` (asym-AC/RSL-RL), eval harnesses | FAPS (smoke only) |

FAPS A6000 rule: **one Isaac process at a time** — before launching, check
`pgrep -f isaac`/`nvidia-smi`; Georg's visual-verification sessions have
priority. All training sweeps go through Slurm on Alex
(`doc/Alex_cluster/alex_quickstart.md`, `tools/train_alex.sh`).

## 2. Phases and gates

```
Phase 1 (all four agents in parallel, ~1 week)
  T1:  S0 terminal bank  ──────────────┐
  T2:  S0 gate/pulse + S1 curriculum ──┤
  T3:  bin-2 + layout randomization ───┼──►  G1 (Georg): merge INF shared lib;
  INF: skrl asym-AC verdict + obs lib  │       visual-verify T2 curriculum best +
       + S5 grasp gates (opt-in) ──────┘       T3 best; T1 bank → T2 seam gate;
                                               decide S3 trigger
Phase 2 (~1–2 weeks)
  T2:  S3 movable holder → single joint policy (on real Task-1 bank)
  T1:  S2 grasp-head training (uses INF keypoint terms)
  T3:  hold-presented gate + real Task-2-bank seam (when available)
  INF: S4 asymmetric-AC retrofit configs + S5 ranking-preservation study
                                        ──►  G2 (Georg): joint policy vs single-
                                             agent deterministic eval; ship-or-
                                             escalate decision (S6 MAPPO only on
                                             measured coordination failure)
Phase 3 (stretch, only what G2 justifies)
  S6 MAPPO ablation (T2)  ·  S8 contact-pinch study (new small agent, FAPS)
  S7 distillation (future work unless time remains)
```

**Gate G1 criteria** (end of Phase 1):
- T2 hem↔hem grasp rate: **> 0.6** → S3 authorized; 0.3–0.6 → iterate curriculum
  once; **< 0.3** → abort to oracle/accessible-corner target (bank 0.679–0.713).
- T1 bank: ≥ 500 presented-terminal states incl. slips; T2 retrain on it must
  not lose > 0.05 present rate vs synthetic bank (else synthetic + slip injection).
- INF: asym-AC verdict (skrl OK / RSL-RL needed) decides Phase-2 training stack.

**Gate G2 criteria** (end of Phase 2):
- Joint policy beats fixed-holder single-agent on deterministic unseen-seed eval
  (coverage AND grasp rate) → it becomes the Task-2 production path; else ship
  single-agent oracle-target and record the negative (thesis-worthy either way).
- S5 ranking preservation: if gated-grasp eval drops success > 0.15 AND reorders
  checkpoints → promote grasp gates into training (new Phase-3 item).

## 3. Instructions for Georg

### Launching (today)

1. Merge/settle the current working tree on `project/tendonbot-sim-rl` (the
   heuristics-study updates), so agents branch from a clean state.
2. Launch all four agents in parallel, each with its prompt file as the task
   description: T2 and T3 on Alex checkouts, T1 and INF on the FAPS server.
   No ordering constraint in Phase 1 — ownership is disjoint.
3. When INF reports the shared observation library + grasp gates ready
   (its first milestone, before its skrl deep-dive): **merge INF early** into
   `project/tendonbot-sim-rl` and tell T1/T2 to rebase — this minimizes
   shared-file drift.

### During phases

- Watch each agent's tracking report (they update as they go, repo rule):
  `doc/reports/{shirt_pick,shirt_present,shirt_distribute}_optimization_tracking.md`
  + INF's new `doc/reports/pipeline_infra_tracking.md`.
- Expect agents to STOP and hand back at their decision gates rather than
  self-authorize expensive retrains — the prompts instruct them to.

### Visual verification duty (the established need_visual_verification loop)

Every agent that produces a candidate checkpoint stages it under
`src/tensegrity_pick/logs/skrl/need_visual_verification/<task>/<run_name>/` with
a `README.md` containing: the exact play command (current-env compatible), the
deterministic eval numbers, and a **"what to look for" checklist** specific to
the change. You play it (`scripts/skrl/play.py … --checkpoint …`, or
`play_zero.py` for setup-only changes), then write **`findings.md`** in the
same folder (free-form; bugs, artifacts, verdict). Agents are instructed to
treat `findings.md` as blocking input for their next iteration — the same loop
that caught the early-gripper-close hack and the hem↔hem geometry issues.

Suggested standing checklist (agents extend it per change):
- Grasp: does the gripper visibly reach and close AT the cloth (no air-grasp
  attach artifacts, no early-close cruising)?
- Presentation: taut horizontal chord at holder height, garment hanging below,
  no bunching/hiding; camera view unoccluded by the arm.
- Holder (S3): repositioning stays smooth, no cloth tearing/overstretch
  (tautness ≤ 1.15), no oscillation between the arms.
- Releases/drops: shirt falls INTO the drum, arm returns without dragging.

### GPU budget guardrails

- Alex: cap Phase-1 sweeps at ~3 seeds × 2 configs per agent before a gate
  review; Phase-2 joint-policy runs are the big-ticket item (10–20 GPU-days
  per report) — schedule after G1, not before.
- FAPS: T1's bank rollout and INF smoke tests are minutes-to-hours; keep the
  GPU free for your play sessions otherwise.

## 4. Cross-agent dependencies (who waits on whom)

| Dependency | Producer → Consumer | Phase |
|---|---|---|
| Real Task-1→2 terminal bank | T1 → T2 | G1 seam gate |
| Observation library terms (keypoints, visibility, noisy grasp flag) | INF → T1, T2 | merge early in Phase 1 |
| Grasp gates (opt-in) + ranking harness | INF → all eval | Phase 2 |
| First-region→expected-coverage lookup table (offline, from study CSVs) | T1 (builds) — study data is read-only input | Phase 1 |
| Real Task-2→3 terminal bank | T2 → T3 | Phase 2, after S3 settles |
| Asym-AC verdict (skrl vs RSL-RL) | INF → T1, T2 training configs | G1 |

Read-only for everyone: `doc/reports/present_heuristics_study.md` + its data
(`doc/reports/data/present_*.csv`, `present_markers.pt`, oracle policy JSON) —
the calibration ground truth all reward/curriculum choices cite.
