# Workstation agent prompt — Reach thesis visual verification

> Paste everything below the line into a fresh agent running **on the FAPS
> workstation** (the one with a display + a non-RTX-segfaulting GPU driver). It
> walks you (with the human at the keyboard for the GUI) through visually
> verifying and recording the final reach policies for the thesis report.

---

You are a coding agent on the FAPS workstation helping verify trained RL reach
policies in the Isaac Sim GUI and capture short videos for a thesis/supervisor
report. Work **with** the human: you run the commands, they watch the viewport
and record. Be concise; confirm each step worked before moving on.

## 0. Context (what you're verifying)

The `reach` task is a per-robot controller-hardening study (6 arms × 4 action
spaces). The final "thesis" policies were trained on **FK-sampled reachable
full-pose targets** with **EMA-smoothed actions**. Unlike older "box" policies,
these actually track target **orientation**, not just position. Full context:
`.../reach/README.md` (§Purpose + thesis results table) and the tracking log
`doc/reports/reach_optimization_tracking.md` (iterations 16–22). The curated
checkpoints and a per-variant checklist live in
`src/tensegrity_pick/logs/skrl/need_visual_verification/reach/thesis_fk/`
(`README.md` there = the authoritative checklist; this prompt is the how-to).

## 1. Setup

```bash
cd /home/robot/studentische-arbeiten          # workstation repo path
git fetch origin
git checkout project/tendonbot-sim-rl          # the base branch — has the reach work merged
git pull --ff-only
git log --oneline -1                           # expect: 02c1f92 (or later) "reach: thesis dataset ..."
git lfs pull                                    # ensure res/ state banks are real files, not pointers

source .config/env_vars.sh                      # auto-detects the faps_workstation profile
source /home/robot/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
unset VIRTUAL_ENV
```

Verify the checkpoints are present (real files, ~95 KB each, not LFS pointers):

```bash
ls -la src/tensegrity_pick/logs/skrl/need_visual_verification/reach/thesis_fk/*/checkpoints/best_agent.pt
```

If any is a ~130-byte pointer stub, run `git lfs pull` again (they should be
plain git blobs, not LFS — if they look like `version https://git-lfs...`, tell
the human; something is misconfigured).

## 2. THE critical gotcha — env vars must match training

Every play command **must** export the same env vars the policy was trained
with, or the environment (targets, action dimensions) won't match the
checkpoint and you'll get a shape error or nonsense behaviour:

```bash
export REACH_FK_TARGETS=1 REACH_TS_EMA=0.2 REACH_FK_HALF_RANGE=1.0
```

- `REACH_FK_TARGETS=1` — reachable FK full-pose targets (not the legacy box).
- `REACH_TS_EMA=0.2` — EMA-smoothed task-space actions (harmless on the joint
  variants; required for the IK/OSC ones).
- `REACH_FK_HALF_RANGE=1.0` — Kinova sampling sector (ignored by the UR arms).
- **One extra:** for `wrist_ab_frank_locked_seed42` **also** export
  `REACH_LOCK_WRIST=1` (its action dim is 6, not 8 — without this it will
  crash). Do **not** set it for any other checkpoint.

Universal play command (num_envs 4 gives a few simultaneous targets to watch):

```bash
python scripts/skrl/play.py --task <PLAY_ID> --num_envs 4 \
    --checkpoint logs/skrl/need_visual_verification/reach/thesis_fk/<DIR>/checkpoints/best_agent.pt
```

(`play.py` opens the GUI by default on the workstation. If it runs headless,
add nothing — it should show a window; if not, tell the human.)

## 3. Pre-flight (do this once, first)

Run checklist item #1 below (`ur5e_f140_joint_seed42`). Confirm:
- The GUI window opens, the arm moves, episodes run full length (no instant
  reset), and **goal-pose axis markers** appear at the targets.
- **If it errors with `FileNotFoundError: ... frame_prim.usd`** — the goal-marker
  UI asset isn't resolving. This is a debug-vis asset-root issue, not a policy
  problem. Fixes, in order of preference:
  1. Ensure Isaac's asset root is configured so `Isaac/Props/UIElements/frame_prim.usd`
     resolves (check `ISAAC_NUCLEUS_DIR` / local Isaac assets; a full Isaac Sim
     install has it). Re-run.
  2. If you can't resolve the asset quickly, the human can still verify motion
     without the goal markers: tell them the fix, and (only if they accept losing
     the visual target markers) you may edit the play run to set the command's
     `debug_vis = False`. Prefer keeping markers on — they make the video.
- Report what you saw before continuing.

## 4. Verification checklist (7 checkpoints)

For each: run it, watch a handful of target changes, note pass/fail + one line of
observation, and — for the ⭐ items — help the human record a short clip. Details
and the "what to look for" per item are in the sibling `README.md`; the mapping:

| # | DIR | PLAY_ID | extra env | ⭐film? | look for |
|---|---|---|---|---|---|
| 1 | `ur5e_f140_joint_seed42` | `Template-Reach-UR5e-F140-Play-v0` | — | ⭐ | Headline: arm matches **position AND orientation** of each marker, incl. large reorientations. |
| 2 | `wrist_ab_frank_active_seed42` | `Template-Reach-UR5e-Frankenstein-Play-v0` | — | ⭐ | Tensegrity **wrist visibly articulates** to hit orientations. Close-up on the wrist. |
| 3 | `wrist_ab_frank_locked_seed42` | `Template-Reach-UR5e-Frankenstein-Play-v0` | `REACH_LOCK_WRIST=1` | ⭐ | Same targets, **wrist locked** → orientation visibly missed. **Film as an A/B pair with #2 — the best supervisor demo.** |
| 4 | `kinova_f140_joint_seed123` | `Template-Reach-Kinova-F140-Play-v0` | — |  | Clean full-pose reach on the 7-DOF arm (was untrainable before the fix). |
| 5 | `kinova_f140_osc_seed2` | `Template-Reach-Kinova-F140-OSC-Play-v0` | — |  | OSC using 7-DOF redundancy: smooth task-space motion, **no divergence**. |
| 6 | `ur5e_f140_ikabs_ema_seed2` | `Template-Reach-UR5e-F140-IK-Abs-Play-v0` | — |  | The EMA fix: **visibly smooth** IK-Abs motion (pre-EMA was jerky). |
| 7 | `ur5e_frankenstein_joint_seed123` | `Template-Reach-UR5e-Frankenstein-Play-v0` | — |  | Frankenstein arm in normal thesis-grid use. |

Also confirm (a passive check, worth a sentence in findings): **no Kinova
`setLimitParams` startup error / popup** on #4/#5 — its absence is itself a
verified fix (tracking log iteration 16).

## 5. Recording the videos

Best quality for a supervisor report = screen-record the Isaac Sim viewport
while the policy runs (the human drives the window; use the OS screen recorder,
e.g. OBS / `wf-recorder` / built-in). Frame the arm + the goal markers; let it
show 3–5 target changes (~15–30 s). For the **#2 vs #3 wrist A/B**, record the
same duration for both with an identical camera angle so they cut together.

Alternative (headless, deterministic clip): add
`--video --video_length 400 --headless` to the play command — Isaac Lab writes
an mp4 under the run's `videos/` dir. Lower visual polish, but reproducible.

## 6. Findings + sign-off

Append notes to `.../need_visual_verification/reach/findings.md` with a
`thesis_fk/` prefix per item (pass/fail + one observation each), the same format
as the existing box-era entries. When done, give the human a 5-line summary:
which items passed, where the recordings are saved, and any surprises.

## Guardrails
- One Isaac Sim process at a time (close the GUI before the next run).
- Don't retrain, re-checkpoint, or edit committed configs — this is verification
  only. The only permitted temporary edit is toggling `debug_vis` per §3, and
  only with the human's OK.
- If a checkpoint's action-dim mismatches the env (shape error), you almost
  certainly forgot an env var from §2 (esp. `REACH_LOCK_WRIST=1` for #3).
