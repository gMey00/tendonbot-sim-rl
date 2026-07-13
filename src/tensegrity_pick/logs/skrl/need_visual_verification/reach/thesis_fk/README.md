# Visual verification — THESIS FK grid (iteration 22)

Selected checkpoints from the final thesis dataset (FK full-pose targets, EMA
everywhere, fixed sampler — see `theses_logs/reach/grid/README.md`). Chosen for
**supervisor-video quality**: each demonstrates one headline finding. Play on a
workstation (not Alex — RTX GUI segfaults on the cluster driver).

## IMPORTANT: env vars must match training

Every play command needs the FK-mode env vars, or the targets/actions won't
match what the policy was trained on:

```bash
export REACH_FK_TARGETS=1 REACH_TS_EMA=0.2 REACH_FK_HALF_RANGE=1.0
python scripts/skrl/play.py --task <Play-ID> --num_envs 4 \
    --checkpoint logs/skrl/need_visual_verification/reach/thesis_fk/<dir>/checkpoints/best_agent.pt
```

## Checklist / video priorities

| # | Dir | Play task ID | Extra env vars | Shows / film this |
|---|---|---|---|---|
| 1 | `ur5e_f140_joint_seed42` (3.1 cm/86 % pose) | `Template-Reach-UR5e-F140-Play-v0` | — | **Headline:** true full-pose reach — arm matches position AND orientation of every marker (box-era policies ignored orientation). Film several targets incl. large reorientations. |
| 2 | `wrist_ab_frank_active_seed42` (4.5 cm) | `Template-Reach-UR5e-Frankenstein-Play-v0` | — | **Wrist capability, active:** tensegrity wrist visibly articulates to hit wrist-requiring orientations. Film close-up of the wrist during target changes. |
| 3 | `wrist_ab_frank_locked_seed42` (5.1 cm, pose 35 % vs 57 %) | `Template-Reach-UR5e-Frankenstein-Play-v0` | `REACH_LOCK_WRIST=1` | **Wrist capability, control:** same targets, wrist locked — orientation visibly missed on wrist-requiring targets. **Film as the A/B pair with #2 — the best supervisor demo.** |
| 4 | `kinova_f140_joint_seed123` (2.8 cm) | `Template-Reach-Kinova-F140-Play-v0` | — | Kinova sector fix: clean full-pose reach on the 7-DOF arm (was untrainable pre-fix). |
| 5 | `kinova_f140_osc_seed2` (2.4 cm — tightest task-space cell) | `Template-Reach-Kinova-F140-OSC-Play-v0` | — | OSC exploiting 7-DOF redundancy: smooth task-space motion, no divergence (contrast with the box-era UR-OSC collapses). |
| 6 | `ur5e_f140_ikabs_ema_seed2` (5.1 cm) | `Template-Reach-UR5e-F140-IK-Abs-Play-v0` | — | The EMA fix: visibly smooth IK-Abs motion (pre-EMA policies were jerky — cf. old findings.md notes). |
| 7 | `ur5e_frankenstein_joint_seed123` (4.4 cm) | `Template-Reach-UR5e-Frankenstein-Play-v0` | — | Frankenstein arm in normal thesis-grid use (wrist participating under joint control). |

Notes for the checklist (same procedure as the box-era verification):
- Confirm no startup errors (the Kinova `setLimitParams` popup is fixed —
  iteration 16; its absence on #4/#5 is itself a verifiable fix).
- Watch for: full-length episodes, no oscillation/jerk (EMA variants should be
  visibly smooth), orientation actually pursued (pose-tracking, not just
  position).
- Findings go into `../findings.md` with a `thesis_fk/` prefix, as before.
