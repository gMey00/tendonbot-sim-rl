# Reach — Visual Verification Findings

Notes from live playback sessions (see `README.md` for the checklist and
per-variant priorities). Logged as issues come up, referenced by variant dir.

> **Resolution status (2026-07-07, iteration 16 in the
> [tracking log](../../../../../../../doc/reports/reach_optimization_tracking.md)):**
> - **Kinova startup errors: FIXED.** Root cause confirmed (USD limits baked by
>   PhysX at `sim.reset()`, before the reset-mode clamp). Fixed with a before-bake
>   spawner wrapper; verified 8→0 `setLimitParams` errors, 0 NaN, identical baked
>   limits, across all 4 action spaces + both Kinova arms.
> - **Orientation / unused wrist: INVESTIGATED.** An opt-in, position-gated
>   orientation-refinement reward (redundant arms only, off by default) recovers
>   orientation where the redundancy is *controller-accessible* — `kinova_f140_osc`
>   9 %→75 % orient for +0.3 cm — and is safely inert where it is not
>   (`ur5e_frankenstein_osc`, wrist excluded from OSC: 0 %→0 %, position kept).
>   See the tracking log Part B for the full before/after table and per-variant
>   root-cause.
> - **Multi-seed follow-up (2026-07-08, iteration 17): OSC is only seed-robust on
>   the redundant Kinova arms.** A 5-seed grid revealed the single-seed(42)
>   "OSC passes" was luck — all four *UR*-OSC variants are bimodal (converge on
>   some seeds, diverge 20–60 cm / collapsed episodes on others). The bimodal
>   approach behaviour the visual check flagged on `ur10_frankenstein_osc` is
>   this instability, not a one-off seed outlier. Kinova-OSC is tight across
>   seeds. Non-OSC spaces are stable.

## Cross-cutting: startup errors on Kinova variants

- **Kinova arm checkpoints throw PhysX joint-limit errors at startup**, and
  one surfaces as a GUI notification popup: `PxArticulationJointReducedCoordinate::setLimitParams()
  only supports limit angles in range [-2Pi, 2Pi] for joints of type
  PxArticulationJointType::eREVOLUTE`. Fires 9 times in a row right after
  the checkpoint loads. Seen identically on both `kinova_frankenstein_osc`
  and `kinova_f140_osc` — likely a Kinova joint-limit config issue (some
  joint's limit range exceeds ±2π), not seed- or controller-specific. Did
  not visibly affect playback behavior in either case, but should be
  checked against the Kinova USD/joint config since it's noisy and pops a
  user-facing error dialog.

## kinova_frankenstein_osc (Priority 1, seed42, best_agent.pt)

- Episodes run full-length and reach the target position — no early
  termination, matches the expected fix (before: collapsed in ~3 steps).
- **Tensegrity wrist unused**: the 2-DOF compliant wrist is held straight
  throughout: the policy isn't actuating it at all, not just resting near
  neutral.
- **Orientation not pursued**: for sampled goals whose orientation was also
  geometrically reachable (not just position), the policy made no visible
  effort to align to it — consistent with position being the sole reward
  term, but worth flagging since a reachable orientation was left on the
  table rather than attempted.
- Same Kinova joint-limit startup errors as noted in "Cross-cutting" above
  (9x `setLimitParams()` PhysX error, one as GUI popup).

## ur5e_f140_osc (Priority 1, seed42, best_agent.pt)

- Same orientation finding as above: no visible effort to match reachable
  target orientations, position-only tracking. (Tensegrity-wrist finding is
  N/A here — this is the rigid F140-gripper arm, no 2-DOF wrist.)

## ur10_frankenstein_osc seed42 (Priority 1, best_agent.pt)

- Same wrist-unused and orientation-not-pursued findings as
  `kinova_frankenstein_osc` / `ur5e_frankenstein_osc` above.
- **Inconsistent approach behavior**: most target positions are reached
  fine, but for some targets the arm just doesn't approach the target at
  all. Consistent with the README's flagged "~20 cm offset, seed-unlucky
  local optimum" — but worth noting the failure looks bimodal (mostly fine
  vs. no attempt) rather than a uniform bias across all targets. Compare
  against seed 7 / seed 123 to see if this bimodality is seed-specific.

## ur10_frankenstein_osc seed7 (Priority 1, best_agent.pt)

- All approaches reach cleanly — no bimodal failure like seed42. Confirms
  seed42's occasional non-approach was a seed-specific local optimum, not a
  config-wide issue.
- Same wrist-unused and orientation-not-pursued findings as the other
  Frankenstein OSC variants above.

## kinova_f140_osc (Priority 2, seed42, best_agent.pt)

- Matches expected clean/tight reach (3.7 cm / 87%).
- **Orientation-seeking, unlike the Frankenstein variants**: this 7-DOF
  redundant arm visibly attempts to match the reachable target orientation
  much more than any of the Priority 1 checkpoints — not perfect, doesn't
  quite converge rotationally, but a clear behavioral difference from the
  "position only, no attempt" pattern seen above.
- Same Kinova joint-limit startup errors as noted in "Cross-cutting" above
  (9x `setLimitParams()` PhysX error, one as GUI popup).

## ur10_f140_osc (Priority 2, seed42, best_agent.pt)

- Positioning matches expectations (3.1 cm / 91%). Orientation tracking is
  worse than `kinova_f140_osc` — back to little/no visible effort toward
  reachable target orientations, closer to the Frankenstein-variant pattern
  than the Kinova one. Orientation-seeking behavior seen on
  `kinova_f140_osc` doesn't carry over to this rigid non-redundant arm
  (makes sense: no redundant DOF to use for orientation without trading off
  position).

## ur10_frankenstein_ikabs (Priority 2, seed42, best_agent.pt)

- Position matches expectations (smooth approach, no jerkiness/overshoot).
- **Wrist behavior differs from the OSC variants**: here the tensegrity
  wrist isn't held straight — the policy just lets it **drop/hang
  passively** rather than actively controlling it. Distinct failure mode
  from the OSC checkpoints (which hold it straight); both amount to "wrist
  not being actuated," but this one looks like it's not being commanded at
  all rather than commanded to a fixed pose.

## kinova_f140_ikabs (Priority 2, seed42, best_agent.pt)

- Confirms it's the weakest checkpoint in the grid (67%): movement is
  **jerky**, and position doesn't always converge to the marker (matches
  the README's "hovers/overshoots" concern).
- Orientation tracking is a bit better than the pure position-only variants
  (consistent with `kinova_f140_osc`'s redundant-arm orientation-seeking),
  but still doesn't converge well.
- Same Kinova joint-limit startup errors as noted in "Cross-cutting" above
  (9x `setLimitParams()` PhysX error, one as GUI popup).

## ur10_f140_ikabs (Priority 2, seed42, best_agent.pt)

- Settles on position as expected (4.7 cm / 85%), a bit jerky like the
  other IK-Abs checkpoints. Same orientation-not-pursued pattern as the
  other rigid non-redundant-arm variants.

## ur5e_f140 (Priority 3, seed42, best_agent.pt — reference/joint-control, unmodified)

- Position matches expectations (tightest in the grid), a bit jerky though.
- **New orientation pattern**: it stably aligns **one** rotation axis but
  gives up on the other(s), rather than either fully tracking or fully
  ignoring orientation. Different from the OSC variants (no attempt at
  all) and from `kinova_f140_osc`/`kinova_f140_ikabs` (partial full-3-axis
  attempt) — this one looks like it's solving orientation along a single
  preferred axis. Worth checking if that's a rewarded sub-objective or an
  emergent shortcut.

## kinova_frankenstein (Priority 3, seed42, best_agent.pt — reference/joint-control, unmodified)

- Position and orientation both look good here (7-DOF redundant arm, plain
  joint control). Movement is a bit **oscillating/jittery** though.
- Wrist is controlled *slightly* — a step up from the pure "hold straight"
  (OSC) and pure "let it hang" (IK-Abs) patterns above — but in the end it
  still mostly settles into a hang-down posture; the wrist isn't doing much
  active work.
- **Suggested improvement direction**: prioritize the main arm joints for
  coarse positioning and use the tensegrity wrist for fine, quick
  orientation adjustment, rather than the current pattern of positioning
  via the arm and then letting the wrist passively hang/settle to
  whatever orientation that implies. None of the checkpoints reviewed so
  far show the wrist being used this way — worth considering as a reward
  shaping / training change for future variants rather than something
  visible in any current checkpoint.
- Same Kinova joint-limit startup errors as noted in "Cross-cutting" above
  (9x `setLimitParams()` PhysX error, one as GUI popup).

## ur10_frankenstein_osc seed123 (Priority 1, best_agent.pt)

- Matches expectations (reaches, slightly looser than seed7). Same
  wrist-unused / orientation-not-pursued findings as all other Frankenstein
  OSC variants.

## ur5e_frankenstein_osc (Priority 1, seed42, best_agent.pt)

- Same two findings as `kinova_frankenstein_osc` above: the 2-DOF tensegrity
  wrist is held straight/unused, and the policy doesn't attempt to match
  reachable target orientations, only position. Appears to be a pattern
  across the Frankenstein OSC variants rather than a one-off.
