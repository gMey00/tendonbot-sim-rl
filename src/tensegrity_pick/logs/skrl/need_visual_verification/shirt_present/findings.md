# Shirt Present — Visual Verification Findings

Notes from live playback of the selected checkpoint (see `README.md`):
`ur5e_f140`, run `2026-07-04_15-21-09_ppo_torch_seed43`, `agent_96000.pt`.

## Working well

- Cloth simulation behaves nicely overall, other than being a bit too
  stretchy.
- The arm does stretch the shirt as intended, even though the current
  "grab the lowest hanging point" strategy isn't a good one — the
  mechanics of reach → grasp → pull work.

## Configuration issues

- **Shirt initial position is too far forward along Y**: it sits close
  enough to the robot's own position axis that the arm quite often
  collides with itself. Should be repositioned to **x = 0.8** — right at
  the edge of the conveyor.
- **Second grasp point should move purely along X relative to the first
  (hanging) grasp point**: the Y and Z coordinates of the 2nd grab point
  should stay the same as the hang point, only X should differ, to
  produce a pure horizontal stretch. Which exact point on the shirt to
  target for that 2nd grasp is still open research (in progress).
- **UR5 sometimes ends up in front of the shirt**, occluding parts of it
  from the −Y inspection camera. Likely needs reward shaping to keep the
  arm/gripper out of the camera's view of the garment.

## Bugs

- **Cloth/robot clipping and unwanted wrap-around**: the cloth sometimes
  clips through solid robot links, and other times wraps around the
  robot's lower links, over-stretching the shirt well beyond what the
  task actually required.
- **Passive tensegrity holder doesn't visually hold the first grasp
  point**: it's supposed to represent the retriever robot gripping the
  shirt at the anchor, but it just hangs down in its initial rest pose
  instead of looking like it's holding the garment. Needs a fix so the
  holder visually grips at the first grasp point.
- **Gripper closes too early — reward hacking**: the UR5 gripper closes
  well before actually reaching the shirt, then approaches with the
  gripper already closed and the proximity-attach latches onto whichever
  cloth vertices happen to come close enough. This is exploiting the
  deterministic proximity-attach (see README: attaches within ~10 cm on
  closing) rather than performing a physically sensible reach-then-grasp.

## Next steps (per user, not a defect in the current checkpoint)

- Presentation/spread optimization is still active research. Planned next
  step: replace the fully random 2-grasp point selection with a
  **hem-to-hem** spread strategy, per the best-performing method found in
  [present_heuristics_study.md](../../../../../../doc/reports/present_heuristics_study.md).
