// diagrams.typ — Native Typst diagrams using fletcher
// Replaces pre-rendered Mermaid PDF diagrams with inline vector graphics.
// All diagrams use FAPS color palette for consistency.
// ============================================================

#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import "@preview/finite:0.5.1" as finite: automaton
#import "@preview/cetz:0.5.0" as cetz
#import "colors.typ": *

// Derived colors for diagrams (FAPS official palette — fills without contour)
#let faps-node-blue       = fau-blau-hell      // FAU-Blau hell: bright; black text readable
#let faps-node-green      = faps-gruen         // FAPS-Grün (official primary green)
#let faps-node-gray       = grau-3             // Grau 3 (official light gray)
#let faps-node-lightgreen = faps-gruen-hell    // FAPS-Grün hell (official)
#let faps-stroke          = fapsgraudunkel     // dark gray — edges, arrow strokes
#let faps-bg              = rgb(242, 242, 242) // F2F2F2 — group backgrounds

// Reusable subtitle text inside coloured principal nodes.
// Black text — FAU-Blau hell and FAPS-Grün fills are bright enough for contrast.
#let faps-subtitle(body) = text(size: 7.5pt, fill: black)[#body]

// Reusable side-annotation card. Used for legend-like callouts that
// must not be confused with topology nodes. Keep the look centralised
// so all diagrams share the same vocabulary.
#let faps-annotation(body) = box(
  inset: 4pt,
  radius: 0pt,
  stroke: none,
  fill: faps-node-lightgreen.lighten(40%),
  text(size: 7pt)[#body],
)


// ── 1. RL Agent–Environment Loop ─────────────────────────────
// Used in: Background § 2.1 (Reinforcement Learning)
// Classical Sutton & Barto MDP interaction loop. Two principal
// nodes are joined by an arced action arrow (top) and an arced
// observation+reward arrow (bottom). A small unit-delay marker
// $z^(-1)$ on the feedback path makes explicit that the agent
// reacts to the environment's previous output. A self-loop on
// the environment marks episode resets, and the discount
// factor $gamma$ is annotated next to the reward symbol.
// Domain-specific contents are placed in dashed annotation
// cards in the four corners so the loop topology stays clean.
#let rl-agent-env-loop() = diagram(
  spacing: (56mm, 18mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 0pt,
  edge-stroke: 0.9pt + faps-stroke,
  label-sep: 3pt,

  // ── Principal nodes ──────────────────────────────────────
  node((0.15, 0), align(center)[
    *Agent*\
    #text(size: 8pt)[Policy $pi_theta (a_t | o_t)$]\
    #faps-subtitle[PPO actor–critic]
  ],
    fill: faps-node-blue, stroke: none,
    width: 50mm, height: 22mm, name: <agent>),

  node((0.85, 0), align(center)[
    *Environment*\
    #text(size: 8pt)[state $s_t$, dynamics $P(s_(t+1) | s_t, a_t)$]\
    #faps-subtitle[PhysX simulator + task logic]
  ],
    fill: faps-node-green, stroke: none,
    width: 50mm, height: 22mm, name: <env>),

  // ── Forward arc: action ──────────────────────────────────
  edge(<agent.north>, <env.north>, "->",
    bend: 45deg,
    label: text(size: 9pt)[*action* $a_t$],
    label-pos: 0.5, label-side: left),

  // ── Feedback arc: observation + reward, via $z^(-1)$ ─────
  // The delay marker is offset downwards so the two half-arcs
  // still read as a single curved feedback path rather than a
  // pair of straight segments meeting at $z^(-1)$.
  node((0.5, 0.8), $z^(-1)$,
    fill: white, stroke: 0.5pt + faps-stroke,
    width: 7mm, height: 7mm, corner-radius: 3.5mm, name: <delay>),

  edge(<env.south>, <delay.east>, "->",
    bend: 22.5deg,
    label: text(size: 8.5pt)[$o_(t+1), r_(t+1)$],
    label-pos: 0.45, label-side: left),
  edge(<delay.west>, <agent.south>, "->",
    bend: 22.5deg,
    label: text(size: 8.5pt)[$o_t, r_t$],
    label-pos: 0.55, label-side: left),

  // ── Episode reset marker — small dashed callout next to env ─
  node((1.35, 0.0), text(size: 7pt, fill: faps-stroke)[`reset` \ #text(size: 6.5pt)[(episode end)]],
    stroke: none, fill: none, name: <reset>),
  edge(<reset>, <env.east>, "->",
    stroke: (dash: "dashed", paint: faps-stroke, thickness: 0.5pt)),

  // ── Side annotations (no incoming/outgoing edges) ────────
  // Placed in the four corners far enough from the arcs to
  // avoid visual collision with the action/observation paths.
  node((-0.1, -0.7), faps-annotation[
    *Action space*\
    #text(size: 6.5pt)[joint-position deltas \
    cable tensions \
    gripper command]
  ], shape: "rect", stroke: none, fill: none, name: <actbox>),

  node((1.1, -0.7), faps-annotation[
    *Observation* $o_t subset.eq s_t$ \
    #text(size: 6.5pt)[$bold(q)$, $dot(bold(q))$, EE pose \
    object pose, gripper state]
  ], shape: "rect", stroke: none, fill: none, name: <obsbox>),

  node((1.1, 0.7), faps-annotation[
    *Reward* $r_t$ \
    #text(size: 6.5pt)[dense shaping \
    sparse success bonus]
  ], shape: "rect", stroke: none, fill: none, name: <rewbox>),
)


// ── 2. Isaac Sim / IsaacLab Software Stack ───────────────────
// Used in: Background § 2.3 (Isaac Sim & IsaacLab)
#let isaac-stack() = diagram(
  spacing: (40mm, 18mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 0pt,
  edge-stroke: 0.8pt + faps-stroke,

  // IsaacLab layer
  node((0.25, 0), align(center)[*MDP Task*\ *Environments*],
    fill: faps-node-blue, stroke: none,
    width: 36mm, height: 15mm, name: <tasks>),
  node((1, 0), align(center)[*Managers*\ #text(size: 7pt)[Obs / Act / Rew / Term]],
    fill: faps-node-blue, stroke: none,
    width: 36mm, height: 15mm, name: <mgrs>),
  node((1.75, 0), align(center)[*Training Utils*\ #text(size: 7pt)[Multi-GPU, Wrappers]],
    fill: faps-node-blue, stroke: none,
    width: 36mm, height: 15mm, name: <train>),

  // Isaac Sim layer
  node((0.25, 1), align(center)[*Extensions*\ #text(size: 7pt)[Cloner, ROS 2]],
    fill: faps-node-green, stroke: none,
    width: 36mm, height: 15mm, name: <ext>),
  node((1, 1), align(center)[*RTX Rendering*\ #text(size: 7pt)[Sensors, Cameras]],
    fill: faps-node-green, stroke: none,
    width: 36mm, height: 15mm, name: <rtx>),
  node((1.75, 1), align(center)[*PhysX 5 Backend*\ #text(size: 7pt)[Rigid, Art., PBD]],
    fill: faps-node-green, stroke: none,
    width: 36mm, height: 15mm, name: <physx>),

  // Foundation layer
  node((0.5, 2), align(center)[*USD Scene Graph*\ #text(size: 7pt)[Data Layer]],
    fill: faps-node-gray, stroke: none,
    width: 48mm, height: 13mm, name: <usd>),
  node((1.5, 2), align(center)[*CUDA / GPU*\ #text(size: 7pt)[Tensor Interface]],
    fill: faps-node-gray, stroke: none,
    width: 48mm, height: 13mm, name: <cuda>),

  // Group outlines
  node(enclose: (<tasks>, <mgrs>, <train>),
    stroke: 0.4pt + faps-node-blue, fill: none, inset: 4mm, snap: -1, name: <lab>),
  node(enclose: (<ext>, <rtx>, <physx>),
    stroke: 0.4pt + faps-stroke, fill: none, inset: 4mm, snap: -1, name: <sim>),
  node(enclose: (<usd>, <cuda>),
    stroke: 0.4pt + faps-stroke, fill: none, inset: 4mm, snap: -1, name: <fnd>),

  // Inter-layer arrows
  edge(<lab>, <sim>, "->", stroke: 1pt + faps-stroke),
  edge(<sim>, <fnd>, "->", stroke: 1pt + faps-stroke),
)


// ── 3. PhysX Solver Pipeline ─────────────────────────────────
// Used in: Background § 2.2 (Simulation)
#let physx-pipeline() = diagram(
  spacing: (25mm, 12mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 0pt,
  edge-stroke: 0.8pt + faps-stroke,

  // Scene authoring
  node((1.2, 0), align(center)[*USD Stage*\ #text(size: 7pt)[UsdPhysics + PhysXSchema]],
    fill: faps-node-blue, stroke: none,
    width: 40mm, height: 14mm, name: <usd>),

  // Runtime binding
  node((1.2, 1), align(center)[*omni.physx*\ #text(size: 7pt)[USD ↔ PhysX]],
    fill: faps-node-green, stroke: none,
    width: 30mm, height: 14mm, name: <bind>),

  // PhysX solvers
  node((0.1, 2), align(center)[*Rigid Body*\ #text(size: 7pt)[(CPU)]],
    fill: faps-node-gray, stroke: none,
    width: 30mm, height: 13mm, name: <rcpu>),
  node((0.8, 2), align(center)[*Rigid Body*\ #text(size: 7pt)[(GPU)]],
    fill: faps-node-gray, stroke: none,
    width: 30mm, height: 13mm, name: <rgpu>),
  node((1.6, 2), align(center)[*Articulations*\ #text(size: 7pt)[(Reduced Coords)]],
    fill: faps-node-gray, stroke: none,
    width: 30mm, height: 13mm, name: <art>),
  node((2.4, 2), align(center)[*PBD Particles*\ #text(size: 7pt)[(GPU)]],
    fill: faps-node-gray, stroke: none,
    width: 30mm, height: 13mm, name: <pbd>),

  edge(<usd>, <bind>, "->"),
  edge(<bind>, <rcpu>, "->"),
  edge(<bind>, <rgpu>, "->"),
  edge(<bind>, <art>, "->"),
  edge(<bind>, <pbd>, "->"),
)


// ── 4. IsaacLab MDP Manager Decomposition ───────────────────
// Used in: Background § 2.3, Methodology § 4.1
#let mdp-managers() = diagram(
  spacing: (22mm, 10mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 0pt,
  edge-stroke: 0.8pt + faps-stroke,

  // Entry
  node((2.5, 0), align(center)[*env.step(actions)*],
    fill: faps-node-blue, stroke: none,
    width: 32mm, height: 12mm, name: <step>),

  // Managers row
  node((0, 1), align(center)[*Action*\ *Mgr*],
    fill: faps-node-lightgreen, stroke: none,
    width: 18mm, height: 14mm, name: <act>),
  node((1, 1), align(center)[*Obs*\ *Mgr*],
    fill: faps-node-lightgreen, stroke: none,
    width: 18mm, height: 14mm, name: <obs>),
  node((2, 1), align(center)[*Reward*\ *Mgr*],
    fill: faps-node-lightgreen, stroke: none,
    width: 18mm, height: 14mm, name: <rew>),
  node((3, 1), align(center)[*Term*\ *Mgr*],
    fill: faps-node-lightgreen, stroke: none,
    width: 18mm, height: 14mm, name: <term>),
  node((4, 1), align(center)[*Event*\ *Mgr*],
    fill: faps-node-lightgreen, stroke: none,
    width: 18mm, height: 14mm, name: <event>),
  node((5, 1), align(center)[*Curric.*\ *Mgr*],
    fill: faps-node-lightgreen, stroke: none,
    width: 18mm, height: 14mm, name: <curr>),

  // Physics sim
  node((2.5, 2), align(center)[*Physics Step*\ #text(size: 7pt)[PhysX]],
    fill: faps-node-green, stroke: none,
    width: 28mm, height: 12mm, name: <sim>),

  // Output
  node((2.5, 3), align(center)[*Policy Input*\ #text(size: 7pt)[$o_t$, $r_t$, done]],
    fill: faps-node-blue, stroke: none,
    width: 32mm, height: 12mm, name: <out>),

  // Edges
  edge(<step>, <act>, "->"),
  edge(<act>, <sim>, [apply], "->", label-side: left),
  edge(<sim>, <obs>, "->"),
  edge(<sim>, <rew>, "->"),
  edge(<sim>, <term>, "->"),
  edge(<sim>, <event>, "->"),
  edge(<event>, <curr>, "->"),
  edge(<obs>, <out>, "->"),
  edge(<rew>, <out>, "->"),
  edge(<term>, <out>, "->"),
)


// ── 6. Tendon Actuation Mapping ──────────────────────────────
// Used in: Background § 2.4, Methodology § 4.4
#let tendon-mapping() = diagram(
  spacing: (32mm, 12mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 0pt,
  edge-stroke: 0.8pt + faps-stroke,

  // Main chain
  node((0, 0), align(center)[*Tendon Space*\ #text(size: 7pt)[Spool lengths /\ Tensions]],
    fill: faps-node-gray, stroke: none,
    width: 30mm, height: 16mm, name: <tendon>),
  node((1, 0), align(center)[*Joint Space*\ #text(size: 7pt)[$q_1, q_2, q_3$]],
    fill: faps-node-green, stroke: none,
    width: 26mm, height: 14mm, name: <joint>),
  node((2, 0), align(center)[*Task Space*\ #text(size: 7pt)[EE Pose]],
    fill: faps-node-blue, stroke: none,
    width: 26mm, height: 14mm, name: <task>),

  edge(<tendon>, <joint>, align(center)[Transmission\ Model], "->", label-side: left),
  edge(<joint>, <task>, align(center)[Forward\ Kinematics], "->", label-side: left),

  // Non-idealities
  node((0, 1), align(center)[#text(size: 7pt)[Friction &\ Hysteresis]],
    stroke: (dash: "dashed", paint: faps-stroke, thickness: 0.4pt),
    fill: white, name: <friction>),
  node((0.6, 1), align(center)[#text(size: 7pt)[Multi-Joint\ Coupling]],
    stroke: (dash: "dashed", paint: faps-stroke, thickness: 0.4pt),
    fill: white, name: <coupling>),

  edge(<friction>, <tendon>, "->",
    stroke: (dash: "dashed", paint: faps-stroke, thickness: 0.5pt)),
  edge(<coupling>, <tendon>, "->",
    stroke: (dash: "dashed", paint: faps-stroke, thickness: 0.5pt)),
)


// ── 7. Methodology Overview Flowchart ────────────────────────
// Used in: Methodology § 4.1 (Overview of Approach)
#let methodology-overview() = diagram(
  spacing: (18mm, 5mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 0pt,
  edge-stroke: 0.8pt + faps-stroke,

  // Stage 1: Construction & Validation
  node((0, 0), align(center)[*Robot Model*\ #text(size: 7pt)[USD Assembly]],
    fill: faps-node-gray, stroke: none,
    width: 28mm, height: 12mm, name: <model>),
  node((1.2, 0), align(center)[*Validation*],
    fill: faps-node-green, stroke: none,
    width: 28mm, height: 12mm, name: <valid>),

  // Branch: actuation modes
  node((-0.5, 1), align(center)[#text(size: 7pt)[PD]], fill: faps-node-lightgreen, stroke: none, width: 14mm, height: 10mm, name: <pd>),
  node((0., 1), align(center)[#text(size: 7pt)[Tendon]], fill: faps-node-lightgreen, stroke: none, width: 14mm, height: 10mm, name: <ten>),
  node((0.5, 1), align(center)[#text(size: 7pt)[Physical]], fill: faps-node-lightgreen, stroke: none, width: 14mm, height: 10mm, name: <phys>),

  // Stage 2: Progressive tasks
  node((1.2, 1), align(center)[*Reach*\ *Task*],
    fill: faps-node-blue, stroke: none,
    width: 26mm, height: 12mm, name: <reach>),
  node((2, 1), align(center)[*Cube Place*\ *Task*],
    fill: faps-node-blue, stroke: none,
    width: 26mm, height: 12mm, name: <place>),
  node((2.8, 1), align(center)[*Cube Sort*\ #text(size: 7pt)[(outlook)]],
    fill: white, stroke: (dash: "dashed", paint: faps-stroke, thickness: 0.5pt),
    width: 26mm, height: 12mm, name: <sort>),

  // Stage 3: Evaluation
  node((1.6, 2), align(center)[*Cross-Variant* *Evaluation*],
    fill: faps-node-green, stroke: none,
    width: 50mm, height: 12mm, name: <eval>),

  // Edges
  edge(<model>, <valid>, "->"),
  edge(<model>, <pd>, "->"),
  edge(<model>, <ten>, "->"),
  edge(<model>, <phys>, "->"),
  edge(<pd>, <reach>, "->"),
  edge(<ten>, <reach>, "->"),
  edge(<phys>, <reach>, "->"),
  edge(<reach>, <place>, "->"),
  edge(<place>, <sort>, "->", stroke: (dash: "dashed", paint: faps-stroke, thickness: 0.5pt)),
  edge(<reach>, <eval>, "->"),
  edge(<place>, <eval>, "->"),
)


// ── 8. Reward Structure (Cube Place) ────────────────────────
// Used in: Methodology § 4.6
//
// Single Gantt-style chart showing every reward term, in which
// task phase it is active, its weight, and the gating conditions
// (was_grasped, was_placed, red curriculum). Replaces the older
// `reward-pipeline()` block diagram and the `reward-gating-fsm()`
// automaton, which together (a) duplicated information and
// (b) overflowed the text width when typeset.
//
// Layout: left column = reward name + weight, right strip =
// six phase columns; coloured bars mark the phases in which a
// term contributes to the per-step reward. Two latch markers on
// the top axis show where `was_grasped` and `was_placed` flip on.
//
// Width budget: 14 cm (fits the 16 cm thesis text width with
// breathing room). Height ≈ 9 cm — substantially less vertical
// real-estate than the previous two-figure stack.
#let reward-structure() = cetz.canvas({
  import cetz.draw: *

  // ── geometry ──────────────────────────────────────────────
  let label-w = 4.2      // left label column width
  let phase-w = 1.55     // each phase column
  let n-phases = 6
  let chart-w = phase-w * n-phases
  let x0 = label-w
  let x1 = x0 + chart-w
  let row-h = 0.44       // bar row height
  let row-gap = 0.05
  let header-h = 0.65    // phase header strip
  let latch-h = 0.55     // latch arrow strip below header

  // ── colour palette (FAPS) ─────────────────────────────────
  let c-approach = faps-node-lightgreen
  let c-grasp    = faps-gruen
  let c-goal     = fau-blau-hell
  let c-success  = fau-blau
  let c-penalty  = sonderfa-echtrot.lighten(40%)
  let c-reg      = grau-3
  let c-grid     = dunkelgrau
  let c-bg-odd   = rgb(248, 248, 248)
  let c-text     = faps-stroke

  // ── phase header ──────────────────────────────────────────
  let phases = (
    "Reach", "Grasp", "Lift", "Transport", "Release", "Hold",
  )
  for (i, p) in phases.enumerate() {
    let xa = x0 + i * phase-w
    let xb = xa + phase-w
    rect((xa, 0), (xb, -header-h),
      fill: hellgrau, stroke: 0.4pt + c-grid)
    content(((xa + xb) / 2, -header-h / 2),
      text(size: 9pt, weight: "bold", fill: c-text)[#p])
  }

  // ── latch strip ───────────────────────────────────────────
  let strip-y0 = -header-h - latch-h - 0.05
  let strip-y1 = strip-y0 + latch-h
  rect((x0, strip-y0), (x1, strip-y1),
    fill: sonderfa-gelb.lighten(80%), stroke: 0.4pt + c-grid)
  // was_grasped: fires at start of Lift phase
  let xg = x0 + 2 * phase-w
  line((xg, strip-y0 - 0.05), (xg, strip-y1 + 0.05),
    stroke: 0.8pt + fapsgruen.darken(20%))
  content((xg + 0.06, (strip-y0 + strip-y1) / 2),
    anchor: "west",
    text(size: 7pt, weight: "bold", fill: fapsgruen.darken(30%))[
      #raw("was_grasped") $arrow.t$
    ])
  // was_placed: fires at start of Hold phase
  let xp = x0 + 5 * phase-w
  line((xp, strip-y0 - 0.05), (xp, strip-y1 + 0.05),
    stroke: 0.8pt + fau-blau)
  content((xp - 0.06, (strip-y0 + strip-y1) / 2),
    anchor: "east",
    text(size: 7pt, weight: "bold", fill: fau-blau)[
      $arrow.t$ #raw("was_placed")
    ])
  content((x0 - 0.15, (strip-y0 + strip-y1) / 2),
    anchor: "east",
    text(size: 8pt, style: "italic", fill: c-text)[Latches])

  // ── reward rows ───────────────────────────────────────────
  let approach = (
    (name: "reaching_object",      weight: "+2",  active: (0, 1), gate: none),
    (name: "reaching_object_fine", weight: "+5",  active: (0, 1), gate: none),
    (name: "red_green_separation", weight: "+3",  active: (0, 1),
      gate: [!#raw("was_grasped")]),
  )
  let grasp = (
    (name: "grasping",       weight: "+3", active: (1, 2), gate: none),
    (name: "lifting_object", weight: "+5", active: (2,),
      gate: [grasp_active]),
    (name: "height_bonus",   weight: "+5", active: (2,),
      gate: [grasp_active]),
  )
  let transport = (
    (name: "goal_tracking",      weight: "+40", active: (2, 3, 4),
      gate: [#raw("was_grasped")]),
    (name: "goal_tracking_fine", weight: "+10", active: (2, 3, 4),
      gate: [#raw("was_grasped")]),
    (name: "release",            weight: "+25", active: (4,),
      gate: [#raw("was_grasped"), above drum]),
  )
  let success = (
    (name: "green_in_target", weight: "+100", active: (4, 5), gate: none),
  )
  let penalties = (
    (name: "belt_contact",      weight: "−10",            active: range(n-phases),
      colour: c-penalty, gate: [finger tip $<$ belt]),
    (name: "cube_off_conveyor", weight: "−5",             active: range(n-phases),
      colour: c-penalty, gate: [cube off belt]),
    (name: "action_rate",       weight: "−1e−4 → −2e−3", active: range(n-phases),
      colour: c-reg, gate: [curriculum ramp]),
    (name: "joint_vel",         weight: "−1e−4 → −2e−3", active: range(n-phases),
      colour: c-reg, gate: [curriculum ramp]),
    (name: "joint_torque",      weight: "−0.025",         active: range(n-phases),
      colour: c-reg, gate: none),
    (name: "arm_utilization",   weight: "+0.25",          active: range(n-phases),
      colour: c-reg, gate: none),
  )

  let sections = (
    (title: [Approach (active before grasp)],                        colour: c-approach, rows: approach),
    (title: [Grasp \& Lift (proximity + closure gated)],             colour: c-grasp,    rows: grasp),
    (title: [Transport \& Release (gated by #raw("was_grasped"))],   colour: c-goal,     rows: transport),
    (title: [Success],                                               colour: c-success,  rows: success),
    (title: [Penalties \& Regularisation (always active)],           colour: c-penalty,  rows: penalties),
  )

  let section-gap = 0.38
  let section-pad = 0.22
  let row-step    = row-h + row-gap

  let cursor = strip-y0 - 0.22
  for sec in sections {
    cursor = cursor - section-gap
    let yc = cursor + 0.20
    line((x0 - label-w + 0.1, yc), (x1, yc),
      stroke: (paint: c-grid, thickness: 0.4pt, dash: "dotted"))
    content((x0 - label-w + 0.15, yc - 0.24),
      anchor: "west",
      text(size: 9pt, weight: "bold", fill: fau-blau)[#sec.title])
    cursor = cursor - section-pad

    for r in sec.rows {
      cursor = cursor - row-step
      let yt = cursor + row-h
      let yb = cursor
      let bar-colour = if "colour" in r { r.colour } else { sec.colour }
      rect((x0, yb), (x1, yt), fill: c-bg-odd, stroke: none)
      for i in range(1, n-phases) {
        let xs = x0 + i * phase-w
        line((xs, yb), (xs, yt), stroke: 0.3pt + c-grid)
      }
      rect((x0, yb), (x1, yt), stroke: 0.3pt + c-grid)
      for i in r.active {
        let xa = x0 + i * phase-w + 0.08
        let xb = x0 + (i + 1) * phase-w - 0.08
        rect((xa, yb + 0.06), (xb, yt - 0.06),
          fill: bar-colour, stroke: none,
          radius: 0.06)
      }
      content((x0 - 0.15, (yt + yb) / 2),
        anchor: "east",
        text(size: 8.5pt, fill: c-text)[#raw(r.name)])
      if r.gate != none {
        content((x1 + 0.12, (yt + yb) / 2),
          anchor: "west",
          text(size: 7.5pt, style: "italic", fill: c-text)[#r.gate])
      }
    }
  }

  // ── legend (2 rows × 3 items, centered under the chart) ──
  let leg-y  = cursor - 0.65
  let leg-y2 = leg-y - 0.52
  let leg-span  = x1 * 0.88
  let leg-step  = leg-span / 3
  let leg-start = (x1 - leg-span) / 2
  let swatch(x, y, colour, label) = {
    rect((x, y - 0.15), (x + 0.35, y + 0.15),
      fill: colour, stroke: none, radius: 0.05)
    content((x + 0.46, y), anchor: "west",
      text(size: 8pt, fill: c-text)[#label])
  }
  swatch(leg-start + 0 * leg-step, leg-y,  c-approach, [Approach])
  swatch(leg-start + 1 * leg-step, leg-y,  c-grasp,    [Grasp / Lift])
  swatch(leg-start + 2 * leg-step, leg-y,  c-goal,     [Transport / Release])
  swatch(leg-start + 0 * leg-step, leg-y2, c-success,  [Success])
  swatch(leg-start + 1 * leg-step, leg-y2, c-penalty,  [Penalty])
  swatch(leg-start + 2 * leg-step, leg-y2, c-reg,      [Regularisation])
})


// ── 8 (legacy). Reward Pipeline (Cube Place) ────────────────
// Kept for backward compatibility; replaced in §4.6 by
// `reward-structure()`. Do not reference in new chapters.
#let reward-pipeline() = diagram(
  spacing: (10mm, 8mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 0pt,
  edge-stroke: 0.8pt + faps-stroke,

  // Phase 1: Approach & Grasp
  node((0, 0), align(center)[*Reach*\ #text(size: 7pt)[approach cube]],
    fill: faps-node-lightgreen, stroke: none,
    width: 22mm, height: 12mm, name: <r1>),
  node((1, 0), align(center)[*Grasp*\ #text(size: 7pt)[close gripper]],
    fill: faps-node-lightgreen, stroke: none,
    width: 22mm, height: 12mm, name: <r2>),
  node((2, 0), align(center)[*Lift*\ #text(size: 7pt)[height bonus]],
    fill: faps-node-lightgreen, stroke: none,
    width: 22mm, height: 12mm, name: <r3>),

  // Latch
  node((3, 0), align(center)[#text(size: 7pt, weight: "bold")[was_grasped\ LATCH]],
    fill: sonderfa-gelb, stroke: none,
    width: 22mm, height: 12mm, name: <latch>),

  // Phase 2: Transport & Release (gated)
  node((4, 0), align(center)[*Transport*\ #text(size: 7pt)[to drum]],
    fill: faps-node-green, stroke: none,
    width: 22mm, height: 12mm, name: <r4>),
  node((5, 0), align(center)[*Release*\ #text(size: 7pt)[in target]],
    fill: faps-node-blue, stroke: none,
    width: 22mm, height: 12mm, name: <r5>),

  // Success
  node((6, 0), align(center)[*Success*\ #text(size: 7pt)[$w = 100$]],
    fill: faps-node-blue, stroke: none,
    width: 20mm, height: 12mm, name: <succ>),

  edge(<r1>, <r2>, "->"),
  edge(<r2>, <r3>, "->"),
  edge(<r3>, <latch>, "->"),
  edge(<latch>, <r4>, "->", label-side: left),
  edge(<r4>, <r5>, "->"),
  edge(<r5>, <succ>, "->"),
)


// ── 8b. Reward-Gating FSM (Cube Place) ──────────────────────
// Used in: Methodology § 4.6 — finite-state view of the
// `was_grasped` latch and the reward-phase transitions.
#let reward-gating-fsm() = automaton(
  (
    approach:  (grasp:     "proximity<0.10m"),
    grasp:     (lift:      "closure & contact",
                approach:  "slip"),
    lift:      (transport: "Δz>0.10m & low velocity"),
    transport: (release:   "above drum & xy-aligned"),
    release:   (success:   "in target & open"),
    success:   (:),
  ),
  initial: "approach",
  final: ("success",),
  layout: finite.layout.linear.with(spacing: 3.0),
  labels: (
    approach: [Approach],
    grasp:    [Grasp],
    lift:     [Lift\ #text(size: 7pt)[#raw("was_grasped") ↑]],
    transport: [Transport],
    release:  [Release],
    success:  [Success],
    "grasp-approach": (label: text(size: 7pt)[slip], curve: -0.8),
  ),
  style: (
    state:   (fill: faps-node-lightgreen, stroke: none),
    success: (fill: faps-node-blue, stroke: none),
    transition: (stroke: 0.8pt + faps-stroke),
  ),
)


// ── 9. Mesh Processing Pipeline (CAD → USD → Isaac Lab) ─────
// Used in: Methodology § 4.3 (Simulation Model Construction)
#let mesh-processing-pipeline() = diagram(
  spacing: (11mm, 8mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 0pt,
  edge-stroke: 0.8pt + faps-stroke,

  // Source
  node((0, 0), align(center)[*Fusion 360*\ #text(size: 7pt)[CAD Assembly]],
    fill: faps-node-gray, stroke: none,
    width: 22mm, height: 20mm, name: <cad>),

  // Blender processing
  node((1, 0), align(center)[*Blender*\ #text(size: 7pt)[FBX Import]],
    fill: faps-node-lightgreen, stroke: none,
    width: 24mm, height: 20mm, name: <blender>),
  node((1, 1), align(center)[#text(size: 7pt)[Join Parts &\ Fix Origins]],
    fill: faps-node-lightgreen, stroke: none,
    width: 22mm, height: 12mm, name: <join>),
  node((1, 2), align(center)[#text(size: 7pt)[Decimation &\ Collision Meshes]],
    fill: faps-node-lightgreen, stroke: none,
    width: 22mm, height: 12mm, name: <decim>),

  // USD Builder
  node((2, 0), align(center)[*USD Builder*\ *Script*\ #text(size: 7pt)[Python + Kit]],
    fill: faps-node-green, stroke: none,
    width: 22mm, height: 20mm, name: <builder>),
  node((2, 1), align(center)[#text(size: 7pt)[ArticulationRoot\ RigidBody APIs]],
    fill: faps-node-green, stroke: none,
    width: 22mm, height: 12mm, name: <apis>),
  node((2, 2), align(center)[#text(size: 7pt)[Mass / Inertia /\ Joint Properties]],
    fill: faps-node-green, stroke: none,
    width: 22mm, height: 12mm, name: <props>),

  // Assembly
  node((3, 0), align(center)[*Robot*\ *Assembler*\ #text(size: 7pt)[USD Composition]],
    fill: faps-node-blue, stroke: none,
    width: 24mm, height: 20mm, name: <assembler>),
  node((3, 1), align(center)[#text(size: 7pt)[Base + Arm]],
    fill: faps-node-green, stroke: none,
    width: 22mm, height: 12mm, name: <arm>),
  node((3, 2), align(center)[#text(size: 7pt)[Robot + Gripper]],
    fill: faps-node-green, stroke: none,
    width: 22mm, height: 12mm, name: <gripper>),
  // Output
  node((4, 0), align(center)[*Isaac Lab*\ *Asset*],
    fill: faps-node-blue, stroke: none,
    width: 22mm, height: 20mm, name: <asset>),

  edge(<cad>, <blender>, [FBX], "->"),
  edge(<blender>, <join>, "->"),
  edge(<join>, <decim>, "->"),
  edge(<decim>, <builder>, "->"),
  edge(<blender>, <builder>, [USDC], "->"),
  edge(<builder>, <apis>, "->"),
  edge(<apis>, <props>, "->"),
  edge(<props>, <assembler>, "->"),
  edge(<builder>, <assembler>, [USD], "->"),
  edge(<assembler>, <arm>, "->"),
  edge(<arm>, <gripper>, "->"),
  edge(<gripper>, <asset>, "->"),
  edge(<assembler>, <asset>, "->"),
)


// ── 10. Antiparallelogram Four‑Bar Linkage Schematic ────────
// Used in: Methodology § 4.2 (Physical Robot Design)
#let antiparallelogram-linkage() = diagram(
  spacing: (24mm, 24mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 50%,
  edge-stroke: 0.8pt + faps-stroke,

  // Joints — four corners of the crossed linkage
  node((0, 0), $A$, fill: faps-node-gray, stroke: none,
    width: 8mm, height: 8mm, name: <A>),
  node((1, 0), $B$, fill: faps-node-gray, stroke: none,
    width: 8mm, height: 8mm, name: <B>),
  node((0.2, 1), $D$, fill: faps-node-green, stroke: none,
    width: 8mm, height: 8mm, name: <D>),
  node((0.8, 1), $C$, fill: faps-node-green, stroke: none,
    width: 8mm, height: 8mm, name: <C>),

  // Frame (base link)
  edge(<A>, <B>, [frame $k_e$], "=", stroke: 1.2pt + faps-stroke, label-side: right),
  // Coupler (platform link)
  edge(<D>, <C>, [coupler $k_e$], "=", stroke: 1.2pt + faps-node-green.darken(20%), label-side: left),
  // Crossed side links
  edge(<A>, <C>, [$l_e$], "-", stroke: 0.8pt + fau-blau),
  edge(<B>, <D>, [$l_e$], "-", stroke: 0.8pt + fau-blau),
)


// ── 11. Kinematic Chain of the 5‑DOF Manipulator ────────────
// Used in: Methodology § 4.2 (Physical Robot Design)
#let kinematic-chain-diagram() = diagram(
  spacing: (26mm, 8mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 0pt,
  edge-stroke: 0.8pt + faps-stroke,

  // Ceiling mount (world)
  node((1, 0), align(center)[*Ceiling*\ #text(size: 7pt)[World Frame]],
    fill: faps-node-gray, stroke: none,
    width: 24mm, height: 12mm, name: <world>),

  // Prismatic base
  node((0.5, 1), align(center)[*Base Y*\ #text(size: 7pt)[Prismatic]],
    fill: faps-node-lightgreen, stroke: none,
    width: 24mm, height: 12mm, name: <baseY>),
  node((1.5, 1), align(center)[*Base Z*\ #text(size: 7pt)[Prismatic]],
    fill: faps-node-lightgreen, stroke: none,
    width: 24mm, height: 12mm, name: <baseZ>),

  // Elbow
  node((1, 2), align(center)[*Elbow*\ #text(size: 7pt)[Revolute, 1 DoF]],
    fill: faps-node-green, stroke: none,
    width: 24mm, height: 12mm, name: <elbow>),

  // Wrist
  node((0.5, 3), align(center)[*Wrist Pitch*\ #text(size: 7pt)[Revolute]],
    fill: faps-node-blue, stroke: none,
    width: 24mm, height: 12mm, name: <pitch>),
  node((1.5, 3), align(center)[*Wrist Roll*\ #text(size: 7pt)[Revolute]],
    fill: faps-node-blue, stroke: none,
    width: 24mm, height: 12mm, name: <roll>),

  // Gripper
  node((1, 4), align(center)[*Robotiq 2F-140*\ #text(size: 7pt)[Gripper]],
    fill: faps-node-gray, stroke: none,
    width: 32mm, height: 12mm, name: <gripper>),

  edge(<world>, <baseY>, "->"),
  edge(<world>, <baseZ>, "->"),
  edge(<baseY>, <elbow>, "->"),
  edge(<baseZ>, <elbow>, "->"),
  edge(<elbow>, <pitch>, "->"),
  edge(<elbow>, <roll>, "->"),
  edge(<pitch>, <gripper>, "->"),
  edge(<roll>, <gripper>, "->"),
)


// ── 12. Tendon Actuation Data Flow ──────────────────────────
// Used in: Methodology § 4.4 (Tendon Actuation in Simulation)
#let tendon-actuation-dataflow() = diagram(
  spacing: (10mm, 10mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 0pt,
  edge-stroke: 0.8pt + faps-stroke,

  // RL policy output
  node((0, 0), align(center)[*Policy*\ #text(size: 7pt)[$bold(a)_t in [-1, 1]^5$]],
    fill: faps-node-blue, stroke: none,
    width: 24mm, height: 16mm, name: <policy>),

  // Tension scaling
  node((1, 0), align(center)[*Tension*\ *Scaling*\ #text(size: 7pt)[$bold(T) = bold(a)_t dot T_max$]],
    fill: faps-node-lightgreen, stroke: none,
    width: 26mm, height: 16mm, name: <scale>),

  // J^T mapping
  node((2, 0), align(center)[*$J^top$ Mapping*\ #text(size: 7pt)[$bold(tau) = J^top bold(T)$]],
    fill: faps-node-green, stroke: none,
    width: 30mm, height: 16mm, name: <jt>),

  // Split
  node((3, -0.4), align(center)[#text(size: 7pt)[Elbow\ Torque]],
    fill: faps-node-gray, stroke: none,
    width: 20mm, height: 11mm, name: <elbow>),
  node((3, 0.4), align(center)[#text(size: 7pt)[Wrist\ Torques]],
    fill: faps-node-gray, stroke: none,
    width: 20mm, height: 11mm, name: <wrist>),

  // PhysX
  node((4, 0), align(center)[*PhysX*\ *Joints*],
    fill: faps-node-blue, stroke: none,
    width: 22mm, height: 16mm, name: <physx>),

  edge(<policy>, <scale>, "->"),
  edge(<scale>, <jt>, "->"),
  edge(<jt>, <elbow>, "->"),
  edge(<jt>, <wrist>, "->"),
  edge(<elbow>, <physx>, "->"),
  edge(<wrist>, <physx>, "->"),
)


// ── 13. Antagonistic Tendon Actuation (Conceptual) ──────────
// Used in: Background § 2.2 (Tendon-Driven Mechanisms)
// Shows a revolute joint actuated by two antagonistic cables routed
// from proximal motors, illustrating bidirectional torque and
// co-contraction stiffness modulation.
#let tendon-antagonistic() = diagram(
  spacing: (28mm, 12mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 0pt,
  edge-stroke: 0.8pt + faps-stroke,

  // Proximal motors
  node((0, 0), align(center)[*Motor 1*\ #text(size: 7pt)[Tension $T_1$]],
    fill: faps-node-gray, stroke: none,
    width: 22mm, height: 12mm, name: <m1>),
  node((0, 1), align(center)[*Motor 2*\ #text(size: 7pt)[Tension $T_2$]],
    fill: faps-node-gray, stroke: none,
    width: 22mm, height: 12mm, name: <m2>),

  // Joint
  node((1, 0.5), align(center)[*Revolute*\ *Joint*\ #text(size: 7pt)[1 DoF]],
    fill: faps-node-green, stroke: none,
    width: 24mm, height: 16mm, name: <joint>),

  // Distal link
  node((2, 0.5), align(center)[*Distal*\ *Link*\ #text(size: 7pt)[low inertia]],
    fill: faps-node-lightgreen, stroke: none,
    width: 22mm, height: 14mm, name: <link>),

  // Result
  node((1, 1.5), align(center)[#text(size: 7pt, weight: "bold")[Net torque]\ #text(size: 6.5pt)[$tau = r(T_1 - T_2)$\ stiffness $prop T_1 + T_2$]],
    fill: white, stroke: (dash: "dashed", paint: faps-stroke, thickness: 0.4pt),
    width: 34mm, height: 14mm, name: <result>),

  edge(<m1>, <joint>, [cable 1], "->", label-side: left,
    stroke: 0.8pt + fau-blau),
  edge(<m2>, <joint>, [cable 2], "->", label-side: left,
    stroke: 0.8pt + fau-blau),
  edge(<joint>, <link>, "->"),
  edge(<joint>, <result>, "-", stroke: 0.4pt + faps-stroke),
)


// ── 14. Tensegrity Structural Concept ───────────────────────
// Used in: Background § 2.2 (Tensegrity Principles)
// Shows the tensegrity paradigm: rigid compression struts connected
// solely by a continuous tension network, bridging rigid and soft robotics.
#let tensegrity-concept() = diagram(
  spacing: (30mm, 10mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 0pt,
  edge-stroke: 0.8pt + faps-stroke,

  // Core principle boxes
  node((0, 0), align(center)[*Compression*\ *Elements*\ #text(size: 7pt)[struts, rigid bars]],
    fill: faps-node-gray, stroke: none,
    width: 28mm, height: 16mm, name: <comp>),
  node((1, 0), align(center)[*Tension*\ *Network*\ #text(size: 7pt)[cables, springs]],
    fill: faps-node-lightgreen, stroke: none,
    width: 28mm, height: 16mm, name: <tens>),

  // Resulting structure
  node((0.5, 1), align(center)[*Tensegrity*\ *Structure*\ #text(size: 7pt)[self-stressed equilibrium]],
    fill: faps-node-green, stroke: none,
    width: 32mm, height: 16mm, name: <structure>),

  // Properties
  node((-0.3, 2), align(center)[#text(size: 7pt)[Inherent\ compliance]],
    fill: white, stroke: (dash: "dashed", paint: faps-stroke, thickness: 0.4pt),
    width: 22mm, height: 11mm, name: <p1>),
  node((0.5, 2), align(center)[#text(size: 7pt)[Lightweight\ high strength]],
    fill: white, stroke: (dash: "dashed", paint: faps-stroke, thickness: 0.4pt),
    width: 22mm, height: 11mm, name: <p2>),
  node((1.3, 2), align(center)[#text(size: 7pt)[Controllable\ stiffness]],
    fill: white, stroke: (dash: "dashed", paint: faps-stroke, thickness: 0.4pt),
    width: 22mm, height: 11mm, name: <p3>),

  // Robotics application
  node((0.5, 3), align(center)[*Robot Design*\ #text(size: 7pt)[rigid links (struts) +\ cable actuation (tension)]],
    fill: faps-node-blue, stroke: none,
    width: 36mm, height: 16mm, name: <robot>),

  edge(<comp>, <structure>, "->"),
  edge(<tens>, <structure>, "->"),
  edge(<structure>, <p1>, "->", stroke: 0.5pt + faps-stroke),
  edge(<structure>, <p2>, "->", stroke: 0.5pt + faps-stroke),
  edge(<structure>, <p3>, "->", stroke: 0.5pt + faps-stroke),
  edge(<p1>, <robot>, "->", stroke: 0.4pt + faps-stroke),
  edge(<p2>, <robot>, "->", stroke: 0.4pt + faps-stroke),
  edge(<p3>, <robot>, "->", stroke: 0.4pt + faps-stroke),
)


// ── 14. Tensegrity Manipulator Kinematic Schematic (Klein-style) ──────
// NOTE: superseded by the matplotlib version exported from
// `doc/Tensegrity_robot/kinematic_schematic.ipynb` to
// `shared/figures/tensegrity_kinematic_schematic.svg`.
// Kept here (commented out) in case we want to return to a fully native
// CeTZ rendering later.
/*
#let tensegrity-kinematic-schematic() = cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Palette
  let c-stroke   = faps-stroke
  let c-link     = fapsblau.darken(10%)
  let c-rev      = fapsgruen.darken(15%)
  let c-pris     = rgb("#CC6600")
  let c-axis-y   = rgb("#33AA33")
  let c-axis-z   = rgb("#3366CC")
  let c-text     = faps-stroke
  let c-inset-bg = rgb(248, 248, 248)
  let c-rod      = fapsblau

  // Helpers ---------------------------------------------------
  let triad(p, label, anchor: "west", len: 0.40) = {
    let x = p.at(0)
    let y = p.at(1)
    let dx = if anchor == "west" { len } else { -len }
    line((x, y), (x + dx, y),
      mark: (end: ">"), stroke: 0.6pt + c-axis-y)
    line((x, y), (x, y + len),
      mark: (end: ">"), stroke: 0.6pt + c-axis-z)
    circle((x, y), radius: 0.06, fill: c-stroke, stroke: none)
    if label != none {
      let lx = if anchor == "west" { x + 0.50 } else { x - 0.50 }
      content((lx, y - 0.05),
        text(size: 6.5pt, fill: c-text)[#label], anchor: anchor)
    }
  }

  let rev-joint(p, label: none, label-anchor: "west", radius: 0.22) = {
    let x = p.at(0)
    let y = p.at(1)
    circle((x, y), radius: radius, fill: white,
      stroke: 0.8pt + c-rev)
    circle((x, y), radius: 0.05, fill: c-rev, stroke: none)
    if label != none {
      let lx = if label-anchor == "west" {
        x + radius + 0.18
      } else {
        x - radius - 0.18
      }
      content((lx, y),
        text(size: 7pt, fill: c-text)[#label], anchor: label-anchor)
    }
  }

  let prism-h(p, label, w: 1.25, h: 0.50) = {
    let x = p.at(0)
    let y = p.at(1)
    rect((x - w/2, y - h/2), (rel: (w, h)),
      fill: white, stroke: 0.7pt + c-pris)
    line((x - w/2 + 0.12, y), (x + w/2 - 0.12, y),
      mark: (start: ">", end: ">"), stroke: 0.6pt + c-pris)
    content((x, y - h/2 - 0.20),
      text(size: 7pt, fill: c-text)[#label], anchor: "north")
  }

  let prism-v(p, label, w: 0.50, h: 1.10) = {
    let x = p.at(0)
    let y = p.at(1)
    rect((x - w/2, y - h/2), (rel: (w, h)),
      fill: white, stroke: 0.7pt + c-pris)
    line((x, y - h/2 + 0.12), (x, y + h/2 - 0.12),
      mark: (start: ">", end: ">"), stroke: 0.6pt + c-pris)
    content((x + w/2 + 0.18, y),
      text(size: 7pt, fill: c-text)[#label], anchor: "west")
  }

  let ceiling(x0, x1, y) = {
    line((x0, y), (x1, y), stroke: 1.0pt + c-stroke)
    let n = 14
    let dx = (x1 - x0) / n
    for i in range(n) {
      let xa = x0 + i * dx
      line((xa, y), (xa - 0.20, y + 0.28), stroke: 0.4pt + c-stroke)
    }
  }

  // ─── MAIN SCHEMATIC ────────────────────────────────────────
  let xc = 5.0
  let cy = 12.5

  ceiling(2.5, 7.5, cy)
  content((7.6, cy + 0.15),
    text(size: 7.5pt, fill: c-text)[*Ceiling — World Frame* $Sigma_W$],
    anchor: "west")

  line((xc, cy), (xc, 11.95), stroke: 1.6pt + c-link)
  prism-h((xc, 11.65),
    [`base_y_joint` — prismatic Y, $plus.minus 0.5$ m])
  line((xc, 11.40), (xc, 10.85), stroke: 1.6pt + c-link)
  prism-v((xc, 10.30),
    [`base_z_joint` — prismatic Z, $-0.5 dots 0$ m])
  line((xc, 9.75), (xc, 9.15), stroke: 1.6pt + c-link)

  triad((xc, 9.10), [LO 0 — `root_link` $(0,0,0)$])
  line((xc, 9.10), (xc, 6.50), stroke: 2.6pt + c-link)
  content((xc - 0.30, 7.80),
    text(size: 7pt, fill: c-text)[$d_2 = 430$ mm], anchor: "east")
  content((xc - 0.30, 7.40),
    text(size: 6.5pt, fill: c-text)[(upper-arm aluminium bracket)],
    anchor: "east")

  rev-joint((xc, 6.50),
    label: [`elbow_joint` (rev., $plus.minus 70 degree$, disc-approx.)],
    label-anchor: "west", radius: 0.26)
  triad((xc - 0.05, 6.50), none, anchor: "east")
  content((xc - 0.50, 6.10),
    text(size: 6.5pt, fill: c-text)[LO 1 — `forearm_link` $(0,0,-0.4975)$],
    anchor: "east")

  line((xc, 6.50), (xc, 4.30), stroke: 2.6pt + c-link)
  content((xc - 0.30, 5.40),
    text(size: 7pt, fill: c-text)[$d_3 = 406$ mm], anchor: "east")
  content((xc - 0.30, 5.00),
    text(size: 6.5pt, fill: c-text)[(forearm aluminium tube)],
    anchor: "east")

  triad((xc - 0.05, 4.25), none, anchor: "east")
  content((xc - 0.50, 4.55),
    text(size: 6.5pt, fill: c-text)[LO 3 — `wrist_intermediate_link` $(0,0,-0.836)$],
    anchor: "east")
  rev-joint((xc, 4.25),
    label: [`wrist_x_joint` (rev., $plus.minus 50 degree$)],
    label-anchor: "west", radius: 0.18)
  line((xc, 4.25), (xc, 3.55), stroke: 1.6pt + c-link)
  rev-joint((xc, 3.55),
    label: [`wrist_y_joint` (rev., $plus.minus 50 degree$)],
    label-anchor: "west", radius: 0.18)

  line((xc, 3.55), (xc, 2.85), stroke: 2.0pt + c-link)
  content((xc - 0.30, 3.20),
    text(size: 7pt, fill: c-text)[$d_4 = 136$ mm], anchor: "east")
  triad((xc, 2.85), [LO 4 — `tool_link` $(0,0,-0.972)$])

  // Robotiq 2F-140 gripper body
  let gx = xc
  let gy = 2.40
  rect((gx - 0.55, gy - 0.30), (rel: (1.10, 0.30)),
    fill: hellgrau, stroke: 0.7pt + c-stroke)
  rect((gx - 0.55, gy - 1.20), (rel: (0.18, 0.95)),
    fill: hellgrau, stroke: 0.7pt + c-stroke)
  rect((gx + 0.37, gy - 1.20), (rel: (0.18, 0.95)),
    fill: hellgrau, stroke: 0.7pt + c-stroke)
  content((gx + 0.85, gy - 0.55),
    text(size: 7pt, fill: c-text)[`finger_joint` — Robotiq 2F-140 gripper],
    anchor: "west")

  // ─── INSET: physical antiparallelogram elbow variant ──────
  let ix = 12.7
  let iy = 6.50
  let iw = 3.6
  let ih = 4.4

  rect((ix - iw/2, iy - ih/2), (rel: (iw, ih)),
    fill: c-inset-bg,
    stroke: (paint: c-stroke, dash: "dashed", thickness: 0.5pt))
  content((ix, iy + ih/2 - 0.30),
    text(size: 7.5pt, fill: c-text, weight: "bold")[Physical elbow variant],
    anchor: "center")
  content((ix, iy + ih/2 - 0.65),
    text(size: 6.5pt, fill: c-text)[antiparallelogram four-bar linkage],
    anchor: "center")

  let A = (ix - 0.50, iy + 0.55)
  let B = (ix + 0.50, iy + 0.55)
  let C = (ix - 0.50, iy - 1.10)
  let D = (ix + 0.50, iy - 1.10)

  let n2 = 8
  let dxh = (B.at(0) - A.at(0)) / n2
  for i in range(n2) {
    let xa = A.at(0) + i * dxh
    line((xa, A.at(1) + 0.10), (xa - 0.16, A.at(1) + 0.36),
      stroke: 0.4pt + c-stroke)
  }
  line((A.at(0) - 0.10, A.at(1) + 0.10),
       (B.at(0) + 0.10, A.at(1) + 0.10),
       stroke: 0.7pt + c-stroke)

  line(A, B, stroke: 1.6pt + c-link)
  content((ix, A.at(1) + 0.55),
    text(size: 6.5pt, fill: c-text)[root_link frame ($k_e = 60$ mm)],
    anchor: "south")

  line(C, D, stroke: 1.6pt + c-link)
  content((B.at(0) + 0.20, D.at(1) - 0.05),
    text(size: 6pt, fill: c-text)[forearm coupler ($k_e$)],
    anchor: "west")

  line(A, D, stroke: 1.6pt + c-rod)
  line(B, C, stroke: 1.6pt + c-rod)
  content((ix + 1.05, iy + 0.10),
    text(size: 6.5pt, fill: c-rod)[$l_e = 150$ mm], anchor: "west")

  for tup in (
    (A, "A", "east"),
    (B, "B", "west"),
    (C, "C", "east"),
    (D, "D", "west"),
  ) {
    let p = tup.at(0)
    let lbl = tup.at(1)
    let ax = tup.at(2)
    circle(p, radius: 0.10, fill: white, stroke: 0.7pt + c-rev)
    circle(p, radius: 0.03, fill: c-rev, stroke: none)
    let off = if ax == "east" { -0.18 } else { 0.18 }
    content((p.at(0) + off, p.at(1)),
      text(size: 6.5pt, fill: c-rev, weight: "bold")[#lbl],
      anchor: ax)
  }

  line((ix, D.at(1) - 0.05), (ix, D.at(1) - 0.55),
    mark: (end: ">"), stroke: 1.4pt + c-link)
  content((ix, D.at(1) - 0.70),
    text(size: 6pt, fill: c-text)[to forearm — loop closure constraint],
    anchor: "north")

  // Connector from main elbow to inset (drawn below the elbow label so it
  // does not strike through the joint annotation)
  line((xc + 0.30, 6.10), (ix - iw/2 - 0.05, 6.10),
    mark: (end: ">"),
    stroke: (paint: c-stroke, dash: "dashed", thickness: 0.5pt))
  content(((xc + ix - iw/2) / 2, 6.25),
    text(size: 6pt, fill: c-stroke)[detail], anchor: "south")

  // ─── Legend ────────────────────────────────────────────────
  let lx = 0.4
  let ly = 1.6
  rect((lx - 0.1, ly - 1.4), (rel: (5.4, 1.7)),
    stroke: 0.4pt + c-stroke, fill: c-inset-bg)
  content((lx, ly + 0.10),
    text(size: 6.5pt, fill: c-text, weight: "bold")[Legend],
    anchor: "west")
  // Row 1: revolute
  circle((lx + 0.20, ly - 0.30), radius: 0.10,
    fill: white, stroke: 0.6pt + c-rev)
  content((lx + 0.45, ly - 0.30),
    text(size: 6.5pt, fill: c-text)[revolute joint], anchor: "west")
  // Row 1 col 2: prismatic
  rect((lx + 2.80, ly - 0.40), (rel: (0.40, 0.20)),
    fill: white, stroke: 0.6pt + c-pris)
  content((lx + 3.30, ly - 0.30),
    text(size: 6.5pt, fill: c-text)[prismatic joint], anchor: "west")
  // Row 2: triad
  line((lx + 0.10, ly - 0.85), (lx + 0.40, ly - 0.85),
    mark: (end: ">"), stroke: 0.5pt + c-axis-y)
  line((lx + 0.10, ly - 0.85), (lx + 0.10, ly - 0.55),
    mark: (end: ">"), stroke: 0.5pt + c-axis-z)
  content((lx + 0.50, ly - 0.75),
    text(size: 6.5pt, fill: c-text)[link-origin frame ($Y$, $Z$)],
    anchor: "west")
  // Row 2 col 2: link
  line((lx + 2.80, ly - 0.75), (lx + 3.20, ly - 0.75),
    stroke: 2.0pt + c-link)
  content((lx + 3.30, ly - 0.75),
    text(size: 6.5pt, fill: c-text)[rigid link], anchor: "west")
  // Row 3: crossed rod (linkage)
  line((lx + 0.10, ly - 1.20), (lx + 0.40, ly - 1.20),
    stroke: 1.4pt + c-rod)
  content((lx + 0.50, ly - 1.20),
    text(size: 6.5pt, fill: c-text)[antiparallelogram side link],
    anchor: "west")
})
*/



// ── 14. Training Infrastructure Stack ────────────────────────
// Used in: Methodology § 4.7 (Training Infrastructure)
#let training-infrastructure-stack() = diagram(
  spacing: (10mm, 16mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 0pt,
  edge-stroke: 0.8pt + faps-stroke,

  // Algorithm layer (top): PPO + skrl
  node((0.5, 0), align(center)[*PPO Agent*\ #text(size: 7pt)[`skrl` 1.3]],
    fill: faps-node-blue, stroke: none,
    width: 43mm, height: 14mm, name: <ppo>),
  node((1.5, 0), align(center)[*Sequential Trainer*\ #text(size: 7pt)[Rollout · Update]],
    fill: faps-node-blue, stroke: none,
    width: 43mm, height: 14mm, name: <trainer>),
  node((2.5, 0), align(center)[*Logging*\ #text(size: 7pt)[TensorBoard · Custom TUI]],
    fill: faps-node-blue, stroke: none,
    width: 43mm, height: 14mm, name: <log>),

  // Environment layer (middle): Gymnasium + IsaacLab task
  node((0.5, 1), align(center)[*Gymnasium Wrapper*\ #text(size: 7pt)[`tensegrity_pick`]],
    fill: faps-node-lightgreen, stroke: none,
    width: 43mm, height: 14mm, name: <gym>),
  node((1.5, 1), align(center)[*IsaacLab Managers*\ #text(size: 7pt)[Obs · Act · Rew · Term · Curr]],
    fill: faps-node-lightgreen, stroke: none,
    width: 43mm, height: 14mm, name: <mgrs>),
  node((2.5, 1), align(center)[*Vectorized Envs*\ #text(size: 7pt)[4 096 parallel · GPU clones]],
    fill: faps-node-lightgreen, stroke: none,
    width: 43mm, height: 14mm, name: <vec>),

  // Simulation layer (third row): IsaacSim + PhysX
  node((0.5, 2), align(center)[*IsaacSim 5.1.0*\ #text(size: 7pt)[USD Stage · Cloner]],
    fill: faps-node-green, stroke: none,
    width: 43mm, height: 14mm, name: <isaac>),
  node((1.5, 2), align(center)[*PhysX 5*\ #text(size: 7pt)[Articulations · Tendons]],
    fill: faps-node-green, stroke: none,
    width: 43mm, height: 14mm, name: <physx>),
  node((2.5, 2), align(center)[*USD Assets*\ #text(size: 7pt)[Robot · Scene · Props]],
    fill: faps-node-green, stroke: none,
    width: 43mm, height: 14mm, name: <usd>),

  // Hardware layer (bottom)
  node((1.0, 3), align(center)[*NVIDIA RTX A6000*\ #text(size: 7pt)[48 GB VRAM · CUDA]],
    fill: faps-node-gray, stroke: none,
    width: 50mm, height: 14mm, name: <gpu>),
  node((2.0, 3), align(center)[*Workstation Host*\ #text(size: 7pt)[Linux · Conda env]],
    fill: faps-node-gray, stroke: none,
    width: 50mm, height: 14mm, name: <host>),

  // Group enclosures
  node(enclose: (<ppo>, <trainer>, <log>),
    stroke: 0.4pt + faps-node-blue, fill: none, inset: 4mm, snap: -1, name: <alg>),
  node(enclose: (<gym>, <mgrs>, <vec>),
    stroke: 0.4pt + faps-stroke, fill: none, inset: 4mm, snap: -1, name: <env>),
  node(enclose: (<isaac>, <physx>, <usd>),
    stroke: 0.4pt + faps-stroke, fill: none, inset: 4mm, snap: -1, name: <sim>),
  node(enclose: (<gpu>, <host>),
    stroke: 0.4pt + faps-stroke, fill: none, inset: 4mm, snap: -1, name: <hw>),

  // Inter-layer arrows
  edge(<alg>, <env>, "->", stroke: 1pt + faps-stroke, label: text(size: 7pt)[actions / obs]),
  edge(<env>, <sim>, "->", stroke: 1pt + faps-stroke, label: text(size: 7pt)[step()]),
  edge(<sim>, <hw>, "->", stroke: 1pt + faps-stroke, label: text(size: 7pt)[CUDA tensors]),
)
