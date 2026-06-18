// diagrams.typ — Native Typst diagrams using fletcher
// Replaces pre-rendered Mermaid PDF diagrams with inline vector graphics.
// All diagrams use FAPS color palette for consistency.
// ============================================================

#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import "colors.typ": *

// Derived colors for diagrams
#let faps-node-blue = fapsblau
#let faps-node-green = fapsgruen
#let faps-node-gray = hellgrau
#let faps-node-lightgreen = rgb(214, 232, 168) // D6E8A8 — light accent
#let faps-stroke = fapsgraudunkel
#let faps-bg = rgb(242, 242, 242) // F2F2F2 — group backgrounds


// ── 1. RL Agent–Environment Loop ─────────────────────────────
// Used in: Background § 2.1 (Reinforcement Learning)
#let rl-agent-env-loop() = diagram(
  spacing: (30mm, 0mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 0pt,
  edge-stroke: 0.8pt + faps-stroke,

  node((0, 0), align(center)[*Agent*\ $pi(a | o)$],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
    width: 26mm, height: 14mm, name: <agent>),
  node((1, 0), align(center)[*Environment*\ MDP],
    fill: faps-node-green, stroke: faps-node-green.darken(20%),
    width: 26mm, height: 14mm, name: <env>),

  edge(<agent>, <env>, [action $a_t$],
    "->", label-side: left, bend: -30deg),
  edge(<env>, <agent>, align(center)[observation $o_(t+1)$\ reward $r_t$],
    "->", label-side: left, bend: -30deg),
)


// ── 2. Isaac Sim / IsaacLab Software Stack ───────────────────
// Used in: Background § 2.3 (Isaac Sim & IsaacLab)
#let isaac-stack() = diagram(
  spacing: (8mm, 8mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 0pt,
  edge-stroke: 0.8pt + faps-stroke,

  // IsaacLab layer
  node((0, 0), align(center)[*MDP Task*\ *Environments*],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
    width: 33mm, height: 15mm, name: <tasks>),
  node((1, 0), align(center)[*Managers*\ #text(size: 7pt)[Obs / Act / Rew / Term]],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
    width: 33mm, height: 15mm, name: <mgrs>),
  node((2, 0), align(center)[*Training Utils*\ #text(size: 7pt)[Multi-GPU, Wrappers]],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
    width: 33mm, height: 15mm, name: <train>),

  // Isaac Sim layer
  node((0, 1), align(center)[*Extensions*\ #text(size: 7pt)[Cloner, ROS 2]],
    fill: faps-node-green, stroke: faps-node-green.darken(20%),
    width: 33mm, height: 15mm, name: <ext>),
  node((1, 1), align(center)[*RTX Rendering*\ #text(size: 7pt)[Sensors, Cameras]],
    fill: faps-node-green, stroke: faps-node-green.darken(20%),
    width: 33mm, height: 15mm, name: <rtx>),
  node((2, 1), align(center)[*PhysX 5 Backend*\ #text(size: 7pt)[Rigid, Art., PBD]],
    fill: faps-node-green, stroke: faps-node-green.darken(20%),
    width: 33mm, height: 15mm, name: <physx>),

  // Foundation layer
  node((0.25, 2), align(center)[*USD Scene Graph*\ #text(size: 7pt)[Data Layer]],
    fill: faps-node-gray, stroke: faps-stroke,
    width: 48mm, height: 13mm, name: <usd>),
  node((1.75, 2), align(center)[*CUDA / GPU*\ #text(size: 7pt)[Tensor Interface]],
    fill: faps-node-gray, stroke: faps-stroke,
    width: 48mm, height: 13mm, name: <cuda>),

  // Group outlines
  node(enclose: (<tasks>, <mgrs>, <train>),
    stroke: 0.4pt + faps-node-blue, fill: none, inset: 8mm, snap: -1, name: <lab>),
  node(enclose: (<ext>, <rtx>, <physx>),
    stroke: 0.4pt + faps-stroke, fill: none, inset: 8mm, snap: -1, name: <sim>),
  node(enclose: (<usd>, <cuda>),
    stroke: 0.4pt + faps-stroke, fill: none, inset: 8mm, snap: -1, name: <fnd>),

  // Inter-layer arrows
  edge(<lab>, <sim>, "->", stroke: 1pt + faps-stroke),
  edge(<sim>, <fnd>, "->", stroke: 1pt + faps-stroke),
)


// ── 3. PhysX Solver Pipeline ─────────────────────────────────
// Used in: Background § 2.2 (Simulation)
#let physx-pipeline() = diagram(
  spacing: (10mm, 10mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 0pt,
  edge-stroke: 0.8pt + faps-stroke,

  // Scene authoring
  node((1, 0), align(center)[*USD Stage*\ #text(size: 7pt)[UsdPhysics + PhysXSchema]],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
    width: 40mm, height: 14mm, name: <usd>),

  // Runtime binding
  node((1, 1), align(center)[*omni.physx*\ #text(size: 7pt)[USD ↔ PhysX]],
    fill: faps-node-green, stroke: faps-node-green.darken(20%),
    width: 30mm, height: 14mm, name: <bind>),

  // PhysX solvers
  node((0, 2), align(center)[*Rigid Body*\ #text(size: 7pt)[(CPU)]],
    fill: faps-node-gray, stroke: faps-stroke,
    width: 24mm, height: 13mm, name: <rcpu>),
  node((0.8, 2), align(center)[*Rigid Body*\ #text(size: 7pt)[(GPU)]],
    fill: faps-node-gray, stroke: faps-stroke,
    width: 24mm, height: 13mm, name: <rgpu>),
  node((1.6, 2), align(center)[*Articulations*\ #text(size: 7pt)[(Reduced Coords)]],
    fill: faps-node-gray, stroke: faps-stroke,
    width: 27mm, height: 13mm, name: <art>),
  node((2.4, 2), align(center)[*PBD Particles*\ #text(size: 7pt)[(GPU)]],
    fill: faps-node-gray, stroke: faps-stroke,
    width: 24mm, height: 13mm, name: <pbd>),

  edge(<usd>, <bind>, "->"),
  edge(<bind>, <rcpu>, "->"),
  edge(<bind>, <rgpu>, "->"),
  edge(<bind>, <art>, "->"),
  edge(<bind>, <pbd>, "->"),
)


// ── 4. IsaacLab MDP Manager Decomposition ───────────────────
// Used in: Background § 2.3, Methodology § 4.1
#let mdp-managers() = diagram(
  spacing: (8mm, 10mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 0pt,
  edge-stroke: 0.8pt + faps-stroke,

  // Entry
  node((1.25, 0), align(center)[*env.step(actions)*],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
    width: 32mm, height: 12mm, name: <step>),

  // Managers row
  node((0, 1), align(center)[*Action*\ *Mgr*],
    fill: faps-node-lightgreen, stroke: faps-stroke,
    width: 18mm, height: 14mm, name: <act>),
  node((0.5, 1), align(center)[*Obs*\ *Mgr*],
    fill: faps-node-lightgreen, stroke: faps-stroke,
    width: 18mm, height: 14mm, name: <obs>),
  node((1, 1), align(center)[*Reward*\ *Mgr*],
    fill: faps-node-lightgreen, stroke: faps-stroke,
    width: 18mm, height: 14mm, name: <rew>),
  node((1.5, 1), align(center)[*Term*\ *Mgr*],
    fill: faps-node-lightgreen, stroke: faps-stroke,
    width: 18mm, height: 14mm, name: <term>),
  node((2, 1), align(center)[*Event*\ *Mgr*],
    fill: faps-node-lightgreen, stroke: faps-stroke,
    width: 18mm, height: 14mm, name: <event>),
  node((2.5, 1), align(center)[*Curric.*\ *Mgr*],
    fill: faps-node-lightgreen, stroke: faps-stroke,
    width: 18mm, height: 14mm, name: <curr>),

  // Physics sim
  node((1.25, 2), align(center)[*Physics Step*\ #text(size: 7pt)[PhysX]],
    fill: faps-node-green, stroke: faps-node-green.darken(20%),
    width: 28mm, height: 12mm, name: <sim>),

  // Output
  node((1.25, 3), align(center)[*Policy Input*\ #text(size: 7pt)[$o_t$, $r_t$, done]],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
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


// ── 5. Format Conversion Pipeline (URDF → USD) ──────────────
// Used in: Methodology § 4.3 (Robot Model Integration)
#let format-conversion-pipeline() = diagram(
  spacing: (16mm, 0mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 0pt,
  edge-stroke: 0.8pt + faps-stroke,

  // Sources
  node((0, 0), align(center)[*URDF*\ #text(size: 7pt)[Kinematics, Inertials,\ Joint Limits]],
    fill: faps-node-gray, stroke: faps-stroke,
    width: 30mm, height: 16mm, name: <urdf>),

  // Importer
  node((1, 0), align(center)[*URDF*\ *Importer*],
    fill: faps-node-green, stroke: faps-node-green.darken(20%),
    width: 22mm, height: 14mm, name: <imp>),

  // USD outputs
  node((2, -0.5), align(center)[*Robot Asset*\ #text(size: 7pt)[UsdPhysics]],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
    width: 28mm, height: 14mm, name: <robot>),
  node((2, 0), align(center)[*Environment*\ #text(size: 7pt)[Objects, Lights]],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
    width: 28mm, height: 14mm, name: <scene>),
  node((2, 0.5), align(center)[*PhysX Schema*\ #text(size: 7pt)[Solver Config]],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
    width: 28mm, height: 14mm, name: <physx>),

  edge(<urdf>, <imp>, "->"),
  edge(<imp>, <robot>, "->"),
  edge(<robot>, <scene>, "-", stroke: 0.4pt + faps-stroke),
  edge(<robot>, <physx>, "-", stroke: 0.4pt + faps-stroke),
)


// ── 6. Tendon Actuation Mapping ──────────────────────────────
// Used in: Background § 2.4, Methodology § 4.4
#let tendon-mapping() = diagram(
  spacing: (20mm, 12mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 0pt,
  edge-stroke: 0.8pt + faps-stroke,

  // Main chain
  node((0, 0), align(center)[*Tendon Space*\ #text(size: 7pt)[Spool lengths /\ Tensions]],
    fill: faps-node-gray, stroke: faps-stroke,
    width: 30mm, height: 16mm, name: <tendon>),
  node((1, 0), align(center)[*Joint Space*\ #text(size: 7pt)[$q_1, q_2, q_3$]],
    fill: faps-node-green, stroke: faps-node-green.darken(20%),
    width: 26mm, height: 14mm, name: <joint>),
  node((2, 0), align(center)[*Task Space*\ #text(size: 7pt)[EE Pose]],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
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
  spacing: (6mm, 10mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 0pt,
  edge-stroke: 0.8pt + faps-stroke,

  // Stage 1: Construction & Validation
  node((0, 0), align(center)[*Robot Model*\ #text(size: 7pt)[USD Assembly]],
    fill: faps-node-gray, stroke: faps-stroke,
    width: 26mm, height: 14mm, name: <model>),
  node((1, 0), align(center)[*Validation*],
    fill: faps-node-green, stroke: faps-node-green.darken(20%),
    width: 22mm, height: 14mm, name: <valid>),

  // Branch: actuation modes
  node((-0.3, 1), align(center)[#text(size: 7pt)[PD]], fill: faps-node-lightgreen, stroke: faps-stroke, width: 14mm, height: 10mm, name: <pd>),
  node((0.3, 1), align(center)[#text(size: 7pt)[Tendon]], fill: faps-node-lightgreen, stroke: faps-stroke, width: 14mm, height: 10mm, name: <ten>),
  node((0.9, 1), align(center)[#text(size: 7pt)[Physical]], fill: faps-node-lightgreen, stroke: faps-stroke, width: 14mm, height: 10mm, name: <phys>),

  // Stage 2: Progressive tasks
  node((1.7, 1), align(center)[*Reach*\ *Task*],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
    width: 22mm, height: 14mm, name: <reach>),
  node((2.5, 1), align(center)[*Cube Place*\ *Task*],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
    width: 22mm, height: 14mm, name: <place>),
  node((3.3, 1), align(center)[*Cube Sort*\ #text(size: 7pt)[(outlook)]],
    fill: white, stroke: (dash: "dashed", paint: faps-stroke, thickness: 0.5pt),
    width: 22mm, height: 14mm, name: <sort>),

  // Stage 3: Evaluation
  node((2.5, 2), align(center)[*Cross-Variant*\ *Evaluation*],
    fill: faps-node-green, stroke: faps-node-green.darken(20%),
    width: 30mm, height: 14mm, name: <eval>),

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


// ── 8. Reward Pipeline (Cube Place) ─────────────────────────
// Used in: Methodology § 4.7
#let reward-pipeline() = diagram(
  spacing: (12mm, 8mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 0pt,
  edge-stroke: 0.8pt + faps-stroke,

  // Phase 1: Approach & Grasp
  node((0, 0), align(center)[*Reach*\ #text(size: 7pt)[approach cube]],
    fill: faps-node-lightgreen, stroke: faps-stroke,
    width: 22mm, height: 12mm, name: <r1>),
  node((1, 0), align(center)[*Grasp*\ #text(size: 7pt)[close gripper]],
    fill: faps-node-lightgreen, stroke: faps-stroke,
    width: 22mm, height: 12mm, name: <r2>),
  node((2, 0), align(center)[*Lift*\ #text(size: 7pt)[height bonus]],
    fill: faps-node-lightgreen, stroke: faps-stroke,
    width: 22mm, height: 12mm, name: <r3>),

  // Latch
  node((2.8, 0), align(center)[#text(size: 7pt, weight: "bold")[was_grasped\ LATCH]],
    fill: rgb(255, 235, 180), stroke: faps-stroke,
    width: 22mm, height: 12mm, name: <latch>),

  // Phase 2: Transport & Release (gated)
  node((3.6, 0), align(center)[*Transport*\ #text(size: 7pt)[to drum]],
    fill: faps-node-green, stroke: faps-node-green.darken(20%),
    width: 22mm, height: 12mm, name: <r4>),
  node((4.4, 0), align(center)[*Release*\ #text(size: 7pt)[in target]],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
    width: 22mm, height: 12mm, name: <r5>),

  // Success
  node((5.2, 0), align(center)[*Success*\ #text(size: 7pt)[$w = 100$]],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
    width: 20mm, height: 12mm, name: <succ>),

  edge(<r1>, <r2>, "->"),
  edge(<r2>, <r3>, "->"),
  edge(<r3>, <latch>, "->"),
  edge(<latch>, <r4>, "->", label-side: left),
  edge(<r4>, <r5>, "->"),
  edge(<r5>, <succ>, "->"),
)
