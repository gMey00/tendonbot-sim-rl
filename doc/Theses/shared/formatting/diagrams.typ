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
// Shows the MDP interaction loop with manipulation-specific details.
#let rl-agent-env-loop() = diagram(
  spacing: (36mm, 12mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 3pt,
  edge-stroke: 0.8pt + faps-stroke,

  // Agent side
  node((0, 0), align(center)[*Agent (Policy)*\ $pi_theta (a_t | o_t)$\ #text(size: 7pt)[PPO actor–critic]],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
    width: 34mm, height: 18mm, name: <agent>),

  // Environment side
  node((1.2, 0), align(center)[*Environment*\ #text(size: 7pt)[Physics Sim (PhysX)\ + Task Logic\ state $s_t$]],
    fill: faps-node-green, stroke: faps-node-green.darken(20%),
    width: 34mm, height: 20mm, name: <env>),

  // Main loop edges — variable-only labels to avoid duplication with boxes
  edge(<agent>, <env>, "->", bend: -30deg, label-side: left,
    label: text(size: 7pt)[$a_t$]),
  edge(<env>, <agent>, "->", bend: -30deg, label-side: left,
    label: text(size: 7pt)[$o_(t+1)$, $r_t$]),

  // Detail boxes — manipulation-specific content
  node((-0.15, 1), align(center)[#text(size: 7pt, weight: "bold")[Actions]\ #text(size: 6.5pt)[joint position deltas,\ cable tensions,\ gripper command]],
    fill: faps-node-lightgreen, stroke: faps-stroke,
    width: 28mm, height: 16mm, name: <actbox>),

  node((0.5, 1), align(center)[#text(size: 7pt, weight: "bold")[Observations]\ #text(size: 6.5pt)[partial: $o_t subset.eq s_t$\ $bold(q)$, $dot(bold(q))$, EE pose,\ object state, gripper]],
    fill: faps-node-lightgreen, stroke: faps-stroke,
    width: 30mm, height: 18mm, name: <obsbox>),

  node((1.3, 1), align(center)[#text(size: 7pt, weight: "bold")[Reward]\ #text(size: 6.5pt)[gated phases:\ reach #sym.arrow.r grasp\ #sym.arrow.r transport #sym.arrow.r place\ shaping + sparse success]],
    fill: faps-node-lightgreen, stroke: faps-stroke,
    width: 28mm, height: 20mm, name: <rewbox>),

  // Detail link edges (thin)
  edge(<agent>, <actbox>, "->", stroke: 0.4pt + faps-stroke),
  edge(<obsbox>, <agent>, "->", stroke: 0.4pt + faps-stroke),
  edge(<env>, <rewbox>, "->", stroke: 0.4pt + faps-stroke),
  edge(<env>, <obsbox>, "->", stroke: 0.4pt + faps-stroke),
)


// ── 2. Isaac Sim / IsaacLab Software Stack ───────────────────
// Used in: Background § 2.3 (Isaac Sim & IsaacLab)
#let isaac-stack() = diagram(
  spacing: (40mm, 15mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 3pt,
  edge-stroke: 0.8pt + faps-stroke,

  // IsaacLab layer
  node((0.25, 0), align(center)[*MDP Task*\ *Environments*],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
    width: 33mm, height: 15mm, name: <tasks>),
  node((1, 0), align(center)[*Managers*\ #text(size: 7pt)[Obs / Act / Rew / Term]],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
    width: 33mm, height: 15mm, name: <mgrs>),
  node((1.75, 0), align(center)[*Training Utils*\ #text(size: 7pt)[Multi-GPU, Wrappers]],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
    width: 33mm, height: 15mm, name: <train>),

  // Isaac Sim layer
  node((0.25, 1), align(center)[*Extensions*\ #text(size: 7pt)[Cloner, ROS 2]],
    fill: faps-node-green, stroke: faps-node-green.darken(20%),
    width: 33mm, height: 15mm, name: <ext>),
  node((1, 1), align(center)[*RTX Rendering*\ #text(size: 7pt)[Sensors, Cameras]],
    fill: faps-node-green, stroke: faps-node-green.darken(20%),
    width: 33mm, height: 15mm, name: <rtx>),
  node((1.75, 1), align(center)[*PhysX 5 Backend*\ #text(size: 7pt)[Rigid, Art., PBD]],
    fill: faps-node-green, stroke: faps-node-green.darken(20%),
    width: 33mm, height: 15mm, name: <physx>),

  // Foundation layer
  node((0.5, 2), align(center)[*USD Scene Graph*\ #text(size: 7pt)[Data Layer]],
    fill: faps-node-gray, stroke: faps-stroke,
    width: 48mm, height: 13mm, name: <usd>),
  node((1.5, 2), align(center)[*CUDA / GPU*\ #text(size: 7pt)[Tensor Interface]],
    fill: faps-node-gray, stroke: faps-stroke,
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
  spacing: (25mm, 8mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 3pt,
  edge-stroke: 0.8pt + faps-stroke,

  // Scene authoring
  node((1.2, 0), align(center)[*USD Stage*\ #text(size: 7pt)[UsdPhysics + PhysXSchema]],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
    width: 40mm, height: 14mm, name: <usd>),

  // Runtime binding
  node((1.2, 1), align(center)[*omni.physx*\ #text(size: 7pt)[USD ↔ PhysX]],
    fill: faps-node-green, stroke: faps-node-green.darken(20%),
    width: 30mm, height: 14mm, name: <bind>),

  // PhysX solvers
  node((0.1, 2), align(center)[*Rigid Body*\ #text(size: 7pt)[(CPU)]],
    fill: faps-node-gray, stroke: faps-stroke,
    width: 30mm, height: 13mm, name: <rcpu>),
  node((0.8, 2), align(center)[*Rigid Body*\ #text(size: 7pt)[(GPU)]],
    fill: faps-node-gray, stroke: faps-stroke,
    width: 30mm, height: 13mm, name: <rgpu>),
  node((1.6, 2), align(center)[*Articulations*\ #text(size: 7pt)[(Reduced Coords)]],
    fill: faps-node-gray, stroke: faps-stroke,
    width: 30mm, height: 13mm, name: <art>),
  node((2.4, 2), align(center)[*PBD Particles*\ #text(size: 7pt)[(GPU)]],
    fill: faps-node-gray, stroke: faps-stroke,
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
  node-corner-radius: 3pt,
  edge-stroke: 0.8pt + faps-stroke,

  // Entry
  node((2.5, 0), align(center)[*env.step(actions)*],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
    width: 32mm, height: 12mm, name: <step>),

  // Managers row
  node((0, 1), align(center)[*Action*\ *Mgr*],
    fill: faps-node-lightgreen, stroke: faps-stroke,
    width: 18mm, height: 14mm, name: <act>),
  node((1, 1), align(center)[*Obs*\ *Mgr*],
    fill: faps-node-lightgreen, stroke: faps-stroke,
    width: 18mm, height: 14mm, name: <obs>),
  node((2, 1), align(center)[*Reward*\ *Mgr*],
    fill: faps-node-lightgreen, stroke: faps-stroke,
    width: 18mm, height: 14mm, name: <rew>),
  node((3, 1), align(center)[*Term*\ *Mgr*],
    fill: faps-node-lightgreen, stroke: faps-stroke,
    width: 18mm, height: 14mm, name: <term>),
  node((4, 1), align(center)[*Event*\ *Mgr*],
    fill: faps-node-lightgreen, stroke: faps-stroke,
    width: 18mm, height: 14mm, name: <event>),
  node((5, 1), align(center)[*Curric.*\ *Mgr*],
    fill: faps-node-lightgreen, stroke: faps-stroke,
    width: 18mm, height: 14mm, name: <curr>),

  // Physics sim
  node((2.5, 2), align(center)[*Physics Step*\ #text(size: 7pt)[PhysX]],
    fill: faps-node-green, stroke: faps-node-green.darken(20%),
    width: 28mm, height: 12mm, name: <sim>),

  // Output
  node((2.5, 3), align(center)[*Policy Input*\ #text(size: 7pt)[$o_t$, $r_t$, done]],
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


// ── 6. Tendon Actuation Mapping ──────────────────────────────
// Used in: Background § 2.4, Methodology § 4.4
#let tendon-mapping() = diagram(
  spacing: (32mm, 12mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 3pt,
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
  spacing: (20mm, 8mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 3pt,
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
  spacing: (10mm, 8mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 3pt,
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
  node((3, 0), align(center)[#text(size: 7pt, weight: "bold")[was_grasped\ LATCH]],
    fill: rgb(255, 235, 180), stroke: faps-stroke,
    width: 22mm, height: 12mm, name: <latch>),

  // Phase 2: Transport & Release (gated)
  node((4, 0), align(center)[*Transport*\ #text(size: 7pt)[to drum]],
    fill: faps-node-green, stroke: faps-node-green.darken(20%),
    width: 22mm, height: 12mm, name: <r4>),
  node((5, 0), align(center)[*Release*\ #text(size: 7pt)[in target]],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
    width: 22mm, height: 12mm, name: <r5>),

  // Success
  node((6, 0), align(center)[*Success*\ #text(size: 7pt)[$w = 100$]],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
    width: 20mm, height: 12mm, name: <succ>),

  edge(<r1>, <r2>, "->"),
  edge(<r2>, <r3>, "->"),
  edge(<r3>, <latch>, "->"),
  edge(<latch>, <r4>, "->", label-side: left),
  edge(<r4>, <r5>, "->"),
  edge(<r5>, <succ>, "->"),
)


// ── 9. Mesh Processing Pipeline (CAD → USD → Isaac Lab) ─────
// Used in: Methodology § 4.3 (Simulation Model Construction)
#let mesh-processing-pipeline() = diagram(
  spacing: (15mm, 12mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 3pt,
  edge-stroke: 0.8pt + faps-stroke,

  // Source
  node((0, 0), align(center)[*Fusion 360*\ #text(size: 7pt)[CAD Assembly]],
    fill: faps-node-gray, stroke: faps-stroke,
    width: 26mm, height: 14mm, name: <cad>),

  // Blender processing
  node((1, 0), align(center)[*Blender*\ #text(size: 7pt)[FBX Import]],
    fill: faps-node-lightgreen, stroke: faps-stroke,
    width: 24mm, height: 14mm, name: <blender>),
  node((1, 1), align(center)[#text(size: 7pt)[Join Parts &\ Fix Origins]],
    fill: faps-node-lightgreen, stroke: faps-stroke,
    width: 22mm, height: 12mm, name: <join>),
  node((1, 2), align(center)[#text(size: 7pt)[Decimation &\ Collision Meshes]],
    fill: faps-node-lightgreen, stroke: faps-stroke,
    width: 22mm, height: 12mm, name: <decim>),

  // USD Builder
  node((2, 0), align(center)[*USD Builder*\ *Script*\ #text(size: 7pt)[Python + Kit]],
    fill: faps-node-green, stroke: faps-node-green.darken(20%),
    width: 26mm, height: 16mm, name: <builder>),
  node((2, 1), align(center)[#text(size: 7pt)[ArticulationRoot\ RigidBody APIs]],
    fill: faps-node-green, stroke: faps-node-green.darken(20%),
    width: 24mm, height: 12mm, name: <apis>),
  node((2, 2), align(center)[#text(size: 7pt)[Mass / Inertia /\ Joint Properties]],
    fill: faps-node-green, stroke: faps-node-green.darken(20%),
    width: 24mm, height: 12mm, name: <props>),

  // Assembly
  node((3, 0), align(center)[*Robot*\ *Assembler*\ #text(size: 7pt)[Base + Arm +\ Gripper]],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
    width: 26mm, height: 16mm, name: <assembler>),

  // Output
  node((4, 0), align(center)[*Isaac Lab*\ *Asset*],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
    width: 22mm, height: 14mm, name: <asset>),

  edge(<cad>, <blender>, [FBX], "->"),
  edge(<blender>, <join>, "->"),
  edge(<join>, <decim>, "->"),
  edge(<decim>, <builder>, "->"),
  edge(<blender>, <builder>, [USDC], "->"),
  edge(<builder>, <apis>, "->"),
  edge(<apis>, <props>, "->"),
  edge(<props>, <assembler>, "->"),
  edge(<builder>, <assembler>, [USD], "->"),
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
  node((0, 0), $A$, fill: faps-node-gray, stroke: faps-stroke,
    width: 8mm, height: 8mm, name: <A>),
  node((1, 0), $B$, fill: faps-node-gray, stroke: faps-stroke,
    width: 8mm, height: 8mm, name: <B>),
  node((0.2, 1), $D$, fill: faps-node-green, stroke: faps-node-green.darken(20%),
    width: 8mm, height: 8mm, name: <D>),
  node((0.8, 1), $C$, fill: faps-node-green, stroke: faps-node-green.darken(20%),
    width: 8mm, height: 8mm, name: <C>),

  // Frame (base link)
  edge(<A>, <B>, [frame $k_e$], "=", stroke: 1.2pt + faps-stroke, label-side: right),
  // Coupler (platform link)
  edge(<D>, <C>, [coupler $k_e$], "=", stroke: 1.2pt + faps-node-green.darken(20%), label-side: left),
  // Crossed side links
  edge(<A>, <C>, [$l_e$], "-", stroke: 0.8pt + faps-node-blue),
  edge(<B>, <D>, [$l_e$], "-", stroke: 0.8pt + faps-node-blue),
)


// ── 11. Kinematic Chain of the 5‑DOF Manipulator ────────────
// Used in: Methodology § 4.2 (Physical Robot Design)
#let kinematic-chain-diagram() = diagram(
  spacing: (26mm, 8mm),
  node-stroke: 0.6pt + faps-stroke,
  node-corner-radius: 3pt,
  edge-stroke: 0.8pt + faps-stroke,

  // Ceiling mount (world)
  node((1, 0), align(center)[*Ceiling*\ #text(size: 7pt)[World Frame]],
    fill: faps-node-gray, stroke: faps-stroke,
    width: 24mm, height: 12mm, name: <world>),

  // Prismatic base
  node((0.5, 1), align(center)[*Base Y*\ #text(size: 7pt)[Prismatic]],
    fill: faps-node-lightgreen, stroke: faps-stroke,
    width: 24mm, height: 12mm, name: <baseY>),
  node((1.5, 1), align(center)[*Base Z*\ #text(size: 7pt)[Prismatic]],
    fill: faps-node-lightgreen, stroke: faps-stroke,
    width: 24mm, height: 12mm, name: <baseZ>),

  // Elbow
  node((1, 2), align(center)[*Elbow*\ #text(size: 7pt)[Revolute, 1 DoF]],
    fill: faps-node-green, stroke: faps-node-green.darken(20%),
    width: 24mm, height: 12mm, name: <elbow>),

  // Wrist
  node((0.5, 3), align(center)[*Wrist Pitch*\ #text(size: 7pt)[Revolute]],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
    width: 24mm, height: 12mm, name: <pitch>),
  node((1.5, 3), align(center)[*Wrist Roll*\ #text(size: 7pt)[Revolute]],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
    width: 24mm, height: 12mm, name: <roll>),

  // Gripper
  node((1, 4), align(center)[*Robotiq 2F-140*\ #text(size: 7pt)[Gripper]],
    fill: faps-node-gray, stroke: faps-stroke,
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
  node-corner-radius: 3pt,
  edge-stroke: 0.8pt + faps-stroke,

  // RL policy output
  node((0, 0), align(center)[*Policy*\ #text(size: 7pt)[$bold(a)_t in [-1, 1]^5$]],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
    width: 24mm, height: 16mm, name: <policy>),

  // Tension scaling
  node((1, 0), align(center)[*Tension*\ *Scaling*\ #text(size: 7pt)[$bold(T) = bold(a)_t dot T_max$]],
    fill: faps-node-lightgreen, stroke: faps-stroke,
    width: 26mm, height: 16mm, name: <scale>),

  // J^T mapping
  node((2, 0), align(center)[*$J^top$ Mapping*\ #text(size: 7pt)[$bold(tau) = J^top bold(T)$]],
    fill: faps-node-green, stroke: faps-node-green.darken(20%),
    width: 30mm, height: 16mm, name: <jt>),

  // Split
  node((3, -0.4), align(center)[#text(size: 7pt)[Elbow\ Torque]],
    fill: faps-node-gray, stroke: faps-stroke,
    width: 20mm, height: 11mm, name: <elbow>),
  node((3, 0.4), align(center)[#text(size: 7pt)[Wrist\ Torques]],
    fill: faps-node-gray, stroke: faps-stroke,
    width: 20mm, height: 11mm, name: <wrist>),

  // PhysX
  node((4, 0), align(center)[*PhysX*\ *Joints*],
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
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
  node-corner-radius: 3pt,
  edge-stroke: 0.8pt + faps-stroke,

  // Proximal motors
  node((0, 0), align(center)[*Motor 1*\ #text(size: 7pt)[Tension $T_1$]],
    fill: faps-node-gray, stroke: faps-stroke,
    width: 22mm, height: 12mm, name: <m1>),
  node((0, 1), align(center)[*Motor 2*\ #text(size: 7pt)[Tension $T_2$]],
    fill: faps-node-gray, stroke: faps-stroke,
    width: 22mm, height: 12mm, name: <m2>),

  // Joint
  node((1, 0.5), align(center)[*Revolute*\ *Joint*\ #text(size: 7pt)[1 DoF]],
    fill: faps-node-green, stroke: faps-node-green.darken(20%),
    width: 24mm, height: 16mm, name: <joint>),

  // Distal link
  node((2, 0.5), align(center)[*Distal*\ *Link*\ #text(size: 7pt)[low inertia]],
    fill: faps-node-lightgreen, stroke: faps-stroke,
    width: 22mm, height: 14mm, name: <link>),

  // Result
  node((1, 1.5), align(center)[#text(size: 7pt, weight: "bold")[Net torque]\ #text(size: 6.5pt)[$tau = r(T_1 - T_2)$\ stiffness $prop T_1 + T_2$]],
    fill: white, stroke: (dash: "dashed", paint: faps-stroke, thickness: 0.4pt),
    width: 34mm, height: 14mm, name: <result>),

  edge(<m1>, <joint>, [cable 1], "->", label-side: left,
    stroke: 0.8pt + faps-node-blue),
  edge(<m2>, <joint>, [cable 2], "->", label-side: left,
    stroke: 0.8pt + faps-node-blue),
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
  node-corner-radius: 3pt,
  edge-stroke: 0.8pt + faps-stroke,

  // Core principle boxes
  node((0, 0), align(center)[*Compression*\ *Elements*\ #text(size: 7pt)[struts, rigid bars]],
    fill: faps-node-gray, stroke: faps-stroke,
    width: 28mm, height: 16mm, name: <comp>),
  node((1, 0), align(center)[*Tension*\ *Network*\ #text(size: 7pt)[cables, springs]],
    fill: faps-node-lightgreen, stroke: faps-stroke,
    width: 28mm, height: 16mm, name: <tens>),

  // Resulting structure
  node((0.5, 1), align(center)[*Tensegrity*\ *Structure*\ #text(size: 7pt)[self-stressed equilibrium]],
    fill: faps-node-green, stroke: faps-node-green.darken(20%),
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
    fill: faps-node-blue, stroke: faps-node-blue.darken(30%),
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
