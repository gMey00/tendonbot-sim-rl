// Acronym System for Typst
// ============================================================
// Provides first-use expansion: "Reinforcement Learning (RL)" → "RL"
// and separate lists for abbreviations and symbols.
//
// Auto-filtering: print-abbreviations() and print-symbols() only list
// entries actually used via ac(), acs(), acl(), or acp() in the document.
// Uses metadata() + query() (not state.final()) to avoid layout non-convergence.

// Label used to tag all acronym-usage metadata elements
#let _ac-used-label = <_ac-used>

// Emit an invisible usage marker
#let _ac-marker(key) = [#metadata(key)#_ac-used-label]

// Acronym database: (key: (short, long, tag, plural))
#let acronyms = (
  // Abbreviations
  "RL":    (short: "RL",    long: "Reinforcement Learning",                     tag: "abbrev"),
  "PPO":   (short: "PPO",   long: "Proximal Policy Optimization",               tag: "abbrev"),
  "MDP":   (short: "MDP",   long: "Markov Decision Process",                    tag: "abbrev"),
  "GAE":   (short: "GAE",   long: "Generalised Advantage Estimation",           tag: "abbrev"),
  "PBD":   (short: "PBD",   long: "Position-Based Dynamics",                    tag: "abbrev"),
  "XPBD":  (short: "XPBD",  long: "Extended Position-Based Dynamics",           tag: "abbrev"),
  "VBD":   (short: "VBD",   long: "Vertex Block Descent",                       tag: "abbrev"),
  "FEM":   (short: "FEM",   long: "Finite Element Method",                      tag: "abbrev"),
  "MPM":   (short: "MPM",   long: "Material Point Method",                      tag: "abbrev"),
  "GPU":   (short: "GPU",   long: "Graphics Processing Unit",                   tag: "abbrev"),
  "ROS":   (short: "ROS",   long: "Robot Operating System",                     tag: "abbrev"),
  "CPU":   (short: "CPU",   long: "Central Processing Unit",                    tag: "abbrev"),
  "DoF":   (short: "DoF",   long: "Degree of Freedom",                          tag: "abbrev", plural: "Degrees of Freedom"),
  "URDF":  (short: "URDF",  long: "Unified Robot Description Format",           tag: "abbrev"),
  "USD":   (short: "USD",   long: "Universal Scene Description",                tag: "abbrev"),
  "TUI":   (short: "TUI",   long: "Text-based User Interface",                  tag: "abbrev"),
  "MJCF":  (short: "MJCF",  long: "MuJoCo XML Format",                         tag: "abbrev"),
  "CAD":   (short: "CAD",   long: "Computer-Aided Design",                      tag: "abbrev"),
  "PD":    (short: "PD",    long: "Proportional-Derivative",                    tag: "abbrev"),
  "PID":   (short: "PID",   long: "Proportional-Integral-Derivative",           tag: "abbrev"),
  "PGS":   (short: "PGS",   long: "Projected Gauss–Seidel",                    tag: "abbrev"),
  "FK":    (short: "FK",    long: "Forward Kinematics",                         tag: "abbrev"),
  "IK":    (short: "IK",    long: "Inverse Kinematics",                         tag: "abbrev"),
  "ICR":   (short: "ICR",   long: "Instantaneous Center of Rotation",           tag: "abbrev"),
  "CDPM":  (short: "CDPM",  long: "Cable-Driven Parallel Mechanism",            tag: "abbrev"),
  "POMDP": (short: "POMDP", long: "Partially Observable Markov Decision Process", tag: "abbrev"),
  "RGBD":  (short: "RGB-D", long: "Red-Green-Blue plus Depth",                  tag: "abbrev"),
  "DR":    (short: "DR",    long: "Domain Randomization",                       tag: "abbrev"),
  "MPC":   (short: "MPC",   long: "Model Predictive Control",                   tag: "abbrev"),
  "IL":    (short: "IL",    long: "Imitation Learning",                         tag: "abbrev"),
  "DVRK":  (short: "dVRK",  long: "da Vinci Research Kit",                     tag: "abbrev"),
  "XML":   (short: "XML",   long: "Extensible Markup Language",                 tag: "abbrev"),
  "API":   (short: "API",   long: "Application Programming Interface",          tag: "abbrev"),
  "SDF":   (short: "SDF",   long: "Signed Distance Field",                     tag: "abbrev"),
  "TGS":   (short: "TGS",   long: "Temporal Gauss–Seidel",                     tag: "abbrev"),
  "NRMSE": (short: "NRMSE", long: "Normalized Root-Mean-Square Error",          tag: "abbrev"),

  // Symbols
  "sym:gamma":   (short: $gamma$,                        long: "Discount factor",                              tag: "symbol"),
  "sym:theta":   (short: $theta$,                        long: "Policy parameters",                            tag: "symbol"),
  "sym:pi":      (short: $pi_theta$,          long: "Policy (parameterized by theta)", tag: "symbol"),
  "sym:J":       (short: $J(theta)$,                     long: "Expected discounted return",                   tag: "symbol"),
  "sym:Ahat":    (short: $hat(A)_t$,                     long: "Advantage estimator",                          tag: "symbol"),
  "sym:epsilon": (short: $epsilon$,                      long: "Clipping parameter (PPO)",                     tag: "symbol"),
  "sym:S":       (short: $cal(S)$,                       long: "State space",                                  tag: "symbol"),
  "sym:A":       (short: $cal(A)$,                       long: "Action space",                                 tag: "symbol"),
  "sym:r":       (short: $r(s_t, a_t)$,                  long: "Reward function",                              tag: "symbol"),
  "sym:h":       (short: $h$,                            long: "Simulation time step",                         tag: "symbol"),
  "sym:kj":      (short: $k_j$,                          long: "Stiffness parameter (PBD)",                    tag: "symbol"),
  "sym:M":       (short: $bold(M)$,                      long: "Mass matrix",                                  tag: "symbol"),
  "sym:w":       (short: $w(bold(q))$,                   long: "Yoshikawa manipulability",                     tag: "symbol"),
  "sym:J_mat":   (short: $bold(J)(bold(q))$,             long: "Geometric Jacobian",                           tag: "symbol"),
  "sym:kappa":   (short: $kappa_"inv"$,                  long: "Inverse condition number",                     tag: "symbol"),
  "sym:tau":     (short: $bold(tau)$,                    long: "Joint torque vector",                          tag: "symbol"),
  "sym:T":       (short: $bold(T)$,                      long: "Tendon tension vector",                        tag: "symbol"),
  "sym:Jt":      (short: $bold(J)^top$,                  long: "Tendon Jacobian transpose",                    tag: "symbol"),
)

// -- Acronym access functions -------------------------------------------------

// First-use expansion: "Long Form (Short)" → subsequent uses: "Short"
// Uses query() to detect previous markers for the same key, which is
// convergence-stable: markers are always emitted (in both branches), so
// their existence doesn't depend on expansion decisions from prior passes.
#let ac(key) = context {
  let entry = acronyms.at(key)
  let prev = query(selector(_ac-used-label).before(here()))
    .filter(m => m.value == key)
  if prev.len() > 0 {
    [#_ac-marker(key)#entry.short]
  } else {
    [#_ac-marker(key)#entry.long (#entry.short)]
  }
}

// Always short form — also emits usage marker
// Uses content block [..] so the marker element is emitted (not discarded)
#let acs(key) = [#_ac-marker(key)#acronyms.at(key).short]

// Always long form — also emits usage marker
#let acl(key) = [#_ac-marker(key)#acronyms.at(key).long]

// Plural form (first use expanded) — also emits usage marker
#let acp(key) = context {
  let entry = acronyms.at(key)
  let long-plural = if "plural" in entry { entry.plural } else { entry.long + "s" }
  let prev = query(selector(_ac-used-label).before(here()))
    .filter(m => m.value == key)
  if prev.len() > 0 {
    [#_ac-marker(key)#entry.short\s]
  } else {
    [#_ac-marker(key)#long-plural (#entry.short\s)]
  }
}

// Reset is a no-op now (kept for API compatibility); first-use detection
// is purely positional via query(), so there is nothing to reset.
#let ac-reset() = {}

// Register one or more symbol keys for the List of Symbols without showing
// any visible output. Use this for symbols that appear only inside math
// equations (where acs()/ac() cannot be used inline).
// Example: #reg-sym("sym:gamma", "sym:theta", "sym:pi")
#let reg-sym(..keys) = {
  for key in keys.pos() {
    _ac-marker(key)
  }
}

// -- Print lists (auto-filtered to used entries only) -------------------------
// query(_ac-used-label) returns all metadata elements emitted by ac/acs/acl/acp
// throughout the entire document. This is convergence-stable (unlike state.final()).

// Print list of abbreviations — only entries used in this document
#let print-abbreviations() = context {
  let used = query(_ac-used-label).map(m => m.value).dedup()
  let abbrevs = acronyms
    .pairs()
    .filter(((k, v)) => v.tag == "abbrev" and used.contains(k))
    .sorted(key: ((k, v)) => v.short)
  if abbrevs.len() > 0 {
    heading(numbering: none)[List of Abbreviations]
    for (key, entry) in abbrevs {
      grid(
        columns: (2.5cm, 1fr),
        gutter: 0.5cm,
        text(entry.short), text(entry.long),
      )
      v(0.1em)
    }
  }
}

// Print list of symbols — only entries used in this document
#let print-symbols() = context {
  let used = query(_ac-used-label).map(m => m.value).dedup()
  let syms = acronyms
    .pairs()
    .filter(((k, v)) => v.tag == "symbol" and used.contains(k))
    .sorted(key: ((k, v)) => str(k))
  if syms.len() > 0 {
    heading(numbering: none)[List of Symbols]
    for (key, entry) in syms {
      grid(
        columns: (2.5cm, 1fr),
        gutter: 0.5cm,
        entry.short, text(entry.long),
      )
      v(0.1em)
    }
  }
}
