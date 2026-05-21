// plots.typ — Shared cetz-plot configuration for FAPS thesis plots
// ============================================================
// Provides consistent styling for all natively rendered plots.
// Import this file in any chapter that uses cetz-plot.
//
// Usage:
//   #import "../../../shared/formatting/plots.typ": *
//   #step-response-plot(data-files: (...), ...)

#import "@preview/cetz:0.5.0": canvas, draw
#import "@preview/cetz-plot:0.1.3": plot, chart
#import "colors.typ": *

// ── Color palette for plots (FAPS official palette) ─────────────────────────
#let plot-blue    = fau-blau             // FAU-Blau (0,47,108) — PD model
#let plot-green   = faps-gruen           // FAPS-Grün (151,193,57) — Tendon model
#let plot-amber   = sonderfa-orange      // Sonderfarbe Orange (245,130,31)
#let plot-red     = sonderfa-echtrot     // Sonderfarbe Echtrot (220,30,38)
#let plot-purple  = hausfarbe-tuertkis   // Hausfarbe Türkis (52,103,125) — Physical model
#let plot-teal    = tf-metallic          // TF Metallic (119,159,181)
#let plot-orange  = sonderfa-orange      // Sonderfarbe Orange (245,130,31)
#let plot-grey    = grau-1               // Grau 1 (149,162,171)

// Amplitude colors for multi-trace step response plots
#let amp-colors = (plot-blue, plot-green, plot-amber)

// Damping sweep colors (high D → low D: grey → teal → teal-dark → orange → blue → red)
#let damping-colors = (
  grau-1,              // D=120 (overdamped) — Grau 1
  tf-metallic,         // D=60  — TF Metallic
  hausfarbe-tuertkis,  // D=30  — Hausfarbe Türkis
  sonderfa-orange,     // D=25  — Sonderfarbe Orange
  fau-blau,            // D=20 (optimal) — FAU-Blau
  sonderfa-echtrot,    // D=15 (underdamped) — Sonderfarbe Echtrot
)

// ── Shared plot dimensions ──────────────────────────────────────────────────
// Based on thesis text width of ~160mm (6.3in) and A4 layout
#let plot-full-width   = 14     // cm — full text width plot
#let plot-half-width   = 6.6   // cm — half text width (for 2-column)
#let plot-height       = 5.0   // cm — default height
#let plot-small-height = 4.0   // cm — for compact subplots

// ── Shared axis styles ──────────────────────────────────────────────────────
#let axis-style = (
  stroke: 0.6pt + luma(60),
  tick: (stroke: 0.4pt + luma(80), length: 0.08),
  grid: (stroke: 0.3pt + luma(210)),
  label: (offset: 0.35),
)

// ── Common axis labels ──────────────────────────────────────────────────────
#let ax-time-s    = [Time (s)]
#let ax-angle-deg = [Angle (°)]

// ── Common axis tick formatters ─────────────────────────────────────────────
// Format large step counts as "Nk" (e.g. 50000 → "50k", 1.5e6 → "1500k").
#let kilo-format = v => {
  let k = v / 1000
  if k == calc.trunc(k) {
    str(calc.trunc(k)) + "k"
  } else {
    str(k) + "k"
  }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

// Parse a CSV file (skipping header) into an array of (x, y) tuples.
// col-x and col-y are 0-based column indices.
// Downsamples data for performance and better visual rendering via `step`.
#let parse-csv(raw, col-x: 0, col-y: 1, step: 50) = {
  let res = ()
  let n = raw.len()
  for i in range(1, n, step: step) {
    if raw.at(i).len() > calc.max(col-x, col-y) {
      res.push((float(raw.at(i).at(col-x)), float(raw.at(i).at(col-y))))
    }
  }
  // Ensure the last point is included for complete curves
  if n > 1 and calc.rem(n - 1, step) != 0 {
      res.push((float(raw.at(n - 1).at(col-x)), float(raw.at(n - 1).at(col-y))))
  }
  res
}

// Parse setpoint column from CSV data
#let parse-csv-setpoint(raw, col-x: 0, col-sp: 2, step: 50) = {
  let res = ()
  let n = raw.len()
  for i in range(1, n, step: step) {
    if raw.at(i).len() > calc.max(col-x, col-sp) {
      res.push((float(raw.at(i).at(col-x)), float(raw.at(i).at(col-sp))))
    }
  }
  if n > 1 and calc.rem(n - 1, step) != 0 {
      res.push((float(raw.at(n - 1).at(col-x)), float(raw.at(n - 1).at(col-sp))))
  }
  res
}
