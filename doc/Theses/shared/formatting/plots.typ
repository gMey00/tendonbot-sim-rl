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

// ── Color palette for plots (matches faps_colors.py) ────────────────────────
#let plot-blue    = fapsblau         // #296193 — PD model
#let plot-green   = fapsgruen        // #97C139 — Tendon model
#let plot-amber   = rgb("#CC8800")   // Klein (2023) reference
#let plot-red     = rgb("#CC3333")   // alert / error
#let plot-purple  = rgb("#7755AA")   // physical tendon model
#let plot-teal    = rgb("#22AAAA")   // teal accent
#let plot-orange  = rgb("#DD6622")   // dark orange
#let plot-grey    = fapsgraudunkel   // neutral

// Amplitude colors for multi-trace step response plots
#let amp-colors = (plot-blue, plot-green, plot-amber)

// Damping sweep colors (high D → low D: grey → blue → green → amber → red)
#let damping-colors = (
  fapsgraudunkel,   // D=120 (overdamped)
  rgb("#22AAAA"),   // D=60
  rgb("#7755AA"),   // D=30
  plot-amber,       // D=25
  fapsblau,         // D=20 (optimal)
  rgb("#CC3333"),   // D=15 (underdamped)
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
