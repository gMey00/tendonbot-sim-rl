#import "@preview/cetz:0.3.4": canvas, draw
#import "@preview/cetz-plot:0.1.1": plot
#set page(width: 180mm, height: 120mm, margin: 10mm)
#set text(size: 9pt, font: ("Nimbus Sans", "Arial"))

// FAPS colors
#let plot-blue   = rgb(41, 97, 147)
#let plot-green  = rgb(151, 193, 57)
#let plot-amber  = rgb("#CC8800")

// Load CSV
#let load-xy(path, cx: 0, cy: 1) = {
  csv(path).slice(1).map(r => (float(r.at(cx)), float(r.at(cy))))
}

// Data paths
#let base = "shared/data/"

// SINGLE STEP RESPONSE TEST
#let d20-actual = load-xy(base + "step_pd_elbow_20.csv")
#let d20-sp     = load-xy(base + "step_pd_elbow_20.csv", cy: 2)
#let d30-actual = load-xy(base + "step_pd_elbow_30.csv")
#let d30-sp     = load-xy(base + "step_pd_elbow_30.csv", cy: 2)
#let d40-actual = load-xy(base + "step_pd_elbow_40.csv")
#let d40-sp     = load-xy(base + "step_pd_elbow_40.csv", cy: 2)

#canvas(length: 1cm, {
  import draw: *

  plot.plot(
    size: (14, 5),
    x-label: [Time (s)],
    y-label: [Angle (deg)],
    x-min: 0, x-max: 4.0,
    y-min: -2, y-max: 45,
    x-tick-step: 0.5,
    y-tick-step: 10,
    x-grid: true,
    y-grid: true,
    legend: "inner-north-east",
    {
      // Setpoints (dashed)
      plot.add(d20-sp, style: (stroke: (paint: plot-blue, dash: "dashed", thickness: 0.8pt)))
      plot.add(d30-sp, style: (stroke: (paint: plot-green, dash: "dashed", thickness: 0.8pt)))
      plot.add(d40-sp, style: (stroke: (paint: plot-amber, dash: "dashed", thickness: 0.8pt)))
      // Actuals (solid)
      plot.add(d20-actual, label: [20° actual], style: (stroke: (paint: plot-blue, thickness: 1.2pt)))
      plot.add(d30-actual, label: [30° actual], style: (stroke: (paint: plot-green, thickness: 1.2pt)))
      plot.add(d40-actual, label: [40° actual], style: (stroke: (paint: plot-amber, thickness: 1.2pt)))
    }
  )
})
