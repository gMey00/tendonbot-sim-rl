# Project-thesis plotting scripts

Python plotting utilities that render every data-driven figure in the
Results chapter (§5) and Appendices A.4 / A.5 / A.7 of the project
thesis.  The cetz-rendered originals were difficult to lay out within
the FAPS textwidth, so they were replaced by matplotlib figures that
are exported as PNG and embedded with `image()`.

## Layout

```
project_thesis/
├── assets/
│   ├── data/                        # ← input CSVs / JSON (read-only)
│   │   ├── step_response/
│   │   └── rl_training/<task>/<variant>/
│   └── figures/
│       ├── results/                 # ← Results §5 outputs
│       └── appendix/                # ← Appendix A.4–A.7 outputs
└── scripts/
    ├── README.md                    # this file
    ├── Makefile                     # convenience targets
    ├── build_all_plots.py           # one-shot regeneration
    ├── common.py                    # shared loaders, smoothing, helpers
    ├── plot_step_metrics.py         # Results — step-response bar overview
    ├── plot_step_responses.py       # Appendix A.7 — time-trace figures
    ├── plot_reach_results.py        # Results §5.2
    ├── plot_place_results.py        # Results §5.3
    ├── plot_appendix_reach.py       # Appendix A.4
    └── plot_appendix_place.py       # Appendix A.5
```

The repo-wide matplotlib style sheet, FAPS colour palette, and helper
functions for figure sizing live in [`/.config/`](../../../.config/)
(`plot_config.py`, `faps_colors.py`, `faps_thesis.mplstyle`).  All
scripts here import them via `common.py`, so the same look-and-feel is
shared across both theses *and* the Isaac Lab task plotting scripts in
`src/tensegrity_pick/scripts/`.

## Output figures

| File | Used in |
|------|---------|
| `figures/results/step_metrics_overview.png` | `chapters/5_0_Results.typ` — `<fig:step_metrics>` |
| `figures/results/reach_convergence.png`     | `chapters/5_0_Results.typ` — `<fig:reach_convergence>` |
| `figures/results/reach_comparison.png`      | `chapters/5_0_Results.typ` — `<fig:reach_comparison>` |
| `figures/results/place_convergence.png`     | `chapters/5_0_Results.typ` — `<fig:place_convergence>` |
| `figures/results/place_curriculum.png`      | `chapters/5_0_Results.typ` — `<fig:curriculum_effect>` |
| `figures/results/place_comparison.png`      | `chapters/5_0_Results.typ` — `<fig:place_comparison>` |
| `figures/appendix/step_pd_grouped.png`         | `appendices/A_7_StepResponses.typ` |
| `figures/appendix/step_tendon_grouped.png`     | `appendices/A_7_StepResponses.typ` |
| `figures/appendix/step_physical_grouped.png`   | `appendices/A_7_StepResponses.typ` |
| `figures/appendix/reach_reward_overview.png`   | `appendices/A_4_ReachMetrics.typ` |
| `figures/appendix/reach_components.png`        | `appendices/A_4_ReachMetrics.typ` |
| `figures/appendix/reach_diagnostics.png`       | `appendices/A_4_ReachMetrics.typ` |
| `figures/appendix/reach_diagnostics_extra.png` | `appendices/A_4_ReachMetrics.typ` |
| `figures/appendix/place_reward_overview.png`   | `appendices/A_5_PlaceMetrics.typ` |
| `figures/appendix/place_metrics.png`           | `appendices/A_5_PlaceMetrics.typ` |
| `figures/appendix/place_manipulation.png`      | `appendices/A_5_PlaceMetrics.typ` |
| `figures/appendix/place_placement.png`         | `appendices/A_5_PlaceMetrics.typ` |
| `figures/appendix/place_regularization.png`    | `appendices/A_5_PlaceMetrics.typ` |
| `figures/appendix/place_diagnostics.png`       | `appendices/A_5_PlaceMetrics.typ` |

## Usage

### Regenerate everything

```bash
# from anywhere in the repo
python doc/Theses/project_thesis/scripts/build_all_plots.py

# or via Make
cd doc/Theses/project_thesis/scripts && make
```

### Regenerate one section

```bash
cd doc/Theses/project_thesis/scripts
make results        # Results §5 only
make appendix       # Appendices A.4–A.7 only

# or invoke a single script
python plot_reach_results.py
```

### Clean

```bash
cd doc/Theses/project_thesis/scripts && make clean
```

### Adding a new plot

1. Add a `plot_<topic>.py` file with a top-level `main()`.
2. Inside, import `common as c`, call `c.init()`, and write outputs to
   `c.RESULTS_FIG_DIR / "<name>.png"` or `c.APPENDIX_FIG_DIR / "..."`.
3. Use the helpers in `common.py` (`load_step_csv`, `load_rl_csv`,
   `plot_variants_curve`, `add_curriculum_marker`, `missing_panel`).
4. Append the module name to the `MODULES` tuple in `build_all_plots.py`
   and add a target to `Makefile`.
5. Embed the figure in the relevant `.typ` chapter via
   `image("../../../assets/figures/.../<name>.png", width: ...)`.

## Dependencies

* `matplotlib` ≥ 3.8
* `numpy` ≥ 1.26
* `pandas` ≥ 2.2

These are the same versions used by the Isaac Lab task plotting
scripts; the `env_isaaclab` conda environment satisfies them.

## Why PNG, not PDF/SVG?

The FAPS template uses `\includegraphics`-style raster scaling at print
sizes of ~16 cm.  PNGs at 300 dpi yield a file ≈ 100 KB per figure,
load instantly in the Typst preview, and avoid the font-embedding
issues that vector exports of matplotlib figures occasionally cause
when the FAPS Helvetica substitute is unavailable on the build host.
If a vector export becomes necessary later, change `savefig.format` in
`/.config/faps_thesis.mplstyle` to `pdf`.
