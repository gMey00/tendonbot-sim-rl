"""Step-response time-trace figures for Appendix A.7.

Produces three grouped figures, one per actuation mode, each containing
panels for Elbow / Wrist-X / Wrist-Y at the recorded amplitudes:

* ``step_pd_grouped.png``        – PD ImplicitActuator
* ``step_tendon_grouped.png``    – Constant-tension tendon
* ``step_physical_grouped.png``  – Physical antiparallelogram tendon

Wrist-X traces have not been re-recorded for the new pipeline; those
panels render a grey "data not recorded" placeholder so the layout
stays consistent and readers immediately see what is missing.

Run::

    python doc/Theses/project_thesis/scripts/plot_step_responses.py
"""

from __future__ import annotations

from typing import Final

import matplotlib.pyplot as plt
import numpy as np

import common as c

# Per-joint amplitude sweeps (matching the on-disk filenames).
ELBOW_AMPS: Final = (20, 30, 40)
WRIST_AMPS: Final = (10, 20, 30)

AMP_COLORS: Final = (c.pcfg.FAPS_BLUE, c.pcfg.FAPS_GREEN, c.pcfg.AMBER)


def _plot_traces(ax: plt.Axes, csv_template: str, amps: tuple[int, ...],
                 *, title: str, y_max: float | None = None) -> None:
    """Plot setpoint (dashed) + measured (solid) for each amplitude."""
    for amp, color in zip(amps, AMP_COLORS):
        try:
            df = c.load_step_csv(csv_template.format(amp=amp))
        except FileNotFoundError:
            continue
        ax.plot(df["time_s"], df["setpoint_deg"],
                color=color, linestyle="--", linewidth=0.7, alpha=0.6)
        ax.plot(df["time_s"], df["actual_deg"],
                color=color, linewidth=1.2, label=f"{amp}°")

    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Angle / °")
    ax.set_xlim(0, 4.0)
    if y_max is not None:
        ax.set_ylim(-2, y_max)
    ax.legend(loc="lower right", fontsize=8, title="Setpoint")


def _three_panel_figure(
    out_name: str,
    *,
    suptitle: str | None,
    elbow_template: str | None,
    wrist_y_template: str | None,
    wrist_x_template: str | None = None,
    elbow_y_max: float = 45,
    wrist_y_max: float = 35,
    wrist_x_message: str = "Re-run on `wrist_x_joint` and re-export CSV traces.",
    elbow_message: str | None = None,
) -> None:
    fig, axes = plt.subplots(
        3, 1,
        figsize=c.pcfg.scaled(11, 11.5, scale=1.0),
        constrained_layout=True,
    )

    # (a) Elbow
    if elbow_template is not None:
        _plot_traces(axes[0], elbow_template, ELBOW_AMPS,
                     title="(a) Elbow joint", y_max=elbow_y_max)
    else:
        axes[0].set_title("(a) Elbow joint", fontsize=10, fontweight="bold", loc="left")
        c.missing_panel(axes[0], elbow_message or "")

    # (b) Wrist X (often missing)
    if wrist_x_template is not None:
        _plot_traces(axes[1], wrist_x_template, WRIST_AMPS,
                     title="(b) Wrist X (pitch about x̂)", y_max=wrist_y_max)
    else:
        axes[1].set_title("(b) Wrist X (pitch about x̂)",
                          fontsize=10, fontweight="bold", loc="left")
        c.missing_panel(axes[1], wrist_x_message)

    # (c) Wrist Y
    if wrist_y_template is not None:
        _plot_traces(axes[2], wrist_y_template, WRIST_AMPS,
                     title="(c) Wrist Y (pitch about ŷ)", y_max=wrist_y_max)
    else:
        axes[2].set_title("(c) Wrist Y (pitch about ŷ)",
                          fontsize=10, fontweight="bold", loc="left")
        c.missing_panel(axes[2], "Re-run on `wrist_y_joint` and re-export.")

    if suptitle and c.pcfg.SHOW_TITLES:
        fig.suptitle(suptitle, fontsize=11, fontweight="bold")

    c.save(fig, c.APPENDIX_FIG_DIR / out_name, tight=False)


def main() -> None:
    c.init()

    _three_panel_figure(
        "step_pd_grouped.png",
        suptitle="PD model — step responses",
        elbow_template="step_pd_elbow_{amp}.csv",
        wrist_y_template="step_pd_wrist_y_{amp}.csv",
    )
    _three_panel_figure(
        "step_tendon_grouped.png",
        suptitle="Constant-tension tendon — step responses",
        elbow_template="step_tendon_elbow_{amp}.csv",
        wrist_y_template="step_tendon_wrist_y_{amp}.csv",
    )
    _three_panel_figure(
        "step_physical_grouped.png",
        suptitle="Physical antiparallelogram tendon — step responses",
        elbow_template=None,
        wrist_y_template=None,
        elbow_message=(
            "Re-run `step_response` on the physical antiparallelogram "
            "elbow (Kp=75, Ki=6, Kd=3) at 20°, 30°, 40°."
        ),
        wrist_x_message=(
            "Wrist behaviour identical to constant-tension tendon — "
            "see `step_tendon_grouped`."
        ),
    )


if __name__ == "__main__":
    main()
