"""Step-response time-trace figures for Appendix A.7.

Produces three grouped figures, one per actuation mode, each containing
panels for Elbow / Wrist-X / Wrist-Y at the recorded amplitudes:

* ``step_pd_grouped.png``        – PD ImplicitActuator
* ``step_tendon_grouped.png``    – Constant-tension tendon
* ``step_physical_grouped.png``  – Physical antiparallelogram tendon

Data is loaded directly from
``src/tensegrity_pick/outputs/model_validation/{pd,tendon,tendon_physical}/data/``
— no intermediate CSV files are required.

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


def _plot_traces(
    ax: plt.Axes,
    variant: str,
    joint: str,
    amps: tuple[int, ...],
    *,
    title: str,
    y_max: float | None = None,
) -> None:
    """Plot setpoint (dashed) + measured (solid) for each amplitude.

    Parameters
    ----------
    variant : ``'pd'``, ``'tendon'``, or ``'physical'``
    joint   : canonical joint name (``'elbow_joint'`` etc.)
    amps    : tuple of integer amplitudes to plot
    """
    for amp, color in zip(amps, AMP_COLORS):
        data = c.load_npz_trial(variant, joint, amp)
        if data is None:
            continue
        t = np.asarray(data["time_s"], dtype=float)
        actual = np.asarray(data["actual_deg"], dtype=float)
        setpoint = np.asarray(data["setpoint_deg"], dtype=float)
        ax.plot(t, setpoint, color=color, linestyle="--", linewidth=0.7, alpha=0.6)
        ax.plot(t, actual, color=color, linewidth=1.2, label=f"{amp}°")

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
    variant: str,
    suptitle: str | None,
    elbow_y_max: float = 45,
    wrist_y_max: float = 35,
) -> None:
    """Render a 3-panel (elbow / wrist-x / wrist-y) step-response figure."""
    fig, axes = plt.subplots(
        3, 1,
        figsize=c.pcfg.scaled(11, 11.5, scale=1.0),
        constrained_layout=True,
    )

    _plot_traces(axes[0], variant, "elbow_joint",   ELBOW_AMPS,
                 title="(a) Elbow joint", y_max=elbow_y_max)
    _plot_traces(axes[1], variant, "wrist_x_joint", WRIST_AMPS,
                 title="(b) Wrist X (pitch about x̂)", y_max=wrist_y_max)
    _plot_traces(axes[2], variant, "wrist_y_joint", WRIST_AMPS,
                 title="(c) Wrist Y (pitch about ŷ)", y_max=wrist_y_max)

    c.save(fig, c.APPENDIX_FIG_DIR / out_name, tight=False)


# Base joint amplitudes (mm).
BASE_Y_AMPS: Final = (50, 100, 200)
BASE_Z_AMPS: Final = (30, 60, 100)


def _plot_base_traces(
    ax: plt.Axes,
    joint: str,
    amps: tuple[int, ...],
    *,
    title: str,
) -> None:
    """Plot base-joint setpoint (dashed) + measured (solid) for each amplitude.

    The y-axis is labelled in mm because the base joints are prismatic.
    For ``base_z_joint`` the traces are translated so the step starts at
    zero, making the three amplitude sweeps directly comparable.
    """
    for amp, color in zip(amps, AMP_COLORS):
        data = c.load_npz_base_trial(joint, amp)
        if data is None:
            continue
        t = np.asarray(data["time_s"], dtype=float)
        actual = np.asarray(data["actual_deg"], dtype=float)
        setpoint = np.asarray(data["setpoint_deg"], dtype=float)
        # Zero-reference: subtract the value at t=0 so all sweeps start at 0.
        offset = setpoint[0]
        ax.plot(t, setpoint - offset, color=color, linestyle="--",
                linewidth=0.7, alpha=0.6)
        ax.plot(t, actual - offset, color=color, linewidth=1.2, label=f"{amp} mm")

    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Position / mm")
    ax.set_xlim(0, 4.0)
    ax.legend(loc="lower right", fontsize=8, title="Setpoint")


def _base_two_panel_figure(out_name: str, *, suptitle: str | None) -> None:
    """Render a 2-panel (base_y / base_z) base-joint step-response figure."""
    fig, axes = plt.subplots(
        2, 1,
        figsize=c.pcfg.scaled(11, 8.0, scale=1.0),
        constrained_layout=True,
    )

    _plot_base_traces(axes[0], "base_y_joint", BASE_Y_AMPS,
                      title="(a) Base Y joint (horizontal translation)")
    _plot_base_traces(axes[1], "base_z_joint", BASE_Z_AMPS,
                      title="(b) Base Z joint (vertical translation, gravity feedforward)")

    if suptitle and c.pcfg.SHOW_TITLES:
        fig.suptitle(suptitle, fontsize=11, fontweight="bold")

    c.save(fig, c.APPENDIX_FIG_DIR / out_name, tight=False)


def main() -> None:
    c.init()

    _three_panel_figure(
        "step_pd_grouped.png",
        variant="pd",
        suptitle="PD model — step responses",
    )
    _three_panel_figure(
        "step_tendon_grouped.png",
        variant="tendon",
        suptitle="Constant-tension tendon — step responses",
    )
    _three_panel_figure(
        "step_physical_grouped.png",
        variant="physical",
        suptitle="Physical antiparallelogram tendon — step responses",
    )
    _base_two_panel_figure(
        "step_base_grouped.png",
        suptitle="Prismatic base joints — step responses",
    )


if __name__ == "__main__":
    main()
