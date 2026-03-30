"""FAPS thesis colour palette.

Official FAPS colours from ``doc/Theses/shared/formatting/Makros.tex``,
extended with colourblind-friendly secondary shades (Tol Bright inspired)
for multi-line plots.  The cycle order puts the most commonly needed
colours first: blue → green → amber → red → purple → teal → orange → grey.
"""

from __future__ import annotations

# ── Official FAPS corporate colours ──────────────────────────────────────────
FAPS_BLUE = "#296193"        # fapsblau  RGB(41, 97, 147)
FAPS_GREEN = "#97C139"       # fapsgruen RGB(151, 193, 57)
FAPS_DARK_GREY = "#5F5F5F"   # fapsgraudunkel RGB(95, 95, 95)
FAPS_MID_GREY = "#CCCCCC"    # dunkelgrau
FAPS_LIGHT_GREY = "#E6E6E6"  # hellgrau

# ── Extended cycle palette (colourblind-friendly) ────────────────────────────
# Eight well-spaced colours for line plots and bar charts.
# Order: anchor colours first, then high-contrast fills.
AMBER = "#CC8800"
MUTED_RED = "#CC3333"
PURPLE = "#7755AA"
TEAL = "#22AAAA"
DARK_ORANGE = "#DD6622"

CYCLE_COLORS: list[str] = [
    FAPS_BLUE,      # 1  primary
    FAPS_GREEN,     # 2  primary
    AMBER,          # 3  warm contrast
    MUTED_RED,      # 4  alert / error
    PURPLE,         # 5
    TEAL,           # 6
    DARK_ORANGE,    # 7
    FAPS_DARK_GREY, # 8  neutral
]

# ── Semantic aliases for specific plot types ─────────────────────────────────
# Model-validation step-response plots
KLEIN_COLOR = AMBER             # Klein (2023) reference data
PD_COLOR = FAPS_BLUE            # PD actuator model
TENDON_COLOR = FAPS_GREEN       # tendon (elbow-approx) model
PHYSICAL_COLOR = PURPLE         # physical tendon model

# Training reward line assignments
REWARD_COLOR = FAPS_BLUE
SUCCESS_COLOR = FAPS_GREEN
PENALTY_COLOR = MUTED_RED

# Workspace visualisation colormaps (sequential, perceptually uniform)
SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP = "RdBu_r"

# ── Matplotlib prop_cycle string (for .mplstyle import) ─────────────────────
CYCLER_STR = ", ".join(f"'{c}'" for c in CYCLE_COLORS)
