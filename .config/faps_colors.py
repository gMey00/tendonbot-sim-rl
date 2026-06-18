"""FAPS thesis colour palette.

Official FAPS/FAU colours from the FAPS PowerPoint style guide (2022) and
the FAU corporate design manual.  All values are taken directly from the
approved colour swatches shown in the FAPS slide template.

Cycle order: FAU-Blau hell → FAPS-Grün → Sonderfa.Orange → Sonderfa.Echtrot
             → Türkis dunkel → Hausfarbe Türkis → Sonderfa.Gelb → Graudunkel
"""

from __future__ import annotations

# ── Official FAPS/FAU colours ─────────────────────────────────────────────────
# Green family
FAPS_GREEN = "#97C139"       # FAPS-Grün          RGB(151, 193, 57)   – primary green
FAPS_GREEN_DARK = "#6A8A22"  # FAPS-Grün dunkel   RGB(106, 138, 34)
FAPS_GREEN_LIGHT = "#C5DE89" # FAPS-Grün hell     RGB(197, 222, 137)

# Blue / Teal family (FAU brand colours)
FAPS_BLUE = "#6C8CC7"        # FAU-Blau hell      RGB(108, 140, 199)  – primary blue
FAU_BLAU = "#002F6C"         # FAU-Blau           RGB(0,   47,  108)
FAU_BLAU_DUNKEL = "#041E42"  # FAU-Blau dunkel    RGB(4,   30,   66)
TEAL = "#34677D"             # Hausfarbe Türkis   RGB(52,  103, 125)
TEAL_DARK = "#004359"        # Türkis dunkel      RGB(0,   67,   89)
TF_METALLIC = "#779FB5"      # TF Metallic        RGB(119, 159, 181)

# Grey family
FAPS_DARK_GREY = "#5F5F5F"   # fapsgraudunkel     RGB(95,  95,   95)  – annotations/edges
FAPS_MID_GREY = "#B0BCC4"    # Grau 2             RGB(176, 188, 196)
FAPS_LIGHT_GREY = "#D1D9DE"  # Grau 3             RGB(209, 217, 222)

# Special / accent colours
AMBER = "#F5821F"            # Sonderfa. Orange   RGB(245, 130,  31)
MUTED_RED = "#DC1E26"        # Sonderfa. Echtrot  RGB(220,  30,  38)
DARK_ORANGE = "#FFCB00"      # Sonderfa. Gelb     RGB(255, 203,   0)
PURPLE = TEAL_DARK           # Türkis dunkel used as "third variant" accent

CYCLE_COLORS: list[str] = [
    FAPS_BLUE,      # 1  FAU-Blau hell  – primary blue
    FAPS_GREEN,     # 2  FAPS-Grün      – primary green
    AMBER,          # 3  Sonderfa. Orange
    MUTED_RED,      # 4  Sonderfa. Echtrot
    TEAL_DARK,      # 5  Türkis dunkel
    TEAL,           # 6  Hausfarbe Türkis
    DARK_ORANGE,    # 7  Sonderfa. Gelb
    FAPS_DARK_GREY, # 8  neutral grey
]

# ── Semantic aliases for specific plot types ─────────────────────────────────
# Model-validation step-response plots
KLEIN_COLOR = AMBER             # Klein (2023) reference data   – Sonderfa. Orange
PD_COLOR = FAPS_BLUE            # PD actuator model             – FAU-Blau hell
TENDON_COLOR = FAPS_GREEN       # tendon (elbow-approx) model   – FAPS-Grün
PHYSICAL_COLOR = TEAL_DARK      # physical tendon model         – Türkis dunkel

# Training reward line assignments
REWARD_COLOR = FAPS_BLUE
SUCCESS_COLOR = FAPS_GREEN
PENALTY_COLOR = MUTED_RED

# Workspace visualisation colormaps (sequential, perceptually uniform)
SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP = "RdBu_r"

# ── Matplotlib prop_cycle string (for .mplstyle import) ─────────────────────
CYCLER_STR = ", ".join(f"'{c}'" for c in CYCLE_COLORS)
