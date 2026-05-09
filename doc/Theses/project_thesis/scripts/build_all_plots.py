"""Single entry-point that regenerates every project-thesis figure.

Run from anywhere::

    python doc/Theses/project_thesis/scripts/build_all_plots.py
    # or:
    cd doc/Theses/project_thesis/scripts && python build_all_plots.py

The script invokes each ``plot_*.py`` in dependency-free order and
prints one progress line per generated PNG.
"""

from __future__ import annotations

import importlib
import time
from typing import Final

import common as c

MODULES: Final = (
    "plot_step_metrics",
    "plot_step_responses",
    "plot_reach_results",
    "plot_place_results",
    "plot_appendix_reach",
    "plot_appendix_place",
)


def main() -> None:
    c.init()
    print(f"Output dir (results): {c.RESULTS_FIG_DIR.relative_to(c.THESIS_DIR)}")
    print(f"Output dir (appendix): {c.APPENDIX_FIG_DIR.relative_to(c.THESIS_DIR)}")
    print()
    for name in MODULES:
        print(f"== {name} ==")
        t0 = time.perf_counter()
        mod = importlib.import_module(name)
        mod.main()
        print(f"   ({time.perf_counter() - t0:.1f} s)\n")
    print("All figures regenerated.")


if __name__ == "__main__":
    main()
