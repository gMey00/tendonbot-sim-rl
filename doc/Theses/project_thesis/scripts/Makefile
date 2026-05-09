# `make` targets for project-thesis figure regeneration.
#
# Usage:
#   make            # regenerate every figure
#   make results    # only Results §5 figures
#   make appendix   # only Appendix A.4–A.7 figures
#   make clean      # remove generated PNGs
#
# A working Python environment with matplotlib, numpy, and pandas is
# required.  The plot configuration lives in ../../../../.config/.

PYTHON ?= python3
SCRIPT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
RESULTS_DIR := $(SCRIPT_DIR)../assets/figures/results
APPENDIX_DIR := $(SCRIPT_DIR)../assets/figures/appendix

.PHONY: all results appendix clean step-metrics step-traces \
        reach-results place-results appendix-reach appendix-place

all:
	@$(PYTHON) $(SCRIPT_DIR)build_all_plots.py

results: step-metrics reach-results place-results

appendix: step-traces appendix-reach appendix-place

step-metrics:
	@$(PYTHON) $(SCRIPT_DIR)plot_step_metrics.py

step-traces:
	@$(PYTHON) $(SCRIPT_DIR)plot_step_responses.py

reach-results:
	@$(PYTHON) $(SCRIPT_DIR)plot_reach_results.py

place-results:
	@$(PYTHON) $(SCRIPT_DIR)plot_place_results.py

appendix-reach:
	@$(PYTHON) $(SCRIPT_DIR)plot_appendix_reach.py

appendix-place:
	@$(PYTHON) $(SCRIPT_DIR)plot_appendix_place.py

clean:
	rm -f $(RESULTS_DIR)/*.png $(APPENDIX_DIR)/*.png
