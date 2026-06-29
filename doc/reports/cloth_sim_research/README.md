# cloth_sim_research — package for an external research AI

We repeatedly failed to get a stable, flat, graspable cloth t-shirt in Isaac Sim 5.1
/ Isaac Lab. This folder packages everything a researching AI needs to find the
canonical fix.

**Start with [`RESEARCH_BRIEF.md`](RESEARCH_BRIEF.md)** — it states the task, the
stack, the asset, the six recurring failure modes (F1–F6), and the prioritized
questions to answer.

Then:
- `data/mesh_probe.txt` — proof the asset is an inflated 3-D shirt (panels ~0.25 m
  apart), not a flat pattern.
- `data/experiments_and_results.md` — every parameter set we tried and its measured
  outcome (divergence / sink / explosion / bunching).
- `references/REFERENCE_PARAMS.md` + the two reference garment loaders — the closest
  known-good Isaac Sim cloth pipelines (GarmentLab, DexGarmentLab).
- `code/` — our full implementation, the validation script, a one-file stability
  sandbox, and the mesh probe.

Note: `code/sandbox_stability_test.py` and `code/mesh_probe.py` write their output to
a hardcoded scratch path and must be run inside the Isaac Sim python env
(`conda activate env_isaaclab`); adapt the output path before running.
