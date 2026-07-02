# Reach play-video rendering

Tooling to render short videos of the **trained** reach policies so they can be
inspected visually (e.g. to diagnose the diverged / weak variants).

> ## ⚠️ The Alex cluster cannot render Isaac Sim 5.1 — render on a workstation
>
> After extensive investigation (see [§ Why not on Alex](#why-not-on-alex)),
> rendering the trained policies **cannot be done on the Alex cluster**:
>
> 1. `rtxpro6k` (Blackwell) nodes ship a **compute-only** NVIDIA driver — no
>    OpenGL/Vulkan userspace at all — so the RTX renderer can't even start.
> 2. On `a40` nodes (which *do* have the graphics driver) rendering gets all the
>    way up — glibc, Vulkan and the GPU are fine inside the container — but
>    **Isaac Sim 5.1's RTX renderer segfaults on the cluster's driver 610.43**
>    (it expects ~535; 610 is a brand-new Blackwell-era driver newer than
>    Isaac Sim 5.1 supports). This is cluster-wide (same driver everywhere).
>
> Training is unaffected (it's PhysX/CUDA only, no RTX renderer). **Render the
> policies on a workstation whose GPU driver Isaac Sim 5.1 supports** (e.g. the
> FAPS workstation) using the workstation workflow below.

## Files

| file | what it does |
|------|--------------|
| `render_play.py`               | Thin variant of `../skrl/play.py`: loads the latest checkpoint, runs the policy deterministically, records **one** framed video. Adds CLI overrides for the camera + scene tiling (`--cam_eye/--cam_lookat/--cam_resolution/--env_spacing/--num_envs`) so the parallel robot copies are framed nicely without touching any task config. **Portable** — works anywhere Isaac Sim can render. |
| `render_reach_workstation.sh`  | **(use this)** Sequential local loop over the reach grid (no Slurm), one Isaac Sim process per variant. For rendering on a workstation GPU. |
| `sync_checkpoints.sh`          | Pull the trained reach checkpoints (~68 MB, gitignored) from Alex to the workstation. |
| `render_reach_alex.sh`         | Submits the grid as Slurm jobs on Alex. **Does not work** (kept for reference / for if the cluster driver is ever fixed). |
| `container/`                   | Apptainer container + a40 submitter built while diagnosing the cluster. Gets glibc 2.35 + Vulkan working on a40 but still hits the RTX-vs-driver crash. See `container/README` notes below. |

## Workstation workflow (the working path)

```bash
# 1) On the workstation: get the trained checkpoints from Alex (~68 MB)
cd <repo>/src/tensegrity_pick/scripts/rendering
./sync_checkpoints.sh <youruser>@alex.nhr.fau.de

# 2) Activate the Isaac Lab env
source <repo>/.config/env_vars.sh
conda activate "$ISAACLAB_ENV_NAME"

# 3) Render. Preview first:
./render_reach_workstation.sh --list
# one variant:
./render_reach_workstation.sh --arms "Kinova-F140" --spaces "joint"
# everything (24 variants, sequential):
./render_reach_workstation.sh
```

Videos land in each variant's own log dir:

```
logs/skrl/reach/<variant>/<run>/videos/play/rl-video-step-0.mp4
```

## Framing

Defaults render a small **2×2 grid of 4 robot copies** in a 3/4 view. All reach
arms mount at world `(0.75, 1.0, 0.75)` inside each env and the copies tile
around the world origin at `--env-spacing`, so one fixed world-space camera
frames every variant. Tunables (defaults shown):

```
--num-envs 4  --env-spacing 3.0  --video-length 1800   # 1800 steps @30fps = 60s
--cam-eye "6.5 -4.5 3.5"  --cam-lookat "0.75 1.0 1.0"  --resolution "1280 720"
```

If a particular arm (the tall UR10, say) is clipped or too small, re-render just
that one with an adjusted camera, e.g.
`--arms "UR10-F140" --spaces "joint" --cam-eye "8 -6 4.5" --cam-lookat "0.75 1.0 1.2"`.

## Notes

- Frame rate = env step rate: `sim.dt (1/60) × decimation (2)` = **30 fps**.
  Episodes are 6 s (180 steps) with one FK-sampled target each, so a 60 s video
  shows ~10 reach attempts per robot.
- `render_play.py` reuses `play.py`'s checkpoint discovery
  (`get_checkpoint_path` over `logs/skrl/reach/<variant>/.*_ppo_torch/checkpoints`)
  and the skrl 1.x→2.x `state_preprocessor` migration, so it loads the exact same
  trained weights `play.py` would.

## Why not on Alex

Full diagnosis captured in `../../../../doc/` and the `container/` scripts. Short
version, in the order the walls were hit:

1. **glibc**: Isaac Sim 5.1 render/USD libs need glibc ≥ 2.35; Alex is AlmaLinux
   9.8 = glibc 2.34. → fixed by the Apptainer container (`container/isaac_render.def`,
   Ubuntu 22.04).
2. **No Vulkan on rtxpro6k**: those nodes have a compute-only driver (no
   `libGLX_nvidia`, no Vulkan ICD). `apptainer --nv` can only forward a driver
   that exists on the host. → unfixable there; moved to `a40`.
3. **Vulkan ICD path on a40**: the host ICD uses an absolute `library_path`; the
   fix is a relative-path ICD (`container/nvidia_icd_relative.json`) so the loader
   finds the `--nv`-injected lib. → Vulkan then works in-container on a40.
4. **RTX renderer vs driver 610.43**: Isaac Sim 5.1's `librtx.scenedb.plugin.so`
   segfaults during startup on driver 610.43 (recommends 535). Cluster-wide, not
   fixable from user space. → render off-cluster.
