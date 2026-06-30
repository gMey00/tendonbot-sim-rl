# Alex Cluster — Quick Start for `tensegrity_pick`

[← Back to documentation index](README.md)

> This guide documents the cluster conventions a newcomer must know, the project's helper
> scripts (`tools/` + `.config/env_vars.sh`), and the everyday commands you'll use.
>
> The following assumes the **`rtxpro6k`** partition (8 × NVIDIA RTX PRO 6000 per
> node). For the verified hardware/software facts of those nodes, see the
> companion document [Alex RTX PRO 6000 Hardware & Environment](alex_rtxpro6k_hardware.md).
>
> Wherever a cluster-wide convention is stated, it is cited to the official
> NHR@FAU documentation so you can verify and go deeper.

---

## 0. The 10-minute version (TL;DR)

```bash
# 1. Log in (proxy-jump through the central dialog server is in your ~/.ssh/config)
ssh alex.nhr.fau.de

# 2. Get the code (on $HOME) and load the project environment
cd "$HOME/studentische-arbeiten"
source .config/env_vars.sh          # auto-detects the 'alex' profile

# 3. ONE-TIME: install Isaac Sim 5.1 + Isaac Lab onto $HPCVAULT (login node only!)
cd tools && bash ./install_IsaacLab_Alex.sh skrl

# 4. Submit a training job (1 GPU, 32 cores, 24 h)
./train_alex.sh Template-Tensegrity-Reach-v0 --headless --num_envs 4096

# 5. Watch it
squeue --me
./watch_alex.sh                     # auto-picks your running job, shows metrics
```

Everything below explains *why* each step looks the way it does, and what to do
when something goes wrong.

---

## 1. Before you start: account & access

FAU HPC accounts do **not** have access to Alex by default. You need an existing
HPC account (created by invitation through your chair's contact person in the
HPC-Portal), and then you must additionally request Alex access through the
Tier3 form and prove a need that goes beyond TinyGPU but stays below the NHR
thresholds.[^nhr-account][^nhr-tier3] HPC accounts created through the portal have
**no password** — authentication is by **SSH key only**, uploaded in the
HPC-Portal.[^nhr-faq]

After your key is uploaded it can take up to two hours to propagate, and after
account creation it takes until the next morning before all filesystem folders
and the Slurm database are ready — so be patient on day one.[^nhr-faq]

### 1.1 Connecting

Almost all NHR@FAU systems sit behind private IP addresses reachable only from
inside the FAU network, so connections are tunneled through the central dialog
server `csnhr.nhr.fau.de` using SSH's *proxy jump* feature.[^nhr-faq][^nhr-ssh]
Configure this once in your local `~/.ssh/config` (see the NHR@FAU SSH
guide[^nhr-ssh]); afterwards a plain

```bash
ssh alex.nhr.fau.de
```

redirects you to one of the two login nodes `alex[12]`.[^nhr-alex] The login nodes
have **no GPUs** — they are for editing, compiling, data management, and job
submission only.[^nhr-alex][^nhr-slurm]

> **Job monitoring (ClusterCockpit):** because portal accounts have no password,
> the per-job monitoring at `monitoring.nhr.fau.de` is reachable only by logging
> into the HPC-Portal and following the ClusterCockpit link to mint a
> session.[^nhr-faq]

---

## 2. The mental model: login node vs. compute node

| | **Login node** (`alex1`/`alex2`) | **Compute node** (`rtxpro6k`) |
| --- | --- | --- |
| GPUs | none | 8 × RTX PRO 6000 (you request 1) |
| Internet | **yes** (full) | **no direct internet** — proxy only |
| What it's for | editing, `git`, **installing**, submitting jobs | running training jobs |
| How you get there | `ssh` | Slurm only (`sbatch` / `salloc`) — never direct[^nhr-slurm] |

Two consequences drive almost every design decision in the helper scripts:

1. **You install on the login node, you train on the compute node.** The
   installer (`install_IsaacLab_Alex.sh`) must run on a login node because it
   needs internet to download Isaac Sim (~15 GB) and pip-install; the training
   wrapper (`train_alex.sh`) must run through Slurm because only compute nodes
   have GPUs.
2. **Compute nodes have no direct internet.** Isaac Sim makes startup network
   calls (asset-root check, extension registry). On Alex these must be routed
   through the cluster HTTP proxy or disabled, otherwise the job hangs. The
   project handles this in two places — see §6.3.

---

## 3. Filesystems: what goes where

NHR@FAU provides several filesystems with very different backup and size
properties; you must put data on the right one.[^nhr-fs] The project layout maps
directly onto them, and `.config/env_vars.sh` encodes that mapping.

| Filesystem (env var) | Backup / snapshots | Project use (`env_vars.sh`) |
| --- | --- | --- |
| `$HOME` (`/home/hpc`) | **backed up + snapshotted**, small quota[^nhr-fs] | **code** → `$HPC` / `$PROJECT_PATH` |
| `$HPCVAULT` (`/home/vault`) | backed up + snapshotted, large[^nhr-fs] | **Isaac Sim + Isaac Lab + conda** → `$VAULT` / `$INSTALL_PATH` |
| `$WORK` | **no backup, no snapshots**[^nhr-fs] | **training outputs + Slurm logs** → `$WORKDIR` / `$SLURM_LOG_DIR` |
| `$TMPDIR` (node-local NVMe) | wiped at job end[^nhr-fs] | **staged assets at runtime** (`--stage`) |

Key rules to internalize:

- **Never put anything large on `$HOME`.** Its quota is small *and* doubled by
  snapshots; that's why the installer forces conda, Isaac Sim, and all pip/conda
  caches onto `$HPCVAULT`.[^nhr-fs]
- **`$WORK` is not backed up.** Only put regenerable data there (logs,
  checkpoints you can reproduce). Anything you can't afford to lose belongs on
  `$HOME` or `$HPCVAULT`.[^nhr-fs]
- **`$TMPDIR` is the fast path for I/O-heavy training.** It's a node-local NVMe
  SSD created per job and deleted when the job ends; staging your `res/` assets
  there reduces pressure on `$WORK` and cuts I/O latency.[^nhr-fs][^nhr-staging]
  This is exactly what `train_alex.sh --stage` does.

Check your usage and quota anytime with:

```bash
shownicerquota.pl      # friendly per-filesystem view (NHR@FAU tool)[^nhr-fs]
quota -s               # generic Unix view[^nhr-fs]
```

---

## 4. The project environment: `.config/env_vars.sh`

Everything starts by sourcing the project's environment file. It lives at
`.config/env_vars.sh` in the repository and **auto-detects** whether you are on
Alex or on the FAPS workstation from the hostname (Alex is detected via the
presence of `$HPCVAULT` or an `alex*`/`login*` hostname):

```bash
source .config/env_vars.sh
# -> env_vars.sh: loading profile 'alex'
```

On the `alex` profile it sets, among others:

| Variable | Meaning | Default |
| --- | --- | --- |
| `HPC`, `VAULT`, `WORKDIR` | roots → `$HOME`, `$HPCVAULT`, `$WORK` | — |
| `INSTALL_PATH` | where Isaac Sim/Lab + conda live | `$VAULT/Isaac` |
| `CONDA_PATH` | private Miniconda fallback location | `$INSTALL_PATH/miniconda3` |
| `PROJECT_PATH` | repository root | `$HPC/$REPO_NAME` |
| `ISAACSIM_PATH`, `ISAACLAB_PATH` | Isaac install dirs | under `$INSTALL_PATH` |
| `ISAACLAB_ENV_NAME` | conda env name | `env_isaaclab` |
| `ALEX_GRES` | GPU resource string | `gpu:rtxpro6k:1` |
| `ALEX_CPUS_PER_GPU` | cores per GPU | `32` |
| `ALEX_TIME` | default wall time | `24:00:00` |
| `SLURM_LOG_DIR` | where job logs land | `$WORK/slurm_logs` |

> **Why 32 cores per GPU?** On Alex each GPU comes with a fixed share of the
> host's cores and memory.[^nhr-alex] On the `rtxpro6k` partition the default is
> `DefCpuPerGPU=32`, so requesting exactly 32 CPUs per GPU keeps you to one task
> per GPU and the matching host-RAM share. All the helper scripts use
> `ALEX_CPUS_PER_GPU` for this.

> **Override before sourcing**, e.g. to force a profile or change defaults:
> ```bash
> MACHINE=alex ALEX_TIME=12:00:00 source .config/env_vars.sh
> ```

The two other profiles (`faps_workstation`, and the `*` fallback) let the same
file be sourced unchanged on a workstation, which is why the helper scripts can
be developed locally and run on Alex without edits.

---

## 5. One-time setup: `tools/install_IsaacLab_Alex.sh`

Run this **once, on a login node**:

```bash
cd "$HOME/studentische-arbeiten/tools"
bash ./install_IsaacLab_Alex.sh skrl     # framework: rl_games|rsl_rl|sb3|skrl|robomimic|none
```

What it does, and the Alex-specific reasons behind each step:

1. **Sources `.config/env_vars.sh`** and redirects all pip/conda caches to
   `$HPCVAULT` so they don't blow the small `$HOME` quota.[^nhr-fs]
2. **Conda:** prefers the cluster-provided conda from `module load python`
   (NHR@FAU ships a Conda installation through the `python` module[^nhr-python]),
   and falls back to a private Miniconda on `$HPCVAULT` only if that's absent.
   Because the cluster conda is read-only, it redirects `envs_dirs` and
   `pkgs_dirs` onto `$HPCVAULT`.
3. **Loads `gcc/15.2.0`.** Isaac Sim 5.x native libraries need `GLIBCXX_3.4.30+`;
   loading a recent GCC module provides it. This is needed both for
   `post_install.sh` and at training runtime — which is why the training and
   watch scripts load it too. Modules are the standard way to set up toolchains
   on all NHR@FAU systems.[^nhr-modules]
4. **Downloads Isaac Sim 5.1.0** to `$HPCVAULT`, removes the ~15 GB archive
   immediately after unzip, and runs the (non-critical, headless) `post_install.sh`.
5. **Clones Isaac Lab**, symlinks `_isaac_sim`, and creates the conda env with
   **Python 3.11** (mandatory for Isaac Sim 5.x) via `isaaclab.sh --conda`, then
   `isaaclab.sh --install <framework>`. It `module load cmake` first so the
   Isaac Lab installer doesn't try the (unavailable) `sudo apt-get` fallback.
6. **Patches `user.config.json` for offline compute nodes** (see §6.3).
7. **Installs the project extension** (`pip install -e source/tensegrity_pick`)
   plus monitoring extras (`plotly`, `tbparse`, `rich`, `plotext`, …) and `btop`
   (via conda-forge, used by `watch_alex.sh`).
8. **Smoke-tests** with `scripts/list_envs.py`.

> If you hit a GLIBC error after the cluster's ongoing AlmaLinux 8 → 9
> migration, simply re-run the installer to refresh — the Isaac Sim binary build
> is compatible with both.

When it finishes, you can re-enter the environment anytime with:

```bash
source "$PROJECT_PATH/.config/env_vars.sh"
module load gcc/15.2.0
module load python && eval "$(conda shell.bash hook)"
conda activate "$ISAACLAB_ENV_NAME"
```

(That same activation block is embedded in the training and watch scripts, so for
normal use you don't run it by hand.)

---

## 6. Running training: `tools/train_alex.sh`

`train_alex.sh` is an `sbatch` wrapper that generates a compliant Alex job script
and submits it. It follows the NHR@FAU job-script conventions: `#!/bin/bash -l`
to initialize the module system, `--export=NONE` plus `unset SLURM_EXPORT_ENV`
for a clean reproducible environment, `--gres=gpu:rtxpro6k:1` for one GPU, and
`module load python` + `conda activate` for the runtime.[^nhr-alex][^nhr-slurm][^nhr-jobscripts]

### 6.1 Basic usage

```bash
# single run
./train_alex.sh Template-Tensegrity-Reach-v0 --headless --num_envs 4096

# 5-seed robustness study as a Slurm array (one task per seed)
./train_alex.sh -s "0 1 2 3 4" Template-Tensegrity-Reach-v0 --headless

# 12 h MAPPO cloth run with email notifications
./train_alex.sh -t 12:00:00 --algorithm MAPPO -m you@fau.de \
    Template-Tensegrity-Shirt-Sort-v0 --headless

# stage assets to node-local NVMe first (faster I/O for asset-heavy tasks)
./train_alex.sh --stage Template-Tensegrity-Reach-v0 --headless

# see the generated job script WITHOUT submitting
./train_alex.sh --dry-run Template-Tensegrity-Reach-v0 --headless
```

Options parsed by the wrapper (must come **before** the task name): `-s/--seeds`,
`-t/--time`, `-n/--num-envs`, `-i/--iters`, `-j/--job-name`, `-m/--mail`,
`--algorithm`, `--script`, `--stage`, `--dry-run`. Anything after the task is
forwarded verbatim to `train.py`.

> **Tip:** always start with `--dry-run` when you change options — it prints the
> exact `#SBATCH` directives that will be submitted, so you can sanity-check the
> partition, GPU count, core count, wall time, and array directive before
> spending queue time.

### 6.2 What the generated job script contains

- `--partition=rtxpro6k`, `--gres=gpu:rtxpro6k:1`, `--ntasks=1`,
  `--cpus-per-task=32`, `--time=<TIME>` — the standard single-GPU Alex
  allocation.[^nhr-alex]
- `--export=NONE` + `unset SLURM_EXPORT_ENV` — the NHR@FAU-recommended
  boilerplate for a clean environment whose modules still propagate to
  `srun`.[^nhr-slurm][^nhr-jobscripts]
- For `-s/--seeds`, a `#SBATCH --array=...` directive where
  `SLURM_ARRAY_TASK_ID` *is* the seed, and per-task logs named
  `…_%A_seed%a.out`. (Array jobs are the documented way to run parameter studies
  like a multi-seed sweep.[^nhr-slurm-adv])
- Optional `--stage` block copying `res/` into `$TMPDIR` and pointing
  `TENSEGRITY_ASSET_ROOT` at it.[^nhr-staging]
- The proxy exports of §6.3, then
  `srun --ntasks=1 python scripts/skrl/train.py --task=… --algorithm=… …`.

### 6.3 Offline compute nodes (important!)

Because compute nodes have no direct internet, Isaac Sim's startup network calls
must be handled. The project does this **twice, defensively**:

1. **At install time**, `install_IsaacLab_Alex.sh` patches
   `user.config.json` so `asset_root` points at the local `res/` directory and
   the extension-registry network checks are disabled.
2. **At run time**, `train_alex.sh` exports the cluster HTTP proxy
   (`http_proxy=http://proxy.nhr.fau.de:80`, plus the upper-case and `no_proxy`
   variants) so any remaining startup call is routed through the proxy instead of
   hanging.

If a job *hangs at startup with no GPU activity*, a missing/incorrect proxy or an
unpatched `user.config.json` is the first thing to check.

---

## 7. Watching a job: `tools/watch_alex.sh`

To inspect a *running* job without consuming a second GPU allocation, attach to
its compute node with the NHR@FAU-documented overlap mechanism
`srun --jobid=<id> --overlap --pty …`.[^nhr-attach] `watch_alex.sh` wraps this:

```bash
./watch_alex.sh                 # auto-pick your single running job, show metrics
./watch_alex.sh 1234567 btop    # btop system+GPU monitor on that job's node
./watch_alex.sh 1234567_3 gpu   # watch nvidia-smi for array task / seed 3
./watch_alex.sh 1234567 shell   # plain interactive shell on the node
```

Modes:

| Mode | What you get |
| --- | --- |
| `train` (default) | the project TUI training monitor (`python -m tools.monitor`) |
| `btop` | `btop` system + GPU resource monitor (installed in the conda env) |
| `gpu` | lightweight `watch -n 2 nvidia-smi` |
| `shell` | interactive shell on the node (for ad-hoc inspection) |

If you omit the job ID and have exactly one running job, it's auto-selected;
with several running jobs it lists them so you can pass the right ID. You can
also pass a mode alone (`./watch_alex.sh btop`) and let it auto-pick the job.
Detach with `Ctrl-C` / `q` — this leaves the training job itself untouched.

> The `btop` mode exposes `libnvidia-ml.so.1` via `LD_LIBRARY_PATH` so btop's
> GPU panel works on the node; if `btop` isn't present it falls back to `top`.

---

## 8. Everyday Slurm commands

These are the core commands you'll use daily (all standard Slurm, applicable on
every NHR@FAU cluster[^nhr-slurm]):

```bash
# submit (normally via the wrapper, but raw form:)
sbatch my_job.sbatch

# your queue (pending + running)
squeue --me

# only running jobs, just the IDs
squeue --me --states=RUNNING --noheader --format='%i'

# why is my job not starting? -> look at the NODELIST(REASON) column
squeue --me

# cancel a job (or an array element)
scancel <jobID>
scancel <jobID>_<arrayidx>

# cluster occupancy (affects your queue time)
sinfo

# interactive single-GPU shell for quick debugging (1 h)
salloc --gres=gpu:rtxpro6k:1 --cpus-per-task=32 --time=1:00:00
```

A few things worth knowing:

- The order of jobs in `squeue` does **not** reflect priority; priority depends
  on waiting time, partition, group, and recent usage (fairshare). If a job is
  stuck, the reason is in the `NODELIST(REASON)` column (e.g. `AssocGrpGRES`
  means all GPUs allotted to your group are currently in use).[^nhr-slurm-adv]
- On Alex, compute nodes are **shared** between jobs — only the GPUs themselves
  are always exclusive to your job.[^nhr-alex][^nhr-slurm-adv]
- Always **purge modules before `salloc`** (`module purge`) to avoid a polluted
  interactive environment.[^nhr-slurm]
- Don't run real work on the login nodes — they're for compiling and submitting
  only.[^nhr-slurm]

---

## 9. Troubleshooting cheat-sheet

| Symptom | Likely cause / fix |
| --- | --- |
| `Permission denied` / asks for password on login | SSH key not uploaded or not yet propagated (≤ 2 h); key not known on `csnhr` for the proxy jump.[^nhr-faq][^nhr-ssh] |
| Account exists but jobs fail the first day | Filesystem folders + Slurm DB are only ready the next morning after account creation.[^nhr-faq] |
| `GLIBCXX_3.4.30 not found` | `module load gcc/15.2.0` before activating conda; re-run the installer after the AlmaLinux 8→9 migration. |
| Job hangs at startup, GPU idle | Offline-node networking: check the proxy exports and the patched `user.config.json` (§6.3). |
| `$HOME` quota exceeded during install | Caches leaked to `$HOME`; ensure `PIP_CACHE_DIR`/`CONDA_PKGS_DIRS` point to `$HPCVAULT` (the installer sets this).[^nhr-fs] |
| Job won't start, `AssocGrpGRES` in `squeue` | Your group's GPU allocation is fully in use — wait or reduce concurrent jobs.[^nhr-slurm-adv] |
| `btop` mode shows no GPU / not found | falls back to `top`; ensure the conda env has `btop` (installed via conda-forge by the installer). |
| Lost a file on `$WORK` | No backup/snapshots on `$WORK` — only `$HOME`/`$HPCVAULT` have snapshots under `.snapshots/`.[^nhr-fs] |

---

## 10. Where to go next

- **Hardware/software facts to cite in the thesis:** companion doc
  [Alex RTX PRO 6000 Hardware & Environment](alex_rtxpro6k_hardware.md).
- **Acknowledging the resources in publications/thesis:** NHR@FAU asks you to
  acknowledge the compute resources; see the official wording.[^nhr-ack]
- **Deeper Slurm usage** (arrays, dependencies, job control): NHR@FAU Slurm
  docs.[^nhr-slurm][^nhr-slurm-adv]

---

## References

[^nhr-account]: NHR@FAU, *Getting an account* — accounts created by chair-contact
  invitation in the HPC-Portal; free of charge for publicly funded research and
  theses. <https://doc.nhr.fau.de/account/>

[^nhr-tier3]: NHR@FAU, *Application for FAU/Tier3 access to GPGPU cluster Alex* —
  Tier3 free basic tier on proof of need beyond TinyGPU; usage checked via
  ClusterCockpit. <https://hpc.fau.de/tier3-access-to-alex/>

[^nhr-faq]: NHR@FAU, *FAQs* — SSH-key-only auth (no password), proxy jump via
  `csnhr`, private IPs reachable only inside FAU, ClusterCockpit via the portal,
  next-morning account readiness, key propagation delay.
  <https://doc.nhr.fau.de/faq/>

[^nhr-ssh]: NHR@FAU, *Connecting via SSH (command line)* — `~/.ssh/config` proxy-
  jump setup through `csnhr.nhr.fau.de`.
  <https://doc.nhr.fau.de/access/ssh-command-line/>

[^nhr-alex]: NHR@FAU, *Alex* cluster documentation — login nodes have no GPU;
  per-GPU host-resource share; partition/`--gres` usage; compute nodes accessed
  only via Slurm. <https://doc.nhr.fau.de/clusters/alex/>

[^nhr-slurm]: NHR@FAU, *Slurm Batch system* — login nodes not for computation;
  `sbatch`/`salloc`/`srun`; `--export=NONE`/`unset SLURM_EXPORT_ENV`; `module
  purge` before `salloc`. <https://doc.nhr.fau.de/batch-processing/batch_system_slurm/>

[^nhr-slurm-adv]: NHR@FAU, *Slurm Advanced Topics* — job arrays for parameter
  studies; priority/fairshare and `NODELIST(REASON)` (e.g. `AssocGrpGRES`); node
  sharing on Alex; `--exclusive` billing.
  <https://doc.nhr.fau.de/batch-processing/advanced-topics-slurm/>

[^nhr-jobscripts]: NHR@FAU, *Slurm Job Script Examples* — GPU job scripts via
  `--gres=gpu:<n>`; recommended `--export=NONE` boilerplate.
  <https://doc.nhr.fau.de/batch-processing/job-script-examples-slurm/>

[^nhr-attach]: NHR@FAU, *Slurm Batch system — Attach to a running job* — attach
  with `srun --jobid=<id> --overlap --pty …` (e.g. to run `nvidia-smi`).
  <https://doc.nhr.fau.de/batch-processing/batch_system_slurm/#attach-to-a-running-job>

[^nhr-fs]: NHR@FAU, *File systems* — `$HOME`/`$HPCVAULT` backed up + snapshotted,
  `$WORK` not; quotas; `$TMPDIR` node-local NVMe wiped at job end; snapshots under
  `.snapshots/`; `shownicerquota.pl`/`quota -s`.
  <https://doc.nhr.fau.de/data/filesystems/>

[^nhr-staging]: NHR@FAU, *Using node-local SSDs (staging)* — stage data onto
  `$TMPDIR` to speed up training and reduce `$WORK` I/O pressure.
  <https://doc.nhr.fau.de/data/staging/>

[^nhr-python]: NHR@FAU, *Python, Conda* — a Conda installation is provided through
  the `python` module. <https://doc.nhr.fau.de/sdt/python/>

[^nhr-modules]: NHR@FAU, *Modules* — software (compilers, toolchains, libraries)
  is provided via environment modules. <https://doc.nhr.fau.de/environment/modules/>

[^nhr-ack]: NHR@FAU, *Acknowledging resource usage* — required acknowledgement
  wording for NHR@FAU compute resources.
  <https://doc.nhr.fau.de/acknowledgment/>