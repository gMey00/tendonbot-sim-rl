# Alex Cluster — RTX PRO 6000 Partition Hardware & Environment

[← Back to documentation index](README.md)

> **Goal**  Document the exact hardware and software environment of the new
> **`rtxpro6k`** GPU partition on the **Alex** cluster (NHR@FAU), for the
> *reproducibility* and *methodology* sections of the master's thesis,
> , and so that new students/coworkers can get productive on the
> same nodes quickly.
>
> All hardware facts below were gathered **directly from the cluster** on
> **2026-06-30** (Slurm `scontrol`/`sinfo`, `lscpu`, `nvidia-smi`, `lsblk`,
> `module avail`, `quota`). Items that cannot be read off the machine were 
> researched from official documentation and datasheets.

---

## 0. At a glance

| Property | Value |
| --- | --- |
| Cluster | Alex (NHR@FAU — Nationales Hochleistungsrechnen, FAU Erlangen-Nürnberg) |
| Partition | `rtxpro6k` |
| Login node used | `alex2.nhr.fau.de` |
| Nodes in partition | 11 (`a[2041,2043,2141-2143,2741,2841,2843,2941-2943]`) |
| GPUs per node | 8 × NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Total GPUs in partition | 88 |
| GPU memory | ~96 GB (97 887 MiB) per GPU, ECC enabled |
| GPU memory bandwidth | 1,597 GB/s per GPU (GDDR7, 512-bit) — see §1.1 |
| CPU per node | 2 × AMD EPYC 9745 (128-core "Turin"/Zen 5c) = 256 physical cores |
| RAM per node | ~760 GB usable (754 GiB physical) |
| Local scratch per node | ~28 TB NVMe RAID (`$TMPDIR`) |
| Max wall time | 24 h (`1-00:00:00`) |
| Max nodes per job | **1** (single-node partition) |
| OS | AlmaLinux 9.8 (kernel `5.14.0-687.12.1.el9_8`) |
| Scheduler | Slurm 25.11.4 |
| GPU driver / CUDA UMD | 610.43.02 / CUDA 13.3 |

---

## 1. GPU — NVIDIA RTX PRO 6000 Blackwell Server Edition

Read from `nvidia-smi` on compute node `a2843`:

| Property | Value (verified) |
| --- | --- |
| Marketing name | NVIDIA RTX PRO 6000 Blackwell **Server Edition** |
| Count per node | 8 |
| Onboard memory | 97 887 MiB ≈ **96 GB** (ECC **enabled**) |
| Architecture | Blackwell |
| Compute capability | **12.0** (`sm_120`) |
| Power cap | **600 W** per GPU |
| PCIe link | **Gen 5 × 16** (max) |
| VBIOS | `98.02.67.00.0A` |
| Driver version (KMD) | `610.43.02` |
| CUDA UMD version | **13.3** |
| MIG | Disabled |
| GPU-to-GPU interconnect | **PCIe only** — `nvidia-smi topo -m` shows **no NVLink** between GPUs on a node |

> **Note on GPU↔CPU affinity:** `nvidia-smi topo -m` reports the queried GPU
> bound to NUMA node 5 / a specific CPU affinity range. With 8 NUMA domains per
> socket pair (see §2), each GPU sits closest to one NUMA node — relevant if you
> pin CPU threads/dataloaders for performance.

### 1.1 Verified compute & memory specs (from NVIDIA datasheet)

The following are the official figures for the **Server Edition** from the NVIDIA
datasheet [^nv-server-ds] (cross-checked against the Lenovo ThinkSystem product
guide [^lenovo]):

| Property | Value (Server Edition) | Source |
| --- | --- | --- |
| CUDA cores | **24,064** | NVIDIA datasheet [^nv-server-ds] |
| Tensor cores | **752** (fifth-generation) | NVIDIA datasheet [^nv-server-ds] |
| RT cores | **188** (fourth-generation) | NVIDIA datasheet [^nv-server-ds] |
| Memory type | **GDDR7 with ECC** | NVIDIA datasheet [^nv-server-ds] |
| Memory interface | **512-bit** | NVIDIA datasheet [^nv-server-ds] |
| Memory bandwidth | **1,597 GB/s** (≈ 1.6 TB/s) | NVIDIA datasheet [^nv-server-ds] |
| FP32 (single-precision) | **120 TFLOPS** (peak) | NVIDIA datasheet [^nv-server-ds] |
| Peak FP4 AI | **4 PFLOPS** (effective FP4 with sparsity) | NVIDIA datasheet [^nv-server-ds] |
| RT Core performance | **355 TFLOPS** (peak) | NVIDIA datasheet [^nv-server-ds] |
| NVENC / NVDEC / JPEG | **4× / 4× / 4×** (9th-gen NVENC, 6th-gen NVDEC, 4:2:2 support) | NVIDIA datasheet [^nv-server-ds] |
| MIG | Up to **4 instances @ 24 GB** each | NVIDIA datasheet [^nv-server-ds] |
| Power consumption | **Up to 600 W (configurable)** | NVIDIA datasheet [^nv-server-ds] |
| Graphics bus | **PCIe 5.0 ×16** | NVIDIA datasheet [^nv-server-ds] |
| Power connector | 1× PCIe CEM5 16-pin | NVIDIA datasheet [^nv-server-ds] |
| Form factor | 4.4" (H) × 10.5" (L), dual-slot, passively cooled | NVIDIA datasheet [^nv-server-ds] |
| Confidential compute / secure boot | Supported / Yes | NVIDIA datasheet [^nv-server-ds] |

> **Note on the on-node `nvidia-smi` MIG reading:** the verified table in §1 above
> reports MIG as *Disabled*. That is the runtime configuration on the Alex nodes;
> the hardware itself **supports** MIG (up to 4 × 24 GB), it is simply not enabled
> in this partition.

**Architecture-level details (Blackwell GB202, fifth-gen Tensor / fourth-gen RT
cores):** The card is based on the Blackwell GB202 GPU. The fifth-generation
Tensor Cores add native **FP4** and **FP6** datatypes alongside FP8, TF32, BF16,
and FP16, and the Blackwell Tensor Core roughly doubles FP8 throughput versus the
previous generation [^nv-arch]. The fourth-generation RT Cores double the
ray-triangle intersection rate of the previous (Ada) generation [^nv-arch].

> **Note — per-precision Tensor throughput (FP16/BF16/TF32/FP8):** NVIDIA's
> *Server Edition* datasheet [^nv-server-ds] publishes only the headline FP32
> (120 TFLOPS), peak FP4 (4 PFLOPS), and RT (355 TFLOPS) numbers; it does **not**
> tabulate a full per-precision matrix (TF32 / BF16 / FP16 / FP8 dense-vs-sparse)
> the way the older A40/A100 datasheets did. If the thesis needs those exact
> intermediate-precision peaks, they are not available as verified vendor figures
> for the Server Edition and should either be omitted or derived (FP4 is the top
> rate; each step down in precision roughly halves it, and the non-sparse/dense
> rate is half of the with-sparsity rate). Treat any such derived number as an
> estimate, not a datasheet value.

### 1.2 Server Edition vs. Workstation Edition

Both editions use the same Blackwell GB202 GPU, 96 GB of GDDR7 with ECC, a 512-bit
interface, 24,064 CUDA cores, 752 Tensor cores, and 188 RT cores [^lenovo] [^nv-arch].
The relevant differences for this work:

| Property | Server Edition (Alex) | Workstation Edition |
| --- | --- | --- |
| Cooling | **Passive** (designed for server airflow / 24-7 data-center duty) | Active (blower/fan) |
| FP32 peak | **120 TFLOPS** | ~125 TFLOPS |
| Memory bandwidth | **1,597 GB/s** | ~1,792 GB/s (higher GDDR7 data rate) |
| Board length | 10.5" (fits 2U/3U servers) | ~12" |
| Power | Up to 600 W configurable (some OEM docs note a 400–600 W configurable range) | Up to 600 W |
| Target | Data-center / NVIDIA-Certified servers, MIG, confidential compute | Desktop workstation |

Sources: NVIDIA Server Edition datasheet [^nv-server-ds], Lenovo ThinkSystem
product guide [^lenovo], NVIDIA RTX Blackwell PRO architecture whitepaper [^nv-arch].
The slightly lower FP32 peak and memory bandwidth of the Server Edition relative to
the Workstation Edition stem from its lower clocks / GDDR7 data rate, a trade-off
for the passive, data-center-optimised thermal design [^lenovo].

> **Note on clocks/TDP:** NVIDIA does not publish base/boost clock figures for the
> Server Edition in the public datasheet [^nv-server-ds]; only the configurable
> power cap (up to 600 W) is given. The `nvidia-smi`-read 600 W cap in §1 is
> therefore the authoritative on-node value; exact base/boost clocks are not a
> verified vendor figure for this SKU.

**Sources / links for §1:**
- NVIDIA RTX PRO 6000 Blackwell Server Edition product page:
  <https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/>
- NVIDIA RTX PRO 6000 Blackwell Server Edition **datasheet (PDF)**:
  <https://www.akamai.com/site/en/documents/corporate/nvidia-rtx-pro-6000-blackwell-server-edition-datasheet.pdf>
- NVIDIA RTX Blackwell PRO GPU Architecture whitepaper (PDF):
  <https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf>
- Lenovo ThinkSystem NVIDIA RTX PRO 6000 Blackwell Server Edition product guide (full spec table):
  <https://lenovopress.lenovo.com/lp2263-thinksystem-nvidia-rtx-pro-6000-blackwell-server-edition-pcie-gen5-gpu>

---

## 2. CPU — AMD EPYC 9745 (per compute node)

Read from `lscpu` on compute node `a2843`:

| Property | Value (verified) |
| --- | --- |
| Model | **AMD EPYC 9745** (128-Core Processor) |
| Microarchitecture | Zen 5c ("Turin Dense") — CPU family 26, model 17, stepping 0 |
| Sockets | 2 |
| Cores per socket | 128 → **256 physical cores per node** |
| Threads per core | **1** (SMT/hyper-threading disabled in BIOS) |
| Logical CPUs | 256 |
| Max boost clock | ~3.71 GHz (`CPU max MHz` 3709) |
| Min clock | ~1.21 GHz; governor `performance` on compute nodes |
| NUMA domains | **8** (NPS configuration), 32 cores each: node0 `0-31` … node7 `224-255` |
| L1d / L1i cache | 12 MiB / 8 MiB (256 instances each) |
| L2 cache | 256 MiB (256 × 1 MiB) |
| L3 cache | 512 MiB total (16 instances = 32 MiB per CCD) |
| ISA highlights | AVX-512 (incl. `avx512_bf16`, `avx512_vnni`, `avx512_vp2intersect`), AES, SHA-NI |

> **Reproducibility note:** the **login node** (`alex2`) has a *different* CPU —
> 2 × **AMD EPYC 7713** (64-core Milan, SMT on, 256 threads). Always quote the
> **compute-node** CPU (EPYC 9745) for benchmarks; never benchmark on the login node.

---

## 3. Node memory, local storage & network

| Property | Value (verified) |
| --- | --- |
| Physical RAM | 791 497 756 kB ≈ **754 GiB** (`MemTotal`) |
| Slurm-usable RAM | `RealMemory=760000` MB (~760 GB) per node |
| Default mem per CPU | `DefMemPerCPU=2875` MB |
| Max mem per CPU | `MaxMemPerCPU=2875` MB (hard cap) |
| Local NVMe scratch | `/tmp` = **~28 TB** software RAID (`md127`) over 2 × 14 TB datacenter NVMe (`HSSD-E0215TP4L1N`) + 1 × 894 GB Samsung `MZQL2960HCJR` |
| Per-job scratch path | `$TMPDIR=/tmp/<jobid>.alex` (node-local, fast, wiped at job end) |
| Primary NIC up | `enp145s0f0np0` (Ethernet/RoCE-class) |

> **Memory budget rule of thumb:** with `MaxMemPerCPU=2875` MB and the partition
> default of `DefCpuPerGPU=32`, a single-GPU allocation (`--cpus-per-task=32`)
> grants ≈ **92 GB host RAM**. Request more CPUs to get more host RAM.

**Interconnect & local storage (researched — NHR@FAU docs):**

NHR@FAU's published Alex documentation [^nhr-alex] does **not yet** list the
`rtxpro6k` nodes (the page still describes the original A40/A100 partitions as of
the 2026-06-30 fetch), so there is **no vendor-confirmed interconnect figure for the
RTX PRO 6000 nodes specifically**. What can be cited:

- For the existing Alex nodes, the per-node Ethernet is **25 GbE**, and only the
  A100 nodes additionally carry **2 × InfiniBand HDR200 HCAs**; the A40 nodes have
  Ethernet only [^nhr-alex]. This matches the probe finding that only an Ethernet
  NIC (`enp145s0f0np0`) was up on the queried `rtxpro6k` node and no `ibstat`
  output was available — consistent with an **Ethernet-only / RoCE-class** node,
  but this should be **confirmed with NHR@FAU support** before being stated as fact
  for the new partition, since the partition is single-node anyway (see §4) and the
  network only matters for staging/IO, not inter-node MPI.
- Alex is also attached to NHR@FAU's NVMe **Lustre** storage (the `anvme`
  workspaces, `/anvme`), reachable via the workspace mechanism, in addition to the
  node-local NVMe `$TMPDIR` [^nhr-alex] [^nhr-fs].
- The node-local NVMe SSD model spec/endurance is not published by NHR@FAU; the
  drive models read off the node (`HSSD-E0215TP4L1N`, Samsung `MZQL2960HCJR`) are
  the authoritative source and need no external citation. NHR@FAU only documents
  that `$TMPDIR` is a node-local NVMe SSD wiped at job end [^nhr-fs].

> **Action:** if an exact interconnect speed for `rtxpro6k` is required, email
> <hpc-support@fau.de> — the public docs had not been updated for this partition at
> the time of writing.

**Sources / links for §3:**
- NHR@FAU Alex cluster documentation (node-detail table: 25 GbE, A100 HDR200 IB):
  <https://doc.nhr.fau.de/clusters/alex/>
- NHR@FAU filesystems documentation (`$TMPDIR`, Lustre `anvme` workspaces):
  <https://doc.nhr.fau.de/data/filesystems/>

---

## 4. Partition & Slurm policy (`rtxpro6k`)

Read from `scontrol show partition rtxpro6k` and `sinfo`:

| Property | Value (verified) |
| --- | --- |
| Nodes | `a[2041,2043,2141-2143,2741,2841,2843,2941-2943]` (11) |
| Total CPUs | 2816 |
| Total GPUs | 88 (`gres/gpu:rtxpro6k=88`) |
| GRES string | `gpu:rtxpro6k:8` per node (`(S:0-1)` — GPUs spread over both sockets) |
| Default time | `00:10:00` |
| **Max time** | `1-00:00:00` (**24 h hard limit**) |
| **Max nodes per job** | **1** (no multi-node jobs on this partition) |
| Default CPUs per GPU | `DefCpuPerGPU=32` |
| Oversubscribe | `OverSubscribe=NO` |
| Preempt mode | `CANCEL` |
| Priority | `PriorityJobFactor=30`, `PriorityTier=20` |

> **Allocation idiom used in this project** (`tools/.config/env_vars.sh`):
> `--gres=gpu:rtxpro6k:1`, `--cpus-per-task=32`, `--time=24:00:00`.
> Because `DefCpuPerGPU=32`, requesting 32 CPUs per GPU is the natural unit and
> keeps one task per GPU.

**Policy details (researched — NHR@FAU docs):**

- **Interactive jobs and `--exclusive`:** Interactive jobs are supported on Alex via
  `salloc` with the same options as `sbatch` (e.g.
  `salloc --gres=gpu:rtxpro6k:1 --time=...`); the environment of the calling shell
  is inherited [^nhr-alex] [^nhr-slurm]. `--exclusive` is available cluster-wide,
  but note that **exclusive jobs are billed for *all* resources of the node
  regardless of what you actually request** [^nhr-slurm-adv]. Because Alex compute
  nodes are shared between jobs (only the GPUs are always granted exclusively),
  exclusive mode is normally only needed for benchmarking [^nhr-slurm-adv].
- **Concurrency limits:** Per-user / per-group GPU and job concurrency on Alex are
  enforced through Slurm associations (e.g. `AssocGrpGRES` appears as a queue reason
  when all GPUs assigned to your group are in use) [^nhr-slurm-adv]; the concrete
  numeric limits are association-specific and are not published as a fixed figure —
  check with `sacctmgr`/ClusterCockpit or NHR@FAU support for your project.
- **Multi-node:** consistent with the verified `MaxNodes=1` on `rtxpro6k`,
  general Alex multi-node jobs already require explicit enabling by
  <hpc-support@fau.de> and an `a100multi`-style QoS [^nhr-alex]; the RTX PRO 6000
  partition is single-node.

> **Accounting / NPL (could NOT be verified — do not invent a rate):** NHR@FAU's
> public documentation does **not** publish a per-GPU-hour "NPL" charge rate for the
> `rtxpro6k` partition (or for Alex in general). For FAU/Tier3 users, Alex access is
> the *free basic tier* for publicly funded research/theses, granted on proof of
> need beyond TinyGPU but below NHR thresholds; usage is monitored via ClusterCockpit
> rather than billed at a published GPU-hour rate [^nhr-tier3] [^nhr-account]. The
> only "NPL" accounting documentation that turns up online belongs to **HLRN/NHR@ZIB
> + Göttingen**, a *different* center, and even there NPL was retired in favour of
> core-hours from Q2 2022 [^hlrn-npl] — so it must **not** be cited as NHR@FAU's
> model. The verified `billing=2816` / `PriorityJobFactor` values in the table above
> are the authoritative on-cluster Slurm settings; the official cost model for this
> partition should be confirmed directly with NHR@FAU (<hpc-support@fau.de>) before
> any GPU-hour cost claim is made in the thesis.

**Sources / links for §4:**
- NHR@FAU Alex documentation (interactive jobs, multi-node policy):
  <https://doc.nhr.fau.de/clusters/alex/>
- NHR@FAU Slurm batch-system docs:
  <https://doc.nhr.fau.de/batch-processing/batch_system_slurm/>
- NHR@FAU Slurm advanced topics (`--exclusive` billing, association limits):
  <https://doc.nhr.fau.de/batch-processing/advanced-topics-slurm/>
- FAU/Tier3 access to Alex (free basic tier, ClusterCockpit monitoring):
  <https://hpc.fau.de/tier3-access-to-alex/>
- NHR@FAU "Getting an account" (free of charge for publicly funded research/theses):
  <https://doc.nhr.fau.de/account/>

---

## 5. Software environment

| Property | Value (verified) |
| --- | --- |
| OS | AlmaLinux 9.8 (Olive Jaguar) |
| Kernel | `5.14.0-687.12.1.el9_8.x86_64` |
| Node feature flags | `hwperf`, `el9` (Slurm `AvailableFeatures`) |
| Scheduler | Slurm 25.11.4 |
| Module system | Environment Modules 5.6.1 |

**Selected modules available** (`module avail`, non-exhaustive):

| Category | Modules (verified present) |
| --- | --- |
| CUDA toolkit | `cuda/12.8.1`, `12.9.1`, `13.0.2`, `13.2.0` |
| cuDNN | `cudnn/8.9.7.29-12`, `9.17.0.29-12`, `9.17.0.29-13` |
| Compilers | `gcc/15.2.0`, `intel/2024.2.1`, `llvm/22.1.2` (default `gcc 11.5.0` toolchains) |
| MPI | `openmpi/5.0.8-*-cuda13.2`, `openmpi/4.1.8-*`, `intelmpi/2021.17.0` |
| Math | `mkl/2024.2.2`, `fftw/3.3.10`, `hdf5/1.14.6` |
| Python | `python/3.12-base`, `uv/0.11.4` |

> **CUDA version note:** the driver advertises **CUDA UMD 13.3**, so toolkits up
> to 13.x are driver-compatible. This project uses a self-managed Conda/Isaac
> stack under `$HPCVAULT` rather than the system CUDA modules (see
> `tools/install_IsaacLab_Alex.sh`).

---

## 6. Filesystems & quotas (this account)

Read from `quota -s` / `df -h` (figures are this user's quota — quotas are
per-user, sizes/policies are cluster-wide):

| Mount | Path (env var) | Quota | Purpose |
| --- | --- | --- | --- |
| `/home/hpc` | `$HOME` | 100 GB soft / 200 GB hard | Small, **backed up** — code lives here |
| `/home/vault` | `$HPCVAULT` | 1 TB soft / 2 TB hard | Large mid/long-term — Isaac Sim/Lab + Conda |
| `/home/woody` | `$WORK` | ~954 GB soft / ~1.4 TB hard | General scratch — training outputs/logs |
| `/tmp` (node-local) | `$TMPDIR` | ~28 TB (shared on node) | Per-job NVMe scratch, wiped at job end |

> Project layout follows this exactly (`tools/.config/env_vars.sh`): code in
> `$HOME`, Isaac + Conda in `$HPCVAULT/Isaac`, Slurm logs/outputs in `$WORK`,
> assets staged to `$TMPDIR` at runtime.

**Quota / backup / retention policy (researched — NHR@FAU docs):**

NHR@FAU's filesystems documentation [^nhr-fs] confirms the following cluster-wide
policies (the *per-user numbers* in your table above come from your own `quota -s`
and are correct for this account; the *defaults / policies* below are the documented
cluster-wide figures, which differ slightly from your account's elevated quotas):

| Mount (env var) | Documented default quota | Backup | Snapshots | Data lifetime |
| --- | --- | --- | --- | --- |
| `/home/hpc` (`$HOME`) | **50 GB** default (no extensions) | **Yes** | **Yes** | Account lifetime |
| `/home/vault` (`$HPCVAULT`) | **500 GB** default | **Yes** | **Yes** | Account lifetime |
| `$WORK` (woody / atuin / shareholder) | Tier3: **1,000 GB**; NHR: per-project | **No** | **No** | Account lifetime |
| `/lustre`, `/anvme` (workspaces) | inodes only | No | No | Workspace lifetime / high-watermark |
| `$TMPDIR` (node-local) | none | No | No | Job runtime |

Key documented facts to cite [^nhr-fs]:
- **`$HOME` and `$HPCVAULT` are backed up *and* snapshotted;** `$WORK` is **neither
  backed up nor snapshotted** — so your project layout (code in `$HOME`, Isaac+Conda
  in `$HPCVAULT`, logs/outputs in `$WORK`) correctly keeps only reproducible/
  regenerable data on `$WORK`.
- **Snapshot schedule:** `$HOME` snapshots every 30 min (6 kept) + 2 h (12) + daily
  (7) + weekly (4), covering ~4 weeks; `$HPCVAULT` daily (7) + weekly (4). Snapshots
  live in hidden `.snapshots/` dirs and are *not* a substitute for backup [^nhr-fs].
- **Inode (file-count) limits** exist on every filesystem; they are "rather tight"
  on `$HPCVAULT` and on the Lustre filesystems (which limit inodes rather than
  volume). Your `shownicerquota.pl` / `quota -s` output shows the exact FileQ/FiHaQ
  columns per filesystem [^nhr-fs].
- **`$WORK` expiry:** there is **no automatic time-based cleanup** of `$WORK`
  (woody/atuin) — its lifetime is the *account* lifetime, not a fixed expiry. By
  contrast, the **Lustre/`anvme` *workspaces* are subject to a workspace lifetime
  and high-watermark deletion** (oldest/largest files purged when the filesystem
  fills), so anything staged there is transient [^nhr-fs].
- Your `$WORK` ≈ 954 GB soft / 1.4 TB hard simply reflects this account's woody
  allocation; the documented Tier3 default is 1,000 GB.

**Sources / links for §6:**
- NHR@FAU filesystems documentation (quotas, backups, snapshots, inode limits,
  high-watermark deletion):
  <https://doc.nhr.fau.de/data/filesystems/>
- NHR@FAU workspaces (Lustre `anvme`, workspace lifetime):
  <https://doc.nhr.fau.de/data/workspaces/>

---

## 7. Ready-to-cite summary for the thesis

> Training was performed on the **Alex** GPU cluster at NHR@FAU. Each job ran on
> a single node of the **`rtxpro6k`** partition, equipped with **8 × NVIDIA RTX
> PRO 6000 Blackwell Server Edition GPUs** (96 GB ECC memory each, compute
> capability 12.0, 600 W, PCIe Gen 5; driver 610.43.02 / CUDA 13.3) and **2 ×
> AMD EPYC 9745** CPUs (256 physical cores total, 8 NUMA domains) with **~760 GB**
> of system RAM and node-local NVMe scratch. Jobs used **1 GPU + 32 CPU cores**
> per run under Slurm 25.11.4 on AlmaLinux 9.8, with a 24 h wall-time limit.

**Status after research (2026-06-30):** The GPU compute-throughput figures (§1) are
now filled from the official NVIDIA Server Edition datasheet and architecture
whitepaper, and the filesystem/quota and Slurm-policy items (§4, §6) are filled from
the NHR@FAU documentation. **Two items remain genuinely open and were intentionally
*not* fabricated:** (a) the exact interconnect speed of the `rtxpro6k` nodes (the
NHR@FAU Alex page had not been updated for this partition — confirm with
<hpc-support@fau.de>), and (b) the official GPU-hour/NPL **cost model** for
`rtxpro6k` (no published NHR@FAU rate exists; the HLRN NPL scheme is a *different*
center and must not be cited). Also still required before final write-up: record the
**exact software stack versions actually used** (Isaac Sim/Lab, PyTorch, CUDA
toolkit, Python) — pull these from the training environment / job logs at the time of
the experiments, not from this hardware doc.

---

## 8. References

[^nv-server-ds]: NVIDIA, *RTX PRO 6000 Blackwell Server Edition — Datasheet* (Apr 2025).
  Technical specifications: 24,064 CUDA cores; 752 fifth-gen Tensor cores; 188
  fourth-gen RT cores; 96 GB GDDR7 with ECC; 512-bit; 1,597 GB/s; FP32 120 TFLOPS;
  peak FP4 4 PFLOPS; RT 355 TFLOPS; up to 600 W configurable; PCIe 5.0 ×16; MIG up
  to 4 × 24 GB; 4× NVENC / 4× NVDEC / 4× JPEG; passive thermal.
  <https://www.akamai.com/site/en/documents/corporate/nvidia-rtx-pro-6000-blackwell-server-edition-datasheet.pdf>
  (product page: <https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/>)

[^nv-arch]: NVIDIA, *NVIDIA RTX Blackwell PRO GPU Architecture* whitepaper, v1.0.
  Blackwell GB202 GPU; fifth-gen Tensor Cores with FP4/FP6 and ~2× FP8 throughput;
  fourth-gen RT Cores with doubled ray-triangle intersection rate; shared
  architecture across Server / Workstation / Max-Q editions.
  <https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf>

[^lenovo]: Lenovo Press, *ThinkSystem NVIDIA RTX PRO 6000 Blackwell Server Edition
  PCIe Gen5 GPU — Product Guide* (LP2263). Full specification table (memory bandwidth
  up to 1,597 GB/s, 512-bit, 24,064 CUDA / 752 Tensor / 188 RT cores, no NVLink,
  600 W, PCIe 5.0 ×16, MIG up to 4 × 24 GB, passive cooling, FHFL dual-slot).
  <https://lenovopress.lenovo.com/lp2263-thinksystem-nvidia-rtx-pro-6000-blackwell-server-edition-pcie-gen5-gpu>

[^nhr-alex]: NHR@FAU, *Alex* cluster documentation (HPC Documentation). Node-detail
  table (per-node network 25 GbE; A100 nodes additionally 2× InfiniBand HDR200; A40
  Ethernet only), partition/Slurm usage, interactive jobs, multi-node-on-request,
  Lustre `anvme` attachment. <https://doc.nhr.fau.de/clusters/alex/>

[^nhr-fs]: NHR@FAU, *File systems* documentation. Quota defaults ($HOME 50 GB,
  $HPCVAULT 500 GB, $WORK Tier3 1,000 GB), backup/snapshot policy, snapshot
  schedules, inode limits, `$TMPDIR` semantics, high-watermark deletion on Lustre.
  <https://doc.nhr.fau.de/data/filesystems/>

[^nhr-slurm]: NHR@FAU, *Slurm Batch system* documentation (general usage, `salloc`
  interactive jobs). <https://doc.nhr.fau.de/batch-processing/batch_system_slurm/>

[^nhr-slurm-adv]: NHR@FAU, *Slurm Advanced Topics* documentation (`--exclusive`
  billing of all node resources; association/GRES concurrency limits; node sharing on
  Alex). <https://doc.nhr.fau.de/batch-processing/advanced-topics-slurm/>

[^nhr-tier3]: NHR@FAU, *Application for FAU/Tier3 access to GPGPU cluster Alex* (free
  basic tier on proof of need below NHR thresholds; usage checked via ClusterCockpit).
  <https://hpc.fau.de/tier3-access-to-alex/>

[^nhr-account]: NHR@FAU, *Getting an account* documentation (HPC accounts free of
  charge for publicly funded research and theses; compute beyond the free basic tier
  requires an NHR project). <https://doc.nhr.fau.de/account/>

[^hlrn-npl]: HLRN / NHR@ZIB + NHR@Göttingen, *NPL accounting* (historical; NPL used
  only until end of Q1 2022, then replaced by core-hours). **Different center — not
  applicable to NHR@FAU; listed only to document why the online "NPL" pages must not
  be cited for Alex.** <https://www.hlrn.de/doc/pages/viewpage.action?pageId=53149948>

---

*Generated 2026-06-30 from live cluster queries on `alex2.nhr.fau.de` /
compute node `a2843`. Re-run `scontrol show partition rtxpro6k`,
`scontrol show node <n>`, `lscpu`, and `nvidia-smi` to refresh if the partition
is reconfigured.*