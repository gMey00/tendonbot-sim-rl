"""Generate the REAL Task-1→2 terminal-state bank (Stage-2 roadmap, S0/T1).

Rolls the trained shirt_pick policy DETERMINISTICALLY (mean actions, seeded
crumpled-bank resets) over full episodes and snapshots each env's state at the
step its stable-present latch fires (``ShirtPickEnv.snapshot_terminal_states``:
particle pos/vel + attach mask + grasp point + joint state).  The episode then
runs on to its natural end so the post-latch outcome can be recorded — the
5–7 % post-latch grasp slips are INCLUDED, not filtered: they are the real
input distribution Task 2 must survive (research report §3 Stage 0).

Episodes that never latch are snapshotted on their last step instead and
stored with ``presented=False`` (filtering is the consumer's choice, same
contract as the snapshot hook).

Per-state metadata (all ``[K]``):
    presented        latch fired (snapshot is AT the latch step)
    slip             grasp was lost at some point AFTER the latch
    pre_latch_drop   grasp was lost (and possibly re-acquired) BEFORE the latch
    end_attached     grasp still active on the episode's final step
    regrasp          re-attach happened after a post-latch slip
    latch_step       env step of the latch (-1 if never)
    grasp_particle   representative grasped particle (attached particle nearest
                     the finger tip at snapshot; -1 if nothing attached)
    n_attached       welded-patch size at snapshot
    region12 / sym8  study region of ``grasp_particle`` on the flat rest shape
                     (regions.assign_regions — verified identical to
                     present_markers.pt's ``particle_region`` after l/r folding)
    rest_xy          flat-rest coords of ``grasp_particle``

Frames: ``pos`` env-local (env origin subtracted, z above ground — same as the
snapshot hook); ``vel`` world (translation-invariant); ``grasp_point_local``
env-local.  Task 2 re-anchors at its holder via the grasp point + attach mask.

Usage::

    cd src/tensegrity_pick
    conda activate env_isaaclab   # keep PYTHONPATH (isaacsim hooks)
    PYTHONUNBUFFERED=1 python scripts/asset_generation/generate_pick_terminal_bank.py \
        --headless --num_envs 64 --seed 7 --target_presented 520 \
        --checkpoint logs/skrl/shirt_pick/2026-07-03_19-33-17_ppo_torch_seed3/checkpoints/agent_12000.pt

Output: res/Props/Cloth/banks/tshirt_pick_terminal_bank.pt (git-lfs).
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import os
import subprocess
import sys
import time

from isaaclab.app import AppLauncher

DEFAULT_CHECKPOINT = (
    "logs/skrl/shirt_pick/2026-07-03_19-33-17_ppo_torch_seed3/checkpoints/agent_12000.pt"
)

parser = argparse.ArgumentParser(description="Generate the Task-1→2 terminal-state bank.")
parser.add_argument("--task", type=str, default="Template-Shirt-Pick-Tensegrity-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
parser.add_argument("--target_presented", type=int, default=520,
                    help="stop once this many presented-terminal states are banked")
parser.add_argument("--max_episodes", type=int, default=1200,
                    help="hard stop (episodes) if the target is never reached")
parser.add_argument("--out", type=str, default=None,
                    help="output .pt (default: res/Props/Cloth/banks/tshirt_pick_terminal_bank.pt)")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import skrl  # noqa: F401, E402
from skrl.utils.runner.torch import Runner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402

import tensegrity_pick.tasks  # noqa: F401, E402
from tensegrity_pick.tasks.manager_based.shared.cloth_sorting_scene_cfg import (  # noqa: E402
    PROJ_ASSETS_PATH,
)

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FLAT_REST_PT = os.path.join(
    REPO_ROOT, "..", "..", "doc", "reports", "data", "present_heuristics_flat_rest.pt"
)
REGIONS_PY = os.path.join(
    REPO_ROOT, "scripts", "model_validation", "present_heuristics", "regions.py"
)
# 12 fine regions → the study's 8 symmetrized classes (present_markers.pt order).
SYM_CLASSES = ["collar", "shoulder", "sleeve", "chest", "side", "belly", "hem_corner", "hem_c"]
SYM_OF = {
    "collar": "collar", "shoulder_l": "shoulder", "shoulder_r": "shoulder",
    "sleeve_l": "sleeve", "sleeve_r": "sleeve", "chest": "chest",
    "side_l": "side", "side_r": "side", "belly": "belly",
    "hem_l": "hem_corner", "hem_r": "hem_corner", "hem_c": "hem_c",
}


def _load_regions():
    spec = importlib.util.spec_from_file_location("regions", REGIONS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    out_path = args_cli.out or os.path.join(
        PROJ_ASSETS_PATH, "Props", "Cloth", "banks", "tshirt_pick_terminal_bank.pt"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ── Offline per-particle region lookup (study assignment, read-only) ──
    regions = _load_regions()
    flat = torch.load(FLAT_REST_PT, map_location="cpu", weights_only=False)
    flat_rest = flat["flat_rest_pos"]                                # [P, 3]
    reg12_per_particle = torch.from_numpy(
        regions.assign_regions(flat_rest[:, 0].numpy(), flat_rest[:, 1].numpy())
    ).long()                                                          # [P]
    sym_lut = torch.tensor(
        [SYM_CLASSES.index(SYM_OF[n]) for n in regions.REGION_NAMES], dtype=torch.long
    )

    # ── Env + deterministic policy (same recipe as evaluate_shirt_pick) ──
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
    experiment_cfg["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed
    # Global torch RNG drives the crumpled-bank sampling/augmentation at reset
    # (cfg.seed alone was observed NOT to differentiate rollouts).
    torch.manual_seed(args_cli.seed)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)
    resume_path = os.path.abspath(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.enable_training_mode(False, apply_to_models=True)

    u = env.unwrapped
    n, dev = u.num_envs, u.device
    p = u.cloth.num_particles
    assert p == flat_rest.shape[0], (
        f"cloth has {p} particles but the study flat rest has {flat_rest.shape[0]}"
    )
    max_ep = int(u.max_episode_length)
    reg12_dev = reg12_per_particle.to(dev)

    # ── Per-env episode trackers (mirror the env's latches: the env clears
    #    its own flags inside step() for done envs, before we see them) ────
    my_latched = torch.zeros(n, dtype=torch.bool, device=dev)
    slip = torch.zeros(n, dtype=torch.bool, device=dev)
    pre_latch_drop = torch.zeros(n, dtype=torch.bool, device=dev)
    regrasp = torch.zeros(n, dtype=torch.bool, device=dev)
    last_attached = torch.zeros(n, dtype=torch.bool, device=dev)
    latch_step = torch.full((n,), -1, dtype=torch.long, device=dev)
    staged: dict[int, dict] = {}  # env → snapshot record (committed at episode end)

    records: list[dict] = []
    n_presented = 0
    episodes = 0
    ep_no_latch = 0
    env_drop_acc = 0.0  # env's own Metrics/drop_rate — cross-check vs slip flags
    t_start = time.perf_counter()

    def snapshot_envs(env_ids: torch.Tensor, presented: bool) -> None:
        """Stage snapshot records for ``env_ids`` (device tensor)."""
        snap = u.snapshot_terminal_states()          # full-size CPU tensors
        tip = u._finger_tip_pos()                    # [N, 3] device
        mask_dev = u.cloth._attach_mask[0]           # [N, P] device, slot 0
        pts = u.cloth.nodal_pos_w                    # [N, P, 3] device
        for e in env_ids.tolist():
            att = torch.nonzero(mask_dev[e], as_tuple=False).squeeze(-1)
            if att.numel() > 0:
                d = (pts[e, att] - tip[e]).norm(dim=-1)
                rep = int(att[d.argmin()])
            else:
                rep = -1
            reg12 = int(reg12_dev[rep]) if rep >= 0 else -1
            staged[e] = {
                "pos": snap["pos"][e].clone(),
                "vel": snap["vel"][e].clone(),
                "attach_mask": snap["attach_mask"][e].clone(),
                "grasp_point_local": snap["grasp_point_local"][e].clone(),
                "joint_pos": snap["joint_pos"][e].clone(),
                "presented": presented,
                "latch_step": int(latch_step[e]) if presented else -1,
                "grasp_particle": rep,
                "n_attached": int(att.numel()),
                "region12": reg12,
                "sym8": int(sym_lut[reg12]) if reg12 >= 0 else -1,
                "rest_xy": flat_rest[rep, :2].clone() if rep >= 0 else torch.zeros(2),
            }

    obs, _ = env.reset()
    # no_grad (NOT inference_mode): snapshot clones must stay normal tensors
    # so they can be stacked/saved after the loop.
    torch.set_grad_enabled(False)
    while simulation_app.is_running() and n_presented < args_cli.target_presented \
            and episodes < args_cli.max_episodes:
        outputs = runner.agent.act(obs, env.state(), timestep=0, timesteps=0)
        actions = outputs[-1].get("mean_actions", outputs[0])
        obs, _, terminated, truncated, _ = env.step(actions)
        done = (terminated | truncated).squeeze(-1) if terminated.dim() > 1 \
            else (terminated | truncated)
        alive = ~done

        # Post-latch slip / pre-latch drop / regrasp tracking (drop_event is
        # already zero for done envs — the env resets before we read it).
        dropped = (u.drop_event > 0.5) & alive
        slip |= dropped & my_latched
        pre_latch_drop |= dropped & ~my_latched
        regrasp |= slip & u.grasp_active & alive
        last_attached[alive] = u.grasp_active[alive]

        # Latch snapshots (u.was_presented is cleared for done envs, so
        # ``newly`` can only fire on alive envs).
        newly = u.was_presented & ~my_latched
        if newly.any():
            ids = torch.nonzero(newly, as_tuple=False).squeeze(-1)
            latch_step[ids] = u.episode_length_buf[ids]
            my_latched |= newly
            snapshot_envs(ids, presented=True)

        # Last-step snapshot for never-latched envs (next step times out).
        about_to_end = (u.episode_length_buf >= max_ep - 1) & ~my_latched & alive
        if about_to_end.any():
            snapshot_envs(
                torch.nonzero(about_to_end, as_tuple=False).squeeze(-1), presented=False
            )

        # Commit finished episodes.
        if done.any():
            n_done = int(done.sum())
            env_drop_acc += float(
                u.extras.get("log", {}).get("Metrics/drop_rate", 0.0)
            ) * n_done
            for e in torch.nonzero(done, as_tuple=False).squeeze(-1).tolist():
                episodes += 1
                rec = staged.pop(e, None)
                if rec is None:
                    # early termination without latch or last-step snapshot
                    ep_no_latch += 1
                else:
                    rec["slip"] = bool(slip[e])
                    rec["pre_latch_drop"] = bool(pre_latch_drop[e])
                    rec["regrasp"] = bool(regrasp[e])
                    rec["end_attached"] = bool(last_attached[e])
                    records.append(rec)
                    if rec["presented"]:
                        n_presented += 1
                    else:
                        ep_no_latch += 1
                my_latched[e] = False
                slip[e] = False
                pre_latch_drop[e] = False
                regrasp[e] = False
                last_attached[e] = False
                latch_step[e] = -1
            elapsed = time.perf_counter() - t_start
            print(f"[bank] episodes {episodes}: presented {n_presented}/"
                  f"{args_cli.target_presented}, no-latch {ep_no_latch}, "
                  f"slips {sum(r['slip'] and r['presented'] for r in records)} "
                  f"({elapsed:.0f}s)", flush=True)

    # ── Assemble + save ───────────────────────────────────────────────────
    if not records:
        raise RuntimeError("no terminal states collected — check the checkpoint/task")

    def stack(key, dtype=None):
        vals = [r[key] for r in records]
        if torch.is_tensor(vals[0]):
            return torch.stack(vals).to(dtype) if dtype else torch.stack(vals)
        return torch.tensor(vals, dtype=dtype)

    presented_t = stack("presented", torch.bool)
    sym8_t = stack("sym8", torch.long)
    reg12_t = stack("region12", torch.long)
    slip_t = stack("slip", torch.bool)

    # First-grasp-region distribution over presented states (thesis figure).
    pres_sym = sym8_t[presented_t]
    dist_rows = []
    print("\nFirst-grasp-region distribution (presented states):")
    for i, name in enumerate(SYM_CLASSES):
        c = int((pres_sym == i).sum())
        frac = c / max(int(presented_t.sum()), 1)
        dist_rows.append({"region": name, "count": c, "frac": round(frac, 4)})
        print(f"  {name:12s} {c:5d}  {frac:6.3f}")

    try:
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, cwd=REPO_ROOT).strip()
    except Exception:
        git_rev = "unknown"
    meta = {
        "generator": "scripts/asset_generation/generate_pick_terminal_bank.py",
        "command": " ".join(sys.argv),
        "checkpoint": resume_path,
        "task": args_cli.task,
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "policy_actions": "deterministic (mean)",
        "num_states": len(records),
        "num_presented": int(presented_t.sum()),
        "num_slip": int((slip_t & presented_t).sum()),
        "episodes_rolled": episodes,
        "episodes_no_latch": ep_no_latch,
        "env_drop_rate": round(env_drop_acc / max(episodes, 1), 4),
        "num_particles": p,
        "frame": "pos/grasp_point_local env-local (z above ground); vel world",
        "regions": {"names12": regions.REGION_NAMES, "sym_classes": SYM_CLASSES,
                    "assignment": "regions.assign_regions on flat-rest (x,y), "
                                  "rep particle = attached particle nearest finger tip at snapshot"},
        "first_grasp_region_distribution": dist_rows,
        "git": git_rev,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    out = {
        "pos": stack("pos"),                          # [K, P, 3] float32
        "vel": stack("vel"),                          # [K, P, 3] float32
        "attach_mask": stack("attach_mask"),          # [K, P] bool
        "grasp_point_local": stack("grasp_point_local"),  # [K, 3]
        "joint_pos": stack("joint_pos"),              # [K, J]
        "presented": presented_t,
        "slip": slip_t,
        "pre_latch_drop": stack("pre_latch_drop", torch.bool),
        "regrasp": stack("regrasp", torch.bool),
        "end_attached": stack("end_attached", torch.bool),
        "latch_step": stack("latch_step", torch.long),
        "grasp_particle": stack("grasp_particle", torch.long),
        "n_attached": stack("n_attached", torch.long),
        "region12": reg12_t,
        "sym8": sym8_t,
        "rest_xy": stack("rest_xy"),
        "meta": meta,
    }
    torch.save(out, out_path)
    dt_total = time.perf_counter() - t_start
    print(f"\nRESULT: saved {len(records)} states ({int(presented_t.sum())} presented, "
          f"{int((slip_t & presented_t).sum())} post-latch slips, {ep_no_latch} non-latched, "
          f"env drop_rate {env_drop_acc / max(episodes, 1):.4f}) "
          f"to {out_path} ({os.path.getsize(out_path)/1024**2:.1f} MB) in {dt_total:.0f}s")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
