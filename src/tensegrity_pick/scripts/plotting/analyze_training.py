#!/usr/bin/env python3
"""Quick training metrics analyzer. Reads TF events and prints key metrics.

Usage:
    python scripts/analyze_training.py [LOG_DIR]
    python scripts/analyze_training.py  # auto-finds latest cube_sort run
"""
import os
import sys
import glob


def find_latest_log(base="logs/skrl/cube_sort"):
    """Find the most recent log directory."""
    dirs = sorted(glob.glob(os.path.join(base, "2026-*")))
    if not dirs:
        print(f"No log directories found under {base}")
        sys.exit(1)
    return dirs[-1]


def main():
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("tensorboard not installed")
        sys.exit(1)

    log_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest_log()
    print(f"Analyzing: {log_dir}\n")

    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    if not tags:
        print("No scalar tags found yet.")
        return

    tagset = set(tags)

    def resolve(tag):
        """Return the actual tag name present, tolerating the optional
        'Info / ' prefix that skrl <= 2.0 put on Episode_Reward/ and Metrics/
        tags but newer versions (skrl 2.1+) drop. Returns None if absent."""
        if tag in tagset:
            return tag
        if tag.startswith("Info / "):
            alt = tag[len("Info / "):]
        else:
            alt = "Info / " + tag
        return alt if alt in tagset else None

    # Key metrics to focus on (written with the historical 'Info / ' prefix;
    # resolve() also matches the newer prefix-less form).
    key_metrics = [
        "Reward / Total reward (mean)",
        "Reward / Total reward (max)",
        "Info / Metrics/green_grasp_rate",
        "Info / Metrics/green_placement_rate",
        "Info / Metrics/green_miss_rate",
        "Info / Metrics/grasp_rate",
        "Info / Metrics/mean_episode_length",
        "Info / Metrics/green_grasped_count",
        "Info / Metrics/green_placed_count",
        "Info / Metrics/red_grabbed_count",
        "Info / Metrics/red_in_drum_count",
        "Policy / Standard deviation",
        "Learning / Learning rate",
    ]

    # Print key metrics (latest, max, trend)
    print(f"{'Metric':<45} {'Latest':>10} {'Max':>10} {'Steps':>8} {'Points':>6}")
    print("-" * 85)

    for tag in key_metrics:
        actual = resolve(tag)
        if actual is None:
            continue
        vals = ea.Scalars(actual)
        if not vals:
            continue
        latest = vals[-1]
        max_val = max(v.value for v in vals)
        # Strip the historical 'Info / ' prefix so old/new runs read identically.
        label = actual[len("Info / "):] if actual.startswith("Info / ") else actual
        print(f"{label:<45} {latest.value:>10.4f} {max_val:>10.4f} {latest.step:>8d} {len(vals):>6d}")

    # Print all reward terms
    print(f"\n{'Reward Terms (Episode Rewards)':<45}")
    print("-" * 85)
    reward_tags = sorted([t for t in tags if "Episode_Reward" in t])
    for tag in reward_tags:
        vals = ea.Scalars(tag)
        if not vals:
            continue
        latest = vals[-1]
        short = tag.split("/")[-1]
        print(f"  {short:<43} {latest.value:>10.4f} step={latest.step}")

    # Status
    import json
    status_file = os.path.join(log_dir, "run_status.json")
    if os.path.exists(status_file):
        with open(status_file) as f:
            status = json.load(f)
        print(f"\nRun status: {status.get('status', 'unknown')}")
        print(f"Target timesteps: {status.get('target_timesteps', '?')}")


if __name__ == "__main__":
    main()
