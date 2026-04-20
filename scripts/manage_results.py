#!/usr/bin/env python3
"""Manage experiment result directories.

List, inspect, compare, and clean up results stored in results/<datetime>/.
"""

import sys
from pathlib import Path
import json
import shutil
from datetime import datetime
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent.parent / "results"


def list_runs():
    """List all result directories with metadata."""
    if not RESULTS_DIR.exists():
        print("No results directory found.")
        return

    runs = sorted(RESULTS_DIR.iterdir())
    if not runs:
        print("No result runs found.")
        return

    print(f"{'Timestamp':<20} {'Commit':<8} {'Dirty':<6} {'Scripts'}")
    print("-" * 70)

    for run_dir in runs:
        if not run_dir.is_dir():
            continue

        meta_file = run_dir / "meta.json"
        meta = {}
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)

        commit = meta.get("commit", "?")[:7]
        dirty = "YES" if meta.get("dirty", True) else "no"

        # Find which scripts produced output
        scripts = []
        for sub in run_dir.iterdir():
            if sub.is_dir() and sub.name != "meta.json":
                scripts.append(sub.name)
        scripts_str = ", ".join(sorted(scripts)) if scripts else "-"

        print(f"{run_dir.name:<20} {commit:<8} {dirty:<6} {scripts_str}")


def show_run(timestamp: str):
    """Show detailed contents of a single result directory."""
    run_dir = RESULTS_DIR / timestamp
    if not run_dir.exists():
        print(f"Run not found: {timestamp}")
        return

    meta_file = run_dir / "meta.json"
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        print(f"Timestamp: {meta.get('timestamp', 'N/A')}")
        print(f"Commit:    {meta.get('commit', 'N/A')}")
        print(f"Dirty:     {'Yes' if meta.get('dirty') else 'No'}")
    else:
        print("No meta.json found.")

    print(f"\nContents:")
    for sub in sorted(run_dir.iterdir()):
        if sub.is_dir():
            print(f"  📁 {sub.name}/")
            for item in sorted(sub.rglob("*")):
                if item.is_file():
                    rel = item.relative_to(sub)
                    size = item.stat().st_size
                    size_str = f"{size}B" if size < 1024 else f"{size/1024:.1f}KB"
                    print(f"      {rel} ({size_str})")


def compare_runs(timestamps: list[str]):
    """Compare experiment results across multiple runs."""
    for ts in timestamps:
        run_dir = RESULTS_DIR / ts
        if not run_dir.exists():
            print(f"Run not found: {ts}")
            return

    print(f"Comparing {len(timestamps)} runs:\n")

    for ts in timestamps:
        run_dir = RESULTS_DIR / ts
        print(f"--- {ts} ---")

        # Try to load common result files
        exp_results = run_dir / "run_experiments" / "logs" / "results.json"
        if exp_results.exists():
            with open(exp_results) as f:
                data = json.load(f)
            if "final_evaluation" in data:
                fe = data["final_evaluation"]
                print(f"  Selected: {data['selected_config']['name']}")
                print(f"  Trial 1:  {fe['trial_1_acc']:.4f}")
                print(f"  Trial 2:  {fe['trial_2_acc']:.4f}")
                print(f"  Mean:     {fe['mean_acc']:.4f}")

        challenge = run_dir / "evaluate_challenge" / "logs" / "challenge_results.txt"
        if challenge.exists():
            with open(challenge) as f:
                lines = f.readlines()
            for line in lines:
                if "Best:" in line or "Mean:" in line:
                    print(f"  {line.strip()}")

        print()


def clean_old(keep: int = 5):
    """Delete oldest result directories, keeping the N most recent."""
    runs = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir()])
    if len(runs) <= keep:
        print(f"Only {len(runs)} runs exist. Nothing to clean (keep={keep}).")
        return

    to_delete = runs[:-keep]
    print(f"Keeping {keep} most recent runs. Deleting {len(to_delete)} old run(s):")

    for run_dir in to_delete:
        print(f"  🗑️  {run_dir.name}")
        shutil.rmtree(run_dir)

    print("Done.")


def clean_dirty():
    """Delete all runs where the workspace was dirty at the time."""
    deleted = 0
    for run_dir in RESULTS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        meta_file = run_dir / "meta.json"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            if meta.get("dirty", True):
                print(f"  🗑️  {run_dir.name} (dirty)")
                shutil.rmtree(run_dir)
                deleted += 1

    if deleted == 0:
        print("No dirty runs found.")
    else:
        print(f"Deleted {deleted} dirty run(s).")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Manage experiment result directories")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List all result runs")

    show_parser = subparsers.add_parser("show", help="Show details of a specific run")
    show_parser.add_argument("timestamp", help="Run timestamp (e.g., 20260420_175640)")

    compare_parser = subparsers.add_parser("compare", help="Compare multiple runs")
    compare_parser.add_argument("timestamps", nargs="+", help="Run timestamps to compare")

    clean_parser = subparsers.add_parser("clean", help="Clean up old results")
    clean_parser.add_argument("--keep", type=int, default=5, help="Number of recent runs to keep")
    clean_parser.add_argument("--dirty-only", action="store_true", help="Only delete dirty runs")

    args = parser.parse_args()

    if args.command == "list":
        list_runs()
    elif args.command == "show":
        show_run(args.timestamp)
    elif args.command == "compare":
        compare_runs(args.timestamps)
    elif args.command == "clean":
        if args.dirty_only:
            clean_dirty()
        else:
            clean_old(args.keep)


if __name__ == "__main__":
    main()
