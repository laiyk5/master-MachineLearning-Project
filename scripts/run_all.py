#!/usr/bin/env python3
"""Run all analysis scripts into a single shared result directory."""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils import get_result_dir, setup_run_logging


SCRIPTS = [
    "run_experiments.py",
    "error_analysis.py",
    "class_distribution.py",
    "sample_error_analysis.py",
    "evaluate_challenge.py",
]


def run_script(script_name: str, run_dir: Path) -> int:
    """Run a single script with the shared run directory."""
    script_path = Path(__file__).parent / script_name
    cmd = [sys.executable, str(script_path), "--run-dir", str(run_dir)]
    print(f"\n{'=' * 80}")
    print(f"Running: {script_name}")
    print(f"{'=' * 80}")
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Run all experiment scripts into one unified result directory."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this run (default: timestamp)",
    )
    args = parser.parse_args()

    run_dir = get_result_dir(run_name=args.run_name)
    print(f"Result directory: {run_dir}")

    with setup_run_logging(run_dir):
        print(f"{'=' * 80}")
        print("RUNNING ALL EXPERIMENT SCRIPTS")
        print(f"{'=' * 80}")
        print(f"Result directory: {run_dir}")

        for script in SCRIPTS:
            rc = run_script(script, run_dir)
            if rc != 0:
                print(f"\n⚠️  {script} exited with code {rc}")

        print(f"\n{'=' * 80}")
        print("ALL SCRIPTS COMPLETE")
        print(f"Results saved to: {run_dir}")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
