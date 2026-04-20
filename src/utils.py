"""Shared utilities for result management and logging."""

import contextlib
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterator


_DEFAULT_BASE = Path(__file__).parent.parent / "results"


def get_git_short_hash() -> str:
    """Return the short git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_git_info() -> dict[str, str | bool]:
    """Return git metadata: short commit hash and dirty workspace status."""
    repo_root = Path(__file__).parent.parent
    try:
        hash_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root,
        )
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain", "scripts/", "src/"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root,
        )
        return {
            "commit": hash_result.stdout.strip(),
            "dirty": bool(dirty_result.stdout.strip()),
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "dirty": False}


def _find_latest_run_dir(base_dir: Path) -> Path | None:
    """Find the most recent result directory with a valid meta.json."""
    if not base_dir.exists():
        return None
    run_dirs = [
        d for d in sorted(base_dir.iterdir(), reverse=True)
        if d.is_dir() and (d / "meta.json").exists()
    ]
    return run_dirs[0] if run_dirs else None


def _find_matching_run_dir(base_dir: Path, prefix: str) -> Path | None:
    """Find the most recent result directory whose name starts with prefix."""
    if not base_dir.exists():
        return None
    run_dirs = [
        d for d in sorted(base_dir.iterdir(), reverse=True)
        if d.is_dir() and (d / "meta.json").exists() and d.name.startswith(prefix)
    ]
    return run_dirs[0] if run_dirs else None


def get_result_dir(
    base_dir: str | Path | None = None,
    run_name: str | None = None,
    reuse: bool = False,
) -> Path:
    """Create and return a result directory.

    Directory format:
        results/<datetime>/
    where <datetime> defaults to the current timestamp.

    A meta.json is written into the directory with run metadata
    (timestamp, git commit hash, dirty workspace flag).

    Args:
        base_dir: Root directory for results. Defaults to ./results.
        run_name: Name for this run. Defaults to current timestamp.
        reuse: If True and no run_name given, try to reuse the latest
            existing run directory if it has the same commit hash and
            is not dirty.

    Returns:
        Path to the result directory (new or reused).
    """
    if base_dir is None:
        base_dir = _DEFAULT_BASE
    else:
        base_dir = Path(base_dir)

    if run_name:
        result_dir = base_dir / run_name
        result_dir.mkdir(parents=True, exist_ok=True)
    elif reuse:
        git_info = get_git_info()
        latest = _find_latest_run_dir(base_dir)
        if latest is not None:
            with open(latest / "meta.json") as f:
                meta = json.load(f)
            if meta.get("commit") == git_info["commit"] and not meta.get("dirty", True):
                return latest
        # No reusable run found — create new
        name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = base_dir / name
        result_dir.mkdir(parents=True, exist_ok=True)
    else:
        name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = base_dir / name
        result_dir.mkdir(parents=True, exist_ok=True)

    # Write meta.json if it doesn't already exist
    meta_path = result_dir / "meta.json"
    if not meta_path.exists():
        git_info = get_git_info()
        meta = {
            "timestamp": datetime.datetime.now().isoformat(),
            "commit": git_info["commit"],
            "dirty": git_info["dirty"],
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    return result_dir


def get_script_output_dir(run_dir: Path, script_name: str) -> Path:
    """Return a script-specific subdirectory inside a run directory.

    Args:
        run_dir: Base run directory (e.g., from resolve_run_dir).
        script_name: Name of the script (used as subdirectory name).

    Returns:
        Path to the script-specific output directory.
    """
    path = run_dir / script_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_run_dir(run_dir: str | Path | None = None) -> Path:
    """Resolve the result directory for a script.

    Priority:
    1. Explicitly provided run_dir argument.
       - Path: use directly.
       - "latest": reuse the most recent result directory.
       - Partial timestamp (e.g., "20260420"): match most recent run
         whose name starts with the prefix.
    2. ML_RUN_DIR environment variable.
    3. Auto-reuse the latest run if same commit and not dirty.
    4. Auto-generate a new timestamped directory.

    Args:
        run_dir: Explicit path, special keyword, or None for auto.

    Returns:
        Path to the result directory.
    """
    if run_dir is not None:
        run_str = str(run_dir)
        if run_str.lower() == "latest":
            latest = _find_latest_run_dir(_DEFAULT_BASE)
            if latest is None:
                raise FileNotFoundError("No existing result runs found.")
            return latest
        # Check if it looks like a partial timestamp (no slashes, just digits/underscores)
        if "/" not in run_str and "\\" not in run_str:
            matched = _find_matching_run_dir(_DEFAULT_BASE, run_str)
            if matched is not None:
                return matched
        # Fall through to treating it as a path
        path = Path(run_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    env_dir = os.environ.get("ML_RUN_DIR")
    if env_dir:
        path = Path(env_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # Auto-reuse if same commit and clean workspace
    return get_result_dir(reuse=True)


class _Tee:
    """Write to a file while still printing to the original stdout."""

    def __init__(self, filepath: Path, original_stdout):
        self.file = open(filepath, "w", encoding="utf-8")
        self.original_stdout = original_stdout

    def write(self, data: str) -> int:
        self.file.write(data)
        self.file.flush()
        return self.original_stdout.write(data)

    def flush(self) -> None:
        self.file.flush()
        self.original_stdout.flush()

    def close(self) -> None:
        self.file.close()


@contextlib.contextmanager
def setup_run_logging(output_dir: Path) -> Iterator[None]:
    """Context manager that tees stdout to a run.log file inside output_dir.

    Usage:
        with setup_run_logging(output_dir):
            print("This goes to console AND run.log")
    """
    log_path = output_dir / "run.log"
    original_stdout = sys.stdout
    tee = _Tee(log_path, original_stdout)
    sys.stdout = tee
    try:
        yield
    finally:
        sys.stdout = original_stdout
        tee.close()
