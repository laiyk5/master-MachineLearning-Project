"""Shared utilities for result management and logging."""

import contextlib
import datetime
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


def get_result_dir(
    base_dir: str | Path | None = None,
    run_name: str | None = None,
) -> Path:
    """Create and return a result directory.

    Directory format:
        results/<git-hash>/<run-name>/
    where <run-name> defaults to a timestamp if not provided.

    Args:
        base_dir: Root directory for results. Defaults to ./results.
        run_name: Name for this run. Defaults to current timestamp.

    Returns:
        Path to the newly created result directory.
    """
    if base_dir is None:
        base_dir = _DEFAULT_BASE
    else:
        base_dir = Path(base_dir)

    short_hash = get_git_short_hash()
    name = run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = base_dir / short_hash / name
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def resolve_run_dir(run_dir: str | Path | None = None) -> Path:
    """Resolve the result directory for a script.

    Priority:
    1. Explicitly provided run_dir argument.
    2. ML_RUN_DIR environment variable.
    3. Auto-generate a new timestamped directory via get_result_dir().

    Args:
        run_dir: Explicit path, or None to use env/fallback.

    Returns:
        Path to the result directory.
    """
    if run_dir is not None:
        path = Path(run_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    env_dir = os.environ.get("ML_RUN_DIR")
    if env_dir:
        path = Path(env_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    return get_result_dir()


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
