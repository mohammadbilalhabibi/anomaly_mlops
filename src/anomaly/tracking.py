"""Thin MLflow helpers. Imported only by pipeline scripts, never by tests or the API."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import mlflow

from anomaly.config import PROJECT_ROOT

EXPERIMENT = "patchcore-mvtec"


def setup(experiment: str = EXPERIMENT) -> None:
    """Point MLflow at the tracking URI from the environment (default: local sqlite)."""
    default = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", default))
    mlflow.set_experiment(experiment)


def git_commit() -> str:
    """Current git commit hash, or 'unknown' outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def log_file(path: Path, artifact_dir: str | None = None) -> None:
    """Log a small file as an artifact if it exists."""
    if path.is_file():
        mlflow.log_artifact(str(path), artifact_path=artifact_dir)
