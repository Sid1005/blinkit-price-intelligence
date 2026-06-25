"""Experiment tracking — W&B when available, offline JSONL fallback otherwise.

The agent cannot create the user's W&B account/project, so the default path writes a
W&B-compatible JSONL record to ``data/runs/``. When ``WANDB_API_KEY`` is set and the
``wandb`` package is installed, the same API also initialises a real W&B run and logs
the identical schema.

Small API:
    run = start_run(project, name, config={...}, tags=[...])
    log_metrics(run, {"macro_f1": 0.8}, step=1)
    log_artifact(run, "data/lora_adapter", type="model")
    finish_run(run, status="finished", summary={...})

Acceptance: ``python -m app.monitoring.experiment_tracking`` works with WANDB_API_KEY
unset and writes JSONL containing run_id, config, metrics, artifacts, git SHA,
timestamps, and status.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app import config


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(config.REPO_ROOT), stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _wandb_available() -> bool:
    if not (config.WANDB_API_KEY or os.environ.get("WANDB_MODE") == "offline"):
        return False
    try:
        import wandb  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


@dataclass
class Run:
    run_id: str
    project: str
    name: str
    backend: str                      # "wandb" | "offline"
    config: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    metrics: list[dict] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)
    git_sha: str = ""
    started_at: str = ""
    finished_at: str = ""
    status: str = "running"
    _wandb_run: object | None = None
    _path: str = ""


def start_run(project: str = "india-commerce-signalforge", name: str | None = None,
              config_dict: dict | None = None, tags: list[str] | None = None) -> Run:
    run_id = uuid.uuid4().hex[:12]
    name = name or f"run-{run_id}"
    run = Run(run_id=run_id, project=project, name=name,
              backend="wandb" if _wandb_available() else "offline",
              config=config_dict or {}, tags=tags or [], git_sha=_git_sha(),
              started_at=datetime.now(timezone.utc).isoformat())
    run._path = str(config.RUNS_DIR / f"{run_id}.json")
    if run.backend == "wandb":
        try:
            import wandb
            run._wandb_run = wandb.init(project=project, name=name,
                                        config=run.config, tags=run.tags, reinit=True)
        except Exception:  # noqa: BLE001 — degrade to offline
            run.backend = "offline"
    _flush(run)
    return run


def log_metrics(run: Run, metrics: dict, step: int | None = None) -> None:
    rec = {"step": step, "ts": time.time(), **metrics}
    run.metrics.append(rec)
    if run.backend == "wandb" and run._wandb_run is not None:
        try:
            run._wandb_run.log(metrics, step=step)
        except Exception:  # noqa: BLE001
            pass
    _flush(run)


def log_artifact(run: Run, path: str, type: str = "model", name: str | None = None) -> None:  # noqa: A002
    rec = {"path": str(path), "type": type, "name": name or os.path.basename(str(path))}
    run.artifacts.append(rec)
    if run.backend == "wandb" and run._wandb_run is not None:
        try:
            import wandb
            art = wandb.Artifact(rec["name"], type=type)
            if os.path.isdir(path):
                art.add_dir(path)
            elif os.path.exists(path):
                art.add_file(path)
            run._wandb_run.log_artifact(art)
        except Exception:  # noqa: BLE001
            pass
    _flush(run)


def finish_run(run: Run, status: str = "finished", summary: dict | None = None) -> str:
    run.status = status
    run.finished_at = datetime.now(timezone.utc).isoformat()
    if summary:
        run.config.setdefault("_summary", {}).update(summary)
    if run.backend == "wandb" and run._wandb_run is not None:
        try:
            if summary:
                run._wandb_run.summary.update(summary)
            run._wandb_run.finish()
        except Exception:  # noqa: BLE001
            pass
    _flush(run)
    return run._path


def _flush(run: Run) -> None:
    payload = {
        "run_id": run.run_id, "project": run.project, "name": run.name,
        "backend": run.backend, "config": run.config, "tags": run.tags,
        "metrics": run.metrics, "artifacts": run.artifacts, "git_sha": run.git_sha,
        "started_at": run.started_at, "finished_at": run.finished_at,
        "status": run.status,
    }
    with open(run._path, "w") as f:
        json.dump(payload, f, indent=2)


def load_runs(limit: int | None = None) -> list[dict]:
    """Read offline run logs newest-first (for the dashboard tracking pane)."""
    files = sorted(config.RUNS_DIR.glob("*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    runs = []
    for fp in files[:limit] if limit else files:
        try:
            runs.append(json.loads(fp.read_text()))
        except Exception:  # noqa: BLE001
            continue
    return runs


if __name__ == "__main__":
    r = start_run(name="smoke-test", config_dict={"lr": 1e-3, "model": "bert-tiny"},
                  tags=["smoke"])
    for step in range(3):
        log_metrics(r, {"loss": 1.0 / (step + 1), "macro_f1": 0.5 + 0.1 * step}, step=step)
    log_artifact(r, str(config.DATA_DIR), type="dataset", name="data-dir")
    path = finish_run(r, summary={"best_macro_f1": 0.7})
    print(f"backend={r.backend} wrote {path}")
    print(json.dumps(load_runs(limit=1)[0]["metrics"], indent=2))
