"""Shared helpers for every Colab-side script.

Every script:
  * gets a run directory  runs/<phase>/<script>_<UTC timestamp>/
  * logs to stdout and <run_dir>/log.txt with UTC ISO timestamps
  * fixes seeds (python, numpy, torch)
  * writes env.json (versions, GPU, git sha) and args.json at start
  * writes summary.json at the end (write_summary)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TRACKED_PACKAGES = [
    "soup-cli", "torch", "transformers", "peft", "trl", "accelerate", "datasets",
    "bitsandbytes", "tokenizers", "safetensors", "numpy", "huggingface-hub",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def capture_env() -> dict:
    env = {
        "utc": utc_iso(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {p: _pkg_version(p) for p in TRACKED_PACKAGES},
        "cuda_available": None,
        "gpus": [],
        "git_sha": None,
    }
    try:
        import torch
        env["cuda_available"] = torch.cuda.is_available()
        env["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                env["gpus"].append({
                    "name": p.name,
                    "total_memory_bytes": p.total_memory,
                    "capability": f"{p.major}.{p.minor}",
                })
    except ImportError:
        pass
    try:
        env["git_sha"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
            text=True, timeout=10,
        ).stdout.strip() or None
    except Exception:
        pass
    return env


class _UTCFormatter(logging.Formatter):
    converter = time.gmtime

    def formatTime(self, record, datefmt=None):  # noqa: N802
        return datetime.fromtimestamp(record.created, timezone.utc).isoformat(
            timespec="milliseconds")


def setup_run(phase: str, script: str, args, runs_root: str | Path | None = None):
    """Create the run dir, configure logging, seed, dump env + args. Returns (run_dir, logger)."""
    runs_root = Path(runs_root) if runs_root else REPO_ROOT / "runs"
    run_dir = runs_root / phase / f"{script}_{utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    logger = logging.getLogger(script)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = _UTCFormatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(run_dir / "log.txt", encoding="utf-8")):
        h.setFormatter(fmt)
        logger.addHandler(h)

    seed = getattr(args, "seed", None)
    if seed is not None:
        set_seed(seed)

    write_json(run_dir / "args.json", vars(args))
    write_json(run_dir / "env.json", capture_env())
    logger.info("run_dir=%s", run_dir)
    logger.info("argv=%s", " ".join(sys.argv))
    return run_dir, logger


def write_json(path: str | Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str),
                          encoding="utf-8")


def write_jsonl(path: str | Path, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def write_summary(run_dir: Path, summary: dict, logger=None) -> None:
    summary = {"finished_utc": utc_iso(), **summary}
    write_json(run_dir / "summary.json", summary)
    if logger:
        logger.info("summary written: %s", run_dir / "summary.json")
