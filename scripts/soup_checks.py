"""Run Soup's built-in checks for one config and record raw output + exit codes.

Checks (all verified to exist in soup-cli 0.77.2 via --help):
  data_doctor : soup data doctor <train file> --model <base> --max-length N --show-mask 2 -o report.json
                NOTE: takes only --model, so it CANNOT see data.chat_template overrides in the config.
  doctor      : soup doctor --config <cfg>
  plan        : soup plan --config <cfg> --state <file>      (Soup's "pre-flight training plan")
  dry_run     : soup train --config <cfg> --dry-run --yes      (config + data validation; the train
                command's analytical VRAM gate also lives here)
plus our own: scripts/inspect_soup_labels.py (what Soup actually puts in the loss).

Nothing is interpreted as pass/fail here except the exit code and, for data doctor, the
overall verdict string; the raw text is kept for the evidence table (Phase 6).

Example:
  python scripts/soup_checks.py --config configs/real.yaml
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import setup_run, write_summary  # noqa: E402

PY = sys.executable


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--skip", nargs="*", default=[], help="check names to skip")
    p.add_argument("--tag", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--phase", default="phase3")
    p.add_argument("--runs-root", default=None)
    return p.parse_args()


def soup_cmd() -> list[str]:
    return [PY, "-m", "soup_cli"]


def run(name, cmd, run_dir, log, timeout=1800):
    env = {**os.environ, "COLUMNS": "200", "PYTHONIOENCODING": "utf-8", "NO_COLOR": "1"}
    log.info("[%s] %s", name, " ".join(cmd))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, env=env)
        rc, out = r.returncode, (r.stdout or "") + ("\n--- stderr ---\n" + r.stderr if r.stderr else "")
    except subprocess.TimeoutExpired as e:
        rc, out = "timeout", f"TIMEOUT after {timeout}s\n{e.stdout or ''}"
    (run_dir / f"{name}.txt").write_text(f"$ {' '.join(cmd)}\n# exit={rc}\n{out}", encoding="utf-8")
    log.info("[%s] exit=%s", name, rc)
    return rc, out


def main():
    args = parse_args()
    tag = args.tag or Path(args.config).stem
    run_dir, log = setup_run(args.phase, f"soup_checks_{tag}", args, args.runs_root)
    from soup_cli.config.loader import load_config
    cfg = load_config(Path(args.config))
    train_file = cfg.data.train if isinstance(cfg.data.train, str) else cfg.data.train[0]
    res = {}
    checks = {
        "data_doctor": soup_cmd() + ["data", "doctor", train_file, "--model", cfg.base,
                                     "--max-length", str(cfg.data.max_length), "--show-mask", "2",
                                     "-o", str(run_dir / "data_doctor_report.json")],
        "doctor": soup_cmd() + ["doctor", "--config", args.config],
        "plan": soup_cmd() + ["plan", "--config", args.config, "--state", str(run_dir / "plan.tfstate")],
        "dry_run": soup_cmd() + ["train", "--config", args.config, "--dry-run", "--yes"],
        "inspect_labels": [PY, str(Path(__file__).with_name("inspect_soup_labels.py")), "--config",
                           args.config, "--phase", args.phase] + (["--runs-root", args.runs_root] if args.runs_root else []),
    }
    for name, cmd in checks.items():
        if name in args.skip:
            continue
        rc, out = run(name, cmd, run_dir, log)
        entry = {"exit": rc}
        if name == "data_doctor":
            m = re.search(r"overall[^A-Z]*\b(OK|MINOR|MAJOR|FAIL\w*|ERROR|CRITICAL)\b", out)
            entry["overall"] = m.group(1) if m else None
            entry["non_ok_checks"] = re.findall(r"│\s*(\w+)\s*│\s*(MINOR|MAJOR|FAIL\w*|ERROR|WARN\w*)\s*│", out)
        if name == "plan":
            m = re.search(r"peak_vram_gb\s*│\s*([\d.]+)", out)
            entry["peak_vram_gb"] = float(m.group(1)) if m else None
        if name == "dry_run":
            entry["says_ready"] = "Ready to train" in out
        res[name] = entry
    write_summary(run_dir, {"config": args.config, "base": cfg.base, "train_file": train_file,
                            "chat_template_override": cfg.data.chat_template, "checks": res}, log)


if __name__ == "__main__":
    main()
