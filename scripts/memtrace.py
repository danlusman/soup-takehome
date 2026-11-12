"""Measurement harness: torch peaks per phase, 1 s nvidia-smi CSV, 1 s RAM CSV, raw nvidia-smi
dumps at phase boundaries, disk usage snapshots.

Library use (inside our own scripts):
    from memtrace import MemTracker
    with MemTracker(run_dir) as mt:
        with mt.phase("load_model"):
            ...
        mt.snapshot_disk("after_save", ["out/merged"])

CLI use (wrap any command, e.g. soup train):
    python scripts/memtrace.py --phase phase3 --tag train_real -- soup train --config cfg.yaml --yes
  * the nvidia-smi and RAM samplers run for the whole command
  * if the command is `soup ...`, it runs IN-PROCESS (soup_cli.cli:run) so we can read
    torch.cuda.max_memory_* at the end, and a TrainerCallback is attached to every
    transformers.Trainer to record per-step peaks (step 1 has no Adam states yet; step 2
    is the steady state). Patching only adds a callback; it does not change training.
  * any other command runs as a subprocess (only nvidia-smi / RAM are seen).

On a machine without nvidia-smi / CUDA (local smoke tests) every GPU field is null and the
reason is logged; nothing is invented.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import setup_run, utc_iso, write_summary  # noqa: E402

SMI_FIELDS = ("timestamp,index,name,memory.used,memory.total,utilization.gpu,"
              "utilization.memory,power.draw,temperature.gpu,clocks.sm,pstate")


def _have_smi() -> bool:
    return shutil.which("nvidia-smi") is not None


def _cuda():
    try:
        import torch
        return torch if torch.cuda.is_available() else None
    except ImportError:
        return None


def _rss_bytes() -> int | None:
    try:
        import psutil
        p = psutil.Process()
        return p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))
    except Exception:  # noqa: BLE001
        return None


class NvidiaSmiSampler:
    """`nvidia-smi --query-gpu=... -l 1 -f file` in the background. Raw CSV, untouched."""

    def __init__(self, path: Path, interval_s: int = 1):
        self.path, self.interval, self.proc = path, interval_s, None

    def start(self):
        if not _have_smi():
            return self
        self.proc = subprocess.Popen(["nvidia-smi", f"--query-gpu={SMI_FIELDS}", "--format=csv",
                                      "-l", str(self.interval), "-f", str(self.path)])
        return self

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class RamSampler(threading.Thread):
    """Process-tree RSS + system used/available RAM every interval, to CSV (UTC ISO time)."""

    def __init__(self, path: Path, interval_s: float = 1.0):
        super().__init__(daemon=True)
        self.path, self.interval, self._halt = path, interval_s, threading.Event()

    def run(self):
        try:
            import psutil
        except ImportError:
            self.path.write_text("psutil not installed\n", encoding="utf-8")
            return
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["utc", "proc_tree_rss_bytes", "sys_used_bytes", "sys_available_bytes"])
            while not self._halt.is_set():
                vm = psutil.virtual_memory()
                w.writerow([utc_iso(), _rss_bytes(), vm.total - vm.available, vm.available])
                f.flush()
                self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=5)


class MemTracker:
    def __init__(self, run_dir: Path, logger=None, smi_dumps: bool = True):
        self.dir = Path(run_dir)
        self.log = logger
        self.smi_dumps = smi_dumps
        self.events: list[dict] = []
        self._n = 0
        self.smi = NvidiaSmiSampler(self.dir / "nvidia_smi_1s.csv")
        self.ram = RamSampler(self.dir / "ram_1s.csv")
        (self.dir / "nvidia_smi_dumps").mkdir(exist_ok=True)

    def __enter__(self):
        self.smi.start()
        self.ram.start()
        if not _have_smi():
            self._info("nvidia-smi not found: GPU sampler disabled")
        self.dump_smi("start")
        return self

    def __exit__(self, *exc):
        self.dump_smi("end")
        self.smi.stop()
        self.ram.stop()
        with open(self.dir / "phases.jsonl", "w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e) + "\n")
        return False

    def _info(self, msg, *a):
        if self.log:
            self.log.info(msg, *a)

    def dump_smi(self, label: str):
        if not (self.smi_dumps and _have_smi()):
            return
        self._n += 1
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
        p = self.dir / "nvidia_smi_dumps" / f"{self._n:03d}_{label}.txt"
        p.write_text(f"# utc={utc_iso()} label={label}\n{out}", encoding="utf-8")

    def torch_stats(self) -> dict:
        t = _cuda()
        if not t:
            return {"cuda": False}
        t.cuda.synchronize()
        return {"cuda": True,
                "allocated": t.cuda.memory_allocated(), "reserved": t.cuda.memory_reserved(),
                "max_allocated": t.cuda.max_memory_allocated(),
                "max_reserved": t.cuda.max_memory_reserved()}

    @contextmanager
    def phase(self, name: str, **extra):
        t = _cuda()
        self.dump_smi(f"{name}_begin")
        before = self.torch_stats()
        if t:
            t.cuda.reset_peak_memory_stats()
        rss0, t0, u0 = _rss_bytes(), time.perf_counter(), utc_iso()
        try:
            yield
        finally:
            after = self.torch_stats()
            ev = {"phase": name, "utc_begin": u0, "utc_end": utc_iso(),
                  "wall_s": round(time.perf_counter() - t0, 3),
                  "torch_before": before, "torch_after": after,
                  "rss_before": rss0, "rss_after": _rss_bytes(), **extra}
            self.events.append(ev)
            self.dump_smi(f"{name}_end")
            if after.get("cuda"):
                self._info("phase %-22s max_alloc=%.3f GiB max_reserved=%.3f GiB alloc_now=%.3f GiB %.1fs",
                           name, after["max_allocated"] / 2**30, after["max_reserved"] / 2**30,
                           after["allocated"] / 2**30, ev["wall_s"])
            else:
                self._info("phase %-22s (no CUDA) rss=%s %.1fs", name, ev["rss_after"], ev["wall_s"])

    def snapshot_disk(self, label: str, paths: list[str]):
        sizes = {}
        for p in paths:
            pp = Path(p)
            sizes[p] = (sum(f.stat().st_size for f in pp.rglob("*") if f.is_file())
                        if pp.is_dir() else pp.stat().st_size if pp.exists() else None)
        du = shutil.disk_usage(self.dir)
        ev = {"phase": f"disk:{label}", "utc": utc_iso(), "sizes_bytes": sizes,
              "disk_used": du.used, "disk_free": du.free}
        self.events.append(ev)
        self._info("disk %s %s free=%.1f GiB", label, sizes, du.free / 2**30)


# ---------------------------------------------------------------- in-process soup wrapping
def _patch_trainer(events: list, every: int):
    """Attach a memory callback to every transformers.Trainer created from now on."""
    import transformers
    from transformers import TrainerCallback

    class MemCallback(TrainerCallback):
        def _rec(self, tag, state):
            t = _cuda()
            if not t:
                return
            events.append({"phase": f"trainer:{tag}", "utc": utc_iso(), "step": state.global_step,
                           "max_allocated": t.cuda.max_memory_allocated(),
                           "max_reserved": t.cuda.max_memory_reserved(),
                           "allocated": t.cuda.memory_allocated()})

        def on_train_begin(self, args, state, control, **kw):
            self._rec("train_begin", state)
            t = _cuda()
            if t:
                t.cuda.reset_peak_memory_stats()

        def on_step_end(self, args, state, control, **kw):
            if state.global_step <= 3 or state.global_step % every == 0:
                self._rec("step_end", state)

        def on_evaluate(self, args, state, control, **kw):
            self._rec("evaluate", state)

        def on_save(self, args, state, control, **kw):
            self._rec("save", state)

        def on_train_end(self, args, state, control, **kw):
            self._rec("train_end", state)

    orig = transformers.Trainer.__init__

    def init(self, *a, **kw):
        orig(self, *a, **kw)
        self.add_callback(MemCallback())

    transformers.Trainer.__init__ = init


def parse_args(argv):
    if "--" not in argv:
        raise SystemExit("usage: memtrace.py [--phase P] [--tag T] -- <command ...>")
    i = argv.index("--")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", default="phase3")
    p.add_argument("--tag", default="wrapped")
    p.add_argument("--step-every", type=int, default=10)
    p.add_argument("--disk-paths", nargs="*", default=[])
    p.add_argument("--runs-root", default=None)
    args = p.parse_args(argv[:i])
    args.command = argv[i + 1:]
    return args


def main():
    args = parse_args(sys.argv[1:])
    run_dir, log = setup_run(args.phase, f"memtrace_{args.tag}", args, args.runs_root)
    rc, err = 0, None
    with MemTracker(run_dir, log) as mt:
        mt.snapshot_disk("before", args.disk_paths)
        with mt.phase("command", command=args.command):
            if args.command and Path(args.command[0]).stem == "soup":
                _patch_trainer(mt.events, args.step_every)
                from soup_cli.cli import run as soup_run
                sys.argv = ["soup", *args.command[1:]]
                try:
                    soup_run()
                except SystemExit as e:  # typer exits via SystemExit
                    rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
                except Exception as e:  # noqa: BLE001  record, don't hide
                    rc, err = 1, repr(e)
                    log.exception("command raised")
            else:
                rc = subprocess.run(args.command).returncode
        mt.snapshot_disk("after", args.disk_paths)
    peak_smi = None
    csv_path = run_dir / "nvidia_smi_1s.csv"
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        try:
            col = [c.strip() for c in rows[0]].index("memory.used [MiB]")
            peak_smi = max(int(r[col].split()[0]) for r in rows[1:] if len(r) > col)
        except (ValueError, IndexError):
            peak_smi = None
    log.info("command exit code=%s peak nvidia-smi memory.used=%s MiB", rc, peak_smi)
    write_summary(run_dir, {"command": args.command, "exit_code": rc, "error": err,
                            "peak_nvidia_smi_used_MiB": peak_smi,
                            "final_torch": mt.torch_stats()}, log)
    sys.exit(rc)


if __name__ == "__main__":
    main()
