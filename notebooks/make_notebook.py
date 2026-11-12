"""Generates notebooks/colab_runner.ipynb (thin wrapper; all logic lives in scripts/).
Run: python notebooks/make_notebook.py"""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n")})


md("""
# Soup take-home — Colab runner (thin wrapper)
All logic is in `scripts/`. This notebook only: mounts Drive, gets the repo, records the environment,
installs pinned deps, runs the phase scripts, and copies `runs/` to Drive after every step
(Colab disconnects). **Runtime → Change runtime type → T4 GPU.**
""")
code("""
# --- config: edit these two lines ---
REPO_SOURCE = "git"   # "git" or "drive_zip"
REPO = "github.com/danlusman/soup-takehome.git"   # private: needs Colab secret GH_TOKEN (read-only)
BRANCH = "main"
DRIVE_ZIP = "/content/drive/MyDrive/soup_takehome/repo.zip"   # if drive_zip
DRIVE_RUNS = "/content/drive/MyDrive/soup_takehome/runs"
""")
code("""
from google.colab import drive
drive.mount('/content/drive')
import os, subprocess
os.makedirs(DRIVE_RUNS, exist_ok=True)
""")
code("""
%cd /content
if REPO_SOURCE == "git":
    from google.colab import userdata
    _tok = userdata.get('GH_TOKEN')   # never printed; removed from .git/config right after clone
    subprocess.run(["rm", "-rf", "repo"], check=True)
    r = subprocess.run(["git", "clone", "-q", "-b", BRANCH, f"https://x-access-token:{_tok}@{REPO}", "repo"],
                       capture_output=True, text=True)
    del _tok
    if r.returncode != 0:
        raise RuntimeError("git clone failed (check GH_TOKEN secret / repo access); stderr hidden to avoid leaking the token")
    subprocess.run(["git", "-C", "repo", "remote", "set-url", "origin", f"https://{REPO}"], check=True)
else:
    !rm -rf repo && mkdir repo && unzip -q $DRIVE_ZIP -d repo
%cd /content/repo
!git rev-parse HEAD 2>/dev/null || echo "no git sha (zip)"
""")
md("## Phase 0 — environment capture (before and after install)")
code("""
from datetime import datetime, timezone
TS = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
ENV = f'runs/phase0/env_{TS}'
os.makedirs(ENV, exist_ok=True)
!python --version > {ENV}/python_version.txt 2>&1
!nvidia-smi > {ENV}/nvidia-smi.txt 2>&1
!nvidia-smi -q > {ENV}/nvidia-smi-q.txt 2>&1
!free -b > {ENV}/free.txt; df -B1 /content > {ENV}/df.txt
!pip freeze > {ENV}/pip_freeze_before.txt
!pip install -q -r requirements.txt 2>&1 | tail -20 | tee {ENV}/pip_install.txt
# Colab preinstalls torchao 0.10.0; peft 0.21.2 refuses torchao < 0.16 when wrapping LoRA on fp16
# linears (ImportError, runs/phase2/mem_probe_*_20261112T1111*). Nothing in our stack uses torchao.
!pip show torchao 2>&1 | head -3 > {ENV}/torchao_before_uninstall.txt; pip uninstall -y torchao > {ENV}/torchao_uninstall.txt 2>&1
!pip freeze > {ENV}/pip_freeze_after.txt
!soup --version > {ENV}/soup_version.txt 2>&1; cat {ENV}/soup_version.txt
""")
code("""
def sync():
    # copy (never delete) everything under runs/ and data/splits to Drive
    !mkdir -p {DRIVE_RUNS} && cp -rn runs/. {DRIVE_RUNS}/ && mkdir -p {DRIVE_RUNS}/../data_splits && cp -rn data/splits/. {DRIVE_RUNS}/../data_splits/ 2>/dev/null; echo synced
sync()
""")
md("## Phase 1 — data + baseline")
code("""
!python scripts/prepare_data.py --tokenizer Qwen/Qwen2.5-1.5B-Instruct 2>&1 | tail -30
sync()
""")
code("""
!python scripts/eval_task.py --model Qwen/Qwen2.5-1.5B-Instruct --data data/processed/heldout.jsonl --tag base --batch-size 32 2>&1 | tail -30
sync()
""")

md("""
## Phase 2 — memory budget + controlled probes
Each probe is a separate process so its torch peaks and nvidia-smi trace are clean.
Probe (e) is a **pre-registered falsification test**: the budget predicts it does NOT fit
(~15.3 GiB incl. context). An OOM there is a result, not a failure — keep the log.
""")
code("""
!python scripts/memory_budget.py --batch 4 --seq 512 --grad-ckpt 2>&1 | tail -15
!python scripts/memory_budget.py --batch 2 --seq 512 2>&1 | tail -12
sync()
""")
code("""
probes = [
  "--batch 4 --seq 512 --grad-ckpt",            # (a) planned real-run shape
  "--batch 2 --seq 512",                         # (b) no checkpointing: tests the activation term
  "--batch 1 --seq 512 --attn eager",            # (c) eager attention: tests the s^2 term
  "--batch 4 --seq 512 --grad-ckpt --base nf4",  # (d) QLoRA weights term (Soup's default)
  "--batch 8 --seq 256",                         # (e) predicted NOT to fit
]
for a in probes:
    print("=== probe", a)
    !python scripts/mem_probe.py {a} 2>&1 | grep -E "phase|step|trainable|Error|error|Killed" | tail -25 || echo "EXIT NONZERO"
    sync()
""")

md("""
## Phase 3 — Soup checks, then training (real, null control, broken template)
**Do NOT evaluate on held-out here.** SHIP criteria get written first (Phase 5).
Each training run is wrapped by `memtrace.py` (nvidia-smi 1 s CSV, RAM CSV, raw dumps,
per-step telemetry incl. fp16 loss scale / skipped steps, LoRA-B norms).
""")
code("""
for cfg in ["real", "null_lr1e-7", "broken_template"]:
    !python scripts/soup_checks.py --config configs/{cfg}.yaml 2>&1 | grep -E "exit=|Error"
sync()
""")
code("""
!mkdir -p runs/phase3
import shutil
def train(cfg):
    out = f"runs/phase3/train_{cfg}_stdout.txt"
    !python scripts/memtrace.py --phase phase3 --tag train_{cfg} --disk-paths outputs/{cfg} -- soup train --config configs/{cfg}.yaml --yes > {out} 2>&1; echo "exit=$?"
    !grep -vE "it/s\\]|Loading weights" {out} | tail -15
    # keep the adapter + trainer_state (not optimizer.pt) in Drive
    !mkdir -p {DRIVE_RUNS}/../outputs/{cfg} && rsync -a --exclude 'optimizer.pt' --exclude 'rng_state.pth' outputs/{cfg}/ {DRIVE_RUNS}/../outputs/{cfg}/
    sync()
train("real")
""")
code("""
train("null_lr1e-7")
""")
code("""
train("broken_template")   # optional (c)
""")

md("""
## Phase 4 — merge (fp16) + quantize (NF4)
`soup merge` is the primary merge (CPU, fp16), wrapped by memtrace (RAM / VRAM / disk sampled 1 s).
`merge_ref.py` is an independent plain-peft merge with per-phase peaks; verify.py diffs the two.
`quantize.py` writes NF4 with Soup's own `merge_4bit`, also for the BASE (noise floor).
""")
code("""
RUNS = ["real", "null_lr1e-7", "broken_template"]
!mkdir -p runs/phase4
for R in RUNS:
    !python scripts/memtrace.py --phase phase4 --tag soup_merge_{R} --disk-paths outputs/merged/{R} -- soup merge --adapter outputs/{R} --output outputs/merged/{R} --dtype float16 > runs/phase4/soup_merge_{R}_stdout.txt 2>&1; echo "soup merge {R} exit=$?"
    !python scripts/merge_ref.py --adapter outputs/{R} --out outputs/merged_ref/{R} --device cpu 2>&1 | grep -E "phase |WARN|Error"
    sync()
# GPU merge peak, measured once (real run only)
!python scripts/merge_ref.py --adapter outputs/real --out outputs/merged_ref_gpu/real --device cuda 2>&1 | grep -E "phase |Error"
sync()
""")
code("""
!python scripts/quantize.py --src-hub Qwen/Qwen2.5-1.5B-Instruct --src outputs/base_fp16 --out outputs/nf4/base 2>&1 | grep -E "phase |Linear4bit|Error"
for R in RUNS:
    !python scripts/quantize.py --src outputs/merged/{R} --out outputs/nf4/{R} 2>&1 | grep -E "phase |Linear4bit|Error"
    sync()
!df -h /content > runs/phase4/df_after.txt; du -sh outputs/* outputs/*/* >> runs/phase4/df_after.txt; cat runs/phase4/df_after.txt
!mkdir -p {DRIVE_RUNS}/../outputs/nf4 && rsync -a outputs/nf4/ {DRIVE_RUNS}/../outputs/nf4/
sync()
""")
md("""
## Phase 5 — verification (criteria were pre-registered in DECISIONS.md D-20/D-21 before this runs)
`broken_template` is reported alongside but is NOT a null control (`--null-runs null_lr1e-7`).
""")
code("""
!mkdir -p runs/phase5
!python scripts/verify.py --runs real null_lr1e-7 broken_template --ship-run real --null-runs null_lr1e-7 > runs/phase5/verify_stdout.txt 2>&1; echo "exit=$?"
!grep -E "trained-ness|SHIP criteria|weights:|KL " runs/phase5/verify_stdout.txt | cut -c1-250
!cat $(ls -d runs/phase5/verify_*/ | tail -1)report.md
sync()
""")

md("""
## Phase 6 — what does `soup ship` actually evaluate? + is the NF4 collapse the method or the artifact?
Run in a FRESH session: Phase 0 cells (clone, install, torchao uninstall) and `sync()` first; this cell then
restores adapters + NF4 artifacts from Drive and re-creates the merged fp16 dirs (they were not kept in Drive).
""")
code("""
!mkdir -p outputs runs/phase6 data/phase6
!rsync -a {DRIVE_RUNS}/../outputs/ outputs/ --exclude 'checkpoint-*'
!ls outputs outputs/nf4
for R in ["real", "null_lr1e-7"]:
    !soup merge --adapter outputs/{R} --output outputs/merged/{R} --dtype float16 > runs/phase6/remerge_{R}.txt 2>&1; echo "remerge {R} exit=$?"
!python scripts/make_ship_tasks.py --data data/processed/heldout.jsonl --out data/phase6/ship_tasks.jsonl
""")
code("""
# soup ship, live. Exit codes: 0 SHIP, 2 DON'T SHIP, 3 usage, 1 runtime. Keep every output.
SHIP = [
  ("real_adapter",   "--adapter outputs/real --noise-floor 2"),
  ("real_nf4_tuned", "--tuned outputs/nf4/real"),     # can it even see the shipped 4-bit artifact?
  ("null_adapter",   "--adapter outputs/null_lr1e-7"),
  ("broken_adapter", "--adapter outputs/broken_template"),
]
for tag, extra in SHIP:
    !soup ship --base Qwen/Qwen2.5-1.5B-Instruct {extra} --task-eval data/phase6/ship_tasks.jsonl --device cuda --output runs/phase6/ship_{tag}.json --emit-evidence runs/phase6/ship_{tag}_evidence.json > runs/phase6/ship_{tag}_stdout.txt 2>&1; echo "soup ship {tag} exit=$?"
    !tail -25 runs/phase6/ship_{tag}_stdout.txt
    sync()
""")
code("""
!python scripts/nf4_study.py --model Qwen/Qwen2.5-1.5B-Instruct --artifact outputs/nf4/base --tag base 2>&1 | grep -E "acc=|Error"
sync()
!python scripts/nf4_study.py --model outputs/merged/real --artifact outputs/nf4/real --tag real 2>&1 | grep -E "acc=|Error"
sync()
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).with_name("colab_runner.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out)
