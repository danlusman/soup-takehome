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

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).with_name("colab_runner.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out)
