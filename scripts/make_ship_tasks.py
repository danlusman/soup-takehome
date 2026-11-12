"""Phase 6: turn our held-out set into a `soup ship --task-eval` file.

Each line: {"prompt": <our exact user prompt>, "expected": <exact target JSON>, "scoring": "exact",
"category": <label word>}. Soup's `exact` scorer = stripped, lower-cased whole-output equality
(soup_cli/eval/custom.py:score_exact), i.e. the same as our strict_acc for this task.
soup ship renders `prompt` as a single user turn through the tokenizer's chat template
(soup_cli/utils/live_eval.py:_render_prompt_template), the same as our eval.

Example:
  python scripts/make_ship_tasks.py --data data/processed/heldout.jsonl --out data/phase6/ship_tasks.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_jsonl, write_jsonl  # noqa: E402
from task_format import LABELS, build_prompt, build_target  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/processed/heldout.jsonl")
    p.add_argument("--out", default="data/phase6/ship_tasks.jsonl")
    a = p.parse_args()
    rows = read_jsonl(a.data)
    out = [{"prompt": build_prompt(r["text"]), "expected": build_target(r["label"]), "scoring": "exact",
            "category": LABELS[r["label"]]} for r in rows]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(a.out, out)
    print(f"wrote {len(out)} tasks to {a.out}")


if __name__ == "__main__":
    main()
