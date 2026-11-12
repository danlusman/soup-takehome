"""Derive every run config from configs/real.yaml so variants differ ONLY in the stated lines.
Writes: null_lr1e-7.yaml, broken_template.yaml and smoke_*.yaml (CPU, Qwen2.5-0.5B, data_smoke).
Prints a unified diff of each variant against real.yaml, so the difference is auditable.

Run: python configs/make_configs.py
"""
import difflib
from pathlib import Path

HERE = Path(__file__).resolve().parent
real = (HERE / "real.yaml").read_text(encoding="utf-8")


def sub(text: str, pairs) -> str:
    for old, new in pairs:
        assert text.count(old) == 1, f"expected exactly one {old!r}"
        text = text.replace(old, new)
    return text


VARIANTS = {
    "null_lr1e-7": [
        ("# (a) REAL RUN.", "# (b) NULL CONTROL: identical to real.yaml except lr + output (DECISIONS D-14)."),
        ("  lr: 2.0e-4\n", "  lr: 1.0e-7                     # NULL CONTROL (D-14)\n"),
        ("output: outputs/real", "output: outputs/null_lr1e-7"),
    ],
    "broken_template": [
        ("# (a) REAL RUN.", "# (c) BROKEN TEMPLATE: identical to real.yaml except chat_template + output (D-15)."),
        ("  # chat_template: unset -> the tokenizer's own Qwen2.5 template (same one eval_task.py uses)\n",
         "  chat_template: qwen2.5          # BROKEN ON PURPOSE: Soup maps it to plain ChatML, no default system prompt (D-15)\n"),
        ("output: outputs/real", "output: outputs/broken_template"),
    ],
}

SMOKE = [
    ("base: Qwen/Qwen2.5-1.5B-Instruct", "base: Qwen/Qwen2.5-0.5B-Instruct"),
    ("  train: data/processed/trainval_soup.jsonl", "  train: data_smoke/processed/trainval_soup.jsonl"),
    ("  val_split: 0.0909", "  val_split: 0.25"),  # smoke: 9 train + 3 val
    ("  epochs: 2\n", "  epochs: 1\n"),
    ("  batch_size: 4 ", "  batch_size: 2 "),
    ("  gradient_accumulation_steps: 4 ", "  gradient_accumulation_steps: 1 "),
    ("  eval_steps: 20", "  eval_steps: 2"),
    ("  save_steps: 50", "  save_steps: 4"),
]


def write(name: str, text: str):
    (HERE / f"{name}.yaml").write_text(text, encoding="utf-8", newline="\n")
    diff = difflib.unified_diff(real.splitlines(), text.splitlines(), "real.yaml", f"{name}.yaml", lineterm="", n=0)
    print("\n".join(diff) or f"{name}: identical")


for name, pairs in VARIANTS.items():
    write(name, sub(real, pairs))
for name, pairs in [("real", []), *VARIANTS.items()]:
    text = sub(sub(real, pairs), SMOKE)
    text = text.replace("output: outputs/", "output: outputs_smoke/")
    write(f"smoke_{name}", text)
