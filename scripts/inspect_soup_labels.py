"""Pre-flight check of what Soup will ACTUALLY train on, using Soup's own code path.

Loads the Soup config with soup_cli.config.loader.load_config, builds the tokenizer exactly
as Soup does (including data.chat_template overrides), runs Soup's row formatter
(soup_cli.data.sft_format.build_format_row) on every row of the training file, and reports:

  * rendered input vs the reference render our eval uses (tokenizer's shipped template):
    identical / different (-> train/eval template mismatch)
  * trained span = tokens with label != -100, decoded
  * whether the trained span == exactly  <target JSON> + <|im_end|>  (what we intend)
  * whether <|im_end|> (the stop token) is trained, whether role-prefix tokens leak into
    the loss, whether any row has ZERO trained tokens (silently contributes nothing)

This replaces reading the colour-only `soup data doctor --show-mask` output, which is lost
when redirected to a file.

Example:
  python scripts/inspect_soup_labels.py --config configs/real.yaml
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_jsonl, setup_run, sha256_text, write_jsonl, write_summary  # noqa: E402
from task_format import build_messages, parse_response  # noqa: E402

IGNORE = -100


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--examples", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--phase", default="phase3")
    p.add_argument("--runs-root", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    run_dir, log = setup_run(args.phase, f"inspect_labels_{Path(args.config).stem}", args, args.runs_root)
    from soup_cli.config.loader import load_config
    from soup_cli.data.sft_format import build_format_row
    from transformers import AutoTokenizer

    cfg = load_config(Path(args.config))
    tok = AutoTokenizer.from_pretrained(cfg.base)          # what Soup will modify in place
    ref = AutoTokenizer.from_pretrained(cfg.base)          # untouched: what eval_task.py uses
    fmt = build_format_row(tok, cfg.data, console=None, training_cfg=cfg.training)
    log.info("soup chat_template override=%r ; template sha soup=%s ref=%s", cfg.data.chat_template,
             sha256_text(tok.chat_template or "")[:12], sha256_text(ref.chat_template or "")[:12])

    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    rows = read_jsonl(cfg.data.train)
    if args.limit:
        rows = rows[:args.limit]
    stats = Counter()
    per_row, examples = [], []
    for i, row in enumerate(rows):
        out = fmt(row)
        ids, labels = list(out["input_ids"]), list(out["labels"])
        trained = [t for t, l_ in zip(ids, labels) if l_ != IGNORE]
        target = row["messages"][-1]["content"]
        want = tok(target, add_special_tokens=False)["input_ids"] + [im_end]
        ref_ids = ref.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False)
        if hasattr(ref_ids, "keys"):
            ref_ids = ref_ids["input_ids"]
        ref_ids = list(ref_ids)
        # reference render ends with "<|im_end|>\n"; soup may or may not keep the trailing "\n"
        same_render = ids == ref_ids or ids == ref_ids[:-1]
        # v2 (added after fault injection showed two misses, runs/phase6/fault_inject_20261112T132518Z):
        # (a) data contract: the training target itself must be a valid task answer (catches empty/garbled targets)
        # (b) serving parity: the training render must equal what OUR eval/serving prompt builder produces for the
        #     same review (catches system prompts / extra turns that exist only in the training data)
        contract_ok = parse_response(target)["strict_valid"]
        user = next((m["content"] for m in row["messages"] if m["role"] == "user"), "")
        review = user.split("Отзыв:\n", 1)[-1]
        label = parse_response(target)["strict_label"]
        parity = False
        if label is not None:
            ev_ids = ref.apply_chat_template(build_messages(review, label), tokenize=True, add_generation_prompt=False)
            ev_ids = list(ev_ids["input_ids"] if hasattr(ev_ids, "keys") else ev_ids)
            parity = ids == ev_ids or ids == ev_ids[:-1]
        r = {
            "target_contract_valid": contract_ok,
            "render_matches_serving_prompt_builder": parity,
            "row": i, "n_input": len(ids), "n_trained": len(trained),
            "trained_text": tok.decode(trained, skip_special_tokens=False),
            "trained_equals_target_plus_im_end": trained == want,
            "trained_equals_target_only": trained == want[:-1],
            "im_end_trained": im_end in trained,
            "role_prefix_in_loss": "assistant" in tok.decode(trained),
            "render_matches_eval_template": same_render,
            "render_len_diff_vs_eval": len(ids) - len(ref_ids),
        }
        per_row.append(r)
        for k in ("trained_equals_target_plus_im_end", "trained_equals_target_only", "im_end_trained",
                  "role_prefix_in_loss", "render_matches_eval_template", "target_contract_valid",
                  "render_matches_serving_prompt_builder"):
            stats[k] += bool(r[k])
        stats["zero_trained_rows"] += r["n_trained"] == 0
        if i < args.examples:
            examples.append({"rendered_input": tok.decode(ids, skip_special_tokens=False),
                             "trained_text": r["trained_text"]})
    write_jsonl(run_dir / "per_row.jsonl", per_row)
    n = len(rows)
    summary = {"config": args.config, "n_rows": n, "chat_template_override": cfg.data.chat_template,
               "soup_template_sha256": sha256_text(tok.chat_template or ""),
               "eval_template_sha256": sha256_text(ref.chat_template or ""),
               "train_on_responses_only": cfg.data.train_on_responses_only,
               "train_on_eot": getattr(cfg.training, "train_on_eot", None),
               "counts": dict(stats), "rates": {k: round(v / n, 4) for k, v in stats.items()},
               "examples": examples}
    for k, v in stats.items():
        log.info("%-36s %d / %d", k, v, n)
    for e in examples[:1]:
        log.info("example rendered input:\n%s", e["rendered_input"])
        log.info("example trained span: %r", e["trained_text"])
    write_summary(run_dir, summary, log)


if __name__ == "__main__":
    main()
