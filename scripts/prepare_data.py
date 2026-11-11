"""Phase 1: build train / val / held-out splits for RuReviews -> strict-JSON sentiment.

Pipeline (every step is counted in summary.json, with sample rows for each drop reason):
  1. load the pinned source revision (all splits pooled; source splits are NOT trusted as
     disjoint, we re-split ourselves)
  2. clean: NFC, strip soft hyphens / zero-width chars, collapse whitespace
  3. filter: too short, U+FFFD mojibake, low Cyrillic ratio (English / translit reviews)
  4. dedupe on a normalized key; texts that appear with conflicting labels are dropped
  5. tokenizer analysis with the *training* chat template: prompt / answer / full lengths,
     truncation rates at several max lengths, prompt-is-prefix-of-full check
  6. drop rows whose full sequence > --max-seq-len (right truncation would cut the label,
     which is the only supervised token span)
  7. hash split: u = sha256(salt + key) -> [0,1). Disjoint pools by u range, then balanced
     per-class selection by hash order. Held-out is fixed by the hash, not by row order.
  8. near-duplicate check held-out vs train (word 3-shingle Jaccard)

Example:
  python scripts/prepare_data.py --tokenizer Qwen/Qwen2.5-1.5B-Instruct
  python scripts/prepare_data.py --tokenizer Qwen/Qwen2.5-0.5B-Instruct \
      --train-per-class 3 --val-per-class 1 --heldout-per-class 3 --out-data data_smoke --phase smoke
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REPO_ROOT, setup_run, sha256_file, sha256_text, write_json,  # noqa: E402
                    write_jsonl, write_summary)
from task_format import LABELS, SOURCE_LABEL_TEXT, build_messages  # noqa: E402

DATASET = "ai-forever/ru-reviews-classification"
REVISION = "0f67d914f396ce22917dc6463ec619799b3b08d2"  # pinned 2026-11-11 via HF API
INVISIBLE = re.compile("[\u00ad\u200b\u200c\u200d\u2060\ufeff]")
CYR = re.compile("[\u0400-\u04FF]")
LAT = re.compile("[A-Za-z]")
WORD = re.compile(r"\w+", re.UNICODE)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--min-chars", type=int, default=15)
    p.add_argument("--min-cyr-ratio", type=float, default=0.8,
                   help="min share of Cyrillic among Cyrillic+Latin letters")
    p.add_argument("--train-per-class", type=int, default=500)
    p.add_argument("--val-per-class", type=int, default=50)
    p.add_argument("--heldout-per-class", type=int, default=84)
    p.add_argument("--heldout-frac", type=float, default=0.10, help="u-range [0, f) is the held-out pool")
    p.add_argument("--val-frac", type=float, default=0.05, help="u-range [hf, hf+vf) is the val pool")
    p.add_argument("--salt", default="soup-rureviews-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-data", default="data", help="relative to repo root")
    p.add_argument("--phase", default="phase1")
    p.add_argument("--runs-root", default=None)
    return p.parse_args()


def clean(text: str) -> str:
    t = unicodedata.normalize("NFC", text)
    t = INVISIBLE.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


def dedupe_key(text: str) -> str:
    return " ".join(WORD.findall(text.lower()))


def cyr_ratio(text: str) -> float:
    c, l_ = len(CYR.findall(text)), len(LAT.findall(text))
    return c / (c + l_) if (c + l_) else 0.0


def hash_u(salt: str, key: str) -> tuple[str, float]:
    h = sha256_text(salt + "\x00" + key)
    return h, int(h[:15], 16) / 16 ** 15


def pct(xs, qs=(0, 50, 90, 95, 99, 100)):
    if not xs:
        return {}
    s = sorted(xs)
    return {f"p{q}": s[min(len(s) - 1, round(q / 100 * (len(s) - 1)))] for q in qs} | {
        "mean": round(statistics.fmean(s), 2), "n": len(s)}


def shingles(text: str, k: int = 3) -> set:
    w = WORD.findall(text.lower())
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def main():
    args = parse_args()
    run_dir, log = setup_run(args.phase, "prepare_data", args, args.runs_root)

    from datasets import load_dataset
    from transformers import AutoConfig, AutoTokenizer

    # 1. load
    ds = load_dataset(args.dataset, revision=args.revision)
    source_counts = {k: len(v) for k, v in ds.items()}
    log.info("source splits: %s", source_counts)
    rows = []
    label_text_mismatch = 0
    for split, d in ds.items():
        for r in d:
            if SOURCE_LABEL_TEXT.get(r["label"]) != r.get("label_text"):
                label_text_mismatch += 1
            rows.append({"id": str(r["id"]), "source_split": split, "label": int(r["label"]),
                         "raw": r["text"]})
    log.info("pooled rows=%d label/label_text mismatches=%d", len(rows), label_text_mismatch)
    if label_text_mismatch:
        log.warning("label id -> label_text mapping is NOT what task_format assumes")

    # 2-3. clean + filter
    drops = Counter()
    drop_examples = defaultdict(list)
    changed_by_clean = 0
    kept = []
    for r in rows:
        t = clean(r["raw"])
        if t != r["raw"].strip():
            changed_by_clean += 1
        reason = None
        if "\ufffd" in t:
            reason = "mojibake_U+FFFD"
        elif len(t) < args.min_chars:
            reason = f"shorter_than_{args.min_chars}_chars"
        elif cyr_ratio(t) < args.min_cyr_ratio:
            reason = f"cyr_ratio_below_{args.min_cyr_ratio}"
        if reason:
            drops[reason] += 1
            if len(drop_examples[reason]) < 5:
                drop_examples[reason].append(r["raw"][:200])
            continue
        r["text"] = t
        kept.append(r)
    log.info("after filters: %d (drops=%s, changed_by_clean=%d)", len(kept), dict(drops), changed_by_clean)

    # 4. dedupe
    groups = defaultdict(list)
    for r in kept:
        groups[dedupe_key(r["text"])].append(r)
    deduped, n_exact_dup_rows, n_conflict_groups, n_conflict_rows = [], 0, 0, 0
    conflict_examples = []
    for key, g in groups.items():
        labels = {r["label"] for r in g}
        if len(labels) > 1:
            n_conflict_groups += 1
            n_conflict_rows += len(g)
            if len(conflict_examples) < 5:
                conflict_examples.append({"text": g[0]["text"][:200],
                                          "labels": Counter(r["label"] for r in g)})
            continue
        n_exact_dup_rows += len(g) - 1
        r = min(g, key=lambda x: (x["source_split"], x["id"]))
        r["key"] = key
        deduped.append(r)
    log.info("dedupe: unique=%d dup_rows_removed=%d conflict_groups=%d conflict_rows_dropped=%d",
             len(deduped), n_exact_dup_rows, n_conflict_groups, n_conflict_rows)

    # 5. tokenizer / template analysis
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    cfg = AutoConfig.from_pretrained(args.tokenizer)
    template_sha = sha256_text(tok.chat_template or "")
    lens_full, lens_prompt, lens_answer = [], [], []
    not_prefix = 0
    by_class_chars = defaultdict(list)
    for r in deduped:
        msgs = build_messages(r["text"], r["label"])
        p_ids = tok.apply_chat_template(msgs[:1], tokenize=True, add_generation_prompt=True)
        f_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        if hasattr(p_ids, "keys"):  # transformers 5 may return a BatchEncoding
            p_ids, f_ids = p_ids["input_ids"], f_ids["input_ids"]
        if list(f_ids[:len(p_ids)]) != list(p_ids):
            not_prefix += 1
        r["n_tok_prompt"], r["n_tok_full"] = len(p_ids), len(f_ids)
        lens_full.append(len(f_ids))
        lens_prompt.append(len(p_ids))
        lens_answer.append(len(f_ids) - len(p_ids))
        by_class_chars[r["label"]].append(len(r["text"]))
    thresholds = [64, 128, 192, 256, 384, 512, 1024, 2048]
    trunc = {str(t): round(sum(x > t for x in lens_full) / max(1, len(lens_full)), 5) for t in thresholds}
    log.info("full-seq token lengths: %s", pct(lens_full))
    log.info("share of full sequences longer than T: %s", trunc)
    if not_prefix:
        log.warning("%d rows: prompt render is NOT a prefix of the full render", not_prefix)

    # 6. length filter
    too_long = [r for r in deduped if r["n_tok_full"] > args.max_seq_len]
    eligible = [r for r in deduped if r["n_tok_full"] <= args.max_seq_len]
    log.info("dropped %d rows with full length > %d tokens", len(too_long), args.max_seq_len)

    # 7. hash split
    pools = {"heldout": defaultdict(list), "val": defaultdict(list), "train": defaultdict(list)}
    for r in eligible:
        r["hash"], u = hash_u(args.salt, r["key"])
        name = ("heldout" if u < args.heldout_frac
                else "val" if u < args.heldout_frac + args.val_frac else "train")
        pools[name][r["label"]].append(r)
    want = {"train": args.train_per_class, "val": args.val_per_class, "heldout": args.heldout_per_class}
    splits = {}
    for name, per_class in want.items():
        sel = []
        for lid in LABELS:
            cand = sorted(pools[name][lid], key=lambda x: x["hash"])
            if len(cand) < per_class:
                raise SystemExit(f"pool {name}/{lid} has {len(cand)} < {per_class}")
            sel.extend(cand[:per_class])
        splits[name] = sorted(sel, key=lambda x: x["hash"])
    pool_sizes = {n: {LABELS[k]: len(v) for k, v in p.items()} for n, p in pools.items()}

    keysets = {n: {r["key"] for r in s} for n, s in splits.items()}
    overlap = {f"{a}&{b}": len(keysets[a] & keysets[b])
               for a, b in [("train", "val"), ("train", "heldout"), ("val", "heldout")]}
    assert not any(overlap.values()), overlap

    # 8. near-duplicates held-out vs train
    train_sh = [shingles(r["text"]) for r in splits["train"]]
    near = []
    for r in splits["heldout"]:
        s = shingles(r["text"])
        best = max((len(s & t) / len(s | t) for t in train_sh if s | t), default=0.0)
        if best >= 0.8:
            near.append({"heldout_hash": r["hash"], "jaccard": round(best, 3), "text": r["text"][:150]})
    log.info("held-out rows with a train near-duplicate (3-shingle Jaccard>=0.8): %d", len(near))

    # write
    out = REPO_ROOT / args.out_data
    (out / "processed").mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(parents=True, exist_ok=True)
    split_meta = {}
    for name, rs in splits.items():
        full = [{"hash": r["hash"], "id": r["id"], "source_split": r["source_split"],
                 "label": r["label"], "label_text": LABELS[r["label"]], "text": r["text"],
                 "n_tok_prompt": r["n_tok_prompt"], "n_tok_full": r["n_tok_full"],
                 "messages": build_messages(r["text"], r["label"])} for r in rs]
        write_jsonl(out / "processed" / f"{name}.jsonl", full)
        files = {f"{name}.jsonl": sha256_file(out / "processed" / f"{name}.jsonl")}
        if name != "heldout":  # Soup only ever sees train/val, messages-only rows
            write_jsonl(out / "processed" / f"{name}_soup.jsonl", [{"messages": x["messages"]} for x in full])
            files[f"{name}_soup.jsonl"] = sha256_file(out / "processed" / f"{name}_soup.jsonl")
        split_meta[name] = {
            "n": len(rs), "label_counts": dict(Counter(LABELS[r["label"]] for r in rs)),
            "source_split_counts": dict(Counter(r["source_split"] for r in rs)),
            "file_sha256": files, "hashes": [r["hash"] for r in rs],
            "tok_full": pct([r["n_tok_full"] for r in rs]),
        }
    write_json(out / "splits" / "split_hashes.json", {
        "dataset": args.dataset, "revision": args.revision, "salt": args.salt,
        "rule": "u=int(sha256(salt+'\\x00'+dedupe_key)[:15],16)/16**15; heldout u<hf; val hf<=u<hf+vf; "
                "train otherwise; per class take lowest hashes",
        "heldout_frac": args.heldout_frac, "val_frac": args.val_frac, "splits": split_meta})

    summary = {
        "dataset": args.dataset, "revision": args.revision, "source_counts": source_counts,
        "label_text_mismatch": label_text_mismatch, "pooled": len(rows),
        "changed_by_clean": changed_by_clean, "drops": dict(drops),
        "drop_examples": dict(drop_examples), "after_filters": len(kept),
        "dedupe": {"unique": len(deduped), "dup_rows_removed": n_exact_dup_rows,
                   "conflict_groups": n_conflict_groups, "conflict_rows_dropped": n_conflict_rows,
                   "conflict_examples": [{"text": c["text"], "labels": dict(c["labels"])} for c in conflict_examples]},
        "tokenizer": args.tokenizer, "chat_template_sha256": template_sha,
        "tokenizer_model_max_length": tok.model_max_length,
        "config_max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
        "eos_token": tok.eos_token, "pad_token": tok.pad_token,
        "tok_len_full": pct(lens_full), "tok_len_prompt": pct(lens_prompt),
        "tok_len_answer": pct(lens_answer), "share_longer_than": trunc,
        "prompt_not_prefix_of_full": not_prefix,
        "chars_by_class": {LABELS[k]: pct(v) for k, v in by_class_chars.items()},
        "max_seq_len": args.max_seq_len, "dropped_too_long": len(too_long),
        "pool_sizes": pool_sizes, "split_overlap": overlap,
        "near_duplicates_heldout_vs_train": near,
        "splits": {n: {k: v for k, v in m.items() if k != "hashes"} for n, m in split_meta.items()},
        "example_rendered_full": tok.apply_chat_template(
            build_messages(splits["train"][0]["text"], splits["train"][0]["label"]), tokenize=False),
    }
    write_summary(run_dir, summary, log)


if __name__ == "__main__":
    main()
