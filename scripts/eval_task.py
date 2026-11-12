"""Phase 1 baseline (and reusable task eval): greedy generation on the held-out set.

Reports, per model:
  strict_acc      : output is exactly {"тональность": "<label>"} AND label correct (invalid = wrong)
  format_rate     : share of strictly valid outputs
  lenient_acc     : exactly one label stem anywhere in the output and it is correct
  acc_among_valid : accuracy restricted to strictly valid outputs
  eos_rate        : generation stopped on an EOS id before --max-new-tokens; which EOS token
  template_leak   : output contains chat-template control tokens (<|im_start|> etc.)
  gen length distribution, confusion matrix, per-class recall, bootstrap 95% CIs
  prompt_len_mismatch: eval-time prompt token count != prepare_data's count (template drift)
  measured throughput (generated tokens / generate() wall time) and peak CUDA memory

Decoding: do_sample=False, repetition_penalty=1.0 by default (pure greedy). Qwen2.5's
shipped generation_config sets sampling params + repetition_penalty; we override and log both.
`generate_rows` / `score` are imported by verify.py so every artifact is scored identically.

Example:
  python scripts/eval_task.py --model Qwen/Qwen2.5-1.5B-Instruct --data data/processed/heldout.jsonl --tag base
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import setup_run, sha256_text, read_jsonl, write_jsonl, write_summary  # noqa: E402
from task_format import LABELS, build_messages, parse_response  # noqa: E402

CONTROL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|im_sep|>")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None, help="optional PEFT adapter dir")
    p.add_argument("--tokenizer", default=None, help="defaults to --model")
    p.add_argument("--data", default="data/processed/heldout.jsonl")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--dtype", choices=["auto", "fp16", "fp32"], default="auto",
                   help="auto = fp16 on CUDA, fp32 on CPU")
    p.add_argument("--load-in-4bit", action="store_true", help="bitsandbytes NF4 (CUDA only)")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--tag", default="base")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--phase", default="phase1")
    p.add_argument("--runs-root", default=None)
    return p.parse_args()


def bootstrap_ci(xs: list[float], n: int, seed: int) -> list[float] | None:
    if not xs or n <= 0:
        return None
    rng = random.Random(seed)
    k = len(xs)
    means = sorted(sum(xs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return [round(means[int(0.025 * n)], 4), round(means[int(0.975 * n) - 1], 4)]


def pct(xs, qs=(0, 50, 90, 99, 100)):
    s = sorted(xs)
    return {f"p{q}": s[min(len(s) - 1, round(q / 100 * (len(s) - 1)))] for q in qs} if s else {}


def load_model(args, log):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    tok.padding_side = "left"
    kw = {}
    if args.load_in_4bit:
        if not cuda:
            raise SystemExit("--load-in-4bit needs CUDA (bitsandbytes)")
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    dtype = {"fp16": torch.float16, "fp32": torch.float32}.get(
        args.dtype, torch.float16 if cuda else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map="cuda" if cuda else "cpu", **kw)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        log.info("loaded adapter %s", args.adapter)
    model.eval()
    log.info("model=%s dtype=%s device=%s 4bit=%s", args.model, dtype, model.device, args.load_in_4bit)
    return model, tok, dtype


def eos_ids_for(model, tok) -> dict[int, str]:
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    eos.add(tok.convert_tokens_to_ids("<|im_end|>"))
    return {i: tok.convert_ids_to_tokens(i) for i in eos}


def generate_rows(model, tok, rows, *, batch_size=16, max_new_tokens=32, repetition_penalty=1.0,
                  sample_kwargs: dict | None = None, seed: int = 0, log=None) -> tuple[list[dict], dict]:
    """Generate for every row and parse. Greedy unless sample_kwargs is given."""
    import torch
    tok.padding_side = "left"
    id2tok = eos_ids_for(model, tok)
    eos_ids = set(id2tok)
    prompts = [tok.apply_chat_template(build_messages(r["text"]), tokenize=False,
                                       add_generation_prompt=True) for r in rows]
    dev = next(model.parameters()).device
    preds, gen_time, gen_tokens, prompt_mismatch = [], 0.0, 0, 0
    if sample_kwargs:
        torch.manual_seed(seed)
    for b in range(0, len(rows), batch_size):
        batch_rows, batch_prompts = rows[b:b + batch_size], prompts[b:b + batch_size]
        enc = tok(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        in_len = enc["input_ids"].shape[1]
        gk = dict(do_sample=True, **sample_kwargs) if sample_kwargs else dict(
            do_sample=False, repetition_penalty=repetition_penalty, temperature=None, top_p=None, top_k=None)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **gk)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_time += time.perf_counter() - t0
        for i, r in enumerate(batch_rows):
            n_prompt = int(enc["attention_mask"][i].sum())
            if r.get("n_tok_prompt") is not None and n_prompt != r["n_tok_prompt"]:
                prompt_mismatch += 1
            gen = out[i, in_len:].tolist()
            stop_pos = next((j for j, t in enumerate(gen) if t in eos_ids), None)
            body = gen if stop_pos is None else gen[:stop_pos]
            n_gen = len(body) + (0 if stop_pos is None else 1)
            gen_tokens += n_gen
            text = tok.decode(body, skip_special_tokens=False)
            parsed = parse_response(text)
            preds.append({
                "hash": r["hash"], "label": r["label"], "output": text,
                "n_gen_tokens": n_gen, "stopped_eos": stop_pos is not None,
                "stop_token": id2tok[gen[stop_pos]] if stop_pos is not None else None,
                "template_leak": any(c in text for c in CONTROL_TOKENS),
                **parsed,
                "strict_correct": parsed["strict_label"] == r["label"],
                "lenient_correct": parsed["lenient_label"] == r["label"],
            })
        if log:
            log.info("batch %d/%d done, elapsed generate %.1fs", b // batch_size + 1,
                     (len(rows) + batch_size - 1) // batch_size, gen_time)
    stats = {"generate_wall_s": round(gen_time, 2),
             "measured_gen_tok_per_s": round(gen_tokens / gen_time, 2) if gen_time else None,
             "prompt_len_mismatch_vs_prepare_data": prompt_mismatch,
             "example_prompt": prompts[0] if prompts else None}
    return preds, stats


def score(preds: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    n = len(preds)
    sc = [float(p["strict_correct"]) for p in preds]
    fv = [float(p["strict_valid"]) for p in preds]
    valid = [p for p in preds if p["strict_valid"]]
    conf = {LABELS[t]: dict(Counter(LABELS[p["strict_label"]] if p["strict_valid"] else "INVALID"
                                     for p in preds if p["label"] == t)) for t in LABELS}
    recall = {LABELS[t]: round(sum(p["strict_correct"] for p in preds if p["label"] == t)
                               / max(1, sum(p["label"] == t for p in preds)), 4) for t in LABELS}
    pred_share = {LABELS[t]: round(sum(p["strict_valid"] and p["strict_label"] == t for p in preds) / n, 4)
                  for t in LABELS}
    f1s = []
    for t in LABELS:
        tp = sum(p["strict_correct"] and p["label"] == t for p in preds)
        fp = sum(p["strict_valid"] and p["strict_label"] == t and p["label"] != t for p in preds)
        fn = sum(p["label"] == t and not p["strict_correct"] for p in preds)
        f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
    return {
        "n": n,
        "strict_acc": round(sum(sc) / n, 4), "strict_acc_ci95": bootstrap_ci(sc, n_boot, seed),
        "format_rate": round(sum(fv) / n, 4), "format_rate_ci95": bootstrap_ci(fv, n_boot, seed),
        "json_parse_rate": round(sum(p["json_parse_ok"] for p in preds) / n, 4),
        "lenient_acc": round(sum(p["lenient_correct"] for p in preds) / n, 4),
        "acc_among_valid": round(sum(p["strict_correct"] for p in valid) / len(valid), 4) if valid else None,
        "macro_f1_strict": round(sum(f1s) / len(f1s), 4),
        "per_class_recall_strict": recall, "predicted_share": pred_share, "confusion_strict": conf,
        "eos_rate": round(sum(p["stopped_eos"] for p in preds) / n, 4),
        "stop_token_counts": dict(Counter(str(p["stop_token"]) for p in preds)),
        "template_leak_rate": round(sum(p["template_leak"] for p in preds) / n, 4),
        "gen_tokens_dist": pct([p["n_gen_tokens"] for p in preds]),
        "sample_outputs": [p["output"] for p in preds[:5]],
    }


def main():
    args = parse_args()
    run_dir, log = setup_run(args.phase, f"eval_{args.tag}", args, args.runs_root)
    import torch

    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[:args.limit]
    log.info("eval rows=%d from %s", len(rows), args.data)
    model, tok, dtype = load_model(args, log)
    shipped_gen_cfg = model.generation_config.to_dict()
    log.info("shipped generation_config=%s", json.dumps(shipped_gen_cfg, default=str))
    log.info("eos ids used for stop detection: %s", eos_ids_for(model, tok))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    preds, stats = generate_rows(model, tok, rows, batch_size=args.batch_size,
                                 max_new_tokens=args.max_new_tokens,
                                 repetition_penalty=args.repetition_penalty, log=log)
    write_jsonl(run_dir / "predictions.jsonl", preds)
    summary = {
        "tag": args.tag, "model": args.model, "adapter": args.adapter, "dtype": str(dtype),
        "load_in_4bit": args.load_in_4bit, "data": args.data,
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens,
                     "repetition_penalty": args.repetition_penalty, "batch_size": args.batch_size},
        "shipped_generation_config": shipped_gen_cfg,
        "chat_template_sha256": sha256_text(tok.chat_template or ""),
        **score(preds, args.bootstrap, args.seed), **stats,
        "throughput_note": "generated (non-pad) tokens / generate() wall time, batch incl. padding waste",
        "cuda_max_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "cuda_max_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
    }
    log.info("strict_acc=%.4f format_rate=%.4f lenient_acc=%.4f eos_rate=%.4f",
             summary["strict_acc"], summary["format_rate"], summary["lenient_acc"], summary["eos_rate"])
    write_summary(run_dir, summary, log)


if __name__ == "__main__":
    main()
