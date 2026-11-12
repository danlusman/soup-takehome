"""Phase 5: does the shipped artifact contain a trained model, and is it good enough to ship?

Compares, on the same held-out prompts and the same template:
  base fp16 · base+adapter_R at scale 0 / 0.5 / 1 (PEFT) · merged_R fp16 (soup merge) ·
  merged_ref_R fp16 (plain peft, weights only) · merged_R NF4 · base NF4 (noise floor)
for every run R given (real run AND null control), and applies the pre-registered rules
(DECISIONS.md D-20 trained-ness, D-21 SHIP). Thresholds are copied from there, not tuned.

Layers:
  1. weights  (CPU, streamed from safetensors): per-module ||ΔW||/||W||, fraction of fp16 elements
     that actually changed in the merged file, the fraction PREDICTED to change from fp16 spacing
     (|ΔW| > half-ulp), (W_merged - W_base) vs (α/r)·B·A consistency, untouched tensors bit-identical,
     soup merge vs reference merge.
  2. function (teacher-forced on the gold answer): KL between artifacts at every answer position and
     at the decision position (first token where the three possible answers differ); noise floor =
     KL(base || base NF4); numeric floor = base batched vs one-row-at-a-time.
  3. behaviour (greedy generation): strict_acc / format / EOS / length / template leak per artifact,
     paired bootstrap of per-prompt correctness, prediction agreement; S6 reload determinism and
     S7 sampled decoding with the shipped generation_config on the shipped NF4 artifact.

Outputs in runs/phase5/verify_<ts>/: verify.json, report.md, predictions_<artifact>.jsonl.

Example (Colab):
  python scripts/verify.py --runs real null_lr1e-7 --ship-run real
CPU smoke (no NF4, no CUDA):
  python scripts/verify.py --base Qwen/Qwen2.5-0.5B-Instruct --data data_smoke/processed/heldout.jsonl \
      --adapter-dir outputs_smoke --merged-dir outputs_smoke/merged --merged-ref-dir outputs_smoke/merged_ref \
      --runs real null_lr1e-7 --ship-run real --no-nf4 --phase smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_jsonl, setup_run, sha256_file, write_json, write_jsonl, write_summary  # noqa: E402
from eval_task import generate_rows, score  # noqa: E402
from task_format import LABELS, build_messages, build_target  # noqa: E402

# ---- pre-registered thresholds (DECISIONS.md D-20 / D-21) ----
T1_MIN_FRAC_CHANGED = 0.10
T2_MIN_FLOOR_MULTIPLE = 3.0
T0_MAX_SCALE0_KL = 1e-3
S2_MIN_GAIN = 0.05
S3_MAX_DROP = 0.02
S3_MIN_AGREE = 0.95
S3_MAX_FLOOR_MULTIPLE = 2.0
S4_MIN_FORMAT, S4_MIN_EOS = 0.98, 0.99
S5_MIN_SHARE, S5_MAX_RECALL_DROP = 0.10, 0.10
S7_MIN_FORMAT = 0.95

NOT_DETECTED = """\
## What this check does NOT detect
- **Wrong gold labels.** RuReviews labels come from star ratings (CONCERNS C-01/C-09). The check measures
  agreement with those labels, not whether the sentiment is actually right.
- **Generalisation.** Only 252 prompts, all from the same distribution and with our exact instruction and
  JSON contract. Nothing here tests other phrasings, other domains, multi-turn use, or inputs over 512 tokens.
- **Damage to other abilities.** We never test general chat, Russian fluency, or safety outside this task.
  A model that only outputs sentiment JSON would pass.
- **Other runtimes and decoders.** The verdict covers transformers + bitsandbytes NF4 on CUDA, greedy decoding,
  plus one sampled check (S7). GGUF/llama.cpp, vLLM and other tokenizer implementations are not covered (C-21).
- **Free-running divergence.** The KL numbers are teacher-forced on the gold answer, so they miss errors that
  compound once the model generates its own tokens. Only the generation metrics see those, and only at the
  label level.
- **Small effects.** With n=252, a CI half-width of about ±6 pp means a real gain of a few points can look like zero.
- **Seed variance.** One training seed and one data order, so we have no estimate of run-to-run variance.
- **Shortcuts.** Accuracy cannot tell real understanding apart from shortcuts such as length or keywords.
  Per-class recall and the confusion matrix are the only partial guard.
- **Weight-level tests show change, not usefulness.** The null control's B matrices are non-zero, which is
  why the weight test is not trusted on its own.
"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--data", default="data/processed/heldout.jsonl")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--runs", nargs="+", default=["real", "null_lr1e-7"])
    p.add_argument("--ship-run", default="real")
    p.add_argument("--null-runs", nargs="*", default=None, help="default: every run except --ship-run")
    p.add_argument("--adapter-dir", default="outputs")
    p.add_argument("--merged-dir", default="outputs/merged")
    p.add_argument("--merged-ref-dir", default="outputs/merged_ref")
    p.add_argument("--nf4-dir", default="outputs/nf4")
    p.add_argument("--base-nf4", default="outputs/nf4/base")
    p.add_argument("--no-nf4", action="store_true", help="skip NF4 artifacts")
    p.add_argument("--allow-cpu-nf4", action="store_true",
                   help="run NF4 artifacts on CPU (bitsandbytes>=0.50 CPU backend; smoke tests only)")
    p.add_argument("--cpu-dtype", choices=["fp32", "bf16"], default="fp32",
                   help="model dtype when no CUDA (bf16 halves RAM for local smoke tests)")
    p.add_argument("--tf-batch", type=int, default=4)
    p.add_argument("--gen-batch", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--phase", default="phase5")
    p.add_argument("--runs-root", default=None)
    return p.parse_args()


# ============================================================ 1. weights
def _index(d: Path) -> dict:
    from safetensors import safe_open
    idx = {}
    for f in sorted(Path(d).glob("*.safetensors")):
        with safe_open(str(f), "pt") as s:
            for k in s.keys():
                idx[k] = f
    return idx


def _get(idx: dict, k: str):
    from safetensors import safe_open
    with safe_open(str(idx[k]), "pt") as s:
        return s.get_tensor(k).clone()  # clone: the mmap-backed tensor is invalid after close (Windows segfault)


def _half_ulp_fp16(w):
    import torch
    a = w.float().abs().clamp_min(2.0 ** -14)  # ignore subnormals
    _, e = torch.frexp(a)                      # a = m * 2^e, m in [0.5, 1)
    return torch.ldexp(torch.ones_like(a), e - 11) / 2  # fp16: 10 mantissa bits -> ulp = 2^(e-11)


def weight_report(base_dir: Path, adapter: Path, merged: Path, merged_ref: Path | None, log) -> dict:
    import torch
    acfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    r, alpha = acfg["r"], acfg["lora_alpha"]
    scale = alpha / (r ** 0.5) if acfg.get("use_rslora") else alpha / r
    if acfg.get("rank_pattern") or acfg.get("alpha_pattern"):
        log.warning("rank/alpha patterns present: per-module scale not modelled")
    ad = _index(adapter)
    bi, mi = _index(base_dir), _index(merged)
    ri = _index(merged_ref) if merged_ref and merged_ref.exists() else None
    mods = {}
    for k in ad:
        if ".lora_A." in k:
            mk = k.replace("base_model.model.", "", 1).replace(".lora_A.weight", ".weight")
            mods[mk] = (k, k.replace(".lora_A.", ".lora_B."))
    per = []
    tot_el = tot_changed = tot_pred = 0
    zero_b = 0
    cast_tot = [0, 0]
    base_dtypes = set()
    for mk, (ka, kb) in sorted(mods.items()):
        A, B = _get(ad, ka).float(), _get(ad, kb).float()
        dW = scale * (B @ A)
        Wb_raw = _get(bi, mk)
        Wm = _get(mi, mk)
        # The checkpoint may be bf16 (Qwen2.5 is). The merge starts from the base CAST to the merged
        # dtype, so that cast is the reference; comparing against raw bf16 would mark ~every element
        # as "changed" even for a no-op adapter.
        Wb = Wb_raw.to(Wm.dtype)
        cast_inexact = int((Wb.float() != Wb_raw.float()).sum())
        cast_nonfinite = int((~torch.isfinite(Wb)).sum())
        diff = Wm.float() - Wb.float()
        changed = (Wm != Wb)
        pred = dW.abs() > _half_ulp_fp16(Wb)
        nb = float(Wb.float().norm())
        cast_tot[0] += cast_inexact
        cast_tot[1] += cast_nonfinite
        row = {"module": mk, "B_norm": float(B.norm()), "A_norm": float(A.norm()),
               "rel_dW_adapter": float(dW.norm()) / nb, "rel_diff_merged": float(diff.norm()) / nb,
               "frac_changed": float(changed.float().mean()),
               "frac_predicted_to_change": float(pred.float().mean()),
               "consistency_resid": float((diff - dW).norm()) / (float(dW.norm()) + 1e-30)}
        if ri is not None and mk in ri:
            Wr = _get(ri, mk)
            row["soup_vs_ref_equal_frac"] = float((Wm == Wr).float().mean())
            row["soup_vs_ref_max_abs"] = float((Wm.float() - Wr.float()).abs().max())
        zero_b += float(B.abs().max()) == 0.0
        tot_el += Wb.numel()
        tot_changed += int(changed.sum())
        tot_pred += int(pred.sum())
        per.append(row)
    adapted = set(mods)
    untouched_equal, untouched_diff, only_base, only_merged = 0, [], [], []
    for k in bi:
        if k in adapted:
            continue
        if k not in mi:
            only_base.append(k)
            continue
        tb_raw, tm = _get(bi, k), _get(mi, k)
        base_dtypes.add(str(tb_raw.dtype))
        tb = tb_raw.to(tm.dtype)
        cast_tot[0] += int((tb.float() != tb_raw.float()).sum())
        cast_tot[1] += int((~torch.isfinite(tb)).sum())
        if torch.equal(tb, tm):
            untouched_equal += 1
        else:
            untouched_diff.append(k)
    only_merged = [k for k in mi if k not in bi]
    merged_dtypes = sorted({str(_get(mi, k).dtype) for k in list(mi)[:50]})

    def agg(key, by):
        out = {}
        for row in per:
            g = by(row["module"])
            out.setdefault(g, []).append(row[key])
        return {g: round(sum(v) / len(v), 6) for g, v in out.items()}

    rep = {
        "lora_r": r, "lora_alpha": alpha, "scale": scale, "n_adapted_modules": len(per),
        "n_B_all_zero": int(zero_b), "B_norm_sum": sum(x["B_norm"] for x in per),
        "frac_changed_all_adapted": tot_changed / max(1, tot_el),
        "frac_predicted_to_change_all_adapted": tot_pred / max(1, tot_el),
        "rel_dW_adapter_median": sorted(x["rel_dW_adapter"] for x in per)[len(per) // 2] if per else None,
        "rel_diff_merged_median": sorted(x["rel_diff_merged"] for x in per)[len(per) // 2] if per else None,
        "consistency_resid_median": sorted(x["consistency_resid"] for x in per)[len(per) // 2] if per else None,
        "by_module_type_frac_changed": agg("frac_changed", lambda m: m.split(".")[-2]),
        "by_layer_rel_diff_merged": agg("rel_diff_merged", lambda m: m.split(".")[2]),
        "base_checkpoint_dtypes": sorted(base_dtypes),
        "base_to_merged_dtype_cast_inexact_elements": cast_tot[0],
        "base_to_merged_dtype_cast_nonfinite_elements": cast_tot[1],
        "untouched_tensors_bit_identical": untouched_equal, "untouched_tensors_differ": untouched_diff,
        "keys_only_in_base": only_base, "keys_only_in_merged": only_merged,
        "merged_dtypes_sample": merged_dtypes,
        "soup_vs_ref_min_equal_frac": min((x.get("soup_vs_ref_equal_frac", 1.0) for x in per), default=None)
        if ri is not None else None,
        "soup_vs_ref_max_abs": max((x.get("soup_vs_ref_max_abs", 0.0) for x in per), default=None)
        if ri is not None else None,
        "per_module": per,
    }
    log.info("weights: frac_changed=%.4f (predicted %.4f) rel_dW_med=%.2e consistency_med=%.3f zeroB=%d untouched_equal=%d differ=%d",
             rep["frac_changed_all_adapted"], rep["frac_predicted_to_change_all_adapted"],
             rep["rel_dW_adapter_median"] or 0, rep["consistency_resid_median"] or 0, zero_b,
             untouched_equal, len(untouched_diff))
    return rep


def file_integrity(base_dir: Path, art: Path) -> dict:
    out = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
                 "vocab.json", "merges.txt"):
        a, b = art / name, base_dir / name
        out[name] = {"in_artifact": a.exists(), "in_base": b.exists(),
                     "sha_equal": (a.exists() and b.exists() and sha256_file(a) == sha256_file(b))}
    try:
        from transformers import AutoTokenizer
        ta, tb = AutoTokenizer.from_pretrained(str(art)), AutoTokenizer.from_pretrained(str(base_dir))
        out["chat_template_equal"] = (ta.chat_template or "") == (tb.chat_template or "")
        probes = ["Отличное платье, рекомендую!", "размер не подошёл :(", '{"тональность": "нейтральный"}',
                  "<|im_start|>user\nтест<|im_end|>\n", "English and 123 mixed"]
        out["tokenizer_equivalent"] = (ta.get_vocab() == tb.get_vocab()
                                       and ta.all_special_tokens == tb.all_special_tokens
                                       and all(ta(x)["input_ids"] == tb(x)["input_ids"] for x in probes))
        probe = build_messages("Тестовый отзыв.")
        out["rendered_prompt_equal"] = (ta.apply_chat_template(probe, tokenize=False, add_generation_prompt=True)
                                        == tb.apply_chat_template(probe, tokenize=False, add_generation_prompt=True))
    except Exception as e:  # noqa: BLE001
        out["tokenizer_load_error"] = repr(e)
    return out


# ============================================================ 2. teacher-forced KL
def decision_offset(tok) -> int:
    seqs = [tok(build_target(l), add_special_tokens=False)["input_ids"] for l in LABELS]
    d = 0
    while all(len(s) > d for s in seqs) and len({s[d] for s in seqs}) == 1:
        d += 1
    return d


def tf_inputs(tok, rows):
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    items = []
    for r in rows:
        msgs = build_messages(r["text"], r["label"])
        p = tok.apply_chat_template(msgs[:1], tokenize=True, add_generation_prompt=True)
        f = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        p = list(p["input_ids"] if hasattr(p, "keys") else p)
        f = list(f["input_ids"] if hasattr(f, "keys") else f)
        end = max(i for i, t in enumerate(f) if t == im_end)  # the assistant's <|im_end|>
        # positions whose next-token prediction is an answer token (JSON ... <|im_end|>)
        items.append({"ids": f[:end + 1], "positions": list(range(len(p) - 1, end))})
    return items


class Scaler:
    """Scale LoRA adapter `name` by s (0 = base, 1 = trained) by editing module.scaling."""

    def __init__(self, model, name):
        self.mods = [m for m in model.modules() if hasattr(m, "scaling") and isinstance(m.scaling, dict)
                     and name in m.scaling]
        self.name = name
        self.orig = [m.scaling[name] for m in self.mods]

    def set(self, s):
        for m, o in zip(self.mods, self.orig):
            m.scaling[self.name] = o * s


def kl_batch(lp_p, lp_q):
    return (lp_p.exp() * (lp_p - lp_q)).sum(-1)  # [B, P]


def tf_logprobs(model, items, dev):
    import torch
    L = max(len(it["ids"]) for it in items)
    ids = torch.full((len(items), L), 0, dtype=torch.long)
    att = torch.zeros((len(items), L), dtype=torch.long)
    for i, it in enumerate(items):  # right padding: positions stay absolute
        ids[i, :len(it["ids"])] = torch.tensor(it["ids"])
        att[i, :len(it["ids"])] = 1
    P = max(len(it["positions"]) for it in items)
    pos = torch.zeros((len(items), P), dtype=torch.long)
    mask = torch.zeros((len(items), P), dtype=torch.bool)
    for i, it in enumerate(items):
        pos[i, :len(it["positions"])] = torch.tensor(it["positions"])
        mask[i, :len(it["positions"])] = True
    with torch.no_grad():
        logits = model(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits
    sel = logits.gather(1, pos.to(dev).unsqueeze(-1).expand(-1, -1, logits.shape[-1]))
    return torch.log_softmax(sel.float(), -1), mask.to(dev)


# ============================================================ 3. paired stats
def paired(a: list[dict], b: list[dict], n_boot: int, seed: int) -> dict:
    import numpy as np
    x = np.array([p["strict_correct"] for p in a], float)
    y = np.array([p["strict_correct"] for p in b], float)
    d = x - y
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    bs = d[idx].mean(1)
    agree = float(np.mean([pa.get("strict_label") == pb.get("strict_label") and pa["output"] == pb["output"]
                           for pa, pb in zip(a, b)]))
    label_agree = float(np.mean([pa.get("strict_label") == pb.get("strict_label") for pa, pb in zip(a, b)]))
    return {"delta_strict_acc": round(float(d.mean()), 4),
            "ci95": [round(float(np.quantile(bs, 0.025)), 4), round(float(np.quantile(bs, 0.975)), 4)],
            "p_delta_le_0": round(float((bs <= 0).mean()), 4),
            "a_only_correct": int(((x == 1) & (y == 0)).sum()), "b_only_correct": int(((x == 0) & (y == 1)).sum()),
            "identical_output_rate": round(agree, 4), "label_agreement": round(label_agree, 4)}


# ============================================================ main
def main():
    args = parse_args()
    run_dir, log = setup_run(args.phase, "verify", args, args.runs_root)
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    cuda = torch.cuda.is_available()
    use_nf4 = (cuda or args.allow_cpu_nf4) and not args.no_nf4
    dev = "cuda" if cuda else "cpu"
    dtype = torch.float16 if cuda else {"fp32": torch.float32, "bf16": torch.bfloat16}[args.cpu_dtype]
    nulls = args.null_runs if args.null_runs is not None else [r for r in args.runs if r != args.ship_run]
    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[:args.limit]
    log.info("rows=%d runs=%s ship=%s nulls=%s nf4=%s device=%s", len(rows), args.runs, args.ship_run,
             nulls, use_nf4, dev)
    base_dir = Path(snapshot_download(args.base, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]))
    paths = {R: {"adapter": Path(args.adapter_dir) / R, "merged": Path(args.merged_dir) / R,
                 "merged_ref": Path(args.merged_ref_dir) / R, "nf4": Path(args.nf4_dir) / R} for R in args.runs}
    res = {"thresholds": {k: v for k, v in globals().items() if k[:2] in ("T0", "T1", "T2", "S2", "S3", "S4", "S5", "S7")},
           "n_rows": len(rows), "runs": {}, "nf4_used": use_nf4, "device": dev}

    # ---------- 1. weights ----------
    for R in args.runs:
        log.info("== weights %s", R)
        res["runs"][R] = {"weights": weight_report(base_dir, paths[R]["adapter"], paths[R]["merged"],
                                                   paths[R]["merged_ref"], log)}
        res["runs"][R]["integrity_merged"] = file_integrity(base_dir, paths[R]["merged"])
        if use_nf4:
            res["runs"][R]["integrity_nf4"] = file_integrity(base_dir, paths[R]["nf4"])

    # ---------- 2. teacher-forced KL ----------
    tok = AutoTokenizer.from_pretrained(args.base)
    items = tf_inputs(tok, rows)
    d_off = decision_offset(tok)
    res["decision_offset_tokens"] = d_off
    log.info("decision position = answer token #%d (first token where the 3 labels differ)", d_off)
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype, device_map=dev).eval()
    peft = PeftModel.from_pretrained(base, str(paths[args.runs[0]]["adapter"]), adapter_name=args.runs[0])
    for R in args.runs[1:]:
        peft.load_adapter(str(paths[R]["adapter"]), adapter_name=R)
    peft.eval()
    base_nf4 = (AutoModelForCausalLM.from_pretrained(args.base_nf4, device_map=dev).eval() if use_nf4 else None)

    def base_lp(batch):
        with peft.disable_adapter():
            return tf_logprobs(peft, batch, dev)

    # numeric floor: base batched vs one row at a time (first 16 rows)
    num = []
    for i in range(0, min(16, len(items)), args.tf_batch):
        bt = items[i:i + args.tf_batch]
        lpb, mb = base_lp(bt)
        for j, it in enumerate(bt):
            lp1, m1 = base_lp([it])
            n = int(m1.sum())
            num.append(float(kl_batch(lpb[j:j + 1, :n], lp1[:, :n]).mean()))
    res["numeric_floor_kl_batch_vs_single"] = sum(num) / len(num) if num else None

    for R in args.runs:
        log.info("== teacher-forced KL %s", R)
        sc = Scaler(peft, R)
        merged = AutoModelForCausalLM.from_pretrained(str(paths[R]["merged"]), dtype=dtype, device_map=dev).eval()
        nf4 = AutoModelForCausalLM.from_pretrained(str(paths[R]["nf4"]), device_map=dev).eval() if use_nf4 else None
        acc = {}

        def add(name, kl, mask):
            a = acc.setdefault(name, {"all_sum": 0.0, "all_n": 0, "dec_sum": 0.0, "dec_n": 0})
            a["all_sum"] += float((kl * mask).sum())
            a["all_n"] += int(mask.sum())
            a["dec_sum"] += float(kl[:, d_off].sum())
            a["dec_n"] += kl.shape[0]

        for i in range(0, len(items), args.tf_batch):
            bt = items[i:i + args.tf_batch]
            lb, m = base_lp(bt)
            peft.set_adapter(R)
            lps = {}
            for s in (0.0, 0.5, 1.0):
                sc.set(s)
                lps[f"adapter@{s}"] = tf_logprobs(peft, bt, dev)[0]
            sc.set(1.0)
            lps["merged"] = tf_logprobs(merged, bt, dev)[0]
            if use_nf4:
                lps["nf4"] = tf_logprobs(nf4, bt, dev)[0]
                lps["base_nf4"] = tf_logprobs(base_nf4, bt, dev)[0]
            add("base||adapter@0", kl_batch(lb, lps["adapter@0.0"]), m)
            add("base||adapter@0.5", kl_batch(lb, lps["adapter@0.5"]), m)
            add("base||adapter@1", kl_batch(lb, lps["adapter@1.0"]), m)
            add("adapter@1||merged", kl_batch(lps["adapter@1.0"], lps["merged"]), m)
            add("base||merged", kl_batch(lb, lps["merged"]), m)
            if use_nf4:
                add("merged||nf4", kl_batch(lps["merged"], lps["nf4"]), m)
                add("base||nf4", kl_batch(lb, lps["nf4"]), m)
                add("base||base_nf4[floor]", kl_batch(lb, lps["base_nf4"]), m)
        res["runs"][R]["kl"] = {k: {"mean_all_answer_positions": v["all_sum"] / max(1, v["all_n"]),
                                    "mean_decision_position": v["dec_sum"] / max(1, v["dec_n"])}
                                for k, v in acc.items()}
        for k, v in res["runs"][R]["kl"].items():
            log.info("  KL %-24s all=%.3e decision=%.3e", k, v["mean_all_answer_positions"], v["mean_decision_position"])
        if not args.skip_generation:
            gen_run(res, R, peft, sc, merged, nf4, tok, rows, args, run_dir, log, paths)
        del merged, nf4
        if cuda:
            torch.cuda.empty_cache()

    # base + base_nf4 generations
    if not args.skip_generation:
        with peft.disable_adapter():
            preds, st = generate_rows(peft, tok, rows, batch_size=args.gen_batch, max_new_tokens=args.max_new_tokens, log=log)
        write_jsonl(run_dir / "predictions_base.jsonl", preds)
        res["base_gen"] = {**score(preds, 2000, args.seed), **st}
        res["_preds"] = {"base": preds}
        if use_nf4:
            p2, st2 = generate_rows(base_nf4, tok, rows, batch_size=args.gen_batch, max_new_tokens=args.max_new_tokens, log=log)
            write_jsonl(run_dir / "predictions_base_nf4.jsonl", p2)
            res["base_nf4_gen"] = {**score(p2, 2000, args.seed), **st2}
            res["_preds"]["base_nf4"] = p2
        for R in args.runs:
            P = res["runs"][R]["_preds"]
            res["runs"][R]["paired"] = {
                "merged_vs_base": paired(P["merged"], preds, args.bootstrap, args.seed),
                "adapter_vs_merged": paired(P["adapter@1"], P["merged"], args.bootstrap, args.seed),
            }
            if use_nf4:
                res["runs"][R]["paired"]["nf4_vs_base"] = paired(P["nf4"], preds, args.bootstrap, args.seed)
                res["runs"][R]["paired"]["nf4_vs_merged"] = paired(P["nf4"], P["merged"], args.bootstrap, args.seed)

    verdict(res, args, nulls, use_nf4, log)
    for R in args.runs:
        res["runs"][R].pop("_preds", None)
    res.pop("_preds", None)
    write_json(run_dir / "verify.json", res)
    (run_dir / "report.md").write_text(render(res, args, nulls, use_nf4), encoding="utf-8")
    log.info("report: %s", run_dir / "report.md")
    write_summary(run_dir, {"verdict": res["verdict"], "trained": {R: res["runs"][R]["trained"] for R in args.runs}}, log)


def gen_run(res, R, peft, sc, merged, nf4, tok, rows, args, run_dir, log, paths):
    from transformers import GenerationConfig
    P, G = {}, {}
    peft.set_adapter(R)
    sc.set(1.0)
    for name, model in (("adapter@1", peft), ("merged", merged), ("nf4", nf4)):
        if model is None:
            continue
        preds, st = generate_rows(model, tok, rows, batch_size=args.gen_batch, max_new_tokens=args.max_new_tokens, log=log)
        write_jsonl(run_dir / f"predictions_{R}_{name}.jsonl", preds)
        P[name], G[name] = preds, {**score(preds, 2000, args.seed), **st}
    if R == args.ship_run:
        ship = nf4 if nf4 is not None else merged
        sub = rows[:32]
        again, _ = generate_rows(ship, tok, sub, batch_size=args.gen_batch, max_new_tokens=args.max_new_tokens)
        first = (P["nf4"] if nf4 is not None else P["merged"])[:32]
        G["rerun_determinism_identical"] = sum(a["output"] == b["output"] for a, b in zip(again, first)) / len(sub)
        art = paths[R]["nf4"] if nf4 is not None else paths[R]["merged"]
        gc = GenerationConfig.from_pretrained(str(art)).to_dict()
        skw = {k: gc[k] for k in ("temperature", "top_p", "top_k", "repetition_penalty") if gc.get(k) is not None}
        sp, _ = generate_rows(ship, tok, rows, batch_size=args.gen_batch, max_new_tokens=args.max_new_tokens,
                              sample_kwargs=skw, seed=args.seed)
        write_jsonl(run_dir / f"predictions_{R}_shipped_sampling.jsonl", sp)
        G["shipped_sampling"] = {"kwargs": skw, **score(sp, 2000, args.seed)}
    res["runs"][R]["gen"] = G
    res["runs"][R]["_preds"] = P


def _ok(x):
    return "PASS" if x is True else "FAIL" if x is False else "NOT_RUN"


def verdict(res, args, nulls, use_nf4, log):
    for R in args.runs:
        rr = res["runs"][R]
        w, kl = rr["weights"], rr["kl"]
        floor = kl.get("base||base_nf4[floor]", {}).get("mean_decision_position")
        t0a = kl["base||adapter@0"]["mean_decision_position"] < T0_MAX_SCALE0_KL
        mvd = kl["adapter@1||merged"]["mean_decision_position"]
        t0b = (mvd < floor) if floor is not None else (mvd < T0_MAX_SCALE0_KL)
        t1 = w["frac_changed_all_adapted"] >= T1_MIN_FRAC_CHANGED
        t2 = (kl["base||merged"]["mean_decision_position"] > T2_MIN_FLOOR_MULTIPLE * floor) if floor is not None else None
        pr = rr.get("paired", {}).get("merged_vs_base")
        t3 = (pr["ci95"][0] > 0 or pr["ci95"][1] < 0) if pr else None
        rr["trained"] = {"T0_scale0_is_base": _ok(t0a), "T0_merged_eq_adapter_on": _ok(t0b),
                         "T1_weights_changed": _ok(t1), "T2_kl_above_floor": _ok(t2), "T3_behaviour_changed": _ok(t3)}
        rr["trained"]["overall"] = ("TRAINED" if all(v == "PASS" for v in rr["trained"].values())
                                    else "NOT TRAINED" if (t1 is False or t3 is False) else "INCONCLUSIVE")
        log.info("trained-ness %s: %s", R, rr["trained"])
    ship = res["runs"][args.ship_run]
    sep = (ship["trained"]["overall"] == "TRAINED"
           and all(res["runs"][n]["trained"]["T1_weights_changed"] == "FAIL"
                   and res["runs"][n]["trained"]["T3_behaviour_changed"] == "FAIL" for n in nulls))
    S = {"S1_trained_and_null_separated": _ok(sep)}
    g = ship.get("gen", {})
    base_g = res.get("base_gen")
    art = "nf4" if use_nf4 else "merged"
    if g and base_g:
        A = g.get(art)
        pv = ship["paired"].get(f"{art}_vs_base")
        S["S2_gain_vs_base"] = _ok(pv["delta_strict_acc"] >= S2_MIN_GAIN and pv["ci95"][0] > 0)
        if use_nf4:
            pq = ship["paired"]["nf4_vs_merged"]
            floor = ship["kl"]["base||base_nf4[floor]"]["mean_decision_position"]
            S["S3_quantization_fidelity"] = _ok(
                A["strict_acc"] >= g["merged"]["strict_acc"] - S3_MAX_DROP and pq["label_agreement"] >= S3_MIN_AGREE
                and ship["kl"]["merged||nf4"]["mean_decision_position"] <= S3_MAX_FLOOR_MULTIPLE * floor)
        else:
            S["S3_quantization_fidelity"] = "NOT_RUN"
        S["S4_contract"] = _ok(A["format_rate"] >= S4_MIN_FORMAT and A["eos_rate"] >= S4_MIN_EOS
                               and A["template_leak_rate"] == 0)
        S["S5_no_collapse"] = _ok(all(v >= S5_MIN_SHARE for v in A["predicted_share"].values())
                                  and all(A["per_class_recall_strict"][c] >= base_g["per_class_recall_strict"][c] - S5_MAX_RECALL_DROP
                                          for c in A["per_class_recall_strict"]))
        integ = ship.get("integrity_nf4" if use_nf4 else "integrity_merged", {})
        S["S6_integrity"] = _ok(bool(integ.get("chat_template_equal")) and bool(integ.get("rendered_prompt_equal"))
                                and bool(integ.get("tokenizer_equivalent"))
                                and integ.get("generation_config.json", {}).get("in_artifact", False)
                                and g.get("rerun_determinism_identical") == 1.0)
        S["S7_shipped_sampling_format"] = _ok(g["shipped_sampling"]["format_rate"] >= S7_MIN_FORMAT)
    else:
        for k in ("S2_gain_vs_base", "S3_quantization_fidelity", "S4_contract", "S5_no_collapse",
                  "S6_integrity", "S7_shipped_sampling_format"):
            S[k] = "NOT_RUN"
    res["ship_criteria"] = S
    res["shipped_artifact"] = art
    res["verdict"] = ("SHIP" if all(v == "PASS" for v in S.values())
                      else "DON'T SHIP" if any(v == "FAIL" for v in S.values()) else "INCOMPLETE")
    log.info("SHIP criteria: %s -> %s", S, res["verdict"])


def render(res, args, nulls, use_nf4) -> str:
    f = lambda x, n=4: "—" if x is None else (f"{x:.{n}e}" if isinstance(x, float) and (abs(x) < 1e-3 and x != 0) else f"{x:.{n}f}" if isinstance(x, float) else str(x))  # noqa: E731
    L = [f"# verify.py report — verdict: **{res['verdict']}** (artifact: {res['shipped_artifact']})", "",
         f"rows={res['n_rows']} · device={res['device']} · NF4 used={use_nf4} · decision position = answer token #{res['decision_offset_tokens']}",
         f"numeric floor KL(base batched‖base single-row) = {f(res['numeric_floor_kl_batch_vs_single'])}", "",
         "## Trained-ness (D-20)", "", "| check | " + " | ".join(args.runs) + " |", "|---|" + "---|" * len(args.runs)]
    for k in ("T0_scale0_is_base", "T0_merged_eq_adapter_on", "T1_weights_changed", "T2_kl_above_floor",
              "T3_behaviour_changed", "overall"):
        L.append(f"| {k} | " + " | ".join(res["runs"][R]["trained"][k] for R in args.runs) + " |")
    L += ["", "## Weights", "", "| metric | " + " | ".join(args.runs) + " |", "|---|" + "---|" * len(args.runs)]
    for k in ("B_norm_sum", "n_B_all_zero", "rel_dW_adapter_median", "frac_predicted_to_change_all_adapted",
              "frac_changed_all_adapted", "rel_diff_merged_median", "consistency_resid_median",
              "untouched_tensors_bit_identical", "soup_vs_ref_min_equal_frac", "soup_vs_ref_max_abs"):
        L.append(f"| {k} | " + " | ".join(f(res['runs'][R]['weights'].get(k)) for R in args.runs) + " |")
    L += ["", "## KL (teacher-forced, mean per position)", "",
          "| pair | " + " | ".join(f"{R} all / decision" for R in args.runs) + " |", "|---|" + "---|" * len(args.runs)]
    keys = list(res["runs"][args.runs[0]]["kl"])
    for k in keys:
        L.append(f"| {k} | " + " | ".join(
            f"{f(res['runs'][R]['kl'][k]['mean_all_answer_positions'])} / {f(res['runs'][R]['kl'][k]['mean_decision_position'])}"
            for R in args.runs) + " |")
    if res.get("base_gen"):
        L += ["", "## Behaviour (greedy, held-out)", "",
              "| artifact | strict_acc [95% CI] | format | lenient | acc|valid | EOS | leak | gen tok p50/p99 | tok/s |",
              "|---|---|---|---|---|---|---|---|---|"]

        def row(name, s):
            return (f"| {name} | {f(s['strict_acc'])} {s['strict_acc_ci95']} | {f(s['format_rate'])} | {f(s['lenient_acc'])} | "
                    f"{f(s['acc_among_valid'])} | {f(s['eos_rate'])} | {f(s['template_leak_rate'])} | "
                    f"{s['gen_tokens_dist'].get('p50')}/{s['gen_tokens_dist'].get('p99')} | {s.get('measured_gen_tok_per_s')} |")
        L.append(row("base fp16", res["base_gen"]))
        if res.get("base_nf4_gen"):
            L.append(row("base NF4", res["base_nf4_gen"]))
        for R in args.runs:
            for name, s in res["runs"][R].get("gen", {}).items():
                if isinstance(s, dict) and "strict_acc" in s and name != "shipped_sampling":
                    L.append(row(f"{R} {name}", s))
        L += ["", "### Paired bootstrap (Δ strict_acc = A − B)", "",
              "| run | comparison | Δ | 95% CI | A-only correct | B-only correct | label agreement |", "|---|---|---|---|---|---|---|"]
        for R in args.runs:
            for k, v in res["runs"][R].get("paired", {}).items():
                L.append(f"| {R} | {k} | {f(v['delta_strict_acc'])} | {v['ci95']} | {v['a_only_correct']} | "
                         f"{v['b_only_correct']} | {f(v['label_agreement'])} |")
        g = res["runs"][args.ship_run].get("gen", {})
        if "shipped_sampling" in g:
            ss = g["shipped_sampling"]
            L += ["", f"Shipped-sampling check ({ss['kwargs']}): format_rate={f(ss['format_rate'])}, strict_acc={f(ss['strict_acc'])}",
                  f"Rerun determinism (first 32 rows identical): {g.get('rerun_determinism_identical')}"]
    L += ["", "## SHIP criteria (D-21)", "", "| criterion | result |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in res["ship_criteria"].items()]
    L += ["", NOT_DETECTED]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
