"""Phase 6: is the NF4 'neutral collapse' a property of 4-bit quantization or of how we built the artifact?

Observed (runs/phase5/verify_20261112T121313Z): base NF4 neutral recall 0.571 -> 0.071; null NF4 0.083;
real NF4 0.690 -> 0.583, merged-vs-NF4 label agreement 0.921 (S3 FAIL). This script evaluates ONE source
model (fp16 dir) under several load-time quantizations, on the same held-out rows, greedy, and reports
accuracy, per-class recall, predicted-class share and label agreement with the fp16 reference:

  fp16            reference
  nf4_dq_fp16     = Soup merge_4bit settings (nf4, double-quant, fp16 compute), explicit fp16 non-quant params
  nf4_nodq_fp16   no double-quant
  nf4_dq_fp32c    fp32 compute dtype
  fp4_dq_fp16     FP4 instead of NF4
  int8            bitsandbytes LLM.int8 (8-bit, for scale)
  artifact        an already-saved quantized dir (--artifact), e.g. outputs/nf4/base

One model in GPU memory at a time. If on-the-fly nf4_dq_fp16 reproduces the artifact, the artifact path is
not the cause; if nf4 variants all collapse but int8 does not, it is 4-bit quantization of this model.

Example (Colab):
  python scripts/nf4_study.py --model outputs/base_fp16 --artifact outputs/nf4/base --tag base
  python scripts/nf4_study.py --model outputs/merged/real --artifact outputs/nf4/real --tag real
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_jsonl, setup_run, write_jsonl, write_summary  # noqa: E402
from eval_task import generate_rows, score  # noqa: E402

VARIANTS = ["fp16", "nf4_dq_fp16", "nf4_nodq_fp16", "nf4_dq_fp32c", "fp4_dq_fp16", "int8", "artifact"]


def load(variant, model_dir, artifact):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    if variant == "artifact":
        return AutoModelForCausalLM.from_pretrained(artifact, device_map="cuda").eval()
    if variant == "fp16":
        return AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16, device_map="cuda").eval()
    if variant == "int8":
        q = BitsAndBytesConfig(load_in_8bit=True)
    else:
        qtype = "fp4" if variant.startswith("fp4") else "nf4"
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=qtype,
                               bnb_4bit_use_double_quant="_dq_" in variant,
                               bnb_4bit_compute_dtype=torch.float32 if variant.endswith("fp32c") else torch.float16)
    return AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16, quantization_config=q,
                                                device_map="cuda").eval()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="fp16 source dir")
    p.add_argument("--artifact", default=None)
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--data", default="data/processed/heldout.jsonl")
    p.add_argument("--variants", nargs="*", default=VARIANTS)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--tag", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--phase", default="phase6")
    a = p.parse_args()
    run_dir, log = setup_run(a.phase, f"nf4_study_{a.tag}", a)
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = read_jsonl(a.data)
    res, ref = {}, None
    for v in a.variants:
        if v == "artifact" and not a.artifact:
            continue
        model = load(v, a.model, a.artifact)
        nonq = sorted({str(p_.dtype) for p_ in model.parameters() if p_.dtype != torch.uint8 and p_.dtype != torch.int8})
        preds, st = generate_rows(model, tok, rows, batch_size=a.batch_size, log=None)
        write_jsonl(run_dir / f"predictions_{v}.jsonl", preds)
        s = score(preds, 1000, a.seed)
        if v == "fp16":
            ref = preds
        agree = (sum(x.get("strict_label") == y.get("strict_label") for x, y in zip(preds, ref)) / len(preds)
                 if ref else None)
        res[v] = {"strict_acc": s["strict_acc"], "format_rate": s["format_rate"],
                  "per_class_recall": s["per_class_recall_strict"], "predicted_share": s["predicted_share"],
                  "label_agreement_with_fp16": agree, "non_quantized_param_dtypes": nonq,
                  "memory_footprint_bytes": model.get_memory_footprint(), "gen_tok_per_s": st["measured_gen_tok_per_s"]}
        log.info("%-14s acc=%.4f agree=%s recall=%s tok/s=%s nonq=%s", v, s["strict_acc"], agree,
                 s["per_class_recall_strict"], st["measured_gen_tok_per_s"], nonq)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write_summary(run_dir, {"model": a.model, "artifact": a.artifact, "results": res}, log)


if __name__ == "__main__":
    main()
