"""Phase 2: controlled memory probe — a few real LoRA training steps at EXACT (b, s, r) so the
measurement is comparable to memory_budget.py term by term.

Why not just read Soup's training peak: Soup pads each batch to its longest row, so the
effective s varies per step, and its run mixes eval/save. Here every batch is exactly b x s
tokens, labels mask all but the last --answer-tokens positions (like our task), and each
phase is measured separately:
  load_weights -> lora_wrap -> [step k: forward, backward, optimizer] for k=1..N
Step 1 has no Adam states yet (allocated lazily inside optimizer.step); step 2 is the
steady state the estimate describes.

Precision mirrors Soup on a T4 (UNVERIFIED for Soup, to be confirmed in Phase 3 logs):
fp16 frozen base, fp32 LoRA params, torch.autocast(fp16) + GradScaler. On CPU (smoke test)
everything is fp32 and no CUDA numbers exist.

Example (Colab):
  python scripts/mem_probe.py --batch 4 --seq 512 --rank 16 --grad-ckpt
  python scripts/mem_probe.py --batch 2 --seq 512 --rank 16            # no checkpointing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import setup_run, write_summary  # noqa: E402
from memtrace import MemTracker  # noqa: E402

ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--targets", default="all-linear")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--base", choices=["fp16", "nf4"], default="fp16")
    p.add_argument("--attn", default="sdpa", help="attn_implementation: sdpa | eager")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--answer-tokens", type=int, default=14)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--phase", default="phase2")
    p.add_argument("--runs-root", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    tag = f"b{args.batch}_s{args.seq}_r{args.rank}_{'ckpt' if args.grad_ckpt else 'nockpt'}_{args.base}_{args.attn}"
    run_dir, log = setup_run(args.phase, f"mem_probe_{tag}", args, args.runs_root)
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    cuda = torch.cuda.is_available()
    dev = "cuda" if cuda else "cpu"
    dtype = torch.float16 if cuda else torch.float32
    targets = ALL_LINEAR if args.targets == "all-linear" else args.targets.split(",")
    result = {"tag": tag, "cuda": cuda}

    with MemTracker(run_dir, log) as mt:
        with mt.phase("load_weights"):
            kw = {}
            if args.base == "nf4":
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype)
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map=dev,
                                                         attn_implementation=args.attn, **kw)
            model.config.use_cache = False
        with mt.phase("lora_wrap"):
            if args.base == "nf4":
                from peft import prepare_model_for_kbit_training
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.grad_ckpt)
            if args.grad_ckpt:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha,
                                                     lora_dropout=args.dropout, target_modules=targets,
                                                     task_type="CAUSAL_LM"))
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            dtypes = sorted({str(p.dtype) for p in model.parameters() if p.requires_grad})
            log.info("trainable params=%d dtypes=%s", trainable, dtypes)
            result.update(trainable_params=trainable, trainable_dtypes=dtypes)
        model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=cuda)
        g = torch.Generator().manual_seed(args.seed)
        vocab = model.config.vocab_size
        losses = []
        for step in range(1, args.steps + 1):
            ids = torch.randint(0, vocab, (args.batch, args.seq), generator=g).to(dev)
            labels = ids.clone()
            labels[:, :-args.answer_tokens] = -100
            with mt.phase(f"step{step}_forward", step=step):
                with torch.autocast(dev, dtype=torch.float16, enabled=cuda):
                    out = model(input_ids=ids, labels=labels)
                loss = out.loss
            with mt.phase(f"step{step}_backward", step=step):
                scaler.scale(loss).backward()
                del out
            with mt.phase(f"step{step}_optimizer", step=step):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            losses.append(float(loss))
            log.info("step %d loss=%.4f scale=%s", step, losses[-1], scaler.get_scale() if cuda else None)
        result["losses"] = losses
        result["phases"] = [{"phase": e["phase"], **{k: e["torch_after"].get(k) for k in
                             ("max_allocated", "max_reserved", "allocated")}, "rss_after": e["rss_after"]}
                            for e in mt.events if "torch_after" in e]
    write_summary(run_dir, result, log)


if __name__ == "__main__":
    main()
