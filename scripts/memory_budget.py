"""Phase 2: analytic memory budget for LoRA training, merge and 4-bit export.

Everything is derived from the model's config.json (no weights loaded) and is a function of
batch size b, padded sequence length s, LoRA rank r, target modules, base precision
(fp16 or NF4), gradient checkpointing, LoRA dropout and attention kernel.

Units: bytes internally; tables print GiB (2**30) because torch.cuda.max_memory_* and
nvidia-smi MiB are binary units. (Soup's estimator prints GB = 1e9; we convert when comparing.)

What each term assumes (all assumptions are printed into the JSON so they can be argued):
  weights    : fp16 = 2 B/param. NF4 = 0.5 B + block-scale overhead for decoder linear layers
               (blocksize 64, fp32 absmax; double-quant shrinks absmax to 8-bit + 2nd-level fp32
               per 256 blocks); embeddings/norms stay 16-bit, or 32-bit if
               peft.prepare_model_for_kbit_training upcasts them (flag).
  lora       : r*(in+out) params per adapted module, kept in fp32 (peft autocast_adapter_dtype).
  grads      : fp32, one per trainable param.
  adam       : 2 fp32 states per trainable param (adamw_torch); 8-bit variants 2 B.
  activations: itemized per token per layer, see activation_breakdown(). Frozen base Linear
               saves nothing for backward (weight has no grad), LoRA_A saves its input,
               dropout saves a 1-byte mask + its output, SiLU*up saves 3 I-sized tensors,
               RMSNorm saves its fp32-upcast input, SDPA saves q, repeated k/v, out, lse.
               With gradient checkpointing: one fp16 hidden state per layer + one layer
               recomputed at a time.
  logits     : T*V elements. fp16 head output (2 B) + fp32 upcast in the loss (4 B) +
               fp32 log-softmax saved by cross-entropy (4 B) [= forward]; backward adds an
               fp32 grad (4 B) and its fp16 cast (2 B). Reported as fwd and peak.
               Checkpointing does NOT reduce it. Only ~14 answer tokens per row carry loss,
               but logits are computed for every position.
  context    : CUDA context + cuBLAS workspace, visible to nvidia-smi but not to
               torch.cuda.max_memory_allocated. A placeholder constant until measured.

Example:
  python scripts/memory_budget.py --model Qwen/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import setup_run, write_json, write_summary  # noqa: E402

GiB = 2 ** 30
FP16, FP32 = 2, 4
ALL_LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


# ----------------------------------------------------------------------------- shape
@dataclass
class Shape:
    hidden: int
    inter: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    tied: bool
    attn_bias: bool = True  # Qwen2: q/k/v have bias, o_proj does not

    @classmethod
    def from_config(cls, cfg: dict) -> "Shape":
        h, nh = cfg["hidden_size"], cfg["num_attention_heads"]
        return cls(hidden=h, inter=cfg["intermediate_size"], layers=cfg["num_hidden_layers"],
                   heads=nh, kv_heads=cfg.get("num_key_value_heads", nh),
                   head_dim=cfg.get("head_dim") or h // nh, vocab=cfg["vocab_size"],
                   tied=bool(cfg.get("tie_word_embeddings", False)))

    def module_dims(self, name: str) -> tuple[int, int]:
        h, i, kv = self.hidden, self.inter, self.kv_heads * self.head_dim
        return {"q_proj": (h, self.heads * self.head_dim), "k_proj": (h, kv), "v_proj": (h, kv),
                "o_proj": (self.heads * self.head_dim, h), "gate_proj": (h, i),
                "up_proj": (h, i), "down_proj": (i, h)}[name]

    def linear_params_per_layer(self) -> int:
        return sum(a * b for a, b in (self.module_dims(m) for m in ALL_LINEAR))

    def bias_params_per_layer(self) -> int:
        kv = self.kv_heads * self.head_dim
        return (self.heads * self.head_dim + 2 * kv) if self.attn_bias else 0

    def param_counts(self) -> dict:
        embed = self.vocab * self.hidden
        head = 0 if self.tied else self.vocab * self.hidden
        lin = self.layers * self.linear_params_per_layer()
        other = self.layers * (self.bias_params_per_layer() + 2 * self.hidden) + self.hidden
        return {"embed": embed, "lm_head_untied": head, "decoder_linear": lin,
                "norms_and_biases": other, "total": embed + head + lin + other}


# ----------------------------------------------------------------------------- run spec
@dataclass
class Run:
    batch: int = 8
    seq: int = 256
    rank: int = 16
    targets: tuple = ALL_LINEAR
    base: str = "fp16"            # fp16 | nf4
    double_quant: bool = True
    kbit_upcast_embed_fp32: bool = True   # prepare_model_for_kbit_training casts non-quantized to fp32
    grad_ckpt: bool = False
    lora_dropout: float = 0.05
    attn: str = "sdpa_mem_efficient"      # sdpa_mem_efficient | eager
    optimizer: str = "adamw_torch"        # adamw_torch | adamw_8bit
    context_bytes: int = int(0.45 * GiB)  # PLACEHOLDER until measured on the T4

    @property
    def tokens(self) -> int:
        return self.batch * self.seq


def nf4_bytes_per_param(double_quant: bool, block: int = 64, block2: int = 256) -> float:
    if double_quant:  # 8-bit absmax per block + fp32 absmax per 256 blocks
        return 0.5 + 1 / block + 4 / (block * block2)
    return 0.5 + 4 / block


def weights_bytes(sh: Shape, run: Run) -> dict:
    pc = sh.param_counts()
    non_lin = pc["embed"] + pc["lm_head_untied"] + pc["norms_and_biases"]
    if run.base == "fp16":
        return {"decoder_linear": pc["decoder_linear"] * FP16, "non_linear": non_lin * FP16}
    bpp = nf4_bytes_per_param(run.double_quant)
    nl_bytes = FP32 if run.kbit_upcast_embed_fp32 else FP16
    return {"decoder_linear": math.ceil(pc["decoder_linear"] * bpp), "non_linear": non_lin * nl_bytes}


def lora_params(sh: Shape, run: Run) -> int:
    return sh.layers * sum(run.rank * sum(sh.module_dims(m)) for m in run.targets)


def activation_breakdown(sh: Shape, run: Run) -> dict:
    """Bytes saved for backward per token per decoder layer (no checkpointing)."""
    H, I = sh.hidden, sh.inter
    kv = sh.kv_heads * sh.head_dim
    q_dim = sh.heads * sh.head_dim
    t = set(run.targets)
    items = {"rmsnorm_x2_fp32_input": 2 * FP32 * H}
    # attention kernel
    if run.attn == "eager":
        # q, repeated k saved by QK^T; fp32 softmax output; fp16 probs + repeated v saved by PV
        items["attn_eager_qkv"] = FP16 * (q_dim + q_dim + q_dim) + FP16 * q_dim  # q,k_rep,v_rep,out
        items["attn_eager_scores_per_token"] = sh.heads * run.seq * (FP32 + FP16)
    else:
        items["attn_sdpa_q_krep_vrep_out"] = FP16 * 4 * q_dim
        items["attn_sdpa_lse"] = FP32 * sh.heads
    # MLP: gate out, up out, silu(gate) saved for the product
    items["mlp_gate_up_act"] = FP16 * 3 * I
    # LoRA inputs: with dropout each module has its own mask (1 B) + dropped copy (2 B);
    # without dropout lora_A saves the (shared) input once per distinct input tensor.
    lora = 0
    if run.lora_dropout > 0:
        for m in t:
            lora += 3 * sh.module_dims(m)[0]
    else:
        groups = {"attn_in": {"q_proj", "k_proj", "v_proj"}, "o_in": {"o_proj"},
                  "mlp_in": {"gate_proj", "up_proj"}, "down_in": {"down_proj"}}
        for g, mods in groups.items():
            if t & mods:
                lora += FP16 * sh.module_dims(next(iter(mods)))[0]
    items["lora_inputs"] = lora
    # without any adapter on down_proj, the MLP product (input of down) is not saved by anyone;
    # o_proj input likewise. That is already reflected: only lora_A saves them.
    return items


def activations_bytes(sh: Shape, run: Run) -> dict:
    per_tok_layer = sum(activation_breakdown(sh, run).values())
    full = run.tokens * sh.layers * per_tok_layer
    boundary = run.tokens * sh.hidden * FP16  # layer input kept by checkpointing
    ckpt = sh.layers * boundary + run.tokens * per_tok_layer
    return {"per_token_per_layer": per_tok_layer, "no_ckpt": full, "with_ckpt": ckpt,
            "used": ckpt if run.grad_ckpt else full}


def logits_bytes(sh: Shape, run: Run) -> dict:
    el = run.tokens * sh.vocab
    fwd = el * (FP16 + FP32 + FP32)          # head out fp16, .float() copy, log-softmax saved
    peak = fwd + el * (FP32 + FP16)           # + grad fp32, grad cast to fp16
    return {"elements": el, "fp16_logits": el * FP16, "fp32_upcast": el * FP32,
            "logsoftmax_saved_fp32": el * FP32, "grad_fp32": el * FP32, "grad_fp16": el * FP16,
            "forward": fwd, "peak": peak}


def train_budget(sh: Shape, run: Run) -> dict:
    w = weights_bytes(sh, run)
    p = lora_params(sh, run)
    opt_b = 8 if run.optimizer == "adamw_torch" else 2
    a = activations_bytes(sh, run)
    lg = logits_bytes(sh, run)
    terms = {
        "weights": sum(w.values()),
        "lora_params_fp32": p * FP32,
        "lora_grads_fp32": p * FP32,
        "adam_states": p * opt_b,
        "activations": a["used"],
        "logits_chain_peak": lg["peak"],
    }
    allocated = sum(terms.values())
    return {"run": asdict(run), "weights_detail": w, "trainable_params": p,
            "activations_detail": a, "activation_items_per_token_layer": activation_breakdown(sh, run),
            "logits_detail": lg, "terms": terms,
            "predicted_max_allocated": allocated,
            "predicted_nvidia_smi_used": allocated + run.context_bytes,
            "notes": ["Step 1 has no Adam states yet; compare step>=2 peaks.",
                      "max_reserved exceeds max_allocated by allocator fragmentation (not modelled).",
                      "TRL may keep logits for token-accuracy / entropy metrics: extra T*V transients."]}


def export_budget(sh: Shape, gguf_bpw: tuple[float, float] = (4.5, 5.0)) -> dict:
    """Merge into fp16 base, then 4-bit. GPU / CPU-RAM / disk peaks (bytes)."""
    pc = sh.param_counts()
    fp16_w = pc["total"] * FP16
    largest = max(a * b for a, b in (sh.module_dims(m) for m in ALL_LINEAR))
    merge_tmp = largest * FP32  # B@A delta for one module, fp32, before cast/add
    nf4_lin = pc["decoder_linear"] * nf4_bytes_per_param(True)
    nf4_total = nf4_lin + (pc["total"] - pc["decoder_linear"]) * FP16
    gguf = [pc["total"] * bpw / 8 for bpw in gguf_bpw]
    kv_per_token = 2 * sh.layers * sh.kv_heads * sh.head_dim * FP16
    return {
        "fp16_weights": fp16_w,
        "merge_gpu_peak": fp16_w + merge_tmp,
        "merge_cpu_ram_peak_low": fp16_w + merge_tmp,
        "merge_cpu_ram_peak_high": 2 * fp16_w,
        "merge_cpu_note": "low = load fp16 directly + in-place merge; high = an extra full copy "
                          "(fp32 load then cast, or state-dict materialized for saving). "
                          "If the base is accidentally loaded fp32: 2x fp16_weights before cast.",
        "disk_hf_cache_base": fp16_w,
        "disk_merged_fp16": fp16_w,
        "nf4_bnb_in_vram": nf4_total,
        "nf4_quantize_gpu_peak": nf4_total + largest * FP16,
        "gguf_q4_k_m_range": gguf,
        "gguf_conversion_f16_intermediate": fp16_w,
        "disk_peak_gguf_path": fp16_w * 3 + gguf[1],
        "disk_peak_gguf_note": "HF cache + merged fp16 + f16 GGUF intermediate + Q4_K_M output",
        "kv_cache_bytes_per_token_fp16": kv_per_token,
    }


def soup_estimate(model: str, run: Run) -> dict | None:
    """Soup's own pre-flight estimator (GB = 1e9), for a three-way comparison."""
    try:
        from soup_cli.utils.model_shape import read_model_shape
        from soup_cli.utils.vram_estimate import RunSpec, estimate_vram
    except Exception as e:  # noqa: BLE001
        return {"error": f"soup import failed: {e!r}"}
    try:
        shape = read_model_shape(model)
        spec = RunSpec(seq_len=run.seq, batch_size=run.batch,
                       quant="none" if run.base == "fp16" else "4bit",
                       peft="lora" if run.base == "fp16" else "qlora", optimizer=run.optimizer,
                       gradient_checkpointing=run.grad_ckpt, lora_r=run.rank,
                       target_modules=list(run.targets))
        est = estimate_vram(shape, spec)
        return {"terms_bytes": est.terms, "total_bytes": est.total_bytes,
                "total_GiB": round(est.total_bytes / GiB, 3), "notes": list(est.notes),
                "lora_targets_used": list(est.lora_targets), "shape_source": shape.shape_source}
    except Exception as e:  # noqa: BLE001
        return {"error": f"soup estimate failed: {e!r}"}


def gib(x) -> float:
    return round(x / GiB, 3)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--config-json", default=None, help="local config.json instead of the hub")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=256, help="padded length per micro-batch")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--targets", default="all-linear", help="all-linear | comma list, e.g. q_proj,v_proj")
    p.add_argument("--base", choices=["fp16", "nf4"], default="fp16")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--attn", choices=["sdpa_mem_efficient", "eager"], default="sdpa_mem_efficient")
    p.add_argument("--optimizer", choices=["adamw_torch", "adamw_8bit"], default="adamw_torch")
    p.add_argument("--context-gib", type=float, default=0.45, help="PLACEHOLDER until measured")
    p.add_argument("--vram-gib", type=float, default=15.0, help="usable T4 VRAM per nvidia-smi (~15 GiB)")
    p.add_argument("--no-soup", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--phase", default="phase2")
    p.add_argument("--runs-root", default=None)
    return p.parse_args()


def load_config(args) -> dict:
    if args.config_json:
        return json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    from huggingface_hub import hf_hub_download
    return json.loads(Path(hf_hub_download(args.model, "config.json")).read_text(encoding="utf-8"))


def main():
    args = parse_args()
    run_dir, log = setup_run(args.phase, "memory_budget", args, args.runs_root)
    cfg = load_config(args)
    sh = Shape.from_config(cfg)
    targets = ALL_LINEAR if args.targets == "all-linear" else tuple(args.targets.split(","))
    base_run = Run(batch=args.batch, seq=args.seq, rank=args.rank, targets=targets, base=args.base,
                   grad_ckpt=args.grad_ckpt, lora_dropout=args.lora_dropout, attn=args.attn,
                   optimizer=args.optimizer, context_bytes=int(args.context_gib * GiB))
    pc = sh.param_counts()
    log.info("shape=%s", asdict(sh))
    log.info("param counts=%s", pc)

    planned = train_budget(sh, base_run)
    export = export_budget(sh)
    soup = None if args.no_soup else soup_estimate(args.model, base_run)

    # grid: how the peak moves with b, s, r, checkpointing, base precision, kernel
    grid = []
    for b, s, r, gc, base, attn in itertools.product([1, 4, 8, 16], [256, 512], [8, 16, 64],
                                                      [False, True], ["fp16", "nf4"],
                                                      ["sdpa_mem_efficient", "eager"]):
        rr = Run(batch=b, seq=s, rank=r, targets=targets, base=base, grad_ckpt=gc,
                 lora_dropout=args.lora_dropout, attn=attn, optimizer=args.optimizer,
                 context_bytes=int(args.context_gib * GiB))
        tb = train_budget(sh, rr)
        grid.append({"batch": b, "seq": s, "rank": r, "grad_ckpt": gc, "base": base, "attn": attn,
                     **{k: gib(v) for k, v in tb["terms"].items()},
                     "max_allocated_GiB": gib(tb["predicted_max_allocated"]),
                     "nvidia_smi_GiB": gib(tb["predicted_nvidia_smi_used"]),
                     "fits": tb["predicted_nvidia_smi_used"] <= args.vram_gib * GiB})
    with open(run_dir / "grid.csv", "w", encoding="utf-8", newline="\n") as f:
        f.write(",".join(grid[0].keys()) + "\n")
        for g in grid:
            f.write(",".join(str(v) for v in g.values()) + "\n")

    write_json(run_dir / "estimates.json", {"config": cfg, "shape": asdict(sh), "param_counts": pc,
                                            "planned_train": planned, "export": export,
                                            "soup_estimate": soup})
    write_template(run_dir / "estimate_vs_measured.md", planned, export, soup, base_run)

    t = planned["terms"]
    log.info("PLANNED b=%d s=%d r=%d targets=%s base=%s ckpt=%s attn=%s",
             base_run.batch, base_run.seq, base_run.rank, ",".join(targets), base_run.base,
             base_run.grad_ckpt, base_run.attn)
    for k, v in t.items():
        log.info("  %-20s %8.3f GiB", k, gib(v))
    log.info("  %-20s %8.3f GiB", "max_allocated", gib(planned["predicted_max_allocated"]))
    log.info("  %-20s %8.3f GiB (context placeholder %.2f)", "nvidia-smi", gib(planned["predicted_nvidia_smi_used"]), args.context_gib)
    log.info("activations no-ckpt %.3f GiB vs ckpt %.3f GiB; logits peak %.3f GiB",
             gib(planned["activations_detail"]["no_ckpt"]), gib(planned["activations_detail"]["with_ckpt"]),
             gib(planned["logits_detail"]["peak"]))
    if soup and "total_bytes" in soup:
        log.info("soup estimate total %.3f GiB terms(GiB)=%s", soup["total_GiB"],
                 {k: gib(v) for k, v in soup["terms_bytes"].items()})
    else:
        log.info("soup estimate: %s", soup)

    write_summary(run_dir, {
        "param_total": pc["total"], "trainable_params": planned["trainable_params"],
        "trainable_pct": round(100 * planned["trainable_params"] / pc["total"], 3),
        "planned_terms_GiB": {k: gib(v) for k, v in t.items()},
        "planned_max_allocated_GiB": gib(planned["predicted_max_allocated"]),
        "planned_nvidia_smi_GiB": gib(planned["predicted_nvidia_smi_used"]),
        "export_GiB": {k: (gib(v) if isinstance(v, (int, float)) else [gib(x) for x in v] if isinstance(v, list) else v)
                       for k, v in export.items()},
        "soup_total_GiB": soup.get("total_GiB") if soup else None,
        "grid_rows": len(grid), "grid_fit_count": sum(g["fits"] for g in grid),
    }, log)


def write_template(path: Path, planned: dict, export: dict, soup: dict | None, run: Run):
    t = planned["terms"]
    st = (soup or {}).get("terms_bytes", {})
    soup_map = {"weights": st.get("weights"), "lora_params_fp32": st.get("adapters"),
                "lora_grads_fp32": st.get("gradients"), "adam_states": st.get("optimizer"),
                "activations": st.get("activations"), "logits_chain_peak": st.get("logits")}
    f = lambda x: "—" if x is None else f"{gib(x):.3f}"  # noqa: E731
    rows = [
        "# Estimate vs measured (GiB = 2^30 bytes)",
        "",
        f"Planned run: b={run.batch}, s={run.seq}, r={run.rank}, targets={','.join(run.targets)}, "
        f"base={run.base}, grad_ckpt={run.grad_ckpt}, dropout={run.lora_dropout}, attn={run.attn}, "
        f"optimizer={run.optimizer}.",
        "Measured values must be copied from a file in runs/ (give the path). Leave TBD otherwise.",
        "",
        "## Training (per GPU)",
        "",
        "| Term | Our estimate | Soup estimate | Measured | Source file | Gap explanation |",
        "|---|---|---|---|---|---|",
    ]
    for k, v in t.items():
        rows.append(f"| {k} | {f(v)} | {f(soup_map.get(k))} | TBD | | |")
    rows += [
        f"| dequant (Soup only) | — | {f(st.get('dequant'))} | n/a | | |",
        f"| **torch max_memory_allocated** | {f(planned['predicted_max_allocated'])} | "
        f"{f((soup or {}).get('total_bytes'))} (incl. its overhead) | TBD | | |",
        "| torch max_memory_reserved | ≥ allocated (fragmentation not modelled) | — | TBD | | |",
        f"| CUDA context (placeholder) | {f(run.context_bytes)} | {f(st.get('overhead'))} | TBD (nvidia-smi before first alloc) | | |",
        f"| **nvidia-smi peak memory.used** | {f(planned['predicted_nvidia_smi_used'])} | — | TBD | | |",
        "",
        "## Export",
        "",
        "| Step | Resource | Estimate | Measured | Source file | Gap explanation |",
        "|---|---|---|---|---|---|",
        f"| merge (GPU) | VRAM peak | {f(export['merge_gpu_peak'])} | TBD | | |",
        f"| merge (CPU) | RAM peak | {f(export['merge_cpu_ram_peak_low'])} – {f(export['merge_cpu_ram_peak_high'])} | TBD | | |",
        f"| merged fp16 | disk | {f(export['disk_merged_fp16'])} | TBD | | |",
        f"| NF4 (bnb) | VRAM resident | {f(export['nf4_bnb_in_vram'])} | TBD | | |",
        f"| GGUF Q4_K_M | disk | {f(export['gguf_q4_k_m_range'][0])} – {f(export['gguf_q4_k_m_range'][1])} | TBD | | |",
        f"| GGUF path | disk peak | {f(export['disk_peak_gguf_path'])} | TBD | | |",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
