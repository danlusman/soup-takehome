# Memory: estimate vs measured (Phase 7)

All numbers are GiB (2^30 B) unless marked MiB. "Measured" values are copied from raw files; the path is
given next to each. Estimates are from `scripts/memory_budget.py`, using the same model config
(`runs/phase2/memory_budget_20261112T111057Z/estimates.json`). Nothing here was re-measured or adjusted.

## 0. Environment the numbers come from

| item | value | source |
|---|---|---|
| GPU | Tesla T4, sm_75, 15360 MiB (nvidia-smi) / 14.56 GiB usable (torch) | `runs/phase0/env_20261112T110028Z/nvidia-smi.txt`; OOM message in `runs/phase2/mem_probe_b8_s256_*_111606Z/log.txt` |
| torch | 2.14.1+cu130, arch list includes sm_75, bf16 native = False | `runs/phase0/torch_check.txt` |
| python | 3.13.16 (local smoke ran 3.12.10) | `runs/phase0/env_*/python_version.txt` |
| stack | soup-cli 0.77.2, transformers 5.22.2, peft 0.21.2, trl 0.29.1, bitsandbytes 0.50.2 | `runs/phase0/env_*/pip_freeze_after.txt` |
| torchao | 0.10.0 preinstalled, **uninstalled** before probes were re-run (it broke peft LoRA wrapping) | `runs/phase0/torchao_*.txt`; failed runs `runs/phase2/mem_probe_*_20261112T1111*` |
| frozen base dtype during Soup training | **torch.float16** (resolves C-17); trainable fp32; `fp16=True`, `bf16=False` | `runs/phase3/memtrace_train_real_*/telemetry.jsonl` (train_begin) |
| fp16 overflow | **0 skipped optimizer steps** in all 3 runs; GradScaler scale constant at 65536 for all 188 steps | same telemetry files, `step` events |

Note: `pip_freeze_after.txt` still lists torchao 0.10.0, because the freeze ran before the uninstall cell.
The uninstall is in `torchao_uninstall.txt`.

## 1. Training: controlled probes (exact b × s, step 2 = steady state)

Probe files: `runs/phase2/mem_probe_<shape>_20261112T1114..T1116*/phases.jsonl` (the second attempt; the first
attempt, `…T1111*`, crashed on torchao and is kept as-is).

| probe | est. forward peak | **meas. forward peak** | est. max_allocated | **meas. max_allocated (step-2 bwd)** | gap (meas − est) | meas. max_reserved | meas. nvidia-smi peak | Soup est. total |
|---|---|---|---|---|---|---|---|---|
| (a) b4 s512 ckpt fp16 | 6.397 | **6.226** | 8.205 | **7.334** | −0.871 | 7.715 | 8041 MiB (7.85) | 7.877 |
| (b) b2 s512 no-ckpt fp16 | 8.080 | **8.091** | 9.018 | **8.668** | −0.350 | 8.881 | 9235 MiB (9.02) | 9.514 |
| (c) b1 s512 eager fp16 | 6.072 | **6.012** | 6.576 | **6.301** | −0.275 | 6.389 | 6683 MiB | — |
| (d) b4 s512 ckpt NF4 base | 5.021 | **5.477** | 6.829 | **6.568** | −0.261 | 7.115 | 7427 MiB | — |
| (e) b8 s256 no-ckpt fp16 | 13.078 | **12.937** | 14.886 | **14.091**, then **OOM in step 3** | −0.795 | 14.258 | 14799 MiB | 15.255 |

Terms I can check individually (same files, `load_weights` / `lora_wrap` phases):

| term | estimate | measured | note |
|---|---|---|---|
| fp16 weights | 2.875 | 2.875 (allocated after load) | exact |
| LoRA params (r16, 7 modules, fp32) | 0.069 | 2.944 − 2.875 = 0.069 | exact; trainable params 18,464,768 = formula |
| Adam states | 0.138 | step1_optimizer − lora_wrap ≈ 3.247 − 2.944 − grads; allocated after step-1 optimizer 3.109 − 2.944 = 0.165 (states + grads) | consistent |
| NF4 weights + fp32-upcast embeddings + LoRA | 1.499 + 0.069 = 1.568 | 1.578 (allocated after lora_wrap, probe d) | within 0.01 |

### Why the gaps

1. **Forward peaks match within ±3% for every fp16 probe** (a, b, c, e). The activation model (per-tensor
   itemisation, LoRA-dropout masks, SDPA saving q/k/v/out) and the forward logits term (fp16 head output, fp32
   upcast, fp32 log-softmax = 10 B/element) are right to first order.
2. **Backward peaks are consistently lower than estimated, and the gap scales with T·V (tokens × vocab)**:
   0.871/0.290 = 3.0 B/el (a), 0.795/0.290 = 2.7 B/el (e), 0.350/0.145 = 2.4 B/el (b), 0.275/0.072 = 3.8 B/el (c).
   I charged 14 B per logit element at the backward peak (fwd 10 + fp32 grad 4 + fp16 grad 2), but these
   tensors are not all alive at the same time: the fp16 logits / fp32 copy are freed while the gradient is
   built. The measured chain is ≈ 11 B/el. Soup's estimator uses 11.6 B/el (`soup_cli/utils/vram_estimate.py`,
   `LOGITS_LOSS_BYTES_PER_ELEMENT`), which is closer than mine. Cause: I modelled the logits chain from first
   principles without checking tensor lifetimes.
3. **NF4 forward is 9% *higher* than estimated (5.477 vs 5.021).** My estimate has no dequantisation buffer and no
   k-bit logits overhead. Soup's estimator models both (`dequant`, `KBIT_LOGITS_EXTRA_BYTES`).
4. **CUDA context / non-PyTorch memory is much smaller than either estimate.** nvidia-smi minus torch reserved is
   about 105 MiB right after loading weights (3107 MiB vs 2.932 GiB reserved,
   `nvidia_smi_dumps/003_load_weights_end.txt` against the `load_weights` torch stats) and about 141 MiB at peak
   in (a) and (b). My placeholder was 0.45 GiB and Soup charges 0.82–1.17 GB. Likely cause, not verified: CUDA
   13 lazy module loading. Soup's total for (a), 7.877, lands close to the nvidia-smi peak (7.85) only because
   its overhead overestimate and its logits underestimate roughly cancel.
5. **The reserved-vs-allocated gap is small in the probes (0.2–0.5 GiB) because every step has the same shape.**

### Probe (e): did it run out of memory as predicted?

Pre-registered: "b8/s256/no-ckpt is predicted NOT to fit (~15.3 GiB incl. context)."
Measured (`runs/phase2/mem_probe_b8_s256_r16_nockpt_fp16_sdpa_20261112T111606Z/log.txt`):
- steps 1 and 2 **completed**, with peak allocated 13.934 and 14.091 GiB, under the 14.56 GiB the card exposes;
- step 3 backward **OOM**: "Tried to allocate 1.16 GiB … 173.81 MiB free … 12.92 GiB allocated by PyTorch,
  1.34 GiB reserved by PyTorch but unallocated".

**Right outcome, wrong reason.** The live tensors fit, and my number (14.89 + 0.45) was 0.8 GiB too high
(cause 2 above). It ran out of memory through allocator fragmentation, which my model doesn't include.
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` might have let it finish (not tested).

## 2. Training: the real Soup runs (variable-length batches)

Source: `runs/phase3/memtrace_train_<run>_*/summary.json`, `telemetry.jsonl` (`train_end`), `nvidia_smi_1s.csv`.

| run | torch max_allocated | torch max_reserved | reserved − allocated | nvidia-smi peak | tokens / runtime |
|---|---|---|---|---|---|
| real | 6.665 (7,156,196,352 B) | 13.840 | **7.18** | 14315 MiB | 514,058 tok / 734.6 s = 699.8 tok/s |
| null_lr1e-7 | 6.665 | 13.840 | 7.18 | 14315 MiB | 514,058 / 740.4 s = 694.3 tok/s |
| broken_template | 6.517 | 14.121 | 7.60 | 14413 MiB | 451,058 / 696.7 s = 647.4 tok/s |

- **The peak allocation is below probe (a)'s 7.334**, because Soup pads each batch to its own longest row
  (mostly under 512; full-length p95 = 259 tokens, `runs/phase1/prepare_data_*/summary.json`) and probe (a)
  is the worst case at s = 512.
- **The reserved memory is twice the allocated memory.** The caching allocator keeps a block for every
  different batch shape. nvidia-smi says the card is 93% full (14315 of 15360 MiB) while live tensors never
  exceeded 6.7 GiB.
  **Answer to "do you trust the reported VRAM?": only after asking which number it is.** nvidia-smi measures
  the allocator's cache, not what the model needs. A planner that reads nvidia-smi would conclude this run
  barely fits on a T4, when it actually needs about 7 GiB.
- **The token counts check the data independently.** 514,058 tokens ≈ 2 epochs × 1,500 rows × 170.0 mean
  tokens (prepare_data). The broken-template run processed exactly 63,000 fewer tokens, which is 3,000
  sequences × 21 tokens, the length of the default system prompt that Soup's `qwen2.5` alias drops.
- **Throughput:** Soup's panel shows "Speed: 0.26 it/s" (`train_real_stdout.txt`), i.e. optimizer steps per
  second. The tokens/s above use the trainer's own `num_tokens` (non-padding) and `train_runtime`, which
  includes the 10 eval passes. Only about 18 of each row's ~171 tokens carry loss
  (`runs/phase3/inspect_labels_real_*`), so supervised tokens/s is about 74. Raw throughput says little about
  how quickly the model learns.

## 3. Export: merge and 4-bit

Sources: `runs/phase4/*/summary.json`, `phases.jsonl`, `ram_1s.csv`, `nvidia_smi_1s.csv`, `df_after.txt`.

| step | resource | estimate | measured | gap explanation |
|---|---|---|---|---|
| merge on GPU (merge_ref, real) | VRAM max_allocated | 2.927 | **3.054** (`merge_ref_real_20261112T120723Z`) | +0.13: the fp32 adapter (0.069) sits on the GPU next to the weights during merge, plus fp32 B·A temporaries. The estimate counted only one module's delta. |
| merge on CPU (`soup merge`, 3 runs) | process-tree RSS peak | 2.93 – 5.75 | **5.69 – 6.65** (`memtrace_soup_merge_*`) | RSS also counts the memory-mapped safetensors pages and the ~0.6 GiB python/torch runtime. System "used" rose from about 1.7 to about 5.1 GiB, so the real extra memory is about 3.4 GiB ≈ 1.2× the fp16 weights. Either way it fits easily in Colab's 12.7 GB. |
| merged fp16 checkpoint | disk | 2.875 (3,087,428,608 B of weights) | **3,098,893,514 B** | +11.5 MB = tokenizer.json, chat template, configs |
| NF4 quantisation (Soup `merge_4bit`) | VRAM max_allocated | 1.090 | **1.099** (`quantize_*`, phase `merge_4bit_load_quantize_save`) | within 1% |
| NF4 reload check | VRAM max_allocated | 1.064 | **1.705** | **not explained.** The artifact is already quantised, so reload shouldn't need fp16 copies. Possibly a transient during transformers 5 loading. Open (C-28). |
| NF4 artifact | disk | 1.064 GiB (1,142,926,720 B; `estimates.json` → `export.nf4_bnb_in_vram`) | **1,154,754,527 B** (real) | +12 MB = tokenizer files plus quant-state metadata |
| whole export path, per run | disk | HF cache + merged + NF4 ≈ 6.8 GiB | 2.9 (base copy) + 2.9 (merged) + 1.1 (NF4) on top of the cache (`df_after.txt`) | matches; total disk used 35 of 113 GB with all 3 runs plus the reference merges |
