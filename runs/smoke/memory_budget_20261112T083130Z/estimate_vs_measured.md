# Estimate vs measured (GiB = 2^30 bytes)

Planned run: b=8, s=256, r=16, targets=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj, base=fp16, grad_ckpt=False, dropout=0.05, attn=sdpa_mem_efficient, optimizer=adamw_torch.
Measured values must be copied from a file in runs/ (give the path). Leave TBD otherwise.

## Training (per GPU)

| Term | Our estimate | Soup estimate | Measured | Source file | Gap explanation |
|---|---|---|---|---|---|
| weights | 2.875 | 2.875 | TBD | | |
| lora_params_fp32 | 0.069 | 0.069 | TBD | | |
| lora_grads_fp32 | 0.069 | 0.069 | TBD | | |
| adam_states | 0.138 | 0.138 | TBD | | |
| activations | 7.098 | 7.457 | TBD | | |
| logits_chain_peak | 4.637 | 3.478 | TBD | | |
| dequant (Soup only) | — | 0.000 | n/a | | |
| **torch max_memory_allocated** | 14.886 | 15.255 (incl. its overhead) | TBD | | |
| torch max_memory_reserved | ≥ allocated (fragmentation not modelled) | — | TBD | | |
| CUDA context (placeholder) | 0.450 | 1.170 | TBD (nvidia-smi before first alloc) | | |
| **nvidia-smi peak memory.used** | 15.336 | — | TBD | | |

## Export

| Step | Resource | Estimate | Measured | Source file | Gap explanation |
|---|---|---|---|---|---|
| merge (GPU) | VRAM peak | 2.927 | TBD | | |
| merge (CPU) | RAM peak | 2.927 – 5.751 | TBD | | |
| merged fp16 | disk | 2.875 | TBD | | |
| NF4 (bnb) | VRAM resident | 1.064 | TBD | | |
| GGUF Q4_K_M | disk | 0.809 – 0.899 | TBD | | |
| GGUF path | disk peak | 9.525 | TBD | | |
