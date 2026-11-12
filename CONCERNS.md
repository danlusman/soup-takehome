# CONCERNS (open doubts)

Status: OPEN unless marked otherwise. Each item says what would resolve it.

- **C-01 Label noise ceiling.** RuReviews labels come from star ratings, not human judgement of the
  text. Neutral (3 stars) is likely the noisiest class. A trained model may "learn" annotator/star
  quirks. Resolve: engineer hand-labels ~50 held-out rows blind, measure agreement with gold.
- **C-02 Formatting-only win.** The base model may already know sentiment; most of the gain may be
  JSON format adherence. Not a bug, but the report must not present format gains as "understanding".
  Resolve: report acc_among_valid and lenient_acc alongside strict_acc.
- **C-03 Soup's chat template vs ours.** We assume Soup renders `messages` with the tokenizer's own
  Qwen template and masks loss to the assistant span. UNVERIFIED. If Soup uses its own template or
  trains on prompt tokens, train/eval mismatch is silent. Resolve in Phase 3 from Soup source + by
  rendering one training batch.
- **C-04 Decoding mismatch with users.** We evaluate pure greedy, but the shipped
  generation_config (sampling + repetition penalty) is what users get by default. Resolve: also eval
  once with shipped defaults before the verdict.
- **C-05 Near-duplicates.** Dedupe is exact on a normalized key; near-duplicates across splits are
  only measured (3-shingle Jaccard>=0.8, held-out vs train), not removed.
- **C-06 Scope vs 4–6 h.** The PDF says 4–6 h of work; the 8-phase plan is larger. Risk of running out
  of time before the parts worth most (proof of training 30%, silent failures 20%).
- **C-07 Colab environment unknown.** Colab Python/torch/CUDA versions not verified; soup-cli needs
  transformers>=5.16.1, so Colab's preinstalled stack gets replaced. Resolve: notebook's first cell
  records `pip freeze`, `nvidia-smi`, python version.
- **C-08 English-language rows.** Cyrillic filter drops English reviews; some mixed-language rows may
  survive. Check the drop examples in the prepare_data summary.
- **C-09 Evidence for C-01 already in the smoke log.** Conflicting-label groups include
  "качество плохое" labelled neutral 6x vs negative 1x, and "Товар не пришел" 6 neutral / 7 negative /
  1 positive (runs/smoke/prepare_data_*/summary.json). 188 such groups were dropped, but the same noise
  very likely exists in non-duplicated rows (unmeasured). The neutral class is the weakest part of the gold labels.
- **C-10 Base model may be a constant predictor.** Smoke: Qwen2.5-0.5B answered "позитивный" for all
  9 rows with 100% format adherence. Only 9 rows and the 0.5B model, so NOT a result about 1.5B — but
  if 1.5B also has near-perfect format, the "format gain" story (C-02) disappears and any gain must
  come from accuracy. Look at the Colab baseline confusion matrix first.
- **C-11 CPU tokenizer pass is slow.** prepare_data tokenizes all ~71k rows (6 min locally). Fine, but
  on Colab's 2 vCPUs expect longer.
- **C-12 CUDA context size is a placeholder (0.45 GiB).** Soup budgets 1.17 GB "overhead". Measure from
  the first nvidia-smi dump vs torch reserved in Phase 2 probes.
- **C-13 Which SDPA kernel runs on the T4?** Flash-attention needs sm80+; we assume PyTorch picks the
  memory-efficient kernel (O(s) memory). If it falls back to the math kernel, activations gain an
  O(s²) term ~ probe (c) eager. Not verified; probe (a) vs (c) will show.
- **C-14 Probes use random token ids and no padding.** Real batches have padding + attention masks
  (may change the SDPA kernel) and real text. Memory should be identical for the same shapes, speed
  may not. The wrapped `soup train` run is the cross-check.
- **C-15 Soup's batch_size: auto probes for the max batch by trying until OOM.** If left on, the
  measured peak is "whatever fit", not our planned shape. Set batch_size explicitly in Phase 3.
- **C-16 TRL metrics may hold logits.** SFTTrainer logs mean_token_accuracy / entropy computed from
  logits — extra T×V transients not in either estimate. Check against measured peak.
- **C-17 Base dtype on T4.** CPU smoke: Soup loaded the frozen base in bf16 (checkpoint dtype). Soup's
  code says pre-Ampere cards get fp16 storage; check `frozen_dtypes` in runs/phase3/*/telemetry.jsonl.
  bf16 storage on a T4 would be emulated and slow, not wrong.
- **C-18 Grad clipping always active.** Smoke grad norms 21–36 vs max_grad_norm 1.0. On the real run,
  if clipping binds every step, the effective update is lr × unit-norm direction; worth stating.
- **C-19 fp16 overflow skips are silent in HF Trainer.** Captured via GradScaler scale + accelerate's
  `optimizer_step_was_skipped` per step (telemetry.jsonl). Not testable on CPU (no scaler).
- **C-20 Soup writes 2 full checkpoints (adapter + optimizer + rng) per run.** For 1.5B r16 expect
  ~74 MB adapter (fp32) + ~150 MB optimizer per checkpoint. Disk, not risk — but copy only what's needed.
- **C-21 Verdict scope.** SHIP/DON'T SHIP covers the NF4 checkpoint run with transformers+bitsandbytes on
  CUDA. A GGUF/llama.cpp build would need its own tokenizer/template/quality verification.
- **C-22 252 held-out rows ⇒ wide CIs.** ±~6 pp at 50–70% accuracy. A +5 pp gain can be real and still
  fail S2's CI condition. That is the intended conservative behaviour, stated in advance.
- **C-23 Thresholds are judgement calls** (10% elements, 3× noise floor, 5 pp, 2 pp, 95%, 98/99%).
  Chosen before results; defend them as "chosen in advance", not as "optimal".
- **C-24 Smoke-run observations about the CHECK (0.5B, 9 rows, 5-step adapter; not results).**
  runs/smoke/verify_20261112T093828Z/report.md. (a) T1/T2 separate real vs null cleanly (87% vs 0.14%
  elements changed; changed fraction matches the fp16 half-ulp prediction to 3 decimals; soup merge ==
  reference merge bit-for-bit). (b) T3 is the binding test at small n: real Δ=+0.33 but CI [0.0, 0.67]
  touches 0 → FAIL, so real is "NOT TRAINED" on 9 rows. Expected; at 252 rows it can still bite (C-22).
  (c) Null control: KL(adapter-on‖merged) = 0.0018 > KL(base‖adapter-on) = 0.0011 at the decision
  position — merge rounding noise is larger than the null adapter's whole effect, as predicted in D-14.
  (d) NF4 noise floor on 0.5B is large (decision-position KL 0.20) and base-NF4 changes greedy answers;
  the floor on 1.5B/T4 is unknown. (e) Shipped sampling (T 0.7, top_p 0.8, top_k 20, rep 1.1) dropped
  format_rate to 0.56 on the 5-step adapter — S7 may be the criterion that fails for real.
- **C-25 (T4, observed) Colab's preinstalled torchao 0.10.0 breaks peft LoRA wrapping on fp16.**
  `ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above
  0.16.0 are supported` in 4/5 first-attempt probes (runs/phase2/mem_probe_*_20261112T1111*); NF4 probe
  unaffected (different dispatch). Loud, not silent, but environment-only: the local venv has no torchao.
  `soup doctor` reported all checks OK with torchao 0.10.0 present. Fixed by uninstalling torchao
  (runs/phase0/torchao_*.txt); notebook Phase 0 now does it.
- **C-26 (T4, observed) Eager attention in fp16 → NaN loss, every step skipped, no error.**
  Probe (c) b1/s512 eager: loss=nan at steps 1–3, GradScaler scale 65536→32768→16384→8192
  (runs/phase2/mem_probe_b1_s512_r16_nockpt_fp16_eager_*). SDPA path is finite. A training run
  configured with eager attention on a T4 would "finish" having learned nothing. Random-token inputs,
  so magnitude may differ on real text; the failure mode is the point. Phase 6 candidate.
- **C-27 (T4, observed) Probe (e) OOM was fragmentation, not capacity.** b8/s256/no-ckpt ran steps 1–2
  (peak allocated 14.09 GiB, reserved 14.31 GiB of 14.56) and OOM'd in step 3 backward with 1.34 GiB
  reserved-but-unallocated (runs/phase2/mem_probe_b8_s256_*). "Predicted not to fit" is half right:
  the live tensors fit; the allocator did not. expandable_segments might make it pass.
