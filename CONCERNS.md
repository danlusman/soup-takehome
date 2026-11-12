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
