# DECISIONS

Every design choice, the alternatives, and why. "UNVERIFIED" = not yet confirmed by running it
or reading the source. Numbers here are only allowed if they come from a file in `runs/`.

---

## D-01 Model: Qwen2.5-1.5B-Instruct (fixed by engineer)
- Fits a T4 for LoRA in fp16 with headroom; the 0.5B sibling shares the tokenizer and chat
  template, so CPU smoke tests exercise the same template/tokenizer code path.
- Alternatives: 0.5B (too weak to show meaningful task change), 3B (tight on T4 for merge in
  fp16 on 12.7 GB CPU RAM — revisit in Phase 2 memory budget).

## D-02 Dataset: RuReviews via `ai-forever/ru-reviews-classification` (engineer picked "B")
- Candidates considered: A) `mteb/HeadlineClassification` (MIT, 6-topic headlines), B) RuReviews
  (Apache-2.0, 3-class sentiment), C) `bond005/NEREL_instruct` entity_recognition (only 603/82/80 rows,
  whole articles, fuzzy targets). RussianSuperGLUE TERRa/DaNetQA excluded: dataset viewer could not
  load it (likely a script-based dataset), UNVERIFIED.
- Source used: `ai-forever/ru-reviews-classification` rather than `mteb/RuReviewsClassification`
  because it carries `label_text` and `id` columns, so the id->label mapping is checkable per row
  instead of assumed. Pinned revision `0f67d914f396ce22917dc6463ec619799b3b08d2` (HF API, 2026-11-11).
  License: Apache-2.0 on the HF card and in the original repo (github.com/sismetanin/rureviews LICENSE).
- Known weakness (deliberately kept, it is the interesting part): labels are **automatic**, derived
  from user star ratings (1-2 neg, 3 neu, 4-5 pos per the original README; README literally says
  "3 and 5", which we read as a typo for "4 and 5" — UNVERIFIED). Expect label noise, especially on
  `neutral`, so the accuracy ceiling is well below 100% and the engineer must spot-check labels.
- Source also contains English reviews (e.g. machine-translated) and mojibake (U+FFFD, soft hyphens),
  seen in the first rows of the dataset viewer. Handled in D-05.

## D-03 Target behaviour: strict one-line JSON `{"тональность": "<негативный|нейтральный|позитивный>"}`
- Why: automatically scorable, and it separates two things a classifier label alone would merge:
  (1) obeying an exact output contract, (2) getting the sentiment right. We report both
  (`format_rate`, `acc_among_valid`) plus `strict_acc` (invalid = wrong) and `lenient_acc`
  (label stem found anywhere), so a "win" that is only formatting is visible as such.
- The instruction (with the allowed labels) is in the prompt for base AND trained model, so the
  baseline is a fair zero-shot baseline, not a straw man that was never told the format.
- Label id -> word: 0 негативный, 1 нейтральный, 2 позитивный. prepare_data.py re-checks every row's
  `label_text` against this mapping and logs mismatches.
- No system message: Qwen2.5's template injects its default system prompt. Both training (via Soup)
  and eval must render through the same template — whether Soup does so is UNVERIFIED until Phase 3.

## D-04 Split: our own hash-based split over all pooled source splits
- Source train/validation/test are pooled and re-split, because we dedupe across them (source splits
  are not assumed disjoint).
- `u = int(sha256(salt + "\0" + dedupe_key)[:15], 16) / 16**15`; held-out pool u<0.10, val pool
  0.10<=u<0.15, train otherwise; within each pool, per class, take the lowest hashes.
  -> held-out membership depends only on the text, not on row order, library version, or seed.
- Sizes: train 500/class = 1500, val 50/class = 150 (for eval-loss during training only),
  held-out 84/class = 252 (never used for any tuning decision: prompt, hyper-params, early stop).
- Balanced classes: accuracy == macro recall, and a constant predictor scores ~33%, a clear floor.
- Dedupe key = lowercased word tokens (punctuation/spacing-insensitive), so trivial variants land in
  the same split. Texts appearing with conflicting labels are dropped entirely.
- Hashes + file sha256 saved in `data/splits/split_hashes.json`.
- Alternative rejected: random split with a seed (depends on row order and library behaviour).

## D-05 Cleaning / filters
- NFC; remove soft hyphen / zero-width chars; collapse whitespace.
- Drop rows containing U+FFFD (irreversibly corrupted), shorter than 15 chars, or with Cyrillic
  share < 0.8 among letters (English/translit reviews are off-task for a Russian model eval).
- Drop rows whose full chat-templated sequence > max_seq_len (512): with right truncation the
  assistant answer — the only supervised span — would be cut off silently.

## D-06 Baseline decoding: pure greedy
- `do_sample=False`, `repetition_penalty=1.0`, max_new_tokens=32. Qwen2.5 ships a generation_config
  with sampling params and a repetition penalty; we override for determinism and log the shipped
  config. Consequence: a user running with shipped defaults gets different decoding (CONCERNS C-04).
- EOS detection: stop on any of generation_config.eos_token_id ∪ {<|im_end|>}; we record which.

## D-07 Local environment
- Local smoke tests: Python 3.12 venv, CPU torch, `soup-cli[train]==0.77.2`.
  soup-cli 0.77.2 requires Python >=3.11,<3.14 and transformers>=5.16.1 (PyPI metadata).

## D-08 torch is not pinned for Colab
- `requirements.txt` pins soup-cli and the stack pip resolved for it (transformers 5.22.2, peft 0.21.2,
  trl 0.29.1, accelerate 1.15.0, datasets 5.2.0, bitsandbytes 0.50.2) but only constrains `torch>=2.6`.
  Reason: Colab ships a CUDA torch; forcing a PyPI torch could pull a build without T4 (sm_75) kernels
  (UNVERIFIED for torch 2.15). The exact torch is recorded in `runs/phase0/*/pip_freeze_after.txt`.
  Trade-off: Colab-side torch differs from the local CPU one (2.15.0+cpu). Rule "pin everything" is
  bent here on purpose; `soup doctor` reports arch support, run it in Phase 0.

## D-09 Min length 15 chars (kept, but it is a real trade-off)
- Smoke run shows it drops legitimate short reviews ("Размер большой", "пошив не очень") along with
  junk ("рьтд"). 1,560 / 75,000 rows. Kept because short reviews are disproportionately the
  conflicting-label ones (see D-02); revisit if the engineer disagrees.

---
# Phase 2 — memory

## D-10 Proposed: LoRA on an fp16 base (`quantization: none`), NOT Soup's default QLoRA (`4bit`)
- Soup 0.77.2 defaults: `training.quantization: 4bit`, `lora.r: 64`, `lora.alpha: 16`,
  `lora.target_modules: auto` (= `q_proj, v_proj` for qwen2 per `soup_cli.utils.model_shape.lora_targets`),
  `gradient_checkpointing: false`, `batch_size: auto`, `optimizer: adamw_torch`, `lr: 2e-5`
  (read from `soup_cli/config/schema.py`).
- Why not QLoRA: the task merges the adapter into the **fp16** base. An adapter trained against
  NF4-dequantized weights is merged into weights it never saw, so "merged" != "what was trained" by
  the quantization error of every adapted layer. With fp16 LoRA, merged ≈ adapter-on up to fp16
  rounding, which is what Phase 5's "merged ≈ adapter-on" check assumes.
- Memory doesn't force QLoRA: estimated weights 2.875 GiB (fp16) vs 1.499 GiB (NF4 incl. fp32-upcast
  embeddings); the peak is dominated by logits/activations, not weights (see runs/phase2 grid).
- Alternative kept as a probe: (d) NF4 base, to measure the weights term Soup would actually use.
- Engineer to confirm in Phase 3.

## D-11 Proposed training shape: b=4, max_length 512, gradient checkpointing ON, r=16, all linear
- Batches pad to their longest row: with b=8 and full-length p95=259 tokens (runs/smoke/prepare_data),
  most batches reach >256, so s must be budgeted at the 512 cap, not the mean.
- Estimate (runs/smoke/memory_budget_*/grid.csv): b4/s512/ckpt ≈ 8.2 GiB allocated; b4/s512 without
  ckpt ≈ 14.9 GiB (too close to 15 GiB). Under checkpointing the **logits chain (4.6 GiB) is the
  biggest term**, bigger than the weights: vocab 151,936 × 2,048 tokens × ~14 B. Only ~14 answer
  tokens per row carry loss; the rest of the logits are pure overhead (fused CE would remove it;
  Soup has `use_cut_ce` / `use_liger`, not used, to keep the stack plain).
- r=16 all-linear = 18.46M trainable params (1.2%); q,v-only at r=16 would be 2.18M.

## D-12 Measurement design
- Own estimator (`scripts/memory_budget.py`), itemized per tensor, PLUS Soup's estimator
  (`soup_cli.utils.vram_estimate.estimate_vram`) as a second, independent column. They agree on
  weights/LoRA/Adam exactly, on activations within 5% (7.10 vs 7.46 GiB for b8/s256/no-ckpt), and
  disagree on logits (4.64 vs 3.48 GiB: we charge ~14 B/element, Soup ~12). Measurement decides.
- Controlled probe (`scripts/mem_probe.py`) instead of reading Soup's training peak: exact b×s every
  step, per-phase torch peaks (load / wrap / fwd / bwd / opt, steps 1–3). Soup's real run is ALSO
  wrapped (`scripts/memtrace.py -- soup train ...`) to see whether its peak matches the probe.
- Pre-registered falsification: probe (e) b8/s256/no-ckpt is predicted not to fit.
- Units: GiB (2^30) everywhere; Soup prints GB (1e9). 15 GiB ≈ 16.1 GB, i.e. the "16 GB" T4.
- Validated locally: parameter formula = 494,032,768 for Qwen2.5-0.5B, exactly the real count;
  LoRA r=8 all-linear formula = 4,399,104 = what peft reported (runs/smoke/mem_probe_*/log.txt).
  The 1.5B formula gives 1,543,714,304 params; ×2 B = 3,087,428,608 B vs model.safetensors
  3,087,467,144 B (difference = 38.5 KB header).

---
# Phase 3 — training

## D-13 Real-run hyper-parameters (configs/real.yaml) — chosen a priori, never tuned on held-out
- lr 2e-4, cosine, warmup 5%, 2 epochs, micro-batch 4 × grad-accum 4 (effective 16, ~188 optimizer
  steps), max_grad_norm 1.0, weight_decay 0 (decay would pull LoRA-B toward 0 and blur the null
  comparison), seed 42 / data_seed 42. LoRA r 16, alpha 32, dropout 0.05, all 7 linear projections.
  Soup's default lr 2e-5 and q/v-only targets are deliberately overridden; 2e-4 is the usual LoRA
  range. No sweep: any tuning would have to use val (150 rows), never held-out.
- Validation: Soup has no val-file option; it takes the LAST `val_split` fraction of the train file
  unshuffled (`soup_cli/data/loader.py:_split_val`). We pass train+val concatenated
  (`trainval_soup.jsonl`) with val_split 0.0909 -> int(1650 × 0.9091) = 1500, and prepare_data.py
  checks with Soup's own `_split_val` that the split is exactly our val rows (smoke: exact_match=True).
- **No held-out evaluation in Phase 3.** SHIP criteria must be written before any held-out number
  exists (Phase 5). Phase 3 only looks at train/val loss and telemetry.

## D-14 Null control: identical config with lr = 1e-7 (configs/null_lr1e-7.yaml)
- Requirement: a run that *finishes normally* (same data, steps, logging, saved adapter) but does not
  meaningfully change the model, so the Phase 5 check must tell it apart from the real run.
- lr = 0: rejected by Soup's schema (`training -> lr: Input should be greater than 0`, verified).
- Mismatched target_modules: does not finish — peft raises `NoMatchingPeftModuleError` (verified on
  0.5B). A null control that crashes tests nothing.
- lr = 1e-7 (2000× smaller than real): Adam's step size is ~lr per step regardless of gradient scale,
  so LoRA-B moves ~1e-7 × 188 ≈ 2e-5 per entry. That makes B **non-zero** — so "LoRA-B != 0" alone
  would wrongly call it trained (the point of this control) — while ΔW = (α/r)·B·A should be far
  below fp16 resolution of the base weights, i.e. the merge may round away entirely. Prediction
  written before running; Phase 5 measures it.
- Alternative not used: an adapter saved but never loaded / scale 0 at inference. That is a
  pipeline failure, tested in Phase 6 (fault injection), not a training control.

## D-15 Broken-template run: `data.chat_template: qwen2.5` (configs/broken_template.yaml)
- Found in Soup source (`soup_cli/data/chat_templates.py`): the registered name "qwen2.5" maps to plain
  ChatML. Qwen2.5's real template injects a default system prompt ("You are Qwen, created by Alibaba
  Cloud...") when none is given; Soup's does not. A user who writes `chat_template: qwen2.5` believes
  they selected the official template.
- Effect: training renders WITHOUT the system turn; eval/inference with the tokenizer's shipped template
  renders WITH it. Soup prints a yellow advisory that the override will be persisted into the saved
  tokenizer, but does not flag the mismatch. Smoke evidence: our inspector reports
  render_matches_eval_template 0/12 for this config vs 12/12 for real.yaml.
- `soup data doctor` cannot see it by construction: it takes `--model`, not `--config`, so it checks the
  tokenizer's own template (verified from `soup data doctor --help`).

## D-16 Loss mask, as Soup actually builds it (verified by running Soup's own formatter)
- `train_on_responses_only: true` on Qwen2.5 (no `{% generation %}` markers) trains on
  `<|im_start|>assistant\n` + JSON + `<|im_end|>\n` on 12/12 smoke rows. Data doctor reports this as
  MINOR ("may include role-prefix tokens"). `<|im_end|>` IS trained (good: the model learns to stop).
- Consequence for "a falling loss proves nothing": of ~18 trained tokens per row, the role prefix and
  the constant JSON skeleton `{"тональность": "` are fully predictable; only ~2–3 label tokens carry
  information. Mean token accuracy was already ≥0.94 at step 1 in the CPU smoke run. Most of the loss
  curve is the model getting certain about constant tokens. Phase 5 must score the label, not loss.
- Kept as Soup's default (harmless for generation, since the generation prompt supplies the prefix).

---
# Phase 4 — merge + quantize

## D-17 Merge: `soup merge --dtype float16` (primary) + independent plain-peft merge (reference)
- `soup merge` (read from soup_cli/commands/merge.py): loads the base in the requested dtype on **CPU**,
  `PeftModel.from_pretrained` + `merge_and_unload`, saves; the tokenizer is loaded **from the adapter
  dir** (so a train-time chat_template override is shipped with the merged model); base revision is
  taken from Soup's run provenance when recorded. generation_config comes along with the base model
  object (UNVERIFIED until we diff the files).
- `scripts/merge_ref.py`: same merge with plain peft, but tokenizer/template/generation_config copied
  from the BASE, per-phase torch/RSS/disk measurement, and a loud warning if the adapter dir carries a
  different template. verify.py diffs the two merged checkpoints tensor by tensor and file by file.
- Never merged into a 4-bit base.

## D-18 4-bit format: bitsandbytes NF4 (double-quant, fp16 compute), written by Soup's `merge_4bit`
- Chosen because: (1) it loads through the same transformers tokenizer/template/generation code our
  eval uses, so the base vs merged vs 4-bit comparison isolates the quantization effect only;
  (2) the noise floor (base vs base-NF4) is measured with exactly the same method;
  (3) it is Soup's own 4-bit save path (`soup merge --save-format 4bit` uses it); we call the
  function directly on the already-verified merged dir so we quantize that exact checkpoint once;
  (4) it runs on the T4 we have.
- Rejected: GGUF Q4_K_M via llama.cpp (`soup export --format gguf`). It is the more portable format
  for end users, but it is mixed precision (some tensors Q6_K, so not strictly "4-bit"), it uses
  llama.cpp's own tokenizer and the template baked into GGUF metadata (a second, separate silent-failure
  surface), and getting full-vocab logits for KL needs a llama.cpp build. Out of budget for 4–6 h.
- Rejected: AWQ/GPTQ — removed from Soup 0.77.0 (docs). torchao int4 — not on the T4 path we know.
- Consequence for the verdict: SHIP/DON'T SHIP applies to **the NF4 checkpoint served with
  transformers + bitsandbytes on CUDA**. It says nothing about a GGUF/Ollama build (CONCERNS C-21).
- `forced=False`: embeddings / tied lm_head stay fp16 (4bit_forced refuses tied embeddings anyway).

---
# Phase 5 — verification. PRE-REGISTERED before any held-out result exists.

## D-19 What verify.py compares
Artifacts per run R ∈ {real, null}: base fp16 · base+adapter_R (scale 0 / 0.5 / 1) · merged_R fp16 (Soup) ·
merged_ref_R fp16 (ours) · merged_R NF4. Plus base NF4 for the noise floor.
Same 252 held-out prompts, same template, greedy decoding (do_sample=False, repetition_penalty=1.0).
Weight level: per-layer/per-module ‖ΔW‖/‖W‖, fraction of elements that changed in the saved fp16,
LoRA-B norms, (W_merged − W_base) vs (α/r)·B·A consistency, untouched tensors bit-identical, Soup merge
vs reference merge. Function level: teacher-forced KL on the gold answer positions and on the label
token position; noise floor = KL(base‖base-NF4). Behaviour: strict_acc / format / EOS / length /
template leak per artifact, paired bootstrap (10,000 resamples, seed 0) on per-prompt correctness.

## D-20 "Meaningfully trained" — a run counts as trained only if ALL hold (merged fp16 vs base):
- T1 weights: ≥ 10% of elements of the adapted weight matrices differ from the base in the saved fp16
  merged checkpoint.
- T2 function: mean KL at the label-token position, KL(base‖merged), is > 3 × the noise floor
  KL(base‖base-NF4) at the same position.
- T3 behaviour: paired-bootstrap 95% CI of Δstrict_acc (merged − base) excludes 0 (either sign: this is
  about change, not quality).
- T0 sanity: adapter scale 0 reproduces base (label-token KL < 1e-3) and merged ≈ adapter-on
  (label-token KL(adapter-on‖merged) < the NF4 noise floor). If T0 fails, the check itself is broken
  and nothing below is interpreted.
- The check is only valid if it **separates**: real must pass T1–T3 and null must fail at least T1 and T3.
  If the null control also "passes", the check does not discriminate → DON'T SHIP (we can't prove anything).

## D-21 SHIP criteria for the shipped artifact = merged_real NF4 (all must hold)
- S1 Trained: real passes T0–T3 and null fails them as stated in D-20.
- S2 Better than base: strict_acc(merged NF4) − strict_acc(base fp16) ≥ +5 pp, paired-bootstrap 95% CI
  lower bound > 0.
- S3 Quantization didn't eat it: strict_acc(merged NF4) ≥ strict_acc(merged fp16) − 2 pp, and greedy
  prediction agreement merged fp16 vs merged NF4 ≥ 95%, and label-token KL(merged‖merged-NF4) ≤ 2 ×
  KL(base‖base-NF4).
- S4 Contract: format_rate ≥ 98%, stopped on EOS ≥ 99%, template_leak = 0 (merged NF4, greedy).
- S5 No collapse: every class predicted on ≥ 10% of rows, and no class recall drops > 10 pp vs base.
- S6 Artifact integrity: shipped chat template + tokenizer files identical (sha256) to the base's;
  generation_config present; the artifact reloads and reproduces its own greedy predictions.
- S7 User decoding: with the shipped generation_config (sampling, T 0.7, rep 1.1; seed 0) format_rate ≥ 95%.
- Any failure → DON'T SHIP, and the report states which criterion failed and what must change first.
- Thresholds are fixed now (commit timestamp is the evidence). If a threshold later looks wrong, the
  original verdict is still reported next to any revised one, with the reason.

### D-21 amendment (made BEFORE any held-out result; see commit time)
- S6 originally said "tokenizer files identical (sha256)". transformers 5 re-serialises tokenizer files on
  save (and moves the template to chat_template.jinja), so byte equality would fail for a correct artifact.
  S6 now requires: chat template string equal, rendered prompt equal, tokenizer functionally equivalent
  (same vocab, same special tokens, same ids on probe strings), generation_config present, and identical
  greedy outputs when the same in-memory artifact is run twice ("rerun determinism", not a disk reload).
  File sha256 equality is still reported, for information only.

## D-22 The "fp16 base" is a bf16 checkpoint cast to fp16 (found during the verify.py smoke test)
- Qwen2.5 safetensors are **bf16** (`torch_dtype: bfloat16` in config.json; confirmed by reading tensors).
  `soup merge --dtype float16` and our merge_ref load it as fp16, i.e. bf16 → fp16 cast first, then merge.
- My first verify.py compared merged fp16 tensors against the raw bf16 tensors, which would have counted
  ~every element as "changed" and let the null control pass T1. Caught on CPU smoke, fixed: the reference is
  base cast to the merged dtype. verify.py now also reports how many elements the bf16→fp16 cast changes
  and how many overflow to inf (fp16 max 65504) — a silent-failure candidate for Phase 6.
- Also found: safetensors `get_tensor` outside the `safe_open` context segfaults on Windows (mmap closed);
  verify.py clones inside the context.

---
# Phase 7 — analysis (after results; nothing above was edited)

## D-23 Verdict and how it was reached
- DON'T SHIP under D-20/D-21 as written: S1, S2 and S3 fail (analysis/verdict.md, runs/phase5/verify_20261112T121313Z).
- The pre-registered T2 reference (NF4 floor) turned out to be a large systematic shift (C-29/C-30). The original
  verdict is reported unchanged. The post-hoc alternative (null-control reference) would flip T2 to PASS, but T3,
  S2 and S3 still fail, so the verdict doesn't depend on that choice.
- Next step if continued: re-evaluate base vs NF4 on ≥1,500 of the 6,863 never-used held-out rows with thresholds
  registered first, and fix the 4-bit step until label agreement with fp16 is ≥95%.
