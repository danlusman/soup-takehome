# Verdict: DON'T SHIP

Rules applied: DECISIONS.md D-20 (trained-ness), D-21 (SHIP) and the D-21 amendment, all committed in
`d76ee50` / `064d807` before any held-out result existed. Thresholds were not changed.
Primary evidence: `runs/phase5/verify_20261112T121313Z/verify.json` and `report.md`, produced by
`scripts/verify.py` on the T4 (`runs/phase5/verify_stdout.txt`, exit 0). Shipped artifact under test:
`outputs/nf4/real` (bitsandbytes NF4, from Soup's `merge_4bit`). n = 252 held-out rows, greedy decoding.

## Step 0: is the check itself broken? (T0, checked first)

| check | real | null | broken template | evidence (verify.json → runs.<R>.kl) |
|---|---|---|---|---|
| adapter scale 0 reproduces base (decision-position KL < 1e-3) | PASS (0.0) | PASS (0.0) | PASS (0.0) | `base‖adapter@0` |
| merged ≈ adapter-on (KL < NF4 floor 0.266) | PASS (1.20e-5) | PASS (2.20e-5) | PASS (1.19e-5) | `adapter@1‖merged` |
| batching / numerical floor | 3.38e-6 | — | — | `numeric_floor_kl_batch_vs_single` |

Additional sanity checks, all consistent:
- Soup's merge equals our independent reference merge **bit for bit** in all three runs
  (`soup_vs_ref_min_equal_frac` = 1.0, `max_abs` = 0.0).
- All **142** non-adapted tensors are bit-identical to the base after the bf16→fp16 cast.
- The fraction of fp16 elements that changed matches the half-ulp prediction to 4 decimals
  (real 0.9361 vs 0.9359; null 0.0586 vs 0.0585).
- Greedy outputs from adapter-on and merged models agree on 100% of labels for real and broken
  (`paired.adapter_vs_merged.label_agreement` = 1.0) and on 99.6% for null.

**Conclusion: T0 holds, so the measuring instrument works and the results below can be read.**

## Step 1: trained-ness (D-20)

| test (threshold) | real | null_lr1e-7 | broken_template |
|---|---|---|---|
| T1 weights: ≥ 10% of adapted fp16 elements changed | **PASS** 0.9361 | FAIL 0.0586 | PASS 0.9417 |
| T2 decision-position KL(base‖merged) > 3 × KL(base‖base-NF4) = 0.798 | **FAIL** 0.3700 (1.39×) | FAIL 3.4e-5 | FAIL 0.4332 (1.63×) |
| T3 paired-bootstrap 95% CI of Δstrict_acc(merged − base) excludes 0 | **FAIL** +0.0357 [−0.0238, +0.0952] | FAIL 0.0 [0, 0] | FAIL +0.0397 [−0.0198, +0.1032] |
| **overall** | **NOT TRAINED** (by the pre-registered rule) | NOT TRAINED | NOT TRAINED |

Separation requirement: real must pass T1–T3 and null must fail T1 and T3. Null does fail T1 and T3, but real
fails T2 and T3, so **the pre-registered check does not certify the real run as trained.**

## Step 2: SHIP criteria (D-21) for `outputs/nf4/real`

| criterion | result | numbers | evidence |
|---|---|---|---|
| S1 trained + null separated | **FAIL** | follows from Step 1 | `verify.json → runs.*.trained` |
| S2 Δ strict_acc(NF4 − base fp16) ≥ +5 pp and CI lower bound > 0 | **FAIL** | +0.0397, CI [−0.0198, +0.0992]; 34 rows only NF4 got right, 24 only base | `runs.real.paired.nf4_vs_base` |
| S3 quantisation fidelity | **FAIL** | acc NF4 0.6984 ≥ merged 0.6944 − 0.02 ✔; label agreement merged vs NF4 **0.9206 < 0.95** ✘; decision KL(merged‖NF4) 0.0274 ≤ 2 × 0.266 ✔ | `runs.real.paired.nf4_vs_merged`, `runs.real.kl` |
| S4 contract: format ≥ 98%, EOS ≥ 99%, template leak = 0 | PASS | 1.000 / 1.000 / 0.000 | `runs.real.gen.nf4` |
| S5 every class predicted on ≥ 10% of rows; no class recall drops > 10 pp vs base | PASS | shares 0.353 / 0.341 / 0.306; recall neg +17.9 pp, neu +1.2 pp, pos **−7.1 pp** | `runs.real.gen.nf4`, `base_gen` |
| S6 artifact integrity | PASS | chat template equal, rendered prompt equal, tokenizer functionally equivalent, generation_config present, rerun determinism 1.0 | `runs.real.integrity_nf4`, `runs.real.gen.rerun_determinism_identical` |
| S7 shipped sampling (T 0.7, top_p 0.8, top_k 20, rep 1.1) format ≥ 95% | PASS | format 1.000, strict_acc 0.6944 | `runs.real.gen.shipped_sampling` |

**Verdict: DON'T SHIP.** S1, S2 and S3 fail.

## What must change first (in order)

1. **Show the gain is real, using more held-out data, before anything else.** The measured gain is
   +4.0 pp (NF4) and +3.6 pp (fp16 merged). With n = 252 the paired CI is about ±6 pp, so the result is
   compatible with zero. The held-out hash pool holds 7,115 rows, of which 6,863 have never been used
   (`runs/phase1/prepare_data_*/summary.json` → `pool_sizes.heldout`, minus the 252 evaluated). Re-evaluating base vs NF4 on ≥ 1,500 of them, with the threshold registered
   **before** the run, is the cheapest next step and needs no retraining. If the gain stays at +4 pp, S2 as
   written (≥ +5 pp) would still fail. Whether +4 pp is worth shipping is a product decision, not something
   this data can settle.
2. **Fix the 4-bit step before shipping a 4-bit model.** NF4 changes 7.9% of the trained model's labels
   (agreement 0.9206), moving predictions from *neutral* to *negative* (neutral recall 0.690 → 0.583,
   negative 0.619 → 0.726). Accuracy barely moves (0.694 → 0.698) only because those errors happen to
   cancel. Options to measure: NF4 without double-quant, keeping the most sensitive layers in fp16, or
   8-bit. Required: label agreement ≥ 95% with the fp16 merged model.
3. **Re-register T2 against a different reference for the next run** (see "Post-hoc observations").
   That doesn't change this verdict.

## Post-hoc observations (not used for the verdict; clearly labelled)

- **The NF4 "noise floor" is not noise.** Quantising the *base* model nearly deletes the neutral class: neutral
  recall goes 0.571 → **0.071**, negative 0.548 → 0.917, accuracy 0.659 → 0.627 (`base_nf4_gen`). Its
  decision-position KL from fp16 (0.266) is therefore a large systematic shift, not a small floor, and T2
  ("3 × the NF4 floor") compared training against a 4-bit failure. Measured against the **null control**
  (3.4e-5) or the numeric floor (3.4e-6), the real run's 0.370 is about 10⁴× larger. If I had pre-registered
  T2 against the null control it would PASS. T3, S2 and S3 would still FAIL, so the verdict would be the same.
- **The trained model survives quantisation much better than the base:** KL(merged‖NF4) at the decision
  position is 0.027 for real vs 0.266 for the base. Training sharpened the label distribution, so the NF4
  rounding flips fewer argmaxes.
- **The broken template is invisible in accuracy:** trained without the system prompt and evaluated with it,
  it reaches 0.698, the same as the real run. What catches it: our integrity check (`chat_template_equal`
  = False, `rendered_prompt_equal` = False, `runs.broken_template.integrity_nf4`), our label inspector
  (`runs/phase3/inspect_labels_broken_template_*`: render mismatch on every row), the 63,000-token deficit in
  telemetry, and merge_ref's warning (`runs/phase4/merge_ref_broken_template_*/log.txt`). Soup's data doctor,
  doctor, plan and dry-run all passed it (`runs/phase3/soup_checks_broken_template_*`), and `soup merge`
  shipped the overridden template without a warning.
- **A falling loss happened without learning:** the null control's eval loss fell monotonically from 0.7535
  to 0.7284 over 188 steps (`runs/phase3/memtrace_train_null_lr1e-7_*/telemetry.jsonl`), and Soup's summary
  printed "Loss: 0.7758 → 0.6929". Its greedy predictions are **identical** to the base (Δ = 0, CI [0, 0]).
- **Most of the real run's loss drop is formatting tokens:** the base already gets 100% format and mean token
  accuracy 0.925 at step 1; eval loss reaches 0.0353 by step 40 of 188 and then stays flat (0.0347 at 188).
- **The unmerged adapter is slower at inference:** about 120 tok/s with the PEFT adapter on vs about 205–210
  merged (`report.md` behaviour table). Same outputs.

## Caveats on this verdict

Read `report.md` → "What this check does NOT detect". The most material ones here: labels are star-derived
(neutral is the noisiest class, C-01/C-09); one seed; 252 rows; the verdict covers transformers +
bitsandbytes on CUDA only (C-21).
