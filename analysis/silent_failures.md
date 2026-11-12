# Silent failures: what breaks without an error, who catches it (Phase 6)

**Method.** Each fault was injected on purpose and run through Soup's built-in checks and ours.
- **Data, template and config faults:** CPU, Qwen2.5-0.5B smoke data. Run by `scripts/fault_inject.py`,
  results in `runs/phase6/fault_inject_*/results.json` (v1 `…T132518Z`, v2 `…T134020Z`). Each check's raw output is
  in its `runs/phase6/soup_checks_<fault>_*` folder.
- **Merge, quantisation and inference faults:** `scripts/fault_inject_artifacts.py`, results in
  `runs/phase6/fault_inject_artifacts_*`.
- **T4 rows** are failures observed in the real 1.5B runs (phases 2–5), not injected.
- **`soup ship` decision rule:** `scripts/ship_rule_test.py`, results in `runs/phase6/ship_rule_test_*`.

**What counts as flagged.** A Soup check flags a fault only if it exits non-zero, or if data doctor reports a non-OK
row that the clean config doesn't. Ours: the label inspector, the leak check, and verify.py (the latter on artifacts).

✅ = flagged, ❌ = passed silently, — = not applicable, n/t = not tested.

| # | stage | fault (what goes wrong silently) | data doctor | doctor / plan / dry-run | soup ship | ours | evidence |
|---|---|---|---|---|---|---|---|
| D1 | data | labels shuffled across rows: well-formed, wrong supervision | ❌ | ❌ ❌ ❌ | n/t (would see lower accuracy, if it is lower) | ❌ before training; only held-out eval can catch it | `soup_checks_D1_*` |
| D2 | data | 3 held-out rows leaked into train | ❌ | ❌ ❌ ❌ | ❌ by design (scores whatever task file it gets) | ✅ leak check: 3 rows match held-out hashes | `soup_checks_D2_*` |
| D3 | data | max_length 96: every answer truncated away | ✅ truncation_risk MAJOR | ❌ ❌ ❌ (dry-run: "Ready to train", exit 0) | — | ✅ zero trained tokens, EOS never trained | `soup_checks_D3_*` |
| D4 | data | assistant role renamed `bot`: nothing is supervised | ✅ unknown_roles MAJOR | ❌ ❌ ❌ (dry-run exit 0) | — | ✅ | `soup_checks_D4_*` |
| D5 | data | empty target `""` | ❌ | ❌ ❌ ❌ | — | v1 ❌ (**our miss**) → v2 ✅ (answer-contract check) | `soup_checks_D5_*` |
| D6 | template | override template never closes the assistant turn, so EOS is never trained | ❌ (only sees the tokenizer's template) | ❌ ❌ ❌ | — | ✅ `<|im_end|>` not trained; render ≠ eval | `soup_checks_D6_*` |
| D7 | template | system prompt only in training data, not at inference | ❌ | ❌ ❌ ❌ | — | v1 ❌ (**our miss**) → v2 ✅ (serving-parity check) | `soup_checks_D7_*` |
| T1 | template | `chat_template: qwen2.5` = Soup's alias for ChatML without the default system prompt; trains on a different prompt than inference | ❌ | ❌ ❌ ❌ | **❌ SHIP** (T4 accuracies, rule test) | ✅ render ≠ eval on every row; T4: −63,000 tokens, `chat_template_equal=False` | `soup_checks_T1_*`; `runs/phase3/*broken*`; `ship_rule_test_*` |
| T2 | template | role prefix `<|im_start|>assistant\n` in the loss (no `{% generation %}` markers) | MINOR | ❌ | — | ✅ measured: 13 of 16–18 trained tokens are fixed in every row | `runs/phase3/inspect_labels_real_*` |
| C1 | training | lr ≈ 0 (null control): the run "finishes", loss and eval loss fall, the model is unchanged | ❌ | ❌ ❌ ❌ | ✅ DON'T SHIP (tie is not > base) | ✅ verify T1/T3 + 10⁴× KL gap | `runs/phase5/verify_*`, `ship_rule_test_*` |
| C2 | training (T4) | eager attention in fp16: loss = NaN from step 1, every step skipped, no exception | — | n/t | n/t | ✅ only because the probe logs loss + GradScaler scale (65536 → 8192) | `runs/phase2/mem_probe_b1_s512_*_eager_*` |
| C3 | training (T4) | Colab's torchao 0.10.0 breaks peft LoRA wrapping (loud ImportError, but only on Colab) | — | `soup doctor` OK with it installed | — | caught by running the probes | `runs/phase2/mem_probe_*T1111*`, `runs/phase0/torchao_*` |
| M1 | merge | `lora_alpha` edited after training (32 → 16): merged at half strength | — | `soup merge` exit 0, no warning | n/t | ❌ verify.py T0 and consistency use the same edited file; ✅ only a cross-check against train-time telemetry | `fault_inject_artifacts_* → M1` |
| M2 | merge (T4) | `soup merge` takes the tokenizer and template from the adapter dir, so a training override ships with the model | — | `soup merge` exit 0, no warning | — | ✅ merge_ref warns; verify S6 `chat_template_equal=False` | `runs/phase4/merge_ref_broken_template_*/log.txt`, `verify.json → runs.broken_template.integrity_*` |
| Q1 | merge cast | bf16 weight > 65504 becomes `inf` after `--dtype float16` | — | `soup merge` exit 0, **no warning** (my harness first matched "overflow" in our own folder name; corrected) | n/t | ✅ verify counts non-finite elements after the cast (1) | `fault_inject_artifacts_* → Q2` |
| Q1b | merge cast (T4) | the real bf16→fp16 cast is inexact on 251,985 elements, none non-finite | — | — | — | ✅ measured | `verify.json → weights.base_to_merged_dtype_cast_*` |
| Q2 | 4-bit (T4) | NF4 changes 7.9% of the trained model's labels (neutral → negative) while accuracy stays the same; base NF4 neutral recall 0.571 → 0.071 | — | — | **❌ SHIP** (accuracy only) | ✅ verify S3 (label agreement 92.1%), per-class recall | `verify.json → runs.real.paired.nf4_vs_merged`, `base_nf4_gen` |
| Q3 | 4-bit (T4) | `merge_4bit` loads without a dtype, so non-quantised params inherit the source config's dtype: base NF4 has **bf16** layers on a T4 with no bf16 hardware, 132 vs ~195 tok/s | — | — | — | ✅ quantize.py reload check records dtypes | `runs/phase4/quantize_*/summary.json → reload_non4bit_param_dtypes`, verify behaviour table |
| I1 | inference | generation_config without `<|im_end|>` | — | — | n/t | ❌ our eval adds `<|im_end|>` to the stop set itself and would hide it; observed impact here is small (the model emits `<|endoftext|>` 2 tokens later) | `fault_inject_artifacts_* → I1` |
| I2 | inference | serving without the chat template (raw prompt): 0/3 valid outputs | — | — | ⚠ soup ship's renderer *silently falls back to raw prompt* if the template fails (`live_eval.py:_render_prompt_template`) | ✅ S6 rendered-prompt check (only for the template we ship) | `fault_inject_artifacts_* → I2` |
| S1 | release gate | `soup ship` decision = `tuned > base` on a point estimate, no CI: **+1 correct row of 252 ⇒ SHIP** | — | — | **❌ the rule itself** | ✅ verify S2 requires a CI lower bound > 0 | `ship_rule_test_* → synthetic` |
| S2 | release gate | `soup ship --adapter` scores base + adapter in **bf16** (on CUDA) or fp32, never the merged or 4-bit artifact; `--tuned <dir>` can load one only if the user passes it | — | — | **does not see the shipped artifact by default** | ✅ verify scores the NF4 file itself | `soup_cli/commands/ship.py` (`_load_generators`, comment "bf16 on CUDA"), `utils/live_eval.py` |

## Does `soup ship` ever see the merged 4-bit artifact?
**Only if you pass it explicitly with `--tuned`.** The documented usage, `soup ship --base <m> --adapter <lora>`,
loads the base plus the adapter on the fly, in bf16 on CUDA (emulated on a T4) or fp32 on CPU. With `--config`, it
uses the *training* quantisation, not the export's. The tokenizer and template come from `--base` (with `--adapter`)
or from the `--tuned` dir. Leg 1 is exact-match accuracy on whatever task file you give it, and the verdict is a
strict point comparison. `--noise-floor` re-runs the base N times, which under greedy decoding measures ≈ 0. So
with our measured numbers it says **SHIP** for the NF4 artifact we reject, and for the broken-template model.
Not done (no T4 time): a live `soup ship` on the 1.5B model, to confirm the leg-2 general-suite scores; in the rule
test they're a labelled hypothetical (equal).

## What I'd change in Soup (so these can't pass `soup ship` again)
1. Make the ship gate statistical: a paired bootstrap or McNemar on per-item correctness, report n and the CI, and
   refuse to SHIP when the CI lower bound ≤ 0 (or below a user-set minimum effect).
2. Evaluate the artifact that ships. `soup ship --artifact <merged|gguf|nf4 dir>` should be the default after
   export, with an agreement check against the adapter-on model (label flips, per-class recall).
3. Have `train --dry-run` fail on data-doctor MAJOR findings, and pass the config's `chat_template` override into
   data doctor.
4. Warn when `soup merge` copies a template that differs from the base's, and on non-finite values after the dtype
   cast; record the train-time `lora_alpha` / `r` in the adapter's provenance and check it at merge time.
5. Give `merge_4bit` an explicit `dtype` for the non-quantised modules.
6. Turn on a NaN/skip watchdog by default under fp16 (`loss_watchdog` exists but defaults to off).
