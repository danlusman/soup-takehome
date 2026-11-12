# verify.py report — verdict: **DON'T SHIP** (artifact: nf4)

rows=9 · device=cpu · NF4 used=True · decision position = answer token #7
numeric floor KL(base batched‖base single-row) = 8.9526e-05

## Trained-ness (D-20)

| check | real | null_lr1e-7 |
|---|---|---|
| T0_scale0_is_base | PASS | PASS |
| T0_merged_eq_adapter_on | PASS | PASS |
| T1_weights_changed | PASS | FAIL |
| T2_kl_above_floor | PASS | FAIL |
| T3_behaviour_changed | FAIL | FAIL |
| overall | NOT TRAINED | NOT TRAINED |

## Weights

| metric | real | null_lr1e-7 |
|---|---|---|
| B_norm_sum | 7.1283 | 0.0045 |
| n_B_all_zero | 0 | 0 |
| rel_dW_adapter_median | 0.0022 | 1.4083e-06 |
| frac_predicted_to_change_all_adapted | 0.8723 | 0.0014 |
| frac_changed_all_adapted | 0.8726 | 0.0014 |
| rel_diff_merged_median | 0.0022 | 1.5064e-07 |
| consistency_resid_median | 0.0951 | 0.9978 |
| untouched_tensors_bit_identical | 122 | 122 |
| soup_vs_ref_min_equal_frac | 1.0000 | 1.0000 |
| soup_vs_ref_max_abs | 0.0000 | 0.0000 |

## KL (teacher-forced, mean per position)

| pair | real all / decision | null_lr1e-7 all / decision |
|---|---|---|
| base||adapter@0 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| base||adapter@0.5 | 0.0400 / 0.5163 | 2.2593e-04 / 0.0025 |
| base||adapter@1 | 0.1683 / 2.1491 | 1.4445e-04 / 0.0011 |
| adapter@1||merged | 3.1650e-04 / 8.8888e-04 | 2.7094e-04 / 0.0018 |
| base||merged | 0.1720 / 2.2092 | 2.7735e-04 / 0.0030 |
| merged||nf4 | 0.0522 / 0.0728 | 0.0348 / 0.2417 |
| base||nf4 | 0.2372 / 2.4516 | 0.0346 / 0.2317 |
| base||base_nf4[floor] | 0.0315 / 0.2045 | 0.0315 / 0.2045 |

## Behaviour (greedy, held-out)

| artifact | strict_acc [95% CI] | format | lenient | acc|valid | EOS | leak | gen tok p50/p99 | tok/s |
|---|---|---|---|---|---|---|---|---|
| base fp16 | 0.3333 [0.0, 0.6667] | 1.0000 | 0.3333 | 0.3333 | 1.0000 | 0.0000 | 13/13 | 2.68 |
| base NF4 | 0.5556 [0.2222, 0.8889] | 0.8889 | 0.6667 | 0.6250 | 1.0000 | 0.0000 | 13/14 | 0.66 |
| real adapter@1 | 0.6667 [0.3333, 1.0] | 1.0000 | 0.6667 | 0.6667 | 1.0000 | 0.0000 | 14/14 | 4.81 |
| real merged | 0.6667 [0.3333, 1.0] | 1.0000 | 0.6667 | 0.6667 | 1.0000 | 0.0000 | 14/14 | 3.98 |
| real nf4 | 0.4444 [0.1111, 0.7778] | 0.7778 | 0.6667 | 0.5714 | 1.0000 | 0.0000 | 13/14 | 1.22 |
| null_lr1e-7 adapter@1 | 0.3333 [0.0, 0.6667] | 1.0000 | 0.3333 | 0.3333 | 1.0000 | 0.0000 | 13/13 | 4.24 |
| null_lr1e-7 merged | 0.3333 [0.0, 0.6667] | 1.0000 | 0.3333 | 0.3333 | 1.0000 | 0.0000 | 13/13 | 4.88 |
| null_lr1e-7 nf4 | 0.5556 [0.2222, 0.8889] | 0.8889 | 0.6667 | 0.6250 | 1.0000 | 0.0000 | 13/14 | 0.76 |

### Paired bootstrap (Δ strict_acc = A − B)

| run | comparison | Δ | 95% CI | A-only correct | B-only correct | label agreement |
|---|---|---|---|---|---|---|
| real | merged_vs_base | 0.3333 | [0.0, 0.6667] | 3 | 0 | 0.3333 |
| real | adapter_vs_merged | 0.0000 | [0.0, 0.0] | 0 | 0 | 1.0000 |
| real | nf4_vs_base | 0.1111 | [0.0, 0.3333] | 1 | 0 | 0.3333 |
| real | nf4_vs_merged | -0.2222 | [-0.5556, 0.0] | 0 | 2 | 0.7778 |
| null_lr1e-7 | merged_vs_base | 0.0000 | [0.0, 0.0] | 0 | 0 | 1.0000 |
| null_lr1e-7 | adapter_vs_merged | 0.0000 | [0.0, 0.0] | 0 | 0 | 1.0000 |
| null_lr1e-7 | nf4_vs_base | 0.2222 | [0.0, 0.5556] | 2 | 0 | 0.6667 |
| null_lr1e-7 | nf4_vs_merged | 0.2222 | [0.0, 0.5556] | 2 | 0 | 0.6667 |

Shipped-sampling check ({'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'repetition_penalty': 1.1}): format_rate=0.5556, strict_acc=0.3333
Rerun determinism (first 32 rows identical): 1.0

## SHIP criteria (D-21)

| criterion | result |
|---|---|
| S1_trained_and_null_separated | FAIL |
| S2_gain_vs_base | FAIL |
| S3_quantization_fidelity | FAIL |
| S4_contract | FAIL |
| S5_no_collapse | FAIL |
| S6_integrity | PASS |
| S7_shipped_sampling_format | FAIL |

## What this check does NOT detect
- **Wrong gold labels.** RuReviews labels come from star ratings (CONCERNS C-01/C-09). The check measures
  agreement with those labels, not whether the sentiment is actually right.
- **Generalisation.** Only 252 prompts, all from the same distribution and with our exact instruction and
  JSON contract. Nothing here tests other phrasings, other domains, multi-turn use, or inputs over 512 tokens.
- **Damage to other abilities.** We never test general chat, Russian fluency, or safety outside this task.
  A model that only outputs sentiment JSON would pass.
- **Other runtimes and decoders.** The verdict covers transformers + bitsandbytes NF4 on CUDA, greedy decoding,
  plus one sampled check (S7). GGUF/llama.cpp, vLLM and other tokenizer implementations are not covered (C-21).
- **Free-running divergence.** The KL numbers are teacher-forced on the gold answer, so they miss errors that
  compound once the model generates its own tokens. Only the generation metrics see those, and only at the
  label level.
- **Small effects.** With n=252, a CI half-width of about ±6 pp means a real gain of a few points can look like zero.
- **Seed variance.** One training seed and one data order, so we have no estimate of run-to-run variance.
- **Shortcuts.** Accuracy cannot tell real understanding apart from shortcuts such as length or keywords.
  Per-class recall and the confusion matrix are the only partial guard.
- **Weight-level tests show change, not usefulness.** The null control's B matrices are non-zero, which is
  why the weight test is not trusted on its own.

