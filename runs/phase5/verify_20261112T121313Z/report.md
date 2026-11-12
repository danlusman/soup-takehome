# verify.py report — verdict: **DON'T SHIP** (artifact: nf4)

rows=252 · device=cuda · NF4 used=True · decision position = answer token #7
numeric floor KL(base batched‖base single-row) = 3.3785e-06

## Trained-ness (D-20)

| check | real | null_lr1e-7 | broken_template |
|---|---|---|---|
| T0_scale0_is_base | PASS | PASS | PASS |
| T0_merged_eq_adapter_on | PASS | PASS | PASS |
| T1_weights_changed | PASS | FAIL | PASS |
| T2_kl_above_floor | FAIL | FAIL | FAIL |
| T3_behaviour_changed | FAIL | FAIL | FAIL |
| overall | NOT TRAINED | NOT TRAINED | NOT TRAINED |

## Weights

| metric | real | null_lr1e-7 | broken_template |
|---|---|---|---|
| B_norm_sum | 48.6252 | 0.3126 | 52.7012 |
| n_B_all_zero | 0 | 0 | 0 |
| rel_dW_adapter_median | 0.0053 | 3.3599e-05 | 0.0058 |
| frac_predicted_to_change_all_adapted | 0.9359 | 0.0585 | 0.9416 |
| frac_changed_all_adapted | 0.9361 | 0.0586 | 0.9417 |
| rel_diff_merged_median | 0.0053 | 1.6508e-05 | 0.0058 |
| consistency_resid_median | 0.0390 | 0.9385 | 0.0359 |
| untouched_tensors_bit_identical | 142 | 142 | 142 |
| soup_vs_ref_min_equal_frac | 1.0000 | 1.0000 | 1.0000 |
| soup_vs_ref_max_abs | 0.0000 | 0.0000 | 0.0000 |

## KL (teacher-forced, mean per position)

| pair | real all / decision | null_lr1e-7 all / decision | broken_template all / decision |
|---|---|---|---|
| base||adapter@0 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| base||adapter@0.5 | 0.0564 / 0.2000 | 3.0210e-06 / 2.1289e-05 | 0.0321 / 0.2284 |
| base||adapter@1 | 0.0998 / 0.3706 | 5.8950e-06 / 2.5989e-05 | 0.0718 / 0.4334 |
| adapter@1||merged | 9.3325e-07 / 1.2035e-05 | 3.5900e-06 / 2.2007e-05 | 9.3099e-07 / 1.1850e-05 |
| base||merged | 0.0997 / 0.3700 | 4.9302e-06 / 3.4234e-05 | 0.0718 / 0.4332 |
| merged||nf4 | 0.0023 / 0.0274 | 0.0241 / 0.2646 | 0.0077 / 0.0991 |
| base||nf4 | 0.0905 / 0.3767 | 0.0242 / 0.2646 | 0.0828 / 0.5320 |
| base||base_nf4[floor] | 0.0235 / 0.2660 | 0.0235 / 0.2660 | 0.0235 / 0.2660 |

## Behaviour (greedy, held-out)

| artifact | strict_acc [95% CI] | format | lenient | acc|valid | EOS | leak | gen tok p50/p99 | tok/s |
|---|---|---|---|---|---|---|---|---|
| base fp16 | 0.6587 [0.5992, 0.7143] | 1.0000 | 0.6587 | 0.6587 | 1.0000 | 0.0000 | 13/14 | 205.92 |
| base NF4 | 0.6270 [0.5675, 0.6865] | 1.0000 | 0.6270 | 0.6270 | 1.0000 | 0.0000 | 14/14 | 132.45 |
| real adapter@1 | 0.6944 [0.6389, 0.754] | 1.0000 | 0.6944 | 0.6944 | 1.0000 | 0.0000 | 13/14 | 119.62 |
| real merged | 0.6944 [0.6389, 0.754] | 1.0000 | 0.6944 | 0.6944 | 1.0000 | 0.0000 | 13/14 | 210.76 |
| real nf4 | 0.6984 [0.6429, 0.754] | 1.0000 | 0.6984 | 0.6984 | 1.0000 | 0.0000 | 13/14 | 193.98 |
| null_lr1e-7 adapter@1 | 0.6587 [0.5992, 0.7143] | 1.0000 | 0.6587 | 0.6587 | 1.0000 | 0.0000 | 13/14 | 118.38 |
| null_lr1e-7 merged | 0.6587 [0.5992, 0.7143] | 1.0000 | 0.6587 | 0.6587 | 1.0000 | 0.0000 | 13/14 | 205.36 |
| null_lr1e-7 nf4 | 0.6310 [0.5714, 0.6905] | 1.0000 | 0.6310 | 0.6310 | 1.0000 | 0.0000 | 14/14 | 197.72 |
| broken_template adapter@1 | 0.6984 [0.6429, 0.754] | 1.0000 | 0.6984 | 0.6984 | 1.0000 | 0.0000 | 13/14 | 118.74 |
| broken_template merged | 0.6984 [0.6429, 0.754] | 1.0000 | 0.6984 | 0.6984 | 1.0000 | 0.0000 | 13/14 | 205.73 |
| broken_template nf4 | 0.6865 [0.631, 0.746] | 1.0000 | 0.6865 | 0.6865 | 1.0000 | 0.0000 | 14/14 | 195.03 |

### Paired bootstrap (Δ strict_acc = A − B)

| run | comparison | Δ | 95% CI | A-only correct | B-only correct | label agreement |
|---|---|---|---|---|---|---|
| real | merged_vs_base | 0.0357 | [-0.0238, 0.0952] | 34 | 25 | 0.7381 |
| real | adapter_vs_merged | 0.0000 | [0.0, 0.0] | 0 | 0 | 1.0000 |
| real | nf4_vs_base | 0.0397 | [-0.0198, 0.0992] | 34 | 24 | 0.7500 |
| real | nf4_vs_merged | 0.0040 | [-0.0278, 0.0357] | 10 | 9 | 0.9206 |
| null_lr1e-7 | merged_vs_base | 0.0000 | [0.0, 0.0] | 0 | 0 | 1.0000 |
| null_lr1e-7 | adapter_vs_merged | 0.0000 | [0.0, 0.0] | 0 | 0 | 0.9960 |
| null_lr1e-7 | nf4_vs_base | -0.0278 | [-0.0952, 0.0397] | 35 | 42 | 0.6468 |
| null_lr1e-7 | nf4_vs_merged | -0.0278 | [-0.0952, 0.0397] | 35 | 42 | 0.6468 |
| broken_template | merged_vs_base | 0.0397 | [-0.0198, 0.1032] | 37 | 27 | 0.7262 |
| broken_template | adapter_vs_merged | 0.0000 | [0.0, 0.0] | 0 | 0 | 1.0000 |
| broken_template | nf4_vs_base | 0.0278 | [-0.0357, 0.0913] | 38 | 31 | 0.6786 |
| broken_template | nf4_vs_merged | -0.0119 | [-0.0675, 0.0437] | 22 | 25 | 0.7897 |

Shipped-sampling check ({'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'repetition_penalty': 1.1}): format_rate=1.0000, strict_acc=0.6944
Rerun determinism (first 32 rows identical): 1.0

## SHIP criteria (D-21)

| criterion | result |
|---|---|
| S1_trained_and_null_separated | FAIL |
| S2_gain_vs_base | FAIL |
| S3_quantization_fidelity | FAIL |
| S4_contract | PASS |
| S5_no_collapse | PASS |
| S6_integrity | PASS |
| S7_shipped_sampling_format | PASS |

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

