# Should we ship it? LoRA on Qwen2.5-1.5B-Instruct (RuReviews sentiment → strict JSON), merged + NF4

**Verdict: DON'T SHIP** under criteria fixed before any held-out result (DECISIONS.md D-20/D-21, commits
`d76ee50`, `064d807`). Every number below comes from a file in `runs/`. Analysis: `analysis/*.md`.

**Setup.** Qwen2.5-1.5B-Instruct. Task: answer a Russian product review with
`{"тональность": "<негативный|нейтральный|позитивный>"}`. RuReviews (Apache-2.0) deduped and hash-split
1500 / 150 / **252 held-out** (`runs/phase1/prepare_data_*`). Soup 0.77.2 on a free Colab T4: LoRA r16/α32,
all 7 linear projections, **fp16 base** (not Soup's default QLoRA, D-10), 2 epochs, 188 steps. Also trained:
a **null control** (lr 1e-7, otherwise identical) and a **broken-template** run (Soup's `qwen2.5` alias).
Merged with `soup merge` (fp16), quantised with Soup's `merge_4bit` (bitsandbytes NF4).

## 1. Memory: estimate vs measured (GiB; `analysis/estimate_vs_measured.md`)

| | estimate | measured | why the gap |
|---|---|---|---|
| fp16 weights / LoRA params | 2.875 / 0.069 | 2.875 / 0.069 | exact |
| probe b4·s512·ckpt, forward peak | 6.40 | 6.23 | activations + logits forward (10 B/el) right to ±3% |
| same, max_allocated (backward) | 8.21 | **7.33** | logits backward ≈ 11 B/el, not my 14 (tensor lifetimes); Soup's estimator uses 11.6 |
| probe b8·s256·no-ckpt ("should not fit") | 14.89 + context | 14.09 for 2 steps, **OOM at step 3** | right outcome, wrong reason: 1.34 GiB fragmentation, not capacity |
| CUDA context / non-torch | 0.45 (Soup 0.8–1.2) | **0.10–0.14** | lazy CUDA module loading (likely) |
| real `soup train` run | — | allocated **6.67**, reserved **13.84**, nvidia-smi **14.3 GB** | variable batch shapes leave the allocator holding a cache twice the live tensors |
| merge on GPU / NF4 quantise | 2.93 / 1.09 | 3.05 / 1.10 | +adapter kept on the GPU / exact |
| NF4 file | 1.064 | 1.075 | quant-state + tokenizer |

Do I trust the reported VRAM? nvidia-smi says the card is 93% full. The run needs about 7 GiB.

## 2. Did it really train? (`runs/phase5/verify_20261112T121313Z/report.md`)

A falling loss proves nothing here. The base already outputs valid JSON 100% of the time and starts at 0.92
token accuracy. 13 of the 16–18 supervised tokens per row are identical in every row (role prefix, JSON
skeleton, `"}`, `<|im_end|>`); in effect one token carries the decision (`runs/phase3/inspect_labels_real_*`, D-16). The **null
control's eval loss fell steadily** (0.754 → 0.728) and Soup printed "Loss 0.776 → 0.693" while its
predictions stayed **identical to the base**. `verify.py` compares base · adapter (scale 0 / 0.5 / 1) ·
merged · NF4 on the same 252 prompts:

| | real | null | broken |
|---|---|---|---|
| T0 scale 0 = base; merged = adapter-on (decision KL) | 0 / 1.2e-5 ✔ | 0 / 2.2e-5 ✔ | 0 / 1.2e-5 ✔ |
| fp16 elements changed in adapted weights (predicted from half-ulp) | **93.6%** (93.6%) | **5.9%** (5.9%) | 94.2% |
| decision-position KL(base‖merged) | **0.370** | **3.4e-5** | 0.433 |
| Δ strict_acc merged − base [95% paired bootstrap CI] | +3.6 pp [−2.4, +9.5] | 0.0 [0, 0] | +4.0 pp [−2.0, +10.3] |

Soup's merge matches our independent peft merge **bit for bit**, and all 142 untouched tensors are
unchanged. The model did change, by about 10⁴× the null control in KL. But the **useful** change, +4 pp on 252
rows, can't be told apart from zero. The pre-registered T2 compared against KL(base‖base-NF4) = 0.266, which
turned out to be a large systematic shift, not noise (§3), so T2 fails. **What the check does NOT detect:**
wrong gold labels, generalisation beyond this prompt and task, damage to other abilities, other runtimes
(GGUF/vLLM), free-running divergence, effects below about ±6 pp, seed variance.

## 3. Silent failures (fault-injected; `analysis/silent_failures.md`)

Injected on purpose (0.5B on CPU) or observed on the T4. ✅ flagged · ❌ passed silently. Full table, with 21
faults, evidence paths and the Soup fixes: `analysis/silent_failures.md`.

| fault | data doctor | doctor / plan / dry-run | soup ship | ours |
|---|---|---|---|---|
| held-out rows leaked into train | ❌ | ❌ | ❌ | ✅ hash check |
| max_length truncates every answer | ✅ MAJOR | ❌ "Ready to train" | — | ✅ |
| template override without EOS | ❌ can't see override | ❌ | — | ✅ |
| `chat_template: qwen2.5` (no system prompt) | ❌ | ❌ | ❌ SHIP | ✅ (T4: −63k tokens) |
| system prompt in train data only | ❌ | ❌ | — | v1 ❌ → v2 ✅ |
| lr ≈ 0, loss still falls | ❌ | ❌ | ✅ (tie) | ✅ verify |
| eager attention fp16 → NaN, all steps skipped (T4) | — | n/t | n/t | ✅ probe telemetry |
| `lora_alpha` edited after training | — | merge exit 0 | n/t | ❌ verify / ✅ telemetry |
| merge ships the adapter's overridden template (T4) | — | merge exit 0 | — | ✅ S6 |
| bf16 weight → `inf` in the fp16 merge | — | merge exit 0, no warning | n/t | ✅ |
| NF4 flips 7.9% of labels, accuracy unchanged (T4) | — | — | ❌ SHIP | ✅ S3 |
| `soup ship` rule: +1 correct row of 252 ⇒ SHIP | — | — | ❌ | ✅ CI required |

`soup ship --adapter` scores base + adapter in bf16 and **never sees the merged or 4-bit file** unless you pass it
with `--tuned`.

## 4. Verdict: DON'T SHIP (`analysis/verdict.md`)

| S1 trained + null separated | S2 ≥ +5 pp, CI > 0 | S3 NF4 fidelity | S4 format/EOS | S5 no collapse | S6 integrity | S7 user sampling |
|---|---|---|---|---|---|---|
| **FAIL** (T2, T3) | **FAIL** +4.0 [−2.0, +9.9] | **FAIL** agreement 92.1% < 95% | PASS | PASS | PASS | PASS |

What must change first:
1. Re-test base vs NF4 on ≥1,500 of the 6,863 never-used held-out rows, thresholds registered first.
2. Fix the 4-bit step until NF4 agrees with fp16 on ≥95% of labels. NF4 moves the trained model from
   *neutral* to *negative* (recall 0.69 → 0.58), and accuracy hides it because the errors cancel.
3. Measure the neutral-class gold-label noise.

`soup ship`, fed the same accuracies, returns **SHIP** for this artifact, and for the broken-template model.

## 5. What surprised me, what still concerns me

What surprised me most was how many wrong things look right. The null control's loss fell and Soup printed
"0.776 → 0.693" for a model whose predictions never changed. The broken template scored the *same* accuracy as the
real run (0.698); only a 63,000-token gap in the telemetry and a template diff gave it away. 4-bit quantisation was
not noise but a directional shift: NF4 almost deletes the base model's *neutral* class (recall 0.57 → 0.07), and on
the trained model it moved 7.9% of labels while accuracy didn't move, because the errors cancel. nvidia-smi showed
14.3 of 15.4 GB for a run that needs about 7. What still concerns me: the verdict rests on 252 star-derived labels
and one seed. My T2 yardstick turned out to be a poor choice. My own checks have blind spots: the label inspector
missed 2 of 9 injected faults until I added two checks, my eval would hide a missing EOS in `generation_config`,
and an edited `lora_alpha` passes my verify. And `soup ship`, given these same numbers, would have shipped it.
