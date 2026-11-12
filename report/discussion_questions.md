# Questions you should be able to answer yourself (no answers here — pointers only)

Rule: answer from the evidence, not from memory. Each question lists where the evidence lives.

## Proving the model trained (30%)
1. The loss fell from 0.776 to 0.026. Why does that prove almost nothing here? What share of the supervised tokens
   carry information? → `runs/phase3/inspect_labels_real_*/per_row.jsonl`, D-16, telemetry `mean_token_accuracy`.
2. The null control's eval loss also fell monotonically. How, with lr = 1e-7, and why did its predictions not change?
   → `runs/phase3/memtrace_train_null_lr1e-7_*/telemetry.jsonl`, `verify.json → runs.null_lr1e-7.paired`.
3. Why is "LoRA-B ≠ 0" not evidence of training? Why did 5.9% of the null's fp16 weights change, and how was that
   predicted before looking? → `verify.json → runs.*.weights` (`frac_predicted_to_change_all_adapted`), D-14.
4. Walk through T0: what would it mean if scale-0 did not reproduce the base, or merged ≠ adapter-on?
   → `scripts/verify.py` (`Scaler`, `kl_batch`), `report.md` KL table.
5. Why did T2 fail, and is that a property of the model or of how T2 was defined? Would you change it, and why only
   for the next run? → D-20, D-23, C-30, `base_nf4_gen` per-class recall.
6. The gain is +4 pp with a CI touching zero. How many rows would you need, and why can't you just evaluate again on the
   same 252? → `verify.json → paired`, `prepare_data_* → pool_sizes`, C-22.
7. Why is the "decision position" token #7, and why measure KL there instead of over the whole answer?
   → `verify.py:decision_offset`, `task_format.py`.
8. What would your check NOT notice? Name one failure that passes all of T0–T3 and S1–S7.
   → `report.md` "What this check does NOT detect", D1 in the fault table.

## Silent failures (20%)
9. Why did the broken template score the same accuracy as the real run, and which signals revealed it anyway?
   → `analysis/verdict.md` post-hoc, `runs/phase3/memtrace_train_broken_template_*` (`num_tokens`),
   `runs/phase4/merge_ref_broken_template_*`.
10. Which of Soup's checks look at the config's chat_template override, and why can `soup data doctor` not?
    → `soup data doctor --help`, `runs/phase6/soup_checks_T1_*`.
11. Data doctor flagged truncation and unknown roles as MAJOR, yet `soup train --dry-run` said "Ready to train".
    What should the contract between those two be? → `runs/phase6/soup_checks_D3_*`, `_D4_*`.
12. Your own inspector missed two injected faults in v1. Which, why, and what changed in v2?
    → `runs/phase6/fault_inject_20261112T132518Z` vs the v2 run, `scripts/inspect_soup_labels.py`.
13. An edited `lora_alpha` passes `soup merge` and your own verify. Why exactly, and what is the only evidence that
    catches it? → `runs/phase6/fault_inject_artifacts_*` (M1), `telemetry.jsonl → train_begin.peft_config`.
14. The bf16 → fp16 cast: how many elements changed in the real model, could any overflow, who would tell you?
    → `verify.json → weights.base_to_merged_dtype_cast_*`, Q2 in the artifact faults.
15. What does `soup ship` actually evaluate: which weights, which dtype, which tokenizer, which decision rule? Does it
    ever see the NF4 file? → `soup_cli/commands/ship.py` (`_load_generators`), `utils/ship_verdict.py:decide_ship`,
    `utils/live_eval.py`, `runs/phase6/ship_rule_test_*`.
16. What would you change in Soup so this failure can't pass `soup ship` again? Be concrete about which failure.

## Memory & hardware (15%)
17. Derive the 2.875 GiB weights and the 0.069 GiB LoRA params from the config. → `scripts/memory_budget.py`.
18. Why is the logits chain the biggest training term under checkpointing, and why was your estimate high by
    ~3 B/element? → `analysis/estimate_vs_measured.md` §1.
19. Probe (e) was "predicted not to fit" and did OOM. Was the prediction right? → `runs/phase2/mem_probe_b8_s256_*`.
20. Do you trust the reported VRAM? Which number would you put in a capacity plan, and why do the probes not show the
    reserved/allocated gap the real run shows? → §2 of the memory analysis, `nvidia_smi_1s.csv`.
21. Do you trust the reported tok/s? Which tok/s, counting what, including what? Why is the unmerged adapter ~1.7×
    slower at inference, and base-NF4 slower than the other NF4 files? → `train_real_stdout.txt`, telemetry
    `num_tokens`, `report.md` behaviour table, `runs/phase4/quantize_base_* → reload_non4bit_param_dtypes`.
22. Why fp16 LoRA instead of Soup's default QLoRA, and what would QLoRA have done to "merged == what was trained"?
    → D-10, probe (d).
23. Eager attention gave NaN from step 1 with no error. What exactly overflowed, and how would you have noticed in a
    real run? → `runs/phase2/mem_probe_b1_s512_*_eager_*`, C-26.
24. With three days on rented H100s, what would you investigate first and why that and not something else?

## Process / AI use
25. Where was the AI wrong, and how was each mistake caught? → `AI_USAGE.md` (rows marked "AI error").
26. Which thresholds are judgement calls, and how do you defend a threshold you'd now change? → C-23, D-21 amendment.
