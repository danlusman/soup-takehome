# AI_USAGE

Assistant: Claude (Claude Code desktop, model claude-opus-5-5) as pair engineer. The engineer runs all
GPU work in Colab; the assistant has no GPU and must not report numbers it has not seen in a raw log.

Columns: what the AI suggested · what the engineer accepted / rejected · what was verified, changed or
redone (and how).

## Phase 1 — data + baseline

| # | AI suggestion | Engineer decision | Verification / changes |
|---|---|---|---|
| 1 | Pointed out the first "task.pdf" paste was the engineer's own brief, not the task document | Engineer supplied the real PDF | PDF read in full; differences noted (4–6 h scope, tok/s question) |
| 2 | Three dataset candidates (Headlines JSON / RuReviews / NEREL), recommended Headlines | **Rejected recommendation, picked RuReviews (B)** | Licenses + sizes checked live via HF API and dataset viewer, not from memory |
| 3 | Use `ai-forever/ru-reviews-classification` (has `label_text`) instead of the mteb copy | pending engineer review | label mapping read from rows; prepare_data re-checks every row |
| 4 | Strict JSON target with Russian label words; instruction also given to base model | pending engineer review | — |
| 5 | Hash-based split over pooled source splits, balanced 500/50/84 per class | pending engineer review | — |
| 6 | Soup command names (`soup data doctor`, `soup plan` as "pre-flight", `soup ship --base --adapter`) | — | read from the cloned Soup source (v0.77.2), NOT yet run; `soup ship` not seeing merged 4-bit is a hypothesis for Phase 6 |
| 7 | Wrote prepare_data.py / eval_task.py / task_format.py / common.py and ran the CPU smoke test (0.5B, 9 held-out rows) | pending engineer review | Smoke run logs in `runs/smoke/`. Found and recorded: console mojibake was a Windows cp1251 display artefact, not data (re-checked with UTF-8 output); label-conflict evidence (C-09); 0.5B constant-predictor behaviour (C-10) |
| 8 | Leave torch unpinned on Colab (`torch>=2.6`) | pending engineer review | Bends the "pin everything" rule; reason in D-08 |

## Phase 2 — memory budget

| # | AI suggestion | Engineer decision | Verification / changes |
|---|---|---|---|
| 9 | Itemized analytic estimator + Soup's own estimator as a second column | pending | Param + LoRA formulas checked exactly against a real 0.5B load (runs/smoke/mem_probe_*) |
| 10 | Train LoRA on fp16 base instead of Soup's default 4-bit QLoRA (D-10) | pending | reasoning only; probe (d) measures the NF4 alternative |
| 11 | Planned shape b=4, s=512, grad-ckpt, r=16 all-linear (D-11) | pending | estimate only until probes run on the T4 |
| 12 | Controlled probe script instead of reading Soup's peak; in-process Trainer callback for the real Soup run | pending | CPU smoke only: CUDA stats, nvidia-smi sampler and the Trainer patch are untested |
| 13 | Probe (e) as a pre-registered "should OOM" test | pending | — |

## Phase 3 — training

| # | AI suggestion | Engineer decision | Verification / changes |
|---|---|---|---|
| 14 | fp16 LoRA instead of Soup's QLoRA default | **Accepted by engineer** | — |
| 15 | Hyper-parameters a priori (D-13), no held-out eval until SHIP criteria exist | pending | — |
| 16 | Null control lr=1e-7 (D-14) | pending | lr=0 and mismatched targets verified to be rejected / crash |
| 17 | Broken-template run via Soup's own `qwen2.5` alias (D-15) | pending | found by reading Soup source; render mismatch confirmed 0/12 vs 12/12 on CPU |
| 18 | `inspect_soup_labels.py`: run Soup's formatter to see the real loss mask | pending | revealed role-prefix tokens in loss (D-16) |
| 19 | Telemetry callback (scale, skipped steps, B norms) injected into Soup's Trainer | pending | CPU smoke run OK; fp16 scaler path untestable on CPU |

## Phase 4–5 — merge, quantize, verify

| # | AI suggestion | Engineer decision | Verification / changes |
|---|---|---|---|
| 20 | `soup merge` primary + plain-peft reference merge, diffed tensor-by-tensor (D-17) | pending | CPU smoke: both merges run; diff reported by verify.py |
| 21 | NF4 via Soup's `merge_4bit` on the already-merged dir; GGUF rejected (D-18) | pending | CPU smoke: bnb 0.50 quantizes on CPU, 168 Linear4bit after reload (24 layers × 7) |
| 22 | Pre-registered T0–T3 / S1–S7 thresholds, committed before any held-out result (D-20/21) | pending | commit d76ee50; S6 amended before results (D-21 amendment) |
| 23 | verify.py weight test vs raw base tensors | **AI error, caught by smoke test** | base is bf16; compare against base cast to fp16 (D-22). Without the smoke run the null control would have passed T1 |
| 24 | verify.py read safetensors lazily | **AI error** | segfault on Windows; fixed by clone inside safe_open |

(Engineer: fill in the "decision" column yourself and add anything you re-did by hand.)
