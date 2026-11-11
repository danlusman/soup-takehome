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
