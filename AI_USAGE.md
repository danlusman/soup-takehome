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

(Engineer: fill in the "decision" column yourself and add anything you re-did by hand.)
