# CPU smoke test of the Phase 1 scripts with Qwen2.5-0.5B-Instruct and ~9 examples.
# Outputs go to runs/smoke/ and data_smoke/ (never to data/ or runs/phase1/).
$ErrorActionPreference = "Stop"
$py = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$tok = "Qwen/Qwen2.5-0.5B-Instruct"

& $py (Join-Path $PSScriptRoot "prepare_data.py") --tokenizer $tok `
    --train-per-class 3 --val-per-class 1 --heldout-per-class 3 `
    --out-data data_smoke --phase smoke
if ($LASTEXITCODE -ne 0) { throw "prepare_data failed" }

& $py (Join-Path $PSScriptRoot "eval_task.py") --model $tok `
    --data data_smoke/processed/heldout.jsonl --tag smoke_base --batch-size 4 --phase smoke --bootstrap 200
if ($LASTEXITCODE -ne 0) { throw "eval_task failed" }
Write-Host "smoke OK"
