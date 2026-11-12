"""Phase 6: what would `soup ship` decide for OUR measured artifacts? (offline --evidence; no model load)

Task scores (leg 1) are the strict accuracies measured on the T4 by verify.py
(runs/phase5/verify_20261112T121313Z/verify.json, 252 held-out rows, greedy). Soup's `exact` scorer equals
our strict_acc for this task (scripts/make_ship_tasks.py).

LEG 2 IS HYPOTHETICAL: we did not run Soup's general suite on the T4 (no time). Each case sets every
bundled benchmark to base == tuned (no regression), so the verdict isolates the leg-1 rule:
    SHIP <=> tuned > base (+ noise floor if given) AND no benchmark regressed   (utils/ship_verdict.py:decide_ship)

Example:
  python scripts/ship_rule_test.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, setup_run, write_json, write_summary  # noqa: E402

VERIFY = REPO_ROOT / "runs" / "phase5" / "verify_20261112T121313Z" / "verify.json"
BENCH = ["mini_mmlu", "mini_common_sense", "mini_instruction", "mini_arithmetic"]


def main():
    import argparse
    a = argparse.Namespace(seed=0)
    run_dir, log = setup_run("phase6", "ship_rule_test", a)
    v = json.loads(VERIFY.read_text(encoding="utf-8"))
    base = v["base_gen"]["strict_acc"]
    g = {R: v["runs"][R]["gen"] for R in v["runs"]}
    cases = {
        "real_nf4 (shipped artifact; our verdict DON'T SHIP)": g["real"]["nf4"]["strict_acc"],
        "real_adapter_on": g["real"]["adapter@1"]["strict_acc"],
        "broken_template_adapter_on": g["broken_template"]["adapter@1"]["strict_acc"],
        "null_merged (identical predictions to base)": g["null_lr1e-7"]["merged"]["strict_acc"],
        "null_nf4": g["null_lr1e-7"]["nf4"]["strict_acc"],
        "synthetic: base + 1 row of 252": round(base + 1 / 252, 6),
    }
    res = {}
    for name, tuned in cases.items():
        ev = {"task": {"mode": "metric", "base": base, "tuned": tuned},
              "benchmarks": {b: {"base": 0.5, "tuned": 0.5} for b in BENCH}}  # leg-2 benchmarks are HYPOTHETICAL (equal); Soup rejects extra keys
        slug = __import__("re").sub(r"[^A-Za-z0-9_]+", "_", name.split(" (")[0]).strip("_")
        evp = run_dir / f"evidence_{slug}.json"
        evp.write_text(json.dumps(ev, indent=2), encoding="utf-8")
        rel = os.path.relpath(evp, REPO_ROOT)  # soup refuses evidence paths outside cwd
        r = subprocess.run([sys.executable, "-m", "soup_cli", "ship", "--evidence", rel,
                            "--output", os.path.relpath(run_dir / f"verdict_{slug}.json", REPO_ROOT)],
                           cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env={**os.environ, "COLUMNS": "200", "NO_COLOR": "1"})
        (run_dir / f"stdout_{slug}.txt").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
        verdict = {0: "SHIP", 2: "DON'T SHIP"}.get(r.returncode, f"exit {r.returncode}")
        res[name] = {"base": base, "tuned": tuned, "delta_pp": round(100 * (tuned - base), 2),
                     "soup_ship_exit": r.returncode, "soup_ship_verdict": verdict}
        log.info("%-55s base=%.4f tuned=%.4f -> %s", name, base, tuned, verdict)
    write_json(run_dir / "results.json", res)
    write_summary(run_dir, {"cases": res, "leg2": "hypothetical equal scores"}, log)


if __name__ == "__main__":
    main()
