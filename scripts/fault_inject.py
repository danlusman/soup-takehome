"""Phase 6: data / template / training-config fault injection, run through Soup's checks AND ours.

For each fault we write a broken copy of the (smoke) training data and/or config under
faults/<id>/, then run:
  * scripts/soup_checks.py  -> soup data doctor, soup doctor --config, soup plan, soup train --dry-run,
                               + our inspect_soup_labels.py (what Soup actually puts in the loss)
  * our leak check          -> every training row's normalized-key hash vs data_smoke/splits/split_hashes.json
and record, per fault, whether each check flagged it (non-zero exit, a non-OK data-doctor row, or
an inspector invariant broken). The raw output of every check is kept in its own run dir; the
summary only says flagged / not flagged and points to the file.

"Flagged" is deliberately literal: a Soup check flags a fault only if its exit code is non-zero
or data doctor reports a non-OK verdict on a row it did not already report for the clean config.

Example:
  python scripts/fault_inject.py --phase phase6
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, read_jsonl, setup_run, sha256_text, write_json, write_jsonl, write_summary  # noqa: E402
from prepare_data import dedupe_key  # noqa: E402

BASE_CFG = REPO_ROOT / "configs" / "smoke_real.yaml"
TRAINVAL = REPO_ROOT / "data_smoke" / "processed" / "trainval_soup.jsonl"
HELDOUT = REPO_ROOT / "data_smoke" / "processed" / "heldout.jsonl"
SPLITS = REPO_ROOT / "data_smoke" / "splits" / "split_hashes.json"
SALT = "soup-rureviews-v1"

NO_EOS_TEMPLATE = ("{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}"
                   "{% if message['role'] != 'assistant' %}<|im_end|>\n{% endif %}{% endfor %}"
                   "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def faults(rows: list[dict], heldout: list[dict]):
    """Yield (fault_id, description, rows, config_replacements)."""
    rng = random.Random(0)
    # D1 labels shuffled across rows: perfectly formatted, wrong supervision
    ans = [r["messages"][-1]["content"] for r in rows]
    perm = ans[:]
    while perm == ans:
        rng.shuffle(perm)
    yield ("D1_shuffled_labels", "assistant answers permuted across rows",
           [{"messages": r["messages"][:-1] + [{"role": "assistant", "content": a}]} for r, a in zip(rows, perm)], [])
    # D2 held-out leak: held-out rows appended to the TRAIN part (before the val tail)
    n_val = 3
    leak = [{"messages": h["messages"]} for h in heldout[:3]]
    yield ("D2_heldout_leak", "3 held-out rows copied into the training part",
           rows[:-n_val] + leak + rows[-n_val:], [("  val_split: 0.25", f"  val_split: {round(n_val / (len(rows) + 3), 4)}")])
    # D3 truncation: max_length below every row's length -> answer cut off
    yield ("D3_truncation_max_len_96", "max_length 96 < every full sequence (121+ tokens)", rows,
           [("  max_length: 512", "  max_length: 96")])
    # D4 wrong role name
    yield ("D4_role_bot", "assistant role renamed to 'bot'",
           [{"messages": r["messages"][:-1] + [{"role": "bot", "content": r["messages"][-1]["content"]}]} for r in rows], [])
    # D5 empty target
    yield ("D5_empty_target", "assistant content is the empty string",
           [{"messages": r["messages"][:-1] + [{"role": "assistant", "content": ""}]} for r in rows], [])
    # D6 template that never closes the assistant turn -> EOS never trained
    yield ("D6_no_eos_template", "data.chat_template override without <|im_end|> after assistant", rows,
           [("  # chat_template: unset -> the tokenizer's own Qwen2.5 template (same one eval_task.py uses)",
             "  chat_template: " + json.dumps(NO_EOS_TEMPLATE))])
    # D7 system prompt present in training data but not at inference
    yield ("D7_system_in_train_only", "custom system message in every training row (eval has none)",
           [{"messages": [{"role": "system", "content": "Ты — классификатор отзывов."}] + r["messages"]} for r in rows], [])
    # T1 Soup's qwen2.5 alias (= plain ChatML, no default system prompt)
    yield ("T1_qwen25_alias", "data.chat_template: qwen2.5 (Soup alias = ChatML without default system prompt)", rows,
           [("  # chat_template: unset -> the tokenizer's own Qwen2.5 template (same one eval_task.py uses)",
             "  chat_template: qwen2.5")])
    # C1 learning rate ~0 (null control) — pre-training checks cannot see effectiveness
    yield ("C1_lr_1e-7", "lr 2e-4 -> 1e-7 (the null control)", rows, [("  lr: 2.0e-4", "  lr: 1.0e-7")])


def leak_check(rows: list[dict]) -> dict:
    split = json.loads(SPLITS.read_text(encoding="utf-8"))
    held = set(split["splits"]["heldout"]["hashes"])
    hits = 0
    for r in rows:
        user = next(m["content"] for m in r["messages"] if m["role"] == "user")
        review = user.split("Отзыв:\n", 1)[-1]
        if sha256_text(SALT + "\x00" + dedupe_key(review)) in held:
            hits += 1
    return {"heldout_rows_in_train_file": hits, "flagged": hits > 0}


def run_checks(cfg: Path, tag: str, phase: str, log) -> Path:
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "soup_checks.py"), "--config", str(cfg),
           "--tag", tag, "--phase", phase]
    log.info("running %s", " ".join(cmd))
    subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    runs = sorted((REPO_ROOT / "runs" / phase).glob(f"soup_checks_{tag}_*"))
    return runs[-1]


def interpret(run: Path, clean: dict | None) -> dict:
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    c = s["checks"]
    out = {}
    dd = c.get("data_doctor", {})
    base_non_ok = {tuple(x) for x in (clean or {}).get("data_doctor", {}).get("non_ok_checks", [])}
    new_non_ok = [x for x in dd.get("non_ok_checks", []) if tuple(x) not in base_non_ok]
    out["soup_data_doctor"] = {"flagged": dd.get("exit") not in (0, None) or bool(new_non_ok),
                               "exit": dd.get("exit"), "overall": dd.get("overall"), "new_non_ok": new_non_ok}
    for k in ("doctor", "plan", "dry_run"):
        out[f"soup_{k}"] = {"flagged": c.get(k, {}).get("exit") not in (0, None), "exit": c.get(k, {}).get("exit")}
    # our inspector
    insp = sorted((REPO_ROOT / "runs" / s.get("phase", "phase6")).glob("inspect_labels_*"))
    ins_summary = None
    txt = (run / "inspect_labels.txt").read_text(encoding="utf-8") if (run / "inspect_labels.txt").exists() else ""
    m = re.search(r"summary written: (.+summary\.json)", txt)
    if m and Path(m.group(1).strip()).exists():
        ins_summary = json.loads(Path(m.group(1).strip()).read_text(encoding="utf-8"))
    if ins_summary:
        rates = ins_summary["rates"]
        bad = []
        if rates.get("render_matches_eval_template", 0) < 1:
            bad.append("render != eval template")
        if rates.get("im_end_trained", 0) < 1:
            bad.append("<|im_end|> not trained")
        if rates.get("zero_trained_rows", 0) > 0:
            bad.append("rows with zero trained tokens")
        if rates.get("trained_equals_target_plus_im_end", 0) == 0 and rates.get("role_prefix_in_loss", 0) == 0:
            bad.append("trained span != target")
        # v2 invariants (absent from v1 inspector output -> treated as not checked, not as failing)
        if "target_contract_valid" in rates and rates["target_contract_valid"] < 1:
            bad.append("training target violates the answer contract")
        if "render_matches_serving_prompt_builder" in rates and rates["render_matches_serving_prompt_builder"] < 1:
            bad.append("training render != serving prompt builder")
        out["ours_inspect_labels"] = {"flagged": bool(bad), "why": bad, "rates": rates}
    else:
        out["ours_inspect_labels"] = {"flagged": c.get("inspect_labels", {}).get("exit") not in (0, None),
                                      "exit": c.get("inspect_labels", {}).get("exit"), "why": ["inspector crashed"]}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--phase", default="phase6")
    args = p.parse_args()
    run_dir, log = setup_run(args.phase, "fault_inject", args)
    rows, heldout = read_jsonl(TRAINVAL), read_jsonl(HELDOUT)
    base_cfg = BASE_CFG.read_text(encoding="utf-8")
    fdir = REPO_ROOT / "faults"
    fdir.mkdir(exist_ok=True)

    log.info("== clean reference")
    clean_run = run_checks(BASE_CFG, "clean", args.phase, log)
    clean = json.loads((clean_run / "summary.json").read_text(encoding="utf-8"))["checks"]
    results = {"clean": {"run": str(clean_run.relative_to(REPO_ROOT)), **interpret(clean_run, None),
                         "ours_leak_check": leak_check(rows)}}
    for fid, desc, frows, repl in faults(rows, heldout):
        if args.only and fid not in args.only:
            continue
        d = fdir / fid
        d.mkdir(parents=True, exist_ok=True)
        data_path = d / "trainval_soup.jsonl"
        write_jsonl(data_path, frows)
        cfg = base_cfg.replace("  train: data_smoke/processed/trainval_soup.jsonl",
                               f"  train: faults/{fid}/trainval_soup.jsonl")
        for old, new in repl:
            assert cfg.count(old) == 1, (fid, old)
            cfg = cfg.replace(old, new)
        cfg = cfg.replace("output: outputs_smoke/real", f"output: outputs_smoke/faults/{fid}")
        cfg_path = d / "config.yaml"
        cfg_path.write_text(cfg, encoding="utf-8", newline="\n")
        log.info("== %s: %s", fid, desc)
        run = run_checks(cfg_path, fid, args.phase, log)
        res = interpret(run, clean)
        res["ours_leak_check"] = leak_check(frows)
        results[fid] = {"description": desc, "run": str(run.relative_to(REPO_ROOT)), **res}
        log.info("   %s", {k: v["flagged"] for k, v in res.items() if isinstance(v, dict) and "flagged" in v})
    write_json(run_dir / "results.json", results)
    write_summary(run_dir, {"faults": {k: {kk: vv.get("flagged") for kk, vv in v.items()
                                           if isinstance(vv, dict) and "flagged" in vv}
                                       for k, v in results.items()}}, log)


if __name__ == "__main__":
    main()
