"""Phase 6: merge / quantization / inference faults on the CPU smoke artifacts (Qwen2.5-0.5B).

M1 adapter_config.json lora_alpha edited after training (32 -> 16). `soup merge` reads the edited value and
   merges at half scale without complaint. Our verify.py weight + T0 checks read the SAME file, so they stay
   self-consistent. Only a comparison against train-time evidence (Soup's telemetry/trainer_state) catches it.
Q2 one base weight that is finite in bf16 but > fp16 max (65504). `soup merge --dtype float16` casts it to inf.
   verify.py's weight report counts non-finite elements after the cast.
I1 generation_config.json eos_token_id stripped of <|im_end|>. A user calling model.generate() with shipped
   defaults never stops at the end of the turn; our eval_task.py adds <|im_end|> to the stop set itself
   (eos_ids_for) and would NOT notice.
I2 serving without the chat template (raw prompt). soup ship's renderer silently falls back to the raw prompt
   when a template fails to render (soup_cli/utils/live_eval.py:_render_prompt_template).

Each fault records: did `soup merge` exit 0, what our checks say, and the direct observation.

Example:
  python scripts/fault_inject_artifacts.py --phase phase6
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, read_jsonl, setup_run, write_json, write_summary  # noqa: E402

SMOKE = REPO_ROOT / "outputs_smoke"
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
FA = SMOKE / "faults_artifacts"


def soup_merge(adapter: Path, out: Path, base: str | None = None) -> tuple[int, str]:
    if out.exists():
        shutil.rmtree(out)
    cmd = [sys.executable, "-m", "soup_cli", "merge", "--adapter", str(adapter), "--output", str(out),
           "--dtype", "float16"] + (["--base", base] if base else [])
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**__import__("os").environ, "COLUMNS": "200", "NO_COLOR": "1"})
    return r.returncode, (r.stdout + r.stderr)[-3000:]


def m1(run_dir, log):
    import verify
    from huggingface_hub import snapshot_download
    src = SMOKE / "real"
    bad = FA / "M1_adapter_alpha16"
    if bad.exists():
        shutil.rmtree(bad)
    shutil.copytree(src, bad, ignore=shutil.ignore_patterns("checkpoint-*"))
    cfgp = bad / "adapter_config.json"
    cfg = json.loads(cfgp.read_text(encoding="utf-8"))
    trained_alpha = cfg["lora_alpha"]
    cfg["lora_alpha"] = trained_alpha // 2
    cfgp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    rc, out = soup_merge(bad, FA / "M1_merged")
    (run_dir / "M1_soup_merge.txt").write_text(out, encoding="utf-8")
    base_dir = Path(snapshot_download(BASE, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]))
    w = verify.weight_report(base_dir, bad, FA / "M1_merged", None, log)
    # train-time truth: Soup's own trainer_state / our telemetry recorded the alpha used in training
    tele = sorted((REPO_ROOT / "runs" / "smoke").glob("memtrace_train_smoke_real_*/telemetry.jsonl"))
    train_alpha = None
    if tele:
        for line in tele[-1].read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            if e.get("event") == "train_begin":
                train_alpha = e["peft_config"]["default"]["lora_alpha"]
    return {"soup_merge_exit": rc, "trained_alpha": trained_alpha, "edited_alpha": cfg["lora_alpha"],
            "verify_scale_used": w["scale"], "verify_consistency_resid_median": w["consistency_resid_median"],
            "verify_frac_changed": w["frac_changed_all_adapted"],
            "verify_flags_it": False,  # consistency is computed against the same (edited) config
            "telemetry_alpha": train_alpha,
            "telemetry_cross_check_flags_it": train_alpha is not None and train_alpha != cfg["lora_alpha"]}


def q2(run_dir, log):
    import torch
    import verify
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file
    src = Path(snapshot_download(BASE, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]))
    bad_base = FA / "Q2_base_overflow"
    if bad_base.exists():
        shutil.rmtree(bad_base)
    shutil.copytree(src, bad_base)
    st = next(bad_base.glob("*.safetensors"))
    sd = load_file(str(st))
    key = "model.layers.0.mlp.down_proj.weight"
    sd[key] = sd[key].clone()
    sd[key][0, 0] = torch.tensor(1.0e5, dtype=sd[key].dtype)  # finite in bf16, > fp16 max
    save_file(sd, str(st), metadata={"format": "pt"})
    del sd
    rc, out = soup_merge(SMOKE / "real", FA / "Q2_merged", base=str(bad_base))
    (run_dir / "Q2_soup_merge.txt").write_text(out, encoding="utf-8")
    w = verify.weight_report(bad_base, SMOKE / "real", FA / "Q2_merged", None, log)
    return {"soup_merge_exit": rc, "injected": f"{key}[0,0] = 1e5 (bf16)",
            "verify_nonfinite_after_cast": w["base_to_merged_dtype_cast_nonfinite_elements"],
            "verify_flags_it": w["base_to_merged_dtype_cast_nonfinite_elements"] > 0,
            "soup_merge_output_mentions_inf_or_overflow": any(s in out.lower() for s in ("inf", "overflow", "nan"))}


def i1_i2(run_dir, log):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from task_format import build_messages, parse_response
    merged = SMOKE / "merged" / "real"
    bad = FA / "I1_genconfig_no_im_end"
    if bad.exists():
        shutil.rmtree(bad)
    shutil.copytree(merged, bad)
    gcp = bad / "generation_config.json"
    gc = json.loads(gcp.read_text(encoding="utf-8"))
    gc["eos_token_id"] = [151643]  # <|endoftext|> only
    gcp.write_text(json.dumps(gc, indent=2), encoding="utf-8")
    rows = read_jsonl(REPO_ROOT / "data_smoke" / "processed" / "heldout.jsonl")[:3]
    tok = AutoTokenizer.from_pretrained(str(bad))
    model = AutoModelForCausalLM.from_pretrained(str(bad), dtype=torch.bfloat16).eval()
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    res = {"I1": [], "I2": []}
    for r in rows:
        p = tok.apply_chat_template(build_messages(r["text"]), tokenize=False, add_generation_prompt=True)
        enc = tok(p, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=40, do_sample=False,
                                 generation_config=GenerationConfig.from_pretrained(str(bad)),
                                 temperature=None, top_p=None, top_k=None)
        gen = out[0, enc["input_ids"].shape[1]:].tolist()
        res["I1"].append({"n_new_tokens": len(gen), "hit_max_new_tokens": len(gen) >= 40,
                          "contains_im_end": im_end in gen,
                          "text": tok.decode(gen, skip_special_tokens=False)[:200]})
        # I2: raw prompt, no chat template
        enc2 = tok(build_messages(r["text"])[0]["content"], return_tensors="pt")
        with torch.no_grad():
            out2 = model.generate(**enc2, max_new_tokens=30, do_sample=False, temperature=None, top_p=None,
                                  top_k=None, pad_token_id=tok.pad_token_id)
        t2 = tok.decode(out2[0, enc2["input_ids"].shape[1]:], skip_special_tokens=True)
        res["I2"].append({"strict_valid": parse_response(t2)["strict_valid"], "text": t2[:200]})
    res["I1_summary"] = {"rows": len(rows), "ran_to_max_new_tokens": sum(x["hit_max_new_tokens"] for x in res["I1"]),
                         "our_eval_would_flag": False,
                         "why_ours_misses": "eval_task.eos_ids_for() always adds <|im_end|> to the stop set"}
    res["I2_summary"] = {"strict_valid_rate": sum(x["strict_valid"] for x in res["I2"]) / len(rows)}
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", default=["M1", "Q2", "I1"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--phase", default="phase6")
    a = p.parse_args()
    run_dir, log = setup_run(a.phase, "fault_inject_artifacts", a)
    FA.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, fn in (("M1", m1), ("Q2", q2), ("I1", i1_i2)):
        if name not in a.only:
            continue
        log.info("== %s", name)
        try:
            res[name] = fn(run_dir, log)
        except Exception as e:  # noqa: BLE001  keep failures as results
            log.exception("%s failed", name)
            res[name] = {"error": repr(e)}
        log.info("%s -> %s", name, json.dumps(res[name], ensure_ascii=False, default=str)[:600])
        write_json(run_dir / "results.json", res)
    write_summary(run_dir, res, log)


if __name__ == "__main__":
    main()
