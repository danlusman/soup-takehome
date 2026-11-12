"""Phase 4: independent reference merge (plain transformers + peft), with per-phase memory.

Soup's `soup merge` is the primary merge (wrapped by memtrace.py). This script exists to
(1) measure merge peaks phase by phase (torch + RSS + disk) and
(2) produce an independent merged checkpoint so verify.py can diff Soup's merge against it.

Differences from `soup merge`, on purpose:
  * tokenizer, chat template and generation_config are copied from the BASE model, never from
    the adapter dir (an adapter dir can carry a chat_template override from training, D-15);
    the adapter's template is hashed and compared, mismatch is logged loudly
  * base is loaded in float16 explicitly and merged into fp16 (never into a 4-bit base)

Example:
  python scripts/merge_ref.py --adapter outputs/real --out outputs/merged_ref/real
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import setup_run, sha256_file, sha256_text, write_summary  # noqa: E402
from memtrace import MemTracker  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True)
    p.add_argument("--base", default=None, help="default: base_model_name_or_path from adapter_config.json")
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                   help="cpu = what soup merge does; cuda = measure the GPU merge peak")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--phase", default="phase4")
    p.add_argument("--runs-root", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    run_dir, log = setup_run(args.phase, f"merge_ref_{Path(args.adapter).name}", args, args.runs_root)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    acfg = json.loads((Path(args.adapter) / "adapter_config.json").read_text(encoding="utf-8"))
    base = args.base or acfg["base_model_name_or_path"]
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    res = {"base": base, "adapter": args.adapter, "adapter_config": acfg, "device": args.device}

    with MemTracker(run_dir, log) as mt:
        with mt.phase("load_base_fp16"):
            model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16, device_map=args.device)
        res["base_param_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
        with mt.phase("load_adapter"):
            model = PeftModel.from_pretrained(model, args.adapter)
        with mt.phase("merge_and_unload"):
            model = model.merge_and_unload()
        res["merged_param_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
        res["leftover_lora_params"] = [n for n, _ in model.named_parameters() if "lora_" in n]
        with mt.phase("save_merged"):
            model.save_pretrained(out, safe_serialization=True)
        del model
        with mt.phase("copy_tokenizer_template_genconfig_from_base"):
            tok = AutoTokenizer.from_pretrained(base)
            tok.save_pretrained(out)
            GenerationConfig.from_pretrained(base).save_pretrained(out)
        mt.snapshot_disk("after_save", [str(out)])

    base_tmpl = tok.chat_template or ""
    try:
        ad_tok = AutoTokenizer.from_pretrained(args.adapter)
        ad_tmpl = ad_tok.chat_template or ""
    except Exception as e:  # noqa: BLE001
        ad_tmpl, res["adapter_tokenizer_error"] = None, repr(e)
    res["template_sha_base"] = sha256_text(base_tmpl)
    res["template_sha_adapter_dir"] = sha256_text(ad_tmpl) if ad_tmpl is not None else None
    res["adapter_template_differs_from_base"] = ad_tmpl is not None and ad_tmpl != base_tmpl
    if res["adapter_template_differs_from_base"]:
        log.warning("ADAPTER DIR CARRIES A DIFFERENT CHAT TEMPLATE THAN THE BASE (train-time override?)")
    res["files"] = {f.name: {"bytes": f.stat().st_size, "sha256": sha256_file(f)}
                    for f in sorted(out.iterdir()) if f.is_file()}
    res["merged_dir_bytes"] = sum(v["bytes"] for v in res["files"].values())
    res["phases"] = [{"phase": e["phase"], "wall_s": e.get("wall_s"),
                      "max_allocated": e.get("torch_after", {}).get("max_allocated"),
                      "rss_after": e.get("rss_after")} for e in mt.events if "torch_after" in e]
    write_summary(run_dir, res, log)


if __name__ == "__main__":
    main()
