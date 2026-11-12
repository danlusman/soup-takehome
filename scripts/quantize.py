"""Phase 4: produce the 4-bit artifact (bitsandbytes NF4, double-quant, fp16 compute) from an
already-merged fp16 checkpoint, using Soup's own writer `soup_cli.utils.save_formats.merge_4bit`.

Why call Soup's function instead of `soup merge --save-format 4bit`: the CLI re-merges the adapter
into a temp fp16 dir and quantizes that, so the 4-bit file would come from a different merge than
the fp16 one we verify. Calling merge_4bit on the verified merged dir quantizes exactly that.

Also used for the noise floor: `--src` = a local copy of the BASE model -> base NF4.
(merge_4bit refuses paths outside cwd, so the base is first snapshot-copied under outputs/.)

Records per-phase peaks (torch, RSS, nvidia-smi dumps), disk size, and a reload check (the saved
artifact loads and its 4-bit layer count is what we expect).

Example:
  python scripts/quantize.py --src outputs/merged/real --out outputs/nf4/real
  python scripts/quantize.py --src-hub Qwen/Qwen2.5-1.5B-Instruct --src outputs/base_fp16 --out outputs/nf4/base
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import setup_run, sha256_file, write_summary  # noqa: E402
from memtrace import MemTracker  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="merged fp16 dir (under cwd)")
    p.add_argument("--src-hub", default=None, help="if set, first snapshot this hub model into --src")
    p.add_argument("--out", required=True)
    p.add_argument("--no-double-quant", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--phase", default="phase4")
    p.add_argument("--runs-root", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    run_dir, log = setup_run(args.phase, f"quantize_{Path(args.out).name}", args, args.runs_root)
    import torch
    from soup_cli.utils.save_formats import merge_4bit

    src, out = Path(args.src), Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    res = {"src": str(src), "out": str(out), "format": "bitsandbytes NF4 (Soup merge_4bit)",
           "double_quant": not args.no_double_quant, "cuda": torch.cuda.is_available()}
    with MemTracker(run_dir, log) as mt:
        if args.src_hub and not src.exists():
            with mt.phase("snapshot_base"):
                from huggingface_hub import snapshot_download
                snapshot_download(args.src_hub, local_dir=str(src),
                                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "merges.txt"])
        mt.snapshot_disk("before", [str(src)])
        with mt.phase("merge_4bit_load_quantize_save"):
            merge_4bit(merged_dir=str(src), output_dir=str(out), forced=False, dtype="float16",
                       double_quant=not args.no_double_quant)
        mt.snapshot_disk("after", [str(src), str(out)])
        with mt.phase("reload_check"):
            from transformers import AutoModelForCausalLM
            m = AutoModelForCausalLM.from_pretrained(str(out), device_map="cuda" if res["cuda"] else "cpu")
            n4 = sum(1 for _, mod in m.named_modules() if type(mod).__name__ == "Linear4bit")
            res["reload_linear4bit_modules"] = n4
            res["reload_quantization_config"] = getattr(m.config, "quantization_config", None)
            res["reload_memory_footprint_bytes"] = m.get_memory_footprint()
            res["reload_non4bit_param_dtypes"] = sorted({str(p.dtype) for n, p in m.named_parameters()
                                                         if p.dtype != torch.uint8})
            del m
    res["files"] = {f.name: {"bytes": f.stat().st_size, "sha256": sha256_file(f)}
                    for f in sorted(out.iterdir()) if f.is_file()}
    res["out_dir_bytes"] = sum(v["bytes"] for v in res["files"].values())
    res["phases"] = [{"phase": e["phase"], "wall_s": e.get("wall_s"),
                      "max_allocated": e.get("torch_after", {}).get("max_allocated"),
                      "rss_after": e.get("rss_after")} for e in mt.events if "torch_after" in e]
    log.info("Linear4bit modules after reload: %s; dir bytes %d", res.get("reload_linear4bit_modules"),
             res["out_dir_bytes"])
    write_summary(run_dir, res, log)


if __name__ == "__main__":
    main()
