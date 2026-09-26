"""NVFP4 post-training quantisation of a Gemma 4 mixture's routed experts, with NVIDIA ModelOpt.

This is the recipe NVIDIA used for `nvidia/Gemma-4-26B-A4B-NVFP4` -- ModelOpt, NVFP4 weights and
activations (W4A4), block 16, max calibration, the routed experts and nothing else -- applied to
a checkpoint of your own with a calibration corpus of your own. The output is an ordinary ModelOpt
Hugging Face export (per-expert `gate_proj` / `up_proj` / `down_proj` with `weight_scale`,
`weight_scale_2` and `input_scale`), which `surogate serve` converts to the `routed-nvfp4` profile
(`surogate/serve/convert/gemma4_moe/exports/routed_nvfp4.py`).

Calibrate on the prompts the model will serve. For a decision model that is decision prompts in
the serving protocol, from training data -- never benchmark rows.

    python -m surogate.serve.tools.nvfp4.modelopt_ptq --model rune-v3 --out rune-v3-nvfp4 \\
        --calibration corpus.txt --split bos

`--split bos` reads a corpus of rendered chats, each starting with `<bos>` (what
`build_imatrix_corpus` writes); `--split blank` reads one sample per blank-line-separated
paragraph. Needs `nvidia-modelopt` (0.47 or newer knows Gemma 4's fused experts) and a GPU with
room for the BF16 model.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import torch

#: The frontend files the export does not write but serving needs.
_CARRIED = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "chat_template.json",
            "generation_config.json", "processor_config.json", "preprocessor_config.json",
            "special_tokens_map.json", "tokenizer.model")


def load_prompts(path: Path, split: str, limit: int | None) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if split == "bos":
        prompts = ["<bos>" + part for part in text.split("<bos>") if part.strip()]
    else:
        prompts = [part.strip() for part in text.split("\n\n") if part.strip()]
    return prompts[:limit] if limit else prompts


def nvfp4_config(mtq, algorithm: str, scope: str) -> dict:
    """ModelOpt's NVFP4 config for `scope`, with the per-expert quantizer lists named too.

    `experts` is ModelOpt's experts-only config -- NVIDIA's recipe for this model. `all` is its
    default config: every text Linear (attention, the dense feed-forward, the experts) but the
    router, the head and the vision tower. Max calibration collects its statistics with
    quantisation off, so the experts come out the same either way; `all` exports the dense
    modules quantised as well, so one run can serve every scope (the converter serves the modules
    an export quantised, and a checkpoint can be assembled from the parts).

    ModelOpt's fused-experts wrapper keeps one weight quantizer per expert in a ModuleList
    (`gate_up_proj_weight_quantizers.<e>`), which `*weight_quantizer` does not match; the extra
    patterns make sure every expert is quantised, and `summarize` checks it.
    """
    cfg = copy.deepcopy(mtq.NVFP4_EXPERTS_ONLY_CFG if scope == "experts" else mtq.NVFP4_DEFAULT_CFG)
    entries = cfg["quant_cfg"]
    nvfp4 = None
    for entry in entries if isinstance(entries, list) else []:
        if entry.get("quantizer_name") in ("*.experts.*weight_quantizer", "*weight_quantizer"):
            nvfp4 = entry["cfg"]
    if nvfp4 is None:
        raise RuntimeError("this ModelOpt has no NVFP4 experts-only config shaped as expected")
    extra = [
        {"quantizer_name": "*.experts.*weight_quantizers*", "cfg": copy.deepcopy(nvfp4)},
        {"quantizer_name": "*.experts.*input_quantizer", "cfg": copy.deepcopy(nvfp4)},
    ]
    # After the enabling entries and before the disabling ones, so the router and the vision
    # tower stay excluded.
    index = next(i for i, e in enumerate(entries) if e.get("enable") is False and e["quantizer_name"] != "*")
    cfg["quant_cfg"] = entries[:index] + extra + entries[index:]
    cfg["algorithm"] = algorithm
    return cfg


def keep_source_config(source: Path, out: Path) -> None:
    """The export's `config.json` as the source's, plus the export's `quantization_config`.

    Quantisation changes no architecture field, but the export re-serialises the config with the
    installed transformers, which can drop values it considers defaults (Gemma 4's
    `global_head_dim` and `num_global_key_value_heads` under 5.17) or respell others. The
    converters read the checkpoint's config as its authority, so the source's is kept verbatim.
    """
    exported = json.loads((out / "config.json").read_text())
    config = json.loads((source / "config.json").read_text())
    config["quantization_config"] = exported["quantization_config"]
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def summarize(model) -> dict[str, int]:
    """How many quantizers are enabled, by kind, and where."""
    from modelopt.torch.quantization.nn import TensorQuantizer

    counts = {"enabled": 0, "disabled": 0, "enabled_outside_experts": 0,
              "expert_weight_quantizers": 0, "expert_input_quantizers": 0}
    for name, module in model.named_modules():
        if not isinstance(module, TensorQuantizer):
            continue
        if module.is_enabled:
            counts["enabled"] += 1
            if ".experts." not in name:
                counts["enabled_outside_experts"] += 1
            elif "weight_quantizer" in name:
                counts["expert_weight_quantizers"] += 1
            elif "input_quantizer" in name:
                counts["expert_input_quantizers"] += 1
        else:
            counts["disabled"] += 1
    return counts


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--split", choices=("bos", "blank"), default="bos")
    parser.add_argument("--limit", type=int, default=None, help="use the first N samples")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--algorithm", default="max", help="ModelOpt calibration algorithm")
    parser.add_argument("--scope", choices=("experts", "all"), default="experts",
                        help="experts: the routed experts only (NVIDIA's recipe); all: every text Linear "
                             "but the router and the head")
    args = parser.parse_args(argv)

    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite a checkpoint")
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", experts_implementation="eager")
    model.eval()
    prompts = load_prompts(args.calibration, args.split, args.limit)
    lengths = []

    def forward_loop(m):
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False,
                                truncation=True, max_length=args.max_length).input_ids.to("cuda")
                lengths.append(int(ids.shape[1]))
                m(input_ids=ids, logits_to_keep=1)
                if (i + 1) % 100 == 0:
                    print(f"calibrated {i + 1}/{len(prompts)}", flush=True)

    cfg = nvfp4_config(mtq, args.algorithm, args.scope)
    model = mtq.quantize(model, cfg, forward_loop)
    counts = summarize(model)
    print("quantizers:", counts, flush=True)
    if args.scope == "experts" and counts["enabled_outside_experts"]:
        raise SystemExit("quantizers outside the experts are enabled; the recipe quantises experts only")
    text = model.config.get_text_config()
    expected = text.num_hidden_layers * text.num_experts * 2
    if counts["expert_weight_quantizers"] < expected:
        raise SystemExit(f"only {counts['expert_weight_quantizers']} of {expected} expert weight "
                         "quantizers are enabled; the config did not reach every expert")
    export_hf_checkpoint(model, export_dir=str(args.out))
    keep_source_config(args.model, args.out)
    for name in _CARRIED:
        source = args.model / name
        if source.is_file() and not (args.out / name).exists():
            shutil.copy2(source, args.out / name)
    summary = {
        "tool": "surogate.serve.tools.nvfp4.modelopt_ptq",
        "source": str(args.model),
        "calibration": str(args.calibration),
        "samples": len(prompts),
        "calibration_tokens": sum(lengths),
        "algorithm": args.algorithm,
        "recipe": ("NVFP4 W4A4, block 16, routed experts only (ModelOpt NVFP4_EXPERTS_ONLY_CFG)"
                   if args.scope == "experts" else
                   "NVFP4 W4A4, block 16, every text Linear but router/head (ModelOpt NVFP4_DEFAULT_CFG)"),
        "quantizers": counts,
        "seconds": round(time.time() - started, 1),
    }
    (args.out / "surogate_ptq.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
