"""Assemble an NVFP4 checkpoint of a chosen scope from a full ModelOpt export and its BF16 source.

ModelOpt's max calibration collects its statistics with quantisation off, so an export that
quantised every text Linear holds the same routed experts -- and the same scales for every other
module -- as one that quantised fewer. This takes such an export and the BF16 checkpoint it was
made from and writes a checkpoint whose quantised modules are exactly the chosen scope: the
modules in scope keep the export's NVFP4 words and scales, every other module gets the BF16
source's weight back, and `quantization_config.ignore` names what was left alone. One
calibration run then serves every scope the accuracy check compares.

Scopes (each includes the ones before it):

    experts    the routed experts (NVIDIA's recipe for this model)
    mlp        + the dense feed-forward beside them (gate, up, down)
    attention  + the attention projections (query, key, value, output)

    python -m surogate.serve.tools.nvfp4.assemble --export rune-v3-nvfp4-all --source rune-v3 \\
        --scope mlp --out rune-v3-nvfp4-mlp

`--weight-search mse` re-quantises the routed experts' weights from the BF16 source: the same
global scales (so the activation scales and alphas the calibration chose still hold), but each
16-value block takes whichever of a few candidate scales around `amax / 6` reconstructs it with
the smallest squared error, instead of `amax / 6` itself. The block scale is the one number NVFP4
adds per 16 weights, and the plain rule spends its whole range on the block's largest value.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

SCOPES = ("experts", "mlp", "attention")
_QUANTISED = ("weight_scale", "weight_scale_2", "input_scale", "weight_packed", "weight_global_scale",
              "input_global_scale")
_DENSE = {
    "mlp": re.compile(r"\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)$"),
    "attention": re.compile(r"\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"),
}


def module_of(name: str) -> tuple[str, str]:
    module, _, leaf = name.rpartition(".")
    return module, leaf


def kept_quantised(module: str, scope: str) -> bool:
    """Whether a quantised module of the export stays quantised in this scope."""
    if ".experts." in module:
        return True
    order = SCOPES.index(scope)
    for index, kind in enumerate(SCOPES[1:], start=1):
        if _DENSE[kind].search(module):
            return order >= index
    return False


#: Block-scale multipliers the MSE search tries around `amax / 6` (1.0 is the plain rule, so the
#: search never does worse than it).
MSE_CANDIDATES = (1.0, 0.75, 0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.2, 1.35, 1.5)
_EXPERT = re.compile(r"^(?P<prefix>.*\.layers\.\d+\.experts)\.(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)$")


def expert_source(module: str, source_index: dict, source: Path, cache: dict):
    """The BF16 weight of one expert projection, sliced from the source's stacked tensors."""
    import torch

    match = _EXPERT.match(module)
    expert, proj = int(match.group("expert")), match.group("proj")
    name = match.group("prefix") + (".down_proj" if proj == "down_proj" else ".gate_up_proj")
    if name not in cache:
        cache.clear()
        with safe_open(str(source / source_index[name]), "pt") as handle:
            cache[name] = handle.get_tensor(name)
    stacked = cache[name]
    if proj == "down_proj":
        return stacked[expert].to(torch.float32)
    half = stacked.shape[1] // 2
    rows = slice(0, half) if proj == "gate_proj" else slice(half, 2 * half)
    return stacked[expert, rows].to(torch.float32)


def mse_requantise(tensors: dict, module: str, source_index: dict, source: Path, cache: dict) -> None:
    """Replace one expert projection's codes and block scales with the MSE-searched ones."""
    import torch

    from surogate.serve.convert.common import nvfp4

    values = expert_source(module, source_index, source, cache)
    divisor = 1.0 / float(tensors[module + ".weight_scale_2"].to(torch.float32))
    packed, scales = nvfp4.quantize(values, divisor, scale_candidates=MSE_CANDIDATES)
    if tuple(packed.shape) != tuple(tensors[module + ".weight"].shape):
        raise ValueError(f"{module}: requantised codes {tuple(packed.shape)} do not match the export")
    tensors[module + ".weight"] = packed
    tensors[module + ".weight_scale"] = scales.view(torch.float8_e4m3fn)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", required=True, type=Path, help="ModelOpt export that quantised every text Linear")
    parser.add_argument("--source", required=True, type=Path, help="the BF16 checkpoint it was made from")
    parser.add_argument("--scope", required=True, choices=SCOPES)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--weight-search", choices=("none", "mse"), default="none",
                        help="re-quantise the routed experts' weights with an MSE block-scale search")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite a checkpoint")

    index = json.loads((args.export / "model.safetensors.index.json").read_text())["weight_map"]
    source_index = json.loads((args.source / "model.safetensors.index.json").read_text())["weight_map"]
    quantised = {module_of(name)[0] for name in index if module_of(name)[1] in _QUANTISED}
    dropped = sorted(m for m in quantised if not kept_quantised(m, args.scope))
    restored = {m + ".weight" for m in dropped}
    missing = sorted(name for name in restored if name not in source_index)
    if missing:
        raise SystemExit(f"the BF16 source lacks {len(missing)} weights the scope needs, e.g. {missing[:3]}")

    args.out.mkdir(parents=True)
    weight_map: dict[str, str] = {}
    by_shard: dict[str, list[str]] = {}
    for name, shard in index.items():
        module, _ = module_of(name)
        if module in dropped:
            continue  # its NVFP4 words and scales; the BF16 weight replaces them
        by_shard.setdefault(shard, []).append(name)
    searched = 0
    source_cache: dict = {}
    for shard, names in sorted(by_shard.items()):
        with safe_open(str(args.export / shard), "pt") as handle:
            tensors = {name: handle.get_tensor(name) for name in names}
        if args.weight_search == "mse":
            experts = sorted({module_of(n)[0] for n in names
                              if module_of(n)[1] == "weight_scale_2" and _EXPERT.match(module_of(n)[0])})
            for module in experts:
                mse_requantise(tensors, module, source_index, args.source, source_cache)
                searched += 1
        save_file(tensors, str(args.out / shard), metadata={"format": "pt"})
        weight_map.update({name: shard for name in names})
    restored_shard = "model-bf16-restored.safetensors"
    tensors = {}
    for name in sorted(restored):
        with safe_open(str(args.source / source_index[name]), "pt") as handle:
            tensors[name] = handle.get_tensor(name)
    if tensors:
        save_file(tensors, str(args.out / restored_shard), metadata={"format": "pt"})
        weight_map.update({name: restored_shard for name in tensors})
    (args.out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": dict(sorted(weight_map.items()))}, indent=2) + "\n")

    config = json.loads((args.export / "config.json").read_text())
    quant = config.get("quantization_config") or {}
    ignore = list(quant.get("ignore") or [])
    ignore.extend(m for m in dropped if m not in ignore)
    quant["ignore"] = ignore
    config["quantization_config"] = quant
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    hf_quant = args.export / "hf_quant_config.json"
    if hf_quant.is_file():
        data = json.loads(hf_quant.read_text())
        exclude = list(data.get("quantization", {}).get("exclude_modules") or [])
        exclude.extend(m for m in dropped if m not in exclude)
        data.setdefault("quantization", {})["exclude_modules"] = exclude
        (args.out / "hf_quant_config.json").write_text(json.dumps(data, indent=4) + "\n")
    for item in args.export.iterdir():
        if item.is_file() and not item.name.endswith(".safetensors") and not (args.out / item.name).exists():
            shutil.copy2(item, args.out / item.name)
    summary = {"export": str(args.export), "source": str(args.source), "scope": args.scope,
               "quantised_modules": len(quantised) - len(dropped), "restored_bf16_modules": len(dropped),
               "weight_search": args.weight_search, "mse_requantised_experts": searched}
    (args.out / "surogate_assemble.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
