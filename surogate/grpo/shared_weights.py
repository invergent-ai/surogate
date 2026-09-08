"""Describe resident Qwen3 BF16 weights without reading or copying their values."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

from surogate.serve.artifact.container import ArtifactIdentity, ArtifactWriter, ResourceSpec, TensorSpec
from surogate.serve.convert.common.checkpoint import dense_geometry, tokenizer_domain
from surogate.serve.convert.qwen3 import inventory
from surogate.serve.convert.qwen3.convert import load_resources, validate_config


@dataclass(frozen=True)
class Binding:
    name: str
    parameter: str
    shape: tuple[int, ...]
    sources: tuple[tuple[str, tuple[int, ...]], ...]
    row_start: int = 0


def qwen3_bindings(config: dict) -> tuple[Binding, ...]:
    """The shared views, including the two halves of training's [up, gate] MLP."""
    g, _ = validate_config(config)
    c, m, d = g.hidden, g.intermediate, g.head_dim
    q, kv = g.query_heads * d, g.kv_heads * d
    embed = "model.embed_tokens.weight"
    tied = bool(config.get("tie_word_embeddings", False))
    result = [
        Binding("text/token_embedding", "embedding", (g.vocab, c), ((embed, (g.vocab, c)),)),
        Binding("text/final_norm", "final_norm", (c,), (("model.norm.weight", (c,)),)),
        Binding("text/output_head", "embedding" if tied else "lm_head", (g.vocab, c),
                ((embed if tied else "lm_head.weight", (g.vocab, c)),)),
    ]
    for layer in range(g.layers):
        obj, param, hf = f"text/layers/{layer}/", f"blocks[{layer}].", f"model.layers.{layer}."
        for name, field, shape, sources, start in (
            ("input_norm", "ln1_weight", (c,), (("input_layernorm.weight", (c,)),), 0),
            ("attention/query_key_value", "qkv_weight", (q + 2 * kv, c),
             (("self_attn.q_proj.weight", (q, c)), ("self_attn.k_proj.weight", (kv, c)),
              ("self_attn.v_proj.weight", (kv, c))), 0),
            ("attention/query_norm", "q_norm_weight", (d,), (("self_attn.q_norm.weight", (d,)),), 0),
            ("attention/key_norm", "k_norm_weight", (d,), (("self_attn.k_norm.weight", (d,)),), 0),
            ("attention/output", "out_weight", (c, q), (("self_attn.o_proj.weight", (c, q)),), 0),
            ("post_attention_norm", "ln2_weight", (c,), (("post_attention_layernorm.weight", (c,)),), 0),
            ("mlp/gate", "mlp_up_weight", (m, c), (("mlp.gate_proj.weight", (m, c)),), m),
            ("mlp/up", "mlp_up_weight", (m, c), (("mlp.up_proj.weight", (m, c)),), 0),
            ("mlp/down", "mlp_down_weight", (c, m), (("mlp.down_proj.weight", (c, m)),), 0),
        ):
            result.append(Binding(obj + name, param + field, shape,
                                  tuple((hf + source, dims) for source, dims in sources), start))
    return tuple(result)


def write_shared_artifact(model_dir: str | Path, output: str | Path) -> tuple[Binding, ...]:
    """Write a small index and tokenizer resources; tensor bytes stay in the checkpoint.

    The normal loader can use the index as an independent BF16 baseline. The
    shared loader instead binds every tensor to the trainer, with zero uploads.
    Only safetensors headers are read here, even for a sharded checkpoint.
    """
    root = Path(model_dir).resolve()
    config = json.loads((root / "config.json").read_text())
    bindings = qwen3_bindings(config)
    index = root / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    external, tensors = [], {}
    for source_id, filename in enumerate(files, 1):
        path = (root / filename).resolve()
        size = path.stat().st_size
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            if header_size > min(size - 8, 64 << 20):
                raise ValueError(f"invalid safetensors header in {path}")
            header = json.loads(stream.read(header_size))
        external.append((str(path), size))
        for name, entry in header.items():
            if name != "__metadata__":
                if name in tensors:
                    raise ValueError(f"duplicate checkpoint tensor: {name}")
                tensors[name] = (source_id, header_size + 8, entry)
    specs = []
    for binding in bindings:
        runs = []
        for name, shape in binding.sources:
            source_id, offset, entry = tensors[name]
            if entry["dtype"] != "BF16" or tuple(entry["shape"]) != shape:
                raise ValueError(f"shared Qwen3 requires BF16 {name} with shape {shape}")
            start, end = entry["data_offsets"]
            if start < 0 or end < start or offset + end > external[source_id - 1][1]:
                raise ValueError(f"invalid checkpoint tensor offsets: {name}")
            runs.append((source_id, offset + start, end - start))
        specs.append(TensorSpec(binding.name, binding.shape, "BF16", "contiguous-le-v1", tuple(runs)))
    resources = load_resources(root)
    specs.extend(ResourceSpec(r.name, "raw-bytes-v1", len(r.data)) for r in resources)
    geometry, _ = validate_config(config)
    with ArtifactWriter(output, ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID, "qwen3"),
                        specs, external=external,
                        geometry=dense_geometry(geometry, token_domain=tokenizer_domain(root)),
                        layer_types=["full_attention"] * geometry.layers) as writer:
        for resource in resources:
            writer.write(resource.name, resource.data)
    return bindings


def borrow_weights(trainer, bindings: tuple[Binding, ...]):
    import torch

    parameters = {name: torch.from_dlpack(value) for name, value in trainer.get_shared_base_weights().items()}
    result = {}
    for binding in bindings:
        tensor = parameters[binding.parameter]
        if binding.name.endswith(("/mlp/gate", "/mlp/up")):
            if tuple(tensor.shape) != (binding.shape[0] * 2, binding.shape[1]):
                raise ValueError(f"unexpected fused MLP shape: {binding.parameter}")
            tensor = tensor.narrow(0, binding.row_start, binding.shape[0])
        if tuple(tensor.shape) != binding.shape or not tensor.is_contiguous() or tensor.dtype != torch.bfloat16:
            raise ValueError(f"trainer tensor cannot be shared as {binding.name}: {tensor.shape}, {tensor.dtype}")
        result[binding.name] = tensor
    return result


def adapter_modules(trainer, scale: float):
    """Snapshot only the adapter into serving's banks; all conversion stays on GPU."""
    import re
    import torch

    weights = {name: torch.from_dlpack(value) for name, value in trainer.get_lora_weights(0).items()}
    pattern = re.compile(r"base_model\.model\.model\.layers\.(\d+)\.(?:self_attn|mlp)\.([a-z_]+)\.lora_A\.weight")
    modules, consumed = [], set()
    for name, tensor in weights.items():
        match = pattern.fullmatch(name)
        if match is None:
            continue
        b_name = name.replace(".lora_A.", ".lora_B.")
        modules.append(dict(layer=int(match[1]), module=match[2], scale=scale,
                            a=tensor.to(torch.bfloat16).contiguous(),
                            b=weights[b_name].to(torch.bfloat16).contiguous()))
        consumed.update((name, b_name))
    if not modules or consumed != weights.keys():
        raise ValueError("shared serving requires standard dense Qwen3 LoRA modules")
    torch.cuda.synchronize(next(iter(weights.values())).device)
    return modules
