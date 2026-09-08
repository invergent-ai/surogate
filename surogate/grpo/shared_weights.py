"""Describe resident training weights for native serving without copying the base."""

from __future__ import annotations

import json
import math
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
    dtype: str = "BF16"
    cast_source: bool = False


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


def qwen3_5_bindings(config: dict) -> tuple[Binding, ...]:
    """Keep Qwen3.5's native query/gate, convolution and FP32 control storage."""
    from surogate.serve.convert.common.qwen3_5 import geometry_from_config

    g = geometry_from_config(config)
    c, m, d = g.hidden, g.intermediate, g.head_dim
    hf_root = "model.language_model" if "text_config" in config else "model"
    embed = f"{hf_root}.embed_tokens.weight"
    result = [
        Binding("text/token_embedding", "embedding", (g.vocab, c), ((embed, (g.vocab, c)),)),
        Binding("text/final_norm", "final_norm", (c,), ((f"{hf_root}.norm.weight", (c,)),)),
        Binding("text/output_head", "embedding" if g.tied_embeddings else "lm_head", (g.vocab, c),
                ((embed if g.tied_embeddings else "lm_head.weight", (g.vocab, c)),)),
    ]
    for layer, kind in enumerate(g.layer_types):
        obj, param, hf = f"text/layers/{layer}/", f"blocks[{layer}].", f"{hf_root}.layers.{layer}."

        def add(name, field, shape, source, *, start=0, dtype="BF16", cast=False):
            result.append(Binding(obj + name, param + field, shape, ((hf + source, shape),),
                                  start, dtype, cast))

        add("input_norm", "ln1_weight", (c,), "input_layernorm.weight")
        add("post_attention_norm", "ln2_weight", (c,), "post_attention_layernorm.weight")
        add("mlp/gate", "mlp_up_weight", (m, c), "mlp.gate_proj.weight", start=m)
        add("mlp/up", "mlp_up_weight", (m, c), "mlp.up_proj.weight")
        add("mlp/down", "mlp_down_weight", (c, m), "mlp.down_proj.weight")
        if kind == "full_attention":
            for name, field, shape, source in (
                ("query_gate", "full_q_proj_weight", (2 * g.query_size, c), "q_proj.weight"),
                ("key", "full_k_proj_weight", (g.kv_size, c), "k_proj.weight"),
                ("value", "full_v_proj_weight", (g.kv_size, c), "v_proj.weight"),
                ("query_norm", "q_norm_weight", (d,), "q_norm.weight"),
                ("key_norm", "k_norm_weight", (d,), "k_norm.weight"),
                ("output", "full_out_weight", (c, g.query_size), "o_proj.weight"),
            ):
                add("attention/" + name, field, shape, "self_attn." + source)
        else:
            for name, field, shape, source in (
                ("query_key_value", "lin_in_proj_qkv_weight", (g.convolution_dim, c), "in_proj_qkv.weight"),
                ("z", "lin_in_proj_z_weight", (g.value_dim, c), "in_proj_z.weight"),
                ("a_projection", "lin_in_proj_a_weight", (g.gdn_value_heads, c), "in_proj_a.weight"),
                ("b_projection", "lin_in_proj_b_weight", (g.gdn_value_heads, c), "in_proj_b.weight"),
                ("convolution_taps", "lin_conv_weight", (g.convolution_dim, 1, g.gdn_conv_kernel), "conv1d.weight"),
                ("output", "lin_out_weight", (c, g.value_dim), "out_proj.weight"),
            ):
                add("gdn/" + name, field, shape, "linear_attn." + source)
            add("gdn/norm", "lin_norm_weight", (g.gdn_value_head_dim,), "linear_attn.norm.weight", cast=True)
            add("gdn/a_log", "lin_A_log", (g.gdn_value_heads,), "linear_attn.A_log", dtype="FP32", cast=True)
            add("gdn/dt_bias", "lin_dt_bias", (g.gdn_value_heads,), "linear_attn.dt_bias", dtype="FP32", cast=True)
    return tuple(result)


def shared_family(config: dict) -> str:
    architectures = config.get("architectures", [])
    if architectures == ["Qwen3ForCausalLM"]:
        family = "qwen3"
    elif architectures in (["Qwen3_5ForCausalLM"], ["Qwen3_5ForConditionalGeneration"]):
        family = "qwen3_5"
    else:
        raise ValueError("native GRPO colocate supports BF16 dense Qwen3 and Qwen3.5 safetensors checkpoints")
    text = config.get("text_config", config)
    if config.get("quantization_config") or text.get("quantization_config") or text.get("num_experts", 0):
        raise ValueError("native GRPO colocate requires an unquantized dense checkpoint")
    return family


def write_shared_artifact(model_dir: str | Path, output: str | Path) -> tuple[Binding, ...]:
    """Write a small index and tokenizer resources; tensor bytes stay in the checkpoint.

    The normal loader can use the index as an independent BF16 baseline. The
    shared loader instead binds every tensor to the trainer, with zero uploads.
    Projection values are never read. Small Qwen3.5 control vectors whose source
    dtype differs from training are cast into the index for the independent loader.
    Shared serving still borrows those vectors directly from the trainer.
    """
    root = Path(model_dir).resolve()
    config = json.loads((root / "config.json").read_text())
    family = shared_family(config)
    bindings = qwen3_bindings(config) if family == "qwen3" else qwen3_5_bindings(config)
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
    specs, payloads = [], {}
    for binding in bindings:
        runs = []
        for name, shape in binding.sources:
            source_id, offset, entry = tensors[name]
            dtype = "F32" if binding.dtype == "FP32" else binding.dtype
            if tuple(entry["shape"]) != shape or (entry["dtype"] != dtype and not
                    (binding.cast_source and entry["dtype"] in ("BF16", "F32"))):
                raise ValueError(f"shared {family} requires {binding.dtype} {name} with shape {shape}")
            start, end = entry["data_offsets"]
            expected_bytes = math.prod(shape) * (4 if entry["dtype"] == "F32" else 2)
            if start < 0 or end - start != expected_bytes or offset + end > external[source_id - 1][1]:
                raise ValueError(f"invalid checkpoint tensor offsets: {name}")
            if entry["dtype"] != dtype:
                import torch
                with open(external[source_id - 1][0], "rb") as stream:
                    stream.seek(offset + start)
                    raw = bytearray(stream.read(end - start))
                vector = torch.frombuffer(raw, dtype=torch.float32 if entry["dtype"] == "F32" else torch.bfloat16)
                payloads[binding.name] = vector.to(torch.float32 if dtype == "F32" else torch.bfloat16).view(torch.uint8).numpy().tobytes()
                continue
            runs.append((source_id, offset + start, end - start))
        specs.append(TensorSpec(binding.name, binding.shape, binding.dtype, "contiguous-le-v1", tuple(runs)))
    if family == "qwen3":
        resources = load_resources(root)
        g, _ = validate_config(config)
        geometry, layer_types = dense_geometry(g, token_domain=tokenizer_domain(root)), ["full_attention"] * g.layers
    else:
        from surogate.serve.convert.qwen3_5.convert import load_resources as hybrid_resources
        from surogate.serve.convert.common.qwen3_5 import geometry_from_config, geometry_block
        resources = hybrid_resources(root)
        g = geometry_from_config(config)
        geometry, layer_types = geometry_block(g, token_domain=tokenizer_domain(root), mtp=False), g.layer_types
    specs.extend(ResourceSpec(r.name, "raw-bytes-v1", len(r.data)) for r in resources)
    with ArtifactWriter(output, ArtifactIdentity(family, inventory.WEIGHTS_ID, family),
                        specs, external=external,
                        geometry=geometry, layer_types=layer_types) as writer:
        for name, data in payloads.items():
            writer.write(name, data)
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
        dtype = torch.float32 if binding.dtype == "FP32" else torch.bfloat16
        if tuple(tensor.shape) != binding.shape or not tensor.is_contiguous() or tensor.dtype != dtype:
            raise ValueError(f"trainer tensor cannot be shared as {binding.name}: {tensor.shape}, {tensor.dtype}")
        result[binding.name] = tensor
    return result


def adapter_modules(trainer, scale: float):
    """Snapshot only the adapter into serving's banks; all conversion stays on GPU."""
    import re
    import torch

    weights = {name: torch.from_dlpack(value) for name, value in trainer.get_lora_weights(0).items()}
    pattern = re.compile(r"base_model\.model\.model\.(?:language_model\.)?layers\.(\d+)\.(?:self_attn|linear_attn|mlp)\.([a-z_]+)\.lora_A\.weight")
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
        raise ValueError("shared serving requires supported dense Qwen LoRA modules")
    torch.cuda.synchronize(next(iter(weights.values())).device)
    return modules
