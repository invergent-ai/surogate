"""Convert the NVFP4 safetensors release of a hyper-connected hybrid checkpoint.

The release is a ModelOpt `MIXED_PRECISION` export: the routed experts are NVFP4 with a
second-level scale per expert and projection, the n-gram table is FP8 with one scale, and
everything else is BF16. The artifact keeps the experts' words and calibration as they are, for
the TRT-LLM runner on the device; the dense projections become W8, as the GGUF path has them;
and the table becomes IQ4_NL, the format the n-gram kernel reads, quantised on the GPU with
llama.cpp's search. Kept at FP8 the table alone is 51 GB, which a 121 GB DGX Spark cannot hold
beside the 74 GB body.

The HF and engine conventions differ in one place: HF stores every hyper-connection, indexer and
n-gram RMS-norm weight zero-centred (the module computes `(1 + w) * x`), and the engine reads the
folded `1 + w`, as the GGUF carries them. The attention q/k norms are the exception the engine
shares with HF, and the GDN norm is not zero-centred in either.

The NextN draft head is converted unless `--no-mtp` asks otherwise. Its experts are FP8 with a
scale per 128 x 128 block in the release; they are dequantised and stored W8, as the GGUF path
stores them, so `--spec mtp` runs them through the same device MoE as any W8 layer. The vision
tower stays in the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import torch

from surogate.serve.artifact.container import ArtifactIdentity, ArtifactWriter
from surogate.serve.artifact.layouts import encode_direct, encode_nvfp4
from surogate.serve.convert.common import conversion as family_conversion
from surogate.serve.convert.common import iq4nl
from surogate.serve.convert.common.checkpoint import tokenizer_domain
from surogate.serve.convert.common.inventory import TensorSpec
from surogate.serve.convert.common.qwen4exp import geometry_block
from surogate.serve.convert.common.recipe import (
    Concat, Reshape, Slice, TensorRecipe, Transpose, attention_qproj_part, materialize_recipe,
    preflight_source_reader, source,
)
from surogate.serve.convert.common.safetensors import ShardReader

from . import inventory as inv

RECIPE_ID = "qwen4exp-modelopt-nvfp4-v1"
WEIGHTS_ID = "routed-nvfp4-hc-v1"

NVFP4 = "NVFP4"
BLOCK_SCALE_LAYOUT = "blockscale-k16-m128x4-v1"
GATE_UP_SUFFIX = "/mlp/routed_gate_up"
DOWN_SUFFIX = "/mlp/routed_down"
SCALE_SUFFIXES = ("_scale", "_act_scale", "_alpha")
_NVFP4_BLOCK = 16

_DIRECT = {inv.BF16: torch.bfloat16, inv.FP32: torch.float32, inv.I32: torch.int32}
_EXPERTS = re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.experts")
MTP_BLOCK = "mtp.layers.0."
MTP_EXPERTS = MTP_BLOCK + "mlp.experts"
MTP_ROUTED = ("mtp/layer/mlp/routed_gate_up", "mtp/layer/mlp/routed_down")
_FP8_BLOCK = 128
_NGRAM_TABLE = re.compile(r"model\.language_model\.layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding")


# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------


def text_config(config: Mapping, *, mtp: bool = False) -> dict:
    """The text tower's config as the geometry resolver reads it; the NextN head only if asked."""
    text = dict(config.get("text_config", config))
    text.update(architectures=["Qwen4ExpForCausalLM"], model_type="qwen4_exp",
                image_token_id=config.get("image_token_id", text.get("image_token_id")),
                mtp_num_hidden_layers=int(text.get("mtp_num_hidden_layers") or 0) if mtp else 0)
    return text


def validate_mtp(text: Mapping) -> None:
    """Refuse a NextN head the engine's draft path would run differently than the release.

    The engine carries one draft block, a full-attention trunk block fed by the shared token
    embedding and the trunk's last wide residual, at the trunk's RoPE.
    """
    if int(text.get("mtp_num_hidden_layers") or 0) != 1:
        raise ValueError("the engine's draft head is one block; mtp_num_hidden_layers must be 1")
    if text.get("mtp_use_dedicated_embeddings"):
        raise ValueError("a NextN head with its own embedding table is not supported")
    head = text.get("mtp") or {}
    if list(head.get("layer_types", ["full_attention"])) != ["full_attention"]:
        raise ValueError(f"the NextN block must be full attention, not {head.get('layer_types')}")
    if head.get("mtp_use_hidden_state_from_layer") is not None:
        raise ValueError("the NextN head must read the trunk's last hidden state")
    trunk = (text.get("rope_parameters") or {}).get("rope_theta", text.get("rope_theta"))
    if "rope_theta" in head and trunk is not None and float(head["rope_theta"]) != float(trunk):
        raise ValueError("the NextN block's RoPE differs from the trunk's, which the engine reuses")


def mtp_expert_quantization(config: Mapping) -> str | None:
    """`FP8_PB_WO` for 128 x 128 block-scaled FP8 draft experts, None for BF16 ones."""
    quantized = (config.get("quantization_config") or {}).get("quantized_layers") or {}
    entry = quantized.get(MTP_EXPERTS)
    if entry is None:
        return None
    if entry.get("quant_algo") == "FP8_PB_WO" and entry.get("group_size") == _FP8_BLOCK:
        return "FP8_PB_WO"
    raise ValueError(f"{MTP_EXPERTS}: quant_algo {entry.get('quant_algo')!r} is not one this "
                     "converter reads")


def validate_quantization(config: Mapping, layers: int) -> int:
    """Refuse anything but the ModelOpt layout this converter reads; return the PLE layer.

    `quantization_config` is the authority, not the tensor names: ModelOpt's `weight_scale_2`
    and `input_scale` multiply where compressed-tensors' global scales divide, so a checkpoint
    that merely looks similar would mis-scale every expert.
    """
    quant = config.get("quantization_config")
    if not isinstance(quant, Mapping) or quant.get("quant_method") != "modelopt":
        raise ValueError("expected a ModelOpt export (quantization_config.quant_method 'modelopt')")
    quantized = quant.get("quantized_layers")
    if not isinstance(quantized, Mapping):
        raise ValueError("ModelOpt quantization_config has no quantized_layers")
    experts: set[int] = set()
    tables: set[int] = set()
    for name, entry in quantized.items():
        algo = entry.get("quant_algo") if isinstance(entry, Mapping) else None
        if name.startswith("mtp."):
            continue  # the NextN head's experts: `mtp_expert_quantization`
        if (match := _EXPERTS.fullmatch(name)) and algo == "NVFP4" and entry.get("group_size") == 16:
            experts.add(int(match[1]))
        elif (match := _NGRAM_TABLE.fullmatch(name)) and algo == "FP8":
            tables.add(int(match[1]))
        else:
            raise ValueError(f"{name}: quant_algo {algo!r} is not one this converter reads")
    if experts != set(range(layers)):
        raise ValueError("every text layer's routed experts must be NVFP4 with 16-value groups")
    if len(tables) > 1:
        raise ValueError("expected at most one FP8 n-gram table")
    return next(iter(tables), -1)


# --------------------------------------------------------------------------------------------
# Inventory: the GGUF path's objects, with the routed experts in NVFP4
# --------------------------------------------------------------------------------------------


def _nvfp4_specs(name: str, shape: tuple[int, int], second_level: int, experts: int) -> list[TensorSpec]:
    return [
        TensorSpec(name, shape, NVFP4, BLOCK_SCALE_LAYOUT),
        TensorSpec(name + "_scale", (second_level,), inv.FP32, inv.CONTIGUOUS_LAYOUT),
        TensorSpec(name + "_act_scale", (experts,), inv.FP32, inv.CONTIGUOUS_LAYOUT),
        TensorSpec(name + "_alpha", (experts,), inv.FP32, inv.CONTIGUOUS_LAYOUT),
    ]


def routed_nvfp4_specs(specs: Sequence, g: inv.Geometry) -> tuple:
    out: list = []
    for spec in specs:
        if isinstance(spec, TensorSpec) and spec.name.startswith("text/layers/"):
            if spec.name.endswith(GATE_UP_SUFFIX):
                out.extend(_nvfp4_specs(spec.name, spec.shape, 2 * g.experts, g.experts))
                continue
            if spec.name.endswith(DOWN_SUFFIX):
                out.extend(_nvfp4_specs(spec.name, spec.shape, g.experts, g.experts))
                continue
        out.append(spec)
    return tuple(out)


def is_routed_object(name: str) -> bool:
    if not name.startswith("text/layers/"):
        return False
    return any(name.endswith(suffix + extra) for suffix in (GATE_UP_SUFFIX, DOWN_SUFFIX)
               for extra in ("",) + SCALE_SUFFIXES)


# --------------------------------------------------------------------------------------------
# Recipes over the HF names (the reader folds `model.language_model.` to `model.`)
# --------------------------------------------------------------------------------------------


def _hyper_connection(g, src: str, dst: str, with_inject: bool) -> list[TensorRecipe]:
    recipes = [
        TensorRecipe(dst + "norm", source(src + "hc_norm.weight", (g.residual,))),
        TensorRecipe(dst + "down", source(src + "input_mix_weight_down.weight", (g.hc_low_rank, g.residual))),
        TensorRecipe(dst + "up", source(src + "input_mix_weight_up.weight", (g.residual, g.hc_low_rank))),
    ]
    if with_inject:
        recipes.append(TensorRecipe(dst + "inject",
                                    source(src + "block_inject_weight.weight", (g.hc_streams, g.residual))))
    return recipes


def _taps(name: str, channels: int, kernel: int):
    """A depthwise Conv1d weight [channels, 1, kernel] as the engine's [kernel, channels]."""
    return Transpose(Reshape(source(name, (channels, 1, kernel)), (channels, kernel)), (1, 0))


def _ple(g, blk: str, dst: str) -> list[TensorRecipe]:
    src = blk + "ple."
    return [
        TensorRecipe(dst + "key", source(src + "key_proj.weight", (g.residual, g.ple_embed))),
        TensorRecipe(dst + "value", source(src + "value_proj.weight", (g.hidden, g.ple_embed))),
        TensorRecipe(dst + "norm_key", source(src + "norm_key.weight", (g.residual,))),
        TensorRecipe(dst + "norm_query", source(src + "norm_query.weight", (g.residual,))),
        TensorRecipe(dst + "norm_conv", source(src + "norm_conv.weight", (g.residual,))),
        TensorRecipe(dst + "convolution", _taps(src + "conv1d.weight", g.residual, g.ple_conv_kernel)),
    ]


def _attention(g, blk: str, dst: str) -> list[TensorRecipe]:
    src = blk + "self_attn."
    q_proj = src + "q_proj.weight"
    indexer = source(src + "indexer.index_qk_proj.weight",
                     ((g.indexer_heads + 1) * g.indexer_head_dim, g.hidden))
    query_rows = g.indexer_heads * g.indexer_head_dim
    part = dict(num_heads=g.query_heads, hidden_size=g.hidden, head_dim=g.head_dim)
    return [
        TensorRecipe(dst + "query_key_gate_value", Concat((
            attention_qproj_part(q_proj, False, **part),
            source(src + "k_proj.weight", (g.kv_size, g.hidden)),
            attention_qproj_part(q_proj, True, **part),
            source(src + "v_proj.weight", (g.kv_size, g.hidden)),
        ), 0)),
        TensorRecipe(dst + "query_norm", source(src + "q_norm.weight", (g.head_dim,))),
        TensorRecipe(dst + "key_norm", source(src + "k_norm.weight", (g.head_dim,))),
        TensorRecipe(dst + "output", source(src + "o_proj.weight", (g.hidden, g.query_size))),
        TensorRecipe(dst + "indexer/query", Slice(indexer, 0, 0, query_rows)),
        TensorRecipe(dst + "indexer/key", Slice(indexer, 0, query_rows, query_rows + g.indexer_head_dim)),
        TensorRecipe(dst + "indexer/query_norm", source(src + "indexer.q_layernorm.weight", (g.indexer_head_dim,))),
        TensorRecipe(dst + "indexer/key_norm", source(src + "indexer.k_layernorm.weight", (g.indexer_head_dim,))),
    ]


def _gdn(g, blk: str, dst: str) -> list[TensorRecipe]:
    src = blk + "linear_attn."
    return [
        TensorRecipe(dst + "a_log", source(src + "A_log", (g.gdn_value_heads,))),
        TensorRecipe(dst + "dt_bias", source(src + "dt_bias", (g.gdn_value_heads,))),
        TensorRecipe(dst + "convolution", _taps(src + "conv1d.weight", g.convolution_dim, g.gdn_conv_kernel)),
        TensorRecipe(dst + "a_b_projection", Concat((
            source(src + "in_proj_a.weight", (g.gdn_value_heads, g.hidden)),
            source(src + "in_proj_b.weight", (g.gdn_value_heads, g.hidden)),
        ), 0)),
        TensorRecipe(dst + "query_key_value_z", Concat((
            source(src + "in_proj_qkv.weight", (g.convolution_dim, g.hidden)),
            source(src + "in_proj_z.weight", (g.value_dim, g.hidden)),
        ), 0)),
        TensorRecipe(dst + "norm", source(src + "norm.weight", (g.gdn_value_head_dim,))),
        TensorRecipe(dst + "output", source(src + "out_proj.weight", (g.hidden, g.value_dim))),
    ]


def _mlp(g, blk: str, dst: str) -> list[TensorRecipe]:
    src = blk + "mlp."
    shared = src + "shared_expert."
    return [
        TensorRecipe(dst + "router_shared_gate", Concat((
            source(src + "gate.weight", (g.experts, g.hidden)),
            source(src + "shared_expert_gate.weight", (1, g.hidden)),
        ), 0)),
        TensorRecipe(dst + "shared_gate_up", Concat((
            source(shared + "gate_proj.weight", (g.shared_intermediate, g.hidden)),
            source(shared + "up_proj.weight", (g.shared_intermediate, g.hidden)),
        ), 0)),
        TensorRecipe(dst + "shared_down", source(shared + "down_proj.weight", (g.hidden, g.shared_intermediate))),
    ]


def build_recipes(g: inv.Geometry) -> tuple[TensorRecipe, ...]:
    """Every dense text object; the routed experts and the PLE table are built separately."""
    recipes = [TensorRecipe("text/token_embedding", source("model.embed_tokens.weight", (g.vocab, g.hidden)))]
    for layer in range(g.layers):
        blk = f"model.layers.{layer}."
        dst = f"text/layers/{layer}/"
        if g.ple_ngram and layer == g.ple_layer:
            recipes += _ple(g, blk, dst + "ple/")
        recipes += _hyper_connection(g, blk + "attn_hyper_connection.", dst + "hc_attn/", True)
        if layer in g.full_attention_layers:
            recipes += _attention(g, blk, dst + "attention/")
        else:
            recipes += _gdn(g, blk, dst + "gdn/")
        recipes += _hyper_connection(g, blk + "mlp_hyper_connection.", dst + "hc_ffn/", True)
        recipes += _mlp(g, blk, dst + "mlp/")
    recipes += _hyper_connection(g, "model.hyper_connection_mixer.", "text/output_hc/", False)
    head = "model.embed_tokens.weight" if g.tied_embeddings else "lm_head.weight"
    recipes.append(TensorRecipe("text/output_head", source(head, (g.vocab, g.hidden))))
    return tuple(recipes)


def build_mtp_recipes(g: inv.Geometry) -> tuple[TensorRecipe, ...]:
    """The NextN head's dense objects; its routed experts are built separately.

    The engine runs one matmul over `[e; h_s]` per residual stream, so the release's two input
    projections are its column halves, the embedding's first.
    """
    layer = "mtp/layer/"
    recipes = [
        TensorRecipe("mtp/embedding_norm", source("mtp.pre_fc_norm_embedding.weight", (g.hidden,))),
        TensorRecipe("mtp/hidden_norm", source("mtp.pre_fc_norm_hidden.weight", (g.residual,))),
        TensorRecipe("mtp/input_projection", Concat((
            source("mtp.fc_embedding.weight", (g.hidden, g.hidden)),
            source("mtp.fc_hidden.weight", (g.hidden, g.hidden)),
        ), 1)),
    ]
    recipes += _hyper_connection(g, MTP_BLOCK + "attn_hyper_connection.", layer + "hc_attn/", True)
    recipes += _attention(g, MTP_BLOCK, layer + "attention/")
    recipes += _hyper_connection(g, MTP_BLOCK + "mlp_hyper_connection.", layer + "hc_ffn/", True)
    recipes += _mlp(g, MTP_BLOCK, layer + "mlp/")
    recipes += _hyper_connection(g, "mtp.hyper_connection_mixer.", "mtp/head_hc/", False)
    return tuple(recipes)


def unit_offset_objects(g: inv.Geometry) -> frozenset[str]:
    """The norms HF stores zero-centred and the engine reads folded (`1 + w`)."""
    names = {"text/output_hc/norm"}
    if g.mtp_layers:
        names.update(("mtp/embedding_norm", "mtp/hidden_norm", "mtp/head_hc/norm",
                      "mtp/layer/hc_attn/norm", "mtp/layer/hc_ffn/norm",
                      "mtp/layer/attention/indexer/query_norm", "mtp/layer/attention/indexer/key_norm"))
    for layer in range(g.layers):
        dst = f"text/layers/{layer}/"
        names.update((dst + "hc_attn/norm", dst + "hc_ffn/norm"))
        if layer in g.full_attention_layers:
            names.update((dst + "attention/indexer/query_norm", dst + "attention/indexer/key_norm"))
        if g.ple_ngram and layer == g.ple_layer:
            names.update(dst + f"ple/{leaf}" for leaf in ("norm_key", "norm_query", "norm_conv"))
    return frozenset(names)


# --------------------------------------------------------------------------------------------
# The n-gram table and its hash constants
# --------------------------------------------------------------------------------------------


class NgramTable:
    """The FP8 n-gram table as IQ4_NL rows, one stored shard at a time.

    The export splits the table into `split_ngram_parts` row ranges, `shard_0` first; the file
    happens to store them in lexicographic order, which is why they are addressed by name.
    """

    def __init__(self, reader: ShardReader, layer: int, parts: int, head_dim: int) -> None:
        self.reader = reader
        self.prefix = f"model.layers.{layer}.ple.ple_embedding."
        self.parts = [self.prefix + f"ngram_embedding.shard_{i}.weight" for i in range(parts)]
        if reader.has(self.prefix + f"ngram_embedding.shard_{parts}.weight"):
            raise ValueError("the checkpoint holds more n-gram shards than split_ngram_parts says")
        metadata = reader.metadata(self.parts + [self.prefix + "ngram_embedding.weight_scale"])
        self.rows = 0
        for name in self.parts:
            shape, dtype = metadata[name].shape, metadata[name].dtype
            if len(shape) != 2 or shape[1] != head_dim or dtype != "F8_E4M3":
                raise ValueError(f"{name}: expected F8_E4M3 rows of {head_dim}, not {dtype} {shape}")
            self.rows += shape[0]
        scale = reader.get(self.prefix + "ngram_embedding.weight_scale").to(torch.float32).reshape(-1)
        if scale.numel() != 1 or not (float(scale[0]) > 0 and math.isfinite(float(scale[0]))):
            raise ValueError("the n-gram table must carry one finite positive scale")
        self.scale = float(scale[0])
        self.head_dim = head_dim

    def hash_words(self, ngram: int, heads: int) -> dict[str, bytes]:
        multipliers = self.reader.get(self.prefix + "layer_multipliers").to(torch.int64).tolist()
        offsets = self.reader.get(self.prefix + "ngram_heads_offsets").to(torch.int64).tolist()
        counts = self.reader.get(self.prefix + "ngram_heads_vocab_sizes").to(torch.int64).tolist()
        if len(multipliers) != ngram or len(offsets) != heads or len(counts) != heads:
            raise ValueError("PLE hash arrays disagree with the configured n-gram and head counts")
        for i, (offset, count) in enumerate(zip(offsets, counts)):
            if (offset < 0 or count <= 0 or offset + count > self.rows or
                    (i and offset < offsets[i - 1] + counts[i - 1])):
                raise ValueError("PLE head ranges overlap or exceed the stored table")
        words: list[int] = []
        for m in multipliers:
            m &= (1 << 64) - 1
            words.extend((m & 0xFFFFFFFF, m >> 32))
        signed = [w - (1 << 32) if w >= (1 << 31) else w for w in words]
        return {
            "text/ple/multipliers": encode_direct(torch.tensor(signed, dtype=torch.int32), inv.I32),
            "text/ple/head_offsets": encode_direct(torch.tensor(offsets, dtype=torch.int32), inv.I32),
            "text/ple/head_vocab_sizes": encode_direct(torch.tensor(counts, dtype=torch.int32), inv.I32),
        }

    def payload(self, device: torch.device, rows_per_chunk: int = 1 << 18) -> Iterator[bytes]:
        started = time.perf_counter()
        for index, name in enumerate(self.parts):
            part = self.reader.get(name)
            if part.dtype != torch.float8_e4m3fn:
                raise TypeError(f"{name} is {part.dtype}, expected float8_e4m3fn")
            for begin in range(0, part.shape[0], rows_per_chunk):
                rows = part[begin:begin + rows_per_chunk].to(device).to(torch.float32) * self.scale
                if not bool(torch.isfinite(rows).all()):
                    raise ValueError(f"{name}: the n-gram table holds a non-finite value")
                yield iq4nl.quantize_rows(rows).cpu().numpy().tobytes()
            del part
            if (index + 1) % 16 == 0 or index + 1 == len(self.parts):
                print(f"n-gram table: {index + 1}/{len(self.parts)} shards as IQ4_NL "
                      f"({time.perf_counter() - started:.0f} s)", flush=True)


# --------------------------------------------------------------------------------------------
# Routed experts
# --------------------------------------------------------------------------------------------


def _positive(tensor: torch.Tensor, what: str) -> float:
    value = float(tensor.reshape(()).to(torch.float32))
    if not (value > 0.0) or not math.isfinite(value):
        raise ValueError(f"{what} is {value}, expected finite positive")
    return value


def expert_source_names(layer: int, g: inv.Geometry) -> Iterator[str]:
    for expert in range(g.experts):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            base = f"model.layers.{layer}.mlp.experts.{expert}.{projection}."
            yield from (base + leaf for leaf in ("weight", "weight_scale", "weight_scale_2", "input_scale"))


def preflight_experts(reader: ShardReader, g: inv.Geometry) -> None:
    for layer in range(g.layers):
        metadata = reader.metadata(tuple(expert_source_names(layer, g)))
        for expert in range(g.experts):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                n, k = (g.hidden, g.intermediate) if projection == "down_proj" else (g.intermediate, g.hidden)
                base = f"model.layers.{layer}.mlp.experts.{expert}.{projection}."
                for leaf, shape, dtype in (("weight", (n, k // 2), "U8"),
                                           ("weight_scale", (n, k // _NVFP4_BLOCK), "F8_E4M3")):
                    actual = metadata[base + leaf]
                    if actual.shape != shape or actual.dtype != dtype:
                        raise ValueError(f"{base + leaf}: stored {actual.dtype} {actual.shape}, "
                                         f"expected {dtype} {shape}")
                for leaf in ("weight_scale_2", "input_scale"):
                    actual = metadata[base + leaf]
                    if math.prod(actual.shape) != 1 or actual.dtype not in ("BF16", "F16", "F32"):
                        raise ValueError(f"{base + leaf}: expected one floating-point scale")


def _projection(reader: ShardReader, layer: int, expert: int, projection: str):
    """One projection's codes, block scales, and ModelOpt's two multipliers."""
    base = f"model.layers.{layer}.mlp.experts.{expert}.{projection}."
    codes = reader.get(base + "weight")
    scales = reader.get(base + "weight_scale")
    if codes.dtype != torch.uint8 or scales.dtype != torch.float8_e4m3fn:
        raise TypeError(f"{base}: expected uint8 codes and float8_e4m3fn scales")
    return (codes, scales.view(torch.uint8), _positive(reader.get(base + "weight_scale_2"), base + "weight_scale_2"),
            _positive(reader.get(base + "input_scale"), base + "input_scale"))


def build_gate_up(reader: ShardReader, layer: int, g: inv.Geometry):
    """One layer's stacked experts, **up first, gate second**, as the TRT-LLM runner reads them.

    ModelOpt's `weight_scale_2` is the weight's second level as a multiplier, which is what the
    engine's scale object holds; `input_scale` is the activation's dequantisation multiplier, so
    the runner's pre-rounding activation scale is its reciprocal and its epilogue alpha (which
    undoes both) their product. The runner carries one activation scale and one alpha per expert,
    so gate and up must agree on both.
    """
    rows = g.experts * 2 * g.intermediate
    codes = torch.empty((rows, g.hidden // 2), dtype=torch.uint8)
    scales = torch.empty((rows, g.hidden // _NVFP4_BLOCK), dtype=torch.uint8)
    second = torch.empty(2 * g.experts, dtype=torch.float32)
    act_scale = torch.empty(g.experts, dtype=torch.float32)
    alpha = torch.empty(g.experts, dtype=torch.float32)
    for expert in range(g.experts):
        halves = []
        for half, projection in enumerate(("up_proj", "gate_proj")):
            plane, scale_plane, weight_scale, input_scale = _projection(reader, layer, expert, projection)
            begin = (expert * 2 + half) * g.intermediate
            codes[begin:begin + g.intermediate] = plane
            scales[begin:begin + g.intermediate] = scale_plane
            second[expert * 2 + half] = weight_scale
            halves.append((weight_scale, input_scale))
        if halves[0] != halves[1]:
            raise ValueError(f"layer {layer} expert {expert}: gate and up disagree on their scales "
                             f"{halves}; the fused runner carries one of each per expert")
        weight_scale, input_scale = halves[0]
        act_scale[expert] = 1.0 / input_scale
        alpha[expert] = input_scale * weight_scale
    payload = encode_nvfp4(codes, scales, torch.tensor(1.0), (rows, g.hidden))
    return payload, second, act_scale, alpha


def build_down(reader: ShardReader, layer: int, g: inv.Geometry):
    rows = g.experts * g.hidden
    codes = torch.empty((rows, g.intermediate // 2), dtype=torch.uint8)
    scales = torch.empty((rows, g.intermediate // _NVFP4_BLOCK), dtype=torch.uint8)
    second = torch.empty(g.experts, dtype=torch.float32)
    act_scale = torch.empty(g.experts, dtype=torch.float32)
    alpha = torch.empty(g.experts, dtype=torch.float32)
    for expert in range(g.experts):
        plane, scale_plane, weight_scale, input_scale = _projection(reader, layer, expert, "down_proj")
        begin = expert * g.hidden
        codes[begin:begin + g.hidden] = plane
        scales[begin:begin + g.hidden] = scale_plane
        second[expert] = weight_scale
        act_scale[expert] = 1.0 / input_scale
        alpha[expert] = input_scale * weight_scale
    payload = encode_nvfp4(codes, scales, torch.tensor(1.0), (rows, g.intermediate))
    return payload, second, act_scale, alpha


class RoutedLayers:
    """Builds a layer's eight routed objects once and hands them out in inventory order."""

    def __init__(self, reader: ShardReader, g: inv.Geometry) -> None:
        self.reader, self.g = reader, g
        self.layer: int | None = None
        self.objects: dict[str, object] = {}

    def payload(self, name: str) -> bytes:
        layer = int(name.split("/")[2])
        if layer != self.layer:
            prefix = f"text/layers/{layer}"
            self.objects = {}
            for suffix, build in ((GATE_UP_SUFFIX, build_gate_up), (DOWN_SUFFIX, build_down)):
                payload, second, act_scale, alpha = build(self.reader, layer, self.g)
                self.objects.update({prefix + suffix: payload, prefix + suffix + "_scale": second,
                                     prefix + suffix + "_act_scale": act_scale,
                                     prefix + suffix + "_alpha": alpha})
            self.layer = layer
        value = self.objects.pop(name)
        return value if isinstance(value, bytes) else encode_direct(value, inv.FP32)


class MtpExperts:
    """The NextN head's routed experts as the logical matrices the W8 encoder quantises.

    Stacked as the GGUF path stacks them: gate then up within each expert, down per expert.
    FP8 block-scaled experts are dequantised first (`weight_scale_inv` multiplies its block).
    """

    def __init__(self, reader: ShardReader, g: inv.Geometry, quantization: str | None) -> None:
        self.reader, self.g, self.fp8 = reader, g, quantization == "FP8_PB_WO"

    def _leaves(self, expert: int, projection: str) -> tuple[str, ...]:
        base = f"{MTP_EXPERTS}.{expert}.{projection}."
        return (base + "weight", base + "weight_scale_inv") if self.fp8 else (base + "weight",)

    def _shape(self, projection: str) -> tuple[int, int]:
        g = self.g
        return (g.hidden, g.intermediate) if projection == "down_proj" else (g.intermediate, g.hidden)

    def preflight(self) -> None:
        names = [leaf for expert in range(self.g.experts)
                 for projection in ("gate_proj", "up_proj", "down_proj")
                 for leaf in self._leaves(expert, projection)]
        metadata = self.reader.metadata(names)
        for expert in range(self.g.experts):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                rows, columns = self._shape(projection)
                leaves = self._leaves(expert, projection)
                expected = [(leaves[0], (rows, columns), "F8_E4M3" if self.fp8 else "BF16")]
                if self.fp8:
                    expected.append((leaves[1], (math.ceil(rows / _FP8_BLOCK), math.ceil(columns / _FP8_BLOCK)),
                                     None))
                for name, shape, dtype in expected:
                    actual = metadata[name]
                    if actual.shape != shape or (dtype and actual.dtype != dtype) or (
                            dtype is None and actual.dtype not in ("BF16", "F16", "F32")):
                        raise ValueError(f"{name}: stored {actual.dtype} {actual.shape}, expected {shape}")

    def _projection(self, expert: int, projection: str, device: torch.device) -> torch.Tensor:
        rows, columns = self._shape(projection)
        leaves = self._leaves(expert, projection)
        weight = self.reader.get(leaves[0]).to(device).to(torch.float32)
        if not self.fp8:
            return weight
        scale = self.reader.get(leaves[1]).to(device).to(torch.float32)
        if not bool(torch.isfinite(scale).all()):
            raise ValueError(f"{leaves[1]}: a block scale is not finite")
        blocks = scale.repeat_interleave(_FP8_BLOCK, 0)[:rows].repeat_interleave(_FP8_BLOCK, 1)[:, :columns]
        return weight * blocks

    def tensor(self, name: str, device: torch.device) -> torch.Tensor:
        g = self.g
        if name == MTP_ROUTED[0]:
            out = torch.empty((g.experts * 2 * g.intermediate, g.hidden), dtype=torch.float32, device=device)
            for expert in range(g.experts):
                for half, projection in enumerate(("gate_proj", "up_proj")):
                    begin = (expert * 2 + half) * g.intermediate
                    out[begin:begin + g.intermediate] = self._projection(expert, projection, device)
            return out
        out = torch.empty((g.experts * g.hidden, g.intermediate), dtype=torch.float32, device=device)
        for expert in range(g.experts):
            out[expert * g.hidden:(expert + 1) * g.hidden] = self._projection(expert, "down_proj", device)
        return out


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


def encode(tensor: torch.Tensor, spec, device: torch.device) -> bytes:
    if spec.format in _DIRECT:
        return encode_direct(tensor.to(_DIRECT[spec.format]), spec.format)
    if spec.format == inv.W8:
        return family_conversion.encode_tensor_payload(tensor, spec, device)
    raise ValueError(f"{spec.name}: nothing here encodes {spec.format}")


def convert(model_dir: str | Path, out_path: str | Path, *, device: str | None = None,
            mtp: bool = True) -> Path:
    started = time.perf_counter()
    model_dir = Path(model_dir)
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = json.loads((model_dir / "config.json").read_text())
    mtp = mtp and int(text_config(config, mtp=True).get("mtp_num_hidden_layers") or 0) > 0
    text = text_config(config, mtp=mtp)
    layers = int(text.get("num_hidden_layers", 0))
    table_layer = validate_quantization(config, layers)
    if mtp:
        validate_mtp(text)

    with ShardReader.for_directory(model_dir) as reader:
        ple_layers = [i - 1 for i in text.get("ple_layer_ids") or []]
        if ple_layers != ([table_layer] if table_layer >= 0 else []):
            raise ValueError("the FP8 n-gram table must sit on the configured PLE layer")
        table = None
        if ple_layers:
            parts = text.get("split_ngram_parts")
            if isinstance(parts, bool) or not isinstance(parts, int) or parts <= 0:
                raise ValueError("config.split_ngram_parts must name the n-gram table's shard count")
            heads = (text["ngram_size"] - 1) * text["heads_per_ngram"]
            table = NgramTable(reader, ple_layers[0], parts, text["ple_embed_dim"] // heads)
        g = inv.geometry_from_config(text, ple_table_rows=table.rows if table else 0,
                                     token_domain=tokenizer_domain(model_dir))
        if g.hidden % _NVFP4_BLOCK or g.intermediate % _NVFP4_BLOCK:
            raise ValueError("routed NVFP4 requires hidden and expert widths divisible by 16")
        tensor_specs, object_specs = inv.active_specs(geometry=g, vision=False, mtp=mtp)
        tensor_specs = routed_nvfp4_specs(tensor_specs, g)
        object_specs = routed_nvfp4_specs(object_specs, g)

        recipes = {item.object_name: item
                   for item in build_recipes(g) + (build_mtp_recipes(g) if mtp else ())}
        unit_offset = unit_offset_objects(g)
        hashes = table.hash_words(g.ple_ngram, g.ple_heads) if table else {}
        draft_experts = MtpExperts(reader, g, mtp_expert_quantization(config)) if mtp else None
        planned = set(recipes) | set(hashes) | {inv.PLE_TABLE_RESOURCE} | (set(MTP_ROUTED) if mtp else set())
        stray = [s.name for s in tensor_specs if s.name not in planned and not is_routed_object(s.name)]
        if stray or not unit_offset <= set(recipes):
            raise ValueError(f"objects with no source: {stray[:8]}")
        preflight = preflight_source_reader(reader, tuple(recipes.values()))
        preflight_experts(reader, g)
        if draft_experts:
            draft_experts.preflight()
        print(f"preflight: {preflight.source_tensor_count} dense sources, "
              f"{g.layers} layers of {g.experts} NVFP4 experts, "
              f"n-gram table {g.ple_table_rows} rows, "
              f"NextN head {'converted' if mtp else 'left out'}", flush=True)

        # A text-only artifact must not offer pixels: the processor configs the release ships
        # stay out, and the engine then refuses `--vision` against it.
        frontend = family_conversion.load_resources(model_dir, inv.RESOURCE_SPECS)
        resources = {item.name: item.data for item in frontend
                     if not item.name.endswith("preprocessor_config.json")}
        plan = family_conversion.build_object_plan(object_specs, resources)
        routed = RoutedLayers(reader, g)
        specs = list(plan.specs)
        print(f"converting {len(specs)} objects to {output}", flush=True)
        with ArtifactWriter(
            output,
            ArtifactIdentity(inv.MODEL_ID, WEIGHTS_ID, architecture="qwen4exp"),
            plan.specs,
            geometry=geometry_block(g),
            layer_types=g.layer_types,
        ) as writer:
            for index, spec in enumerate(specs, start=1):
                t0 = time.perf_counter()
                if spec.name in resources:
                    payload = resources[spec.name]
                elif spec.name == inv.PLE_TABLE_RESOURCE:
                    payload = table.payload(target)
                elif spec.name in hashes:
                    payload = hashes[spec.name]
                elif is_routed_object(spec.name):
                    payload = routed.payload(spec.name)
                elif spec.name in MTP_ROUTED:
                    tensor = draft_experts.tensor(spec.name, target)
                    payload = encode(tensor, spec, target)
                    del tensor
                else:
                    tensor = materialize_recipe(recipes[spec.name], reader)
                    if spec.name in unit_offset:
                        tensor = tensor.to(torch.float32) + 1.0
                    payload = encode(tensor, spec, target)
                    del tensor
                writer.write(spec.name, payload)
                del payload
                if index % 50 == 0 or index == len(specs):
                    print(f"[{index}/{len(specs)}] {spec.name} ({time.perf_counter() - t0:.1f}s)", flush=True)

    report = {
        "recipe_id": RECIPE_ID,
        "model_id": inv.MODEL_ID,
        "weights_id": WEIGHTS_ID,
        "geometry": geometry_block(g),
        "layer_types": list(g.layer_types),
        "source": str(model_dir),
        "device": str(target),
        "elapsed_seconds": time.perf_counter() - started,
        "bytes": output.stat().st_size,
        "objects": len(specs),
    }
    Path(str(output) + ".conversion.json").write_text(json.dumps(report, indent=2))
    print(f"conversion finished in {report['elapsed_seconds']:.0f} s -> {output} "
          f"({report['bytes'] / 1e9:.2f} GB)", flush=True)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="the release's directory (config.json, shards)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None, help="where the dense and table quantisers run")
    # The ingest path asks every converter the same way; this one never converts either.
    parser.add_argument("--no-vision", action="store_true", help="accepted; the tower is never converted")
    parser.add_argument("--no-mtp", action="store_true", help="leave the NextN draft head out")
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device, mtp=not args.no_mtp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
