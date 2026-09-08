"""Resolve hyper-connected hybrid checkpoints, including their PLE table metadata."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, fields
from types import SimpleNamespace

from .checkpoint import positive_int
from . import qwen3_5 as hybrid


@dataclass(frozen=True, kw_only=True)
class Geometry(hybrid.Geometry):
    hc_streams: int
    hc_low_rank: int
    indexer_heads: int
    indexer_head_dim: int
    indexer_top_k: int
    indexer_block: int
    ple_layer: int
    ple_ngram: int
    ple_heads_per_ngram: int
    ple_head_dim: int
    ple_conv_kernel: int
    ple_table_rows: int
    ple_eos_token: int
    ple_image_token: int

    @property
    def residual(self):
        return self.hidden * self.hc_streams

    @property
    def attention_fused_rows(self):
        return 2 * self.query_size + 2 * self.kv_size

    @property
    def gdn_fused_rows(self):
        return self.convolution_dim + self.value_dim

    @property
    def router_rows(self):
        return self.experts + 1

    @property
    def ple_heads(self):
        return (self.ple_ngram - 1) * self.ple_heads_per_ngram if self.ple_ngram else 0

    @property
    def ple_embed(self):
        return self.ple_heads * self.ple_head_dim

    @property
    def ple_table_row_bytes(self):
        return self.ple_head_dim // 32 * 18


def geometry_from_config(config: Mapping, *, ple_table_rows: int = 0,
                         token_domain: int | None = None) -> Geometry:
    source = deepcopy(dict(config))
    text = source.get("text_config", source)
    if not isinstance(text, dict):
        raise ValueError("config.text_config must be an object")
    for name in ("hc_count", "hc_lowrank", "indexer_n_heads", "indexer_head_dim",
                 "indexer_budget", "indexer_compress_ratio"):
        positive_int(text, name)
    if text.get("output_gate_type") != "sigmoid":
        raise ValueError("qwen4exp currently requires a sigmoid GDN output gate")
    if text.get("indexer_kv_heads", 1) != 1:
        raise ValueError("qwen4exp currently supports one indexer KV head")
    layers = text.get("ple_layer_ids")
    if not isinstance(layers, list) or len(layers) > 1 or any(
        isinstance(i, bool) or not isinstance(i, int) or not 1 <= i <= positive_int(text, "num_hidden_layers") for i in layers
    ):
        raise ValueError("qwen4exp supports zero or one PLE layer, named by its one-based layer id")
    if layers:
        for name in ("ngram_size", "heads_per_ngram", "ple_embed_dim", "ple_conv_kernel_size"):
            positive_int(text, name)
        heads = (text["ngram_size"] - 1) * text["heads_per_ngram"]
        if heads <= 0 or text["ple_embed_dim"] % heads:
            raise ValueError("PLE embedding width must be divisible by its head count")
        if not 2 <= text["ngram_size"] <= 3 or heads > 16:
            raise ValueError("the PLE kernel supports n-grams of order 2..3 and at most 16 heads")
        head = text["ple_embed_dim"] // heads
        if head % 32:
            raise ValueError("IQ4_NL PLE rows require head width divisible by 32")
        positive_int({"ple_table_rows": ple_table_rows}, "ple_table_rows")
        for name, value in (("ple_eos_token_id", text.get("eos_token_id")),
                            ("ple_image_token_id", source.get("image_token_id", text.get("image_token_id")))):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < positive_int(text, "vocab_size"):
                raise ValueError(f"{name} must index the checkpoint vocabulary")
    else:
        head = 0
        ple_table_rows = 0
        for name in ("ngram_size", "heads_per_ngram", "ple_embed_dim", "ple_conv_kernel_size"):
            text[name] = 0
    base = hybrid.geometry_from_config(source, mixture=True, token_domain=token_domain,
                                       architecture_base="Qwen4Exp")
    g = Geometry(**{f.name: getattr(base, f.name) for f in fields(base)},
                 hc_streams=text["hc_count"], hc_low_rank=text["hc_lowrank"],
                 indexer_heads=text["indexer_n_heads"], indexer_head_dim=text["indexer_head_dim"],
                 indexer_top_k=text["indexer_budget"], indexer_block=text["indexer_compress_ratio"],
                 ple_layer=layers[0] - 1 if layers else 0, ple_ngram=text["ngram_size"],
                 ple_heads_per_ngram=text["heads_per_ngram"], ple_head_dim=head,
                 ple_conv_kernel=text["ple_conv_kernel_size"], ple_table_rows=ple_table_rows,
                 ple_eos_token=text.get("eos_token_id", 0) if layers else 0,
                 ple_image_token=source.get("image_token_id", text.get("image_token_id", 0)) if layers else 0)
    if max(g.residual, g.indexer_heads * g.indexer_head_dim,
           (g.ple_conv_kernel - 1) * g.ple_ngram, g.ple_heads) > 2147483647:
        raise ValueError("qwen4exp projection or state dimensions exceed int32")
    geometry_block(g)
    return g


def geometry_block(g: Geometry) -> dict:
    from surogate.serve.artifact.geometry import validate_resolved_geometry
    values = hybrid.geometry_block(g)
    values.update(residual=g.residual, draft_vocab=0, **{
        name: getattr(g, name) for name in (
            "hc_streams", "hc_low_rank", "indexer_heads", "indexer_head_dim",
            "indexer_top_k", "indexer_block", "ple_layer", "ple_ngram",
            "ple_heads_per_ngram", "ple_head_dim", "ple_conv_kernel", "ple_table_rows",
            "ple_eos_token", "ple_image_token",
        )
    })
    return validate_resolved_geometry(values)


def geometry_from_gguf(source, *, token_domain: int | None = None) -> Geometry:
    # Resolve the trunk from the first shard; the optional NextN file has separate metadata.
    reader = source.readers[0]
    if reader.kv("general.architecture") != "qwen4exp":
        raise ValueError("GGUF does not declare the qwen4exp architecture")
    # Tensor descriptors can live in any shard. The shared normalizer consumes GGML
    # dimension order; GgufSource exposes logical (NumPy) order instead.
    def tensor(name):
        item = source.tensors.get(name)
        return SimpleNamespace(shape=tuple(reversed(item.shape))) if item is not None else None

    config = hybrid.config_from_gguf(SimpleNamespace(kv=reader.kv, tensor=tensor), "qwen4exp")
    config.update(architectures=["Qwen4ExpForCausalLM"], model_type="qwen4_exp")

    def required(name):
        value = reader.kv("qwen4exp." + name)
        if value is None:
            raise ValueError("GGUF is missing qwen4exp." + name)
        return value

    interval = positive_int(config, "full_attention_interval")
    layers = positive_int(config, "num_hidden_layers")
    if required("attention.value_length") != config["head_dim"]:
        raise ValueError("qwen4exp requires matching key and value head widths")
    schedule = ["full_attention" if (i + 1) % interval == 0
                else "linear_attention" for i in range(config["num_hidden_layers"])]
    ratios = required("attention.compress_ratios")
    if not isinstance(ratios, list) or len(ratios) != len(schedule):
        raise ValueError("attention.compress_ratios must match the trunk layer count")
    blocks = set()
    for ratio, kind in zip(ratios, schedule):
        if kind == "full_attention":
            positive_int({"compress_ratio": ratio}, "compress_ratio")
            blocks.add(ratio)
        elif isinstance(ratio, bool) or not isinstance(ratio, int) or ratio != 0:
            raise ValueError("linear attention layers cannot carry an indexer compression ratio")
    if len(blocks) != 1:
        raise ValueError("qwen4exp currently requires one compression ratio across attention layers")
    ple = reader.kv("qwen4exp.ple.layers", [])
    if not isinstance(ple, list) or any(isinstance(i, bool) or not isinstance(i, int) or
                                        not 0 <= i < layers for i in ple):
        raise ValueError("GGUF ple.layers must contain valid zero-based layer indices")
    config.update(hc_count=required("hyper_connection.count"), hc_lowrank=required("hyper_connection.low_rank"),
                  indexer_n_heads=required("attention.indexer.head_count"),
                  indexer_head_dim=required("attention.indexer.key_length"),
                  indexer_budget=required("attention.indexer.top_k"), indexer_compress_ratio=next(iter(blocks)),
                  indexer_kv_heads=1, output_gate_type="sigmoid", layer_types=schedule,
                  ple_layer_ids=[i + 1 for i in ple])
    table_rows = 0
    if ple:
        table = source.tensor("per_layer_token_embd.weight")
        head = required("embedding_length_per_layer_input")
        if table.type_name != "IQ4_NL" or len(table.shape) != 2 or table.shape[1] != head:
            raise ValueError("PLE table must be an IQ4_NL matrix of the declared head width")
        table_rows = table.shape[0]
        config.update(ngram_size=required("ple.ngram_size"), heads_per_ngram=required("ple.heads_per_ngram"),
                      ple_conv_kernel_size=required("ple.conv_kernel"),
                      eos_token_id=required("ple.eos_token_id"), image_token_id=required("ple.image_token_id"))
        config["ple_embed_dim"] = (config["ngram_size"] - 1) * config["heads_per_ngram"] * head
    config["mtp_num_hidden_layers"] = int(
        f"blk.{config['num_hidden_layers']}.nextn.eh_proj.weight" in source.tensors)
    g = geometry_from_config(config, ple_table_rows=table_rows, token_domain=token_domain)
    if ple:
        multipliers = required("ple.layer_multipliers")
        offsets = required("ple.head_offsets")
        vocab_sizes = required("ple.head_vocab_sizes")
        if (not all(isinstance(a, list) for a in (multipliers, offsets, vocab_sizes)) or
            len(multipliers) != g.ple_ngram or len(offsets) != g.ple_heads or len(vocab_sizes) != g.ple_heads):
            raise ValueError("PLE hash arrays disagree with the configured n-gram and head counts")
        for i, (offset, count) in enumerate(zip(offsets, vocab_sizes)):
            if (isinstance(offset, bool) or not isinstance(offset, int) or offset < 0 or
                isinstance(count, bool) or not isinstance(count, int) or count <= 0 or
                offset + count > table_rows or (i and offset < offsets[i - 1] + vocab_sizes[i - 1])):
                raise ValueError("PLE head ranges overlap or exceed the stored table")
        if any(isinstance(m, bool) or not isinstance(m, int) or not 0 <= m < 2**64 for m in multipliers):
            raise ValueError("PLE multipliers must be uint64 values")
    return g
