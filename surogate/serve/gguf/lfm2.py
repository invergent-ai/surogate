"""Recover an LFM2 checkpoint configuration from its GGUF metadata."""

from __future__ import annotations

import math

from surogate.serve.convert.common.checkpoint import positive_int


def config_from_gguf(reader) -> dict:
    def kv(name, default=None):
        field = reader.get_field("lfm2." + name)
        return default if field is None else field.contents()

    dimensions = {
        "hidden_size": kv("embedding_length"),
        "num_hidden_layers": kv("block_count"),
        "intermediate_size": kv("feed_forward_length"),
        "conv_L_cache": kv("shortconv.l_cache"),
        "max_position_embeddings": kv("context_length"),
    }
    for name in dimensions:
        positive_int(dimensions, name)
    layers = dimensions["num_hidden_layers"]

    def per_layer(name):
        value = kv(name)
        values = list(value) if isinstance(value, (list, tuple)) else [value] * layers
        if len(values) != layers or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values
        ):
            raise ValueError(f"GGUF lfm2.{name} must declare a nonnegative count for each layer")
        return values

    kv_counts = per_layer("attention.head_count_kv")
    query_counts = per_layer("attention.head_count")
    attention = [i for i, count in enumerate(kv_counts) if count > 0]
    if not attention:
        raise ValueError("LFM2 serving requires at least one attention layer")
    heads = {query_counts[i] for i in attention}
    kv_heads = {kv_counts[i] for i in attention}
    if len(heads) != 1 or 0 in heads or len(kv_heads) != 1:
        raise ValueError("LFM2 serving requires uniform query and KV head counts in attention layers")
    query_heads = heads.pop()
    if dimensions["hidden_size"] % query_heads or query_heads % next(iter(kv_heads)):
        raise ValueError("LFM2 GGUF hidden size and query/KV head counts must divide evenly")
    head_dim = dimensions["hidden_size"] // query_heads
    for key in ("attention.key_length", "attention.value_length", "rope.dimension_count"):
        if kv(key, head_dim) != head_dim:
            raise ValueError(f"LFM2 serving does not support GGUF lfm2.{key} != hidden_size / heads")
    if kv("attention.sliding_window", 0):
        raise ValueError("LFM2 serving does not support sliding-window attention")
    if kv("attention.causal", True) is not True or kv("pooling_type", 0):
        raise ValueError("LFM2 GGUF serving supports causal text generation, not embedding models")
    if kv("expert_count", 0):
        raise ValueError("LFM2 GGUF serving does not support mixture-of-experts checkpoints")
    for key in ("rope.freq_base", "attention.layer_norm_rms_epsilon"):
        value = kv(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"GGUF lfm2.{key} must be a positive finite number")

    tokens = reader.get_field("tokenizer.ggml.tokens")
    if tokens is None:
        raise ValueError("LFM2 GGUF must declare its tokenizer vocabulary")
    vocab = len(tokens.contents())
    if kv("vocab_size", vocab) != vocab:
        raise ValueError("LFM2 GGUF vocab_size disagrees with its tokenizer vocabulary")
    from surogate.serve.gguf.frontend import extract_generation_config

    return {
        **dimensions,
        **extract_generation_config(reader),
        "architectures": ["Lfm2ForCausalLM"],
        "model_type": "lfm2",
        "num_attention_heads": query_heads,
        "num_key_value_heads": kv_heads.pop(),
        "vocab_size": vocab,
        "norm_eps": kv("attention.layer_norm_rms_epsilon"),
        "rope_theta": kv("rope.freq_base"),
        "layer_types": ["full_attention" if i in attention else "conv" for i in range(layers)],
        # GGUF already stores the adjusted FFN width, so do not apply HF's adjustment again.
        "block_auto_adjust_ff_dim": False,
        "conv_bias": False,
        "tie_word_embeddings": reader.tensor("output.weight") is None,
    }
