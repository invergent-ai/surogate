# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# GGUF ingest for `surogate serve` (design/serve-engine-plan.md §5.1, §5.2b).
#
# Bridge GGUF to the input the checkpoint converter already
# accepts — a temporary HF-layout directory with BF16 safetensors shards. Fully
# offline: the tokenizer and chat template are reconstructed from the GGUF's
# own KV metadata (frontend.py), configuration is resolved from that metadata, and family
# modules (qwen35.py) invert llama.cpp's export transforms. The converter then
# runs its source and shape preflight checks. Costs one temporary BF16
# materialization on disk (~2 bytes/param, deleted after conversion); a
# reader-injection path that avoids the temp copy is a tracked follow-up.
# K-quant sources are dequantized to BF16 and re-encoded by the recipe (one
# documented double quantization; native K-quant repack is the plan's §5.2 path).

from __future__ import annotations

import json
from pathlib import Path

_SHARD_BYTES = 8 << 30


def _arch_kv(reader, arch: str, key: str, default=None):
    field = reader.get_field(f"{arch}.{key}")
    if field is None:
        return default
    return field.contents()


def open_gguf(gguf_path: Path):
    """Open a GGUF via the lean metadata parser (surogate/serve/gguf/lean.py).

    gguf-py's GGUFReader parses ALL KV eagerly (~10s on a 250k-token
    vocabulary); the lean parser indexes spans in ~0.1s and parses arrays on
    demand, exposing a get_field() facade so KV consumers work unchanged.
    gguf-py stays only as the dequantizer (and as the parser oracle in tests).
    """
    from surogate.serve.gguf.lean import LeanGguf, LeanGgufError

    try:
        return LeanGguf(gguf_path)
    except (LeanGgufError, OSError) as exc:
        raise SystemExit(f"surogate serve: '{gguf_path}' is not a readable GGUF file ({exc}).")


def read_gguf_summary(gguf_path: Path, reader=None) -> dict:
    """Cheap metadata pass: architecture + geometry from GGUF KV, no tensor data."""
    if reader is None:
        reader = open_gguf(gguf_path)
    arch_field = reader.get_field("general.architecture")
    arch = arch_field.contents() if arch_field is not None else ""
    summary = {
        "architecture": arch,
        "hidden_size": _arch_kv(reader, arch, "embedding_length", 0),
        "num_hidden_layers": _arch_kv(reader, arch, "block_count", 0),
        "num_attention_heads": _arch_kv(reader, arch, "attention.head_count", 0),
        "num_key_value_heads": _arch_kv(reader, arch, "attention.head_count_kv", 0),
        "tensor_count": len(reader.tensors),
        "quant_types": sorted({t.type_name for t in reader.tensors}),
        # What tells one architecture string's variants apart. Gemma 4 gives its dense pair,
        # its E-series and its mixture one `general.architecture`, exactly as their HF configs
        # give them one `model_type`, so the routing needs the shape and not the name.
        "num_experts": _arch_kv(reader, arch, "expert_count", 0),
        "per_layer_input_dim": _arch_kv(reader, arch, "embedding_length_per_layer_input", 0),
        "kv_shared_layers": _arch_kv(reader, arch, "attention.shared_kv_layers", 0),
    }
    del reader
    return summary


#: gguf-py lists several HF aliases for one tensor, and `hf_preference` below cannot tell which
#: spelling a given family's checkpoint actually uses — it only prefers `model.`-rooted names.
#: Where the alias it lands on is not the one the converter's recipe names, say so here. These
#: are substring rewrites on the HF side of the map, applied after it is built.
_HF_ALIAS_FIXUPS: dict[str, tuple[tuple[str, str], ...]] = {
    "lfm2": (("model.pre_ln", "model.embedding_norm"),
             (".input_layernorm", ".operator_norm"),
             (".post_attention_layernorm", ".ffn_norm"),
             ("self_attn.o_proj", "self_attn.out_proj"),
             ("mlp.gate_proj", "feed_forward.w1"),
             ("mlp.down_proj", "feed_forward.w2"),
             ("mlp.up_proj", "feed_forward.w3")),
    # Qwen3's per-head norms are `q_norm`/`k_norm` in the checkpoint; the generic map reaches
    # them by their `q_layernorm`/`k_layernorm` alias.
    "qwen3": (("self_attn.q_layernorm", "self_attn.q_norm"),
              ("self_attn.k_layernorm", "self_attn.k_norm")),
    # Gemma 3 spells them the same way Qwen3 does.
    "gemma3": (("self_attn.q_layernorm", "self_attn.q_norm"),
               ("self_attn.k_layernorm", "self_attn.k_norm")),
    # Qwen3-MoE spells its per-head norms like the dense Qwen3, and puts its router under
    # `mlp.gate` where the generic map reaches it as Mixtral's `block_sparse_moe.gate`.
    "qwen3moe": (("self_attn.q_layernorm", "self_attn.q_norm"),
                 ("self_attn.k_layernorm", "self_attn.k_norm"),
                 ("block_sparse_moe.gate", "mlp.gate")),
    # Gemma 4. One architecture string covers the dense pair, the E-series and the mixture, so
    # these fixups serve all three targets.
    "gemma4": (("self_attn.q_layernorm", "self_attn.q_norm"),
               ("self_attn.k_layernorm", "self_attn.k_norm"),
               # The router is `router.proj`, where the generic map reaches it by Mixtral's
               # `block_sparse_moe.gate`.
               ("block_sparse_moe.gate", "router.proj"),
               # The expert bank hangs off the layer, not off `mlp`: `experts.gate_up_proj`,
               # not `mlp.experts.gate_up_proj`.
               ("mlp.experts.", "experts."),
               ),
}


#: Tensors a GGUF carries beside a weight, under a `.scale` suffix gguf-py's name map does not
#: emit -- it enumerates `.weight` and `.bias` only.
#:
#: Gemma 4's mixture needs both. Its router applies a learned per-channel scale to its input,
#: and its experts a learned per-expert scale to the finished routing weights; llama.cpp stores
#: the first beside the router projection and the second beside the *down* experts, which is
#: the same association the safetensors converter derives. Neither is optional: without the
#: per-expert scale every token's experts are weighted wrongly by up to a tenth.
#: Checkpoint entries that carry no `.weight`, by architecture.
#:
#: Gemma 4 has three. `layer_scalar` is an `nn.Buffer` of one element. The two expert banks are
#: stacked parameters -- `[experts, 2 * intermediate, hidden]` and `[experts, hidden,
#: intermediate]` -- stored under their bare names. Everything else in the model, its per-head
#: norms and its router projection included, does carry the suffix, so this is a list and not a
#: rule about shapes.
_BARE_HF_NAMES: dict[str, tuple[str, ...]] = {
    "gemma4": ("layer_scalar", "experts.gate_up_proj", "experts.down_proj"),
}


_SCALE_SIDECARS: dict[str, tuple[tuple[str, str], ...]] = {
    "gemma4": (("blk.{bid}.ffn_gate_inp.scale", "model.layers.{bid}.router.scale"),
               ("blk.{bid}.ffn_down_exps.scale", "model.layers.{bid}.router.per_expert_scale")),
}


def _hf_name_map(arch: str, n_layers: int) -> dict[str, str]:
    """gguf tensor name -> HF tensor name, via gguf-py's canonical mapping."""
    import gguf

    model_arch = None
    for candidate in gguf.MODEL_ARCH:
        if gguf.MODEL_ARCH_NAMES.get(candidate) == arch:
            model_arch = candidate
            break
    if model_arch is None:
        raise SystemExit(f"surogate serve: GGUF architecture '{arch}' has no gguf-py mapping.")

    tmap = gguf.get_tensor_name_map(model_arch, n_layers)
    # tmap.mapping is the FORWARD map: {hf_or_gguf alias -> (MODEL_TENSOR,
    # gguf base name)}, with the gguf name itself included as an alias. Invert
    # it, preferring the canonical HF spelling among the aliases.
    # gguf-py lists every HF spelling any architecture has used for a tensor, and the reverse
    # map has to choose one. Root first, then spelling: `model.` outranks `language_model.`
    # (the multimodal wrapper's prefix, which a text-only checkpoint does not carry), and
    # `.mlp.` outranks `.feed_forward.` (used by Llama 4 and afmoe, where every other family
    # says `mlp`). Without the second rank the choice is whichever alias gguf-py happened to
    # list first, which put Gemma 3's up projection under `feed_forward` and lost it.
    def hf_preference(name: str) -> int:
        rank = 0 if name.startswith(("model.", "lm_head")) else (
            1 if name.startswith("language_model.") else 2)
        if ".feed_forward." in name:
            rank += 4
        return rank

    reverse: dict[str, str] = {}
    for alias, (_tid, gguf_base) in tmap.mapping.items():
        if alias == gguf_base:
            continue
        prev = reverse.get(gguf_base)
        if prev is None or hf_preference(alias) < hf_preference(prev):
            reverse[gguf_base] = alias

    fixups = _HF_ALIAS_FIXUPS.get("lfm2" if arch == "lfm2moe" else arch, ())
    if arch == "lfm2moe":
        fixups += (
            ("block_sparse_moe.gate", "feed_forward.gate"),
            ("mlp.experts.gate_proj", "feed_forward.experts.w1"),
            ("mlp.experts.up_proj", "feed_forward.experts.w3"),
            ("mlp.experts.down_proj", "feed_forward.experts.w2"),
        )
    out: dict[str, str] = {}
    # Base names are stored without the trailing ".weight"/".bias"; GGUF tensor
    # names carry the suffix. Emit both suffixed forms.
    for gguf_base, hf_base in reverse.items():
        for wrong, right in fixups:
            hf_base = hf_base.replace(wrong, right)
        for suffix in (".weight", ".bias"):
            out[gguf_base + suffix] = hf_base + suffix
        if arch == "lfm2moe" and gguf_base.endswith(".exp_probs_b"):
            layer = gguf_base.split(".")[1]
            out[gguf_base + ".bias"] = f"model.layers.{layer}.feed_forward.expert_bias"
        # Some checkpoint entries are not `.weight` at all: a buffer, or a stacked parameter
        # the checkpoint stores under its bare name. The generic loop appends a suffix to
        # every base, so those are corrected here rather than guessed at.
        if any(hf_base.endswith(bare) for bare in _BARE_HF_NAMES.get(arch, ())):
            for suffix in (".weight", ".bias"):
                out.pop(gguf_base + suffix, None)
            out[gguf_base + ".weight"] = hf_base
    for gguf_template, hf_template in _SCALE_SIDECARS.get(arch, ()):
        for layer in range(n_layers):
            out[gguf_template.format(bid=layer)] = hf_template.format(bid=layer)
    return out



def _family_or_generic(fam, gguf_name: str, n_main: int, name_map: dict[str, str]) -> str | None:
    """A family's own name if it has one, else gguf-py's.

    gguf-py's `get_tensor_name_map` is the universal table -- it is maintained upstream for
    every architecture llama.cpp supports, so it is the base and a new architecture needs no
    entry here. A family table exists only to *override* the handful of names where the generic
    map's alias preference picks wrong for us (ssm_a -> A_log, attn_gate -> self_attn.gate_proj).
    Anything it does not name falls through, which is how a routed MoE's expert tensors resolve
    without a single hand-written line.
    """
    return fam.hf_name_for(gguf_name, n_main) or name_map.get(gguf_name)


def _first_scalar(value, default: int) -> int:
    """One number from a key that may be a scalar or a per-layer array."""
    if isinstance(value, (list, tuple)):
        return int(value[0]) if len(value) else int(default)
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return int(value.flat[0]) if value.size else int(default)
    except Exception:
        pass
    return int(value or default)


def _rounded_eps(value: float) -> float:
    """A GGUF stores the norm epsilon as float32, so 1e-6 reads back as
    9.999999974752427e-07 and an exact-match config check fails on it. The value is always a
    round decimal in the checkpoint it came from, so read it as one."""
    return float(f"{value:.1e}")


def synthesised_config(reader, arch: str) -> dict | None:
    """`config.json` for an architecture the GGUF fully describes, or None.

    Dimensions and execution settings come from GGUF metadata. Family-specific normalization
    translates those fields into the same configuration consumed by safetensors conversion.
    """
    if arch in ("lfm2", "lfm2moe"):
        from surogate.serve.gguf.lfm2 import config_from_gguf
        return config_from_gguf(reader, arch)
    if arch in ("qwen35", "qwen35moe", "qwen38", "qwen3_5", "qwen3_6",
                "qwen3_8", "qwen3_5_moe", "qwen3_6_moe"):
        from surogate.serve.convert.common.qwen3_5 import config_from_gguf
        config = config_from_gguf(reader, arch)
        for key in ("eos_token_id", "bos_token_id"):
            value = reader.kv(f"tokenizer.ggml.{key}")
            if value is not None:
                config[key] = value
        return config

    def kv(key, default=None):
        return _arch_kv(reader, arch, key, default)

    hidden = int(kv("embedding_length", 0) or 0)
    heads = int(kv("attention.head_count", 0) or 0)
    layers = int(kv("block_count", 0) or 0)
    if not (hidden and heads and layers):
        return None
    tokens = reader.get_field("tokenizer.ggml.tokens")
    if tokens is None:
        return None
    vocab = len(tokens.contents())
    # Whether the *file* ties, which is not always what the original checkpoint said. Qwen3-0.6B
    # declares `tie_word_embeddings: true`, and llama.cpp's converter still writes a separate
    # `output.weight` — at Q6_K, where `token_embd.weight` is Q4_K — because quantising one
    # shared tensor would damage whichever of the two roles wanted the higher precision. The
    # GGUF is the checkpoint here, so it decides.
    tied = reader.tensor("output.weight") is None
    # The engine seeds its stop tokens from these, and a GGUF always carries them.
    special = {}
    for key, member in (("eos_token_id", "eos_token_id"), ("bos_token_id", "bos_token_id"),
                        ("padding_token_id", "pad_token_id")):
        field = reader.get_field(f"tokenizer.ggml.{key}")
        if field is not None:
            special[member] = int(field.contents())
    if "eos_token_id" not in special:
        return None
    common = {
        **special,
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "intermediate_size": _first_scalar(kv("feed_forward_length", 0), 0),
        "num_attention_heads": heads,
        # A family may state this per layer -- Gemma 4 does, because its windowed and global
        # layers differ -- so take the first entry here and let the architecture's own branch
        # below say what each geometry is.
        "num_key_value_heads": _first_scalar(kv("attention.head_count_kv", heads), heads),
        "head_dim": int(kv("attention.key_length", 0) or 0) or hidden // heads,
        "vocab_size": vocab,
        "max_position_embeddings": int(kv("context_length", 0) or 0),
        "rope_theta": float(kv("rope.freq_base", 0.0) or 0.0),
        "rms_norm_eps": _rounded_eps(float(kv("attention.layer_norm_rms_epsilon", 0.0) or 0.0)),
        "tie_word_embeddings": tied,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "rope_scaling": None,
        "torch_dtype": "bfloat16",
        "initializer_range": 0.02,
        "use_cache": True,
    }
    if arch == "qwen3":
        return {**common, "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
                "hidden_act": "silu", "sliding_window": None, "use_sliding_window": False,
                "max_window_layers": layers}
    if arch == "qwen3moe":
        # Qwen3-30B-A3B and its siblings: the same attention as the dense Qwen3 over a routed
        # mixture. Everything the converter reads is in the file -- the expert count, how many
        # a token uses, and their FFN width, which is `expert_feed_forward_length` and not the
        # `feed_forward_length` beside it: this architecture has no dense MLP, and reading that
        # one would size every expert eight times too wide.
        #
        # `expert_shared_feed_forward_length` is what a mixture with an always-on expert states.
        # This family has none, and the zero says so all the way down to the router, which then
        # carries one row per expert and no gate.
        shared = int(kv("expert_shared_feed_forward_length", 0) or 0)
        return {**common,
                "architectures": ["Qwen3MoeForCausalLM"],
                "model_type": "qwen3_moe",
                "hidden_act": "silu",
                "num_experts": int(kv("expert_count", 0) or 0),
                "num_experts_per_tok": int(kv("expert_used_count", 0) or 0),
                "moe_intermediate_size": int(kv("expert_feed_forward_length", 0) or 0),
                "shared_expert_intermediate_size": shared,
                "norm_topk_prob": True,
                "decoder_sparse_step": 1,
                "mlp_only_layers": [],
                "sliding_window": None,
                "use_sliding_window": False,
                "max_window_layers": layers}
    if arch == "llama":
        from surogate.serve.gguf.frontend import extract_generation_config
        return {**common, "architectures": ["LlamaForCausalLM"], "model_type": "llama",
                "eos_token_id": extract_generation_config(reader)["eos_token_id"],
                "hidden_act": "silu", "mlp_bias": False, "pretraining_tp": 1}
    if arch == "gemma3":
        from surogate.serve.gguf.frontend import extract_generation_config
        # Gemma 3 alternates five sliding-window layers with one full-attention layer. The GGUF
        # states the window but not the period, because llama.cpp holds the same 6 as a constant
        # of the architecture; the converter accepts that period in place of a `layer_types` list.
        return {**common,
                "eos_token_id": extract_generation_config(reader)["eos_token_id"],
                "architectures": ["Gemma3ForCausalLM"],
                "model_type": "gemma3_text",
                "hidden_activation": "gelu_pytorch_tanh",
                "sliding_window": int(kv("attention.sliding_window", 0) or 0),
                "sliding_window_pattern": 6,
                "rope_local_base_freq": float(kv("rope.local.freq_base", 10000.0) or 10000.0),
                "query_pre_attn_scalar": common["head_dim"],
                "use_bidirectional_attention": False,
                "attn_logit_softcapping": None,
                "final_logit_softcapping": None}
    if arch == "gemma4":
        return _gemma4_config(reader, kv, common, layers, heads)
    return None


def _as_list(value, layers: int) -> list:
    """A per-layer GGUF array as a list, or a scalar broadcast over the layers.

    Gemma 4 states several quantities per layer where every other family states one: which
    layers look through the window, how many key/value heads each has, and -- on the E-series --
    how wide each feed-forward is. A scalar means "the same everywhere", which is what the
    dense sizes say about their key/value count and what E2B does not.
    """
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return [value] * layers


#: What ends a Gemma 4 turn. `tokenizer.ggml.eos_token_id` names one token; the family stops
#: on three, and llama.cpp names the same three outright (its end-of-generation list, with the
#: entries commented `gemma4`). The chat template closes every assistant turn with `<turn|>`,
#: so a run seeded with `<eos>` alone never stops: the model answers, opens a fresh thought
#: channel and answers again until the token budget runs out.
_GEMMA4_EOG_TOKENS = ("<eos>", "<turn|>", "<|tool_response>")


def _gemma4_eos_ids(reader, stated: int):
    """The turn-ending ids this file actually carries, or the one it stated.

    Looked up by name in the file's own vocabulary rather than assumed: a token this export
    does not have is one it cannot stop on, and inventing an id would stop on whatever else
    happens to sit at that index.
    """
    field = reader.get_field("tokenizer.ggml.tokens")
    if field is None:
        return stated
    index = {str(token): i for i, token in enumerate(field.contents())}
    ids = [index[name] for name in _GEMMA4_EOG_TOKENS if name in index]
    if stated not in ids:
        ids.insert(0, stated)
    return ids if len(ids) > 1 else stated


def _gemma4_config(reader, kv, common: dict, layers: int, heads: int) -> dict:
    """A Gemma 4 config from the GGUF's own metadata.

    Almost nothing here is inferred, which is the difference from Gemma 3 above: llama.cpp
    writes Gemma 4's window schedule as a per-layer boolean array rather than leaving a period
    to be assumed, and it writes both attention geometries -- `key_length` for the global
    layers, `key_length_swa` for the windowed ones -- beside a per-layer key/value head count.
    So the schedule and the two geometries are read, not derived.

    One field is not in the file and is taken as an architecture constant: the global layers'
    `partial_rotary_factor`. llama.cpp states `rope.dimension_count` as the whole 512-wide head
    because its proportional rope appends zero frequencies rather than rotating a prefix, which
    is the same thing said differently and cannot be inverted to the factor. Every published
    Gemma 4 states 0.25.
    """
    windowed = [bool(x) for x in _as_list(kv("attention.sliding_window_pattern", True), layers)]
    kv_heads = [int(x) for x in _as_list(kv("attention.head_count_kv", 1), layers)]
    ffn = [int(x) for x in _as_list(kv("feed_forward_length", 0), layers)]
    global_head = int(kv("attention.key_length", 0) or 0)
    window_head = int(kv("attention.key_length_swa", 0) or 0) or global_head
    # The two key/value counts, taken from the layers that actually use each geometry rather
    # than from a single number that cannot describe both.
    window_kv = next((n for n, w in zip(kv_heads, windowed) if w), kv_heads[0])
    global_kv = next((n for n, w in zip(kv_heads, windowed) if not w), window_kv)
    experts = int(kv("expert_count", 0) or 0)
    per_layer_input = int(kv("embedding_length_per_layer_input", 0) or 0)
    shared_kv = int(kv("attention.shared_kv_layers", 0) or 0)
    # `attention_k_eq_v` is a property of the tensors, not of the metadata: a global layer that
    # reuses its key projection as its value ships no `attn_v`. The 12B ships one on 40 of its
    # 48 layers and the 8 without are exactly the global ones.
    #
    # Only the layers that *own* their key and value can be asked. A shared-KV layer ships no
    # `attn_v` either, for an unrelated reason -- it reads an earlier layer's planes -- and
    # counting one of those reads the E-series as `k_eq_v` when it is not: E2B's globals at 19,
    # 24, 29 and 34 all sit inside its twenty shared layers, so the question was being answered
    # by a layer that has no answer. Its owning globals (4, 9, 14) each ship a value, and the
    # published config agrees: `attention_k_eq_v` is false for the E-series and true for the
    # dense sizes. Read wrongly, three value projections vanish from the artifact and those
    # layers attend to their keys instead -- fluent, confident and wrong.
    owns_kv = layers - shared_kv
    k_eq_v = any(not w and reader.tensor(f"blk.{i}.attn_v.weight") is None
                 for i, w in enumerate(windowed[:owns_kv]))
    # The E-series widens exactly the layers that share their key/value planes, which llama.cpp
    # states as a feed-forward length that differs layer by layer. E2B does; E4B does not.
    dense_ffn = ffn[0] if ffn else 0
    double_wide = bool(ffn) and max(ffn) > min(ffn)
    text = {
        **common,
        "eos_token_id": _gemma4_eos_ids(reader, int(common["eos_token_id"])),
        "model_type": "gemma4_text",
        "architectures": ["Gemma4ForCausalLM"],
        "hidden_activation": "gelu_pytorch_tanh",
        "intermediate_size": dense_ffn,
        "num_key_value_heads": window_kv,
        "num_global_key_value_heads": global_kv,
        "head_dim": window_head,
        "global_head_dim": global_head,
        "layer_types": ["sliding_attention" if w else "full_attention" for w in windowed],
        "sliding_window": int(kv("attention.sliding_window", 0) or 0),
        "attention_k_eq_v": k_eq_v,
        "final_logit_softcapping": float(kv("final_logit_softcapping", 0.0) or 0.0) or None,
        "attn_logit_softcapping": None,
        "rope_parameters": {
            "full_attention": {
                "rope_theta": float(kv("rope.freq_base", 1.0e6) or 1.0e6),
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
            },
            "sliding_attention": {
                "rope_theta": float(kv("rope.freq_base_swa", 1.0e4) or 1.0e4),
                "rope_type": "default",
            },
        },
        "enable_moe_block": experts > 0,
        "num_experts": experts,
        "top_k_experts": int(kv("expert_used_count", 0) or 0),
        "moe_intermediate_size": int(kv("expert_feed_forward_length", 0) or 0),
        "hidden_size_per_layer_input": per_layer_input,
        "vocab_size_per_layer_input": common["vocab_size"] if per_layer_input else 0,
        "num_kv_shared_layers": shared_kv,
        "use_double_wide_mlp": double_wide,
    }
    return text


def _has_export_transform(arch: str, hf_name: str) -> bool:
    """Whether reading this tensor back means undoing something, which decides whether it can
    be moved into the artifact bit-exactly or has to go through the dequantise path."""
    if arch in ("lfm2", "lfm2moe"):
        return hf_name.endswith(".conv.conv.weight")
    if arch == "gemma3":
        return hf_name.endswith("norm.weight")
    if arch == "llama":
        return hf_name.endswith(("self_attn.q_proj.weight", "self_attn.k_proj.weight"))
    return False


def _invert_export_transform(arch: str, hf_name: str, tensor, heads: int, kv_heads: int):
    """Undo what llama.cpp's converter did to a tensor's *values* on the way in.

    A GGUF is not a renamed checkpoint. `conversion/` folds things into the weights so its
    runtime does not have to, and reading the file back means undoing them. Two here, both
    quoted from that source:

    * Gemma folds the +1 its norm applies (`conversion/gemma.py`: `if
      name.endswith("norm.weight"): data_torch = data_torch + 1`), so a Gemma GGUF's norm is
      the checkpoint's plus one — checked exactly against `gemma-3-270m-it`, difference 1.0
      with no error anywhere in the tensor.
    * Llama permutes Q and K so its rotary can read halves contiguously
      (`conversion/llama.py: permute`). The inverse is the same reshape with the swap the
      other way round. It is a row permutation, not arithmetic, but it still has to happen
      before the rows mean anything.

    Left alone, neither is loud: TinyLlama answered "The capital of France is" with fluent,
    confident, wrong text, and Gemma 3 produced multilingual noise.
    """
    if arch in ("lfm2", "lfm2moe") and hf_name.endswith(".conv.conv.weight"):
        # llama.cpp squeezes the depthwise channel axis: [hidden, 1, taps] -> [hidden, taps].
        return tensor.unsqueeze(1) if tensor.ndim == 2 else tensor
    if arch == "gemma3":
        # Every norm, and only norms: `_norm.weight` covers input/post/pre/final and the
        # per-head q_norm/k_norm, all of which Gemma's converter folds.
        if hf_name.endswith("norm.weight"):
            return tensor - 1.0
        return tensor
    if arch == "llama":
        if hf_name.endswith("self_attn.q_proj.weight"):
            return _unpermute(tensor, heads)
        if hf_name.endswith("self_attn.k_proj.weight"):
            return _unpermute(tensor, kv_heads)
        return tensor
    return tensor


def _unpermute(tensor, heads: int):
    """The inverse of llama.cpp's Q/K permutation: split each head's rows into two halves and
    interleave them back, which is `permute`'s reshape with the axes swapped the other way."""
    rows = tensor.shape[0]
    return (tensor.reshape(heads, rows // heads // 2, 2, *tensor.shape[1:])
            .swapaxes(1, 2)
            .reshape(tensor.shape))



def build_hf_dir_from_gguf(
    gguf_path: Path,
    target_key: str,
    work_dir: Path,
    *,
    repack_planner=None,
    reader=None,
    echo=print,
) -> Path:
    """Materialize a temporary HF-layout model dir from a GGUF file.

    With a ``repack_planner`` (PATCHES.md #14), 2D Q8_0 tensors whose
    inverse transform is a row identity become repack CANDIDATES; the planner
    (backed by the converter's own recipes) returns the subset its artifact
    profile actually repacks — e.g. Q8_0 sources of BF16-profile objects stay
    on the dequant path. Planned tensors are not dequantized; they are
    recorded in ``gguf_repack.json`` for a bit-exact move into the artifact."""
    import numpy as np
    import torch
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize
    from safetensors.torch import save_file

    work_dir.mkdir(parents=True, exist_ok=True)
    if reader is None:
        reader = open_gguf(gguf_path)
    payload_mm = np.memmap(gguf_path, dtype=np.uint8, mode="r")
    arch = reader.get_field("general.architecture").contents()

    # Frontend and configuration both come from this GGUF's metadata.
    from surogate.serve.gguf.frontend import write_frontend

    write_frontend(reader, arch, work_dir, echo=echo)
    derived = synthesised_config(reader, arch)
    if derived is None:
        raise SystemExit(f"surogate serve: architecture '{arch}' has no checkpoint config normalizer")
    (work_dir / "config.json").write_text(json.dumps(derived, indent=2))
    echo(f"surogate serve: config.json synthesized from the GGUF's own metadata ({arch})")

    # 2. Dequantize tensors to BF16 and write sharded safetensors with HF names.
    n_layers = int(_arch_kv(reader, arch, "block_count", 0))

    # Family-specific handling: llama.cpp does NOT store HF-layout tensors for
    # the qwen35 family — it folds norms (+1), stores -exp(A_log), renames
    # dt_bias, squeezes conv1d, reorders V heads, and remaps mtp.* into extra
    # layers. surogate/serve/gguf/qwen35.py inverts all of that; skipping it
    # would produce silently damaged weights.
    qwen35_family = arch in ("qwen35", "qwen35moe")
    if qwen35_family:
        from surogate.serve.gguf import qwen35 as fam

        n_mtp = int(_arch_kv(reader, arch, "nextn_predict_layers", 0) or 0)
        n_main = n_layers - n_mtp
        num_v = int(_arch_kv(reader, arch, "ssm.time_step_rank", 0) or 0)
        inner = int(_arch_kv(reader, arch, "ssm.inner_size", 0) or 0)
        geom = fam.GdnGeometry(
            num_k_heads=int(_arch_kv(reader, arch, "ssm.group_count", 0) or 0),
            num_v_heads=num_v,
            head_k_dim=int(_arch_kv(reader, arch, "ssm.state_size", 0) or 0),
            head_v_dim=(inner // num_v) if num_v else 0,
        )
        echo(f"surogate serve: qwen35 inverse transforms active "
             f"(layers {n_main}+{n_mtp} mtp, GDN {geom.num_k_heads}k/{geom.num_v_heads}v)")
        if n_mtp == 0:
            # Community exports frequently strip nextn; the converter emits
            # the no-MTP artifact variant and the engine refuses --spec mtp
            # with a clear error (PATCHES.md #15).
            echo(
                "surogate serve: this GGUF was exported without the MTP (nextn) "
                "block — converting the no-MTP artifact variant; speculative "
                "decode (--spec mtp) will be unavailable for it."
            )
    name_map = _hf_name_map(arch, n_layers)
    # These feed one thing only -- `_invert_export_transform`'s Llama Q/K unpermute -- and a
    # family that states them per layer, as Gemma 4 does for its two attention geometries, has
    # no such transform to invert. So one number is enough, and where the file states a list
    # `_first_scalar` takes its head rather than failing on a key this arch never reads.
    export_heads = _first_scalar(_arch_kv(reader, arch, "attention.head_count", 0), 0)
    export_kv_heads = _first_scalar(
        _arch_kv(reader, arch, "attention.head_count_kv", export_heads), export_heads
    )

    # Pre-walk: collect candidates, let the converter's recipes pick the
    # subset it will repack; everything else takes the dequant path below.
    repack_sources: dict[str, str] = {}
    if repack_planner is not None:
        # Candidates carry every 2D row-identity tensor with its GGUF type;
        # the planner (backed by the converter's repack module) keeps only the
        # types that move into the artifact profile bit-exactly.
        candidates: dict[str, dict] = {}
        for tensor in reader.tensors:
            hf = (_family_or_generic(fam, tensor.name, n_main, name_map) if qwen35_family
                  else name_map.get(tensor.name))
            if hf is None or len(tensor.shape) < 2:
                continue
            # GGUF ne order is innermost-first, so the checkpoint shape is its reverse.
            # Rank is carried whole: a routed MoE stacks its experts, and [experts, out, in]
            # is [experts*out, in] to the row algebra.
            shape = tuple(int(extent) for extent in reversed(tensor.shape))
            rows = 1
            for extent in shape[:-1]:
                rows *= extent
            # A tensor whose inverse is a row permutation is still readable from the file: the
            # candidate carries the map and the planner turns it into runs. Only a *value*
            # transform forces the dequantise path.
            row_perm = (
                fam.inverse_row_permutation(hf, geom, rows) if qwen35_family else None
            )
            # A column permutation cannot be a gather -- runs describe rows -- but when it moves
            # whole quantisation groups the loader carries it as a map instead.
            col_groups = (
                fam.inverse_column_group_map(hf, geom, 32) if qwen35_family else None
            )
            if (
                qwen35_family
                and row_perm is None
                and col_groups is None
                and not fam.inverse_is_row_identity(hf, geom)
            ):
                continue
            candidates[hf] = {
                "name": tensor.name,
                "shape": list(shape),
                "rows": rows,
                "k": shape[-1],
                "offset": int(tensor.data_offset),
                "type": tensor.type_name,
                "row_perm": None if row_perm is None else [int(v) for v in row_perm],
                "col_groups": None if col_groups is None else [int(v) for v in col_groups],
            }
        for name in list(candidates):
            if _has_export_transform(arch, name):
                candidates.pop(name)
        repack_sources = repack_planner(gguf_path, candidates)
        if set(repack_sources) - set(candidates):
            raise SystemExit("surogate serve: repack planner returned non-candidate sources.")

    weight_map: dict[str, str] = {}
    total_bytes = 0
    shard_idx = 0
    shard: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    n_tensors = len(reader.tensors)

    def flush():
        nonlocal shard, shard_bytes, shard_idx
        if not shard:
            return
        shard_idx += 1
        fname = f"model-{shard_idx:05d}.safetensors"
        save_file(shard, str(work_dir / fname))
        for key in shard:
            weight_map[key] = fname
        shard = {}
        shard_bytes = 0

    for i, tensor in enumerate(reader.tensors):
        if qwen35_family:
            hf_name = _family_or_generic(fam, tensor.name, n_main, name_map)
        else:
            hf_name = name_map.get(tensor.name)
        if hf_name is None:
            raise SystemExit(
                f"surogate serve: GGUF tensor '{tensor.name}' has no HF mapping for "
                f"arch '{arch}' — refusing rather than dropping weights."
            )
        if hf_name in repack_sources:
            continue
        payload = reader.payload_view(tensor, payload_mm)
        data = dequantize(payload, GGMLQuantizationType(tensor.type_id))
        # GGUF stores dims innermost-first; HF convention is the reverse.
        array = np.ascontiguousarray(data.reshape(tuple(reversed(tensor.shape))))
        t = torch.from_numpy(array)
        if qwen35_family:
            # Undo llama.cpp's export transforms (fp32 math, then narrow).
            t = fam.invert_tensor(hf_name, t, geom)
        else:
            t = _invert_export_transform(arch, hf_name, t, export_heads, export_kv_heads)
        # LFM2-MoE's expert-selection correction is trained/stored in FP32.
        dtype = torch.float32 if arch == "lfm2moe" and hf_name.endswith(".expert_bias") else torch.bfloat16
        t = t.to(dtype)
        shard[hf_name] = t
        shard_bytes += t.numel() * t.element_size()
        total_bytes += t.numel() * t.element_size()
        if shard_bytes >= _SHARD_BYTES:
            flush()
        if (i + 1) % 100 == 0 or i + 1 == n_tensors:
            echo(f"surogate serve: dequantized {i + 1}/{n_tensors} tensors "
                 f"({total_bytes / (1 << 30):.1f} GiB BF16)")
    flush()
    del reader

    index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
    (work_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=1))
    if repack_sources:
        (work_dir / "gguf_repack.json").write_text(
            json.dumps(
                {"gguf_path": str(gguf_path.resolve()), "sources": repack_sources},
                indent=1,
            )
        )
        # Name the types the file actually holds. This said "Q8_0" from when that was the
        # only one repacked, and went on saying it for every K-quant since -- a Gemma 4 QAT
        # export is Q4_0 throughout, and the line called all 328 of them Q8_0.
        kinds = sorted({entry.get("type", "?") for entry in repack_sources.values()
                        if isinstance(entry, dict)})
        echo(
            f"surogate serve: {len(repack_sources)} "
            f"{'/'.join(kinds) if kinds else 'quantised'} tensors marked for "
            f"bit-exact repack (dequantized only {len(weight_map)})"
        )
    return work_dir


def gguf_converter_key(gguf_path: Path, reader=None):
    """Which converter module builds this GGUF's artifact.

    The engine target's own name. One target per architecture and one converter to match, so
    the two agree; the vendored `resources/<key>/` directories are keyed the same way.
    """
    return gguf_target_key(gguf_path, reader)


def gguf_target_key(gguf_path: Path, reader=None):
    """Map a GGUF file to a registered converter target key, or None.

    Architecture strings follow llama.cpp/gguf-py naming: the Qwen3.5/3.6
    family (both Qwen3_5ForCausalLM) is 'qwen35'; MoE is 'qwen35moe'. Older
    HF-style spellings are accepted defensively. Validated against a real
    Qwen3.6-27B GGUF before this path is called supported.
    """
    s = read_gguf_summary(gguf_path, reader)
    arch = s["architecture"]
    hidden = int(s["hidden_size"] or 0)
    layers = int(s["num_hidden_layers"] or 0)
    # Qwen3.5, 3.6 and 3.8 are one interleaved gated-delta architecture at different sizes, so
    # they are one target and one converter; the artifact declares the dimensions it binds
    # against, and the architecture string tells 3.8 apart where the dimensions cannot.
    # Qwen3-30B-A3B and its siblings: plain attention over a routed mixture with no always-on
    # expert. llama.cpp spells it `qwen3moe`, which is one character from the interleaved
    # gated-delta family's `qwen35moe` and was once folded into it.
    if arch == "qwen3moe" and hidden > 0 and layers > 0:
        return "qwen3_moe"
    # `qwen3moe` is deliberately not in the list below. It is llama.cpp's name for Qwen3-30B-A3B --
    # plain attention with a routed mixture and no always-on expert -- and it was accepted here
    # as a defensive spelling of the interleaved gated-delta family, which would have bound a
    # 48-layer dense-attention checkpoint against a target that expects a linear mixer at three
    # layers in four. Nothing spells the 3.5 family that way; the collision was the whole of the
    # reason it was listed.
    if arch in ("qwen35moe", "qwen3_6_moe", "qwen3_5_moe") and hidden > 0:
        return "qwen3_5_moe"
    if arch in ("qwen35", "qwen3_5", "qwen3_6", "qwen38", "qwen3_8") and hidden > 0 and layers > 0:
        return "qwen3_5"
    # Dense decoders whose engine target is one compiled geometry. The gates below are that
    # geometry: a differently sized Qwen3 or Llama has no target to be served by yet, and is
    # refused with the summary rather than converted against the wrong config.
    # Any size of the plain dense Qwen3: the target reads its dimensions from the artifact
    # rather than compiling them, so what has to match is the architecture, not the size.
    if arch == "qwen3" and hidden > 0 and layers > 0:
        return "qwen3"
    if arch == "llama" and hidden > 0 and layers > 0:
        return "llama"
    if arch == "lfm2" and hidden > 0 and layers > 0:
        return "lfm2"
    if arch == "lfm2moe" and hidden > 0 and layers > 0:
        return "lfm2_moe"
    if arch == "gemma3" and hidden > 0 and layers > 0:
        return "gemma3"
    # Gemma 4. llama.cpp gives all five published checkpoints one architecture string, exactly
    # as their HF configs give them one `model_type`, so the shape of the model picks the
    # target here for the same reasons it does in `ingest.converter_for_config`: a mixture and
    # an E-series are different architectures wearing the same name, and deriving either from
    # the dense blocks builds an artifact that is wrong rather than absent.
    if arch == "gemma4" and hidden > 0 and layers > 0:
        experts = int(s.get("num_experts") or 0)
        if experts > 0:
            return "gemma4_moe"
        # The E-series is told from the dense sizes by the two things only it carries: a
        # per-layer input embedding, and a tail of layers that project a query and nothing else.
        if int(s.get("per_layer_input_dim") or 0) > 0 or int(s.get("kv_shared_layers") or 0) > 0:
            return "gemma4_e"
        return "gemma4"
    if arch == "qwen4exp":
        # Qwen3.8-Flash-Next: converted straight from the GGUF (no HF bridge).
        return "qwen4exp"
    if arch == "glm5next" and hidden > 0 and layers > 0:
        return "glm5_next"
    return None
