# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# GGUF ingest for `surogate serve` (design/serve-engine-plan.md §5.1, §5.2b).
#
# Strategy (v0): bridge GGUF to the exact input the vendored converter already
# accepts — a temporary HF-layout directory with BF16 safetensors shards. Fully
# offline: the tokenizer and chat template are reconstructed from the GGUF's
# own KV metadata (frontend.py), the checkpoint-invariant config files come
# from vendored per-target resources (surogate/serve/resources/), and family
# modules (qwen35.py) invert llama.cpp's export transforms. The converter then
# runs with its own preflight checks (SINFER_ALLOW_DERIVED_FRONTEND downgrades
# the tokenizer pinned-hash check to a recorded warning — the reconstruction is
# semantically equivalent, not byte-identical). Costs one temporary BF16
# materialization on disk (~2 bytes/param, deleted after conversion); a
# reader-injection path that avoids the temp copy is a tracked follow-up.
# K-quant sources are dequantized to BF16 and re-encoded by the recipe (one
# documented double quantization; native K-quant repack is the plan's §5.2 path).

from __future__ import annotations

import json
import shutil
from pathlib import Path

# Checkpoint-invariant static resources per registered target, vendored under
# surogate/serve/resources/<target_key>/ (see its README for provenance). The
# tokenizer and chat template are NOT static files — they are reconstructed
# from the GGUF itself (frontend.py) so local GGUF serving stays fully offline.
_RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"

_STATIC_RESOURCE_FILES = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
]

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
    }
    del reader
    return summary


#: gguf-py lists several HF aliases for one tensor, and `hf_preference` below cannot tell which
#: spelling a given family's checkpoint actually uses — it only prefers `model.`-rooted names.
#: Where the alias it lands on is not the one the converter's recipe names, say so here. These
#: are substring rewrites on the HF side of the map, applied after it is built.
_HF_ALIAS_FIXUPS: dict[str, tuple[tuple[str, str], ...]] = {
    # Qwen3's per-head norms are `q_norm`/`k_norm` in the checkpoint; the generic map reaches
    # them by their `q_layernorm`/`k_layernorm` alias.
    "qwen3": (("self_attn.q_layernorm", "self_attn.q_norm"),
              ("self_attn.k_layernorm", "self_attn.k_norm")),
    # Gemma 3 spells them the same way Qwen3 does.
    "gemma3": (("self_attn.q_layernorm", "self_attn.q_norm"),
               ("self_attn.k_layernorm", "self_attn.k_norm")),
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

    fixups = _HF_ALIAS_FIXUPS.get(arch, ())
    out: dict[str, str] = {}
    # Base names are stored without the trailing ".weight"/".bias"; GGUF tensor
    # names carry the suffix. Emit both suffixed forms.
    for gguf_base, hf_base in reverse.items():
        for wrong, right in fixups:
            hf_base = hf_base.replace(wrong, right)
        for suffix in (".weight", ".bias"):
            out[gguf_base + suffix] = hf_base + suffix
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


def _rounded_eps(value: float) -> float:
    """A GGUF stores the norm epsilon as float32, so 1e-6 reads back as
    9.999999974752427e-07 and an exact-match config check fails on it. The value is always a
    round decimal in the checkpoint it came from, so read it as one."""
    return float(f"{value:.1e}")


def synthesised_config(reader, arch: str) -> dict | None:
    """`config.json` for an architecture the GGUF fully describes, or None.

    The Qwen3.5 family's config carries things no GGUF holds (per-layer `layer_types`, the
    attention output gate), so those targets vendor a file under `serve/resources/`. A plain
    dense decoder does not: every member its converter reads is either a constant of the
    architecture or a number in the GGUF's own metadata. Synthesising it keeps a family from
    needing one vendored config per model size, and keeps us from vendoring config files whose
    licence is not ours to vendor.
    """
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
        "intermediate_size": int(kv("feed_forward_length", 0) or 0),
        "num_attention_heads": heads,
        "num_key_value_heads": int(kv("attention.head_count_kv", heads) or heads),
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
    if arch == "llama":
        return {**common, "architectures": ["LlamaForCausalLM"], "model_type": "llama",
                "hidden_act": "silu", "mlp_bias": False, "pretraining_tp": 1}
    if arch == "gemma3":
        # Gemma 3 alternates five sliding-window layers with one full-attention layer. The GGUF
        # states the window but not the period, because llama.cpp holds the same 6 as a constant
        # of the architecture; the converter accepts that period in place of a `layer_types` list.
        return {**common,
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
    return None


def _has_export_transform(arch: str, hf_name: str) -> bool:
    """Whether reading this tensor back means undoing something, which decides whether it can
    be moved into the artifact bit-exactly or has to go through the dequantise path."""
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



def _apply_gguf_dimensions(config: dict, reader, arch: str) -> bool:
    """Overwrite a vendored config's text dimensions with this GGUF's own. True if any moved.

    A vendored config is one size of a family. The file being converted may be another, and
    every dimension the converter reads has to describe the file, not the template.
    """
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config

    def kv(suffix, default=None):
        try:
            value = reader.kv(f"{arch}.{suffix}")
        except Exception:
            return default
        return default if value is None else int(value)

    heads = kv("attention.head_count")
    inner = kv("ssm.inner_size")
    state = kv("ssm.state_size")
    moved = False
    updates = {
        # the text core: llama.cpp counts the MTP (nextn) block among the blocks
        "num_hidden_layers": (kv("block_count") or 0) - (kv("nextn_predict_layers") or 0),
        "hidden_size": kv("embedding_length"),
        "intermediate_size": kv("feed_forward_length"),
        "num_attention_heads": heads,
        "num_key_value_heads": kv("attention.head_count_kv"),
        "head_dim": kv("attention.key_length"),
        "linear_num_key_heads": kv("ssm.group_count"),
        "linear_key_head_dim": state,
        "linear_value_head_dim": state,
        "linear_conv_kernel_dim": kv("ssm.conv_kernel"),
        "linear_num_value_heads": (inner // state) if inner and state else None,
    }
    for name, value in updates.items():
        if value is None or name not in text or text[name] == value:
            continue
        text[name] = value
        moved = True
    # The layer schedule is one entry per layer, so a template written for another size states
    # the wrong number of them. It is not free-form: full attention falls on every fourth layer
    # from the fourth, and the converter refuses a list that says otherwise.
    layers = updates["num_hidden_layers"]
    if layers and isinstance(text.get("layer_types"), list) and len(text["layer_types"]) != layers:
        interval = 4
        text["layer_types"] = [
            "full_attention" if layer >= interval - 1 and (layer - (interval - 1)) % interval == 0
            else "linear_attention"
            for layer in range(layers)
        ]
        moved = True
    # Two generations share these dimensions and differ only in how their exports quantise, so
    # the architecture is the only thing that tells them apart.
    if arch in ("qwen38", "qwen3_8"):
        for holder in (config, text):
            if holder.get("model_type", "").startswith("qwen3_5"):
                holder["model_type"] = holder["model_type"].replace("qwen3_5", "qwen3_8")
                moved = True
    return moved


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

    # 1. Frontend: tokenizer + chat template reconstructed from the GGUF
    #    itself (fully offline); checkpoint-invariant config files from the
    #    vendored per-target resources.
    from surogate.serve.gguf.frontend import write_frontend

    write_frontend(reader, arch, work_dir, echo=echo)
    static_dir = _RESOURCES_DIR / target_key
    for fname in _STATIC_RESOURCE_FILES:
        src = static_dir / fname
        if src.is_file():
            shutil.copy(src, work_dir / fname)
    if (work_dir / "config.json").is_file():
        # A vendored config describes one size of the family -- the tower, the rope
        # parameters, the special token ids -- and this file may be another size. Take the
        # dimensions from the GGUF, which is the checkpoint actually being converted, and
        # leave everything else as vendored.
        vendored = json.loads((work_dir / "config.json").read_text(encoding="utf-8"))
        if _apply_gguf_dimensions(vendored, reader, arch):
            (work_dir / "config.json").write_text(json.dumps(vendored, indent=2))
            echo("surogate serve: config dimensions taken from the GGUF, the rest vendored")
    else:
        derived = synthesised_config(reader, arch)
        if derived is None:
            raise SystemExit(
                f"surogate serve: missing vendored config.json for target '{target_key}' "
                f"(expected at {static_dir}), and architecture '{arch}' has no synthesised one."
            )
        (work_dir / "config.json").write_text(json.dumps(derived, indent=2))
        echo(f"surogate serve: config.json synthesised from the GGUF's own metadata ({arch})")

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
    export_heads = int(_arch_kv(reader, arch, "attention.head_count", 0) or 0)
    export_kv_heads = int(_arch_kv(reader, arch, "attention.head_count_kv", export_heads)
                          or export_heads)

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
        t = t.to(torch.bfloat16)
        shard[hf_name] = t
        shard_bytes += t.numel() * 2
        total_bytes += t.numel() * 2
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
        echo(
            f"surogate serve: {len(repack_sources)} Q8_0 tensors marked for "
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
    # `qwen3moe` is deliberately not in this list. It is llama.cpp's name for Qwen3-30B-A3B --
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
    if arch == "gemma3" and hidden > 0 and layers > 0:
        return "gemma3"
    if arch == "qwen4exp":
        # Qwen3.8-Flash-Next: converted straight from the GGUF (no HF bridge).
        return "qwen4exp"
    return None
