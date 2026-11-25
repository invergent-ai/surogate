"""The complete ordered object inventory of an Inkling-2 artifact (ARTIFACT.md section 4).

Each entry names its artifact object, its stored format and shape, and the checkpoint tensors it
is made of. The converter builds payloads from it and the verifier checks every object against
its sources through an independent decode, so the mapping is stated exactly once, here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .checkpoint import MAIN_PREFIX, VISION_PREFIX, Geometry
from .fp8_attention import SCHEMES, STORAGE_FORMAT

CONTIGUOUS = "contiguous-le-v1"
NVFP4_LAYOUT = "blockscale-k16-m128x4-v1"
MXFP8_LAYOUT = "mxfp8-k32-v1"
ROW_F32_LAYOUT = "row-scale-f32-v1"
GRID_LAYOUT = "fp8-1x128-f32-v1"
LAYOUT_OF = {"BF16": CONTIGUOUS, "FP32": CONTIGUOUS, "NVFP4": NVFP4_LAYOUT, "MXFP8": MXFP8_LAYOUT,
             "FP8_E4M3FN_ROW_F32S": ROW_F32_LAYOUT, "FP8_E4M3FN_1X128_F32S": GRID_LAYOUT}
#: Prefix of a source in ``mtp.safetensors`` (the main file's names start with ``model.``).
MTP = "mtp:"
GATE_UP_ORDERS = ("interleaved", "split")
LOOP_STREAMS = ("attn", "k", "v", "mlp")


@dataclass(frozen=True)
class Entry:
    name: str
    kind: str  # direct | nvfp4 | mxfp8 | fp8
    format: str
    shape: tuple[int, ...]
    #: Checkpoint tensors in order. direct: the object is their flattened concatenation.
    #: nvfp4 / mxfp8: (codes, block scales). fp8: the BF16 slices quantised one by one.
    sources: tuple[str, ...]
    extra: dict = field(default_factory=dict)

    @property
    def layout(self) -> str:
        return LAYOUT_OF[self.format]


def _t(name: str) -> str:
    return MAIN_PREFIX + name


def _direct(entries: list[Entry], name: str, shape, sources, fmt: str = "BF16") -> None:
    entries.append(Entry(name, "direct", fmt, tuple(int(d) for d in shape), tuple(sources)))


def parse_attention(option: str) -> tuple[str, ...]:
    """``bf16[,fp8-<scheme>...]`` -> the FP8 schemes (BF16 is always carried)."""
    schemes: list[str] = []
    for item in (part.strip() for part in option.split(",") if part.strip()):
        if item == "bf16":
            continue
        if not item.startswith("fp8-") or item[4:] not in SCHEMES:
            raise ValueError(f"--attention item {item!r}: expected bf16 or fp8-<{'|'.join(SCHEMES)}>")
        if item[4:] not in schemes:
            schemes.append(item[4:])
    return tuple(schemes)


def vision_geometry(vision: Mapping, text_hidden: int) -> dict[str, int | float]:
    """The root ``vision_geometry`` member for the checkpoint's ``vision_config`` (a PE-style ViT, a 2 x 2 merge)."""
    hidden, heads = int(vision["hidden_size"]), int(vision["num_attention_heads"])
    patch, merge = int(vision["native_patch_size"]), int(vision["adapter_merge_size"])
    grid = int(vision["posemb_grid_size"])
    if (int(vision["text_hidden_size"]) != text_hidden or merge != 2 or vision.get("projector_bias", False) or
            int(vision.get("temporal_patch_size", 2)) != 2 or hidden % heads or (hidden // heads) % 4):
        raise ValueError(f"unsupported Inkling-2 vision_config: {dict(vision)}")
    return {
        "inkling": 1, "layers": int(vision["num_hidden_layers"]), "hidden": hidden,
        "intermediate": int(hidden * float(vision["mlp_ratio"])), "heads": heads,
        "patch_dim": 3 * patch * patch, "merge": merge, "position_embeddings": grid * grid,
        # 2D RoPE over the whole head: x on the first half of its pairs, y on the second (vLLM's apply_rotary_qk_).
        "rotary_dim": hidden // heads, "rope_theta": 10000.0,
        "projector_hidden": int(vision["projection_hidden_dim"]), "output_hidden": text_hidden,
        # vLLM's per-image encoder-cache cap at --max-num-batched-tokens 4096 (inkling_vision.h).
        "max_image_tokens": 4096, "norm_epsilon": float(vision["layer_norm_eps"]),
    }


def _v(name: str) -> str:
    return VISION_PREFIX + name


def vision_entries(e: list[Entry], vision: Mapping, text_hidden: int) -> None:
    """``vision/*``: every ``model.visual.*`` tensor, BF16 bit copies (q / k / v stay separate)."""
    vg = vision_geometry(vision, text_hidden)
    h, m = vg["hidden"], vg["intermediate"]
    b = "backbone."
    _direct(e, "vision/patch_embedding", (h, vg["patch_dim"]), [_v(b + "embeddings.patch_embedding.weight")])
    _direct(e, "vision/position_embedding", (vg["position_embeddings"], h), [_v(b + "embeddings.position_embedding")])
    _direct(e, "vision/class_embedding", (h,), [_v(b + "embeddings.class_embedding")])
    _direct(e, "vision/cls_position_embedding", (h,), [_v(b + "embeddings.cls_position_embedding")])
    for part in ("weight", "bias"):
        _direct(e, f"vision/pre_norm/{part}", (h,), [_v(b + f"ln_pre.{part}")])
    for i in range(vg["layers"]):
        p, src = f"vision/layers/{i}/", _v(b + f"layers.{i}.")
        for norm, source in (("norm1", "ln_1"), ("norm2", "ln_2")):
            for part in ("weight", "bias"):
                _direct(e, p + f"{norm}/{part}", (h,), [src + f"{source}.{part}"])
        for role, source in (("query", "q_proj"), ("key", "k_proj"), ("value", "v_proj"), ("output", "out_proj")):
            _direct(e, p + f"attention/{role}", (h, h), [src + f"self_attn.{source}.weight"])
            _direct(e, p + f"attention/{role}_bias", (h,), [src + f"self_attn.{source}.bias"])
        _direct(e, p + "mlp/fc1", (m, h), [src + "mlp.c_fc.weight"])
        _direct(e, p + "mlp/fc1_bias", (m,), [src + "mlp.c_fc.bias"])
        _direct(e, p + "mlp/fc2", (h, m), [src + "mlp.c_proj.weight"])
        _direct(e, p + "mlp/fc2_bias", (h,), [src + "mlp.c_proj.bias"])
        _direct(e, p + "ls1", (h,), [src + "ls_1.gamma"])
        _direct(e, p + "ls2", (h,), [src + "ls_2.gamma"])
    _direct(e, "vision/projector/in", (vg["projector_hidden"], 4 * h), [_v("proj_in.weight")])
    _direct(e, "vision/projector/out", (text_hidden, vg["projector_hidden"]), [_v("proj_out.weight")])


def build_inventory(g: Geometry, *, fp8_schemes: tuple[str, ...] = (), gate_up_order: str = "interleaved",
                    mtp: bool = True, vision: Mapping | None = None) -> list[Entry]:
    if gate_up_order not in GATE_UP_ORDERS:
        raise ValueError(f"gate_up_order must be one of {GATE_UP_ORDERS}")
    h, e = g.hidden, []
    _direct(e, "text/embedding", (g.vocab_padded, h), [_t("embed.weight")])
    _direct(e, "text/embedding_norm", (h,), [_t("embed_norm.weight")])
    _direct(e, "text/final_norm", (h,), [_t("norm.weight")])
    _direct(e, "text/output_head", (g.vocab_padded, h), [_t("unembed.weight")])
    for layer in range(1, g.layers):
        rows = 3 if g.is_global(layer) else 1
        _direct(e, f"text/attn_res/attn_query/{layer}", (rows, h),
                [_t(f"attn_res.attn_queries.{layer - 1}")], "FP32")
    for layer in range(g.layers):
        _direct(e, f"text/attn_res/mlp_query/{layer}", (1, h), [_t(f"attn_res.mlp_queries.{layer}")], "FP32")
    _direct(e, "text/attn_res/output_query", (1, h), [_t("attn_res.output_query")], "FP32")
    core = g.core_end - g.core_start
    _direct(e, "text/loop/cross_pass_queries", (2 * core, h),
            [_t("middle_cycle_cross_pass_queries")], "FP32")
    _direct(e, "text/loop/cross_pass_kv_queries", (2 * core, h),
            [_t("middle_cycle_cross_pass_kv_queries")], "FP32")

    qkvr_rows = sum(rows for _, rows in g.qkvr_slices)
    for layer in range(g.layers):
        p, src = f"text/layers/{layer}/", f"layers.{layer}."
        if g.is_global(layer):
            _direct(e, p + "attn/input_norm", (3, h),
                    [_t(src + f"{s}_input_norm.weight") for s in ("q", "k", "v")])
        else:
            _direct(e, p + "attn/input_norm", (1, h), [_t(src + "attn_norm.weight")])
        qkvr_sources = [_t(src + f"attn.{name}.weight") for name, _ in g.qkvr_slices]
        _direct(e, p + "attn/qkvr", (qkvr_rows, h), qkvr_sources)
        _direct(e, p + "attn/output", (h, g.q_rows), [_t(src + "attn.wo_ud.weight")])
        _direct(e, p + "attn/q_scale", (g.query_heads, g.head_dim), [_t(src + "attn.q_l2_clip.scale")])
        _direct(e, p + "attn/r_scale", (g.query_heads, g.d_rel), [_t(src + "attn.rel_l2_clip.scale")])
        _direct(e, p + "attn/rel_proj", (g.d_rel, g.rel_extent), [_t(src + "attn.rel_logits_proj.proj")])
        k = g.conv_kernel
        if g.is_global(layer):
            for s in ("k", "v"):
                _direct(e, p + f"attn/{s}_conv_static", (k, g.kv_rows), [_t(src + f"attn.{s}_sconv.static_w")])
                _direct(e, p + f"attn/{s}_conv_proj", (g.kv_heads * k, h),
                        [_t(src + f"attn.{s}_sconv.weight_proj")])
        else:
            for s in ("k", "v"):
                _direct(e, p + f"attn/{s}_conv", (g.kv_rows, k), [_t(src + f"attn.{s}_sconv.weight")])
        for scheme in fp8_schemes:
            fmt = STORAGE_FORMAT[scheme]
            e.append(Entry(p + f"attn/fp8-{scheme}/qkvr", "fp8", fmt, (qkvr_rows, h),
                           tuple(qkvr_sources), {"scheme": scheme}))
            e.append(Entry(p + f"attn/fp8-{scheme}/output", "fp8", fmt, (h, g.q_rows),
                           (_t(src + "attn.wo_ud.weight"),), {"scheme": scheme}))
        _direct(e, p + "attn_out_conv", (h, k), [_t(src + "attn_sconv.weight")])
        _direct(e, p + "mlp_out_conv", (h, k), [_t(src + "mlp_sconv.weight")])
        _direct(e, p + "moe/input_norm", (h,), [_t(src + "mlp_norm.weight")])
        _direct(e, p + "moe/router", (g.experts, h), [_t(src + "mlp.gate.weight")])
        _direct(e, p + "moe/router_bias", (g.experts,), [_t(src + "mlp.gate.bias")], "FP32")
        experts = src + "mlp.experts."
        shapes = {"w13": (2 * g.intermediate, h), "w2": (h, g.intermediate)}
        for proj in ("w13", "w2"):
            n, kk = shapes[proj]
            extra = {"layer": layer, "projection": proj,
                     "gate_up_order": gate_up_order if proj == "w13" else "n/a",
                     "per_expert_rows": n, "experts": g.experts}
            if g.expert_format[layer] == "nvfp4":
                e.append(Entry(p + f"moe/{proj}", "nvfp4", "NVFP4", (g.experts * n, kk),
                               (_t(experts + f"{proj}_weight"), _t(experts + f"{proj}_weight_scale")), extra))
                _direct(e, p + f"moe/{proj}_amax", (g.experts,), [_t(experts + f"{proj}_weight_amax")], "FP32")
                _direct(e, p + f"moe/{proj}_outer_scale_denominator", (1,),
                        [_t(experts + f"{proj}_weight_outer_scale_denominator")], "FP32")
            else:
                e.append(Entry(p + f"moe/{proj}", "mxfp8", "MXFP8", (g.experts * n, kk),
                               (_t(experts + f"{proj}_weight"), _t(experts + f"{proj}_weight_scale")), extra))
        if g.is_core(layer):
            rank = g.gate_rank
            _direct(e, p + "loop/down", (len(LOOP_STREAMS), rank, 2 * h),
                    [_t(src + f"controllers.{s}.down_weight") for s in LOOP_STREAMS])
            gain_forget = []
            for s in LOOP_STREAMS:
                gain_forget += [_t(src + f"controllers.{s}.gain_weight"), _t(src + f"controllers.{s}.forget_weight")]
            _direct(e, p + "loop/gain_forget", (len(LOOP_STREAMS), 2 * h, rank), gain_forget)

    if mtp and g.draft is not None:
        d = g.draft
        n_features = len(d.target_visits)
        _direct(e, "mtp/fuse_weight", (d.layers + 1, n_features), [MTP + "fuse_weight"])
        _direct(e, "mtp/mask_embed", (d.hidden,), [MTP + "mask_embed"])
        _direct(e, "mtp/final_norm", (d.hidden,), [MTP + "norm.weight"])
        q_rows, kv_rows = d.query_heads * d.head_dim, d.kv_heads * d.head_dim
        for layer in range(d.layers):
            p, src = f"mtp/layers/{layer}/", f"{MTP}layers.{layer}."
            _direct(e, p + "feature_norm", (d.hidden,), [src + "dflare_norm.weight"])
            _direct(e, p + "input_norm", (d.hidden,), [src + "input_layernorm.weight"])
            _direct(e, p + "post_attention_norm", (d.hidden,), [src + "post_attention_layernorm.weight"])
            _direct(e, p + "attn/q", (q_rows, d.hidden), [src + "self_attn.q_proj.weight"])
            _direct(e, p + "attn/k", (kv_rows, d.hidden), [src + "self_attn.k_proj.weight"])
            _direct(e, p + "attn/v", (kv_rows, d.hidden), [src + "self_attn.v_proj.weight"])
            _direct(e, p + "attn/o", (d.hidden, q_rows), [src + "self_attn.o_proj.weight"])
            _direct(e, p + "attn/k_ctx", (kv_rows, d.hidden), [src + "self_attn.k_proj_ctx.weight"])
            _direct(e, p + "attn/v_ctx", (kv_rows, d.hidden), [src + "self_attn.v_proj_ctx.weight"])
            _direct(e, p + "attn/q_norm", (d.head_dim,), [src + "self_attn.q_norm.weight"])
            _direct(e, p + "attn/k_norm", (d.head_dim,), [src + "self_attn.k_norm.weight"])
            _direct(e, p + "attn/k_ctx_norm", (d.head_dim,), [src + "self_attn.k_norm_ctx.weight"])
            _direct(e, p + "mlp/gate", (d.intermediate, d.hidden), [src + "mlp.gate_proj.weight"])
            _direct(e, p + "mlp/up", (d.intermediate, d.hidden), [src + "mlp.up_proj.weight"])
            _direct(e, p + "mlp/down", (d.hidden, d.intermediate), [src + "mlp.down_proj.weight"])
        r = d.markov_rank
        _direct(e, "mtp/markov/feature_norm", (d.hidden,), [MTP + "markov_head.dflare_norm.weight"])
        _direct(e, "mtp/markov/hidden_out", (d.hidden, r), [MTP + "markov_head.hidden_out.weight"])
        _direct(e, "mtp/markov/hidden_out_bias", (d.hidden,), [MTP + "markov_head.hidden_out.bias"])
        _direct(e, "mtp/markov/w1", (g.vocab_padded, r), [MTP + "markov_head.markov_w1.weight"])
        _direct(e, "mtp/markov/rnn_proj", (3 * r, 2 * r + 2 * d.hidden), [MTP + "markov_head.rnn_proj.weight"])
        _direct(e, "mtp/markov/rnn_proj_bias", (3 * r,), [MTP + "markov_head.rnn_proj.bias"])
    if vision is not None:
        vision_entries(e, vision, h)
    return e


def covered_sources(entries: list[Entry]) -> list[str]:
    """Every checkpoint tensor the stored (non-derived) objects read, with repeats."""
    return [s for entry in entries if entry.kind != "fp8" for s in entry.sources]


__all__ = ["Entry", "GATE_UP_ORDERS", "LAYOUT_OF", "LOOP_STREAMS", "MTP", "build_inventory",
           "covered_sources", "parse_attention", "vision_entries", "vision_geometry"]
