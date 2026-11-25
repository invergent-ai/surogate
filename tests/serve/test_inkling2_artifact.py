"""Inkling-2 artifact (ARTIFACT.md I-1): converter + reader round trip on a tiny synthetic
``inkling2_loop`` checkpoint with the real tensor names, NVFP4 and MXFP8 expert layers, a loop core,
a global layer inside the core and a DSpark drafter. CPU only, seconds.

The real 20 GB conversion is checked by the same verifier (``python -m
surogate.serve.convert.inkling2.verify``); this test proves the verifier and the formats.
"""

from __future__ import annotations

import json
import struct

import pytest
import torch
from safetensors.torch import save_file

from surogate.serve.artifact.container import Artifact
from surogate.serve.artifact.layouts import (
    decode_fp8_grid_words,
    decode_mxfp8_words,
    dequantize_fp8_grid,
    dequantize_mxfp8,
    encode_fp8_grid,
    encode_mxfp8,
    encoded_size,
    fp8_grid_geometry,
    mxfp8_geometry,
)
from surogate.serve.artifact.numeric import decode_e8m0_word, valid_mxfp8_scale_word
from surogate.serve.convert.inkling2.convert import convert
from surogate.serve.convert.inkling2.fp8_attention import SCHEMES, quantize
from surogate.serve.convert.inkling2.verify import Mismatch, Verifier
from surogate.serve.convert.inkling2.checkpoint import Checkpoint

H, LAYERS, VOCAB, VOCAB_REAL = 128, 4, 256, 250
QH, KVH, HD, DREL, REL, WIN, K = 2, 1, 64, 16, 32, 32, 4
EXPERTS, INTER, RANK = 2, 64, 8
LOCAL = [0, 1, 3]
CORE = (1, 3)
PASSES = 2
NVFP4_LAYERS = [0, 1, 2]


def _config() -> dict:
    return {
        "model_type": "inkling2_loop", "eos_token_id": 249,
        "architectures": ["Inkling2LoopForConditionalGeneration"],
        "text_config": {
            "model_max_length": 4096, "hidden_size": H, "num_hidden_layers": LAYERS, "vocab_size": VOCAB,
            "num_attention_heads": QH, "num_key_value_heads": KVH, "head_dim": HD, "d_rel": DREL,
            "rel_extent": REL, "log_scaling_n_floor": 64, "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-6,
            "use_value_norm_no_gamma": True, "qk_l2_clip_threshold": 22.627416997969522,
            "rel_l2_clip_threshold": 8.0, "local_layer_ids": LOCAL, "sconv_kernel_size": K,
            "unpadded_vocab_size": VOCAB_REAL, "final_logit_softcapping": 30.0, "swa_head_dim": HD,
            "swa_num_attention_heads": QH, "swa_num_key_value_heads": KVH, "sliding_window_size": WIN,
            "n_layers_per_attn_res_block": 2, "attn_res_normalize_values": True, "n_routed_experts": EXPERTS,
            "num_experts_per_tok": 1, "n_shared_experts": 0, "intermediate_size": INTER,
            "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 8.0, "use_gate_bias": True,
            "loop_config": {"core_start": CORE[0], "core_end": CORE[1], "num_passes": PASSES,
                            "gate_rank": RANK, "is_varloop": False},
        },
        "mtp_config": {
            "hidden_size": H, "num_hidden_layers": 2, "num_attention_heads": QH, "num_key_value_heads": KVH,
            "head_dim": HD, "intermediate_size": 256, "max_position_embeddings": 4096, "rms_norm_eps": 1e-6,
            "rope_theta": 1e6, "local_layer_ids": [0], "sliding_window_size": WIN, "block_size": 4,
            "markov_rank": RANK, "markov_head_type": "rnn_hidden", "target_layer_ids": [1, 4, 6],
        },
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
    }


def _bf16(*shape, gen):
    return (torch.randn(*shape, generator=gen) * 0.05).to(torch.bfloat16)


VH, VHEADS, VLAYERS, VGRID, VPROJ = 64, 4, 2, 4, 96


def _vision_config() -> dict:
    return {"model_type": "inkling2_vision", "text_hidden_size": H, "temporal_patch_size": 2, "native_patch_size": 20,
            "adapter_merge_size": 2, "projection_hidden_dim": VPROJ, "projector_bias": False, "hidden_size": VH,
            "num_hidden_layers": VLAYERS, "num_attention_heads": VHEADS, "mlp_ratio": 2.0,
            "posemb_grid_size": VGRID, "layer_norm_eps": 1e-05}


def _vision_tensors(gen) -> dict[str, torch.Tensor]:
    v, b = "model.visual.", "model.visual.backbone."
    t = {b + "embeddings.class_embedding": _bf16(VH, gen=gen),
         b + "embeddings.cls_position_embedding": _bf16(VH, gen=gen),
         b + "embeddings.patch_embedding.weight": _bf16(VH, 1200, gen=gen),
         b + "embeddings.position_embedding": _bf16(VGRID * VGRID, VH, gen=gen),
         b + "ln_pre.weight": _bf16(VH, gen=gen), b + "ln_pre.bias": _bf16(VH, gen=gen),
         v + "proj_in.weight": _bf16(VPROJ, 4 * VH, gen=gen), v + "proj_out.weight": _bf16(H, VPROJ, gen=gen)}
    for i in range(VLAYERS):
        p = b + f"layers.{i}."
        for norm in ("ln_1", "ln_2"):
            t[p + f"{norm}.weight"], t[p + f"{norm}.bias"] = _bf16(VH, gen=gen), _bf16(VH, gen=gen)
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            t[p + f"self_attn.{proj}.weight"], t[p + f"self_attn.{proj}.bias"] = _bf16(VH, VH, gen=gen), _bf16(VH, gen=gen)
        t[p + "mlp.c_fc.weight"], t[p + "mlp.c_fc.bias"] = _bf16(2 * VH, VH, gen=gen), _bf16(2 * VH, gen=gen)
        t[p + "mlp.c_proj.weight"], t[p + "mlp.c_proj.bias"] = _bf16(VH, 2 * VH, gen=gen), _bf16(VH, gen=gen)
        t[p + "ls_1.gamma"], t[p + "ls_2.gamma"] = _bf16(VH, gen=gen), _bf16(VH, gen=gen)
    return t


def _checkpoint(tmp_path, seed=0, vision=False):
    gen = torch.Generator().manual_seed(seed)
    t: dict[str, torch.Tensor] = {}
    p = "model.llm."
    t[p + "embed.weight"] = _bf16(VOCAB, H, gen=gen)
    t[p + "embed_norm.weight"] = _bf16(H, gen=gen)
    t[p + "norm.weight"] = _bf16(H, gen=gen)
    t[p + "unembed.weight"] = _bf16(VOCAB, H, gen=gen)
    for layer in range(1, LAYERS):
        rows = 1 if layer in LOCAL else 3
        t[p + f"attn_res.attn_queries.{layer - 1}"] = torch.randn(rows, H, generator=gen).squeeze(0)
    for layer in range(LAYERS):
        t[p + f"attn_res.mlp_queries.{layer}"] = torch.randn(H, generator=gen)
    t[p + "attn_res.output_query"] = torch.randn(H, generator=gen)
    core = CORE[1] - CORE[0]
    t[p + "middle_cycle_cross_pass_queries"] = torch.randn(2 * core, H, generator=gen)
    t[p + "middle_cycle_cross_pass_kv_queries"] = torch.randn(2 * core, H, generator=gen)
    quant_groups = {"nvfp4": [], "mxfp8": []}
    for layer in range(LAYERS):
        q = p + f"layers.{layer}."
        local = layer in LOCAL
        if local:
            t[q + "attn_norm.weight"] = _bf16(H, gen=gen)
            t[q + "attn.k_sconv.weight"] = _bf16(KVH * HD, 1, K, gen=gen)
            t[q + "attn.v_sconv.weight"] = _bf16(KVH * HD, 1, K, gen=gen)
        else:
            for s in "qkv":
                t[q + f"{s}_input_norm.weight"] = _bf16(H, gen=gen)
            for s in "kv":
                t[q + f"attn.{s}_sconv.static_w"] = _bf16(K, KVH * HD, gen=gen)
                t[q + f"attn.{s}_sconv.weight_proj"] = _bf16(KVH * K, H, gen=gen)
        for name, rows in (("wq_du", QH * HD), ("wr_du", QH * DREL), ("gate_attn", QH * HD),
                           ("attn_sink_proj", QH), ("wk_dv", KVH * HD), ("wv_dv", KVH * HD)):
            t[q + f"attn.{name}.weight"] = _bf16(rows, H, gen=gen)
        t[q + "attn.attn_sink_proj.weight"][0].zero_()  # an all-zero row: the scale-1.0 rule
        t[q + "attn.wo_ud.weight"] = _bf16(H, QH * HD, gen=gen)
        t[q + "attn.q_l2_clip.scale"] = _bf16(QH, HD, gen=gen)
        t[q + "attn.rel_l2_clip.scale"] = _bf16(QH, DREL, gen=gen)
        t[q + "attn.rel_logits_proj.proj"] = _bf16(DREL, REL, gen=gen)
        t[q + "attn_sconv.weight"] = _bf16(H, 1, K, gen=gen)
        t[q + "mlp_sconv.weight"] = _bf16(H, 1, K, gen=gen)
        t[q + "mlp_norm.weight"] = _bf16(H, gen=gen)
        t[q + "mlp.gate.weight"] = _bf16(EXPERTS, H, gen=gen)
        t[q + "mlp.gate.bias"] = torch.randn(EXPERTS, generator=gen)
        e = q + "mlp.experts."
        for proj, (n, k) in (("w13", (2 * INTER, H)), ("w2", (H, INTER))):
            if layer in NVFP4_LAYERS:
                t[e + f"{proj}_weight"] = torch.randint(0, 256, (EXPERTS, n, k // 2), generator=gen, dtype=torch.uint8)
                t[e + f"{proj}_weight_scale"] = torch.randint(0x20, 0x7F, (EXPERTS, n, k // 16), generator=gen,
                                                              dtype=torch.uint8).view(torch.float8_e4m3fn)
                t[e + f"{proj}_weight_amax"] = torch.rand(EXPERTS, generator=gen) + 0.1
                t[e + f"{proj}_weight_outer_scale_denominator"] = torch.tensor([2688.0])
                quant_groups["nvfp4"].append(e + f"{proj}_weight")
            else:
                codes = torch.randint(0, 256, (EXPERTS, n, k), generator=gen, dtype=torch.uint8)
                codes[(codes & 0x7F) == 0x7F] = 0x00
                t[e + f"{proj}_weight"] = codes.view(torch.float8_e4m3fn)
                t[e + f"{proj}_weight_scale"] = torch.randint(110, 126, (EXPERTS, n, k // 32), generator=gen,
                                                              dtype=torch.uint8)
                quant_groups["mxfp8"].append(e + f"{proj}_weight")
        if CORE[0] <= layer < CORE[1]:
            for s in ("attn", "k", "v", "mlp"):
                t[q + f"controllers.{s}.down_weight"] = _bf16(RANK, 2 * H, gen=gen)
                t[q + f"controllers.{s}.gain_weight"] = _bf16(H, RANK, gen=gen)
                t[q + f"controllers.{s}.forget_weight"] = _bf16(H, RANK, gen=gen)
    t["model.visual.proj_out.weight"] = _bf16(H, 64, gen=gen)  # never carried (text only)
    m: dict[str, torch.Tensor] = {"fuse_weight": _bf16(3, 3, gen=gen), "mask_embed": _bf16(H, gen=gen),
                                  "norm.weight": _bf16(H, gen=gen)}
    for d in range(2):
        s = f"layers.{d}."
        for name in ("dflare_norm", "input_layernorm", "post_attention_layernorm"):
            m[s + name + ".weight"] = _bf16(H, gen=gen)
        for name, shape in (("q_proj", (QH * HD, H)), ("k_proj", (KVH * HD, H)), ("v_proj", (KVH * HD, H)),
                            ("o_proj", (H, QH * HD)), ("k_proj_ctx", (KVH * HD, H)),
                            ("v_proj_ctx", (KVH * HD, H)), ("q_norm", (HD,)), ("k_norm", (HD,)),
                            ("k_norm_ctx", (HD,))):
            m[s + f"self_attn.{name}.weight"] = _bf16(*shape, gen=gen)
        for name, shape in (("gate_proj", (256, H)), ("up_proj", (256, H)), ("down_proj", (H, 256))):
            m[s + f"mlp.{name}.weight"] = _bf16(*shape, gen=gen)
    m["markov_head.dflare_norm.weight"] = _bf16(H, gen=gen)
    m["markov_head.hidden_out.weight"] = _bf16(H, RANK, gen=gen)
    m["markov_head.hidden_out.bias"] = _bf16(H, gen=gen)
    m["markov_head.markov_w1.weight"] = _bf16(VOCAB, RANK, gen=gen)
    m["markov_head.rnn_proj.weight"] = _bf16(3 * RANK, 2 * RANK + 2 * H, gen=gen)
    m["markov_head.rnn_proj.bias"] = _bf16(3 * RANK, gen=gen)

    config = _config()
    if vision:
        t.update(_vision_tensors(gen))
        config["vision_config"] = _vision_config()

    model = tmp_path / "model"
    model.mkdir()
    save_file({k: v.contiguous() for k, v in t.items()}, str(model / "model-00001-of-00001.safetensors"),
              metadata={"format": "pt"})
    save_file({k: v.contiguous() for k, v in m.items()}, str(model / "mtp.safetensors"), metadata={"format": "pt"})
    (model / "config.json").write_text(json.dumps(config))
    groups = [{"compute_dtype": "nvfp4", "tensors": quant_groups["nvfp4"]},
              {"compute_dtype": "mxfp8", "tensors": quant_groups["mxfp8"]}]
    (model / "hf_quant_config.json").write_text(json.dumps({"model": {"parameter_groups": groups},
                                                            "mtp": {"parameter_groups": []},
                                                            "quant_method": "inkling2"}))
    (model / "tokenizer.json").write_text('{"model": {}}')
    (model / "tokenizer_config.json").write_text('{"eos_token": "<eos>"}')
    (model / "special_tokens_map.json").write_text('{"eos_token": "<eos>"}')
    (model / "generation_config.json").write_text('{"eos_token_id": 249}')
    (model / "chat_template.jinja").write_text("{{ messages }}")
    return model, t, m


@pytest.mark.parametrize("gate_up_order", ["interleaved", "split"])
def test_convert_and_verify_every_object(tmp_path, gate_up_order):
    model, tensors, _ = _checkpoint(tmp_path)
    out = tmp_path / "tiny.sinfer"
    schemes = SCHEMES
    convert(model, out, fp8_schemes=schemes, gate_up_order=gate_up_order, log=lambda m: None)
    ckpt = Checkpoint(model)
    with Artifact(out) as artifact:
        stats = Verifier(artifact, ckpt, chunk_rows=64, log=lambda m: None).run(
            fp8_schemes=schemes, gate_up_order=gate_up_order, mtp=True)
        assert stats["objects_nvfp4"] == 2 * len(NVFP4_LAYERS)
        assert stats["objects_mxfp8"] == 2 * (LAYERS - len(NVFP4_LAYERS))
        assert stats["objects_fp8"] == 2 * LAYERS * len(schemes)
        assert artifact.identity.architecture == "inkling2"
        assert artifact.layer_types == ["sliding_attention", "sliding_attention", "full_attention",
                                        "sliding_attention"]
        assert not any(o.name.startswith("vision") for o in artifact.objects)
        config = json.loads(bytes(artifact.payload("inkling2/config.json")))
        assert config["visits"]["table"] == [[0, 0], [1, 0], [2, 0], [1, 1], [2, 1], [3, 0]]
        assert config["layers"][2]["visits"] == [2, 4]
        assert config["dspark"]["target_visits"] == [1, 4, 6]
        assert config["moe"]["gate_up_rows"] == gate_up_order
        # The fused qkvr's FP8 slices keep their own scales: the zero sink row has scale 1.0
        # under the per-channel scheme, and is all-zero codes.
        obj = artifact.find("text/layers/0/attn/fp8-channel/qkvr")
        n, k = obj.shape
        payload = bytes(artifact.payload(obj))
        sink_row = QH * HD + QH * DREL + QH * HD
        assert payload[sink_row * k: (sink_row + 1) * k] == bytes(k)
        offset = (n * k + 255) // 256 * 256
        assert struct.unpack_from("<f", payload, offset + 4 * sink_row)[0] == 1.0


def test_vision_tower_is_carried_and_verified(tmp_path):
    model, tensors, _ = _checkpoint(tmp_path, vision=True)
    out = tmp_path / "tiny-vl.sinfer"
    convert(model, out, log=lambda m: None)
    with Artifact(out) as artifact:
        stats = Verifier(artifact, Checkpoint(model), chunk_rows=64, log=lambda m: None).run(
            fp8_schemes=(), gate_up_order="interleaved", mtp=True)
        names = [o.name for o in artifact.objects if o.name.startswith("vision/")]
        # 6 globals + 18 per layer + the two projector matrices: every model.visual.* tensor, once.
        assert len(names) == 6 + 18 * VLAYERS + 2
        assert artifact.vision_geometry == {
            "inkling": 1, "layers": VLAYERS, "hidden": VH, "intermediate": 2 * VH, "heads": VHEADS,
            "patch_dim": 1200, "merge": 2, "position_embeddings": VGRID * VGRID, "rotary_dim": VH // VHEADS,
            "rope_theta": 10000.0, "projector_hidden": VPROJ, "output_hidden": H, "max_image_tokens": 4096,
            "norm_epsilon": 1e-05}
        q = artifact.find("vision/layers/1/attention/query")
        assert bytes(artifact.payload(q)) == tensors["model.visual.backbone.layers.1.self_attn.q_proj.weight"] \
            .view(torch.int16).numpy().tobytes()
        assert stats["objects_direct"] >= len(names)
    text_only = tmp_path / "tiny-text.sinfer"
    convert(model, text_only, vision=False, log=lambda m: None)
    with Artifact(text_only) as artifact:
        assert not any(o.name.startswith("vision/") for o in artifact.objects)
        assert artifact.vision_geometry == {}
        Verifier(artifact, Checkpoint(model), chunk_rows=64, log=lambda m: None).run(
            fp8_schemes=(), gate_up_order="interleaved", mtp=True, vision=False)


def test_verifier_catches_a_flipped_word(tmp_path):
    model, _, _ = _checkpoint(tmp_path, seed=1)
    out = tmp_path / "tiny.sinfer"
    convert(model, out, log=lambda m: None)
    with Artifact(out) as artifact:
        obj = artifact.find("text/layers/3/moe/w2")
        position = artifact.payload_offset + obj.offset + 5
    raw = bytearray(out.read_bytes())
    raw[position] ^= 0x01
    out.write_bytes(bytes(raw))
    with Artifact(out) as artifact:
        with pytest.raises(Mismatch):
            Verifier(artifact, Checkpoint(model), chunk_rows=64, log=lambda m: None).run(
                fp8_schemes=(), gate_up_order="interleaved", mtp=True)


def test_mxfp8_layout_round_trip_and_values():
    shape = (3, 64)
    geometry = mxfp8_geometry("MXFP8", shape)
    assert (geometry.code_plane_bytes, geometry.scale_plane_offset, geometry.scale_plane_bytes,
            geometry.payload_bytes) == (192, 256, 6, 262)
    assert encoded_size("mxfp8-k32-v1", "MXFP8", shape) == 262
    codes = torch.tensor([[0x38] * 32 + [0xB8] * 32, [0x40] * 64, [0x00] * 64], dtype=torch.uint8)
    scales = torch.tensor([[127, 128], [126, 0], [1, 254]], dtype=torch.uint8)
    payload = encode_mxfp8(codes, scales, shape)
    c, s = decode_mxfp8_words(payload, shape)
    assert torch.equal(c, codes) and torch.equal(s, scales)
    values = dequantize_mxfp8(payload, shape)
    assert values[0, 0] == 1.0 and values[0, 40] == -2.0
    assert values[1, 0] == 1.0 and values[1, 40] == 2.0 * 2.0**-127
    assert decode_e8m0_word(127) == 1.0 and decode_e8m0_word(0) == 2.0**-127
    assert valid_mxfp8_scale_word(254) and not valid_mxfp8_scale_word(255)
    with pytest.raises(ValueError):
        encode_mxfp8(codes, torch.full((3, 2), 255, dtype=torch.uint8), shape)
    bad = codes.clone()
    bad[0, 0] = 0x7F
    with pytest.raises(ValueError):
        encode_mxfp8(bad, scales, shape)
    with pytest.raises(ValueError):
        mxfp8_geometry("MXFP8", (2, 48))


def test_fp8_grid_layout_and_quantiser():
    shape = (24, 256)
    geometry = fp8_grid_geometry("FP8_E4M3FN_1X128_F32S", shape)
    assert (geometry.grid_rows, geometry.grid_columns, geometry.scale_plane_offset,
            geometry.payload_bytes) == (24, 2, 6144, 6144 + 24 * 2 * 4)
    w = (torch.randn(shape, generator=torch.Generator().manual_seed(3)) * 0.02).to(torch.bfloat16)
    for scheme in SCHEMES:
        q = quantize(w, scheme)
        payload = encode_fp8_grid(q.codes, q.grid, shape)
        codes, grid = decode_fp8_grid_words(payload, shape)
        assert torch.equal(codes, q.codes) and torch.equal(grid, q.grid)
        values = dequantize_fp8_grid(payload, shape)
        assert torch.equal(values, q.dequantize())
        relative = float((values - w.float()).norm() / w.float().norm())
        assert relative < 0.08, (scheme, relative)
        # the largest magnitude of each cell maps to +-448 exactly
        assert int((q.codes & 0x7F).max()) == 0x7E
    # 24 rows = one partial 128-row block: every row repeats its column block's scale
    grid = quantize(w, "b128").grid
    assert torch.equal(grid, grid[:1].expand_as(grid))
    assert torch.equal(grid[0], quantize(w, "g1x128").grid.amax(dim=0))
