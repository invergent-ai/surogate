"""Exercise the numerical references with artifacts unlike their original fixtures."""

from importlib import import_module
import json

import pytest
import torch
import torch.nn.functional as F

from tests.serve.test_qwen3_5_checkpoint_config import config_for as dense_config
from tests.serve.test_qwen3_5_moe_checkpoint_config import config_for as moe_config
from surogate.serve.convert.common.recipe import source_requirements


@pytest.fixture(
    scope="module", params=[("qwen3_5", False), ("qwen3_5", True), ("qwen3_5_moe", False), ("qwen3_5_moe", True)]
)
def checkpoint(request, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("reference-checkpoint")
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    family, mtp = request.param
    config = (moe_config if family.endswith("_moe") else dense_config)(hidden=128, tied=True)
    config.update(
        mtp_num_hidden_layers=int(mtp),
        max_position_embeddings=16,
        layer_types=["linear_attention", "full_attention", "full_attention", "linear_attention"],
    )
    root = tmp_path / "renamed-checkpoint"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(config))
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, **{f"word{i}": i for i in range(1, 500)}}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    tokenizer.save_pretrained(root)
    (root / "generation_config.json").write_text(json.dumps({"eos_token_id": [7, 19]}))
    inv = import_module(f"surogate.serve.convert.{family}.inventory")
    recipe = import_module(f"surogate.serve.convert.{family}.recipe")
    g = inv.geometry_from_config(config, token_domain=500)
    tensors = {}
    generator = torch.Generator().manual_seed(71)
    for name, source in source_requirements(recipe.build_recipes(g)).items():
        tensors[name] = (torch.randn(source.shape, generator=generator) * 0.02).to(
            torch.float32 if source.dtype == "F32" else torch.bfloat16
        )
    save_file(tensors, root / "model.safetensors")
    output = tmp_path / "arbitrary-label.sinfer"
    convert = import_module(f"surogate.serve.convert.{family}.convert").convert
    if family.endswith("_moe"):
        convert(root, None, output, device="cpu")
    else:
        convert(root, output, device="cpu")
    return family, output, mtp


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_reference_prefill_decode_use_checkpoint_geometry(checkpoint, kv_dtype):
    family, output, mtp = checkpoint
    RefModel = import_module(f"surogate.serve.tools.reference.{family}.model").RefModel
    model = RefModel(
        output,
        device="cpu",
        prefill_chunk=2,
        compile_codec=False,
        mtp_draft_tokens=int(mtp),
        draft_head=mtp,
        kv_dtype=kv_dtype,
    )
    try:
        cfg = model.config
        assert (cfg.hidden, cfg.kv_heads, cfg.head_dim, cfg.gdn_k_dim, cfg.gdn_v_dim) == (128, 1, 128, 32, 64)
        assert cfg.full_index(1) == 0 and cfg.full_index(2) == 1
        assert cfg.gdn_index(3) == 1
        assert cfg.rms_eps == 2e-5 and cfg.rope_theta == 543210.0
        assert cfg.max_position_embeddings == 16
        assert cfg.attention_scale == pytest.approx(128**-0.5)
        assert cfg.gdn_scale == pytest.approx(32**-0.5)
        assert cfg.draft_vocab == 500
        if family == "qwen3_5":
            assert model.binding.text.output_head is model.binding.text.token_embedding
        assert (model.binding.mtp is not None) == mtp
        if family.endswith("_moe"):
            bank = model.binding.text.layers[0].moe.routed_gate_up
            assert (cfg.experts, cfg.experts_per_token, cfg.shared_intermediate) == (4, 2, 128)
            assert (bank.experts, bank.rows_per_expert) == (4, 128)
        token = model.prefill([1, 2, 3], capacity=8)
        assert 0 <= token < 500
        assert 0 <= model.decode(token) < 500
        assert model.state.position == 4
        assert model.state.ssm[0].shape == (1, 4, 32, 64)
        assert model.state.kv._k[0].shape == (8, 1, 128)
        assert torch.isfinite(model.last_hidden).all()
        generated = model.generate([1, 2], 4)
        assert len(generated) == 4 and all(0 <= token < 500 for token in generated)
        with pytest.raises(ValueError, match="maximum context"):
            model.prepare(17)
    finally:
        model.close()


def test_reference_frontend_accepts_checkpoint_tokenizer_and_text_only_resources(checkpoint):
    from surogate.serve.tools.reference.common.frontend import Frontend

    family, output, _ = checkpoint
    Binding = import_module(f"surogate.serve.tools.reference.{family}.bindings").ArtifactBinding
    with Binding.open(output) as binding:
        frontend = Frontend(binding)
        batch = frontend.process_text("word3 word4", thinking=False)
        assert batch.input_ids.tolist() == [3, 4]
        assert not batch.has_vision
        assert batch.position_ids.tolist() == [[0, 1]] * 3
        assert frontend.default_stop_token_ids == {7, 19}
        assert frontend.decode([3, 4]) == "word3 word4"


@pytest.mark.parametrize("family,config", [("qwen3_5", dense_config), ("qwen3_5_moe", moe_config)])
def test_reference_refuses_missing_checkpoint_fields(family, config):
    inv = import_module(f"surogate.serve.convert.{family}.inventory")
    resolve = import_module(f"surogate.serve.tools.reference.{family}.config").model_config_from_declared
    g = inv.geometry_from_config(config(), token_domain=500)
    from surogate.serve.convert.common.qwen3_5 import geometry_block

    geometry = geometry_block(g)
    for key in ("hidden", "gdn_value_head_dim", "rms_epsilon", "max_context", "mtp_layers", "draft_vocab"):
        missing = dict(geometry)
        del missing[key]
        with pytest.raises(ValueError, match=key):
            resolve(missing, layer_types=g.layer_types)
    with pytest.raises(ValueError, match="layer_types"):
        resolve(geometry, layer_types=["full_attention"])


def test_vision_ops_use_operand_dimensions_and_checkpoint_parameters():
    from surogate.serve.tools.reference.common import vision_ops as ops
    from surogate.serve.tools.reference.common.multimodal import build_mrope_positions

    generator = torch.Generator().manual_seed(83)
    q, k, v = [torch.randn(18, 3, 16, generator=generator).to(torch.bfloat16) for _ in range(3)]
    actual = ops.vision_attention(q, k, v, torch.tensor([0, 9, 18]))
    expected = torch.cat(
        [
            F.scaled_dot_product_attention(
                q[a:b].float().transpose(0, 1), k[a:b].float().transpose(0, 1), v[a:b].float().transpose(0, 1)
            ).transpose(0, 1)
            for a, b in ((0, 9), (9, 18))
        ]
    )
    torch.testing.assert_close(actual.float(), expected, atol=0.004, rtol=0.016)
    grid = torch.tensor([[1, 3, 6]])
    positions = ops.vision_position_ids(grid, merge=3)
    table = torch.arange(9 * 5).reshape(9, 5).to(torch.bfloat16)
    embedded = ops.interpolate_position_embedding(table, grid, merge=3)
    linear = F.interpolate(
        table.float().reshape(3, 3, 5).permute(2, 0, 1)[None], size=(3, 6), mode="bilinear", align_corners=True
    )
    torch.testing.assert_close(embedded, linear[0, :, positions[:, 0], positions[:, 1]].t().to(torch.bfloat16))
    rotated, _ = ops.apply_vision_rope(q, k, positions, theta=123.0)
    inv = 123.0 ** (-torch.arange(0, 8, 2).float() / 8)
    phase = (positions[..., None] * inv).flatten(1).repeat(1, 2)[:, None]
    half = torch.cat((-q[..., 8:], q[..., :8]), dim=-1).float()
    torch.testing.assert_close(rotated, (q.float() * phase.cos() + half * phase.sin()).to(torch.bfloat16))
    types = torch.tensor([0, 1, 1, 0])
    mrope, delta = build_mrope_positions(types, grid, None, spatial_merge=3)
    assert mrope.tolist() == [[0, 1, 1, 3], [0, 1, 1, 3], [0, 1, 2, 3]]
    assert delta == 0
    weight, bias = torch.ones(16), torch.zeros(16)
    torch.testing.assert_close(
        ops.layer_norm(q, weight, bias, eps=0.1), F.layer_norm(q, (16,), weight.bfloat16(), bias.bfloat16(), 0.1)
    )


@pytest.mark.parametrize(
    "family,schedule",
    [
        ("qwen3_5", ["full_attention"] * 4),
        ("qwen3_5", ["linear_attention"] * 4),
        ("qwen3_5", ["full_attention", "linear_attention", "full_attention", "linear_attention"]),
        ("qwen3_5_moe", ["full_attention", "linear_attention", "full_attention", "linear_attention"]),
    ],
)
def test_bf16_reference_vision_and_mixed_projection_layouts(tmp_path, family, schedule):
    from dataclasses import replace
    from surogate.serve.artifact.container import ArtifactIdentity, ArtifactWriter, ResourceSpec, TensorSpec
    from surogate.serve.convert.common.qwen3_5 import geometry_block, vision_geometry_block

    config = (moe_config if family.endswith("_moe") else dense_config)(vision=True)
    config["layer_types"] = schedule
    config["vision_config"]["spatial_merge_size"] = 3
    inv = import_module(f"surogate.serve.convert.{family}.inventory")
    g = inv.geometry_from_config(config, token_domain=500)
    specs = {
        s.name: replace(s, format="BF16", layout="contiguous-le-v1") if len(s.shape) == 2 else s
        for s in inv.build_tensor_specs(g)
    }
    if family == "qwen3_5":
        # Different layers may store different fused parents.
        full = "text/layers/0/attention/query_key_gate_value"
        if full in specs:
            spec = specs.pop(full)
            for name in ("query_key", "gate_value"):
                key = "text/layers/0/attention/" + name
                specs[key] = replace(spec, name=key, shape=(spec.shape[0] // 2, spec.shape[1]))
        fused = "text/layers/1/gdn/query_key_value_z"
        if fused in specs:
            spec = specs.pop(fused)
            for name, rows in (
                ("query_key", 2 * g.gdn_key_heads * g.gdn_key_head_dim),
                ("value_z", 2 * g.gdn_value_heads * g.gdn_value_head_dim),
            ):
                key = "text/layers/1/gdn/" + name
                specs[key] = replace(spec, name=key, shape=(rows, g.hidden))
    resources = [
        ResourceSpec(f"frontend/{name}.json", "raw-bytes-v1", 2)
        for name in ("tokenizer", "tokenizer_config", "generation_config")
    ]
    output = tmp_path / "any-model-name.sinfer"
    vision = vision_geometry_block(config, text_hidden=g.hidden)
    vision.update(rope_theta=123.0, norm_epsilon=0.03)
    generator = torch.Generator().manual_seed(91)
    with ArtifactWriter(
        output,
        ArtifactIdentity("renamed", "bf16", family),
        [*resources, *(TensorSpec(s.name, s.shape, s.format, s.layout) for s in specs.values())],
        geometry=geometry_block(g),
        vision_geometry=vision,
        layer_types=schedule,
    ) as writer:
        for resource in resources:
            writer.write(resource.name, b"{}")
        for spec in specs.values():
            if spec.format == "I32":
                tensor = torch.arange(g.draft_vocab, dtype=torch.int32)
            else:
                tensor = (torch.randn(spec.shape, generator=generator) * 0.02).to(
                    torch.float32 if spec.format == "FP32" else torch.bfloat16
                )
            writer.write(spec.name, tensor.view(torch.uint8).numpy().tobytes())
    model = import_module(f"surogate.serve.tools.reference.{family}.model").RefModel(
        output, device="cpu", compile_codec=False, mtp_draft_tokens=1, draft_head=True
    )
    encoder = import_module(f"surogate.serve.tools.reference.{family}.vision").VisionEncoder(
        model.binding, "cpu", compile_codec=False
    )
    try:
        assert model.config.layer_types == tuple(schedule)
        assert model.binding.vision_config.heads == 3
        assert model.binding.vision_config.patch_dim == 48
        assert model.binding.vision_config.rope_theta == 123.0
        assert model.binding.vision_config.norm_eps == 0.03
        pixels = torch.randn(18, 48, generator=generator).bfloat16()
        result = encoder.encode(pixels, torch.tensor([[1, 3, 6]]), None, None)
        assert result.image_embeddings.shape == (2, 128)
        assert torch.isfinite(result.image_embeddings).all()
        token = model.prefill([1, 3], capacity=6)
        assert 0 <= model.decode(token) < 500
    finally:
        encoder.close()
        model.close()
