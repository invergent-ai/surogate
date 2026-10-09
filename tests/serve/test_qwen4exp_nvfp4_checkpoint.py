"""The NVFP4 safetensors release of a hyper-connected hybrid converts with its words intact."""

import json

import torch
from safetensors.torch import save_file

from surogate.serve.artifact.container import Artifact
from surogate.serve.artifact.layouts import decode_nvfp4_words, dequantize_row_split, decode_direct
from surogate.serve.convert.common import iq4nl
from surogate.serve.convert.common.recipe import source_requirements
from surogate.serve.convert.qwen4exp import checkpoint, inventory as inv
from tests.serve.test_qwen4exp_checkpoint_config import config_for

PARTS, PART_ROWS = 2, 100


def _release(path):
    """A tiny ModelOpt export: NVFP4 experts, one FP8 n-gram table, BF16 everything else."""
    config = config_for(vision=True)
    text = config["text_config"]
    text["split_ngram_parts"] = PARTS
    g = inv.geometry_from_config(checkpoint.text_config(config), ple_table_rows=PARTS * PART_ROWS,
                                 token_domain=500)
    generator = torch.Generator().manual_seed(5)
    tensors = {}
    for name, item in source_requirements(checkpoint.build_recipes(g)).items():
        tensors[name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name] = (
            torch.randn(item.shape, generator=generator) * 0.1).to(torch.bfloat16)
    quantized = {}
    for layer in range(g.layers):
        quantized[f"model.language_model.layers.{layer}.mlp.experts"] = {"quant_algo": "NVFP4", "group_size": 16}
        for expert in range(g.experts):
            input_scale = torch.tensor(0.01 * (expert + 1))
            weight_scale = torch.tensor(0.002 * (layer + 1))
            for projection, (n, k) in (("gate_proj", (g.intermediate, g.hidden)),
                                       ("up_proj", (g.intermediate, g.hidden)),
                                       ("down_proj", (g.hidden, g.intermediate))):
                base = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}."
                tensors[base + "weight"] = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=generator)
                tensors[base + "weight_scale"] = (torch.rand((n, k // 16), generator=generator) * 8 + 1).to(
                    torch.float8_e4m3fn)
                tensors[base + "weight_scale_2"] = weight_scale.clone()
                tensors[base + "input_scale"] = input_scale.clone() * (2 if projection == "down_proj" else 1)
    ple = f"model.language_model.layers.{g.ple_layer}.ple.ple_embedding."
    quantized[ple.removesuffix(".") + ".ngram_embedding"] = {"quant_algo": "FP8"}
    for part in range(PARTS):
        tensors[ple + f"ngram_embedding.shard_{part}.weight"] = (
            torch.randn((PART_ROWS, g.ple_head_dim), generator=generator) * 100).clamp(-440, 440).to(
                torch.float8_e4m3fn)
    tensors[ple + "ngram_embedding.weight_scale"] = torch.tensor([0.002], dtype=torch.bfloat16)
    tensors[ple + "layer_multipliers"] = torch.tensor([(1 << 40) + 3, 7], dtype=torch.int64)
    tensors[ple + "ngram_heads_offsets"] = torch.tensor([0, 90], dtype=torch.int64)
    tensors[ple + "ngram_heads_vocab_sizes"] = torch.tensor([90, 110], dtype=torch.int64)
    tensors["model.visual.blocks.0.norm1.weight"] = torch.ones(8, dtype=torch.bfloat16)
    path.mkdir()
    save_file(tensors, path / "model.safetensors")
    config["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                                     "quantized_layers": quantized}
    (path / "config.json").write_text(json.dumps(config))
    (path / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"a": 0, "b": 1}},
                                                     "added_tokens": [{"id": 499}]}))
    (path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    (path / "generation_config.json").write_text(json.dumps({"eos_token_id": 1}))
    (path / "preprocessor_config.json").write_text(json.dumps({"patch_size": 16}))
    return g, tensors


def test_release_converts(tmp_path):
    g, stored = _release(tmp_path / "release")
    output = checkpoint.convert(tmp_path / "release", tmp_path / "out.sinfer", device="cpu")
    layer = "model.language_model.layers.0."
    with Artifact(output) as artifact:
        assert artifact.identity.weights_id == checkpoint.WEIGHTS_ID
        names = {obj.name for obj in artifact.objects}
        assert "frontend/preprocessor_config.json" not in names
        assert not any(name.startswith(("vision/", "mtp/")) for name in names)

        def direct(name):
            obj = artifact.find(name)
            return decode_direct(bytes(artifact.payload(obj)), obj.format, tuple(obj.shape))

        # Zero-centred HF norms are stored folded; the attention q/k norms are not.
        hc = stored[layer + "attn_hyper_connection.hc_norm.weight"].float()
        assert torch.equal(direct("text/layers/0/hc_attn/norm"), hc + 1)
        attn = f"model.language_model.layers.{g.full_attention_layers[0]}.self_attn."
        prefix = f"text/layers/{g.full_attention_layers[0]}/attention/"
        assert torch.equal(direct(prefix + "query_norm").float(), stored[attn + "q_norm.weight"].float())
        assert torch.equal(direct(prefix + "indexer/key_norm").float(),
                           (stored[attn + "indexer.k_layernorm.weight"].float() + 1).bfloat16().float())
        qk = stored[attn + "indexer.index_qk_proj.weight"]
        assert torch.equal(direct(prefix + "indexer/key"), qk[g.indexer_heads * g.indexer_head_dim:])

        # Dense projections are W8 within a W8 step of the release.
        obj = artifact.find("text/layers/0/gdn/query_key_value_z")
        values = dequantize_row_split(bytes(artifact.payload(obj)), obj.format, tuple(obj.shape),
                                      dtype=torch.float32)
        reference = torch.cat([stored[layer + "linear_attn.in_proj_qkv.weight"],
                               stored[layer + "linear_attn.in_proj_z.weight"]]).float()
        assert torch.allclose(values, reference, atol=float(reference.abs().max()) / 100)

        # Routed experts: the release's words, [up; gate] per expert, ModelOpt's multipliers.
        obj = artifact.find("text/layers/0/mlp/routed_gate_up")
        assert obj.format == checkpoint.NVFP4
        codes, scales, divisor = decode_nvfp4_words(bytes(artifact.payload(obj)), tuple(obj.shape))
        expert = 1
        up = stored[layer + f"mlp.experts.{expert}.up_proj.weight"]
        gate = stored[layer + f"mlp.experts.{expert}.gate_proj.weight"]
        rows = slice(expert * 2 * g.intermediate, (expert + 1) * 2 * g.intermediate)
        assert torch.equal(codes[rows], torch.cat([up, gate]))
        assert float(divisor) == 1.0
        second = direct("text/layers/0/mlp/routed_gate_up_scale")
        act = direct("text/layers/0/mlp/routed_gate_up_act_scale")
        alpha = direct("text/layers/0/mlp/routed_gate_up_alpha")
        assert torch.allclose(second[2 * expert:2 * expert + 2], torch.tensor([0.002, 0.002]))
        assert torch.allclose(act[expert], torch.tensor(1 / 0.02))
        assert torch.allclose(alpha[expert], torch.tensor(0.02 * 0.002))
        down_act = direct("text/layers/0/mlp/routed_down_act_scale")
        assert torch.allclose(down_act[expert], torch.tensor(1 / 0.04))

        # The n-gram table: shard 0's rows first, as IQ4_NL of the FP8 values times their scale.
        obj = artifact.find(inv.PLE_TABLE_RESOURCE)
        assert tuple(obj.shape) == (PARTS * PART_ROWS, g.ple_head_dim)
        table = torch.frombuffer(bytearray(artifact.payload(obj)), dtype=torch.uint8).reshape(PARTS * PART_ROWS, -1)
        back = iq4nl.dequantize_rows(table, g.ple_head_dim)
        ple = f"model.language_model.layers.{g.ple_layer}.ple.ple_embedding.ngram_embedding."
        fp8 = torch.cat([stored[ple + f"shard_{i}.weight"].float() for i in range(PARTS)]) * 0.002
        cos = torch.nn.functional.cosine_similarity(back, fp8, dim=1)
        assert float(cos.min()) > 0.98
        assert direct("text/ple/multipliers").tolist() == [3, 1 << 8, 7, 0]
        assert direct("text/ple/head_vocab_sizes").tolist() == [90, 110]
