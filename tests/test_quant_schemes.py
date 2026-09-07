# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# Per-module scheme resolution for compressed-tensors checkpoints
# (surogate/core/model/quant_schemes.py). The unit cases build a tiny Llama
# on the meta device and describe a checkpoint as a dict of tensor shapes, so
# nothing here reads a shard or touches a GPU; what is under test is that the
# config's own targets/ignore rules -- run through the library's matcher --
# decide the format of every module, including leaves no instantiated model
# lists, and that a checkpoint contradicting its config is refused rather
# than guessed at. The last case runs the same code on a real 24 GB export
# when it is on disk, reading headers only.

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("compressed_tensors")
transformers = pytest.importorskip("transformers")

from compressed_tensors.quantization import QuantizationConfig  # noqa: E402
from torch import nn  # noqa: E402

from surogate.core.model import quant_schemes as qs  # noqa: E402

REAL_EXPORT = Path("/home/densemax2/work/models/hf/Qwen3.6-35B-A3B-NVFP4-redhat-vllm")


def tiny_llama() -> nn.Module:
    config = transformers.LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
    )
    with torch.device("meta"):
        return transformers.LlamaForCausalLM(config)


def nvfp4_config(ignore: list[str]) -> QuantizationConfig:
    return QuantizationConfig.model_validate(
        {
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "ignore": ignore,
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "format": "nvfp4-pack-quantized",
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                        "strategy": "tensor_group",
                        "group_size": 16,
                        "symmetric": True,
                        "dynamic": False,
                    },
                }
            },
        }
    )


def checkpoint_for(skeleton: nn.Module, packed: set[str]) -> dict[str, tuple[int, ...]]:
    """Tensor shapes a compressed-tensors export would write for this skeleton.

    Every ``Linear`` in ``packed`` gets the packed trio; every other module
    gets its parameters as plain tensors.
    """

    shapes: dict[str, tuple[int, ...]] = {}
    for name, module in skeleton.named_modules():
        if isinstance(module, nn.Linear):
            out_f, in_f = module.weight.shape
            if name in packed:
                shapes[f"{name}.weight_packed"] = (out_f, in_f // 2)
                shapes[f"{name}.weight_scale"] = (out_f, in_f // 16)
                shapes[f"{name}.weight_global_scale"] = ()
            else:
                shapes[f"{name}.weight"] = (out_f, in_f)
        else:
            for parameter, tensor in module.named_parameters(recurse=False):
                shapes[f"{name}.{parameter}"] = tuple(tensor.shape)
    return shapes


def linear_names(skeleton: nn.Module) -> set[str]:
    return {name for name, module in skeleton.named_modules() if isinstance(module, nn.Linear)}


# ---------------------------------------------------------------------------
# the config decides
# ---------------------------------------------------------------------------


def test_targets_and_ignore_decide_every_module() -> None:
    skeleton = tiny_llama()
    config = nvfp4_config(ignore=["lm_head", "re:.*k_proj$"])
    packed = {n for n in linear_names(skeleton) if n != "lm_head" and not n.endswith("k_proj")}
    modules = qs.checkpoint_modules(checkpoint_for(skeleton, packed))

    resolved = qs.resolve_schemes(config, modules, skeleton)
    resolved.check_against_checkpoint()

    quantized = {item.module.name for item in resolved.quantized()}
    assert quantized == packed
    assert resolved.modules["lm_head"].scheme is None
    assert resolved.modules["model.layers.1.self_attn.k_proj"].scheme is None
    assert resolved.modules["model.norm"].scheme is None
    q_proj = resolved.modules["model.layers.0.self_attn.q_proj"]
    assert q_proj.group == "group_0"
    assert q_proj.scheme.weights.num_bits == 4
    assert q_proj.scheme.weights.group_size == 16


def test_a_leaf_the_skeleton_does_not_list_is_still_resolved() -> None:
    # transformers 5 fuses routed experts into one module, so a checkpoint's
    # per-expert `experts.N.gate_proj` has no entry in named_modules(). The
    # file packs it; that makes it a Linear for a class-name target.
    skeleton = tiny_llama()
    config = nvfp4_config(ignore=["lm_head"])
    packed = linear_names(skeleton) - {"lm_head"}
    shapes = checkpoint_for(skeleton, packed)
    leaf = "model.layers.0.mlp.experts.3.gate_proj"
    shapes[f"{leaf}.weight_packed"] = (128, 32)
    shapes[f"{leaf}.weight_scale"] = (128, 4)
    shapes[f"{leaf}.weight_global_scale"] = ()

    resolved = qs.resolve_schemes(config, qs.checkpoint_modules(shapes), skeleton)
    resolved.check_against_checkpoint()

    assert resolved.modules[leaf].scheme is not None
    assert resolved.modules[leaf].group == "group_0"


def test_ignore_needs_no_class() -> None:
    # A module outside the skeleton entirely (an MTP head stored beside the
    # model) is left alone by a regex that names it, without ever needing to
    # know what class it is.
    skeleton = tiny_llama()
    config = nvfp4_config(ignore=["lm_head", "re:^mtp.*"])
    shapes = checkpoint_for(skeleton, linear_names(skeleton) - {"lm_head"})
    shapes["mtp.fc.weight"] = (64, 128)
    shapes["mtp.layers.0.mlp.gate.weight"] = (8, 64)

    resolved = qs.resolve_schemes(config, qs.checkpoint_modules(shapes), skeleton)
    resolved.check_against_checkpoint()

    assert resolved.modules["mtp.fc"].scheme is None
    assert resolved.modules["mtp.layers.0.mlp.gate"].scheme is None


# ---------------------------------------------------------------------------
# a checkpoint that contradicts its config is refused, not guessed at
# ---------------------------------------------------------------------------


def test_a_packed_module_the_config_ignores_is_refused() -> None:
    skeleton = tiny_llama()
    config = nvfp4_config(ignore=["lm_head", "re:.*k_proj$"])
    # the file packs k_proj anyway
    modules = qs.checkpoint_modules(checkpoint_for(skeleton, linear_names(skeleton) - {"lm_head"}))

    resolved = qs.resolve_schemes(config, modules, skeleton)
    with pytest.raises(ValueError, match="match no config group"):
        resolved.check_against_checkpoint()


def test_a_packed_module_no_target_covers_is_refused() -> None:
    skeleton = tiny_llama()
    config = nvfp4_config(ignore=["lm_head"])
    shapes = checkpoint_for(skeleton, linear_names(skeleton) - {"lm_head"})
    # a norm, which `Linear` does not cover, shows up packed
    shapes["model.norm.weight_packed"] = (64,)
    del shapes["model.norm.weight"]

    with pytest.raises(ValueError, match="neither ignores nor targets"):
        qs.resolve_schemes(config, qs.checkpoint_modules(shapes), skeleton)


def test_a_module_the_config_quantizes_but_the_file_did_not_pack_is_refused() -> None:
    skeleton = tiny_llama()
    config = nvfp4_config(ignore=["lm_head"])
    # o_proj is a Linear the config targets, but the file stores it plain
    modules = qs.checkpoint_modules(
        checkpoint_for(skeleton, linear_names(skeleton) - {"lm_head", "model.layers.0.self_attn.o_proj"})
    )

    resolved = qs.resolve_schemes(config, modules, skeleton)
    with pytest.raises(ValueError, match="have no weight_packed"):
        resolved.check_against_checkpoint()


def test_a_non_compressed_tensors_config_is_not_ours() -> None:
    assert qs.quantization_config_of({"quantization_config": {"quant_method": "modelopt"}}) is None
    assert qs.quantization_config_of({}) is None


# ---------------------------------------------------------------------------
# the real thing
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not REAL_EXPORT.is_dir(), reason="RedHatAI 35B-A3B NVFP4 export not on this disk")
def test_redhat_35b_a3b_resolves_exactly_what_it_packed() -> None:
    resolved = qs.resolve_checkpoint(REAL_EXPORT)
    assert resolved is not None
    quantized = sum(1 for _ in resolved.quantized())
    packed = sum(1 for item in resolved.modules.values() if item.module.packed)
    # 40 layers x (256 experts x 3 + shared expert x 3) + 10 full-attention layers x 4
    assert quantized == packed == 30880
    assert all(item.scheme is None for name, item in resolved.modules.items() if name.startswith("mtp."))
