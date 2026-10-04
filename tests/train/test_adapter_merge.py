"""`surogate merge` (surogate/utils/adapter_merge.py) on CPU, against tiny synthetic checkpoints.

The adapters are named the way the trainer's export writes them (lora_weights_manager.cpp
iterate_tensors): ``base_model.model.model.layers.{L}`` whatever the checkpoint's own layer
path, routed experts per expert (``mlp.experts.{e}.{gate,up,down}_proj``) and the shared
expert as ``mlp.shared_experts.*``. Expected results are computed independently, from the
HF semantics of each checkpoint layout.
"""

import filecmp
import json
import math

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from surogate.utils.adapter_merge import (
    AdapterMergeError,
    FusedExpertLayout,
    _plan_lora_merge,
    merge_adapter,
)

PREFIX = "base_model.model."


def _adapter_pair(module: str, a_shape=(2, 3), b_shape=(4, 2)):
    return {
        f"{PREFIX}{module}.lora_A.weight": torch.ones(*a_shape),
        f"{PREFIX}{module}.lora_B.weight": torch.ones(*b_shape),
    }


def _random_pair(gen, module: str, in_features: int, out_features: int, rank: int, dtype=torch.bfloat16):
    return {
        f"{PREFIX}{module}.lora_A.weight": torch.randn(rank, in_features, generator=gen).to(dtype),
        f"{PREFIX}{module}.lora_B.weight": torch.randn(out_features, rank, generator=gen).to(dtype),
    }


def _delta(adapter, module: str, scaling: float) -> torch.Tensor:
    a = adapter[f"{PREFIX}{module}.lora_A.weight"].float()
    b = adapter[f"{PREFIX}{module}.lora_B.weight"].float()
    return scaling * (b @ a)


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.uint8)


def _write_adapter(path, tensors, rank, alpha, **config):
    path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path / "adapter_model.safetensors"))
    cfg = {"peft_type": "LORA", "r": rank, "lora_alpha": alpha, "use_rslora": False, **config}
    (path / "adapter_config.json").write_text(json.dumps(cfg))


def _load_dir(path) -> dict[str, torch.Tensor]:
    out = {}
    for shard in sorted(path.glob("*.safetensors")):
        out.update(load_file(str(shard)))
    return out


def _shapes(tensors: dict[str, torch.Tensor]) -> dict[str, tuple[int, ...]]:
    return {key: tuple(t.shape) for key, t in tensors.items()}


# ---------------------------------------------------------------------------
# Prefix resolution (planner only)
# ---------------------------------------------------------------------------


def test_lora_lookup_prefers_qwen_language_model_over_mtp():
    adapter = _adapter_pair("model.layers.0.mlp.down_proj")
    base = {"mtp.layers.0.mlp.down_proj.weight": (4, 3), "model.language_model.layers.0.mlp.down_proj.weight": (4, 3)}

    plan = _plan_lora_merge(adapter, base)

    assert list(plan.targets) == ["model.language_model.layers.0.mlp.down_proj.weight"]


def test_lora_lookup_keeps_unambiguous_prefix_remap():
    adapter = _adapter_pair("model.layers.3.self_attn.q_proj")
    base = {"language_model.model.layers.3.self_attn.q_proj.weight": (4, 3)}

    plan = _plan_lora_merge(adapter, base)

    assert list(plan.targets) == ["language_model.model.layers.3.self_attn.q_proj.weight"]


def test_lora_lookup_rejects_ambiguous_suffix_matches():
    adapter = _adapter_pair("other.layers.0.mlp.down_proj")
    base = {"model.layers.0.mlp.down_proj.weight": (4, 3), "mtp.layers.0.mlp.down_proj.weight": (4, 3)}

    with pytest.raises(ValueError, match="ambiguous LoRA target"):
        _plan_lora_merge(adapter, base)


def test_prefix_probe_matches_whole_dotted_components_only():
    adapter = _adapter_pair("model.layers.0.mlp.down_proj")
    base = {"model.language_model.sublayers.0.mlp.down_proj.weight": (4, 3)}

    plan = _plan_lora_merge(adapter, base)

    assert plan.missing == ["model.layers.0.mlp.down_proj"]


def test_prefix_remap_is_found_from_any_pair_not_only_the_first():
    # #272 cause 3: the first pair is a routed expert of a fused checkpoint. Its per-expert
    # name matches nothing, which must not hide the model.layers -> model.language_model.layers
    # remap from the attention pairs (or from the expert itself, via the fused tensor).
    adapter = {
        **_adapter_pair("model.layers.0.mlp.experts.0.gate_proj", (2, 8), (6, 2)),
        **_adapter_pair("model.layers.0.self_attn.q_proj", (2, 8), (16, 2)),
    }
    base = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": (4, 12, 8),
        "model.language_model.layers.0.self_attn.q_proj.weight": (16, 8),
        "mtp.layers.0.mlp.experts.gate_up_proj": (4, 12, 8),
        "mtp.layers.0.self_attn.q_proj.weight": (16, 8),
    }

    plan = _plan_lora_merge(adapter, base)

    assert plan.missing == []
    assert plan.prefix_remaps == [("model.", "model.language_model.")]
    assert set(plan.targets) == {
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.self_attn.q_proj.weight",
    }


def test_per_expert_checkpoints_keep_per_expert_targets():
    # Qwen3-MoE-style checkpoints store every expert separately: no fused slicing.
    adapter = {
        **_adapter_pair("model.layers.0.mlp.experts.3.gate_proj", (2, 8), (6, 2)),
        **_adapter_pair("model.layers.0.mlp.experts.3.down_proj", (2, 6), (8, 2)),
    }
    base = {
        "model.layers.0.mlp.experts.3.gate_proj.weight": (6, 8),
        "model.layers.0.mlp.experts.3.down_proj.weight": (8, 6),
    }

    plan = _plan_lora_merge(adapter, base)

    assert plan.missing == []
    assert {key: [t.expert for t, *_ in apps] for key, apps in plan.targets.items()} == {
        "model.layers.0.mlp.experts.3.gate_proj.weight": [None],
        "model.layers.0.mlp.experts.3.down_proj.weight": [None],
    }


def test_shared_experts_keep_their_name_when_the_checkpoint_has_it():
    # DeepSeek/GLM checkpoints name the shared expert as the export does.
    adapter = _adapter_pair("model.layers.0.mlp.shared_experts.up_proj", (2, 8), (5, 2))
    base = {"model.layers.0.mlp.shared_experts.up_proj.weight": (5, 8)}

    assert list(_plan_lora_merge(adapter, base).targets) == ["model.layers.0.mlp.shared_experts.up_proj.weight"]


def test_misshapen_pairs_are_rejected():
    adapter = _adapter_pair("model.layers.0.mlp.experts.9.gate_proj", (2, 8), (6, 2))  # only 4 experts
    base = {"model.layers.0.mlp.experts.gate_up_proj": (4, 12, 8)}
    with pytest.raises(AdapterMergeError, match="do not fit"):
        _plan_lora_merge(adapter, base)

    adapter = _adapter_pair("model.layers.0.self_attn.q_proj", (2, 8), (16, 2))
    with pytest.raises(AdapterMergeError, match="do not fit"):
        _plan_lora_merge(adapter, {"model.layers.0.self_attn.q_proj.weight": (16, 10)})


def test_unpaired_lora_halves_are_rejected():
    adapter = _adapter_pair("model.layers.0.self_attn.q_proj")
    del adapter[f"{PREFIX}model.layers.0.self_attn.q_proj.lora_B.weight"]
    with pytest.raises(AdapterMergeError, match="no matching lora_A/lora_B"):
        _plan_lora_merge(adapter, {"model.layers.0.self_attn.q_proj.weight": (4, 3)})


# ---------------------------------------------------------------------------
# Qwen3.5/3.6-MoE: fused routed experts, shared_expert, model.language_model prefix
# ---------------------------------------------------------------------------

HIDDEN, MOE_FF, SHARED_FF, EXPERTS, RANK, ALPHA = 8, 6, 5, 4, 2, 4
FULL_ATTENTION = {1}
LAYERS = 2


def _qwen3_5_moe_checkpoint(tmp_path, gen):
    """A two-layer qwen3_5_moe checkpoint in the layout Qwen ships, three shards."""
    base = tmp_path / "base"
    base.mkdir()

    def rnd(*shape):
        return torch.randn(*shape, generator=gen).to(torch.bfloat16)

    shards: list[dict[str, torch.Tensor]] = [{}, {}, {}]
    shards[0]["model.language_model.embed_tokens.weight"] = rnd(32, HIDDEN)
    shards[1]["model.language_model.norm.weight"] = rnd(HIDDEN)
    shards[1]["lm_head.weight"] = rnd(32, HIDDEN)
    for layer in range(LAYERS):
        pre = f"model.language_model.layers.{layer}"
        shard = shards[layer]
        shard[f"{pre}.input_layernorm.weight"] = rnd(HIDDEN)
        shard[f"{pre}.post_attention_layernorm.weight"] = rnd(HIDDEN)
        shard[f"{pre}.mlp.gate.weight"] = rnd(EXPERTS, HIDDEN)
        shard[f"{pre}.mlp.experts.gate_up_proj"] = rnd(EXPERTS, 2 * MOE_FF, HIDDEN)
        shard[f"{pre}.mlp.experts.down_proj"] = rnd(EXPERTS, HIDDEN, MOE_FF)
        shard[f"{pre}.mlp.shared_expert.gate_proj.weight"] = rnd(SHARED_FF, HIDDEN)
        shard[f"{pre}.mlp.shared_expert.up_proj.weight"] = rnd(SHARED_FF, HIDDEN)
        shard[f"{pre}.mlp.shared_expert.down_proj.weight"] = rnd(HIDDEN, SHARED_FF)
        shard[f"{pre}.mlp.shared_expert_gate.weight"] = rnd(1, HIDDEN)
        if layer in FULL_ATTENTION:
            shard[f"{pre}.self_attn.q_proj.weight"] = rnd(16, HIDDEN)
            shard[f"{pre}.self_attn.k_proj.weight"] = rnd(4, HIDDEN)
            shard[f"{pre}.self_attn.v_proj.weight"] = rnd(4, HIDDEN)
            shard[f"{pre}.self_attn.o_proj.weight"] = rnd(HIDDEN, 8)
        else:
            shard[f"{pre}.linear_attn.in_proj_qkv.weight"] = rnd(12, HIDDEN)
            shard[f"{pre}.linear_attn.out_proj.weight"] = rnd(HIDDEN, 6)
    # The MTP head repeats the layer's module names under mtp.layers.0: bait for the prefix probe.
    mtp = shards[2]
    mtp["mtp.fc.weight"] = rnd(HIDDEN, 2 * HIDDEN)
    mtp["mtp.layers.0.self_attn.q_proj.weight"] = rnd(16, HIDDEN)
    mtp["mtp.layers.0.mlp.experts.gate_up_proj"] = rnd(EXPERTS, 2 * MOE_FF, HIDDEN)
    mtp["mtp.layers.0.mlp.experts.down_proj"] = rnd(EXPERTS, HIDDEN, MOE_FF)
    mtp["mtp.layers.0.mlp.shared_expert.up_proj.weight"] = rnd(SHARED_FF, HIDDEN)

    names = [f"model-{i + 1:05d}-of-00003.safetensors" for i in range(3)]
    weight_map = {}
    for name, shard in zip(names, shards):
        save_file(shard, str(base / name), metadata={"format": "pt", "origin": "fixture"})
        weight_map.update(dict.fromkeys(shard, name))
    (base / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {"model_type": "qwen3_5_moe_text", "hidden_size": HIDDEN, "moe_intermediate_size": MOE_FF},
    }
    (base / "config.json").write_text(json.dumps(config))
    (base / "tokenizer.json").write_text("{}")
    tensors = {key: t for shard in shards for key, t in shard.items()}
    return base, names, tensors


def _qwen3_5_moe_adapter(gen):
    """The trainer's export of a q/k/v/o/gate/up/down adapter, routed experts first."""
    adapter = {}
    for layer in range(LAYERS):
        mlp = f"model.layers.{layer}.mlp"
        for e in range(EXPERTS):
            adapter |= _random_pair(gen, f"{mlp}.experts.{e}.gate_proj", HIDDEN, MOE_FF, RANK)
            adapter |= _random_pair(gen, f"{mlp}.experts.{e}.up_proj", HIDDEN, MOE_FF, RANK)
            adapter |= _random_pair(gen, f"{mlp}.experts.{e}.down_proj", MOE_FF, HIDDEN, RANK)
    for layer in range(LAYERS):
        mlp = f"model.layers.{layer}.mlp"
        adapter |= _random_pair(gen, f"{mlp}.shared_experts.up_proj", HIDDEN, SHARED_FF, RANK)
        adapter |= _random_pair(gen, f"{mlp}.shared_experts.down_proj", SHARED_FF, HIDDEN, RANK)
    for layer in sorted(FULL_ATTENTION):
        attn = f"model.layers.{layer}.self_attn"
        adapter |= _random_pair(gen, f"{attn}.q_proj", HIDDEN, 16, RANK)
        adapter |= _random_pair(gen, f"{attn}.k_proj", HIDDEN, 4, RANK)
        adapter |= _random_pair(gen, f"{attn}.v_proj", HIDDEN, 4, RANK)
        adapter |= _random_pair(gen, f"{attn}.o_proj", 8, HIDDEN, RANK)
    return adapter


def _qwen3_5_moe_reference(base, adapter, scaling):
    """W + scaling * B @ A per HF Qwen3.5-MoE semantics: gate_up_proj[e] = [gate; up] rows, [out, in]."""
    expected = dict(base)
    for layer in range(LAYERS):
        src, dst = f"model.layers.{layer}", f"model.language_model.layers.{layer}"
        gate_up = base[f"{dst}.mlp.experts.gate_up_proj"].clone()
        down = base[f"{dst}.mlp.experts.down_proj"].clone()
        for e in range(EXPERTS):
            gate = _delta(adapter, f"{src}.mlp.experts.{e}.gate_proj", scaling)
            up = _delta(adapter, f"{src}.mlp.experts.{e}.up_proj", scaling)
            gate_up[e] = (gate_up[e].float() + torch.cat([gate, up])).to(torch.bfloat16)
            down[e] = (down[e].float() + _delta(adapter, f"{src}.mlp.experts.{e}.down_proj", scaling)).to(
                torch.bfloat16
            )
        expected[f"{dst}.mlp.experts.gate_up_proj"] = gate_up
        expected[f"{dst}.mlp.experts.down_proj"] = down
        for proj in ("up_proj", "down_proj"):
            key = f"{dst}.mlp.shared_expert.{proj}.weight"
            delta = _delta(adapter, f"{src}.mlp.shared_experts.{proj}", scaling)
            expected[key] = (base[key].float() + delta).to(torch.bfloat16)
        if layer in FULL_ATTENTION:
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                key = f"{dst}.self_attn.{proj}.weight"
                delta = _delta(adapter, f"{src}.self_attn.{proj}", scaling)
                expected[key] = (base[key].float() + delta).to(torch.bfloat16)
    return expected


def test_qwen3_5_moe_fused_experts_merge_every_pair(tmp_path):
    gen = torch.Generator().manual_seed(272)
    base_dir, shard_names, base = _qwen3_5_moe_checkpoint(tmp_path, gen)
    adapter = _qwen3_5_moe_adapter(gen)
    _write_adapter(tmp_path / "adapter", adapter, RANK, ALPHA)
    scaling = ALPHA / RANK

    plan = _plan_lora_merge(adapter, _shapes(base))
    assert plan.missing == []
    assert plan.num_resolved == plan.num_pairs == len(adapter) // 2
    assert plan.num_fused == LAYERS * EXPERTS * 3

    out = tmp_path / "merged"
    merge_adapter(str(base_dir), str(tmp_path / "adapter"), str(out))

    result = _load_dir(out)
    assert result.keys() == base.keys()
    expected = _qwen3_5_moe_reference(base, adapter, scaling)
    touched = {key for key in base if not torch.equal(_bits(expected[key]), _bits(base[key]))}
    assert len(touched) == 2 * LAYERS + 2 * LAYERS + 4 * len(FULL_ATTENTION)
    for key in base:
        assert result[key].dtype == base[key].dtype, key
        assert torch.equal(_bits(result[key]), _bits(expected[key])), key
    for key in base.keys() - touched:  # untouched tensors keep their exact bytes
        assert torch.equal(_bits(result[key]), _bits(base[key])), key

    # The MTP-only shard has no target: copied as a file. Merged shards keep their metadata.
    assert filecmp.cmp(base_dir / shard_names[2], out / shard_names[2], shallow=False)
    with safe_open(str(out / shard_names[0]), framework="pt") as f:
        assert f.metadata() == {"format": "pt", "origin": "fixture"}
    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        assert (out / name).read_bytes() == (base_dir / name).read_bytes()
    assert not list(tmp_path.glob(".*merging-*"))
    (tmp_path / "plain").mkdir()  # the output dir is staged, but made like any other directory
    assert out.stat().st_mode == (tmp_path / "plain").stat().st_mode


def test_fused_expert_gate_and_up_land_in_their_halves(tmp_path):
    # Only expert 1's gate and expert 2's up are adapted: gate rows first, then up rows.
    gen = torch.Generator().manual_seed(1)
    base_dir, _, base = _qwen3_5_moe_checkpoint(tmp_path, gen)
    adapter = {
        **_random_pair(gen, "model.layers.0.mlp.experts.1.gate_proj", HIDDEN, MOE_FF, RANK),
        **_random_pair(gen, "model.layers.0.mlp.experts.2.up_proj", HIDDEN, MOE_FF, RANK),
    }
    _write_adapter(tmp_path / "adapter", adapter, RANK, ALPHA)
    merge_adapter(str(base_dir), str(tmp_path / "adapter"), str(tmp_path / "out"))

    key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    before, after = base[key], _load_dir(tmp_path / "out")[key]
    changed = (before != after).any(dim=-1)  # [E, 2M] rows that changed
    assert changed[1, :MOE_FF].all() and not changed[1, MOE_FF:].any()
    assert changed[2, MOE_FF:].all() and not changed[2, :MOE_FF].any()
    assert not changed[0].any() and not changed[3].any()


def test_missing_target_fails_and_writes_nothing(tmp_path):
    gen = torch.Generator().manual_seed(2)
    base_dir, _, _ = _qwen3_5_moe_checkpoint(tmp_path, gen)
    adapter = _qwen3_5_moe_adapter(gen)
    adapter |= _random_pair(gen, "model.layers.7.mlp.experts.0.gate_proj", HIDDEN, MOE_FF, RANK)  # no layer 7
    _write_adapter(tmp_path / "adapter", adapter, RANK, ALPHA)
    out = tmp_path / "merged"

    with pytest.raises(AdapterMergeError, match=rf"1 of {len(adapter) // 2} LoRA pairs have no target"):
        merge_adapter(str(base_dir), str(tmp_path / "adapter"), str(out))
    assert not out.exists()
    assert not list(tmp_path.glob(".*merging-*"))


def test_merge_failure_leaves_an_existing_output_dir_untouched(tmp_path, monkeypatch):
    # The SFT trainer merges into the directory that holds the adapter. A merge that fails
    # half way (here: while writing a shard) must leave it as it was.
    import surogate.utils.adapter_merge as adapter_merge

    gen = torch.Generator().manual_seed(3)
    base_dir, _, _ = _qwen3_5_moe_checkpoint(tmp_path, gen)
    out = tmp_path / "run"
    _write_adapter(out, _qwen3_5_moe_adapter(gen), RANK, ALPHA)
    before = {p.name: p.read_bytes() for p in out.iterdir()}

    calls = []

    def failing_save_file(tensors, filename, metadata=None):
        calls.append(filename)
        if len(calls) == 2:
            raise OSError("disk full")
        save_file(tensors, filename, metadata=metadata)

    monkeypatch.setattr(adapter_merge, "save_file", failing_save_file)
    with pytest.raises(OSError, match="disk full"):
        merge_adapter(str(base_dir), str(out), str(out))
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    assert not list(tmp_path.glob(".*merging-*"))


def test_merge_into_the_adapter_dir_moves_the_adapter_aside(tmp_path):
    gen = torch.Generator().manual_seed(4)
    base_dir, shard_names, _ = _qwen3_5_moe_checkpoint(tmp_path, gen)
    out = tmp_path / "run"
    _write_adapter(out, _qwen3_5_moe_adapter(gen), RANK, ALPHA)

    merge_adapter(str(base_dir), str(out), str(out))

    assert (out / "adapter" / "adapter_model.safetensors").exists()
    assert not (out / "adapter_config.json").exists()
    assert all((out / name).exists() for name in shard_names)


def test_cli_exits_nonzero_on_missing_target(tmp_path):
    from surogate.cli.merge import main

    gen = torch.Generator().manual_seed(5)
    base_dir, _, _ = _qwen3_5_moe_checkpoint(tmp_path, gen)
    adapter = _qwen3_5_moe_adapter(gen)
    _write_adapter(tmp_path / "good", adapter, RANK, ALPHA)
    adapter |= _random_pair(gen, "model.layers.0.mlp.nope", HIDDEN, 4, RANK)
    _write_adapter(tmp_path / "bad", adapter, RANK, ALPHA)

    argv = ["--base-model", str(base_dir), "--checkpoint-dir"]
    assert main([*argv, str(tmp_path / "bad"), "--output", str(tmp_path / "bad_out")]) == 1
    assert not (tmp_path / "bad_out").exists()
    assert main([*argv, str(tmp_path / "good"), "--output", str(tmp_path / "good_out")]) == 0
    assert (tmp_path / "good_out" / "model-00001-of-00003.safetensors").exists()


# ---------------------------------------------------------------------------
# Other layouts
# ---------------------------------------------------------------------------


def test_dense_merge_is_unchanged(tmp_path):
    # The non-fused path: W' = (W.float() + scaling * B @ A).to(W.dtype), every other tensor as is.
    gen = torch.Generator().manual_seed(6)
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    weights = {
        "model.embed_tokens.weight": torch.randn(16, 8, generator=gen).bfloat16(),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8, generator=gen).bfloat16(),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(8, 8, generator=gen).bfloat16(),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(12, 8, generator=gen).bfloat16(),
        "model.layers.0.mlp.down_proj.weight": torch.randn(8, 12, generator=gen).bfloat16(),
    }
    save_file(weights, str(base_dir / "model.safetensors"))
    adapter = {
        **_random_pair(gen, "model.layers.0.self_attn.q_proj", 8, 8, 4),
        **_random_pair(gen, "model.layers.0.mlp.gate_proj", 8, 12, 4),
        **_random_pair(gen, "model.layers.0.mlp.down_proj", 12, 8, 4),
    }
    _write_adapter(tmp_path / "adapter", adapter, 4, 8)

    merge_adapter(str(base_dir), str(tmp_path / "adapter"), str(tmp_path / "merged"))

    result = load_file(str(tmp_path / "merged" / "model.safetensors"))
    assert result.keys() == weights.keys()
    for name, original in weights.items():
        module = name.removesuffix(".weight")
        if f"{PREFIX}{module}.lora_A.weight" not in adapter:
            assert torch.equal(_bits(result[name]), _bits(original))
            continue
        assert torch.equal(result[name], (original.float() + _delta(adapter, module, 2.0)).bfloat16())


def test_rslora_scaling(tmp_path):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    weight = torch.zeros(4, 3)
    save_file({"model.layers.0.self_attn.q_proj.weight": weight}, str(base_dir / "model.safetensors"))
    _write_adapter(tmp_path / "adapter", _adapter_pair("model.layers.0.self_attn.q_proj"), 2, 4, use_rslora=True)

    merge_adapter(str(base_dir), str(tmp_path / "adapter"), str(tmp_path / "out"))

    merged = load_file(str(tmp_path / "out" / "model.safetensors"))["model.layers.0.self_attn.q_proj.weight"]
    assert torch.allclose(merged, torch.full((4, 3), 2.0 * 4 / math.sqrt(2)))


def test_gpt_oss_fused_experts_are_in_out_and_interleaved(tmp_path):
    # GPT-OSS: gate_up_proj [E, C, 2M] used as x @ W with gate = [..., ::2], up = [..., 1::2];
    # down_proj [E, M, C]. C == M makes down_proj square, so the layout must come from config.json.
    gen = torch.Generator().manual_seed(7)
    C = M = 6
    E = 3
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    base = {
        "model.layers.0.mlp.experts.gate_up_proj": torch.randn(E, C, 2 * M, generator=gen).bfloat16(),
        "model.layers.0.mlp.experts.down_proj": torch.randn(E, M, C, generator=gen).bfloat16(),
        "model.layers.0.mlp.router.weight": torch.randn(E, C, generator=gen).bfloat16(),
    }
    save_file(base, str(base_dir / "model.safetensors"))
    (base_dir / "config.json").write_text(json.dumps({"model_type": "gpt_oss"}))
    adapter = {}
    for e in range(E):
        adapter |= _random_pair(gen, f"model.layers.0.mlp.experts.{e}.gate_proj", C, M, RANK)
        adapter |= _random_pair(gen, f"model.layers.0.mlp.experts.{e}.up_proj", C, M, RANK)
        adapter |= _random_pair(gen, f"model.layers.0.mlp.experts.{e}.down_proj", M, C, RANK)
    _write_adapter(tmp_path / "adapter", adapter, RANK, ALPHA)

    merge_adapter(str(base_dir), str(tmp_path / "adapter"), str(tmp_path / "out"))

    result = load_file(str(tmp_path / "out" / "model.safetensors"))
    gate_up = base["model.layers.0.mlp.experts.gate_up_proj"].float().clone()
    down = base["model.layers.0.mlp.experts.down_proj"].float().clone()
    for e in range(E):
        gate_up[e, :, 0::2] += _delta(adapter, f"model.layers.0.mlp.experts.{e}.gate_proj", 2.0).T
        gate_up[e, :, 1::2] += _delta(adapter, f"model.layers.0.mlp.experts.{e}.up_proj", 2.0).T
        down[e] += _delta(adapter, f"model.layers.0.mlp.experts.{e}.down_proj", 2.0).T
    assert torch.equal(result["model.layers.0.mlp.experts.gate_up_proj"], gate_up.bfloat16())
    assert torch.equal(result["model.layers.0.mlp.experts.down_proj"], down.bfloat16())
    assert torch.equal(
        _bits(result["model.layers.0.mlp.router.weight"]), _bits(base["model.layers.0.mlp.router.weight"])
    )


def test_fused_gate_up_adapter_rows_are_up_then_gate():
    # A per-expert gate_up_proj adapter (lora_target_modules: gate_up_proj) is in the runtime's
    # [up; gate] row order (moe_grouped_gemm_gate_up.cpp; serve's lora_bind.h reads it the same
    # way); a gate-first checkpoint gets its halves exchanged.
    from surogate.utils.adapter_merge import _merge_tensor

    adapter = _adapter_pair("model.layers.0.mlp.experts.1.gate_up_proj", (1, 2), (4, 1))
    adapter[f"{PREFIX}model.layers.0.mlp.experts.1.gate_up_proj.lora_B.weight"] = torch.tensor(
        [[1.0], [2.0], [10.0], [20.0]]
    )
    plan = _plan_lora_merge(adapter, {"model.layers.0.mlp.experts.gate_up_proj": (2, 4, 2)}, FusedExpertLayout())

    merged = _merge_tensor(torch.zeros(2, 4, 2), plan.targets["model.layers.0.mlp.experts.gate_up_proj"], 1.0)

    assert merged[1, :, 0].tolist() == [10.0, 20.0, 1.0, 2.0]  # gate rows first, then up
    assert not merged[0].any()


def test_quantized_targets_are_refused(tmp_path):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    weight = torch.zeros(4, 3).to(torch.float8_e4m3fn)
    save_file({"model.layers.0.self_attn.q_proj.weight": weight}, str(base_dir / "model.safetensors"))
    _write_adapter(tmp_path / "adapter", _adapter_pair("model.layers.0.self_attn.q_proj"), 2, 4)

    with pytest.raises(AdapterMergeError, match="F8_E4M3"):
        merge_adapter(str(base_dir), str(tmp_path / "adapter"), str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()
