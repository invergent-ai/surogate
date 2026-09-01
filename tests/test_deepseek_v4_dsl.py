"""CPU-only DSL compile tests for DeepSeek-V4 (``DeepseekV4ForCausalLM``).

No GPU, no weights, no C++ extension: a hand-written mini config is pushed through the
Python DSL and the emitted IR is asserted on — the two-axis layer schedule (attention
type x MoE type), block counts, the shared-KV MQA attention skeleton (one KV head, per-head
sink, sliding window, grouped output projection), the sqrt-softplus MoE, the HF weight
mapping for both a ``hash_moe`` and a ``moe`` layer, and the explicit deferral flags.

Run: pytest tests/test_deepseek_v4_dsl.py -q
"""

from __future__ import annotations

import json

import pytest

import surogate.dsl.models  # noqa: F401 - registers nn-style models
from surogate.dsl.models.deepseek_v4 import (
    DEEPSEEK_V4_DEFERRED_HF_TENSORS,
    DEEPSEEK_V4_DEFERRED_MECHANISMS,
)
from surogate.dsl.py_compiler import compile_model_for_hf


def _mini_config(**overrides) -> dict:
    """6 layers, so the default schedule yields both HCA and CSA and both hash_moe and moe."""

    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "vocab_size": 256,
        "hidden_size": 64,
        "moe_intermediate_size": 32,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "q_lora_rank": 16,
        "o_groups": 2,
        "o_lora_rank": 8,
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "scoring_func": "sqrtsoftplus",
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.5,
        "max_position_embeddings": 1024,
        "rms_norm_eps": 1e-6,
        "sliding_window": 8,
        "swiglu_limit": 10.0,
        "partial_rotary_factor": 0.25,
        "compress_rates": {"compressed_sparse_attention": 4, "heavily_compressed_attention": 128},
        "rope_theta": 10000.0,
        "compress_rope_theta": 160000.0,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "index_n_heads": 4,
        "index_head_dim": 8,
        "index_topk": 16,
        "num_nextn_predict_layers": 1,
        "tie_word_embeddings": False,
    }
    config.update(overrides)
    return config


def _compile(config: dict | None = None) -> dict:
    raw = compile_model_for_hf("DeepseekV4ForCausalLM", config or _mini_config())
    result = json.loads(raw) if isinstance(raw, str) else raw
    assert result.get("success"), f"DSL compilation failed: {result.get('errors')}"
    return result["modules"][0]


def _compile_failure(config: dict) -> dict:
    raw = compile_model_for_hf("DeepseekV4ForCausalLM", config)
    return json.loads(raw) if isinstance(raw, str) else raw


def _ops(ir: dict) -> list[dict]:
    return ir["forward"]["operations"]


def _kernels(ir: dict) -> list[str]:
    return [op["kernel_type"] for op in _ops(ir)]


def _block_schemas(ir: dict) -> list[dict]:
    return ir["forward"]["metadata"]["block_schemas"]


@pytest.fixture(scope="module")
def ir() -> dict:
    return _compile()


# ---------------------------------------------------------------------------
# Layer schedule
# ---------------------------------------------------------------------------


class TestSchedule:
    def test_default_schedule_matches_hf_post_init(self, ir):
        """HF default: 2x HCA bootstrap + an interleave over the remaining n-2 slots
        (which lands as HCA on layers 0/1/2, then CSA on odd / HCA on even), crossed with
        3x hash_moe then moe."""

        config = ir["config"]
        assert config["hybrid_pattern"] == "hhhCHC"
        assert config["n_hca_hash_blocks"] == 3
        assert config["n_csa_moe_blocks"] == 2
        assert config["n_hca_moe_blocks"] == 1
        assert config["n_sliding_moe_blocks"] == 0
        assert config["n_sliding_hash_blocks"] == 0
        assert config["n_csa_hash_blocks"] == 0
        assert config["num_hash_layers"] == 3

        by_layer = [entry["block_type"] for entry in _block_schemas(ir)]
        assert by_layer == ["hca_hash", "hca_hash", "hca_hash", "csa_moe", "hca_moe", "csa_moe"]

    def test_explicit_schedule_including_sliding_layers(self):
        ir = _compile(
            _mini_config(
                layer_types=[
                    "sliding_attention",
                    "compressed_sparse_attention",
                    "heavily_compressed_attention",
                    "sliding_attention",
                    "compressed_sparse_attention",
                    "heavily_compressed_attention",
                ],
                mlp_layer_types=["hash_moe", "hash_moe", "moe", "moe", "moe", "moe"],
            )
        )
        assert ir["config"]["hybrid_pattern"] == "scHSCH"
        assert ir["config"]["n_sliding_hash_blocks"] == 1
        assert ir["config"]["n_csa_hash_blocks"] == 1
        assert ir["config"]["n_hca_moe_blocks"] == 2
        assert ir["config"]["n_sliding_moe_blocks"] == 1
        assert ir["config"]["n_csa_moe_blocks"] == 1

    def test_rejects_unknown_layer_type(self):
        result = _compile_failure(_mini_config(layer_types=["full_attention"] * 6))
        assert not result.get("success")

    def test_rejects_multi_kv_head(self):
        """V4 is shared-KV MQA: one KV head, and K and V are the same tensor."""

        result = _compile_failure(_mini_config(num_key_value_heads=8))
        assert not result.get("success")

    def test_rejects_foreign_scoring_function(self):
        result = _compile_failure(_mini_config(scoring_func="sigmoid"))
        assert not result.get("success")


# ---------------------------------------------------------------------------
# Attention skeleton
# ---------------------------------------------------------------------------


class TestAttention:
    def test_sliding_window_and_sinks_on_every_layer(self, ir):
        attn_ops = [op for op in _ops(ir) if op["kernel_type"] == "flash_attention"]
        assert len(attn_ops) == 6  # every layer, including the CSA/HCA ones
        for op in attn_ops:
            assert op["attrs"]["window_size"] == 8
            assert op["attrs"]["causal"] is True
            assert op["attrs"]["softmax_scale"] == pytest.approx(32**-0.5)
            # gpt-oss style per-head sink is a second input to the attention op
            assert len(op["inputs"]) == 2
            assert op["inputs"][1].endswith(".sinks")

    def test_shared_kv_head_is_one_tensor_used_twice(self, ir):
        """One `kv_proj` per layer feeds both the K and the V slot of the packed QKV."""

        params = ir["forward"]["params"]
        assert params["blocks[0].kv_proj_weight"]["shape"] == [32, 64]  # [head_dim, C], ONE head
        # a copy node per layer is the second (V) reference to the same projected tensor
        assert _kernels(ir).count("copy") == 6
        concat = [op for op in _ops(ir) if op["kernel_type"] == "concat"]
        assert len(concat) == 6
        # [num_query_heads, 1 key head, 1 value head]
        assert concat[0]["attrs"]["split_size"] == [4, 1, 1]

    def test_low_rank_query_and_unweighted_q_norm(self, ir):
        params = ir["forward"]["params"]
        assert params["blocks[0].q_a_proj_weight"]["shape"] == [16, 64]  # [q_lora_rank, C]
        assert params["blocks[0].q_a_norm_weight"]["shape"] == [16]
        assert params["blocks[0].q_b_proj_weight"]["shape"] == [128, 16]  # [heads*head_dim, rank]
        # `q_b_norm` is DeepseekV4UnweightedRMSNorm — no checkpoint tensor, and the graph
        # feeds a ones vector to the rmsnorm instead (one per layer).
        assert "blocks[0].q_b_norm_weight" not in params
        assert _kernels(ir).count("ones") == 6

    def test_grouped_output_projection(self, ir):
        params = ir["forward"]["params"]
        # o_a_proj is block-diagonal: o_groups x (heads*head_dim/o_groups -> o_lora_rank)
        assert params["blocks[0].o_a_proj_weight"]["shape"] == [2 * 8, (4 * 32) // 2]
        assert params["blocks[0].o_b_proj_weight"]["shape"] == [64, 2 * 8]
        # one batched GEMM per layer runs the g independent group projections
        assert _kernels(ir).count("batched_matmul") == 6

    def test_partial_rope_table_width(self, ir):
        # partial_rotary_factor 0.25 of head_dim 32 -> 8 rotated channels, 4 (cos, sin) pairs
        assert ir["config"]["rotary_dim"] == 8
        assert ir["forward"]["params"]["blocks[0].rope_freqs"]["shape"] == [1024, 4, 2]
        rope_ops = [op for op in _ops(ir) if op["kernel_type"] == "rope"]
        assert len(rope_ops) == 6
        assert rope_ops[0]["attrs"]["rotary_dim"] == 8


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------


class TestMoE:
    def test_moe_on_every_layer_with_shared_expert(self, ir):
        kernels = _kernels(ir)
        # No `first_k_dense_replace`: every one of the 6 layers is sparse.
        assert kernels.count("moe_topk") == 6
        assert kernels.count("moe_grouped_gemm_gate_up") == 6
        assert kernels.count("moe_grouped_gemm_down") == 6
        assert kernels.count("swiglu") == 6
        # ...plus one ungated shared expert per layer (silu * up, then down).
        assert kernels.count("silu") == 6

    def test_batched_expert_layout(self, ir):
        params = ir["forward"]["params"]
        assert params["blocks[0].experts_gate_up"]["shape"] == [8, 2 * 32, 64]  # [E, 2*I, C]
        assert params["blocks[0].experts_down"]["shape"] == [8, 64, 32]  # [E, C, I]
        assert params["blocks[0].shared_expert_gate"]["shape"] == [32, 64]
        assert params["blocks[0].shared_expert_down"]["shape"] == [64, 32]

    def test_router_score_is_softplus_and_bias_is_selection_only(self, ir):
        """`sqrtsoftplus` is `softplus(x).sqrt()`; the DSL has softplus but no elementwise
        sqrt, so the emitted score is the softplus and `router_sqrt_deferred` says so."""

        assert _kernels(ir).count("softplus") == 6
        assert ir["config"]["router_sqrt_deferred"] is True

        topk = [op for op in _ops(ir) if op["kernel_type"] == "moe_topk"]
        assert len(topk) == 6
        # Layer 3 is a `moe` layer: selection uses (score + e_score_correction_bias) while
        # the weights come from the unbiased score.
        moe_layer = next(op for op in topk if op["inputs"][0].startswith("blocks[3]."))
        assert moe_layer["inputs"][1] == "blocks[3].e_score_correction_bias"
        # HF's DeepseekV4TopKRouter renormalises unconditionally and folds
        # routed_scaling_factor into the weights.
        assert moe_layer["attrs"]["normalize"] is True
        assert moe_layer["attrs"]["scaling_factor"] == pytest.approx(1.5)
        assert moe_layer["attrs"]["top_k"] == 2

        # Layer 0 is a `hash_moe` layer: no correction bias tensor at all.
        hash_layer = next(op for op in topk if op["inputs"][0].startswith("blocks[0]."))
        assert len(hash_layer["inputs"]) == 1

    def test_hash_table_declared_on_bootstrap_layers_only(self, ir):
        params = ir["forward"]["params"]
        assert params["blocks[0].tid2eid"]["shape"] == [256, 2]  # [vocab, top_k]
        assert "blocks[0].e_score_correction_bias" not in params
        assert "blocks[3].tid2eid" not in params
        assert params["blocks[3].e_score_correction_bias"]["shape"] == [8]


# ---------------------------------------------------------------------------
# HF weight mapping
# ---------------------------------------------------------------------------


class TestHFMapping:
    def test_model_level_names(self, ir):
        mapping = ir["hf_mapping"]
        assert mapping["embedding"] == "model.embed_tokens.weight"
        # V4 keeps a real final norm (the mHC head collapses the streams before it),
        # unlike qwen4_exp where the output mix replaced it.
        assert mapping["final_norm"] == "model.norm.weight"
        assert mapping["lm_head"] == "lm_head.weight"
        assert not any("{prefix}" in str(v) for v in mapping.values())

    def test_attention_names(self, ir):
        mapping = ir["hf_mapping"]
        prefix = "model.layers.{layer}.self_attn"
        assert mapping["q_a_proj_weight"] == f"{prefix}.q_a_proj.weight"
        assert mapping["q_a_norm_weight"] == f"{prefix}.q_a_norm.weight"
        assert mapping["q_b_proj_weight"] == f"{prefix}.q_b_proj.weight"
        assert mapping["kv_proj_weight"] == f"{prefix}.kv_proj.weight"
        assert mapping["kv_norm_weight"] == f"{prefix}.kv_norm.weight"
        assert mapping["o_a_proj_weight"] == f"{prefix}.o_a_proj.weight"
        assert mapping["o_b_proj_weight"] == f"{prefix}.o_b_proj.weight"
        assert mapping["sinks"] == f"{prefix}.sinks"
        assert mapping["ln1_weight"] == "model.layers.{layer}.input_layernorm.weight"
        assert mapping["ln2_weight"] == "model.layers.{layer}.post_attention_layernorm.weight"

    def test_expert_halves_are_swapped_not_passed_through(self, ir):
        """V4 stores gate_up_proj as [gate; up]; surogate's swiglu kernel reads [up; gate].
        The mapping names the repack so a loader that cannot do it fails loudly instead of
        training with gate and up exchanged."""

        spec = ir["hf_mapping"]["experts_gate_up"]
        assert spec["type"] == "transform"
        assert spec["source"] == "model.layers.{layer}.mlp.experts.gate_up_proj"
        assert spec["fn"] == "swap_gate_up_halves"
        assert ir["hf_mapping"]["experts_down"] == "model.layers.{layer}.mlp.experts.down_proj"

    def test_shared_expert_is_plural_in_v4(self, ir):
        mapping = ir["hf_mapping"]
        assert mapping["shared_expert_gate"] == "model.layers.{layer}.mlp.shared_experts.gate_proj.weight"
        assert mapping["shared_expert_up"] == "model.layers.{layer}.mlp.shared_experts.up_proj.weight"
        assert mapping["shared_expert_down"] == "model.layers.{layer}.mlp.shared_experts.down_proj.weight"

    def test_per_layer_router_tensor_follows_the_mlp_schedule(self, ir):
        mapping = ir["hf_mapping"]
        # hash_moe layer 0
        assert mapping["blocks[0].tid2eid"] == "model.layers.0.mlp.gate.tid2eid"
        assert "blocks[0].e_score_correction_bias" not in mapping
        # moe layer 3
        assert mapping["blocks[3].e_score_correction_bias"] == "model.layers.3.mlp.gate.e_score_correction_bias"
        assert "blocks[3].tid2eid" not in mapping
        # the router matrix itself is on both
        assert mapping["blocks[0].router_weight"] == "model.layers.0.mlp.gate.weight"
        assert mapping["blocks[3].router_weight"] == "model.layers.3.mlp.gate.weight"

    def test_csa_layer_carries_the_same_attention_tensors_as_hca(self, ir):
        """The attention geometry is identical across layer types; only the deferred
        compressor branch differs."""

        mapping = ir["hf_mapping"]
        for layer in (0, 3):  # hca_hash, csa_moe
            assert mapping[f"blocks[{layer}].q_a_proj_weight"] == f"model.layers.{layer}.self_attn.q_a_proj.weight"
            assert mapping[f"blocks[{layer}].sinks"] == f"model.layers.{layer}.self_attn.sinks"
            assert mapping[f"blocks[{layer}].o_a_proj_weight"] == f"model.layers.{layer}.self_attn.o_a_proj.weight"


# ---------------------------------------------------------------------------
# Block schemas
# ---------------------------------------------------------------------------


class TestBlockSchemas:
    def test_families_and_routing_kinds(self, ir):
        schemas = {entry["layer"]: entry["schema"] for entry in _block_schemas(ir)}
        assert schemas[0]["attrs"]["block_family"] == "deepseek_v4_hca_hash_moe"
        assert schemas[0]["attrs"]["attention_kind"] == "hca"
        assert schemas[0]["attrs"]["mlp_kind"] == "hash_moe"
        assert schemas[0]["routing"]["kind"] == "hash_topk"
        assert schemas[0]["routing"]["scoring_bias"] is False

        assert schemas[3]["attrs"]["block_family"] == "deepseek_v4_csa_moe"
        assert schemas[3]["attrs"]["attention_kind"] == "csa"
        assert schemas[3]["routing"]["kind"] == "topk_sqrtsoftplus"
        assert schemas[3]["routing"]["scoring_bias"] is True
        assert schemas[3]["routing"]["norm_topk_prob"] is True

    def test_expert_parallel_contract(self, ir):
        schema = _block_schemas(ir)[3]["schema"]
        slots = {slot["name"]: slot for slot in schema["slots"]}
        assert slots["router_weight"]["distribution"]["kind"] == "router_replicated"
        for name in ("experts_gate_up", "experts_down"):
            assert slots[name]["grouped"] is True
            assert slots[name]["distribution"]["kind"] == "expert_parallel"
            assert slots[name]["distribution"]["global_experts"] == "num_experts"
        assert schema["ep_topology"]["ep_size_param"] == "ep_size"

    def test_compressor_serve_objects_follow_the_attention_schedule(self):
        ir = _compile(
            _mini_config(
                layer_types=[
                    "sliding_attention",
                    "compressed_sparse_attention",
                    "heavily_compressed_attention",
                ]
                * 2,
                mlp_layer_types=["moe"] * 6,
            )
        )
        by_layer = {entry["layer"]: entry["schema"]["serve_objects"] for entry in _block_schemas(ir)}
        names = {layer: {obj["name"] for obj in objs} for layer, objs in by_layer.items()}

        # sliding layers have no long-range branch at all
        assert not any(n.startswith("attention/compressor/") for n in names[0])
        assert not any(n.startswith("attention/indexer/") for n in names[0])
        # CSA carries both the (double-width) compressor and the Lightning Indexer
        assert "attention/compressor/key_value" in names[1]
        assert "attention/indexer/scorer" in names[1]
        # HCA carries the compressor but never an indexer
        assert "attention/compressor/key_value" in names[2]
        assert not any(n.startswith("attention/indexer/") for n in names[2])

        # every layer carries the two mHC sites, declared without components because the
        # mechanism is deferred
        for layer in range(6):
            objs = {obj["name"]: obj for obj in by_layer[layer]}
            assert objs["attn_hc/mix"]["components"] == []
            assert objs["ffn_hc/mix"]["components"] == []


# ---------------------------------------------------------------------------
# Deferrals
# ---------------------------------------------------------------------------


class TestDeferrals:
    def test_declaration_is_flagged_as_not_numerically_faithful(self, ir):
        config = ir["config"]
        assert config["numerically_faithful"] is False
        for flag in (
            "mhc_deferred",
            "compressor_deferred",
            "indexer_deferred",
            "rope_deferred",
            "hash_routing_deferred",
            "router_sqrt_deferred",
            "swiglu_clamp_deferred",
            "mtp_deferred",
        ):
            assert config[flag] is True, flag
        assert len(DEEPSEEK_V4_DEFERRED_MECHANISMS) == 9

    def test_deferred_geometry_still_reaches_the_runtime_config(self, ir):
        """Hyperparameters of the deferred subsystems are captured so a serve-spec
        generator (and any future lowering) reads them from one place."""

        config = ir["config"]
        assert config["hc_mult"] == 4
        assert config["hc_sinkhorn_iters"] == 20
        assert config["compress_rate_csa"] == 4
        assert config["compress_rate_hca"] == 128
        assert config["index_n_heads"] == 4
        assert config["index_head_dim"] == 8
        assert config["index_topk"] == 16
        assert config["num_nextn_predict_layers"] == 1

    def test_deferred_tensors_are_named_but_not_loaded(self, ir):
        """mHC / compressor / indexer tensors are recorded by name, but the graph does not
        allocate them — a deferral must not become a silently-untrained parameter."""

        params = ir["forward"]["params"]
        assert "blocks[0].attn_hc_fn" not in params
        assert "blocks[3].compressor_kv_proj" not in params
        assert "hc_head_fn" not in ir["params"]

        assert DEEPSEEK_V4_DEFERRED_HF_TENSORS["attn_hc.fn"] == "model.layers.{layer}.attn_hc.fn"
        assert DEEPSEEK_V4_DEFERRED_HF_TENSORS["hc_head.fn"] == "model.hc_head.hc_fn"
        assert (
            DEEPSEEK_V4_DEFERRED_HF_TENSORS["indexer.scorer"]
            == "model.layers.{layer}.self_attn.compressor.indexer.scorer.weights_proj.weight"
        )
        # none of them is wired into the block mapping
        mapping = ir["hf_mapping"]
        assert not any(k.endswith("attn_hc_fn") or "compressor" in k for k in mapping)
