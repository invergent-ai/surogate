"""CPU-only DSL compile tests for GLM-5.3-Flash (``glm5_next``).

No GPU, no weights, no C++ extension: a hand-written mini config goes through
the Python DSL and the emitted IR is asserted on — the two-axis block schedule
(3x KDA + 1x MLA, dense before ``first_k_dense_replace``), the manifold
hyper-connections on every layer, the KDA and NoPE-MLA graphs, the sigmoid MoE
router, and the HF weight mapping for both mixer types and the experts.

Run: pytest tests/test_glm5_next_dsl.py -v --no-header
"""

from __future__ import annotations

import json

import pytest

import surogate.dsl.models  # noqa: F401 - registers nn-style models
from surogate.dsl.py_compiler import compile_model_for_hf


ARCH = "Glm5NextForConditionalGeneration"


def _mini_text_config(**overrides) -> dict:
    """8 layers: the real 3-KDA + 1-MLA cycle twice, with 3 leading dense layers.

    That schedule exercises three of the four block shapes (kda, kda_moe,
    mla_moe) and leaves mla_dense — which the released checkpoint never hits
    either — for its own test.
    """

    config = {
        "model_type": "glm5_next_text",
        "vocab_size": 256,
        "hidden_size": 64,
        "intermediate_size": 128,
        "moe_intermediate_size": 32,
        "num_hidden_layers": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        # MoE
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "n_group": 1,
        "topk_group": 1,
        "swiglu_limit": 10.0,
        "first_k_dense_replace": 3,
        # MLA (NoPE)
        "q_lora_rank": 24,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 0,
        "v_head_dim": 16,
        # KDA
        "linear_num_heads": 4,
        "linear_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "linear_lower_bound": -5.0,
        # mHC
        "hc_mult": 4,
        "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 20,
        # DSA indexer (declared, deferred)
        "index_topk": 64,
        "index_head_dim": 8,
        "index_n_heads": 2,
        "index_kpool": 4,
        "index_kpool_always_select_tail": True,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-5,
    }
    config.update(overrides)
    return config


def _full_config(text: dict | None = None, **overrides) -> dict:
    config = {
        "architectures": [ARCH],
        "model_type": "glm5_next",
        "text_config": text if text is not None else _mini_text_config(),
        "vision_config": {"model_type": "glm5_next_vision", "depth": 1},
        "tie_word_embeddings": False,
    }
    config.update(overrides)
    return config


def _raw_compile(config: dict) -> dict:
    raw = compile_model_for_hf(ARCH, config)
    return json.loads(raw) if isinstance(raw, str) else raw


def _compile(config: dict | None = None) -> dict:
    result = _raw_compile(config or _full_config())
    assert result.get("success"), f"DSL compilation failed: {result.get('errors')}"
    return result["modules"][0]


def _ops(ir: dict) -> list[str]:
    return [op["kernel_type"] for op in ir["forward"]["operations"]]


@pytest.fixture(scope="module")
def ir() -> dict:
    return _compile()


# ---------------------------------------------------------------------------
# Block schedule
# ---------------------------------------------------------------------------


class TestBlockSchedule:
    def test_two_axes_produce_the_expected_pattern(self, ir):
        config = ir["config"]
        # K = kda+dense, k = kda+moe, m = mla+moe. Layers 0-2 are dense, and
        # every 4th layer (3, 7) is MLA.
        assert config["hybrid_pattern"] == "KKKmkkkm"
        assert config["n_kda_blocks"] == 3
        assert config["n_kda_moe_blocks"] == 3
        assert config["n_mla_moe_blocks"] == 2
        assert config["n_mla_blocks"] == 0

    def test_layer_types_default_to_the_3_plus_1_cycle(self):
        """With no ``layer_types``, every 4th layer is MLA (i % 4 == 3)."""
        text = _mini_text_config(num_hidden_layers=9, first_k_dense_replace=0)
        text.pop("layer_types", None)
        ir = _compile(_full_config(text))
        assert ir["config"]["hybrid_pattern"] == "kkkmkkkmk"

    def test_full_attention_label_is_accepted_as_mla(self):
        types = ["linear_attention"] * 3 + ["full_attention"]
        ir = _compile(_full_config(_mini_text_config(num_hidden_layers=4, layer_types=types)))
        assert ir["config"]["hybrid_pattern"] == "KKKm"

    def test_explicit_mlp_layer_types_override_first_k_dense_replace(self):
        ir = _compile(
            _full_config(
                _mini_text_config(
                    first_k_dense_replace=3,
                    mlp_layer_types=["dense"] * 4 + ["sparse"] * 4,
                )
            )
        )
        # Layer 3 is MLA and now dense -> the fourth block shape appears.
        assert ir["config"]["hybrid_pattern"] == "KKKMkkkm"
        assert ir["config"]["n_mla_blocks"] == 1

    def test_all_sparse_when_no_dense_prefix(self):
        ir = _compile(_full_config(_mini_text_config(first_k_dense_replace=0)))
        assert ir["config"]["hybrid_pattern"] == "kkkmkkkm"
        assert ir["config"]["n_kda_blocks"] == 0


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


class TestGraph:
    def test_manifold_hyper_connections_on_every_layer_and_site(self, ir):
        ops = _ops(ir)
        # One mHC gate head per sublayer site: 8 layers x 2.
        assert ops.count("mhc_gates") == 16
        text = json.dumps(ir)
        assert "hc_attn_logits" in text
        assert "hc_ffn_logits" in text
        assert "hc_attn_combine_res" in text
        assert "hc_ffn_combine_res" in text

    def test_layer_norms_survive_beside_the_hyper_connections(self, ir):
        """Unlike qwen4_exp, GLM-5.3 keeps both pre-norms and a final norm."""
        params = ir["forward"]["params"]
        assert "blocks[0].ln1_weight" in params
        assert "blocks[0].ln2_weight" in params
        assert "final_norm" in ir["hf_mapping"]

    def test_kda_graph(self, ir):
        ops = _ops(ir)
        assert ops.count("kda_decay") == 6
        assert ops.count("chunk_kimi_delta_rule") == 6
        # Shared depthwise causal conv over the fused q|k|v, then the sigmoid-
        # gated per-head output norm.
        assert ops.count("mamba_conv1d") == 6
        assert ops.count("mamba_gated_rmsnorm") == 6
        assert '"gate_activation": "sigmoid"' in json.dumps(ir)

    def test_mla_graph_is_dense_and_nope(self, ir):
        ops = _ops(ir)
        assert ops.count("flash_attention_qkv") == 2
        # NoPE: nothing rotary anywhere in the text stack.
        assert "rope" not in ops
        assert "mrope" not in ops
        assert "qkv_qk_norm_rope" not in ops
        assert "freq_cis" not in json.dumps(ir["activation_layout"])

    def test_moe_router_is_sigmoid_with_a_selection_bias(self, ir):
        ops = _ops(ir)
        assert ops.count("moe_sigmoid") == 5
        assert "moe_softmax" not in ops
        assert ops.count("moe_grouped_gemm_gate_up") == 5
        topk = [op for op in ir["forward"]["operations"] if op["kernel_type"] == "moe_topk"][0]
        attrs = topk.get("attributes") or topk.get("attrs") or {}
        assert attrs.get("scaling_factor") == pytest.approx(2.5)
        # The correction bias is an extra input to the top-k, not a bias add.
        assert any("e_score_correction_bias" in str(i) for i in topk["inputs"])

    def test_shared_expert_is_ungated(self, ir):
        """GLM-5.3 adds the shared expert plainly; qwen4_exp sigmoid-gates it."""
        params = ir["forward"]["params"]
        assert "blocks[3].shared_expert_gate" in params
        assert "blocks[3].shared_expert_gate_proj_weight" not in params

    def test_residual_is_hc_mult_streams_wide(self, ir):
        slots = {s["name"]: s for s in ir["activation_layout"]["slots"]}
        assert "residual0" in slots
        assert "hc_mult * d_model" in str(slots["residual0"]["shape"])


# ---------------------------------------------------------------------------
# Parameter shapes
# ---------------------------------------------------------------------------


class TestParamShapes:
    def test_hyper_connection_shapes(self, ir):
        params = ir["forward"]["params"]
        # fn: [(2 + hc) * hc, hc * d_model] = [24, 256]; base [(2+hc)*hc]; scale [3].
        assert params["blocks[0].hc_attn_fn"]["shape"] == [24, 256]
        assert params["blocks[0].hc_attn_base"]["shape"] == [24]
        assert params["blocks[0].hc_ffn_scale"]["shape"] == [3]

    def test_kda_shapes(self, ir):
        params = ir["forward"]["params"]
        # 4 heads x 8 dim = 32; the fused q|k|v is 3x that.
        assert params["blocks[0].kda_qkv_weight"]["shape"] == [96, "C"]
        assert params["blocks[0].kda_conv_weight"]["shape"] == [96, 1, 4]
        assert params["blocks[0].kda_f_a_weight"]["shape"] == [8, "C"]
        assert params["blocks[0].kda_f_b_weight"]["shape"] == [32, 8]
        # dt_bias is per channel, A_log per head -- that is the KDA difference.
        assert params["blocks[0].kda_dt_bias"]["shape"] == [32]
        assert params["blocks[0].kda_A_log"]["shape"] == [4]
        assert params["blocks[0].kda_out_weight"]["shape"] == ["C", 32]

    def test_mla_shapes(self, ir):
        params = ir["forward"]["params"]
        assert params["blocks[3].mla_q_a_weight"]["shape"] == [24, "C"]
        # 4 heads x (qk_nope 16 + rope 0)
        assert params["blocks[3].mla_q_b_weight"]["shape"] == [64, 24]
        # NoPE, so the latent projection is kv_lora_rank wide with no rope tail.
        assert params["blocks[3].mla_kv_a_weight"]["shape"] == [16, "C"]
        # 4 heads x (qk_nope 16 + v 16)
        assert params["blocks[3].mla_kv_b_weight"]["shape"] == [128, 16]
        assert params["blocks[3].mla_out_weight"]["shape"] == ["C", 64]


# ---------------------------------------------------------------------------
# HF weight mapping
# ---------------------------------------------------------------------------


class TestHfMapping:
    def test_model_level_names(self, ir):
        mapping = ir["hf_mapping"]
        assert mapping["embedding"] == "model.language_model.embed_tokens.weight"
        assert mapping["final_norm"] == "model.language_model.norm.weight"
        assert mapping["lm_head"] == "lm_head.weight"

    def test_hyper_connections_are_flat_on_the_layer(self, ir):
        mapping = ir["hf_mapping"]
        prefix = "model.language_model.layers.{layer}"
        assert mapping["hc_attn_fn"] == f"{prefix}.hc_attn_fn"
        assert mapping["hc_attn_base"] == f"{prefix}.hc_attn_base"
        assert mapping["hc_attn_scale"] == f"{prefix}.hc_attn_scale"
        assert mapping["hc_ffn_fn"] == f"{prefix}.hc_ffn_fn"
        assert mapping["ln1_weight"] == f"{prefix}.input_layernorm.weight"
        assert mapping["ln2_weight"] == f"{prefix}.post_attention_layernorm.weight"

    def test_kda_layer_mapping(self, ir):
        mapping = ir["hf_mapping"]
        attn = "model.language_model.layers.0.self_attn"
        assert mapping["blocks[0].kda_qkv_weight"]["sources"] == [
            f"{attn}.q_proj.weight",
            f"{attn}.k_proj.weight",
            f"{attn}.v_proj.weight",
        ]
        # The checkpoint keeps one conv per projection; the runtime wants one
        # grouped conv over the concatenation.
        assert mapping["blocks[0].kda_conv_weight"]["sources"] == [
            f"{attn}.q_conv1d.weight",
            f"{attn}.k_conv1d.weight",
            f"{attn}.v_conv1d.weight",
        ]
        # Forget-gate tensors sit directly under self_attn on disk.
        assert mapping["blocks[0].kda_f_a_weight"] == f"{attn}.f_a_proj.weight"
        assert mapping["blocks[0].kda_dt_bias"] == f"{attn}.dt_bias"
        assert mapping["blocks[0].kda_A_log"] == f"{attn}.A_log"
        assert mapping["blocks[0].kda_o_norm_weight"] == f"{attn}.o_norm.weight"
        assert mapping["blocks[0].kda_out_weight"] == f"{attn}.o_proj.weight"
        # A KDA layer carries no MLA tensors.
        assert "blocks[0].mla_q_a_weight" not in mapping

    def test_mla_layer_mapping(self, ir):
        mapping = ir["hf_mapping"]
        attn = "model.language_model.layers.3.self_attn"
        assert mapping["blocks[3].mla_q_a_weight"] == f"{attn}.q_a_proj.weight"
        assert mapping["blocks[3].mla_q_a_norm_weight"] == f"{attn}.q_a_layernorm.weight"
        assert mapping["blocks[3].mla_kv_a_weight"] == f"{attn}.kv_a_proj_with_mqa.weight"
        assert mapping["blocks[3].mla_kv_a_norm_weight"] == f"{attn}.kv_a_layernorm.weight"
        assert mapping["blocks[3].mla_kv_b_weight"] == f"{attn}.kv_b_proj.weight"
        assert mapping["blocks[3].mla_out_weight"] == f"{attn}.o_proj.weight"
        # An MLA layer carries no KDA tensors.
        assert "blocks[3].kda_qkv_weight" not in mapping

    def test_expert_and_dense_ffn_mapping(self, ir):
        mapping = ir["hf_mapping"]
        mlp0 = "model.language_model.layers.0.mlp"
        mlp3 = "model.language_model.layers.3.mlp"
        # Dense layers: fused [up; gate].
        assert mapping["blocks[0].mlp_up_weight"]["sources"] == [
            f"{mlp0}.up_proj.weight",
            f"{mlp0}.gate_proj.weight",
        ]
        assert mapping["blocks[0].mlp_down_weight"] == f"{mlp0}.down_proj.weight"
        assert "blocks[0].router_weight" not in mapping
        # Sparse layers: per-expert tensors, stacked (and gate/up fused).
        gate_up = mapping["blocks[3].experts_gate_up"]
        assert gate_up["type"] == "stack_experts"
        assert gate_up["pattern"] == f"{mlp3}.experts.{{expert}}.gate_proj.weight"
        assert gate_up["fuse_gate_up"] is True
        assert mapping["blocks[3].experts_down"]["pattern"] == (
            f"{mlp3}.experts.{{expert}}.down_proj.weight"
        )
        assert mapping["blocks[3].router_weight"] == f"{mlp3}.gate.weight"
        assert mapping["blocks[3].e_score_correction_bias"] == f"{mlp3}.gate.e_score_correction_bias"
        # Shared expert is plural on disk and has no gate row.
        assert mapping["blocks[3].shared_expert_gate"] == f"{mlp3}.shared_experts.gate_proj.weight"
        assert mapping["blocks[3].shared_expert_down"] == f"{mlp3}.shared_experts.down_proj.weight"


# ---------------------------------------------------------------------------
# Config plumbing, including the deferred subsystems
# ---------------------------------------------------------------------------


class TestConfig:
    def test_deferred_subsystem_hyperparameters_reach_the_runtime_config(self, ir):
        """The DSA indexer is not in the graph, but a serve-spec generator has
        to be able to read its geometry off the compiled config."""
        config = ir["config"]
        assert config["index_topk"] == 64
        assert config["index_n_heads"] == 2
        assert config["index_head_dim"] == 8
        assert config["index_kpool"] == 4
        assert config["has_dsa_indexer"] is True
        # Clamp limit is carried but deliberately not applied in the graph.
        assert config["swiglu_limit"] == pytest.approx(10.0)

    def test_shared_expert_width_is_moe_width_times_count(self, ir):
        config = ir["config"]
        assert config["shared_expert_intermediate"] == 32
        ir2 = _compile(_full_config(_mini_text_config(n_shared_experts=2)))
        assert ir2["config"]["shared_expert_intermediate"] == 64

    def test_linear_attn_config_is_read_when_the_flat_keys_are_absent(self):
        ir = _compile(
            _full_config(
                _mini_text_config(
                    linear_head_dim=None,
                    linear_num_heads=None,
                    linear_conv_kernel_dim=None,
                    linear_lower_bound=None,
                    linear_attn_config={
                        "head_dim": 16,
                        "num_heads": 2,
                        "short_conv_kernel_size": 3,
                        "gate_lower_bound": -4.0,
                    },
                )
            )
        )
        config = ir["config"]
        # The resolved geometry reaches the exported config, not the flat keys.
        assert config["linear_head_dim"] == 16
        assert config["linear_num_heads"] == 2
        assert config["linear_conv_kernel_dim"] == 3
        # 2 heads x 16 dim = 32; fused q|k|v = 96, conv kernel 3.
        assert ir["forward"]["params"]["blocks[0].kda_conv_weight"]["shape"] == [96, 1, 3]
        assert ir["forward"]["params"]["blocks[0].kda_f_a_weight"]["shape"] == [16, "C"]

    def test_kda_geometry_stated_twice_and_disagreeing_is_rejected(self):
        result = _raw_compile(
            _full_config(
                _mini_text_config(
                    linear_head_dim=8,
                    linear_attn_config={"head_dim": 16},
                )
            )
        )
        assert not result.get("success")

    def test_hf_config_binding(self, ir):
        assert ir["hf_config"]["architecture"] == ARCH
        assert ir["hf_config"]["model_type"] == "glm5_next"


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


class TestRejections:
    def test_rejects_rotary_head_dim(self):
        result = _raw_compile(_full_config(_mini_text_config(qk_rope_head_dim=8)))
        assert not result.get("success")

    def test_rejects_unknown_layer_type(self):
        result = _raw_compile(
            _full_config(_mini_text_config(num_hidden_layers=2, layer_types=["conv", "conv"]))
        )
        assert not result.get("success")

    def test_rejects_layer_types_length_mismatch(self):
        result = _raw_compile(
            _full_config(_mini_text_config(layer_types=["linear_attention"] * 3))
        )
        assert not result.get("success")

    def test_rejects_index_topk_not_divisible_by_kpool(self):
        result = _raw_compile(_full_config(_mini_text_config(index_topk=63, index_kpool=4)))
        assert not result.get("success")
