"""Tests for the torch.nn-like DSL authoring layer (surogate.dsl.nn)."""

import pytest

from surogate.dsl import nn
from surogate.dsl.specs import (
    ActivationScope,
    BlockSpec,
    ParamKind,
    SharePolicy,
)

# ============================================================================
# Fixtures — reusable block instances
# ============================================================================

QWEN3_CONFIG = dict(
    d_model=1024,
    num_query_heads=16,
    num_kv_heads=8,
    head_size=64,
    d_ff=3072,
    max_seq=4096,
    eps=1e-6,
    use_qkv_bias=False,
    use_qk_norm=True,
)


def _make_qwen3_block(**overrides):
    from surogate.dsl.blocks.qwen3 import Qwen3Block

    cfg = {**QWEN3_CONFIG, **overrides}
    return Qwen3Block(**cfg)


# ============================================================================
# Basic compilation
# ============================================================================


class TestBlockCompile:
    """Verify that nn.Block.compile() produces a valid BlockSpec."""

    def test_compile_returns_blockspec(self):
        blk = _make_qwen3_block()
        spec = blk.compile()
        assert isinstance(spec, BlockSpec)
        assert spec.name == "Qwen3Block"

    def test_params_are_collected(self):
        blk = _make_qwen3_block()
        spec = blk.compile()
        param_names = set(spec.params.keys())

        # Attention params (canonical names)
        assert "qkv_weight" in param_names
        assert "out_weight" in param_names
        assert "rope_freqs" in param_names
        # QK-norm params
        assert "q_norm_weight" in param_names
        assert "k_norm_weight" in param_names
        # Norm params
        assert "ln1_weight" in param_names
        assert "ln2_weight" in param_names
        # MLP params
        assert "mlp_up_weight" in param_names
        assert "mlp_down_weight" in param_names

    def test_param_shapes(self):
        blk = _make_qwen3_block()
        spec = blk.compile()
        qkv = spec.params["qkv_weight"]
        assert qkv.shape == ("QKV", "C")
        assert qkv.kind == ParamKind.TENSOR

    def test_frozen_param(self):
        blk = _make_qwen3_block()
        spec = blk.compile()
        freqs = spec.params["rope_freqs"]
        assert freqs.frozen is True
        assert freqs.dtype == "fp32"

    def test_quantizable_false_for_norms(self):
        blk = _make_qwen3_block()
        spec = blk.compile()
        assert spec.params["ln1_weight"].quantizable is False
        assert spec.params["ln2_weight"].quantizable is False
        assert spec.params["q_norm_weight"].quantizable is False


class TestActivationSlots:
    """Verify auto-generated activation slots."""

    def test_forward_slots_exist(self):
        blk = _make_qwen3_block()
        spec = blk.compile()
        slot_names = {s.name for s in spec.activations.slots}

        # Attention norm outputs (canonical names)
        assert "res_ffn" in slot_names
        assert "ln1" in slot_names
        assert "ln1_rstd" in slot_names
        # Attention outputs
        assert "qkv" in slot_names
        assert "qkv_rope" in slot_names
        assert "att" in slot_names
        assert "lse" in slot_names
        assert "att_out" in slot_names
        # MLP norm
        assert "res_att" in slot_names
        assert "ln2" in slot_names
        # MLP
        assert "mlp_up" in slot_names
        assert "swiglu" in slot_names
        assert "mlp_down" in slot_names

    def test_share_policies(self):
        blk = _make_qwen3_block()
        spec = blk.compile()
        slot_map = {s.name: s for s in spec.activations.slots}

        # Norm rstds should be per_layer (save=True)
        assert slot_map["ln1_rstd"].share_policy == SharePolicy.PER_LAYER
        assert slot_map["ln1_rstd"].save_for_backward is True
        # Attention output should be always_recompute
        assert slot_map["att"].share_policy == SharePolicy.ALWAYS_RECOMPUTE
        # MLP outputs should be when_recomputed
        assert slot_map["mlp_up"].share_policy == SharePolicy.WHEN_RECOMPUTED

    def test_gradient_slots_auto_generated(self):
        blk = _make_qwen3_block()
        spec = blk.compile()
        grad_names = {s.name for s in spec.activations.gradient_slots}

        for fwd_slot in spec.activations.slots:
            if fwd_slot.scope == ActivationScope.BLOCK:
                assert f"d_{fwd_slot.name}" in grad_names

    def test_qk_norm_conditional_slots(self):
        blk = _make_qwen3_block(use_qk_norm=True)
        spec = blk.compile()
        slot_names = {s.name for s in spec.activations.slots}
        assert "q_rstd" in slot_names
        assert "k_rstd" in slot_names

    def test_qk_norm_slots_carry_condition(self):
        """Conditional slots are always declared but carry a condition expression."""
        blk = _make_qwen3_block(use_qk_norm=True)
        spec = blk.compile()
        slot_map = {s.name: s for s in spec.activations.slots}
        assert slot_map["q_rstd"].condition_expr == "use_qk_norm"
        assert slot_map["k_rstd"].condition_expr == "use_qk_norm"


class TestGraphNodes:
    """Verify the traced graph has the expected operations."""

    def test_graph_has_nodes(self):
        blk = _make_qwen3_block()
        spec = blk.compile()
        graph = spec.forward._traced_graph
        assert len(graph.nodes) > 0

    def test_graph_contains_expected_ops(self):
        blk = _make_qwen3_block()
        spec = blk.compile()
        graph = spec.forward._traced_graph
        op_types = {n.op for n in graph.nodes}
        assert "fused_residual_rmsnorm" in op_types
        assert "matmul" in op_types
        assert "view" in op_types
        assert "flash_attention" in op_types
        assert "swiglu" in op_types


class TestConditionalBias:
    """Verify conditional params with use_qkv_bias."""

    def test_bias_params_always_declared_with_condition(self):
        """Conditional params are always declared (with condition), not omitted."""
        blk = _make_qwen3_block(use_qkv_bias=False)
        spec = blk.compile()
        # Bias params exist with a condition, regardless of use_qkv_bias value
        assert "qkv_bias" in spec.params
        assert "out_bias" in spec.params
        assert spec.params["qkv_bias"].optional is True

    def test_bias_params_when_enabled(self):
        blk = _make_qwen3_block(use_qkv_bias=True)
        spec = blk.compile()
        assert "qkv_bias" in spec.params
        assert "out_bias" in spec.params


class TestMinimalBlock:
    """Test a minimal custom block to verify the framework."""

    def test_simple_mlp_block(self):
        class SimpleMLP(nn.Block):
            def __init__(self, d_model, d_ff):
                super().__init__()
                self.norm = nn.RMSNorm(d_model)
                self.mlp = nn.SwiGLUMLP(d_model, d_ff)

            def forward(self, x):
                h = self.norm(x)
                return self.mlp(h)

        blk = SimpleMLP(512, 1024)
        spec = blk.compile()

        assert isinstance(spec, BlockSpec)
        assert "norm_weight" in spec.params
        assert "mlp_up_weight" in spec.params
        assert "mlp_down_weight" in spec.params
        assert spec.params["norm_weight"].quantizable is False

    def test_attention_only_block(self):
        class AttnBlock(nn.Block):
            def __init__(self):
                super().__init__()
                self.attn = nn.GQAAttention(
                    d_model=512,
                    num_query_heads=8,
                    num_kv_heads=4,
                    head_size=64,
                    max_seq=2048,
                )

            def forward(self, x, position_ids):
                return self.attn(x, position_ids)

        blk = AttnBlock()
        spec = blk.compile()
        assert "attn_qkv_weight" in spec.params
        assert "attn_out_weight" in spec.params
        assert "attn_rope_freqs" in spec.params


# ============================================================================
# MoE block compilation
# ============================================================================


class TestMoEBlock:
    """Verify Qwen3MoEBlock compilation."""

    def test_moe_block_compiles(self):
        from surogate.dsl.blocks.qwen3_moe import Qwen3MoEBlock

        blk = Qwen3MoEBlock(
            d_model=2048,
            num_query_heads=16,
            num_kv_heads=4,
            head_size=128,
            d_ff=1408,
            max_seq=4096,
            num_experts=64,
            num_experts_per_tok=8,
        )
        spec = blk.compile()
        assert isinstance(spec, BlockSpec)
        ops = {n.op for n in spec.forward._traced_graph.nodes}
        assert "moe_grouped_gemm_gate_up" in ops
        assert "moe_unpermute" in ops

    def test_moe_block_with_shared_expert(self):
        from surogate.dsl.blocks.qwen3_moe import Qwen3MoEBlock

        blk = Qwen3MoEBlock(
            d_model=2048,
            num_query_heads=16,
            num_kv_heads=4,
            head_size=128,
            d_ff=1408,
            max_seq=4096,
            num_experts=64,
            num_experts_per_tok=8,
            use_shared_expert=True,
            shared_expert_intermediate=5632,
        )
        spec = blk.compile()
        ops = {n.op for n in spec.forward._traced_graph.nodes}
        assert "silu" in ops  # shared expert gate activation
        assert "add" in ops  # moe_out + shared_out
