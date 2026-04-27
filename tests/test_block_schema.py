import json

from surogate.dsl.block_schema import BlockSchema, DistributionDecl
from surogate.dsl.blocks.gemma4 import Gemma4FullMoEBlock, Gemma4SharedKVBlock, Gemma4SlidingBlock
from surogate.dsl.blocks.gpt_oss import GptOssBlock
from surogate.dsl.blocks.nemotron_h import NemotronHMamba2Block, NemotronHMoEBlock
from surogate.dsl.blocks.qwen3_5_moe import Qwen3_5MoEAttentionBlock, Qwen3_5MoELinearBlock
from surogate.dsl.blocks.qwen3_moe import Qwen3MoEBlock
from surogate.dsl.py_lowering import lower_block_spec
from surogate.dsl.py_compiler import _module_ir_to_dict, compile_block_spec, compile_model


def test_block_schema_distribution_factories():
    replicated = DistributionDecl.replicated()
    sharded = DistributionDecl.sharded_dim(dim=0, mode="zero2", num_shards="dp_size")
    expert = DistributionDecl.expert_parallel(global_experts="num_experts")

    assert replicated.kind == "replicated"
    assert sharded.kind == "sharded_dim"
    assert sharded.shard_dim == 0
    assert sharded.mode == "zero2"
    assert expert.kind == "expert_parallel"
    assert expert.experts_per_rank == "auto"


def test_nemotron_mamba_schema_attaches_to_block_spec():
    block = NemotronHMamba2Block(d_model=256, mamba_num_heads=8, mamba_head_dim=32, n_groups=4)
    spec = block.compile()

    assert isinstance(spec.schema, BlockSchema)
    assert spec.schema.attrs["block_family"] == "nemotron_mamba2"
    assert spec.schema.get_slot("ssm_state").save_for_backward is True
    assert spec.schema.get_slot("in_proj_weight").distribution.kind == "sharded_dim"
    assert spec.schema.get_slot("in_proj_weight").streaming_hint.prefetch_distance == 2


def test_nemotron_moe_schema_declares_routing_and_ep():
    block = NemotronHMoEBlock(d_model=256, moe_intermediate_size=512, num_experts=8, num_experts_per_tok=2)
    schema = block.compile().schema

    assert schema.routing.kind == "topk_sigmoid"
    assert schema.routing.scoring_bias is True
    assert schema.ep_topology.ep_size_param == "ep_size"
    assert schema.get_slot("router_weight").distribution.kind == "router_replicated"
    assert schema.get_slot("experts_up").grouped is True
    assert schema.get_slot("experts_up").distribution.kind == "expert_parallel"
    assert schema.get_slot("permuted_input").distribution.kind == "expert_parallel"


def test_acceptance_moe_blocks_declare_expert_parallel_schema():
    qwen = Qwen3MoEBlock(
        d_model=256,
        num_query_heads=4,
        num_kv_heads=2,
        head_size=64,
        d_ff=512,
        max_seq=2048,
        num_experts=4,
        num_experts_per_tok=2,
    ).compile()
    gpt_oss = GptOssBlock(
        d_model=256,
        num_query_heads=4,
        num_kv_heads=2,
        head_size=64,
        d_ff=512,
        max_seq=2048,
        num_experts=8,
        num_experts_per_tok=4,
    ).compile()

    for spec in (qwen, gpt_oss):
        assert spec.schema.routing.kind == "topk_softmax"
        assert spec.schema.get_slot("experts_gate_up").distribution.kind == "expert_parallel"
        assert spec.schema.get_slot("experts_down").distribution.kind == "expert_parallel"


def test_qwen3_5_moe_blocks_declare_schema():
    attn = Qwen3_5MoEAttentionBlock(
        d_model=256,
        num_query_heads=4,
        num_kv_heads=2,
        head_size=64,
        d_ff=512,
        max_seq=2048,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate=128,
    ).compile()
    linear = Qwen3_5MoELinearBlock(
        d_model=256,
        d_ff=512,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate=128,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
    ).compile()

    assert attn.schema.attrs["block_family"] == "qwen3_5_moe_attention"
    assert linear.schema.attrs["block_family"] == "qwen3_5_moe_linear"
    assert attn.schema.routing.shared_experts == "shared_expert_intermediate"
    assert linear.schema.get_slot("mixer_conv_state").save_for_backward is True
    assert attn.schema.get_slot("experts_gate_up").distribution.kind == "expert_parallel"
    assert linear.schema.get_slot("permuted_input").distribution.kind == "expert_parallel"


def test_gemma4_blocks_declare_dense_and_moe_schema():
    dense = Gemma4SlidingBlock(
        d_model=256,
        num_query_heads=4,
        num_kv_heads=2,
        head_size=64,
        d_ff=512,
        max_seq=2048,
    ).compile()
    shared = Gemma4SharedKVBlock(
        d_model=256,
        num_query_heads=4,
        num_kv_heads=2,
        head_size=64,
        d_ff=512,
        max_seq=2048,
    ).compile()
    moe = Gemma4FullMoEBlock(
        d_model=256,
        num_query_heads=4,
        num_kv_heads=2,
        head_size=64,
        d_ff=512,
        max_seq=2048,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
    ).compile()

    assert dense.schema.attrs["block_family"] == "gemma4_sliding"
    assert shared.schema.attrs["shared_kv"] is True
    assert shared.schema.get_slot("self_attn_q_weight").kind == "param"
    assert moe.schema.attrs["block_family"] == "gemma4_full_moe"
    assert moe.schema.routing.kind == "topk_softmax"
    assert moe.schema.get_slot("experts_gate_up").distribution.kind == "expert_parallel"


def test_block_schema_lowers_to_block_ir_dict():
    spec = Qwen3MoEBlock(
        d_model=256,
        num_query_heads=4,
        num_kv_heads=2,
        head_size=64,
        d_ff=512,
        max_seq=2048,
        num_experts=4,
        num_experts_per_tok=2,
    ).compile()

    ir = compile_block_spec(
        spec,
        {
            "d_model": 256,
            "num_query_heads": 4,
            "num_kv_heads": 2,
            "head_size": 64,
            "d_ff": 512,
            "max_seq": 2048,
            "num_experts": 4,
            "num_experts_per_tok": 2,
        },
    )
    payload = _module_ir_to_dict(ir)

    assert payload["block_schema"]["routing"]["kind"] == "topk_softmax"
    assert payload["block_schema"]["ep_topology"]["ep_size_param"] == "ep_size"
    expert_slot = next(slot for slot in payload["block_schema"]["slots"] if slot["name"] == "experts_gate_up")
    assert expert_slot["distribution"]["kind"] == "expert_parallel"
    assert expert_slot["streaming_hint"]["prefetch_distance"] == 1


def test_legacy_lowerer_preserves_block_schema_metadata():
    spec = NemotronHMamba2Block(d_model=256, mamba_num_heads=8, mamba_head_dim=32, n_groups=4).compile()

    ir = lower_block_spec(spec)

    assert ir.block_schema["attrs"]["block_family"] == "nemotron_mamba2"
    assert ir.block_schema["slots"][0]["name"] == "ssm_state"


def test_model_compile_emits_per_layer_block_schema_metadata():
    from surogate.dsl.models.qwen3_moe import Qwen3MoEModel  # noqa: F401

    payload = json.loads(
        compile_model(
            "Qwen3MoEModel",
            {
                "vocab_size": 32000,
                "d_model": 256,
                "n_layers": 2,
                "num_query_heads": 4,
                "num_kv_heads": 2,
                "d_ff": 512,
                "max_seq": 2048,
                "head_size": 64,
                "eps": 1e-6,
                "use_qkv_bias": False,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "shared_expert_intermediate": 256,
            },
            raise_on_error=True,
        )
    )

    forward = payload["modules"][0]["forward"]
    records = forward["metadata"]["block_schemas"]

    assert [record["layer"] for record in records] == [0, 1]
    assert records[0]["blocks_param"] == "blocks"
    assert records[0]["block_type"] == "Qwen3MoEBlock"
    assert records[0]["schema"]["routing"]["kind"] == "topk_softmax"
    assert records[0]["schema"]["ep_topology"]["ep_size_param"] == "ep_size"
