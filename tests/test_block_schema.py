from surogate.dsl.block_schema import BlockSchema, DistributionDecl
from surogate.dsl.blocks.gpt_oss import GptOssBlock
from surogate.dsl.blocks.nemotron_h import NemotronHMamba2Block, NemotronHMoEBlock
from surogate.dsl.blocks.qwen3_moe import Qwen3MoEBlock


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
