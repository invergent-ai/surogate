from surogate.train.trainer import (
    _flatten_arena_summary,
    _flatten_descriptor_summary,
    _percentile_summary,
    _summarize_block_schemas,
)


def test_percentile_summary_is_stable_for_small_samples():
    assert _percentile_summary([]) == {}
    assert _percentile_summary([30.0, 10.0, 20.0]) == {
        "mean": 20.0,
        "p50": 20.0,
        "p95": 30.0,
    }


def test_flatten_descriptor_summary_totals_fusion_candidates():
    summary = {
        "forward": {
            "num_ops": 10,
            "fusion_candidate_starts": 2,
            "matmul_fp8_forward_eligible_ops": 4,
            "matmul_fp4_forward_eligible_ops": 3,
            "moe_fp8_grouped_eligible_ops": 1,
            "moe_fp4_grouped_eligible_ops": 2,
            "lora_slices": 3,
            "lora_schema_slot_slices": 3,
            "name": "forward",
        },
        "backward": {
            "num_ops": 12,
            "fusion_candidate_starts": 3,
            "matmul_fp8_backward_eligible_ops": 5,
            "matmul_fp4_backward_eligible_ops": 2,
            "moe_fp8_grouped_eligible_ops": 3,
            "moe_fp4_grouped_eligible_ops": 4,
            "lora_slices": 2,
            "lora_schema_slot_slices": 2,
            "grouped_lora_schema_slot_slices": 1,
            "name": "backward",
        },
    }

    flattened = _flatten_descriptor_summary(summary)

    assert flattened["forward_num_ops"] == 10
    assert flattened["backward_num_ops"] == 12
    assert flattened["forward_fusion_candidate_starts"] == 2
    assert flattened["backward_fusion_candidate_starts"] == 3
    assert flattened["fusion_candidate_starts"] == 5
    assert flattened["matmul_fp8_forward_eligible_ops"] == 4
    assert flattened["matmul_fp8_backward_eligible_ops"] == 5
    assert flattened["matmul_fp4_forward_eligible_ops"] == 3
    assert flattened["matmul_fp4_backward_eligible_ops"] == 2
    assert flattened["moe_fp8_grouped_eligible_ops"] == 4
    assert flattened["moe_fp4_grouped_eligible_ops"] == 6
    assert flattened["lora_slices"] == 5
    assert flattened["lora_schema_slot_slices"] == 5
    assert flattened["grouped_lora_schema_slot_slices"] == 1


def test_flatten_arena_summary_keeps_top_level_graph_and_region_counts():
    summary = {
        "arenas_allocated": True,
        "arena_fwd_stack_bytes": 1024,
        "arena_save_for_bwd_block_bases": [0, 512],
        "forward": {
            "num_tensors": 10,
            "fwd_stack_peak": 2048,
            "regions": [
                {"region": "FwdStack", "tid_count": 3, "tid_bytes": 4096},
                {"region": "SaveForBwd", "tid_count": 1, "tid_bytes": 512},
            ],
        },
        "backward": {
            "num_tensors": 12,
            "bwd_stack_peak": 3072,
            "regions": [],
        },
    }

    flattened = _flatten_arena_summary(summary)

    assert flattened["arenas_allocated"] == 1
    assert flattened["arena_fwd_stack_bytes"] == 1024
    assert "arena_save_for_bwd_block_bases" not in flattened
    assert flattened["forward_num_tensors"] == 10
    assert flattened["forward_fwd_stack_peak"] == 2048
    assert flattened["forward_region_fwdstack_tid_count"] == 3
    assert flattened["forward_region_fwdstack_tid_bytes"] == 4096
    assert flattened["forward_region_saveforbwd_tid_bytes"] == 512
    assert flattened["backward_bwd_stack_peak"] == 3072


def test_summarize_block_schemas_counts_layer_storage_and_distribution():
    ir_json = """
{
  "modules": [
    {
      "config": {"n_layers": 2},
      "forward": {
        "metadata": {
          "block_schemas": [
            {
              "layer": 0,
              "schema": {
                "routing": {"kind": "topk_softmax"},
                "ep_topology": {"ep_size_param": "ep_size"},
                "attrs": {"block_family": "qwen3_moe"},
                "slots": [
                  {
                    "name": "experts_gate_up",
                    "kind": "param",
                    "shape": ["E", "2M", "C"],
                    "residency": "auto",
                    "streaming_hint": {"prefetch_distance": 1},
                    "distribution": {"kind": "expert_parallel"},
                    "grouped": true
                  },
                  {
                    "name": "mlp_down",
                    "kind": "activation",
                    "lifetime": "block",
                    "shape": ["B", "T", "C"],
                    "residency": "cpu_pinned_stream",
                    "distribution": {"kind": "expert_parallel"},
                    "save_for_backward": true
                  }
                ]
              }
            }
          ]
        }
      }
    }
  ]
}
"""

    summary = _summarize_block_schemas(ir_json)

    assert summary["block_schema_records"] == 1
    assert summary["block_schema_expected_layers"] == 2
    assert summary["block_schema_missing_layers"] == 1
    assert summary["block_schema_moe_layers"] == 1
    assert summary["block_schema_routing_layers"] == 1
    assert summary["block_schema_ep_layers"] == 1
    assert summary["block_schema_slots"] == 2
    assert summary["block_schema_param_slots"] == 1
    assert summary["block_schema_activation_slots"] == 1
    assert summary["block_schema_layer_lifetime_slots"] == 1
    assert summary["block_schema_block_lifetime_slots"] == 1
    assert summary["block_schema_shape_slots"] == 2
    assert summary["block_schema_activation_shape_slots"] == 1
    assert summary["block_schema_save_for_backward_slots"] == 1
    assert summary["block_schema_grouped_slots"] == 1
    assert summary["block_schema_expert_parallel_slots"] == 2
    assert summary["block_schema_expert_parallel_param_slots"] == 1
    assert summary["block_schema_auto_resident_slots"] == 1
    assert summary["block_schema_cpu_stream_slots"] == 1
    assert summary["hook_after_produce_targets"] == 1
    assert summary["hook_before_consume_targets"] == 1
    assert summary["hook_after_all_to_all_targets"] == 1
    assert summary["hook_after_reduce_scatter_targets"] == 1
