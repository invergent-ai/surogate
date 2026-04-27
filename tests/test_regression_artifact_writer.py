from surogate.train.trainer import _flatten_descriptor_summary, _percentile_summary


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
            "name": "forward",
        },
        "backward": {
            "num_ops": 12,
            "fusion_candidate_starts": 3,
            "matmul_fp8_backward_eligible_ops": 5,
            "matmul_fp4_backward_eligible_ops": 2,
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
