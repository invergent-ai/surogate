import inspect
from pathlib import Path

import numpy as np
import pytest

from surogate.grpo.trainer import (
    GRPOTrainer,
    MAX_REPLAY_ANCHORED_MISMATCH_KL,
    NativeGRPOUpdateRejected,
    _stack_grpo_rank_rows,
    _validate_replay_anchored_native_update,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _micro_batch(token: int, *, positions: list[int], replay: bool = False) -> dict[str, np.ndarray | None]:
    length = len(positions)
    loss_mask = np.zeros(length, dtype=bool)
    loss_mask[-2:] = True
    return {
        "input_ids": np.full((1, length), token, dtype=np.int32),
        "position_ids": np.asarray(positions, dtype=np.int32).reshape(1, length),
        "targets": np.full((1, length), token + 1, dtype=np.int32),
        "advantages": np.full((1, length), float(token), dtype=np.float32),
        "inference_logprobs": np.full((1, length), -0.5, dtype=np.float32),
        "loss_mask": loss_mask.reshape(1, length),
        "teacher_logprobs": None,
        "temperatures": np.ones((1, length), dtype=np.float32),
        "opd_reference_logprobs": np.zeros((1, length), dtype=np.float32),
        "hindsight_logprobs": np.zeros((1, length), dtype=np.float32),
        "hindsight_mask": np.zeros((1, length), dtype=bool),
        "replay_mask": (loss_mask if replay else np.zeros(length, dtype=bool)).reshape(1, length),
        "replay_weights": np.where(loss_mask & replay, 2.0, 1.0).astype(np.float32).reshape(1, length),
    }


def test_rank_batch_preserves_distinct_rows_masks_and_sample_ranges():
    batch = _stack_grpo_rank_rows(
        [
            _micro_batch(11, positions=[0, 1, 2, 0, 1]),
            _micro_batch(22, positions=[0, 1, 2, 3], replay=True),
        ],
        rank_width=3,
        seq_len=8,
    )

    assert batch["input_ids"].shape == (3, 8)
    assert batch["targets"].shape == (3, 8)
    assert batch["position_ids"].shape == (3, 8)
    assert np.all(batch["input_ids"][0, :5] == 11)
    assert np.all(batch["input_ids"][1, :4] == 22)
    assert np.all(batch["loss_mask"].reshape(3, 8)[2] == 0)
    assert int(batch["loss_mask"].sum()) == 4
    assert int(batch["replay_mask"].sum()) == 2
    assert float(batch["replay_weights"].reshape(3, 8)[1].sum()) == 10.0

    assert batch["samples_per_rank"] == 2
    assert batch["valid_rows"] == 2
    assert batch["sample_starts"].reshape(3, 2).tolist() == [[0, 3], [0, -1], [-1, -1]]
    assert batch["sample_ends"].reshape(3, 2).tolist() == [[3, 5], [4, -1], [-1, -1]]


def test_rank_batch_global_loss_normalization_compensates_gradient_average():
    global_selected_tokens = 9287.0
    rank_width = 6.0
    native_loss_scale = global_selected_tokens / rank_width
    per_rank_unscaled_gradients = np.asarray([2.0, -1.0, 4.0, 0.5, -0.25, 3.0])

    averaged_native_gradient = np.mean(per_rank_unscaled_gradients / native_loss_scale)
    exact_global_gradient = np.sum(per_rank_unscaled_gradients) / global_selected_tokens
    np.testing.assert_allclose(averaged_native_gradient, exact_global_gradient, rtol=1e-15)


def test_native_binding_offsets_every_rank_local_grpo_surface():
    source = (REPO_ROOT / "csrc/src/binding/py_train.cpp").read_text()
    assert "inference_logprobs + token_offset" in source
    assert "advantages + token_offset" in source
    assert "loss_mask + token_offset" in source
    assert "sample_starts + sample_offset" in source
    assert "teacher_logprobs ? teacher_logprobs + token_offset" in source
    assert "replay_mask ? replay_mask + token_offset" in source
    assert "replay_weights ? replay_weights + token_offset" in source


def _valid_native_update_metrics() -> dict[str, float]:
    return {
        "policy_loss": 1.25,
        "mismatch_kl": 0.02,
        "masked_mismatch_kl": 0.01,
        "unmasked_mismatch_kl": 0.03,
        "teacher_kl": 0.0,
        "opd_loss": 0.0,
        "replay_loss": 0.75,
        "replay_tokens": 2_448.0,
        "replay_weight_sum": 2_448.0,
        "total_tokens": 2_816.0,
        "policy_sample_count": 4.0,
    }


def test_replay_anchored_native_update_gate_accepts_complete_finite_signal():
    _validate_replay_anchored_native_update(
        metrics=_valid_native_update_metrics(),
        grad_norms=[0.5] * 6,
        valid_token_count=1_064,
        expected_loss_scale=2_816,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("policy_loss", float("nan")),
        ("mismatch_kl", float("inf")),
        ("replay_loss", 0.0),
        ("replay_tokens", 0.0),
        ("replay_weight_sum", 0.0),
        ("policy_sample_count", 0.0),
    ],
)
def test_replay_anchored_native_update_gate_rejects_bad_native_metrics(field: str, value: float):
    metrics = _valid_native_update_metrics()
    metrics[field] = value
    with pytest.raises(NativeGRPOUpdateRejected):
        _validate_replay_anchored_native_update(
            metrics=metrics,
            grad_norms=[0.5] * 6,
            valid_token_count=1_064,
            expected_loss_scale=2_816,
        )


@pytest.mark.parametrize("grad_norms", [[], [0.5, float("nan")], [0.5, 0.0]])
def test_replay_anchored_native_update_gate_rejects_missing_nonfinite_or_zero_gradient(
    grad_norms: list[float],
):
    with pytest.raises(NativeGRPOUpdateRejected):
        _validate_replay_anchored_native_update(
            metrics=_valid_native_update_metrics(),
            grad_norms=grad_norms,
            valid_token_count=1_064,
            expected_loss_scale=2_816,
        )


def test_replay_anchored_native_update_gate_rejects_global_token_mismatch():
    metrics = _valid_native_update_metrics()
    metrics["total_tokens"] = 2_815.0
    with pytest.raises(NativeGRPOUpdateRejected, match="selected-token count"):
        _validate_replay_anchored_native_update(
            metrics=metrics,
            grad_norms=[0.5] * 6,
            valid_token_count=1_064,
            expected_loss_scale=2_816,
        )


def test_replay_anchored_native_update_gate_rejects_excessive_mismatch_kl():
    metrics = _valid_native_update_metrics()
    metrics["mismatch_kl"] = MAX_REPLAY_ANCHORED_MISMATCH_KL + 1e-6
    with pytest.raises(NativeGRPOUpdateRejected, match="mismatch KL"):
        _validate_replay_anchored_native_update(
            metrics=metrics,
            grad_norms=[0.5] * 6,
            valid_token_count=1_064,
            expected_loss_scale=2_816,
        )


def test_replay_anchored_native_update_gate_rejects_nonpositive_last_row_vtc():
    with pytest.raises(NativeGRPOUpdateRejected, match="valid-token count"):
        _validate_replay_anchored_native_update(
            metrics=_valid_native_update_metrics(),
            grad_norms=[0.5] * 6,
            valid_token_count=0,
            expected_loss_scale=2_816,
        )


def test_replay_gate_precedes_optimizer_mutation_and_candidate_export():
    source = inspect.getsource(GRPOTrainer.train)
    gate = source.index("_validate_replay_anchored_native_update(")
    update = source.index("self.trainer.update_with_config(", gate)
    final_broadcast = source.rindex("self.broadcast.broadcast(")
    final_export = source.rindex("self.trainer.export_")

    assert gate < update < final_broadcast
    assert gate < update < final_export
