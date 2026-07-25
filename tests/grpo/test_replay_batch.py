import hashlib

import msgspec
import pytest

from surogate.core.config.grpo_orch_config import GRPOOrchestratorConfig
from surogate.grpo.orchestrator.replay import ReplayBatchError, ReplayBatchSource
from surogate.grpo.transport import TrainingBatch, TrainingSample
from surogate.utils.dict import DictDefault


def _sample(token: int, *, replay: bool = True) -> TrainingSample:
    return TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[token, token + 1],
        completion_mask=[True, True],
        completion_logprobs=[-1.0, -1.0],
        completion_temperatures=[1.0, 1.0],
        advantage=0.0,
        reward=0.0,
        replay_mask=[False, False, replay, replay],
    )


def _config(tmp_path, samples, *, samples_per_step=1):
    path = tmp_path / "replay.bin"
    path.write_bytes(msgspec.msgpack.encode(TrainingBatch(examples=samples, step=0)))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    config = GRPOOrchestratorConfig(
        DictDefault(
            {
                "replay": {
                    "path": str(path),
                    "sha256": digest,
                    "samples_per_step": samples_per_step,
                    "seed": 19,
                }
            }
        )
    )
    return config.replay


def test_replay_source_is_hash_bound_deterministic_and_returns_copies(tmp_path):
    source = ReplayBatchSource.load(_config(tmp_path, [_sample(10), _sample(20), _sample(30)]))

    first = source.examples_for_step(0)
    assert first == source.examples_for_step(0)
    first[0].completion_ids[0] = 999
    assert source.examples_for_step(0)[0].completion_ids[0] != 999
    assert source.examples_for_step(0) != source.examples_for_step(1)


def test_replay_source_rejects_samples_without_selected_replay_tokens(tmp_path):
    with pytest.raises(ReplayBatchError, match="selects no replay tokens"):
        ReplayBatchSource.load(_config(tmp_path, [_sample(10, replay=False)]))


def test_replay_source_validates_explicit_replay_weights(tmp_path):
    sample = _sample(10)
    sample.replay_weights = [1.0, 1.0, 2.0, 0.5]
    source = ReplayBatchSource.load(_config(tmp_path, [sample]))
    assert source.samples[0].replay_weights == [1.0, 1.0, 2.0, 0.5]

    sample.replay_weights = [2.0, 1.0, 2.0, 0.5]
    with pytest.raises(ReplayBatchError, match="weights a non-replay token"):
        ReplayBatchSource.load(_config(tmp_path, [sample]))
