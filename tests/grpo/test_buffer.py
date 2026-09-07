from __future__ import annotations

import pytest
from datasets import Dataset

from surogate.core.config.grpo_orch_config import GRPOBufferConfig
from surogate.grpo.orchestrator.buffer import Buffer
from surogate.utils.dict import DictDefault


def _dataset() -> Dataset:
    # verifiers >= 0.2: EnvGroup routes via info["env_id"] instead of a `task` column.
    return Dataset.from_list(
        [
            {"example_id": 0, "info": {"env_id": "env_a"}, "prompt": [{"role": "user", "content": "a"}]},
            {"example_id": 1, "info": {"env_id": "env_a"}, "prompt": [{"role": "user", "content": "b"}]},
        ]
    )


def _rollout(example_id: int, reward: float) -> dict:
    return {
        "example_id": example_id,
        "info": {"env_id": "env_a"},
        "reward": reward,
        "trajectory": [{"role": "assistant", "content": "{}"}],
        "error": None,
    }


def _buffer(**config_overrides) -> Buffer:
    config = GRPOBufferConfig(
        DictDefault(
            {
                "easy_threshold": 1.0,
                "hard_threshold": 0.0,
                "online_difficulty_filtering": True,
                **config_overrides,
            }
        )
    )
    return Buffer(_dataset(), ["env_a"], config)


def test_online_difficulty_filtering_evicts_saturated_examples():
    buffer = _buffer()

    buffer.update([_rollout(0, 0.0), _rollout(0, 0.0)])
    buffer.update([_rollout(1, 1.0), _rollout(1, 1.0)])

    assert sum(len(examples) for examples in buffer.example_buffer.values()) == 0
    assert len(buffer.hard_examples) == 1
    assert len(buffer.easy_examples) == 1
    assert buffer.rollout_buffer == []


def test_sampling_empty_normal_pool_recycles_via_starvation_guard():
    """Pre-guard behavior raised here; the per-env starvation guard now returns
    a weighted env's easy/hard/flat examples to normal instead of leaving the
    env silently unschedulable (recycle fractions unset does not disable it)."""
    buffer = _buffer()
    buffer.update([_rollout(0, 0.0), _rollout(0, 0.0)])
    buffer.update([_rollout(1, 1.0), _rollout(1, 1.0)])

    sampled = buffer.sample_examples(n=1)
    assert sampled[0]["example_id"] in {0, 1}
    assert sum(len(examples) for examples in buffer.example_buffer.values()) == 2
    assert not buffer.easy_examples and not buffer.hard_examples


def test_sampling_raises_only_when_every_pool_is_truly_empty():
    buffer = _buffer()
    buffer.example_buffer = {"env_a": {}}
    buffer.easy_examples = []
    buffer.hard_examples = []
    buffer.flat_examples = []

    with pytest.raises(ValueError, match="No environments left with examples"):
        buffer.sample_examples(n=1)


def test_starvation_guard_also_drains_the_flat_pool():
    """The carpet filter added a third sideline pool AFTER the starvation guard
    was written. A lane whose grader returns one partial-credit value for every
    rollout (measured: a terminal grader emitting exactly 0.50) sends its whole
    registry to flat, so draining only easy+hard would leave it starved."""
    # Recycle fractions pinned to 0 so the GLOBAL valve cannot rescue anything —
    # only the per-env starvation guard can, which is what this test is about.
    buffer = _buffer(
        flat_group_filtering=True, easy_threshold=None, hard_threshold=None,
        recycle_easy_fraction=0.0, recycle_hard_fraction=0.0, recycle_flat_fraction=0.0)
    buffer.update([_rollout(0, 0.5), _rollout(0, 0.5)])  # pure carpet -> flat
    buffer.update([_rollout(1, 0.5), _rollout(1, 0.5)])
    assert buffer.flat_examples, "setup: both examples should be sidelined as flat"
    assert sum(len(e) for e in buffer.example_buffer.values()) == 0

    sampled = buffer.sample_examples(n=1)

    assert len(sampled) == 1
    assert not buffer.flat_examples, "flat pool must be drained for a starved env"


def test_sampling_recycles_easy_and_hard_examples_when_normal_pool_is_empty():
    buffer = _buffer(recycle_easy_fraction=1.0, recycle_hard_fraction=1.0)
    buffer.update([_rollout(0, 0.0), _rollout(0, 0.0)])
    buffer.update([_rollout(1, 1.0), _rollout(1, 1.0)])

    sampled = buffer.sample_examples(n=1)

    assert sampled[0]["example_id"] in {0, 1}
    assert sum(len(examples) for examples in buffer.example_buffer.values()) == 2
    metrics = buffer.get_metrics()
    assert metrics["recycled_examples/easy"] == 1
    assert metrics["recycled_examples/hard"] == 1


def test_one_use_sampling_never_returns_a_task_twice_and_persists_consumption(
    tmp_path,
):
    buffer = _buffer(sample_without_replacement=True)

    sampled = buffer.sample_examples(n=2)

    assert {row["example_id"] for row in sampled} == {0, 1}
    assert len(buffer.consumed_examples) == 2
    with pytest.raises(ValueError, match="No environments left with examples"):
        buffer.sample_examples(n=1)

    checkpoint = tmp_path / "buffer"
    buffer.save(checkpoint)
    restored = _buffer(sample_without_replacement=True)
    restored.load(checkpoint)
    assert {row["example_id"] for row in restored.consumed_examples} == {0, 1}
    with pytest.raises(ValueError, match="No environments left with examples"):
        restored.sample_examples(n=1)


def test_flat_group_filtering_excludes_zero_variance_groups():
    buffer = _buffer(flat_group_filtering=True, easy_threshold=None, hard_threshold=None)

    # All-0.5 group: mean-based mechanisms cannot see it; variance can.
    buffer.update([_rollout(0, 0.5), _rollout(0, 0.5), _rollout(0, 0.5)])
    assert buffer.rollout_buffer == []
    assert len(buffer.flat_examples) == 1
    assert 0 not in buffer.example_buffer["env_a"]

    # Variant group passes through untouched.
    buffer.update([_rollout(1, 0.0), _rollout(1, 1.0)])
    assert len(buffer.rollout_buffer) == 2
    assert 1 in buffer.example_buffer["env_a"]


def test_flat_group_filtering_off_by_default_keeps_flat_rollouts():
    buffer = _buffer(easy_threshold=None, hard_threshold=None, online_difficulty_filtering=False)

    buffer.update([_rollout(0, 0.5), _rollout(0, 0.5)])
    assert len(buffer.rollout_buffer) == 2
    assert buffer.flat_examples == []


def test_flat_examples_recycle_back_to_normal():
    buffer = _buffer(
        flat_group_filtering=True,
        easy_threshold=None,
        hard_threshold=None,
        recycle_flat_fraction=1.0,
        normal_pool_min_examples=2,
    )
    buffer.update([_rollout(0, 0.5), _rollout(0, 0.5)])
    assert len(buffer.flat_examples) == 1
    # Normal pool (1 left) <= min (2) triggers recycling on the next check.
    buffer._recycle_examples_if_needed()
    assert buffer.flat_examples == []
    assert 0 in buffer.example_buffer["env_a"]


def test_load_tolerates_checkpoints_predating_flat_pool(tmp_path):
    # Checkpoints saved before the flat pool existed have no
    # flat_examples.jsonl; load() must treat absent as empty (this
    # crashed a live resume on 2026-08-16).
    saver = _buffer()
    saver.update([_rollout(0, 0.0), _rollout(0, 0.0)])
    saver.save(tmp_path)
    (tmp_path / "flat_examples.jsonl").unlink()

    loader = _buffer()
    loader.load(tmp_path)
    assert loader.flat_examples == []


def test_near_flat_group_filtered_by_epsilon():
    """61-63/64 identical rewards with one outlier — the measured live pattern
    the exact-match trigger misses — must be filtered when spread <= epsilon."""
    buffer = _buffer(flat_group_filtering=True, flat_group_epsilon=0.15)
    rollouts = [_rollout(0, 0.5) for _ in range(63)] + [_rollout(0, 0.4)]
    buffer.update(rollouts)
    assert any(e.get("example_id") == 0 for e in buffer.flat_examples)


def test_group_with_a_win_survives_epsilon_filter():
    """One full-credit rollout (spread 0.5) must keep the group trainable."""
    buffer = _buffer(flat_group_filtering=True, flat_group_epsilon=0.15)
    rollouts = [_rollout(1, 0.5) for _ in range(63)] + [_rollout(1, 1.0)]
    buffer.update(rollouts)
    assert not any(e.get("example_id") == 1 for e in buffer.flat_examples)


def test_wal_persists_completed_rollouts_across_restart(tmp_path):
    """A mid-step orchestrator bounce must not lose delivered groups: update()
    appends to the WAL, and a fresh buffer replays it after checkpoint load
    (the 2026-08-20/21 bounces lost 128-192 rollouts each without this)."""
    spool = tmp_path / "live_spool"
    ckpt = tmp_path / "ckpt"

    first = _buffer()
    first.attach_wal(spool)
    first.update([_rollout(0, 1.0), _rollout(0, 0.0)])
    first.save(ckpt)  # checkpoint boundary: WAL truncated, ckpt owns rollouts
    assert not (spool / "rollout_wal.jsonl").exists()
    first.update([_rollout(1, 1.0), _rollout(1, 0.0)])  # post-ckpt delta
    # crash here: `first` dies with 2 undelivered rollouts in memory + WAL

    resumed = _buffer()
    resumed.attach_wal(spool)
    resumed.load(ckpt)
    n_ckpt = len(resumed.rollout_buffer)
    restored = resumed.replay_wal()
    assert restored == 2
    assert len(resumed.rollout_buffer) == n_ckpt + 2


def test_wal_replay_is_idempotent(tmp_path):
    """Double replay (or replay overlapping checkpoint contents) must not
    duplicate rollouts — dedupe is by full-record hash."""
    spool = tmp_path / "live_spool"
    buffer = _buffer()
    buffer.attach_wal(spool)
    buffer.update([_rollout(0, 1.0), _rollout(0, 0.0)])
    n = len(buffer.rollout_buffer)
    assert buffer.replay_wal() == 0  # already in memory
    assert buffer.replay_wal() == 0
    assert len(buffer.rollout_buffer) == n


def test_wal_unarmed_is_a_noop(tmp_path):
    """Without attach_wal (e.g. the val buffer) nothing is written or replayed."""
    buffer = _buffer()
    buffer.update([_rollout(0, 1.0), _rollout(0, 0.0)])
    assert buffer.replay_wal() == 0
    assert not list(tmp_path.iterdir())


def test_dispersion_sampling_biases_toward_high_gradient_tasks(tmp_path):
    """The frontier signal is reward DISPERSION: a task whose groups produce
    real spread is boosted; a partial-credit CARPET (uniform outcomes, high
    mean) is demoted even though a mean-band rule would boost it."""
    import random as _random

    buffer = _buffer(midband_sampling_boost=4.0)
    env = buffer.env_names[0]
    ids = list(buffer.example_buffer[env].keys())[:2]
    buffer.example_std_ema[f"{env}:{ids[0]}"] = 0.39   # 12-win hard group
    buffer.example_std_ema[f"{env}:{ids[1]}"] = 0.088  # 61x0.5 carpet (measured)
    # the carpet's MEAN sits dead centre of the old 0.2-0.7 band, which is
    # exactly the case the mean-based rule got wrong
    buffer.example_reward_ema[f"{env}:{ids[1]}"] = 0.48

    _random.seed(7)
    counts = {i: 0 for i in ids}
    for _ in range(2000):
        counts[buffer._pick_example(env)["example_id"]] += 1
    assert counts[ids[0]] > 6 * counts[ids[1]]

    # unseen tasks keep weight 1.0 -> beat carpets, lose to high-dispersion
    del buffer.example_std_ema[f"{env}:{ids[0]}"]
    counts = {i: 0 for i in ids}
    for _ in range(2000):
        counts[buffer._pick_example(env)["example_id"]] += 1
    assert counts[ids[0]] > 2 * counts[ids[1]]


def test_dispersion_off_by_default_is_uniform():
    buffer = _buffer()
    env = buffer.env_names[0]
    buffer.example_std_ema[f"{env}:0"] = 0.39
    import random as _random
    _random.seed(3)
    picks = {buffer._pick_example(env)["example_id"] for _ in range(50)}
    assert len(picks) > 1


def test_flat_std_filter_catches_carpets_that_spread_misses():
    """61x0.5 + 2x0.0 has spread 0.50 (passes the epsilon filter) but std
    0.088 — ~no gradient. The std criterion must drop it while keeping a
    genuine sparse-win group (12x1.0 + 52x0.0, std 0.39)."""
    carpet = _buffer(flat_group_filtering=True, flat_group_epsilon=0.15,
                     flat_group_std_min=0.12)
    rollouts = [_rollout(0, 0.5) for _ in range(61)] + [_rollout(0, 0.0) for _ in range(2)] + [_rollout(0, 0.4)]
    carpet.update(rollouts)
    assert carpet.rollout_buffer == [], "0.5-carpet must be filtered on std"

    sparse = _buffer(flat_group_filtering=True, flat_group_epsilon=0.15,
                     flat_group_std_min=0.12)
    rollouts = [_rollout(0, 1.0) for _ in range(12)] + [_rollout(0, 0.0) for _ in range(52)]
    sparse.update(rollouts)
    assert len(sparse.rollout_buffer) == 64, "sparse-win group must survive"


def test_std_ema_updates_and_persists(tmp_path):
    buffer = _buffer()
    buffer.update([_rollout(0, 1.0) for _ in range(2)] + [_rollout(0, 0.0) for _ in range(2)])
    env = buffer.env_names[0]
    assert buffer.example_std_ema[f"{env}:0"] > 0.4
    buffer.save(tmp_path)
    resumed = _buffer()
    resumed.load(tmp_path)
    assert resumed.example_std_ema[f"{env}:0"] > 0.4


def test_reward_ema_updates_and_persists(tmp_path):
    buffer = _buffer()
    buffer.update([_rollout(0, 1.0), _rollout(0, 0.0)])  # group mean 0.5
    env = buffer.env_names[0]
    assert abs(buffer.example_reward_ema[f"{env}:0"] - 0.5) < 1e-9
    buffer.update([_rollout(0, 1.0), _rollout(0, 1.0)])  # ema -> 0.7*0.5+0.3*1
    assert abs(buffer.example_reward_ema[f"{env}:0"] - 0.65) < 1e-9

    buffer.save(tmp_path)
    resumed = _buffer()
    resumed.load(tmp_path)
    assert abs(resumed.example_reward_ema[f"{env}:0"] - 0.65) < 1e-9


def _long_dataset():
    from datasets import Dataset
    return Dataset.from_list([
        {"example_id": 0, "task": "env_a", "prompt": [{"role": "user", "content": "L" * 9000}]},
        {"example_id": 1, "task": "env_a", "prompt": [{"role": "user", "content": "L" * 9000}]},
        {"example_id": 2, "task": "env_a", "prompt": [{"role": "user", "content": "s" * 100}]},
    ])


def _buffer_long(**overrides) -> Buffer:
    config = GRPOBufferConfig(DictDefault({"easy_threshold": 1.0, "hard_threshold": 0.0, **overrides}))
    return Buffer(_long_dataset(), ["env_a"], config)


def test_vtc_rescue_guarantees_a_short_prompt_example(monkeypatch):
    """Every sample's prompt longer than the chunk => engine vtc=0 => the whole
    step is discarded (measured: step 103 lost ~2h). Sampling must swap in a
    short-prompt example so the trainer's reorder guard has a rescue micro."""
    import random as _random
    buffer = _buffer_long(vtc_min_short_prompt_examples=1, vtc_short_prompt_max_chars=6000)
    # force the draw toward the two LONG examples
    monkeypatch.setattr(buffer, "_pick_example",
                        lambda env: buffer.example_buffer[env][_random.choice([0, 1])])
    sampled = buffer.sample_examples(n=4)
    shorts = [e for e in sampled if Buffer._prompt_chars(e) <= 6000]
    assert len(shorts) >= 1, "batch must contain at least one short-prompt example"


def test_vtc_rescue_off_by_default_leaves_the_draw_untouched(monkeypatch):
    import random as _random
    buffer = _buffer_long()          # knob unset => feature off
    monkeypatch.setattr(buffer, "_pick_example",
                        lambda env: buffer.example_buffer[env][_random.choice([0, 1])])
    sampled = buffer.sample_examples(n=4)
    assert all(Buffer._prompt_chars(e) > 6000 for e in sampled)


def test_vtc_rescue_noop_when_batch_already_has_a_short_prompt():
    buffer = _buffer_long(vtc_min_short_prompt_examples=1, vtc_short_prompt_max_chars=6000)
    sampled = buffer.sample_examples(n=4)
    assert sum(1 for e in sampled if Buffer._prompt_chars(e) <= 6000) >= 1


def test_vtc_rescue_does_not_hijack_every_single_example_draw(monkeypatch):
    """The orchestrator draws ONE example per group, so a per-draw guarantee
    becomes 'every group must be short'. That collapsed step 104 into an
    all-CRM batch pinned to one worker. Across a window of single draws only
    ~1 may be forced short — the rest keep their sampled env ratios."""
    import random as _random
    buffer = _buffer_long(vtc_min_short_prompt_examples=1, vtc_short_prompt_max_chars=6000,
                          vtc_rescue_window=4)
    monkeypatch.setattr(buffer, "_pick_example",
                        lambda env: buffer.example_buffer[env][_random.choice([0, 1])])
    drawn = [buffer.sample_examples(n=1)[0] for _ in range(8)]
    shorts = sum(1 for e in drawn if Buffer._prompt_chars(e) <= 6000)
    assert shorts <= 2, f"window=4 over 8 draws must force <=2 short, got {shorts}"
    assert shorts >= 1, "but the step must still get its valid chunk-0 micro"


def test_vtc_rescue_still_fires_within_a_window_of_single_draws(monkeypatch):
    """Every window of `window` consecutive groups must hold a short example,
    otherwise the step that spans them is discarded for vtc=0."""
    import random as _random
    buffer = _buffer_long(vtc_min_short_prompt_examples=1, vtc_short_prompt_max_chars=6000,
                          vtc_rescue_window=4)
    monkeypatch.setattr(buffer, "_pick_example",
                        lambda env: buffer.example_buffer[env][_random.choice([0, 1])])
    drawn = [buffer.sample_examples(n=1)[0] for _ in range(4)]
    assert any(Buffer._prompt_chars(e) <= 6000 for e in drawn)


def test_vtc_rescue_batch_draw_still_rescues_immediately(monkeypatch):
    """A full-step draw (n == window) has no later draw to fall back on, so it
    must be rescued in place rather than deferred."""
    import random as _random
    buffer = _buffer_long(vtc_min_short_prompt_examples=1, vtc_short_prompt_max_chars=6000,
                          vtc_rescue_window=4)
    monkeypatch.setattr(buffer, "_pick_example",
                        lambda env: buffer.example_buffer[env][_random.choice([0, 1])])
    sampled = buffer.sample_examples(n=4)
    assert sum(1 for e in sampled if Buffer._prompt_chars(e) <= 6000) >= 1
def _two_env_buffer(**config_overrides) -> Buffer:
    dataset = Dataset.from_list(
        [
            {"example_id": 0, "task": "env_full", "prompt": [{"role": "user", "content": "a"}]},
            {"example_id": 1, "task": "env_full", "prompt": [{"role": "user", "content": "b"}]},
            {"example_id": 2, "task": "env_full", "prompt": [{"role": "user", "content": "c"}]},
            {"example_id": 3, "task": "env_starved", "prompt": [{"role": "user", "content": "d"}]},
        ]
    )
    config = GRPOBufferConfig(
        DictDefault(
            {
                "easy_threshold": 1.0,
                "hard_threshold": 0.5,
                "online_difficulty_filtering": True,
                "normal_pool_min_examples": 0,
                "recycle_easy_fraction": 0.15,
                "recycle_hard_fraction": 0.15,
                **config_overrides,
            }
        )
    )
    return Buffer(dataset, ["env_full", "env_starved"], config)


def test_starved_env_recycles_its_own_pool_even_when_global_floor_never_trips():
    """An env whose EVERY example was classified hard must not silently drop out
    of the deal: the global normal-pool floor cannot trip while other envs stay
    full, so the per-env guard returns the starved env's examples to normal."""
    buffer = _two_env_buffer()

    starved_rollout = {
        "example_id": 3,
        "task": "env_starved",
        "reward": 0.0,
        "trajectory": [{"role": "assistant", "content": "{}"}],
        "error": None,
    }
    buffer.update([starved_rollout, starved_rollout])

    # env_starved fully drained to hard; env_full keeps the global count high.
    assert len(buffer.example_buffer["env_starved"]) == 0
    assert len(buffer.example_buffer["env_full"]) == 3
    assert len(buffer.hard_examples) == 1

    buffer.sample_examples(n=1)

    assert len(buffer.example_buffer["env_starved"]) == 1, (
        "per-env starvation guard must return the starved env's examples to normal"
    )
    assert len(buffer.hard_examples) == 0
    assert len(buffer.example_buffer["env_full"]) == 3
